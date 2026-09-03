import streamlit as st
import pandas as pd
import json
import tempfile
import os
import sys
import time
from pathlib import Path
import plotly.graph_objects as go
import saqc

# Add the project root to the path so we can import 'src' modules when running via `streamlit run`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inspect_data import load_series, summarise_series, format_summary
from src.agent import run_agent, RunSummary

st.set_page_config(page_title="Water Sensor QC", layout="wide")

st.title("Agentic Water-Quality Data Quality Control")
st.markdown("""
This tool uses an LLM reasoning engine to inspect water-quality time series, automatically run QC operations, and generate a cleaned dataset with an explanatory report.
""")

st.markdown("""
<style>
    /* Hide the 'Limit 200MB per file' text on the file uploader */
    [data-testid="stFileUploadDropzone"] small {
        display: none !important;
    }
    .stFileUploader small {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

if "run_results" not in st.session_state:
    st.session_state.run_results = None
if "history" not in st.session_state:
    st.session_state.history = []
if "current_file" not in st.session_state:
    st.session_state.current_file = None

MOCK_DATA_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'mock')))


def _load_mock_decisions():
    """Load pre-generated mock decisions from data/mock/63680_decisions.json."""
    decisions_path = MOCK_DATA_DIR / "63680_decisions.json"
    if decisions_path.exists():
        return json.loads(decisions_path.read_text())
    return []


def _load_mock2_decisions():
    decisions_path = MOCK_DATA_DIR / "01467200_l1_decisions.json"
    if decisions_path.exists():
        return json.loads(decisions_path.read_text())
    return []

def mock_run_agent(qc, max_steps, log_dir):
    """A mock version of run_agent for testing without an API key.

    When the uploaded dataset matches the mock 63680 series (672 rows), loads
    pre-generated flags and decisions from data/mock/.  Otherwise falls back to
    simple midpoint-based flags so any CSV still produces visible output.
    """
    with st.spinner("Mock agent is thinking..."):
        time.sleep(2)

    df = qc.data.to_pandas().copy()
    if df.index.name != "datetime":
        df = df.reset_index()

    decisions = []

    # Check if this looks like the mock 63680 dataset
    clean_path = MOCK_DATA_DIR / "63680_clean.csv"
    mock2_clean_path = MOCK_DATA_DIR / "01467200_l1_clean.csv"
    
    if len(df) == 672 and clean_path.exists():
        # Load pre-generated flags from the mock clean file
        mock_clean = pd.read_csv(clean_path)
        df["flag"] = mock_clean["flag"].values
        decisions = _load_mock_decisions()
    elif len(df) == 210816 and mock2_clean_path.exists():
        # Load pre-generated flags from the mock2 clean file
        mock_clean = pd.read_csv(mock2_clean_path)
        df["flag"] = mock_clean["flag"].values
        decisions = _load_mock2_decisions()
    else:
        # Fallback: simple midpoint-based flags for any other CSV
        df["flag"] = None
        if len(df) > 0:
            mid = len(df) // 2
            if mid > 0:
                df.iloc[mid:mid+3, df.columns.get_loc("flag")] = "Spike"
                df.iloc[mid+10:mid+13, df.columns.get_loc("flag")] = "Plateau"
                if mid+25 < len(df):
                    df.iloc[mid+25, df.columns.get_loc("flag")] = "Borderline Spike"

    report = (
        "Mock Report\n"
        "===========\n\n"
        "The agent successfully ran in mock mode. No real API calls were made.\n\n"
        "**Findings:**\n"
        "- 2 spike events detected and deleted (1-sample and 2-sample excursions)\n"
        "- 1 storm peak flagged but **kept** as genuine signal (14-sample width, "
        "5.5 h recovery)\n"
        "- 1 plateau (stuck sensor) detected and deleted (13 samples at exactly 4.2 FNU)\n"
        "- 1 gap (8 samples, 2 h) imputed with rolling median\n"
    )

    summary = RunSummary(
        steps=8,
        input_tokens=12400,
        output_tokens=3200,
        est_cost_usd=0.085,
        log_path="logs/mock_run.jsonl"
    )
    return qc, df, report, summary, decisions

def build_decisions_from_log(log_path_str: str, df: pd.DataFrame) -> list[dict]:
    """Parse a real JSONL run log to build the per-decision UI structure."""
    from src.workbench.visualize_log import load_events, parse_run

    path = Path(log_path_str)
    if not path.exists():
        return []

    events = load_events(path)
    run_log = parse_run(events, path)

    decisions = []
    decision_id = 1

    for step in run_log.steps:
        for call in step.calls:
            if not call.flagged:
                continue
                
            # Convert flagged timestamps to a pandas DatetimeIndex for easy clustering
            dt_series = pd.Series(pd.to_datetime(call.flagged))
            
            # Split into distinct segments if points are separated by more than 24 hours
            gap_mask = dt_series.diff() > pd.Timedelta("24h")
            group_ids = gap_mask.cumsum()
            
            for _, group in dt_series.groupby(group_ids):
                start_dt = str(group.iloc[0])
                end_dt = str(group.iloc[-1])
                flagged_list = [str(t) for t in group]

                # Fetch value from clean_df
                try:
                    val = df.loc[df['datetime'] == start_dt, 'value'].iloc[0]
                except (KeyError, IndexError):
                    val = None

                # Extract neighbourhood (+/- 16 samples around the first point in this segment)
                try:
                    idx_start = df[df['datetime'] == start_dt].index[0]
                    nb_start = str(df.iloc[max(0, idx_start - 16)]['datetime'])
                    nb_end = str(df.iloc[min(len(df)-1, idx_start + 16)]['datetime'])
                except (KeyError, IndexError):
                    nb_start = start_dt
                    nb_end = end_dt

                d = {
                    "id": decision_id,
                    "segment_start": start_dt,
                    "segment_end": end_dt,
                    "value": val,
                    "action": "flagged", # Real runs haven't written the final flag yet
                    "anomaly_type": call.name.replace('flag_', ''),
                    "flagged_by": call.name,
                    "n_points": len(flagged_list),
                    "flagged_timestamps": flagged_list,
                    "reasoning": step.text or "No reasoning provided by agent.",
                    "thinking": step.thinking,
                    "neighborhood_start": nb_start,
                    "neighborhood_end": nb_end,
                    "params": json.dumps(call.params)
                }
                decisions.append(d)
                decision_id += 1

    return decisions

def plot_results(original_df, clean_df):
    """Generate a Plotly chart showing the series and flagged points."""
    fig = go.Figure()

    # Ensure datetime is a column, not just index
    if "datetime" not in original_df.columns:
        orig = original_df.reset_index()
    else:
        orig = original_df.copy()

    if "datetime" not in clean_df.columns:
        clean = clean_df.reset_index()
    else:
        clean = clean_df.copy()

    fig.add_trace(go.Scatter(
        x=orig["datetime"],
        y=orig["value"],
        mode='lines',
        name='Original',
        line=dict(color='rgba(31, 119, 180, 0.7)'), # Plotly default blue, semi-transparent
        hoverinfo='x+y'
    ))

    # If we have flags, plot them
    if "flag" in clean.columns:
        # Standard anomalies
        anomalies = clean[clean["flag"].notna() & (clean["flag"] != "Borderline Spike")]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies["datetime"],
                y=orig.loc[anomalies.index, "value"] if "value" in orig.columns else anomalies["value"],
                mode='markers',
                name='Flagged Anomaly',
                marker=dict(color='red', size=8, symbol='x'),
                text=anomalies["flag"],
                hovertemplate="Time: %{x}<br>Value: %{y}<br>Flagged by: %{text}<extra></extra>"
            ))

        # Borderline anomalies
        borderline = clean[clean["flag"] == "Borderline Spike"]
        if not borderline.empty:
            fig.add_trace(go.Scatter(
                x=borderline["datetime"],
                y=orig.loc[borderline.index, "value"] if "value" in orig.columns else borderline["value"],
                mode='markers',
                name='Borderline Case (Click Me!)',
                marker=dict(color='orange', size=12, symbol='star', line=dict(width=2, color='DarkSlateGrey')),
                text=borderline["flag"],
                hovertemplate="Time: %{x}<br>Value: %{y}<br>Flagged by: %{text}<extra></extra>"
            ))

    fig.update_layout(
        title=dict(text="Quality Control Results", font=dict(color="black")),
        xaxis=dict(title="Time", color="black", gridcolor="#e5e5e5"),
        yaxis=dict(title="Value", color="black", gridcolor="#e5e5e5"),
        hovermode="x unified",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="white", bordercolor="#e5e5e5", borderwidth=1),
        clickmode='event+select'
    )

    return fig

if "show_history" not in st.session_state:
    st.session_state.show_history = False

# Sidebar for config
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Anthropic API Key", type="password", help="Leave blank to use ANTHROPIC_API_KEY from environment or .env")
    use_mock = st.checkbox("Use Mock Mode (No API required)", value=True, help="Simulate an agent run without making API calls.")
    max_steps = st.number_input("Max Agent Steps", min_value=1, max_value=25, value=25)

    st.divider()
    if st.session_state.show_history:
        if st.button("Back to QC Tool", use_container_width=True):
            st.session_state.show_history = False
            st.rerun()
    else:
        if st.button("Past Results", use_container_width=True):
            st.session_state.show_history = True
            st.rerun()

if not st.session_state.show_history:
    # 1. Upload Panel
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload a CSV file (must have 'datetime' and 'value' columns)", type=['csv'])

    # Reset results if a new file is uploaded
    if uploaded_file is not None and uploaded_file.name != st.session_state.current_file:
        st.session_state.run_results = None
        st.session_state.current_file = uploaded_file.name

    if uploaded_file is not None:
        # Save to a temporary file so our existing pandas/Path logic works seamlessly
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            # 2. Preview Panel
            st.header("2. Data Preview")

            with st.spinner("Loading and validating..."):
                df = load_series(tmp_path)
                summary = summarise_series(df, path=uploaded_file.name)

            def format_date(iso_str):
                if not iso_str: return "N/A"
                # Format like "Jul 01, 2023"
                return pd.to_datetime(iso_str).strftime("%b %d, %Y")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{summary.n_rows:,}")
            with col2:
                st.metric("Missing Values", f"{summary.n_nan:,} ({summary.pct_nan:.2f}%)")
            with col3:
                st.metric("Inferred Frequency", str(summary.inferred_frequency))

            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("Start Date", format_date(summary.time_start))
            with col5:
                st.metric("End Date", format_date(summary.time_end))

            st.dataframe(df.head())

            # 3. Run Panel
            st.header("3. Run Agentic QC")

            if st.button("Run QC", type="primary"):
                if not use_mock and not api_key and not os.environ.get("ANTHROPIC_API_KEY"):
                    st.error("API Key required. Please provide it in the sidebar or check 'Use Mock Mode'.")
                else:
                    if api_key:
                        os.environ["ANTHROPIC_API_KEY"] = api_key

                    data = df.set_index("datetime")
                    qc = saqc.SaQC(data)

                    with st.spinner("Agent is analyzing the data and running QC tools. This may take a few minutes..."):
                        start_time = time.time()
                        try:
                            decisions = []
                            if use_mock:
                                final_qc, clean_df, report, run_summary, decisions = mock_run_agent(qc, max_steps=max_steps, log_dir="logs")
                            else:
                                final_qc, clean_df, report, run_summary = run_agent(qc, max_steps=max_steps, log_dir="logs")
                                decisions = build_decisions_from_log(run_summary.log_path, clean_df)

                            elapsed = time.time() - start_time

                            res = {
                                "filename": uploaded_file.name,
                                "clean_df": clean_df,
                                "report": report,
                                "run_summary": run_summary,
                                "elapsed": elapsed,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "decisions": decisions,
                            }
                            st.session_state.run_results = res
                            st.session_state.history.append(res)

                        except Exception as e:
                            st.error(f"An error occurred during the agent run: {str(e)}")

            if st.session_state.run_results is not None:
                res = st.session_state.run_results
                st.success(f"QC completed in {res['elapsed']:.1f} seconds! (Used {res['run_summary'].steps} tool calls)")

                # 4. Results Visualization
                st.header("4. Results Visualization")

                # Summary Dashboard — compute from actual flags
                st.subheader("Summary Dashboard")
                clean = res["clean_df"]
                decisions = res.get("decisions", [])
                if decisions:
                    action_counts = {}
                    type_counts = {}
                    for d in decisions:
                        a = d.get("action", "flag")
                        t = d.get("anomaly_type", "unknown")
                        action_counts[a] = action_counts.get(a, 0) + d.get("n_points", 1)
                        type_counts[t] = type_counts.get(t, 0) + 1
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Spikes", type_counts.get("spike", 0))
                    m2.metric("Plateaus", type_counts.get("plateau", 0))
                    m3.metric("Gaps", type_counts.get("gap", 0))
                    m4.metric("Kept (real)", type_counts.get("storm_peak", 0) + type_counts.get("level_shift", 0))
                    m5.metric("Decisions", len(decisions))
                else:
                    flag_counts = clean["flag"].dropna().value_counts() if "flag" in clean.columns else pd.Series(dtype=int)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Flagged", int(flag_counts.sum()) if len(flag_counts) else 0)
                    m2.metric("Flag Types", len(flag_counts))
                    m3.metric("Clean Rows", int(clean["flag"].isna().sum()) if "flag" in clean.columns else len(clean))

                st.markdown("---")
                st.subheader("Interactive Time-Series")
                st.markdown("Click on the orange star (**Borderline Case**) or any red cross (**Flagged Anomaly**) below to view the agent's reasoning.")

                fig = plot_results(df, res["clean_df"])

                # Capture click events on the chart
                chart_event = st.plotly_chart(fig, use_container_width=True, theme=None, on_select="rerun")

                clicked_dt_str = None
                if chart_event and "selection" in chart_event and chart_event["selection"]["points"]:
                    clicked_dt_str = chart_event["selection"]["points"][0]["x"]

                # --------------------------------------------------------
                # Per-Decision Agent Reasoning (new section)
                # --------------------------------------------------------
                if decisions:
                    st.markdown("---")
                    st.subheader("Per-Decision Agent Reasoning")
                    st.markdown(
                        "Select a flagged anomaly below to see the agent's "
                        "decision, reasoning, and internal deliberation."
                    )

                    ACTION_COLORS = {
                        "delete": "#dc2626",
                        "keep": "#16a34a",
                        "impute": "#2563eb",
                        "correct": "#d97706",
                        "flagged": "#ef4444",
                    }

                    options = [
                        f"#{d['id']}  {d['anomaly_type'].replace('_', ' ').title()} "
                        f"at {d['segment_start'][:16]}  \u2014  {d['action'].upper()}"
                        for d in decisions
                    ]

                    # Try to map the clicked point to a decision index
                    default_idx = 0
                    if clicked_dt_str:
                        # Convert to pandas datetime for robust comparison
                        try:
                            clicked_dt = pd.to_datetime(clicked_dt_str)
                            for i, d in enumerate(decisions):
                                d_start = pd.to_datetime(d["segment_start"])
                                d_end = pd.to_datetime(d["segment_end"])
                                if d_start <= clicked_dt <= d_end:
                                    default_idx = i
                                    break
                        except Exception:
                            pass

                    selected_idx = st.selectbox(
                        "Flagged anomaly:",
                        range(len(options)),
                        index=default_idx,
                        format_func=lambda i: options[i],
                        key=f"decision_select_{default_idx}" # Force update if click changes
                    )

                    d = decisions[selected_idx]
                    action = d["action"]
                    action_color = ACTION_COLORS.get(action, "#64748b")

                    # --- Zoomed neighbourhood chart ---
                    nb_start = pd.to_datetime(d["neighborhood_start"])
                    nb_end = pd.to_datetime(d["neighborhood_end"])
                    seg_start = pd.to_datetime(d["segment_start"])
                    seg_end = pd.to_datetime(d["segment_end"])

                    dt_col = pd.to_datetime(df["datetime"])
                    mask = (dt_col >= nb_start) & (dt_col <= nb_end)
                    nb_df = df.loc[mask].copy()

                    fig_zoom = go.Figure()
                    fig_zoom.add_trace(go.Scatter(
                        x=nb_df["datetime"], y=nb_df["value"],
                        mode="lines+markers",
                        name="series",
                        line=dict(color="rgba(100,116,139,0.6)", width=1.5),
                        marker=dict(size=4, color="#64748b"),
                        hovertemplate="%{x}<br>%{y:.2f} FNU<extra></extra>",
                    ))

                    # Highlight the flagged segment
                    seg_mask = (dt_col >= seg_start) & (dt_col <= seg_end)
                    seg_df = df.loc[seg_mask]
                    if not seg_df.empty and d.get("value") is not None:
                        fig_zoom.add_trace(go.Scatter(
                            x=seg_df["datetime"],
                            y=seg_df["value"],
                            mode="markers",
                            name=f"{d['anomaly_type']} ({action})",
                            marker=dict(
                                size=14, color=action_color,
                                symbol="circle-open", line=dict(width=3, color=action_color),
                            ),
                            hovertemplate=(
                                "%{x}<br>%{y:.2f} FNU<br>"
                                f"{d['anomaly_type']} \u2014 {action}<extra></extra>"
                            ),
                        ))

                    fig_zoom.update_layout(
                        height=300,
                        margin=dict(l=50, r=20, t=30, b=40),
                        xaxis=dict(title="", gridcolor="#1e293b", showgrid=True),
                        yaxis=dict(title="FNU", gridcolor="#1e293b", showgrid=True),
                        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                        font=dict(color="#cbd5e1"),
                        showlegend=True,
                        legend=dict(orientation="h", y=1.12, x=0, font=dict(color="#cbd5e1")),
                        hovermode="closest",
                    )
                    st.plotly_chart(fig_zoom, use_container_width=True, theme=None)

                    # --- Tabs & Decision Card ---
                    tab1, tab2, tab3 = st.tabs([
                        "Why this verdict",
                        f"What the Run Did ({res['run_summary'].steps} steps)",
                        "Measurements & Labels"
                    ])

                    with tab1:
                        params_str = d.get('params', '{}')
                        raw_reasoning = d.get('reasoning', '')
                        if ". " in raw_reasoning:
                            sentences = raw_reasoning.split(". ")
                            metrics_str = sentences[0]
                            prose_str = ". ".join(sentences[1:]).replace("\n", "<br>")

                            metrics_items = metrics_str.split(", ")
                            metrics_html = '<table style="width: 100%; margin-bottom: 1rem; border-collapse: collapse; font-size: 0.9rem;">'
                            for item in metrics_items:
                                if "=" in item:
                                    k, v = item.split("=", 1)
                                else:
                                    k, v = "observation", item
                                metrics_html += (
                                    f'<tr><td style="padding: 6px 10px; border: 1px solid #e2e8f0; font-weight: 600; '
                                    f'width: 35%; color: #475569;">{k.strip()}</td>'
                                    f'<td style="padding: 6px 10px; border: 1px solid #e2e8f0;">{v.strip()}</td></tr>'
                                )
                            metrics_html += '</table>'
                            reasoning_html = metrics_html + f'<p style="color: inherit; font-size: 0.95rem; line-height: 1.5;">{prose_str}</p>'
                        else:
                            reasoning_html = raw_reasoning.replace("\n", "<br>")
                        anomaly_type = d.get("anomaly_type", "unknown").replace('_', ' ')

                        st.markdown(f"**The run's answer: this is an anomaly \u2014 type {anomaly_type}.**")

                        st.markdown(
                            "<span style='color: #64748b; font-size: 0.75rem; font-weight: 700; "
                            "letter-spacing: 0.05em; text-transform: uppercase;'>WHAT FLAGGED IT</span>",
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f"The detector put this row forward as a candidate. A flag is a candidate, not the answer:\n"
                            f"- **{d['flagged_by']}** flagged it \u2014 called with parameters `{params_str}`"
                        )

                        st.markdown(
                            "<span style='color: #64748b; font-size: 0.75rem; font-weight: 700; "
                            "letter-spacing: 0.05em; text-transform: uppercase;'>AGENT REASONING</span>",
                            unsafe_allow_html=True
                        )

                        thinking_html = ""
                        if d.get("thinking"):
                            thinking_content = d['thinking'].replace('\n', '<br>')
                            thinking_html = f"""
<div style="border-left: 4px solid #22c55e; padding: 12px 16px; margin-top: 16px; background-color: rgba(34, 197, 94, 0.05); border-radius: 0 4px 4px 0;">
    <span style="color: #16a34a; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; display: block; margin-bottom: 6px;">THE AGENT'S WORKING</span>
    <div style="color: inherit; font-size: 0.95rem; line-height: 1.5;">{thinking_content}</div>
</div>
"""

                        content_html = f"""
<div style="color: inherit; font-size: 0.95rem; line-height: 1.5; margin-bottom: 1rem;">
    {reasoning_html}
    {thinking_html}
</div>
"""
                        st.markdown(content_html, unsafe_allow_html=True)

                    with tab2:
                        st.markdown(f"**Total Agent Steps:** {res['run_summary'].steps}")
                        st.markdown(f"**Total Tokens:** {res['run_summary'].input_tokens + res['run_summary'].output_tokens:,}")
                        st.markdown("This tab would contain the full sequence of tools the agent ran during this session.")

                    with tab3:
                        st.markdown(f"**Segment:** {d['segment_start']} to {d['segment_end']}")
                        st.markdown(f"**Data Points in Segment:** {d['n_points']}")
                        st.markdown("This tab would contain the raw segment dataframe and ground-truth comparison labels.")

                # 5. Report Panel
                st.header("5. Agent Report")
                st.info("The agent generated the following report based on its analysis:")
                st.markdown(res["report"])

                with st.expander("Run Summary (Tokens & Cost)"):
                    st.json({
                        "steps": res["run_summary"].steps,
                        "input_tokens": res["run_summary"].input_tokens,
                        "output_tokens": res["run_summary"].output_tokens,
                        "est_cost_usd": res["run_summary"].est_cost_usd,
                        "log_path": res["run_summary"].log_path
                    })

                # 6. Download Panel
                st.header("6. Download Results")

                # Generate Flag JSON
                flags = []
                if res["clean_df"] is not None:
                    for _, row in res["clean_df"].iterrows():
                        if pd.notna(row.get("flag")):
                            dt = row.get("datetime", row.name)
                            flags.append({
                                "datetime": str(dt),
                                "flagged_by": str(row["flag"]),
                                "action": "flag",
                                "reason": ""
                            })

                csv_data = res["clean_df"].to_csv(index=False).encode('utf-8')
                json_data = json.dumps(flags, indent=2).encode('utf-8')

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download Cleaned CSV",
                        data=csv_data,
                        file_name=f"{res['filename'].replace('.csv', '')}_clean.csv",
                        mime="text/csv"
                    )
                with col2:
                    st.download_button(
                        label="Download Flag Log (JSON)",
                        data=json_data,
                        file_name=f"{res['filename'].replace('.csv', '')}_flags.json",
                        mime="application/json"
                    )

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        st.info("Please upload a CSV file to begin.")

else:
    st.header("Past Uploads & Runs")
    if not st.session_state.history:
        st.info("No runs yet in this session.")
    else:
        for idx, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"{item['timestamp']} - {item['filename']}"):
                st.write(f"**Elapsed:** {item['elapsed']:.1f}s | **Steps:** {item['run_summary'].steps}")
                st.markdown(item["report"])

                # Let user re-download from history
                flags = []
                for _, row in item["clean_df"].iterrows():
                    if pd.notna(row.get("flag")):
                        dt = row.get("datetime", row.name)
                        flags.append({
                            "datetime": str(dt),
                            "flagged_by": str(row["flag"]),
                            "action": "flag",
                            "reason": ""
                        })

                csv_data = item["clean_df"].to_csv(index=False).encode('utf-8')
                json_data = json.dumps(flags, indent=2).encode('utf-8')

                hc1, hc2 = st.columns(2)
                with hc1:
                    st.download_button(
                        label=f"Download CSV",
                        data=csv_data,
                        file_name=f"{item['filename'].replace('.csv', '')}_clean.csv",
                        mime="text/csv",
                        key=f"hist_csv_{idx}"
                    )
                with hc2:
                    st.download_button(
                        label=f"Download Flags",
                        data=json_data,
                        file_name=f"{item['filename'].replace('.csv', '')}_flags.json",
                        mime="application/json",
                        key=f"hist_json_{idx}"
                    )
