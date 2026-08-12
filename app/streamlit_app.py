import streamlit as st
import pandas as pd
import json
import tempfile
import os
import time
from pathlib import Path
import plotly.graph_objects as go
import saqc

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

def mock_run_agent(qc, max_steps, log_dir):
    """A mock version of run_agent for testing without an API key."""
    # Simulate processing time
    with st.spinner("Mock agent is thinking..."):
        time.sleep(2)
        
    df = qc.data.to_pandas().copy()
    if df.index.name != "datetime":
        df = df.reset_index()
        
    df["flag"] = None
    
    # Randomly flag some data just for visualization if any data exists
    if len(df) > 0:
        mid_point = len(df) // 2
        if mid_point > 0:
            df.iloc[mid_point:mid_point+5, df.columns.get_loc("flag")] = "flag_spike_unilof"
            if "value" in df.columns:
                df.iloc[mid_point:mid_point+5, df.columns.get_loc("value")] = None

    report = "Mock Report:\n\nThe agent successfully ran in mock mode. No real API calls were made.\nFound 5 simulated spikes."
    
    summary = RunSummary(
        steps=2,
        input_tokens=1000,
        output_tokens=200,
        est_cost_usd=0.005,
        log_path="logs/mock_run.jsonl"
    )
    return qc, df, report, summary

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
        anomalies = clean[clean["flag"].notna()]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies["datetime"],
                y=orig.loc[anomalies.index, "value"] if "value" in orig.columns else anomalies["value"],
                mode='markers',
                name='Flagged',
                marker=dict(color='red', size=8, symbol='x'),
                text=anomalies["flag"],
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
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="white", bordercolor="#e5e5e5", borderwidth=1)
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
                            if use_mock:
                                final_qc, clean_df, report, run_summary = mock_run_agent(qc, max_steps=max_steps, log_dir="logs")
                            else:
                                final_qc, clean_df, report, run_summary = run_agent(qc, max_steps=max_steps, log_dir="logs")
                                
                            elapsed = time.time() - start_time
                            
                            res = {
                                "filename": uploaded_file.name,
                                "clean_df": clean_df,
                                "report": report,
                                "run_summary": run_summary,
                                "elapsed": elapsed,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            st.session_state.run_results = res
                            st.session_state.history.append(res)
                            
                        except Exception as e:
                            st.error(f"An error occurred during the agent run: {str(e)}")

            if st.session_state.run_results is not None:
                res = st.session_state.run_results
                st.success(f"QC completed in {res['elapsed']:.1f} seconds! (Used {res['run_summary'].steps} tool calls)")
                
                # 4. Results Plot
                st.header("4. Results Visualization")
                fig = plot_results(df, res["clean_df"])
                st.plotly_chart(fig, use_container_width=True, theme=None)
                
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
