"""Tests for src/agent_tools/schemas.py — Anthropic tool schema definitions.

These tests require only the stdlib + the schemas module itself (no saqc, no
pandas), so they run in any Python environment including Python 3.13.

What is verified:
  - Every schema has the three fields the Anthropic Messages API requires.
  - Every schema's input_schema is a valid JSON-schema object.
  - Required params listed in 'required' are actually defined in 'properties'.
  - The full schema list is JSON-serialisable (Anthropic rejects non-serialisable tools).
  - The expected tool names are present and no unexpected extras snuck in.
  - correct_drift and drift-related tools are absent (§9.2 — drift removed).
  - TOOL_SCHEMA_BY_NAME is a consistent index of TOOL_SCHEMAS.
"""

import json

import pytest

from src.agent_tools.schemas import TOOL_SCHEMAS, TOOL_SCHEMA_BY_NAME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_TOOLS = {
    # Utility
    "inspect_dataset",
    "get_flag_summary",
    "export_clean_data",
    # Detection
    "flag_range",
    "flag_constants",
    "flag_plateau",
    "flag_spike_unilof",
    "flag_zscore",
    "flag_jumps",
    "flag_nan",
    # Context — the two aggregators, plus the primitives that answer the §6
    # decisions the aggregate blurs: width, recovery time, level shift, and local
    # noise (the one that measures the STRETCH rather than the point). The other
    # four primitives stay library-only; describe_point already returns them all.
    "describe_point",
    "describe_points",
    "slope_context",
    "excursion_context",
    "recovery_context",
    "level_shift_context",
    "noise_context",
    # Action
    "impute_rolling",
}

REMOVED_TOOLS = {
    "correct_drift",         # §9.2 — drift removed
    "flag_drift",
    "correctDrift",
}


# ---------------------------------------------------------------------------
# Top-level list sanity
# ---------------------------------------------------------------------------

def test_schema_count():
    assert len(TOOL_SCHEMAS) == len(EXPECTED_TOOLS), (
        f"Expected {len(EXPECTED_TOOLS)} schemas, got {len(TOOL_SCHEMAS)}. "
        f"Missing: {EXPECTED_TOOLS - {s['name'] for s in TOOL_SCHEMAS}}"
    )


def test_expected_tool_names_present():
    names = {s["name"] for s in TOOL_SCHEMAS}
    missing = EXPECTED_TOOLS - names
    assert not missing, f"Missing tool schemas: {missing}"


def test_no_drift_tools():
    """correct_drift and drift-related tools must not appear (§9.2)."""
    names = {s["name"] for s in TOOL_SCHEMAS}
    present = REMOVED_TOOLS & names
    assert not present, f"Drift tools must be absent but found: {present}"


def test_no_duplicate_names():
    names = [s["name"] for s in TOOL_SCHEMAS]
    assert len(names) == len(set(names)), "Duplicate tool names in TOOL_SCHEMAS"


# ---------------------------------------------------------------------------
# Per-schema structure (Anthropic Messages API contract)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda s: s["name"])
def test_schema_has_required_top_level_keys(schema):
    """Anthropic requires: name, description, input_schema."""
    for key in ("name", "description", "input_schema"):
        assert key in schema, f"{schema.get('name', '?')} missing top-level key '{key}'"


@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda s: s["name"])
def test_name_is_non_empty_string(schema):
    assert isinstance(schema["name"], str) and schema["name"].strip()


@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda s: s["name"])
def test_description_is_non_empty_string(schema):
    name = schema["name"]
    desc = schema["description"]
    assert isinstance(desc, str) and len(desc.strip()) > 10, (
        f"{name}: description is too short or empty"
    )


@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda s: s["name"])
def test_input_schema_is_object_type(schema):
    """Anthropic requires input_schema to be a JSON Schema of type 'object'."""
    isc = schema["input_schema"]
    assert isinstance(isc, dict), f"{schema['name']}: input_schema must be a dict"
    assert isc.get("type") == "object", (
        f"{schema['name']}: input_schema.type must be 'object', got {isc.get('type')!r}"
    )


@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda s: s["name"])
def test_input_schema_has_properties(schema):
    isc = schema["input_schema"]
    assert "properties" in isc, f"{schema['name']}: input_schema must have 'properties'"
    assert isinstance(isc["properties"], dict)


@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda s: s["name"])
def test_required_params_exist_in_properties(schema):
    """Every param listed in 'required' must be defined in 'properties'."""
    isc = schema["input_schema"]
    required = isc.get("required", [])
    properties = isc.get("properties", {})
    missing = [r for r in required if r not in properties]
    assert not missing, (
        f"{schema['name']}: required params {missing} not defined in properties"
    )


@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda s: s["name"])
def test_each_property_has_type_and_description(schema):
    """Each property must have at least 'type' and 'description'."""
    for prop_name, prop_def in schema["input_schema"]["properties"].items():
        assert "type" in prop_def, (
            f"{schema['name']}.{prop_name}: missing 'type'"
        )
        assert "description" in prop_def and prop_def["description"].strip(), (
            f"{schema['name']}.{prop_name}: missing or empty 'description'"
        )


# ---------------------------------------------------------------------------
# JSON serialisability — Anthropic will reject schemas that aren't
# ---------------------------------------------------------------------------

def test_full_schema_list_is_json_serialisable():
    try:
        json.dumps(TOOL_SCHEMAS)
    except (TypeError, ValueError) as e:
        pytest.fail(f"TOOL_SCHEMAS is not JSON-serialisable: {e}")


@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda s: s["name"])
def test_individual_schema_is_json_serialisable(schema):
    try:
        json.dumps(schema)
    except (TypeError, ValueError) as e:
        pytest.fail(f"{schema['name']} schema is not JSON-serialisable: {e}")


# ---------------------------------------------------------------------------
# TOOL_SCHEMA_BY_NAME consistency
# ---------------------------------------------------------------------------

def test_schema_by_name_covers_all_tools():
    names_in_list = {s["name"] for s in TOOL_SCHEMAS}
    names_in_dict = set(TOOL_SCHEMA_BY_NAME.keys())
    assert names_in_list == names_in_dict, (
        f"TOOL_SCHEMA_BY_NAME is out of sync with TOOL_SCHEMAS. "
        f"Extra in dict: {names_in_dict - names_in_list}. "
        f"Missing from dict: {names_in_list - names_in_dict}."
    )


def test_schema_by_name_values_match_list():
    for name, schema in TOOL_SCHEMA_BY_NAME.items():
        assert schema["name"] == name, (
            f"TOOL_SCHEMA_BY_NAME['{name}'] has name='{schema['name']}'"
        )


# ---------------------------------------------------------------------------
# Spot-check key parameter details from §7.2
# ---------------------------------------------------------------------------

def test_flag_spike_unilof_has_thresh_and_n():
    s = TOOL_SCHEMA_BY_NAME["flag_spike_unilof"]
    props = s["input_schema"]["properties"]
    assert "thresh" in props
    assert "n" in props
    assert props["n"].get("default") == 20


def test_flag_constants_thresh_default_small():
    """§7.1: thresh must be much smaller than signal noise sd — default should be ≤ 0.05."""
    s = TOOL_SCHEMA_BY_NAME["flag_constants"]
    thresh_prop = s["input_schema"]["properties"]["thresh"]
    default = thresh_prop.get("default")
    assert default is not None and default <= 0.05, (
        f"flag_constants thresh default should be ≤ 0.05, got {default}"
    )


def test_flag_jumps_required_params():
    """§7: flag_jumps requires both thresh and window."""
    s = TOOL_SCHEMA_BY_NAME["flag_jumps"]
    required = s["input_schema"].get("required", [])
    assert "thresh" in required
    assert "window" in required


def test_impute_rolling_window_required():
    """§7.1: window must exceed the gap length — must be required, not optional."""
    s = TOOL_SCHEMA_BY_NAME["impute_rolling"]
    required = s["input_schema"].get("required", [])
    assert "window" in required


def test_flag_zscore_window_required():
    s = TOOL_SCHEMA_BY_NAME["flag_zscore"]
    required = s["input_schema"].get("required", [])
    assert "window" in required


def test_flag_range_has_min_and_max():
    s = TOOL_SCHEMA_BY_NAME["flag_range"]
    props = s["input_schema"]["properties"]
    assert "min" in props and "max" in props
