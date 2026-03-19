"""
Centralized output path management for the NPS analysis pipeline.

Defines the standard directory structure:

    output/
    ├── reports/{report_group}/                     # Final deliverables (adaptive cards)
    │   ├── interpreter_{period_range}_{execution_date}.json
    │   └── summarizer_{period_range}_{execution_date}.json
    └── logging/{report_group}/                     # Debug / audit trail
        ├── summarizer/{period_range}/
        ├── interpreter/{period_range}/
        └── causal/{period_range}/{node_path}/

report_group is derived from focus_touchpoint ("general" when None, "cabin-crew" for "Cabin Crew", etc.).
period_range is "{start_date}_{end_date}" (e.g. "2025-03-10_2025-03-17").
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


OUTPUT_DIR_NAME = "output"


def get_output_base() -> Path:
    return Path.cwd() / OUTPUT_DIR_NAME


def resolve_report_group(focus_touchpoint: Optional[str] = None) -> str:
    """Normalize the report group name to lowercase-hyphenated form.

    Examples: None → "general", "Cabin Crew" → "cabin-crew"
    """
    raw = focus_touchpoint if focus_touchpoint else "General"
    return raw.strip().lower().replace(" ", "-")


def format_period_range(
    date_param: Optional[str] = None,
    start_date=None,
    end_date=None,
) -> str:
    """Normalise various date representations into a standard period range string.

    Handles:
      - "2025-03-10 to 2025-03-17"  →  "2025-03-10_2025-03-17"
      - start_date + end_date (datetime or str)
      - Single date string           →  "2025-03-17_2025-03-17"
    """
    if date_param and " to " in str(date_param):
        parts = str(date_param).split(" to ")
        return f"{parts[0].strip()}_{parts[1].strip()}"

    if start_date and end_date:
        s = start_date.strftime("%Y-%m-%d") if hasattr(start_date, "strftime") else str(start_date)
        e = end_date.strftime("%Y-%m-%d") if hasattr(end_date, "strftime") else str(end_date)
        return f"{s}_{e}"

    if date_param:
        clean = str(date_param).strip()
        return f"{clean}_{clean}"

    today = datetime.now().strftime("%Y-%m-%d")
    return f"{today}_{today}"


# ---------------------------------------------------------------------------
# Local path builders
# ---------------------------------------------------------------------------

def get_report_path(
    report_group: str,
    agent_type: str,
    period_range: str,
    execution_date: Optional[str] = None,
) -> Path:
    """Path for a report file (adaptive card).

    agent_type: "interpreter" | "summarizer"
    execution_date: YYYY-MM-DD of the run (appended to filename when given).
    """
    exe = execution_date or datetime.now().strftime("%Y-%m-%d")
    return get_output_base() / "reports" / report_group / f"{agent_type}_{period_range}_{exe}.json"


def get_logging_path(
    report_group: str,
    agent_type: str,
    period_range: str,
    filename: str,
    node_path: Optional[str] = None,
) -> Path:
    """Path for a logging / debug file.

    agent_type: "interpreter" | "summarizer" | "causal"
    node_path:  only for causal – e.g. "Global_SH_IB"
    """
    base = get_output_base() / "logging" / report_group / agent_type / period_range
    if node_path:
        base = base / node_path
    return base / filename


# ---------------------------------------------------------------------------
# S3 key builders (mirror the local structure under the existing prefix)
# ---------------------------------------------------------------------------

def get_s3_report_key(
    base_prefix: str,
    report_group: str,
    agent_type: str,
    period_range: str,
    execution_date: Optional[str] = None,
) -> str:
    exe = execution_date or datetime.now().strftime("%Y-%m-%d")
    return f"{base_prefix}{report_group}/{agent_type}_{period_range}_{exe}.json"


def get_s3_logging_key(
    base_prefix: str,
    report_group: str,
    agent_type: str,
    period_range: str,
    filename: str,
    node_path: Optional[str] = None,
) -> str:
    key = f"{base_prefix}logging/{report_group}/{agent_type}/{period_range}"
    if node_path:
        key += f"/{node_path}"
    return f"{key}/{filename}"


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

def build_execution_metadata(
    model: str,
    study_mode: str = "comparative",
    anomaly_detection_mode: str = "vslast",
    causal_filter: Optional[str] = None,
    comparison_start_date: Optional[str] = None,
    comparison_end_date: Optional[str] = None,
    aggregation_days: int = 7,
    baseline_periods: int = 7,
    segment: str = "Global",
    focus_touchpoint: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a standard metadata dict to embed in every saved file."""
    meta: Dict[str, Any] = {
        "model": model,
        "study_mode": study_mode,
        "anomaly_detection_mode": anomaly_detection_mode,
        "aggregation_days": aggregation_days,
        "baseline_periods": baseline_periods,
        "segment": segment,
        "execution_timestamp": datetime.now().isoformat() + "Z",
    }
    if causal_filter:
        meta["causal_filter"] = causal_filter
    if comparison_start_date:
        meta["comparison_period"] = {
            "start": str(comparison_start_date),
            "end": str(comparison_end_date),
        }
    if focus_touchpoint:
        meta["focus_touchpoint"] = focus_touchpoint
    meta.update(extra)
    return meta


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_minified_json(path: Path, data: Any) -> None:
    """Write *data* as minified JSON (no indent, compact separators)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            pass
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def save_pretty_json(path: Path, data: Any) -> None:
    """Write *data* as pretty-printed JSON (for logging / debug files)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
