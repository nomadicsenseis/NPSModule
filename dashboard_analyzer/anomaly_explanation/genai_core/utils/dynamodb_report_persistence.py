"""
DynamoDB persistence for CATIA Proactiva agent reports.

Writes structured logging data from causal, interpreter, and summarizer agents
into two DynamoDB tables:
  - CUSTOMER_CATIA_CAUSAL_REPORTS    (one item per node_path per execution)
  - CUSTOMER_CATIA_SYNTHESIS_REPORT  (one item per agent_type per execution)

Uses the same AWS session resolver as the rest of the pipeline.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from .aws_session import get_aws_session

logger = logging.getLogger(__name__)

CAUSAL_TABLE = os.getenv("DYNAMODB_CAUSAL_TABLE", "CUSTOMER_CATIA_CAUSAL_REPORTS")
SYNTHESIS_TABLE = os.getenv("DYNAMODB_SYNTHESIS_TABLE", "CUSTOMER_CATIA_SYNTHESIS_REPORT")
DEFAULT_TTL_SECONDS = int(os.getenv("DYNAMODB_REPORT_TTL_SECONDS", "7776000"))  # 90 days
MAX_ITEM_BYTES = 350_000  # safety margin under DynamoDB's 400 KB limit


def _safe_json(obj: Any) -> str:
    """Serialize *obj* to a JSON string, returning '{}' / '[]' on failure."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(obj)


def _safe_str(value: Any, max_length: int = 0) -> Optional[str]:
    """Convert *value* to string, optionally truncating."""
    if value is None:
        return None
    s = str(value)
    if max_length and len(s) > max_length:
        return s[:max_length] + "...[truncated]"
    return s


def _safe_number(value: Any) -> Optional[Decimal]:
    """Convert *value* to Decimal for DynamoDB, returning None on failure."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _ttl_epoch(ttl_seconds: int = DEFAULT_TTL_SECONDS) -> int:
    return int(time.time()) + ttl_seconds


def _strip_none(d: dict) -> dict:
    """Remove keys whose value is None (DynamoDB rejects explicit None)."""
    return {k: v for k, v in d.items() if v is not None}


def _parse_node_path(node_path: str) -> dict:
    """Derive haul / cabin / company from a node_path like 'Global/SH/Economy/IB'."""
    parts = node_path.split("/") if node_path else []
    haul = parts[1] if len(parts) >= 2 and parts[1] in ("LH", "SH") else None
    cabin_raw = parts[2] if len(parts) >= 3 else None
    cabin = "Premium EC" if cabin_raw == "Premium" else cabin_raw
    company = parts[3] if len(parts) >= 4 and parts[3] in ("IB", "YW") else None
    segment = "/".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "Global")
    return {"segment": segment, "haul": haul, "cabin": cabin, "company": company}


class DynamoDBReportPersistence:
    """Writes agent report items to DynamoDB."""

    def __init__(
        self,
        environment: str = "prod",
        region_name: str = "eu-west-1",
        causal_table: str = CAUSAL_TABLE,
        synthesis_table: str = SYNTHESIS_TABLE,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        self.environment = environment
        self.causal_table = causal_table
        self.synthesis_table = synthesis_table
        self.ttl_seconds = ttl_seconds

        session = get_aws_session(environment=environment, use_sandbox=False)
        self.dynamodb = session.resource("dynamodb", region_name=region_name)
        self._causal = self.dynamodb.Table(causal_table)
        self._synthesis = self.dynamodb.Table(synthesis_table)
        logger.info(
            f"[DynamoDB] Report persistence ready — causal={causal_table}, "
            f"synthesis={synthesis_table}, env={environment}"
        )

    # ------------------------------------------------------------------
    # Causal agent
    # ------------------------------------------------------------------

    def save_causal_report(
        self,
        *,
        execution_id: str,
        node_path: str,
        agent: Any,
        analysis_start_date: Optional[str] = None,
        analysis_end_date: Optional[str] = None,
        final_synthesis: Optional[str] = None,
        execution_duration_ms: Optional[float] = None,
        status: str = "completed",
        error_message: Optional[str] = None,
        s3_report_key: Optional[str] = None,
    ) -> Optional[str]:
        """Persist a single causal-agent run (one node_path) to DynamoDB.

        Reads structured data directly from the agent's internal state
        (``collected_data``, ``tracker``, etc.) — no agent changes needed.

        Key schema:
          PK  ``REPORT_ID``       = causal_explanation#{node_path}#{start}#{end}
          SK  ``EXPORT_TIMESTAMP`` = ISO-8601 timestamp (allows multiple runs)

        Returns the ``REPORT_ID`` on success, ``None`` on failure.
        """
        start_part = analysis_start_date or "unknown"
        end_part = analysis_end_date or "unknown"
        report_id = f"causal_explanation#{node_path}#{start_part}#{end_part}"
        export_timestamp = datetime.utcnow().isoformat()
        np = _parse_node_path(node_path)

        collected = getattr(agent, "collected_data", {}) or {}
        tracker = getattr(agent, "tracker", None)

        # Build verbatims: prefer 'verbatims', fallback to 'verbatims_conversation'
        verbatims_raw = collected.get("verbatims") or collected.get("verbatims_conversation")

        # survey_count lives inside explanatory_drivers dict
        exp_drivers = collected.get("explanatory_drivers")
        survey_count = None
        if isinstance(exp_drivers, dict):
            survey_count = exp_drivers.get("survey_count")

        comp_start = None
        comp_end = None
        if hasattr(agent, "comparison_start_date") and agent.comparison_start_date:
            comp_start = (
                agent.comparison_start_date.strftime("%Y-%m-%d")
                if hasattr(agent.comparison_start_date, "strftime")
                else str(agent.comparison_start_date)
            )
        if hasattr(agent, "comparison_end_date") and agent.comparison_end_date:
            comp_end = (
                agent.comparison_end_date.strftime("%Y-%m-%d")
                if hasattr(agent.comparison_end_date, "strftime")
                else str(agent.comparison_end_date)
            )

        tools_used = []
        if tracker:
            tools_used = list(set(
                msg.get("metadata", {}).get("tool_name")
                for msg in getattr(tracker, "conversation_log", [])
                if msg.get("metadata", {}).get("tool_name")
            ))

        item = _strip_none({
            # Keys (PK + SK)
            "REPORT_ID": report_id,
            "EXPORT_TIMESTAMP": export_timestamp,
            # Correlation
            "execution_id": execution_id,
            # Segment dimensions
            "node_path": node_path,
            "segment": np["segment"],
            "haul": np["haul"],
            "cabin": np["cabin"],
            "company": np["company"],
            # Execution context
            "execution_timestamp": export_timestamp,
            "analysis_start_date": analysis_start_date,
            "analysis_end_date": analysis_end_date,
            "comparison_start_date": comp_start,
            "comparison_end_date": comp_end,
            "causal_filter": getattr(agent, "causal_filter", None),
            "anomaly_type": getattr(agent, "current_anomaly_type", None),
            "anomaly_detection_mode": getattr(agent, "detection_mode", None),
            "study_mode": getattr(agent, "study_mode", None),
            "focus_touchpoint": getattr(agent, "focus_touchpoint", None),
            "report_group": getattr(agent, "report_group", None),
            "llm_type": agent.llm_type.value if hasattr(agent, "llm_type") else None,
            "environment": getattr(agent, "environment", None),
            # Tool results
            "explanatory_drivers_result": _safe_json(exp_drivers),
            "operative_data_result": _safe_json(collected.get("operative_data")),
            "ncs_result": _safe_json(collected.get("ncs_data")),
            "routes_result": _safe_json(collected.get("routes_data")),
            "verbatims_result": _safe_json(verbatims_raw),
            "customer_profile_result": _safe_json(collected.get("customer_profile")),
            "focus_touchpoint_result": _safe_json(collected.get("focus_touchpoint_enrichment")),
            # Final output
            "final_synthesis": final_synthesis,
            "identified_routes": getattr(tracker, "identified_routes", []) or [],
            "survey_count": _safe_number(survey_count),
            # Operational metrics
            "tools_used": tools_used if tools_used else None,
            "iteration_count": _safe_number(getattr(tracker, "iteration_count", None)),
            "total_messages": _safe_number(
                len(getattr(tracker, "conversation_log", []))
            ),
            "investigation_success": len(getattr(tracker, "previous_explanations", [])) > 0,
            "dax_queries_count": _safe_number(
                len(getattr(tracker, "dax_queries", []))
            ),
            "execution_duration_ms": _safe_number(execution_duration_ms),
            "status": status,
            "error_message": error_message,
            "s3_report_key": s3_report_key,
            "conversation_log": _safe_json(getattr(tracker, "conversation_log", [])),
            "ttl": _ttl_epoch(self.ttl_seconds),
        })

        return self._put_item(self._causal, item, "causal", report_id)

    # ------------------------------------------------------------------
    # Interpreter agent
    # ------------------------------------------------------------------

    def save_interpreter_report(
        self,
        *,
        execution_id: str,
        agent: Any,
        step_responses: Optional[List[Dict]] = None,
        final_interpretation: Optional[str] = None,
        final_adaptive_card_json: Optional[str] = None,
        nma_list: Optional[List[str]] = None,
        execution_duration_ms: Optional[float] = None,
        status: str = "completed",
        error_message: Optional[str] = None,
        s3_report_key: Optional[str] = None,
        source_causal_report_ids: Optional[List[str]] = None,
        analysis_date: Optional[str] = None,
        period_type: str = "weekly",
    ) -> Optional[str]:
        """Persist an interpreter run to DynamoDB."""
        synthesis_id = str(uuid.uuid4())
        tracker = getattr(agent, "conversation_tracker", None)

        # Map step_responses list to named columns
        step_map = {}
        for sr in (step_responses or []):
            step_map[sr.get("step", "")] = sr.get("content", "")

        comp_start = getattr(agent, "comparison_start_date", None)
        comp_end = getattr(agent, "comparison_end_date", None)

        item = _strip_none({
            # Keys
            "synthesis_id": synthesis_id,
            "agent_type": "interpreter",
            # Correlation
            "execution_id": execution_id,
            # Context
            "execution_timestamp": datetime.utcnow().isoformat() + "Z",
            "analysis_date": analysis_date,
            "period_type": period_type,
            "period_range": None,  # caller can set
            "analysis_start_date": _safe_str(comp_start),
            "analysis_end_date": _safe_str(comp_end),
            "comparison_start_date": _safe_str(comp_start),
            "comparison_end_date": _safe_str(comp_end),
            "causal_filter": getattr(agent, "causal_filter", None),
            "anomaly_detection_mode": getattr(agent, "anomaly_detection_mode", None),
            "study_mode": getattr(agent, "study_mode", None),
            "segment": getattr(agent, "segment", "Global"),
            "focus_touchpoint": getattr(agent, "focus_touchpoint", None),
            "report_group": getattr(agent, "report_group", None),
            "llm_type": agent.llm_type.value if hasattr(agent, "llm_type") else None,
            "aggregation_days": _safe_number(getattr(agent, "aggregation_days", None)),
            "baseline_periods": _safe_number(getattr(agent, "baseline_periods", None)),
            "environment": getattr(agent, "environment", None),
            # Interpreter steps
            "step1_company_level_diagnosis": step_map.get("COMPANY_LEVEL_DIAGNOSIS"),
            "step2_cabin_level_diagnosis": step_map.get("CABIN_LEVEL_DIAGNOSIS"),
            "step3_radio_global_diagnosis": step_map.get("RADIO_GLOBAL_DIAGNOSIS"),
            "step4_nma_identification": step_map.get("NMA_IDENTIFICATION"),
            "step4b_evidence_extraction": step_map.get("EVIDENCE_EXTRACTION"),
            "step4c_cabin_radio_reflection": step_map.get("CABIN_RADIO_REFLECTION"),
            "step5_executive_synthesis": step_map.get("EXECUTIVE_SYNTHESIS"),
            "step6_adaptive_card_json": step_map.get("ADAPTIVE_CARD_GENERATION"),
            "step7_tone_modernization": step_map.get("TONE_MODERNIZATION"),
            # Interpreter metrics
            "hierarchy_total_nodes": _safe_number(
                len(getattr(tracker, "hierarchy_structure", {}) or {})
            ),
            "hierarchy_generations": _safe_number(
                len(set(
                    r.get("generation", 0)
                    for r in getattr(agent, "hierarchical_reflections", [])
                )) if getattr(agent, "hierarchical_reflections", None) else None
            ),
            "hierarchy_summary": _safe_json(
                tracker.get_hierarchy_summary() if tracker and hasattr(tracker, "get_hierarchy_summary") else None
            ),
            "nma_list": nma_list,
            "generation_reflections": _safe_json(
                getattr(agent, "hierarchical_reflections", None)
            ),
            # Summarizer-specific columns (null for interpreter)
            # Common finals
            "final_adaptive_card_json": final_adaptive_card_json,
            "final_executive_synthesis": final_interpretation,
            "llm_calls_count": None,
            "execution_duration_ms": _safe_number(execution_duration_ms),
            "conversation_log": _safe_json(
                getattr(tracker, "conversation_log", []) if tracker else []
            ),
            "status": status,
            "error_message": error_message,
            "s3_report_key": s3_report_key,
            "source_causal_report_ids": source_causal_report_ids,
            "ttl": _ttl_epoch(self.ttl_seconds),
        })

        return self._put_item(self._synthesis, item, "interpreter", synthesis_id)

    # ------------------------------------------------------------------
    # Summarizer agent
    # ------------------------------------------------------------------

    def save_summarizer_report(
        self,
        *,
        execution_id: str,
        agent: Any,
        all_conversations: Optional[Dict[str, Any]] = None,
        optimization_debug: Optional[Dict[str, Any]] = None,
        final_adaptive_card_json: Optional[str] = None,
        final_executive_synthesis: Optional[str] = None,
        adaptive_card_size_kb: Optional[float] = None,
        daily_analysis_dates: Optional[List[str]] = None,
        execution_duration_ms: Optional[float] = None,
        status: str = "completed",
        error_message: Optional[str] = None,
        s3_report_key: Optional[str] = None,
        source_causal_report_ids: Optional[List[str]] = None,
        analysis_date: Optional[str] = None,
        period_type: str = "weekly",
    ) -> Optional[str]:
        """Persist a summarizer run to DynamoDB."""
        synthesis_id = str(uuid.uuid4())
        convs = all_conversations or {}
        opt = optimization_debug or {}

        comp_start = getattr(agent, "comparison_start_date", None)
        comp_end = getattr(agent, "comparison_end_date", None)

        # Extract optimization sub-step results
        step6_opts = opt.get("step6_size_optimization", [])
        step6_map = {}
        for entry in (step6_opts if isinstance(step6_opts, list) else []):
            step_name = entry.get("step", "")
            step6_map[step_name] = _safe_json(entry)

        item = _strip_none({
            # Keys
            "synthesis_id": synthesis_id,
            "agent_type": "summarizer",
            # Correlation
            "execution_id": execution_id,
            # Context
            "execution_timestamp": datetime.utcnow().isoformat() + "Z",
            "analysis_date": analysis_date,
            "period_type": period_type,
            "period_range": None,
            "analysis_start_date": _safe_str(comp_start),
            "analysis_end_date": _safe_str(comp_end),
            "comparison_start_date": _safe_str(comp_start),
            "comparison_end_date": _safe_str(comp_end),
            "causal_filter": getattr(agent, "causal_filter", None),
            "anomaly_detection_mode": getattr(agent, "anomaly_detection_mode", None),
            "study_mode": getattr(agent, "study_mode", None),
            "segment": getattr(agent, "segment", "Global"),
            "focus_touchpoint": getattr(agent, "focus_touchpoint", None),
            "report_group": getattr(agent, "report_group", None),
            "llm_type": agent.llm_type.value if hasattr(agent, "llm_type") else None,
            "aggregation_days": _safe_number(getattr(agent, "aggregation_days", None)),
            "baseline_periods": _safe_number(getattr(agent, "baseline_periods", None)),
            "environment": getattr(agent, "environment", None),
            # Summarizer main steps
            "summ_step1_section_connections": _safe_json(convs.get("step1_section_connections")),
            "summ_step2_full_report": _safe_json(convs.get("step2_full_report")),
            "summ_step3_executive_synthesis": _safe_json(convs.get("step3_executive_synthesis")),
            "summ_step4_adaptive_card": _safe_json(convs.get("step4_adaptive_card")),
            # Summarizer optimization steps
            "summ_step5_modernize_tone": _safe_json(opt.get("step5_modernize_tone")),
            "summ_step6a_trim_low_impact_days": step6_map.get("step6a_trim_low_impact_days"),
            "summ_step6b_remove_subsegment_daily": step6_map.get("step6b_remove_subsegment_daily_context"),
            "summ_step6c_summarize_cabin_haul": step6_map.get("step6c_summarize_cabin_haul_company_weekly"),
            "summ_step6d_shorten_cabin_haul": step6_map.get("step6d_shorten_cabin_haul_weekly"),
            "summ_step6e_shorten_overall_global": step6_map.get("step6e_shorten_overall_global"),
            "summ_step7_validate_and_fix_json": _safe_json(opt.get("step7_validate_and_fix_json")),
            # Summarizer metrics
            "adaptive_card_size_kb": _safe_number(adaptive_card_size_kb),
            "num_daily_analyses_included": _safe_number(
                len(daily_analysis_dates) if daily_analysis_dates else None
            ),
            "daily_analysis_dates": daily_analysis_dates,
            # Common finals
            "final_adaptive_card_json": final_adaptive_card_json,
            "final_executive_synthesis": final_executive_synthesis,
            "llm_calls_count": None,
            "execution_duration_ms": _safe_number(execution_duration_ms),
            "conversation_log": None,  # summarizer stores steps individually
            "status": status,
            "error_message": error_message,
            "s3_report_key": s3_report_key,
            "source_causal_report_ids": source_causal_report_ids,
            "ttl": _ttl_epoch(self.ttl_seconds),
        })

        return self._put_item(self._synthesis, item, "summarizer", synthesis_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _put_item(
        self, table, item: dict, label: str, item_id: str
    ) -> Optional[str]:
        """Write an item, handling size limits and errors."""
        try:
            raw_size = len(json.dumps(item, default=str).encode("utf-8"))
            if raw_size > MAX_ITEM_BYTES:
                logger.warning(
                    f"[DynamoDB] {label} item {item_id} is {raw_size / 1024:.1f} KB, "
                    f"trimming large fields"
                )
                item = self._trim_large_fields(item, raw_size)

            table.put_item(Item=item)
            logger.info(f"[DynamoDB] ✅ {label} report saved: {item_id}")
            return item_id
        except ClientError as e:
            logger.error(f"[DynamoDB] ❌ Failed to save {label} report: {e}")
            return None
        except Exception as e:
            logger.error(f"[DynamoDB] ❌ Unexpected error saving {label} report: {e}")
            return None

    @staticmethod
    def _trim_large_fields(item: dict, current_size: int) -> dict:
        """Progressively trim the largest string fields to fit under the limit."""
        TRIM_CANDIDATES = [
            "conversation_log",
            "generation_reflections",
            "summ_step1_section_connections",
            "summ_step2_full_report",
            "summ_step4_adaptive_card",
            "final_adaptive_card_json",
            "routes_result",
            "verbatims_result",
            "ncs_result",
        ]
        for field in TRIM_CANDIDATES:
            if current_size <= MAX_ITEM_BYTES:
                break
            val = item.get(field)
            if isinstance(val, str) and len(val) > 5000:
                saved = len(val.encode("utf-8"))
                item[field] = val[:2000] + f"...[trimmed, original {saved} bytes]"
                current_size -= saved - len(item[field].encode("utf-8"))
                logger.warning(f"[DynamoDB] Trimmed '{field}' to fit size limit")
        return item
