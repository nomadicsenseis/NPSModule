"""
Anomaly Interpreter Agent

A specialized AI agent for interpreting NPS anomaly trees and generating conclusions
about operational desempeño and customer perception correlations.
"""

import asyncio
import os
import yaml
import logging
import json
import re
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import importlib.resources

# Import GenAI Core components (adjusted for new location)
from ..agents.agent import Agent
from ..llms.openai_llm import OpenAiLLM
from ..llms.aws_llm import AWSLLM
from ..utils.enums import LLMType, MessageType, AgentName, get_agent_conversations_folder
from ..utils.output_paths import (
    resolve_report_group, format_period_range,
    get_report_path, get_logging_path, get_s3_report_key, get_s3_logging_key,
    build_execution_metadata, save_minified_json, save_pretty_json,
)
from ..message_history import MessageHistory

# Import S3 uploader
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from dashboard_analyzer.data_collection.s3_report_uploader import S3ReportUploader

# Get the directory containing this file for relative path resolution
_AGENT_DIR = Path(__file__).resolve().parent
# Config is at: anomaly_explanation/config/prompts/ (2 levels up from agents/, then into config/prompts/)
_CONFIG_DIR = _AGENT_DIR.parent.parent / "config" / "prompts"
# Project root is 4 levels up from agents/ (agents → genai_core → anomaly_explanation → dashboard_analyzer → root)
_PROJECT_ROOT = _AGENT_DIR.parent.parent.parent.parent

class HierarchicalConversationTracker:
    """Track hierarchical conversation workflow with generation-by-generation analysis"""
    
    def __init__(self):
        self.conversation_log = []
        self.generation_count = 0
        self.hierarchy_structure = {}
        self.generation_reflections = []
        self.generation_analysis = {}
        
    def reset_tracker(self):
        """Reset for new hierarchical investigation"""
        self.conversation_log = []
        self.generation_count = 0
        self.hierarchy_structure = {}
        self.generation_reflections = []
        self.generation_analysis = {}
        
    def log_message(self, message_type: str, content: str, metadata: Optional[Dict] = None):
        """Log a message in the hierarchical conversation"""
        self.conversation_log.append({
            'generation': self.generation_count,
            'type': message_type,
            'content': content,
            'metadata': metadata or {},
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
    
    def set_hierarchy_structure(self, hierarchy: Dict[str, Any]):
        """Store the parsed hierarchy structure"""
        self.hierarchy_structure = hierarchy
    
    def start_generation(self, generation_level: int, nodes: List[str]):
        """Start analysis of a new generation"""
        self.generation_count = generation_level
        self.log_message("GENERATION_START", f"Starting analysis of generation {generation_level}", {
            'generation_level': generation_level,
            'nodes': nodes,
            'node_count': len(nodes)
        })
    
    def add_generation_reflection(self, generation: int, nodes: List[str], reflection: str):
        """Add a reflection for a specific generation"""
        reflection_data = {
            'generation': generation,
            'nodes': nodes,
            'reflection': reflection,
            'timestamp': datetime.now().isoformat(),
            'reflection_length': len(reflection)
        }
        self.generation_reflections.append(reflection_data)
        self.log_message("GENERATION_REFLECTION", f"Generation {generation} reflection captured", {
            'generation': generation,
            'nodes': nodes,
            'reflection_length': len(reflection)
        })
    
    def add_generation_analysis(self, generation: int, analysis_data: Dict[str, Any]):
        """Add detailed analysis data for a generation"""
        self.generation_analysis[generation] = analysis_data
    
    def get_conversation_summary(self) -> Dict[str, int]:
        """Get summary of conversation message types"""
        summary = {}
        for entry in self.conversation_log:
            msg_type = entry['type']
            summary[msg_type] = summary.get(msg_type, 0) + 1
        return summary
    
    def get_hierarchy_summary(self) -> Dict[str, Any]:
        """Get summary of the hierarchy analyzed"""
        if not self.hierarchy_structure:
            return {}
        
        return {
            'total_nodes': len(self.hierarchy_structure),
            'generations_analyzed': len(set(entry['generation'] for entry in self.generation_reflections)),
            'nodes_by_level': {
                level: [path for path, data in self.hierarchy_structure.items() if data.get('level') == level]
                for level in set(data.get('level', 0) for data in self.hierarchy_structure.values())
            },
            'parent_child_relationships': {
                path: data.get('children', []) 
                for path, data in self.hierarchy_structure.items() 
                if data.get('children')
            }
        }


class AnomalyInterpreterAgent:
    """
    Specialized agent for interpreting anomaly trees and generating actionable conclusions.
    
    Features:
    - Multi-LLM support (OpenAI, AWS Bedrock)
    - External prompt configuration via YAML
    - Professional anomaly analysis with operational insights
    - Hierarchical generation-by-generation analysis with AI reflections
    - Conversation tracking and export functionality
    """
    
    def __init__(
        self,
        llm_type: Optional[LLMType] = None,
        config_path: str = "dashboard_analyzer/anomaly_explanation/config/prompts/anomaly_interpreter.yaml",
        logger: Optional[logging.Logger] = None,
        study_mode: str = "comparative",
        environment: str = "prod",
        anomaly_detection_mode: str = "vslast",
        causal_filter: Optional[str] = None,
        comparison_start_date: Optional[str] = None,
        comparison_end_date: Optional[str] = None,
        aggregation_days: int = 7,
        baseline_periods: int = 7,
        segment: str = "Global",
        focus_touchpoint: Optional[str] = None,
    ):
        """
        Initialize the Anomaly Interpreter Agent.

        Args:
            llm_type: Type of LLM to use (supports OpenAI and AWS Bedrock models)
            config_path: Path to YAML configuration file with prompts
            logger: Optional logger instance
            study_mode: Study mode - "single" or "comparative"
            environment: Environment type ("local" or "prod")
            anomaly_detection_mode: How anomalies are detected ("vslast", "mean", "target")
            causal_filter: Comparison filter (e.g. "vs L7d", "vs Sel. Period")
            comparison_start_date: Start of comparison period (comparative mode)
            comparison_end_date: End of comparison period (comparative mode)
            aggregation_days: Days per analysis period
            baseline_periods: Number of baseline periods
            segment: Root segment ("Global", etc.)
            focus_touchpoint: Touchpoint focus (None → "General", "Cabin-Crew", etc.)
        """
        # Use default LLM type if none provided
        if llm_type is None:
            from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import get_default_llm_type
            llm_type = get_default_llm_type()
        self.llm_type = llm_type
        self.logger = logger or self._setup_logger()
        self.study_mode = study_mode
        self.environment = environment

        # Execution context for output paths and metadata
        self.anomaly_detection_mode = anomaly_detection_mode
        self.causal_filter = causal_filter
        self.comparison_start_date = comparison_start_date
        self.comparison_end_date = comparison_end_date
        self.aggregation_days = aggregation_days
        self.baseline_periods = baseline_periods
        self.segment = segment
        self.focus_touchpoint = focus_touchpoint
        self.report_group = resolve_report_group(focus_touchpoint)
        
        # Load environment variables from .env in current working directory only if not in prod
        if self.environment != "prod":
            dotenv_path = Path.cwd() / '.env'
            if dotenv_path.exists():
                load_dotenv(dotenv_path)
        
        # Load prompt configuration
        self.config = self._load_prompt_config(config_path)
        
        # Initialize LLM based on type
        self.llm = self._create_llm(llm_type)
        
        # Initialize base agent
        self.agent = Agent(llm=self.llm, logger=self.logger)
        
        # Conversation tracking for hierarchical analysis
        self.hierarchical_reflections = []
        self.generation_data = {}
        self.conversation_tracker = HierarchicalConversationTracker()
        
        # Initialize S3 uploader with environment
        self.s3_uploader = S3ReportUploader(environment=environment)
        
        # desempeño metrics
        self.total_processing_time = 0.0
        self.total_hierarchical_calls = 0
        
        # Store last generated Adaptive Card for summary agent
        self.last_adaptive_card = None
        self.last_optimized_adaptive_card = None
        
        self.logger.info(f"🤖 AnomalyInterpreterAgent initialized with {llm_type.value}")

    def _setup_logger(self) -> logging.Logger:
        """Setup default logger for the agent."""
        logger = logging.getLogger(f"AnomalyInterpreterAgent")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _load_prompt_config(self, config_path: str) -> Dict[str, Any]:
        """Load prompt configuration from YAML file using importlib.resources."""
        try:
            # Try loading as package resource first
            config_filename = Path(config_path).name
            package_path = "dashboard_analyzer.anomaly_explanation.config.prompts"
            
            try:
                ref = importlib.resources.files(package_path) / config_filename
                with ref.open('r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"Prompt configuration loaded from package resource: {package_path}/{config_filename}")
            except (ImportError, FileNotFoundError, TypeError) as e:
                # Fallback to direct file path
                self.logger.debug(f"Could not load from package ({e}), trying fallback path logic")
                full_path = _CONFIG_DIR / config_filename
                
                if not full_path.exists():
                    # Try relative to workspace root if _CONFIG_DIR resolution failed
                    full_path = Path("/workspace") / config_path
                    if not full_path.exists():
                        raise FileNotFoundError(f"Prompt config file not found at {full_path}")

                with open(full_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"Prompt configuration loaded from {full_path}")
            
            # Validate that the config has the expected structure
            required_keys = ['comparative_prompts', 'single_prompts', 'hierarchical_diagnostic_helpers']
            missing_keys = [k for k in required_keys if k not in config]
            if missing_keys:
                self.logger.warning(f"⚠️ Config loaded but missing keys: {missing_keys}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load prompt config from {config_path}: {e}")
            # Return a comprehensive fallback configuration
            return self._get_fallback_config()
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Return a comprehensive fallback configuration when YAML loading fails"""
        base_system_prompt = """Eres un analista de datos experto en la industria aérea, especializado en interpretar datos de Net Promoter Score (NPS).
Tu objetivo es generar una interpretación clara, concisa y ejecutiva de las anomalías detectadas."""
        
        context_template = """Analiza la siguiente información sobre anomalías de NPS para el día {date}.

<tree_data>
{tree_data}
</tree_data>

Confirma que has recibido la información y estás listo para el análisis paso a paso."""

        return {
            "comparative_prompts": {
                "system_prompt": base_system_prompt,
                "input_templates": {
                    "hierarchical_context_setup": context_template,
                    "hierarchical_analysis": "📊 **ANÁLISIS JERÁRQUICO**\n\n{tree_data}\n\n**Fecha:** {date}",
                    "single_node_analysis": "📊 **ANÁLISIS INTEGRADO**\n\n{tree_data}\n\n**Fecha:** {date}"
                }
            },
            "single_prompts": {
                "system_prompt": base_system_prompt,
                "input_templates": {
                    "hierarchical_context_setup": context_template,
                    "hierarchical_analysis": "📊 **ANÁLISIS JERÁRQUICO**\n\n{tree_data}\n\n**Fecha:** {date}",
                    "single_node_analysis": "📊 **ANÁLISIS INTEGRADO**\n\n{tree_data}\n\n**Fecha:** {date}"
                }
            },
            "hierarchical_diagnostic_helpers": {
                "step1_company_level_diagnosis": "🏢 **PASO 1: DIAGNÓSTICO A NIVEL COMPAÑÍA**\n\nAnaliza si las causas están localizadas a nivel de compañía (IB/YW) en Short Haul.",
                "step2_cabin_level_diagnosis": "✈️ **PASO 2: DIAGNÓSTICO A NIVEL DE CABINA**\n\nAnaliza las diferencias entre Economy, Business y Premium.",
                "step3_radio_global_diagnosis": "📡 **PASO 3: DIAGNÓSTICO A NIVEL RADIO/GLOBAL**\n\nAnaliza las diferencias entre Long Haul y Short Haul.",
                "step4_nma_identification": "🎯 **PASO 4: IDENTIFICACIÓN DE NMAs**\n\nIdentifica los Nodos Máximo Afectados usando la lógica de burbujeo.",
                "step4b_evidence_extraction": "📋 **PASO 4B: EXTRACCIÓN DE EVIDENCIAS**\n\nVuelve al contexto inicial y extrae TODAS las evidencias para cada NMA.",
                "step5_executive_synthesis": "📋 **PASO 5: SÍNTESIS EJECUTIVA**\n\nGenera un resumen ejecutivo con las principales conclusiones y recomendaciones."
            }
        }

    
    def _get_system_prompt(self, mode: str = None) -> str:
        """Get system prompt for specified mode"""
        if mode is None:
            mode = self.study_mode
        
        if mode == "single":
            return self.config.get('single_prompts', {}).get('system_prompt', '')
        else:
            return self.config.get('comparative_prompts', {}).get('system_prompt', '')
    
    def _get_input_template(self, template_name: str, mode: str = None) -> str:
        """Get input template for specified mode and template name"""
        if mode is None:
            mode = self.study_mode
        
        if mode == "single":
            templates = self.config.get('single_prompts', {}).get('input_templates', {})
        else:
            templates = self.config.get('comparative_prompts', {}).get('input_templates', {})
        
        template = templates.get(template_name)
        if template:
            return template
        
        # Log warning but return a usable fallback instead of error message
        self.logger.warning(f"⚠️ Template '{template_name}' not found for mode '{mode}', using fallback")
        fallback_config = self._get_fallback_config()
        fallback_templates = fallback_config.get('comparative_prompts' if mode != 'single' else 'single_prompts', {}).get('input_templates', {})
        return fallback_templates.get(template_name, f"Analiza los datos proporcionados para {template_name}.")
    
    def _get_hierarchical_helper(self, step_name: str) -> str:
        """Get hierarchical diagnostic helper (shared between modes)"""
        helpers = self.config.get('hierarchical_diagnostic_helpers', {})
        helper = helpers.get(step_name)
        if helper:
            return helper
        
        # Log warning but return a usable fallback instead of error message
        self.logger.warning(f"⚠️ Helper '{step_name}' not found, using fallback")
        fallback_config = self._get_fallback_config()
        fallback_helpers = fallback_config.get('hierarchical_diagnostic_helpers', {})
        return fallback_helpers.get(step_name, f"Ejecuta el paso de análisis: {step_name}")
    
    def _get_config_value(self, keys: List[str]) -> Optional[str]:
        """Get a value from config using a list of keys (path)"""
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return None
            else:
                return None
        return value
    
    def _interpreter_report_context(
        self, date: Optional[str], segment: Optional[str]
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Build analysis_date, date_ranges, and params for comprehensive-report-shaped JSON."""
        seg = (segment or self.segment or "Global").strip() or "Global"
        if date and " to " in str(date):
            parts = [p.strip() for p in str(date).split(" to ", 1)]
            comp_start, comp_end = parts[0], parts[1]
            analysis_date = comp_end
        elif date:
            d = str(date).strip()
            comp_start = self.comparison_start_date or d
            comp_end = self.comparison_end_date or d
            analysis_date = d
        else:
            analysis_date = datetime.now().strftime("%Y-%m-%d")
            comp_start = self.comparison_start_date or analysis_date
            comp_end = self.comparison_end_date or analysis_date

        date_ranges = {
            "analysis_date": analysis_date,
            "comparison_start_date": comp_start,
            "comparison_end_date": comp_end,
        }
        execution_metadata = {
            "execution_date": datetime.now().isoformat() + "Z",
            "analysis_date": analysis_date,
            "segment": seg,
            "explanation_mode": "interpreter_hierarchical",
            "causal_filter": self.causal_filter or "N/A",
        }
        weekly_analysis_params = {
            "anomaly_detection_mode": self.anomaly_detection_mode,
            "baseline_periods": self.baseline_periods,
            "aggregation_days": self.aggregation_days,
            "periods": 1,
            "study_mode": self.study_mode,
        }
        daily_analysis_params: Dict[str, Any] = {}
        return analysis_date, date_ranges, execution_metadata, weekly_analysis_params, daily_analysis_params

    async def _save_adaptive_card(self, card_json: str, date: Optional[str], segment: Optional[str]):
        """Save report using the same JSON envelope as summarizer S3 comprehensive reports.

        The Adaptive Card (optimized) is stored under ``final_synthesis`` (dict in prod when parseable).
        Only comparative (weekly) reports are persisted; single (daily) cards are skipped.
        """
        if self.study_mode == "single":
            self.logger.info(
                f"⏭️ Skipping report save for single/daily interpreter ({self._measure_kb(card_json):.2f} KB)"
            )
            return

        try:
            period_range = format_period_range(date_param=date)
            output_path = get_report_path(self.report_group, "interpreter", period_range)
            (
                _analysis_date,
                date_ranges,
                execution_metadata,
                weekly_analysis_params,
                daily_analysis_params,
            ) = self._interpreter_report_context(date, segment)

            # Match summarizer: final_synthesis = parsed Adaptive Card object when JSON is valid
            try:
                final_synthesis: Any = json.loads(card_json)
            except Exception:
                final_synthesis = card_json

            report_doc = self.s3_uploader.build_comprehensive_report_document(
                execution_metadata=execution_metadata,
                weekly_analysis_params=weekly_analysis_params,
                daily_analysis_params=daily_analysis_params,
                date_ranges=date_ranges,
                final_synthesis=final_synthesis,
            )
            body = json.dumps(report_doc, ensure_ascii=False, separators=(",", ":"))
            save_minified_json(output_path, body)
            self.logger.info(
                f"💾 Interpreter report (comprehensive shape) saved to: {output_path} ({self._measure_kb(body):.2f} KB)"
            )

            # Upload to S3 in production
            if self.environment == "prod":
                try:
                    s3_key = get_s3_report_key(
                        self.s3_uploader.base_prefix, self.report_group, "interpreter", period_range,
                    )
                    self.s3_uploader.s3_client.put_object(
                        Bucket=self.s3_uploader.bucket_name,
                        Key=s3_key,
                        Body=body.encode("utf-8"),
                        ContentType="application/json",
                        Metadata={
                            "content_type": "comprehensive_analysis",
                            "report_group": self.report_group,
                            "period_range": period_range,
                            "agent": "interpreter",
                        },
                    )
                    self.logger.info(f"📤 Interpreter comprehensive report uploaded to S3: {s3_key}")
                except Exception as s3_error:
                    self.logger.warning(f"⚠️ Failed to upload interpreter report to S3: {s3_error}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save Adaptive Card: {e}")
    
    def _measure_kb(self, payload: str) -> float:
        """Return UTF-8 payload size in KB."""
        if payload is None:
            return 0.0
        return len(payload.encode('utf-8')) / 1024.0
    
    def _clean_json_response(self, text: str) -> str:
        """Extract JSON from LLM response - simple and direct."""
        if not text:
            return "{}"
        
        cleaned = text.strip()
        
        # Extract from markdown code blocks if present
        if "```" in cleaned:
            parts = cleaned.split("```")
            for part in parts[1:]:  # Skip first part (before first ```)
                content = part.split("```")[0] if "```" in part else part
                # Remove language identifier if present
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
                if content.startswith("{"):
                    cleaned = content
                    break
        
        # Find first '{' and last '}'
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end+1]
            
        return cleaned
    
    def _minify_json(self, json_payload: str) -> str:
        """Minify JSON string if possible to reduce size."""
        if not json_payload:
            return json_payload
        
        cleaned = self._clean_json_response(json_payload)
        try:
            parsed = json.loads(cleaned)
            return json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
        except Exception:
            # If parsing fails, just return the cleaned text as a last resort
            return cleaned.strip()
    
    async def _optimize_adaptive_card_size(self, adaptive_card_json: str) -> str:
        """
        Optimize Adaptive Card size through progressive reduction.

        Target size: 24 KB
        Reduction order:
        1. Minify JSON
        2. Trim company details (IB/YW)
        3. Trim LH cabins
        4. Trim SH cabins

        After reduction, ALWAYS run a final JSON validation & fix step.
        """
        TARGET_KB = 24

        # Step 1: Minify JSON
        best_json = self._minify_json(adaptive_card_json)
        current_kb = self._measure_kb(best_json)
        self.logger.info(f"📦 Adaptive Card size after minification: {current_kb:.2f} KB")

        if current_kb > TARGET_KB:
            reduction_steps = [
                'step8a_trim_company_details',
                'step8b_trim_lh_cabins',
                'step8c_trim_sh_cabins',
            ]

            last_step_applied = "minify_only"
            best_kb = current_kb

            for step_key in reduction_steps:
                step_config = self._get_config_value([step_key, 'system_prompt'])
                step_input_template = self._get_config_value([step_key, 'input_template'])

                if not step_config or not step_input_template:
                    self.logger.warning(f"⚠️ Missing config for {step_key}, skipping")
                    continue

                self.logger.info(f"🔄 Applying optimization step: {step_key}...")

                escaped_for_llm = best_json.replace('"', '\\"')

                message_history = MessageHistory(logger=self.logger)
                message_history.create_and_add_message(
                    content=step_config.format(target_kb=TARGET_KB),
                    message_type=MessageType.SYSTEM
                )
                message_history.create_and_add_message(
                    content=step_input_template.format(current_json=escaped_for_llm, target_kb=TARGET_KB),
                    message_type=MessageType.USER
                )

                response, _, _ = await self.agent.invoke(messages=message_history.get_messages())
                optimized = response.content if hasattr(response, 'content') else str(response)

                optimized_clean = self._clean_json_response(optimized)
                optimized_min = self._minify_json(optimized_clean)
                size_kb = self._measure_kb(optimized_min)

                self.logger.info(f"📦 Adaptive Card size after {step_key}: {size_kb:.2f} KB")

                if size_kb < best_kb:
                    best_json = optimized_min
                    best_kb = size_kb
                    last_step_applied = step_key

                if size_kb <= TARGET_KB:
                    self.logger.info(f"✅ Adaptive Card fits after {step_key} ({size_kb:.2f} KB)")
                    break

            if best_kb > TARGET_KB:
                self.logger.warning(
                    f"⚠️ Adaptive Card still over limit after {last_step_applied}: "
                    f"{best_kb:.2f} KB (target: {TARGET_KB} KB)"
                )

        # --- STEP 8E: REDISTRIBUTE CONTENT (ALWAYS RUNS AFTER SIZE OPTIMIZATION) ---
        # This step reorganizes content between sections for better structure
        # It runs on the card that has already achieved the target size
        redistribute_config = self._get_config_value(['step8e_redistribute_content', 'system_prompt'])
        redistribute_input_template = self._get_config_value(['step8e_redistribute_content', 'input_template'])

        if redistribute_config and redistribute_input_template:
            self.logger.info("🔄 Applying content redistribution for better structure...")

            escaped_for_llm = best_json.replace('"', '\\"')

            message_history = MessageHistory(logger=self.logger)
            message_history.create_and_add_message(
                content=redistribute_config,
                message_type=MessageType.SYSTEM
            )
            message_history.create_and_add_message(
                content=redistribute_input_template.format(current_json=escaped_for_llm),
                message_type=MessageType.USER
            )

            try:
                response, _, _ = await self.agent.invoke(messages=message_history.get_messages())
                redistributed = response.content if hasattr(response, 'content') else str(response)

                redistributed_clean = self._clean_json_response(redistributed)
                redistributed_min = self._minify_json(redistributed_clean)
                redistributed_kb = self._measure_kb(redistributed_min)

                # Update best JSON with redistributed version
                best_json = redistributed_min
                best_kb = redistributed_kb
                
                self.logger.info(f"📦 Adaptive Card size after content redistribution: {best_kb:.2f} KB")
                self.logger.info("✅ Content redistribution applied successfully")

            except Exception as e:
                self.logger.warning(f"⚠️ Failed to apply content redistribution: {e}")
        else:
            self.logger.warning("⚠️ Missing config for step8e_redistribute_content, skipping")

        # --- FINAL STEP: JSON Validation and Typo Correction (ALWAYS runs) ---
        pre_validation_json = best_json
        step_config = self._get_config_value(['step8d_validate_and_fix_json', 'system_prompt'])
        step_input_template = self._get_config_value(['step8d_validate_and_fix_json', 'input_template'])

        if step_config and step_input_template:
            self.logger.info("🔄 Applying final JSON validation and typo correction...")

            escaped_for_llm = best_json.replace('"', '\\"')

            message_history = MessageHistory(logger=self.logger)
            message_history.create_and_add_message(
                content=step_config,
                message_type=MessageType.SYSTEM
            )
            message_history.create_and_add_message(
                content=step_input_template.format(current_json=escaped_for_llm),
                message_type=MessageType.USER
            )

            try:
                response, _, _ = await self.agent.invoke(messages=message_history.get_messages())
                validated = response.content if hasattr(response, 'content') else str(response)

                validated_clean = self._clean_json_response(validated)
                validated_min = self._minify_json(validated_clean)

                # Only accept if the result is valid JSON
                try:
                    json.loads(validated_min)
                    best_json = validated_min
                    best_kb = self._measure_kb(best_json)
                    self.logger.info(f"📦 Adaptive Card size after validation: {best_kb:.2f} KB")
                    self.logger.info("✅ Final JSON is valid and parseable")
                except json.JSONDecodeError as e:
                    self.logger.error(f"❌ Validation step returned invalid JSON: {e}")
                    self.logger.warning("⚠️ Keeping pre-validation JSON")
                    best_json = pre_validation_json

            except Exception as e:
                self.logger.warning(f"⚠️ Failed to apply JSON validation: {e}")
                best_json = pre_validation_json
        else:
            self.logger.warning("⚠️ Missing config for step8d_validate_and_fix_json, skipping")

        best_kb = self._measure_kb(best_json)
        if best_kb > TARGET_KB:
            self.logger.warning(f"⚠️ Adaptive Card over limit: {best_kb:.2f} KB (target: {TARGET_KB} KB)")
        else:
            self.logger.info(f"✅ Adaptive Card final size: {best_kb:.2f} KB")

        return best_json

    def _create_llm(self, llm_type: LLMType):
        """Create LLM instance"""
        # OpenAI models
        if llm_type in [LLMType.GPT4o, LLMType.O3, LLMType.O3_MINI, LLMType.O4_MINI, LLMType.GPT_5_2]:
            return self._create_openai_llm(llm_type)
        
        # AWS Bedrock models
        elif llm_type in [
            LLMType.CLAUDE_V2, LLMType.CLAUDE_INSTANT, LLMType.CLAUDE_3_HAIKU, LLMType.CLAUDE_3_5_HAIKU,
            LLMType.CLAUDE_3_OPUS, LLMType.CLAUDE_3_5_SONNET, LLMType.CLAUDE_3_5_SONNET_V2, LLMType.CLAUDE_3_7_SONNET,
            LLMType.CLAUDE_SONNET_4, LLMType.CLAUDE_OPUS_4_5, LLMType.CLAUDE_OPUS_4_6, LLMType.LLAMA3_70, 
            LLMType.LLAMA3_1_70, LLMType.LLAMA3_1_405,
            # New models
            LLMType.AMAZON_NOVA_2_LITE, LLMType.AMAZON_NOVA_PRO, LLMType.AMAZON_TITAN_EMBED_TEXT_V2,
            LLMType.CLAUDE_HAIKU_4_5, LLMType.CLAUDE_SONNET_4_5, LLMType.GPT_OSS_120B
        ]:
            return self._create_aws_llm(llm_type)
        
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

    def _create_openai_llm(self, llm_type: LLMType) -> OpenAiLLM:
        """Create OpenAI/Azure OpenAI LLM instance using unified credential strategy."""
        from dashboard_analyzer.anomaly_explanation.genai_core.utils.openai_session import (
            get_openai_credentials, is_azure_openai_configured, is_openai_platform_configured
        )
        
        # Get credentials based on current environment (local vs prod)
        creds = get_openai_credentials(environment=self.environment)
        
        is_azure = is_azure_openai_configured(creds)
        is_platform = is_openai_platform_configured(creds)
        
        if not is_azure and not is_platform:
            raise ValueError("No OpenAI or Azure OpenAI credentials found in environment variables")
        
        # Priority: Azure (to maintain legacy) if configured, otherwise OpenAI Platform
        if is_azure:
            return OpenAiLLM(
                llm_type=llm_type,
                api_key=creds['azure_api_key'] or "",
                api_base=creds['azure_endpoint'] or "",
                api_version=creds['azure_api_version'] or "2024-12-01-preview",
                api_dep_gpt=creds['azure_deployment'] or "",
                temperature=1.0  # Default temperature for O4-MINI compatibility
            )
        else:
            return OpenAiLLM(
                llm_type=llm_type,
                api_key=creds['openai_api_key'] or "",
                project_id=creds['openai_project_id'],
                temperature=1.0
            )

    def _create_aws_llm(self, llm_type: LLMType) -> AWSLLM:
        """Create AWS Bedrock LLM instance using unified credential strategy."""
        from dashboard_analyzer.anomaly_explanation.genai_core.utils.aws_session import get_aws_credentials
        
        # Get credentials based on current environment (local vs prod)
        creds = get_aws_credentials(environment=self.environment)
        
        return AWSLLM(
            llm_type=llm_type,
            region_name=creds['region_name'],
            aws_access_key_id=creds['aws_access_key_id'],
            aws_secret_access_key=creds['aws_secret_access_key'],
            aws_session_token=creds['aws_session_token'],
            profile_name=os.getenv("AWS_PROFILE"),
            environment=self.environment
        )
    
    async def interpret_anomaly_tree(self, tree_data: str, date: Optional[str] = None, segment: Optional[str] = None) -> str:
        """
        Interpret a causal agent explanation using conversational hierarchical methodology.
        
        Args:
            tree_data: The causal agent explanation/narrative (not an anomaly tree)
            date: Optional date for context (e.g., "2025-05-24")
            segment: Optional segment for analysis
            
        Returns:
            Structured interpretation following the conversational hierarchical methodology
        """
        self.logger.info("🔄 Using CONVERSATIONAL hierarchical interpretation (self-conversation enabled)")
        
        # Always use the conversational method that includes self-conversation
        return await self.interpret_anomaly_tree_hierarchical(tree_data, date, segment)

    async def interpret_anomaly_tree_hierarchical(self, tree_data: str, date: Optional[str] = None, segment: Optional[str] = None) -> str:
        """
        Perform hierarchical interpretation using conversational step-by-step reasoning.
        
        Args:
            tree_data: Combined explanations from multiple hierarchical nodes
            date: Optional date for context
            
        Returns:
            Comprehensive hierarchical interpretation with step-by-step analysis
        """
        start_time = datetime.now()
        self.conversation_tracker.reset_tracker()
        
        try:
            # Skip parsing - just use the tree_data as-is
            print("🔍 DEBUG INTERPRETER: Skipping parsing, using tree_data as-is", file=sys.stderr)
            self.logger.info("🔄 Using tree_data directly without parsing")
            
            print("🔍 DEBUG INTERPRETER: Creating message history...", file=sys.stderr)
            # Create message history for conversational analysis
            message_history = MessageHistory(logger=self.logger)
            print("🔍 DEBUG INTERPRETER: Message history created", file=sys.stderr)
            
            print("🔍 DEBUG INTERPRETER: Getting system prompt...", file=sys.stderr)
            # Add system prompt based on study mode
            system_prompt = self._get_system_prompt()
            print("🔍 DEBUG INTERPRETER: System prompt obtained", file=sys.stderr)
            
            print("🔍 DEBUG INTERPRETER: Adding system message...", file=sys.stderr)
            message_history.create_and_add_message(
                content=system_prompt,
                message_type=MessageType.SYSTEM
            )
            print("🔍 DEBUG INTERPRETER: System message added", file=sys.stderr)
            
            print("🔍 DEBUG INTERPRETER: Creating context message...", file=sys.stderr)
            # Step 1: Provide all context to the model and get acknowledgment
            context_message = self._get_input_template('hierarchical_context_setup').format(
                tree_data=tree_data,
                date=date if date else 'No especificada'
            )
            print("🔍 DEBUG INTERPRETER: Context message created", file=sys.stderr)
            
            # DEBUG: Print exactly what's being passed to the interpreter
            print(f"\n🔍 DEBUG INTERPRETER INPUT:")
            print(f"=" * 80)
            print(f"📅 Date: {date if date else 'No especificada'}")
            print(f"📊 Tree data length: {len(tree_data)} chars")
            print(f"🎯 Tree data content:")
            print("-" * 40)
            print(tree_data)
            print("-" * 40)
            print(f"📝 Final context message length: {len(context_message)} chars")
            print(f"📝 Final context message preview:")
            print(context_message[:500] + "..." if len(context_message) > 500 else context_message)
            print(f"=" * 80)
            
            message_history.create_and_add_message(
                content=context_message,
                message_type=MessageType.USER
            )
            self.conversation_tracker.log_message("CONTEXT_SETUP", context_message, {
                'date': date
            })

            # Get AI acknowledgment of context
            print("🔍 DEBUG INTERPRETER: About to call OpenAI for context acknowledgment", file=sys.stderr)
            messages = message_history.get_messages()
            print(f"🔍 DEBUG INTERPRETER: Sending {len(messages)} messages to OpenAI", file=sys.stderr)
            for i, msg in enumerate(messages):
                msg_type = getattr(msg, 'type', 'unknown')
                msg_content = getattr(msg, 'content', '')
                print(f"🔍 DEBUG INTERPRETER: Message {i}: type={msg_type}, length={len(msg_content)}", file=sys.stderr)
                print(f"🔍 DEBUG INTERPRETER: Message {i} content preview: {msg_content[:200]}...", file=sys.stderr)
            
            context_response, _, _ = await self.agent.invoke(messages=messages)
            print("🔍 DEBUG INTERPRETER: OpenAI context acknowledgment completed", file=sys.stderr)
            message_history.add_message(context_response)
            
            self.conversation_tracker.log_message("CONTEXT_ACKNOWLEDGED", context_response.content, {
                'response_length': len(context_response.content)
            })
            
            self.logger.info(f"💭 Context acknowledged: {context_response.content[:100]}...")

            # Determine applicable steps dynamically using the provided segment
            detected_segment = segment if segment else self._extract_primary_segment_from_data(tree_data)
            applicable_step_keys = self._get_applicable_steps_for_segment(detected_segment)
            
            # Step-by-step conversational analysis
            conversation_steps = []
            step_name_mapping = {
                'step1_company_level_diagnosis': "COMPANY_LEVEL_DIAGNOSIS",
                'step2_cabin_level_diagnosis': "CABIN_LEVEL_DIAGNOSIS", 
                'step3_radio_global_diagnosis': "RADIO_GLOBAL_DIAGNOSIS",
                'step4_nma_identification': "NMA_IDENTIFICATION",
                'step4b_evidence_extraction': "EVIDENCE_EXTRACTION",
                'step4c_cabin_radio_reflection': "CABIN_RADIO_REFLECTION",
                'step5_executive_synthesis': "EXECUTIVE_SYNTHESIS"
            }
            
            for step_key in applicable_step_keys:
                if step_key in step_name_mapping:
                    conversation_steps.append((step_key, step_name_mapping[step_key]))
            
            # Always add step4b (evidence extraction) after step4 if step4 is included
            step_keys_in_conversation = [s[0] for s in conversation_steps]
            if 'step4_nma_identification' in step_keys_in_conversation and 'step4b_evidence_extraction' not in step_keys_in_conversation:
                # Find position of step4 and insert step4b after it
                step4_index = step_keys_in_conversation.index('step4_nma_identification')
                conversation_steps.insert(step4_index + 1, ('step4b_evidence_extraction', "EVIDENCE_EXTRACTION"))
            
            # Add step4c (cabin-radio reflection) after step4b if segment has cabin-radios
            cabin_radios = self._get_cabin_radios_for_segment(detected_segment)
            has_cabin_radios = cabin_radios['sh_cabins'] or cabin_radios['lh_cabins']
            
            step_keys_in_conversation = [s[0] for s in conversation_steps]
            if has_cabin_radios and 'step4b_evidence_extraction' in step_keys_in_conversation and 'step4c_cabin_radio_reflection' not in step_keys_in_conversation:
                # Find position of step4b and insert step4c after it
                step4b_index = step_keys_in_conversation.index('step4b_evidence_extraction')
                conversation_steps.insert(step4b_index + 1, ('step4c_cabin_radio_reflection', "CABIN_RADIO_REFLECTION"))
            
            # Always add step5 for final synthesis if not already included
            if 'step5_executive_synthesis' not in [s[0] for s in conversation_steps]:
                conversation_steps.append(('step5_executive_synthesis', "EXECUTIVE_SYNTHESIS"))
            
            self.logger.info(f"🔄 Using {len(conversation_steps)} applicable steps for segment '{detected_segment}': {[s[0] for s in conversation_steps]}")
            
            step_responses = []
            
            # Pre-extract data for step 4C and 5 placeholders
            cabin_radio_values_table = ""
            cabin_radios_list = ""
            segment_reference = ""
            
            if has_cabin_radios:
                cabin_radio_values_table = self._extract_cabin_radio_values(tree_data, cabin_radios)
                cabins_list_parts = []
                if cabin_radios['sh_cabins']:
                    cabins_list_parts.append(f"**Cabinas SH (con análisis IB/YW):** {', '.join(cabin_radios['sh_cabins'])}")
                if cabin_radios['lh_cabins']:
                    cabins_list_parts.append(f"**Cabinas LH (resumen directo):** {', '.join(cabin_radios['lh_cabins'])}")
                cabin_radios_list = "\n".join(cabins_list_parts)
            
            segment_reference = self._extract_segment_reference(tree_data, detected_segment)
            
            for step_key, step_name in conversation_steps:
                self.logger.info(f"🔍 Executing diagnostic step: {step_name}")
                
                # Get helper prompt for this step
                helper_prompt = self._get_hierarchical_helper(step_key)
                
                # For cabin-radio reflection (step 4C), replace placeholders with extracted values
                if step_key == 'step4c_cabin_radio_reflection':
                    helper_prompt = helper_prompt.replace('{CABIN_RADIO_VALUES_TABLE}', cabin_radio_values_table)
                    helper_prompt = helper_prompt.replace('{CABIN_RADIOS_LIST}', cabin_radios_list)
                
                # For executive synthesis (step 5), replace placeholders
                if step_key == 'step5_executive_synthesis':
                    helper_prompt = helper_prompt.replace('{SEGMENT_REFERENCE}', segment_reference)
                
                message_history.create_and_add_message(
                    content=helper_prompt,
                    message_type=MessageType.USER
                )
                
                self.conversation_tracker.log_message("HELPER_PROMPT", helper_prompt, {
                    'step': step_name,
                    'prompt_length': len(helper_prompt)
                })
                
                self.logger.info(f"💬 Asking AI: {helper_prompt[:100]}...")
                
                # Get AI response for this step
                print(f"🔍 DEBUG INTERPRETER: About to call OpenAI for step: {step_name}", file=sys.stderr)
                step_messages = message_history.get_messages()
                print(f"🔍 DEBUG INTERPRETER: Sending {len(step_messages)} messages to OpenAI for {step_name}", file=sys.stderr)
                
                # Show the last message (the one we just added)
                if step_messages:
                    last_msg = step_messages[-1]
                    last_content = getattr(last_msg, 'content', '')
                    print(f"🔍 DEBUG INTERPRETER: Last message for {step_name}: {last_content[:300]}...", file=sys.stderr)
                
                step_response, _, _ = await self.agent.invoke(
                    messages=step_messages
                )
                print(f"🔍 DEBUG INTERPRETER: OpenAI step '{step_name}' completed", file=sys.stderr)
                
                step_content = step_response.content.strip()
                step_responses.append({
                    'step': step_name,
                    'content': step_content
                })
                
                message_history.add_message(step_response)
                
                self.conversation_tracker.log_message("STEP_RESPONSE", step_content, {
                    'step': step_name,
                    'response_length': len(step_content)
                })
                
                self.logger.info(f"💭 AI Response for {step_name}: {step_content[:150]}...")
                self.logger.info(f"✅ {step_name} completed ({len(step_content)} chars)")
            
            # STEP 6: Generate Adaptive Card (NEW)
            executive_synthesis_content = next((s['content'] for s in step_responses if s['step'] == 'EXECUTIVE_SYNTHESIS'), None)
            adaptive_card_json = None
            
            if executive_synthesis_content:
                self.logger.info(f"🎨 Generating Adaptive Card from executive synthesis...")
                
                # Get step 6 prompt
                step6_prompt = self._get_config_value(['step6_generate_adaptive_card', 'input_template'])
                if step6_prompt:
                    date_range = f"{date}" if date else "Período no especificado"
                    step6_input = step6_prompt.format(
                        executive_synthesis=executive_synthesis_content,
                        date_range=date_range
                    )
                    
                    # Get system prompt for step 6
                    step6_system = self._get_config_value(['step6_generate_adaptive_card', 'system_prompt'])
                    if step6_system:
                        # Create new message history for step 6
                        step6_history = MessageHistory(logger=self.logger)
                        step6_history.create_and_add_message(content=step6_system, message_type=MessageType.SYSTEM)
                        step6_history.create_and_add_message(content=step6_input, message_type=MessageType.USER)
                        
                        # Log step 6
                        self.conversation_tracker.log_message("HELPER_PROMPT", step6_input, {
                            'step': 'ADAPTIVE_CARD_GENERATION',
                            'prompt_length': len(step6_input)
                        })
                        
                        step6_response, _, _ = await self.agent.invoke(messages=step6_history.get_messages())
                        adaptive_card_json = step6_response.content.strip()
                        
                        step_responses.append({
                            'step': 'ADAPTIVE_CARD_GENERATION',
                            'content': adaptive_card_json
                        })
                        
                        # Log step 6 response
                        self.conversation_tracker.log_message("STEP_RESPONSE", adaptive_card_json, {
                            'step': 'ADAPTIVE_CARD_GENERATION',
                            'response_length': len(adaptive_card_json)
                        })
                        
                        self.logger.info(f"✅ ADAPTIVE_CARD_GENERATION completed ({len(adaptive_card_json)} chars)")
                        
                        # STEP 7: Modernize Tone (NEW)
                        self.logger.info(f"✍️ Modernizing tone of Adaptive Card...")
                        
                        step7_prompt = self._get_config_value(['step7_modernize_tone', 'input_template'])
                        if step7_prompt:
                            step7_input = step7_prompt.format(current_json=adaptive_card_json)
                            
                            step7_system = self._get_config_value(['step7_modernize_tone', 'system_prompt'])
                            if step7_system:
                                step7_history = MessageHistory(logger=self.logger)
                                step7_history.create_and_add_message(content=step7_system, message_type=MessageType.SYSTEM)
                                step7_history.create_and_add_message(content=step7_input, message_type=MessageType.USER)
                                
                                # Log step 7
                                self.conversation_tracker.log_message("HELPER_PROMPT", step7_input, {
                                    'step': 'TONE_MODERNIZATION',
                                    'prompt_length': len(step7_input)
                                })
                                
                                step7_response, _, _ = await self.agent.invoke(messages=step7_history.get_messages())
                                modernized_card_json = step7_response.content.strip()
                                
                                step_responses.append({
                                    'step': 'TONE_MODERNIZATION',
                                    'content': modernized_card_json
                                })
                                
                                # Log step 7 response
                                self.conversation_tracker.log_message("STEP_RESPONSE", modernized_card_json, {
                                    'step': 'TONE_MODERNIZATION',
                                    'response_length': len(modernized_card_json)
                                })
                                
                                self.logger.info(f"✅ TONE_MODERNIZATION completed ({len(modernized_card_json)} chars)")
                                
                                # STEP 8: Optimize Adaptive Card size if needed
                                self.logger.info(f"📦 Optimizing Adaptive Card size...")
                                self.logger.info(f"📊 Input size before optimization: {self._measure_kb(modernized_card_json):.2f} KB")
                                
                                try:
                                    optimized_card_json = await self._optimize_adaptive_card_size(modernized_card_json)
                                    self.logger.info(f"📊 Output size after optimization: {self._measure_kb(optimized_card_json):.2f} KB")
                                except Exception as opt_error:
                                    self.logger.error(f"❌ Optimization failed: {opt_error}, using minified version")
                                    optimized_card_json = self._minify_json(modernized_card_json)
                                
                                # Save the optimized adaptive card to a file (reduced version for storage)
                                await self._save_adaptive_card(optimized_card_json, date, segment)
                                
                                # Store both versions: unreduced for summary agent, optimized for return
                                self.last_adaptive_card = modernized_card_json
                                self.last_optimized_adaptive_card = optimized_card_json
            
            # Compile final response from all steps
            print("🔍 DEBUG INTERPRETER: Compiling final interpretation...", file=sys.stderr)
            final_interpretation = self._compile_final_interpretation(
                step_responses,
                {}  # Empty hierarchy since we're not parsing
            )
            print(f"🔍 DEBUG INTERPRETER: Final interpretation compiled, length: {len(final_interpretation)}", file=sys.stderr)
            print(f"🔍 DEBUG INTERPRETER: Final interpretation preview: {final_interpretation[:500]}...", file=sys.stderr)
            
            # Store step_responses for DynamoDB persistence
            self._last_step_responses = step_responses
            self._last_final_interpretation = final_interpretation

            # Update desempeño metrics
            end_time = datetime.now()
            self.total_processing_time = (end_time - start_time).total_seconds()
            self.total_hierarchical_calls += 1
            
            self.logger.info(f"✅ Hierarchical interpretation completed in {self.total_processing_time:.2f}s")
            
            # Export successful conversation for debugging
            conversation_file = await self.export_hierarchical_conversation(date=date)
            if conversation_file:
                self.logger.info(f"🗂️ Conversación jerárquica guardada: {conversation_file}")
            
            # Combine interpretation + adaptive card (same structure as summarizer)
            optimized_card = getattr(self, 'last_optimized_adaptive_card', None)
            if optimized_card:
                final_output = f"{final_interpretation}\n\n---ADAPTIVE_CARD_JSON---\n\n{optimized_card}"
            else:
                final_output = final_interpretation

            print("🔍 DEBUG INTERPRETER: About to return final interpretation", file=sys.stderr)
            print(f"🔍 DEBUG INTERPRETER: Returning {len(final_output)} characters", file=sys.stderr)
            return final_output
            
        except Exception as e:
            self.logger.error(f"❌ Error in hierarchical interpretation: {str(e)}")
            error_msg = f"Error durante la interpretación jerárquica: {str(e)}"
            
            # Export error conversation for debugging
            await self.export_hierarchical_conversation(date, error_msg)
            
            return f"❌ Error en la interpretación jerárquica: {str(e)}"
    
    def _parse_hierarchy_from_explanations(self, tree_data: str) -> Dict[str, Any]:
        """Parse hierarchical structure from combined causal explanations."""
        print("🔍 DEBUG PARSER: Starting _parse_hierarchy_from_explanations", file=sys.stderr)
        print(f"🔍 DEBUG PARSER: tree_data length: {len(tree_data)}", file=sys.stderr)
        
        hierarchy = {}
        
        # First try to parse NODO: format (legacy)
        print("🔍 DEBUG PARSER: Trying NODO: format", file=sys.stderr)
        nodo_pattern = re.compile(r"NODO:\s*(.*?)\n(.*?)(?=\nNODO:|\Z)", re.DOTALL)
        nodo_matches = nodo_pattern.findall(tree_data)
        print(f"🔍 DEBUG PARSER: Found {len(nodo_matches)} NODO matches", file=sys.stderr)
        
        if nodo_matches:
            # Legacy NODO: format
            for match in nodo_matches:
                node_path = match[0].strip()
                content = match[1].strip()
                
                # Determine level
                level = len(node_path.split('/'))
                
                # Determine parent
                parent_path = '/'.join(node_path.split('/')[:-1]) if '/' in node_path else None

                hierarchy[node_path] = {
                    'type': 'node',
                    'level': level,
                    'path': node_path,
                    'parent': parent_path,
                    'children': [],
                    'content': content  # The full explanation for this node
                }
        else:
            # New hierarchical tree format - parse line by line
            print("🔍 DEBUG PARSER: Using new hierarchical tree format parser", file=sys.stderr)
            self.logger.info("Using new hierarchical tree format parser")
            lines = tree_data.split('\n')
            print(f"🔍 DEBUG PARSER: Split into {len(lines)} lines", file=sys.stderr)
            current_node_path = None
            current_content = []
            
            # Extract segment filter from debug information if available
            segment_context = ""
            for line in lines[:10]:  # Check first 10 lines for context
                if 'segment_filter:' in line:
                    segment_context = line
                    break
            
            for line in lines:
                original_line = line
                line = line.strip()
                if not line:
                    continue
                
                # Look for node indicators with anomaly states or NPS data
                node_indicators = ['POSITIVE ANOMALY', 'NEGATIVE ANOMALY', 'Normal', 'No Data']
                has_node_indicator = any(indicator in line for indicator in node_indicators)
                has_nps_data = 'NPS:' in line and ('vs' in line or 'baseline' in line)
                
                if (has_node_indicator or has_nps_data) and ':' in line:
                    # Save previous node if exists
                    if current_node_path:
                        hierarchy[current_node_path] = {
                            'type': 'node',
                            'level': len(current_node_path.split('/')),
                            'path': current_node_path,
                            'parent': '/'.join(current_node_path.split('/')[:-1]) if '/' in current_node_path else None,
                            'children': [],
                            'content': '\n'.join(current_content).strip()
                        }
                    
                    # Extract node path and start new node
                    node_name = line.split(':')[0].strip()
                    
                    # Clean up node name (remove tree symbols)
                    node_name = node_name.replace('├─', '').replace('└─', '').replace('  ', '').strip()
                    
                    # Try to determine the full node path based on context
                    current_node_path = self._infer_node_path(node_name, tree_data, segment_context)
                    
                    current_content = [original_line]  # Keep original formatting
                else:
                    # Add content to current node
                    if current_node_path:
                        current_content.append(original_line)  # Keep original formatting
            
            # Save last node if exists
            if current_node_path:
                hierarchy[current_node_path] = {
                    'type': 'node',
                    'level': len(current_node_path.split('/')),
                    'path': current_node_path,
                    'parent': '/'.join(current_node_path.split('/')[:-1]) if '/' in current_node_path else None,
                    'children': [],
                    'content': '\n'.join(current_content).strip()
                }
        
        # Build parent-child relationships
        for node_path, node_data in hierarchy.items():
            parent_path = node_data.get('parent')
            if parent_path and parent_path in hierarchy:
                if node_path not in hierarchy[parent_path]['children']:
                    hierarchy[parent_path]['children'].append(node_path)
        
        return hierarchy
    
    def _infer_node_path(self, node_name: str, tree_data: str, segment_context: str) -> str:
        """Infer the full node path from the node name and context."""
        # If already a full path, return as is
        if 'Global/' in node_name:
            return node_name
        
        # Try to extract from segment_filter context
        if 'segment_filter:' in segment_context:
            filter_part = segment_context.split('segment_filter:')[1].strip()
            if 'Business/LH' in filter_part:
                if 'Business' in node_name:
                    return 'Global/LH/Business'
            elif 'Economy/SH' in filter_part:
                if 'Economy' in node_name:
                    return 'Global/SH/Economy'
                elif 'IB' in node_name:
                    return 'Global/SH/Economy/IB'
                elif 'YW' in node_name:
                    return 'Global/SH/Economy/YW'
        
        # Look for clues in the tree_data
        if 'Global/LH/Business' in tree_data and 'Business' in node_name:
            return 'Global/LH/Business'
        elif 'Global/SH/Economy' in tree_data and 'Economy' in node_name:
            return 'Global/SH/Economy'
        elif 'Global/SH/Economy/IB' in tree_data and 'IB' in node_name:
            return 'Global/SH/Economy/IB'
        elif 'Global/SH/Economy/YW' in tree_data and 'YW' in node_name:
            return 'Global/SH/Economy/YW'
        
        # Default mappings based on common patterns
        if 'Business' in node_name:
            return 'Global/LH/Business'
        elif 'Economy' in node_name:
            return 'Global/SH/Economy'
        elif node_name == 'IB':
            return 'Global/SH/Economy/IB'  # Most common
        elif node_name == 'YW':
            return 'Global/SH/Economy/YW'  # Most common
        else:
            return f'Global/{node_name}'
    
    def _format_hierarchy_structure(self, hierarchy: Dict[str, Any]) -> str:
        """Format hierarchy structure for display."""
        result = []
        
        # Find root nodes (no parent)
        root_nodes = [path for path, data in hierarchy.items() if data.get('parent') is None]
        
        def format_node(node_path: str, indent: str = "") -> str:
            node_data = hierarchy.get(node_path, {})
            children = node_data.get('children', [])
            
            lines = [f"{indent}📊 {node_path}"]
            
            for child in children:
                lines.append(format_node(child, indent + "  "))
            
            return '\n'.join(lines)
        
        for root in sorted(root_nodes):
            result.append(format_node(root))
        
        return '\n'.join(result)
    
    def _order_generations_bottom_up(self, hierarchy: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Order nodes by generation level (bottom-up)."""
        # Group by depth level
        levels = {}
        for path, data in hierarchy.items():
            level = data.get('level', 0)
            if level not in levels:
                levels[level] = []
            levels[level].append(data)
        
        # Return in reverse order (deepest first)
        result = []
        for level in sorted(levels.keys(), reverse=True):
            result.append(levels[level])
        
        return result
    
    def _generate_generation_helper_prompt(self, generation_level: int, nodes: List[Dict[str, Any]], hierarchy: Dict[str, Any]) -> str:
        """Generate helper prompt for analyzing a specific generation."""
        
        node_paths = [f"• {node['path']}" for node in nodes]
        
        return self.config['hierarchical_prompts']['generation_analysis'].format(
            generation_level=generation_level + 1,
            node_paths='\n'.join(node_paths)
        )
    
    def _generate_comprehensive_summary_prompt(self, hierarchy: Dict[str, Any]) -> str:
        """Generate comprehensive summary prompt asking for final conclusions, causes, impacts, routes, and profiles."""
        
        total_nodes = len(hierarchy)
        analyzed_segments = ', '.join(hierarchy.keys())
        
        return self.config['hierarchical_prompts']['comprehensive_summary'].format(
            total_nodes=total_nodes,
            analyzed_segments=analyzed_segments
        )
    
    def _generate_synthesis_prompt(self, hierarchy: Dict[str, Any]) -> str:
        """Generate final synthesis prompt across all generations."""
        
        total_nodes = len(hierarchy)
        
        return self.config['hierarchical_prompts']['final_synthesis'].format(
            total_nodes=total_nodes
        )
    
    def _compile_final_interpretation(self, step_responses: List[Dict[str, str]], hierarchy: Dict[str, Any]) -> str:
        """
        Compile final interpretation from all conversational steps.
        
        Args:
            step_responses: List of step responses with step name and content
            hierarchy: Parsed hierarchy structure
            
        Returns:
            Compiled final interpretation with all analysis steps
        """
        try:
            # Header with analysis overview
            total_nodes = len(hierarchy)
            
            final_interpretation = f"""
📊 **ANÁLISIS JERÁRQUICO COMPLETO DE ANOMALÍAS NPS**

**Nodos analizados:** {total_nodes} ({', '.join(sorted(list(hierarchy.keys())))})

---
"""
            
            # Add each step response with proper formatting
            step_titles = {
                'COMPANY_LEVEL_DIAGNOSIS': '📊 DIAGNÓSTICO A NIVEL DE EMPRESA',
                'CABIN_LEVEL_DIAGNOSIS': '💺 DIAGNÓSTICO A NIVEL DE CABINA', 
                'RADIO_GLOBAL_DIAGNOSIS': '🌎 DIAGNÓSTICO GLOBAL POR RADIO',
                'NMA_IDENTIFICATION': '🎯 IDENTIFICACIÓN DE NODOS MÁXIMO AFECTADOS',
                'EVIDENCE_EXTRACTION': '📋 EXTRACCIÓN DE EVIDENCIAS',
                'EXECUTIVE_SYNTHESIS': '📋 SÍNTESIS EJECUTIVA FINAL'
            }
            
            for step_data in step_responses:
                step_name = step_data['step']
                step_content = step_data['content']
                step_title = step_titles.get(step_name, step_name.replace('_', ' ').title())
                
                final_interpretation += f"""
## {step_title}

{step_content}

---
"""
            
            # Footer with completion summary
            final_interpretation += f"""
✅ **ANÁLISIS COMPLETADO**

- **Nodos procesados:** {total_nodes}
- **Pasos de análisis:** {len(step_responses)}
- **Metodología:** Análisis conversacional paso a paso
- **Resultado:** Interpretación jerárquica completa con razonamiento estructurado

*Este análisis utiliza metodología conversacional para simular el razonamiento paso a paso de un analista experto, similar al proceso de investigación causal.*
"""
            
            return final_interpretation.strip()
            
        except Exception as e:
            self.logger.error(f"❌ Error compiling final interpretation: {str(e)}")
            # Fallback: return the final synthesis step if available
            final_synthesis = next((s['content'] for s in step_responses if s['step'] == 'EXECUTIVE_SYNTHESIS'), None)
            if final_synthesis:
                return f"📋 **SÍNTESIS FINAL**\n\n{final_synthesis}"

            # Last resort: return a generic error
            return "⚠️ **ANÁLISIS PARCIAL** (Error en compilación)"
    
    async def export_hierarchical_conversation(self, date: Optional[str] = None, error: Optional[str] = None, execution_id: Optional[str] = None, source_causal_report_ids: Optional[List[str]] = None) -> str:
        """Export the hierarchical conversation log to the logging directory and DynamoDB."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            period_range = format_period_range(date_param=date)
            filename = f"interpreter_{timestamp}.json"

            full_path = get_logging_path(self.report_group, "interpreter", period_range, filename)

            conversation_data = {
                "metadata": build_execution_metadata(
                    model=self.llm_type.value,
                    study_mode=self.study_mode,
                    anomaly_detection_mode=self.anomaly_detection_mode,
                    causal_filter=self.causal_filter,
                    comparison_start_date=self.comparison_start_date,
                    comparison_end_date=self.comparison_end_date,
                    aggregation_days=self.aggregation_days,
                    baseline_periods=self.baseline_periods,
                    segment=self.segment,
                    focus_touchpoint=self.focus_touchpoint,
                    agent_type="anomaly_interpreter",
                    analysis_type="hierarchical_generation_by_generation",
                    analysis_date=date,
                    total_generations=len(set(r['generation'] for r in self.hierarchical_reflections)),
                    total_nodes_analyzed=len(self.conversation_tracker.hierarchy_structure),
                    error=error,
                ),
                "hierarchy_structure": self.conversation_tracker.hierarchy_structure,
                "hierarchy_summary": self.conversation_tracker.get_hierarchy_summary(),
                "conversation_log": self.conversation_tracker.conversation_log,
                "conversation_summary": self.conversation_tracker.get_conversation_summary(),
                "generation_reflections": self.hierarchical_reflections,
                "generation_analysis": self.generation_data,
            }

            save_pretty_json(full_path, conversation_data)
            self.logger.info(f"📝 Hierarchical conversation exported to: {full_path}")

            s3_key = None
            # Upload to S3 in production
            if self.environment == "prod":
                try:
                    s3_key = get_s3_logging_key(
                        self.s3_uploader.base_prefix, self.report_group,
                        "interpreter", period_range, filename,
                    )
                    self.s3_uploader.s3_client.put_object(
                        Bucket=self.s3_uploader.bucket_name,
                        Key=s3_key,
                        Body=json.dumps(conversation_data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                        ContentType="application/json",
                    )
                    self.logger.info(f"📤 Interpreter conversation uploaded to S3: {s3_key}")
                except Exception as s3_err:
                    self.logger.warning(f"⚠️ Failed to upload interpreter conversation to S3: {s3_err}")

            # Persist to DynamoDB
            if execution_id:
                try:
                    from ..utils.dynamodb_report_persistence import DynamoDBReportPersistence
                    dynamo = DynamoDBReportPersistence(environment=self.environment)
                    adaptive_card = getattr(self, "last_adaptive_card", None)
                    synthesis_id = dynamo.save_interpreter_report(
                        execution_id=execution_id,
                        agent=self,
                        step_responses=getattr(self, "_last_step_responses", None),
                        final_interpretation=getattr(self, "_last_final_interpretation", None),
                        final_adaptive_card_json=adaptive_card,
                        execution_duration_ms=self.total_processing_time * 1000 if self.total_processing_time else None,
                        status="failed" if error else "completed",
                        error_message=error,
                        s3_report_key=s3_key,
                        source_causal_report_ids=source_causal_report_ids,
                        analysis_date=date,
                    )
                    if synthesis_id:
                        self.logger.info(f"📊 Interpreter report persisted to DynamoDB: {synthesis_id}")
                except Exception as ddb_err:
                    self.logger.warning(f"⚠️ Failed to persist interpreter report to DynamoDB: {ddb_err}")

            return str(full_path)

        except Exception as e:
            self.logger.error(f"❌ Failed to export hierarchical conversation: {e}")
            return ""

    def get_desempeño_metrics(self) -> Dict[str, Any]:
        """Get agent desempeño metrics."""
        return {
            "num_calls": self.agent.num_calls,
            "total_time": self.agent.total_time,
            "avg_time": self.agent.avg_time,
            "last_execution_time": self.agent.last_execution_time,
            "input_tokens": self.agent.input_tokens,
            "output_tokens": self.agent.output_tokens,
            "money_spent": self.agent.money_spent,
            "llm_type": self.llm_type.value,
            "hierarchical_reflections_count": len(self.hierarchical_reflections),
            "conversation_messages_count": len(self.conversation_tracker.conversation_log)
        }

    def _get_applicable_steps_for_segment(self, segment: str) -> List[str]:
        """
        Determina qué pasos del análisis jerárquico aplican según el segmento.
        
        Args:
            segment: El segmento a analizar
            
        Returns:
            Lista de claves de prompts que deben ejecutarse
        """
        segment_mapping = {
            # Base keys
            'Global': ['step1_company_level_diagnosis', 'step2_cabin_level_diagnosis', 
                       'step3_radio_global_diagnosis', 'step4_nma_identification'],
            'LH': ['step2_cabin_level_diagnosis', 'step4_nma_identification'],
            'SH': ['step1_company_level_diagnosis', 'step2_cabin_level_diagnosis', 
                   'step4_nma_identification'],
            'Economy LH': ['step4_nma_identification'],
            'Business LH': ['step4_nma_identification'],
            'Premium LH': ['step4_nma_identification'],
            'Economy SH': ['step1_company_level_diagnosis', 'step4_nma_identification'],
            'Business SH': ['step1_company_level_diagnosis', 'step4_nma_identification'],
            'Premium SH': ['step1_company_level_diagnosis', 'step4_nma_identification'],
            'IB': ['step4_nma_identification'],
            'YW': ['step4_nma_identification'],
            
            # Path aliases (for robustness when extracting from tree paths)
            'Global/LH': ['step2_cabin_level_diagnosis', 'step4_nma_identification'],
            'Global/SH': ['step1_company_level_diagnosis', 'step2_cabin_level_diagnosis', 
                   'step4_nma_identification'],
            'Global/LH/Economy': ['step4_nma_identification'],
            'Global/LH/Business': ['step4_nma_identification'],
            'Global/LH/Premium': ['step4_nma_identification'],
            'Global/SH/Economy': ['step1_company_level_diagnosis', 'step4_nma_identification'],
            'Global/SH/Business': ['step1_company_level_diagnosis', 'step4_nma_identification'],
            'Global/SH/Premium': ['step1_company_level_diagnosis', 'step4_nma_identification'],
            'Global/SH/Economy/IB': ['step4_nma_identification'],
            'Global/SH/Economy/YW': ['step4_nma_identification'],
            'Global/SH/Business/IB': ['step4_nma_identification'],
            'Global/SH/Business/YW': ['step4_nma_identification'],
            
            # Readable aliases (from print outputs)
            'Long Haul (LH)': ['step2_cabin_level_diagnosis', 'step4_nma_identification'],
            'Short Haul (SH)': ['step1_company_level_diagnosis', 'step2_cabin_level_diagnosis', 
                   'step4_nma_identification'],
            'SH Economy': ['step1_company_level_diagnosis', 'step4_nma_identification'],
            'SH Business': ['step1_company_level_diagnosis', 'step4_nma_identification'],
            'LH Economy': ['step4_nma_identification'],
            'LH Business': ['step4_nma_identification'],
            'LH Premium': ['step4_nma_identification'],
        }
        return segment_mapping.get(segment, ['step4_nma_identification'])  # default
    
    def _extract_primary_segment_from_data(self, tree_data: str) -> str:
        """
        Determines the primary segment for analysis from the tree data.
        It scans the tree data to find the root node being analyzed.

        Args:
            tree_data: The hierarchical data containing node explanations.

        Returns:
            The detected primary segment (e.g. 'Global', 'Global/SH', etc.)
        """
        # Try to find the first node definition in the text
        # Support multiple formats:
        # 1. Legacy: "NODO: Global/SH"
        # 2. Visual Tree: "Global/SH: STATUS" or "├── Global/SH: STATUS"
        
        # Pattern 1: Legacy NODO
        nodo_matches = re.findall(r"NODO:\s*([^\n]+)", tree_data)
        if nodo_matches:
            primary_segment = nodo_matches[0].strip()
            self.logger.info(f"Detected primary segment (NODO format): {primary_segment}")
            return primary_segment
            
        # Pattern 2: Visual Tree (looking for "Segment Path: Status")
        # Matches lines starting with optional tree chars, then path, then colon, then status
        # We look for lines that contain ANOMALY or Normal to be sure it's a node line
        tree_lines = tree_data.split('\n')
        for line in tree_lines:
            if ':' in line and ('ANOMALY' in line or 'Normal' in line or 'No Data' in line):
                # Extract the part before the colon
                node_part = line.split(':')[0]
                # Clean tree characters
                clean_node = node_part.replace('├─', '').replace('└─', '').replace('│', '').replace('  ', '').strip()
                # Clean parenthesized abbreviations e.g. "Long Haul (LH)" -> "Global/LH" normalization is tricky here
                # So we prefer to rely on the clean path if possible. 
                # Assuming the tree generator outputs clean paths or we can infer them.
                
                # If the node name is simple (e.g. "LH"), we might need context, 
                # but usually the root node of the passed text is what we want.
                
                # Special handling: if we see "Global" at the start, that's definitely it
                if "Global" in clean_node:
                    # Take the first one we see
                    self.logger.info(f"Detected primary segment (Tree format): {clean_node}")
                    return clean_node
                
        # Fallback 1: Try to find just the first line with a colon that looks like a node
        for line in tree_lines:
            if ':' in line and ('ANOMALY' in line or 'Normal' in line):
                 node_part = line.split(':')[0]
                 clean_node = node_part.replace('├─', '').replace('└─', '').replace('│', '').replace('  ', '').strip()
                 self.logger.info(f"Detected primary segment (Fallback): {clean_node}")
                 return clean_node

        # Fallback 2: Default to Global if nothing found
        self.logger.info("Could not detect segment from data. Defaulting to 'Global'.")
        return 'Global'

    def _detect_segment_level(self, segment: str) -> str:
        """
        Detecta el nivel jerárquico del segmento.
        
        Args:
            segment: El segmento a analizar
            
        Returns:
            'company', 'cabin', 'radio', 'global'
        """
        if segment in ['IB', 'YW']:
            return 'company'
        elif segment in ['Economy LH', 'Business LH', 'Premium LH', 
                         'Economy SH', 'Business SH', 'Premium SH']:
            return 'cabin'
        elif segment in ['LH', 'SH']:
            return 'radio'
        elif segment == 'Global':
            return 'global'
        else:
            return 'unknown'

    def get_segment_hierarchy(self, segment: str) -> Dict[str, Any]:
        """
        Devuelve la estructura jerárquica completa para un segmento dado.
        Incluye el segmento raíz y TODAS las agregaciones por debajo.
        
        Args:
            segment: El segmento a analizar (ej: 'Global', 'SH', 'Economy SH')
            
        Returns:
            Dict con:
                - 'root': El segmento raíz normalizado
                - 'sections': Lista ordenada de secciones para el informe
                - 'hierarchy': Dict con la estructura jerárquica completa
                - 'sh_cabins': Cabinas SH con análisis IB/YW
                - 'lh_cabins': Cabinas LH (resumen directo)
        """
        segment = self._normalize_segment_name(segment)
        
        # Definir la jerarquía completa
        full_hierarchy = {
            'Global': {
                'sections': ['Global', 'SH', 'LH', 'Business SH', 'Business SH IB', 'Business SH YW', 
                            'Economy SH', 'Economy SH IB', 'Economy SH YW',
                            'Business LH', 'Premium LH', 'Economy LH'],
                'sh_cabins': ['Economy SH', 'Business SH'],
                'lh_cabins': ['Economy LH', 'Business LH', 'Premium LH'],
                'children': {
                    'SH': ['Economy SH', 'Business SH'],
                    'LH': ['Economy LH', 'Business LH', 'Premium LH'],
                    'Economy SH': ['Economy SH IB', 'Economy SH YW'],
                    'Business SH': ['Business SH IB', 'Business SH YW'],
                }
            },
            'SH': {
                'sections': ['SH', 'Business SH', 'Business SH IB', 'Business SH YW',
                            'Economy SH', 'Economy SH IB', 'Economy SH YW'],
                'sh_cabins': ['Economy SH', 'Business SH'],
                'lh_cabins': [],
                'children': {
                    'SH': ['Economy SH', 'Business SH'],
                    'Economy SH': ['Economy SH IB', 'Economy SH YW'],
                    'Business SH': ['Business SH IB', 'Business SH YW'],
                }
            },
            'LH': {
                'sections': ['LH', 'Economy LH', 'Business LH', 'Premium LH'],
                'sh_cabins': [],
                'lh_cabins': ['Economy LH', 'Business LH', 'Premium LH'],
                'children': {
                    'LH': ['Economy LH', 'Business LH', 'Premium LH'],
                }
            },
            'Economy SH': {
                'sections': ['Economy SH', 'Economy SH IB', 'Economy SH YW'],
                'sh_cabins': ['Economy SH'],
                'lh_cabins': [],
                'children': {
                    'Economy SH': ['Economy SH IB', 'Economy SH YW'],
                }
            },
            'Business SH': {
                'sections': ['Business SH', 'Business SH IB', 'Business SH YW'],
                'sh_cabins': ['Business SH'],
                'lh_cabins': [],
                'children': {
                    'Business SH': ['Business SH IB', 'Business SH YW'],
                }
            },
            'Economy LH': {
                'sections': ['Economy LH'],
                'sh_cabins': [],
                'lh_cabins': ['Economy LH'],
                'children': {}
            },
            'Business LH': {
                'sections': ['Business LH'],
                'sh_cabins': [],
                'lh_cabins': ['Business LH'],
                'children': {}
            },
            'Premium LH': {
                'sections': ['Premium LH'],
                'sh_cabins': [],
                'lh_cabins': ['Premium LH'],
                'children': {}
            },
            'IB': {
                'sections': ['IB'],
                'sh_cabins': [],
                'lh_cabins': [],
                'children': {}
            },
            'YW': {
                'sections': ['YW'],
                'sh_cabins': [],
                'lh_cabins': [],
                'children': {}
            },
        }
        
        hierarchy_data = full_hierarchy.get(segment, {
            'sections': [segment],
            'sh_cabins': [],
            'lh_cabins': [],
            'children': {}
        })
        
        return {
            'root': segment,
            'sections': hierarchy_data['sections'],
            'hierarchy': hierarchy_data['children'],
            'sh_cabins': hierarchy_data['sh_cabins'],
            'lh_cabins': hierarchy_data['lh_cabins']
        }

    def _get_cabin_radios_for_segment(self, segment: str) -> Dict[str, List[str]]:
        """
        Retorna las cabinas-radio que aplican para un segmento dado,
        clasificadas por tipo de reflexión (SH con análisis IB/YW, LH con resumen directo).
        
        Args:
            segment: El segmento normalizado a analizar
            
        Returns:
            Dict con 'sh_cabins' (requieren análisis IB/YW) y 'lh_cabins' (resumen directo)
        """
        hierarchy = self.get_segment_hierarchy(segment)
        return {
            'sh_cabins': hierarchy['sh_cabins'],
            'lh_cabins': hierarchy['lh_cabins']
        }

    def _normalize_segment_name(self, segment: str) -> str:
        """
        Normaliza el nombre del segmento a un formato estándar.
        
        Args:
            segment: El segmento en cualquier formato
            
        Returns:
            El nombre normalizado del segmento
        """
        if segment in ['Global', 'Global/']:
            return 'Global'
        elif segment in ['Global/SH', 'Short Haul (SH)', 'SH']:
            return 'SH'
        elif segment in ['Global/LH', 'Long Haul (LH)', 'LH']:
            return 'LH'
        elif segment in ['Global/SH/Economy', 'SH Economy']:
            return 'Economy SH'
        elif segment in ['Global/SH/Business', 'SH Business']:
            return 'Business SH'
        elif segment in ['Global/SH/Premium', 'SH Premium']:
            return 'Premium SH'
        elif segment in ['Global/LH/Economy', 'LH Economy']:
            return 'Economy LH'
        elif segment in ['Global/LH/Business', 'LH Business']:
            return 'Business LH'
        elif segment in ['Global/LH/Premium', 'LH Premium']:
            return 'Premium LH'
        elif 'IB' in segment:
            return 'IB'
        elif 'YW' in segment:
            return 'YW'
        return segment

    def _extract_cabin_radio_values(self, tree_data: str, cabin_radios: Dict[str, List[str]]) -> str:
        """
        Extrae los valores exactos de NPS para las cabinas-radio especificadas del tree_data.
        
        Args:
            tree_data: Los datos del árbol con las explicaciones causales
            cabin_radios: Dict con 'sh_cabins' y 'lh_cabins'
            
        Returns:
            Tabla markdown con los valores exactos de cada cabina y sus compañías (para SH)
        """
        lines = []
        all_cabins = cabin_radios['sh_cabins'] + cabin_radios['lh_cabins']
        
        if not all_cabins:
            return "No hay cabinas-radio para analizar en este segmento."
        
        # TABLA 1: RADIOS (SH, LH) - valores agregados por radio
        lines.append("**VALORES DE RADIOS (SH, LH):**")
        lines.append("| Radio | NPS Actual | Variación | Estado |")
        lines.append("|-------|------------|-----------|--------|")
        
        # Extraer SH si hay cabinas SH
        if cabin_radios['sh_cabins']:
            sh_nps, sh_diff, sh_state = self._parse_radio_values_from_tree(tree_data, 'SH')
            lines.append(f"| SH | {sh_nps} | {sh_diff} | {sh_state} |")
        
        # Extraer LH si hay cabinas LH
        if cabin_radios['lh_cabins']:
            lh_nps, lh_diff, lh_state = self._parse_radio_values_from_tree(tree_data, 'LH')
            lines.append(f"| LH | {lh_nps} | {lh_diff} | {lh_state} |")
        
        lines.append("")
        
        # TABLA 2: CABINAS-RADIO
        lines.append("**VALORES DE CABINAS-RADIO:**")
        lines.append("| Cabina | NPS Actual | Variación | Estado |")
        lines.append("|--------|------------|-----------|--------|")
        
        for cabin in all_cabins:
            nps, diff, state = self._parse_cabin_values_from_tree(tree_data, cabin)
            lines.append(f"| {cabin} | {nps} | {diff} | {state} |")
        
        # TABLA 3: COMPAÑÍAS para cabinas SH (IB, YW por cada cabina)
        if cabin_radios['sh_cabins']:
            lines.append("")
            lines.append("**VALORES DE COMPAÑÍAS SH (IB, YW por cabina):**")
            lines.append("| Cabina | Compañía | NPS Actual | Variación | Estado |")
            lines.append("|--------|----------|------------|-----------|--------|")
            
            for cabin in cabin_radios['sh_cabins']:
                # Extraer IB
                ib_nps, ib_diff, ib_state = self._parse_company_values_from_tree(tree_data, cabin, 'IB')
                lines.append(f"| {cabin} | IB | {ib_nps} | {ib_diff} | {ib_state} |")
                
                # Extraer YW
                yw_nps, yw_diff, yw_state = self._parse_company_values_from_tree(tree_data, cabin, 'YW')
                lines.append(f"| {cabin} | YW | {yw_nps} | {yw_diff} | {yw_state} |")
        
        return "\n".join(lines)
    
    def _parse_radio_values_from_tree(self, tree_data: str, radio: str) -> Tuple[str, str, str]:
        """
        Parsea los valores de NPS de un radio específico (SH o LH) del tree_data.
        
        Args:
            tree_data: Los datos del árbol
            radio: 'SH' o 'LH'
            
        Returns:
            Tupla (nps_actual, variacion, estado)
        """
        # Determinar marcadores del radio
        if radio == 'SH':
            radio_marker = 'Short Haul'
        else:
            radio_marker = 'Long Haul'
        
        # Patrón para buscar el radio: "Short Haul (SH): Normal (+6.8 pts - within normal range) (NPS: 37.49..."
        # o "Global/SH: Normal (+6.8 pts) (NPS: 37.49..."
        patterns = [
            # Patrón 1: "Short Haul (SH): Normal (+6.8 pts...) (NPS: 37.49...)"
            rf'{radio_marker}\s*\({radio}\)[:\s]+(?P<state>NEGATIVE ANOMALY|POSITIVE ANOMALY|Normal)\s*\((?P<diff>[+-]?\d+\.?\d*)\s*pts?[^)]*\)\s*\(NPS:\s*(?P<nps>[\d.]+)',
            # Patrón 2: "Global/SH: Normal (+6.8 pts) (NPS: 37.49...)"
            rf'Global/{radio}[:\s]+(?P<state>NEGATIVE ANOMALY|POSITIVE ANOMALY|Normal)\s*\((?P<diff>[+-]?\d+\.?\d*)\s*pts?[^)]*\)\s*\(NPS:\s*(?P<nps>[\d.]+)',
            # Patrón 3: Más flexible - solo el radio marker seguido de estado
            rf'{radio_marker}[:\s]+(?P<state>NEGATIVE ANOMALY|POSITIVE ANOMALY|Normal)\s*\((?P<diff>[+-]?\d+\.?\d*)\s*pts?[^)]*\)\s*\(NPS:\s*(?P<nps>[\d.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, tree_data, re.IGNORECASE)
            if match:
                nps = float(match.group('nps'))
                diff = float(match.group('diff'))
                state = match.group('state')
                return f"{nps:.1f}", f"{diff:+.1f}", state
        
        return "N/A", "N/A", "N/A"

    def _parse_cabin_values_from_tree(self, tree_data: str, cabin: str) -> Tuple[str, str, str]:
        """
        Parsea los valores de NPS de una cabina específica del tree_data.
        
        Args:
            tree_data: Los datos del árbol
            cabin: El nombre de la cabina (ej: 'Economy SH', 'Business LH')
            
        Returns:
            Tupla (nps_actual, variacion, estado)
        """
        # Determinar tipo de cabina y radio
        if 'SH' in cabin:
            cabin_type = cabin.replace(' SH', '')
            radio_marker = 'Short Haul'
            radio_abbrev = 'SH'
        else:
            cabin_type = cabin.replace(' LH', '')
            radio_marker = 'Long Haul'
            radio_abbrev = 'LH'
        
        # Patrón mejorado que maneja:
        # - "Economy: NEGATIVE ANOMALY (-4.4 pts) (NPS: 31.91..."
        # - "Business: Normal (+3.5 pts - within normal range) (NPS: 37.05..."
        pattern = rf'{cabin_type}[:\s]+(?P<state>NEGATIVE ANOMALY|POSITIVE ANOMALY|Normal)\s*\((?P<diff>[+-]?\d+\.?\d*)\s*pts?[^)]*\)\s*\(NPS:\s*(?P<nps>[\d.]+)'
        
        # Dividir el tree_data en secciones por radio
        lines = tree_data.split('\n')
        in_correct_radio = False
        found_radio_start = False
        
        for line in lines:
            # Detectar inicio del radio correcto
            if radio_marker in line or f'({radio_abbrev})' in line:
                in_correct_radio = True
                found_radio_start = True
                continue
            
            # Detectar si salimos del radio (otro radio empieza)
            if found_radio_start and in_correct_radio:
                other_radio = 'Long Haul' if radio_abbrev == 'SH' else 'Short Haul'
                other_abbrev = 'LH' if radio_abbrev == 'SH' else 'SH'
                if other_radio in line or f'({other_abbrev})' in line:
                    in_correct_radio = False
            
            # Buscar la cabina en el contexto correcto
            if in_correct_radio and cabin_type in line:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    nps = float(match.group('nps'))
                    diff = float(match.group('diff'))
                    state = match.group('state')
                    return f"{nps:.1f}", f"{diff:+.1f}", state
        
        # Fallback: buscar en todo el documento con un patrón más específico que incluya el path
        # Esto ayuda a distinguir entre Economy SH y Economy LH
        path_pattern = rf'Global/{radio_abbrev}/{cabin_type}[:\s]+(?P<state>NEGATIVE ANOMALY|POSITIVE ANOMALY|Normal)\s*\((?P<diff>[+-]?\d+\.?\d*)\s*pts?[^)]*\)\s*\(NPS:\s*(?P<nps>[\d.]+)'
        match = re.search(path_pattern, tree_data, re.IGNORECASE)
        if match:
            nps = float(match.group('nps'))
            diff = float(match.group('diff'))
            state = match.group('state')
            return f"{nps:.1f}", f"{diff:+.1f}", state
        
        return "N/A", "N/A", "N/A"

    def _parse_company_values_from_tree(self, tree_data: str, cabin: str, company: str) -> Tuple[str, str, str]:
        """
        Parsea los valores de NPS de una compañía específica dentro de una cabina del tree_data.
        
        Args:
            tree_data: Los datos del árbol
            cabin: El nombre de la cabina (ej: 'Economy SH', 'Business SH')
            company: 'IB' o 'YW'
            
        Returns:
            Tupla (nps_actual, variacion, estado)
        """
        # Determinar tipo de cabina
        cabin_type = cabin.replace(' SH', '').replace(' LH', '')
        radio_abbrev = 'SH' if 'SH' in cabin else 'LH'
        
        # Patrón mejorado que maneja "- within normal range" y otros textos
        pattern = rf'{company}[:\s]+(?P<state>NEGATIVE ANOMALY|POSITIVE ANOMALY|Normal)\s*\((?P<diff>[+-]?\d+\.?\d*)\s*pts?[^)]*\)\s*\(NPS:\s*(?P<nps>[\d.]+)'
        
        # Buscar primero el path específico en el tree_data
        # Global/SH/Economy/IB o Global/SH/Business/IB
        path_pattern = rf'Global/{radio_abbrev}/{cabin_type}/{company}[:\s]+(?P<state>NEGATIVE ANOMALY|POSITIVE ANOMALY|Normal)\s*\((?P<diff>[+-]?\d+\.?\d*)\s*pts?[^)]*\)\s*\(NPS:\s*(?P<nps>[\d.]+)'
        match = re.search(path_pattern, tree_data, re.IGNORECASE)
        if match:
            nps = float(match.group('nps'))
            diff = float(match.group('diff'))
            state = match.group('state')
            return f"{nps:.1f}", f"{diff:+.1f}", state
        
        # Fallback: buscar en el contexto de la cabina correcta
        lines = tree_data.split('\n')
        in_correct_cabin = False
        cabin_depth = 0
        
        for line in lines:
            # Detectar si estamos en la cabina correcta (bajo el radio correcto)
            if cabin_type in line and ('ANOMALY' in line or 'Normal' in line):
                # Verificar que estamos en el radio correcto
                # Buscamos hacia atrás para confirmar el radio
                in_correct_cabin = True
                cabin_depth = line.count('│') + line.count('├') + line.count('└')
            
            # Detectar si salimos de la cabina
            if in_correct_cabin:
                current_depth = line.count('│') + line.count('├') + line.count('└')
                # Si encontramos otra cabina al mismo nivel o superior, salimos
                if re.search(r'(Economy|Business|Premium)[:\s]+(NEGATIVE ANOMALY|POSITIVE ANOMALY|Normal)', line) and cabin_type not in line:
                    if current_depth <= cabin_depth:
                        in_correct_cabin = False
            
            # Buscar la compañía dentro de la cabina
            if in_correct_cabin and company in line:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    nps = float(match.group('nps'))
                    diff = float(match.group('diff'))
                    state = match.group('state')
                    return f"{nps:.1f}", f"{diff:+.1f}", state
        
        return "N/A", "N/A", "N/A"

    def _extract_segment_reference(self, tree_data: str, segment: str) -> str:
        """
        Extrae los valores de referencia del segmento raíz del tree_data.
        
        Args:
            tree_data: Los datos del árbol
            segment: El segmento raíz normalizado
            
        Returns:
            String con los valores de referencia del segmento
        """
        segment_norm = self._normalize_segment_name(segment)
        
        # Buscar el patrón del segmento raíz
        # Formato: "Global: NEGATIVE ANOMALY (-3.6 pts) (NPS: 24.99... vs baseline: 28.61...)"
        if segment_norm == 'Global':
            pattern = r'Global[:\s]+(?P<state>NEGATIVE ANOMALY|POSITIVE ANOMALY|Normal)\s*\((?P<diff>[+-]?\d+\.?\d*)\s*pts?\)\s*\(NPS:\s*(?P<nps>[\d.]+)'
        elif segment_norm in ['SH', 'LH']:
            pattern = rf'(?:Short Haul|Long Haul|{segment_norm})[:\s]+(?P<state>NEGATIVE ANOMALY|POSITIVE ANOMALY|Normal)\s*\((?P<diff>[+-]?\d+\.?\d*)\s*pts?\)\s*\(NPS:\s*(?P<nps>[\d.]+)'
        else:
            # Para cabinas específicas
            cabin_type = segment_norm.replace(' SH', '').replace(' LH', '')
            pattern = rf'{cabin_type}[:\s]+(?P<state>NEGATIVE ANOMALY|POSITIVE ANOMALY|Normal)\s*\((?P<diff>[+-]?\d+\.?\d*)\s*pts?\)\s*\(NPS:\s*(?P<nps>[\d.]+)'
        
        match = re.search(pattern, tree_data, re.IGNORECASE)
        if match:
            nps = float(match.group('nps'))
            diff = float(match.group('diff'))
            state = match.group('state')
            return f"• Segmento: {segment_norm}\n• NPS: {nps:.1f} ({diff:+.1f} pts)\n• Estado: {state}"
        
        return f"• Segmento: {segment_norm}\n• NPS: N/A\n• Estado: N/A"

    def _get_cabin_sections_for_segment(self, segment: str) -> str:
        """
        Genera las secciones de cabina dinámicas según el segmento seleccionado.
        Incluye TODAS las agregaciones jerárquicas bajo el segmento.
        
        Args:
            segment: El segmento a analizar
            
        Returns:
            String con las secciones de cabina relevantes para el prompt
        """
        # Normalize segment path to dictionary keys
        segment = self._normalize_segment_name(segment)

        # Plantillas para cada tipo de agregación
        radio_template = """
    **{radio}: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] El radio {radio} registró un NPS de [valor] con [variación] pts vs semana anterior. [Resumen de cómo las cabinas bajo este radio contribuyen al resultado total. Menciona las cabinas que más impactan al resultado.]
"""
        
        cabin_sh_template = """
    **{cabin}: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] La cabina {cabin_name} de SH registró un NPS de [valor cabina] con [variación] pts vs L7d. **Desglose por compañía:** IB obtuvo [NPS IB] ([diff IB] pts) y YW [NPS YW] ([diff YW] pts). [Breve explicación de la dinámica: si ambas suben/bajan = efecto conjunto; si una sube y otra baja = se compensan; si solo una tiene variación = esa domina]
"""
        
        company_template = """
    **{company}: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] La compañía {company} [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor] con [variación] pts vs semana anterior. [Para segmentos con variaciones: "La causa principal fue [hipótesis con datos], especialmente en rutas como [top rutas], afectando a perfiles [perfiles específicos]."]
"""
        
        cabin_lh_template = """
    **{cabin}: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] La cabina {cabin_name} de LH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] con [variación] pts vs la semana anterior. [Para segmentos con variaciones: "La causa principal fue [hipótesis con datos (drivers, operativa, NCS, verbatims) que la respaldan]. Esta [mejora/deterioro] se reflejó especialmente en rutas como [top rutas], mientras que los perfiles más reactivos incluyen [perfiles específicos]."]
"""
        
        # Construir las secciones según el segmento
        cabin_sections = {
            'Global': f"""
{radio_template.format(radio='SH')}
{radio_template.format(radio='LH')}
{cabin_sh_template.format(cabin='BUSINESS SH', cabin_name='Business')}
{company_template.format(company='BUSINESS SH IB')}
{company_template.format(company='BUSINESS SH YW')}
{cabin_sh_template.format(cabin='ECONOMY SH', cabin_name='Economy')}
{company_template.format(company='ECONOMY SH IB')}
{company_template.format(company='ECONOMY SH YW')}
{cabin_lh_template.format(cabin='BUSINESS LH', cabin_name='Business')}
{cabin_lh_template.format(cabin='PREMIUM LH', cabin_name='Premium')}
{cabin_lh_template.format(cabin='ECONOMY LH', cabin_name='Economy')}
""",
            'SH': f"""
{cabin_sh_template.format(cabin='BUSINESS SH', cabin_name='Business')}
{company_template.format(company='BUSINESS SH IB')}
{company_template.format(company='BUSINESS SH YW')}
{cabin_sh_template.format(cabin='ECONOMY SH', cabin_name='Economy')}
{company_template.format(company='ECONOMY SH IB')}
{company_template.format(company='ECONOMY SH YW')}
""",
            'LH': f"""
{cabin_lh_template.format(cabin='ECONOMY LH', cabin_name='Economy')}
{cabin_lh_template.format(cabin='BUSINESS LH', cabin_name='Business')}
{cabin_lh_template.format(cabin='PREMIUM LH', cabin_name='Premium')}
""",
            'Economy SH': f"""
{cabin_sh_template.format(cabin='ECONOMY SH', cabin_name='Economy')}
{company_template.format(company='ECONOMY SH IB')}
{company_template.format(company='ECONOMY SH YW')}

**IMPORTANTE:** Analiza el comportamiento agregado de la cabina completa. MENCIONA EXPLÍCITAMENTE los valores NPS de cada compañía por separado antes de explicar el efecto neto en la cabina.
""",
            'Business SH': f"""
{cabin_sh_template.format(cabin='BUSINESS SH', cabin_name='Business')}
{company_template.format(company='BUSINESS SH IB')}
{company_template.format(company='BUSINESS SH YW')}

**IMPORTANTE:** Analiza el comportamiento agregado de la cabina completa. MENCIONA EXPLÍCITAMENTE los valores NPS de cada compañía por separado antes de explicar el efecto neto en la cabina.
""",
            'Economy LH': cabin_lh_template.format(cabin='ECONOMY LH', cabin_name='Economy'),
            'Business LH': cabin_lh_template.format(cabin='BUSINESS LH', cabin_name='Business'),
            'Premium LH': cabin_lh_template.format(cabin='PREMIUM LH', cabin_name='Premium'),
            'IB': company_template.format(company='IB'),
            'YW': company_template.format(company='YW'),
        }
        return cabin_sections.get(segment, "")


async def interpret_anomaly_tree(
    tree_data: str, 
    date: Optional[str] = None,
    segment: Optional[str] = None,
    llm_type: Optional[LLMType] = None
) -> str:
    """
    Convenience function to interpret an anomaly tree without creating an agent instance.
    
    Args:
        tree_data: Anomaly tree data to interpret
        date: Optional date context
        llm_type: LLM type to use
        
    Returns:
        Interpretation result
    """
    # Use default LLM type if none provided
    if llm_type is None:
        from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import get_default_llm_type
        llm_type = get_default_llm_type()
    agent = AnomalyInterpreterAgent(
        llm_type=llm_type,
        config_path="dashboard_analyzer/anomaly_explanation/config/prompts/anomaly_interpreter.yaml"
    )
    return await agent.interpret_anomaly_tree(tree_data, date, segment)


if __name__ == "__main__":
    # Example usage
    async def main():
        # Example anomaly tree data
        sample_tree = """
        Global: NEGATIVE ANOMALY (-12.5 pts)
          └─ Pattern: Negative anomaly driven by negative LH and negative SH
        
        ├─ LH (LH): NEGATIVE ANOMALY (-15.2 pts)
        │  └─ Analysis: High load factors (85-95%) across major routes
        │  
        │  ├─ Economy: NEGATIVE ANOMALY (-8.4 pts)
        │  │  └─ Analysis: Cabin crew satisfaction issues
        │  │
        │  ├─ Business: NEGATIVE ANOMALY (-22.1 pts)
        │  │  └─ Analysis: Food quality complaints on intercontinental routes
        │  │
        │  └─ Premium: NORMAL (0.2 pts)
        │     └─ Analysis: Maintained service standards
        │
        └─ SH (SH): NEGATIVE ANOMALY (-8.7 pts)
           └─ Analysis: Punctuality issues affecting customer satisfaction
        """
        
        agent = AnomalyInterpreterAgent(
            config_path="../../config/prompts/anomaly_interpreter.yaml"
        )
        result = await agent.interpret_anomaly_tree(sample_tree, "2025-05-24")
        print(result)
    
    asyncio.run(main()) 