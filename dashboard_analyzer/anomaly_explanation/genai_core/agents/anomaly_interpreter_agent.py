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
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import importlib.resources

# Import GenAI Core components (adjusted for new location)
from ..agents.agent import Agent
from ..llms.openai_llm import OpenAiLLM
from ..llms.aws_llm import AWSLLM
from ..utils.enums import LLMType, MessageType, AgentName
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
        environment: str = "prod"
    ):
        """
        Initialize the Anomaly Interpreter Agent.
        
        Args:
            llm_type: Type of LLM to use (supports OpenAI and AWS Bedrock models)
            config_path: Path to YAML configuration file with prompts
            logger: Optional logger instance
            study_mode: Study mode - "single" or "comparative"
            environment: Environment type ("local" or "prod")
        """
        # Use default LLM type if none provided
        if llm_type is None:
            from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import get_default_llm_type
            llm_type = get_default_llm_type()
        self.llm_type = llm_type
        self.logger = logger or self._setup_logger()
        self.study_mode = study_mode
        self.environment = environment
        
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

    def _create_llm(self, llm_type: LLMType):
        """Create LLM instance"""
        if llm_type in [LLMType.GPT4o, LLMType.O3, LLMType.O3_MINI, LLMType.O4_MINI]:
            return self._create_openai_llm(llm_type)
        else:
            return self._create_aws_llm(llm_type)

    def _create_openai_llm(self, llm_type: LLMType) -> OpenAiLLM:
        """Create OpenAI/Azure OpenAI LLM instance."""
        # Get credentials from environment variables
        api_key = os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("AZURE_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        if not all([api_key, api_base, deployment_name]):
            raise ValueError("Missing required OpenAI/Azure OpenAI environment variables")
        
        return OpenAiLLM(
            llm_type=llm_type,
            api_key=api_key or "",
            api_base=api_base or "",
            api_version=api_version,
            api_dep_gpt=deployment_name or "",
            temperature=1.0  # Default temperature for O4-MINI compatibility
        )

    def _create_aws_llm(self, llm_type: LLMType) -> AWSLLM:
        """Create AWS Bedrock LLM instance."""
        region_name = os.getenv("AWS_REGION", "us-east-1")
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        profile_name = os.getenv("AWS_PROFILE")
        
        return AWSLLM(
            llm_type=llm_type,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            profile_name=profile_name
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
            
            # Always add step5 for final synthesis if not already included
            if 'step5_executive_synthesis' not in [s[0] for s in conversation_steps]:
                conversation_steps.append(('step5_executive_synthesis', "EXECUTIVE_SYNTHESIS"))
            
            self.logger.info(f"🔄 Using {len(conversation_steps)} applicable steps for segment '{detected_segment}': {[s[0] for s in conversation_steps]}")
            
            step_responses = []
            
            for step_key, step_name in conversation_steps:
                self.logger.info(f"🔍 Executing diagnostic step: {step_name}")
                
                # Get helper prompt for this step
                helper_prompt = self._get_hierarchical_helper(step_key)
                
                # For executive synthesis, replace CABIN_SECTIONS placeholder with dynamic content
                if step_key == 'step5_executive_synthesis':
                    cabin_sections = self._get_cabin_sections_for_segment(detected_segment)
                    helper_prompt = helper_prompt.replace('{CABIN_SECTIONS}', cabin_sections)
                
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
            
            # Compile final response from all steps
            print("🔍 DEBUG INTERPRETER: Compiling final interpretation...", file=sys.stderr)
            final_interpretation = self._compile_final_interpretation(
                step_responses,
                {}  # Empty hierarchy since we're not parsing
            )
            print(f"🔍 DEBUG INTERPRETER: Final interpretation compiled, length: {len(final_interpretation)}", file=sys.stderr)
            print(f"🔍 DEBUG INTERPRETER: Final interpretation preview: {final_interpretation[:500]}...", file=sys.stderr)
            
            # Update desempeño metrics
            end_time = datetime.now()
            self.total_processing_time = (end_time - start_time).total_seconds()
            self.total_hierarchical_calls += 1
            
            self.logger.info(f"✅ Hierarchical interpretation completed in {self.total_processing_time:.2f}s")
            
            # Export successful conversation for debugging
            conversation_file = await self.export_hierarchical_conversation(date=date)
            if conversation_file:
                self.logger.info(f"🗂️ Conversación jerárquica guardada: {conversation_file}")
            
            print("🔍 DEBUG INTERPRETER: About to return final interpretation", file=sys.stderr)
            print(f"🔍 DEBUG INTERPRETER: Returning {len(final_interpretation)} characters", file=sys.stderr)
            return final_interpretation
            
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
    
    async def export_hierarchical_conversation(self, date: Optional[str] = None, error: Optional[str] = None) -> str:
        """Export the hierarchical conversation log to JSON file and S3 (in production)"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use analysis date if provided, otherwise use current date
            period_identifier = date if date else timestamp[:8]  # Extract YYYYMMDD from timestamp
            filename = f"interpreter_{period_identifier}_{timestamp}.json"
            
            # Create agent_conversations directory structure in current working directory
            base_dir = Path.cwd() / 'agent_conversations' / 'anomaly_interpreter'
            base_dir.mkdir(parents=True, exist_ok=True)
            
            full_path = base_dir / filename
            
            conversation_data = {
                "metadata": {
                    "agent_type": "anomaly_interpreter",
                    "analysis_type": "hierarchical_generation_by_generation",
                    "export_timestamp": datetime.now().isoformat(),
                    "analysis_date": date,
                    "llm_type": self.llm_type.value,
                    "total_generations": len(set(r['generation'] for r in self.hierarchical_reflections)),
                    "total_nodes_analyzed": len(self.conversation_tracker.hierarchy_structure),
                    "error": error
                },
                "hierarchy_structure": self.conversation_tracker.hierarchy_structure,
                "hierarchy_summary": self.conversation_tracker.get_hierarchy_summary(),
                "conversation_log": self.conversation_tracker.conversation_log,
                "conversation_summary": self.conversation_tracker.get_conversation_summary(),
                "generation_reflections": self.hierarchical_reflections,
                "generation_analysis": self.generation_data
            }
            
            # Save locally
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"📝 Hierarchical conversation exported to: {full_path}")
            
            # Upload to S3 in production
            try:
                s3_key = await self.s3_uploader.upload_interpreter_conversation(conversation_data, filename)
                if s3_key:
                    self.logger.info(f"📤 Interpreter conversation uploaded to S3: {s3_key}")
                else:
                    self.logger.info("🔧 S3 upload skipped (local environment or failed)")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to upload to S3: {e}")
            
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

    def _get_cabin_sections_for_segment(self, segment: str) -> str:
        """
        Genera las secciones de cabina dinámicas según el segmento seleccionado.
        
        Args:
            segment: El segmento a analizar
            
        Returns:
            String con las secciones de cabina relevantes para el prompt
        """
        # Normalize segment path to dictionary keys
        if segment in ['Global', 'Global/']: 
            segment = 'Global'
        elif segment in ['Global/SH', 'Short Haul (SH)']: 
            segment = 'SH'
        elif segment in ['Global/LH', 'Long Haul (LH)']: 
            segment = 'LH'
        elif segment in ['Global/SH/Economy', 'SH Economy']: 
            segment = 'Economy SH'
        elif segment in ['Global/SH/Business', 'SH Business']: 
            segment = 'Business SH'
        elif segment in ['Global/SH/Premium', 'SH Premium']: 
            segment = 'Premium SH'
        elif segment in ['Global/LH/Economy', 'LH Economy']: 
            segment = 'Economy LH'
        elif segment in ['Global/LH/Business', 'LH Business']: 
            segment = 'Business LH'
        elif segment in ['Global/LH/Premium', 'LH Premium']: 
            segment = 'Premium LH'
        elif 'IB' in segment and 'Economy' in segment: # e.g. Global/SH/Economy/IB
            segment = 'IB'
        elif 'YW' in segment and 'Economy' in segment: # e.g. Global/SH/Economy/YW
            segment = 'YW'
        elif segment == 'IB':
            segment = 'IB'
        elif segment == 'YW':
            segment = 'YW'

        cabin_sections = {
            'Global': """
    **ECONOMY SH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] La cabina Economy de SH registró un NPS de [valor cabina] con [variación] pts vs L7d. **Desglose por compañía:** IB obtuvo [NPS IB] ([diff IB] pts) y YW [NPS YW] ([diff YW] pts). [Breve explicación de la dinámica: si ambas suben/bajan = efecto conjunto; si una sube y otra baja = se compensan; si solo una tiene variación = esa domina]
    
    **BUSINESS SH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] El segmento Business de SH registró un NPS de [valor cabina] con [variación] pts vs L7d. **Desglose por compañía:** IB obtuvo [NPS IB] ([diff IB] pts) y YW [NPS YW] ([diff YW] pts). [Breve explicación de la dinámica y causas principales de cada compañía con sus evidencias clave]
    
    **ECONOMY LH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] La cabina Economy de LH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha período]) con una [variación de NPS_diff cabina] puntos respecto a la semana anterior. [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "La causa principal fue [hipótesis con datos (drivers, operativa, NCS, verbatims) que la respaldan], complementada por [hipótesis secundarias (si las hubiera)]. Esta [mejora/deterioro] se reflejó especialmente en rutas como [top rutas con NPS y diff], mientras que los perfiles más reactivos incluyen [perfiles específicos]."]
    
    **BUSINESS LH: [Título]**  
    [PÁRRAFO NARRATIVO FLUIDO] La cabina Business de LH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha período]) con una [variación de NPS_diff cabina] puntos respecto al período anterior.     [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "Los drivers principales fueron [causas con SHAP], impactando especialmente las rutas [rutas específicas] y perfiles [perfiles específicos]."]
    
    **PREMIUM LH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] El segmento Premium de LH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha]) con [diff cabina] puntos de [variación] vs la semana anterior. [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "Las causas dominantes fueron [drivers SHAP], especialmente evidentes en [rutas top] y entre [perfiles reactivos]."]
            """,
            'SH': """
    **ECONOMY SH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] La cabina Economy de SH registró un NPS de [valor cabina] con [variación] pts vs L7d. **Desglose por compañía:** IB obtuvo [NPS IB] ([diff IB] pts) y YW [NPS YW] ([diff YW] pts). [Breve explicación de la dinámica: si ambas suben/bajan = efecto conjunto; si una sube y otra baja = se compensan; si solo una tiene variación = esa domina]
    
    **BUSINESS SH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] El segmento Business de SH registró un NPS de [valor cabina] con [variación] pts vs L7d. **Desglose por compañía:** IB obtuvo [NPS IB] ([diff IB] pts) y YW [NPS YW] ([diff YW] pts). [Breve explicación de la dinámica y causas principales de cada compañía con sus evidencias clave]
            """,
            'LH': """
    **ECONOMY LH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] La cabina Economy de LH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha período]) con una [variación de NPS_diff cabina] puntos respecto a la semana anterior. [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "La causa principal fue [hipótesis con datos (drivers, operativa, NCS, verbatims) que la respaldan], complementada por [hipótesis secundarias (si las hubiera)]. Esta [mejora/deterioro] se reflejó especialmente en rutas como [top rutas con NPS y diff], mientras que los perfiles más reactivos incluyen [perfiles específicos]."]
    
    **BUSINESS LH: [Título]**  
    [PÁRRAFO NARRATIVO FLUIDO] La cabina Business de LH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha]) con [diff cabina] puntos de [variación] vs período anterior. [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "Los drivers principales fueron [causas con SHAP], impactando especialmente las rutas transatlánticas [rutas específicas] y perfiles [perfiles específicos]."]
    
    **PREMIUM LH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] El segmento Premium de LH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha]) con [diff cabina] puntos de [variación] vs la semana anterior. [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "Las causas dominantes fueron [drivers SHAP], especialmente evidentes en [rutas top] y entre [perfiles reactivos]."]
            """,
            'Economy SH': """
    **ECONOMY SH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] La cabina Economy de SH [descripción - para segmentos estables usar: "mantuvo desempeño estable"] durante la semana del [fecha], registrando un NPS de [valor cabina] ([fecha período]) con una [variación de NPS_diff cabina] puntos respecto a la semana anterior. [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "La causa principal fue [hipótesis con datos (drivers, operativa, NCS, verbatims) que la respaldan], complementada por [hipótesis secundarias (si las hubiera)]. Esta [mejora/deterioro] se reflejó especialmente en rutas como [top rutas con NPS y diff], mientras que los perfiles más reactivos incluyen [perfiles específicos]."]
    
    **IMPORTANTE:** Si hay subsegmentos IB y YW disponibles, analiza ambos y reporta el comportamiento agregado de la cabina completa. MENCIONA EXPLÍCITAMENTE los valores NPS de cada compañía por separado (si una compañía no aparece en el árbol, indica que "mantuvo desempeño estable") antes de explicar el efecto neto en la cabina.
            """,
            'Business SH': """
    **BUSINESS SH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] El segmento Business de SH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha]) con una [variación de diff cabina] puntos vs la semana anterior. [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "Esta evolución se explica principalmente por [causas SHAP], siendo especialmente visible en rutas como [rutas top] y entre perfiles [perfiles reactivos]."]
    
    **IMPORTANTE:** Si hay subsegmentos IB y YW disponibles, analiza ambos y reporta el comportamiento agregado de la cabina completa. MENCIONA EXPLÍCITAMENTE los valores NPS de cada compañía por separado (si una compañía no aparece en el árbol, indica que "mantuvo desempeño estable") antes de explicar el efecto neto en la cabina.
            """,
            'Premium SH': """
    **PREMIUM SH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] El segmento Premium de SH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha]) con [diff cabina] puntos de [variación] vs la semana anterior. [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "Las causas dominantes fueron [drivers SHAP], especialmente evidentes en [rutas top] y entre perfiles [perfiles reactivos]."]
            """,
            'Economy LH': """
    **ECONOMY LH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] La cabina Economy de LH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha período]) con una [variación de NPS_diff cabina] puntos respecto a la semana anterior. [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "La causa principal fue [hipótesis con datos (drivers, operativa, NCS, verbatims) que la respaldan], complementada por [hipótesis secundarias (si las hubiera)]. Esta [mejora/deterioro] se reflejó especialmente en rutas como [top rutas con NPS y diff], mientras que los perfiles más reactivos incluyen [perfiles específicos]."]
            """,
            'Business LH': """
    **BUSINESS LH: [Título]**  
    [PÁRRAFO NARRATIVO FLUIDO] La cabina Business de LH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha]) con [diff cabina] puntos de [variación] vs período anterior. [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "Los drivers principales fueron [causas con SHAP], impactando especialmente las rutas transatlánticas [rutas específicas] y perfiles [perfiles específicos]."]
            """,
            'Premium LH': """
    **PREMIUM LH: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] El segmento Premium de LH [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha]) con [diff cabina] puntos de [variación] vs la semana anterior. [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "Las causas dominantes fueron [drivers SHAP], especialmente evidentes en [rutas top] y entre [perfiles reactivos]."]
            """,
            'IB': """
    **IB: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] La compañía IB [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha período]) con una [variación de NPS_diff cabina] puntos respecto a la semana anterior. [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "La causa principal fue [hipótesis con datos (drivers, operativa, NCS, verbatims) que la respaldan], complementada por [hipótesis secundarias (si las hubiera)]. Esta [mejora/deterioro] se reflejó especialmente en rutas como [top rutas con NPS y diff], mientras que los perfiles más reactivos incluyen [perfiles específicos]."]
            """,
            'YW': """
    **YW: [Título]**
    [PÁRRAFO NARRATIVO FLUIDO] La compañía YW [descripción - para segmentos estables usar: "mantuvo desempeño estable"], registrando un NPS de [valor cabina] ([fecha período]) con una [variación de NPS_diff cabina] puntos respecto a la semana anterior. [Para segmentos estables: "No se detectaron cambios significativos, manteniendo niveles consistentes de satisfacción." | Para segmentos con variaciones: "La causa principal fue [hipótesis con datos (drivers, operativa, NCS, verbatims) que la respaldan], complementada por [hipótesis secundarias (si las hubiera)]. Esta [mejora/deterioro] se reflejó especialmente en rutas como [top rutas con NPS y diff], mientras que los perfiles más reactivos incluyen [perfiles específicos]."]
            """
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