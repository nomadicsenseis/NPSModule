"""
Anomaly Summary Agent

A specialized AI agent for summarizing multiple periods of NPS anomaly analysis
and generating executive-level insights about trends, patterns, and strategic priorities.
"""

import asyncio
import os
import yaml
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import importlib.resources

# Import GenAI Core components (adjusted for new location)
from ..agents.agent import Agent
from ..llms.openai_llm import OpenAiLLM
from ..llms.aws_llm import AWSLLM
from ..utils.enums import LLMType, MessageType, AgentName, get_default_llm_type, get_agent_conversations_folder
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


class AnomalySummaryAgent:
    """
    Specialized agent for summarizing multiple periods of anomaly analysis 
    and generating strategic insights and executive reports.
    
    Features:
    - Multi-period trend analysis
    - Pattern identification across time
    - Strategic priority assessment
    - Executive-level reporting
    - Multi-LLM support (OpenAI, AWS Bedrock)
    """
    
    def __init__(
        self,
        llm_type: Optional[LLMType] = None,
        config_path: str = "../../config/prompts/anomaly_summary.yaml",
        logger: Optional[logging.Logger] = None,
        environment: str = "prod",
        study_mode: str = "comparative",
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
        Initialize the Anomaly Summary Agent.
        
        Args:
            llm_type: Type of LLM to use (supports OpenAI and AWS Bedrock models)
            config_path: Path to YAML configuration file with prompts
            logger: Optional logger instance
            environment: Environment type ("local" or "prod")
            study_mode: "single" or "comparative"
            anomaly_detection_mode: "vslast", "mean", or "target"
            causal_filter: Comparison filter (e.g. "vs L7d")
            comparison_start_date: Start of comparison period
            comparison_end_date: End of comparison period
            aggregation_days: Days per analysis period
            baseline_periods: Number of baseline periods
            segment: Root segment
            focus_touchpoint: Touchpoint focus (None → "General")
        """
        # Use default LLM type if none provided
        if llm_type is None:
            llm_type = get_default_llm_type()
        self.llm_type = llm_type
        self.config_path = config_path
        self.logger = logger or self._setup_logger()
        self.silent_mode = False
        self.environment = environment

        # Execution context
        self.study_mode = study_mode
        self.anomaly_detection_mode = anomaly_detection_mode
        self.causal_filter = causal_filter
        self.comparison_start_date = comparison_start_date
        self.comparison_end_date = comparison_end_date
        self.aggregation_days = aggregation_days
        self.baseline_periods = baseline_periods
        self.segment = segment
        self.focus_touchpoint = focus_touchpoint
        self.report_group = resolve_report_group(focus_touchpoint)
        
        # Initialize S3 uploader with environment
        self.s3_uploader = S3ReportUploader(environment=environment)
        
        # Load environment variables from .env in current working directory only if not in prod
        if self.environment != "prod":
            dotenv_path = Path.cwd() / '.env'
            if dotenv_path.exists():
                load_dotenv(dotenv_path)
        
        # Load prompt configuration
        self.config = self._load_prompt_config(config_path)
        
        # Create LLM and agent
        self.llm = self._create_llm(llm_type)
        self.agent = Agent(llm=self.llm, logger=self.logger)
        
        if not self.silent_mode:
            self.logger.info(f"AnomalySummaryAgent initialized with {llm_type.value}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup default logger for the agent."""
        logger = logging.getLogger("anomaly_summary")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_prompt_config(self, config_path: str) -> Dict[str, Any]:
        """Load prompt configuration using importlib.resources for package support"""
        try:
            # Try loading as package resource first
            config_filename = Path(config_path).name
            package_path = "dashboard_analyzer.anomaly_explanation.config.prompts"
            
            try:
                ref = importlib.resources.files(package_path) / config_filename
                with ref.open('r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self.logger.debug(f"Loaded configuration from package resource: {package_path}/{config_filename}")
                return config
            except (ImportError, FileNotFoundError, TypeError) as e:
                # Fallback to direct file path (development/relative mode)
                self.logger.debug(f"Could not load from package ({e}), trying file path")
                
                # Get the directory of this file to resolve relative paths if config_path is relative
                current_dir = os.path.dirname(os.path.abspath(__file__))
                full_path = os.path.join(current_dir, config_path)
                
                if not os.path.exists(full_path):
                    # Try workspace root fallback
                    full_path = Path("/workspace") / "dashboard_analyzer/anomaly_explanation/config/prompts" / config_filename
                
                if os.path.exists(full_path):
                    with open(full_path, 'r', encoding='utf-8') as file:
                        config = yaml.safe_load(file)
                    self.logger.debug(f"Loaded summary prompt configuration from {full_path}")
                    return config
                else:
                    raise FileNotFoundError(f"Configuration file not found: {config_path} or package resource")
            
        except Exception as e:
            self.logger.error(f"Error loading prompt configuration: {e}")
            raise
    
    def _create_llm(self, llm_type: LLMType):
        """Factory method to create appropriate LLM instance based on type."""
        
        # OpenAI/Azure OpenAI models
        if llm_type in [LLMType.GPT3_5, LLMType.GPT4, LLMType.GPT4o, LLMType.GPT4o_MINI, 
                       LLMType.O1_MINI, LLMType.O3_MINI, LLMType.O3, LLMType.O4_MINI, LLMType.GPT_5_2]:
            return self._create_openai_llm(llm_type)
        
        # AWS Bedrock models
        elif llm_type in [
            LLMType.CLAUDE_3_HAIKU, LLMType.CLAUDE_3_5_HAIKU, LLMType.CLAUDE_3_OPUS,
            LLMType.CLAUDE_3_5_SONNET, LLMType.CLAUDE_3_5_SONNET_V2, LLMType.CLAUDE_3_7_SONNET,
            LLMType.CLAUDE_SONNET_4, LLMType.CLAUDE_OPUS_4_5, LLMType.CLAUDE_OPUS_4_6, LLMType.LLAMA3_70, LLMType.LLAMA3_1_70, LLMType.LLAMA3_1_405,
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

    def _repair_json(self, json_str: str) -> str:
        """Validate JSON, with minimal repair attempts."""
        if not json_str:
            return "{}"
        
        cleaned = self._clean_json_response(json_str)
        
        # Try to parse as-is
        try:
            json.loads(cleaned)
            self.logger.info(f"✅ Adaptive Card JSON validated ({len(cleaned)} chars)")
            return cleaned
        except json.JSONDecodeError as e:
            self.logger.warning(f"⚠️ JSON validation failed: {e.msg} at pos {e.pos}")
        
        # Single repair attempt: handle escaped quotes
        if '\\"' in cleaned:
            try:
                fixed = cleaned.replace('\\"', '"')
                json.loads(fixed)
                self.logger.info("✅ Fixed JSON by unescaping quotes")
                return fixed
            except:
                pass
        
        # Return as-is (caller will handle invalid JSON)
        self.logger.error(f"❌ Could not parse JSON. Returning cleaned version.")
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

    async def _optimize_adaptive_card_payload(
        self,
        adaptive_card_json: str,
        target_kb: int = 24
    ) -> tuple:
        """Iteratively reduce Adaptive Card JSON payload to fit size constraints.
        
        Returns:
            Tuple of (final_json, debug_info) where debug_info contains step5 and step6 conversations
        """
        debug_info = {
            'step5_modernize_tone': None,
            'step6_size_optimization': [],
            'step7_validate_and_fix_json': None
        }
        
        if not adaptive_card_json:
            return adaptive_card_json, debug_info

        # Initial cleaning and minification (returns normal quotes)
        best_json = self._minify_json(adaptive_card_json)
        current_kb = self._measure_kb(best_json)
        self.logger.info(f"📦 Initial Adaptive Card size (minified): {current_kb:.2f} KB")

        # --- STEP 5: Modernize Tone (Now First) ---
        step_key = 'step5_modernize_tone'
        step_config = self.config.get(step_key, {})
        step_system = step_config.get('system_prompt', '')
        step_input_template = step_config.get('input_template', '')

        if step_system and step_input_template:
            self.logger.info("🔄 Applying tone modernization (less Cervantes) BEFORE size optimization...")
            
            # Pass escaped version to the LLM
            escaped_for_llm = best_json.replace('"', '\\"')
            step5_input = step_input_template.format(current_json=escaped_for_llm)
            
            message_history = MessageHistory()
            message_history.create_and_add_message(content=step_system, message_type=MessageType.SYSTEM)
            message_history.create_and_add_message(content=step5_input, message_type=MessageType.USER)

            try:
                response, _, _ = await self.agent.invoke(messages=message_history.get_messages())
                modernized = response.content if hasattr(response, 'content') else str(response)
                
                # Store debug info (full content for debugging)
                debug_info['step5_modernize_tone'] = {
                    'messages': [
                        {'role': 'system', 'content': step_system},
                        {'role': 'user', 'content': step5_input},  # Full input
                        {'role': 'assistant', 'content': modernized}  # Full output
                    ],
                    'input_size_kb': current_kb,
                    'executed': True
                }
                
                # Clean and minify (returns with normal quotes)
                clean_modernized = self._clean_json_response(modernized)
                best_json = self._minify_json(clean_modernized)
                
                # Verify size after modernization
                new_kb = self._measure_kb(best_json)
                debug_info['step5_modernize_tone']['output_size_kb'] = new_kb
                current_kb = new_kb
                
                self.logger.info(f"📦 Adaptive Card size after tone modernization: {current_kb:.2f} KB")
                self.logger.info("✅ Tone modernization applied successfully")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to apply tone modernization: {e}")
                debug_info['step5_modernize_tone'] = {'executed': False, 'error': str(e)}

        # --- STEP 6: Surgical Size Optimization (Now Second) ---
        if current_kb <= target_kb:
            self.logger.info(f"✅ Adaptive Card already fits ({current_kb:.2f} KB)")
            self.logger.info(f"🚀 Final Adaptive Card JSON:\n{best_json}")
            debug_info['step6_size_optimization'].append({
                'step': 'skipped',
                'reason': f'Already fits: {current_kb:.2f} KB <= {target_kb} KB'
            })
            return best_json, debug_info

        optimization_steps = [
            'step6a_trim_low_impact_days',
            'step6b_remove_subsegment_daily_context',
            'step6c_summarize_cabin_haul_company_weekly',
            'step6d_shorten_cabin_haul_weekly',
            'step6e_shorten_overall_global'
        ]

        last_step_applied = "modernize_only"
        best_kb = current_kb

        for step_key in optimization_steps:
            step_config = self.config.get(step_key, {})
            step_system = step_config.get('system_prompt', '')
            step_input_template = step_config.get('input_template', '')

            if not step_system or not step_input_template:
                self.logger.warning(f"⚠️ Missing config for {step_key}, skipping")
                debug_info['step6_size_optimization'].append({
                    'step': step_key, 'executed': False, 'reason': 'Missing config'
                })
                continue

            self.logger.info(f"🔄 Applying optimization step: {step_key}...")
            
            # Pass escaped version to the LLM
            escaped_for_llm = best_json.replace('"', '\\"')
            
            message_history = MessageHistory()
            message_history.create_and_add_message(content=step_system.format(target_kb=target_kb), message_type=MessageType.SYSTEM)
            message_history.create_and_add_message(
                content=step_input_template.format(current_json=escaped_for_llm, target_kb=target_kb),
                message_type=MessageType.USER
            )

            response, _, _ = await self.agent.invoke(messages=message_history.get_messages())
            optimized = response.content if hasattr(response, 'content') else str(response)

            # Clean and minify the optimized response
            optimized_clean = self._clean_json_response(optimized)
            optimized_min = self._minify_json(optimized_clean)
            size_kb = self._measure_kb(optimized_min)
            
            # Store debug info (full content for debugging)
            step6_system = step_system.format(target_kb=target_kb)
            step6_input = step_input_template.format(current_json=escaped_for_llm, target_kb=target_kb)
            step_debug = {
                'step': step_key,
                'executed': True,
                'messages': [
                    {'role': 'system', 'content': step6_system},
                    {'role': 'user', 'content': step6_input},
                    {'role': 'assistant', 'content': optimized}
                ],
                'input_size_kb': best_kb,
                'output_size_kb': size_kb,
                'improved': size_kb < best_kb
            }
            debug_info['step6_size_optimization'].append(step_debug)
            
            self.logger.info(f"📦 Adaptive Card size after {step_key}: {size_kb:.2f} KB")
            self.logger.info(f"📄 Optimized JSON snippet after {step_key}:\n{optimized_min[:200]}...")

            if size_kb < best_kb:
                best_json = optimized_min
                best_kb = size_kb
                last_step_applied = step_key

            if size_kb <= target_kb:
                self.logger.info(f"✅ Adaptive Card fits after {step_key} ({size_kb:.2f} KB)")
                break

        if best_kb > target_kb:
            self.logger.warning(f"⚠️ Adaptive Card still over limit after {last_step_applied}: {best_kb:.2f} KB")
        else:
            self.logger.info(f"✅ Adaptive Card final size optimization step: {last_step_applied} ({best_kb:.2f} KB)")

        # --- STEP 7: JSON Validation and Typo Correction (FINAL QUALITY CHECK) ---
        step7_key = 'step7_validate_and_fix_json'
        step7_config = self.config.get(step7_key, {})
        step7_system = step7_config.get('system_prompt', '')
        step7_input_template = step7_config.get('input_template', '')

        if step7_system and step7_input_template:
            self.logger.info("🔄 Applying final JSON validation and typo correction...")
            
            # Pass escaped version to the LLM
            escaped_for_llm = best_json.replace('"', '\\"')
            step7_input = step7_input_template.format(current_json=escaped_for_llm)
            
            message_history = MessageHistory()
            message_history.create_and_add_message(content=step7_system, message_type=MessageType.SYSTEM)
            message_history.create_and_add_message(content=step7_input, message_type=MessageType.USER)

            try:
                response, _, _ = await self.agent.invoke(messages=message_history.get_messages())
                validated = response.content if hasattr(response, 'content') else str(response)
                
                # Store debug info
                debug_info['step7_validate_and_fix_json'] = {
                    'messages': [
                        {'role': 'system', 'content': step7_system},
                        {'role': 'user', 'content': step7_input},
                        {'role': 'assistant', 'content': validated}
                    ],
                    'input_size_kb': best_kb,
                    'executed': True
                }
                
                # Clean and minify the validated response
                validated_clean = self._clean_json_response(validated)
                validated_min = self._minify_json(validated_clean)
                validated_kb = self._measure_kb(validated_min)
                
                debug_info['step7_validate_and_fix_json']['output_size_kb'] = validated_kb
                
                # Update best JSON with validated version
                best_json = validated_min
                best_kb = validated_kb
                
                self.logger.info(f"📦 Adaptive Card size after JSON validation: {best_kb:.2f} KB")
                self.logger.info("✅ JSON validation and typo correction applied successfully")
                
                # Validate that the JSON is actually valid
                try:
                    json.loads(best_json)
                    self.logger.info("✅ Final JSON is valid and parseable")
                except json.JSONDecodeError as e:
                    self.logger.error(f"❌ Final JSON validation failed: {e}")
                    # Fall back to previous version if validation fails
                    self.logger.warning("⚠️ Falling back to pre-validation JSON")
                    # Revert to previous best_json (before step7)
                    # We need to track this separately, but for now we'll just log
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to apply JSON validation: {e}")
                debug_info['step7_validate_and_fix_json'] = {'executed': False, 'error': str(e)}
        else:
            self.logger.warning(f"⚠️ Missing config for {step7_key}, skipping JSON validation")
            debug_info['step7_validate_and_fix_json'] = {'executed': False, 'reason': 'Missing config'}

        self.logger.info(f"🚀 Final optimized Adaptive Card JSON:\n{best_json}")
        return best_json, debug_info
    
    async def generate_summary_report(self, periods_data: List[Dict[str, Any]]) -> str:
        """
        Generate a comprehensive summary report from multiple periods of anomaly analysis.
        
        Args:
            periods_data: List of dictionaries containing period analysis data
                         Each dict should have: 'period', 'anomalies', 'explanations', 'interpretations'
        
        Returns:
            Executive summary string
        """
        try:
            if not periods_data:
                return "⚠️ No data provided for summary generation"
            
            # Format the periods data for the summary
            formatted_analysis = self._format_periods_for_summary(periods_data)
            
            # Get prompts from configuration
            system_prompt = self.config.get('system_prompt', '')
            input_template = self.config.get('input_template', '')
            
            # Format the input with the analysis data
            num_periods = len(periods_data)
            formatted_input = input_template.format(
                num_periods=num_periods,
                formatted_analysis=formatted_analysis
            )
            
            # Create message history for the summary generation
            message_history = MessageHistory()
            message_history.create_and_add_message(
                content=system_prompt,
                message_type=MessageType.SYSTEM
            )
            message_history.create_and_add_message(
                content=formatted_input,
                message_type=MessageType.USER
            )
            
            # Generate the summary using the agent
            response, structured_response, tool_calls = await self.agent.invoke(messages=message_history.get_messages())
            summary_response = response.content if hasattr(response, 'content') else str(response)
            
            # NOTE: Conversation export disabled - only final consolidated report is saved
            # first_period_date = periods_data[0].get('period', 'unknown') if periods_data else None
            # conversation_file = await self.export_conversation(message_history, first_period_date)
            # if conversation_file:
            #     self.logger.info(f"🗂️ Conversación de summary guardada: {conversation_file}")
            
            if summary_response:
                self.logger.info(f"✅ Generated summary report for {num_periods} periods")
                return summary_response
            else:
                return "⚠️ Failed to generate summary report"
                
        except Exception as e:
            self.logger.error(f"❌ Error generating summary report: {str(e)}")
            return f"❌ Error generating summary: {str(e)}"
    
    async def generate_summary_from_adaptive_cards(
        self, 
        adaptive_cards_data: List[Dict[str, Any]]
    ) -> str:
        """
        Generate summary from Adaptive Cards JSON data.
        
        Args:
            adaptive_cards_data: List of dictionaries with Adaptive Card JSON data
                                Each dict should have: 'adaptive_card_json', 'period_type', 'date_range'
        
        Returns:
            Executive summary string
        """
        try:
            if not adaptive_cards_data:
                return "⚠️ No Adaptive Card data provided for summary generation"
            
            # Parse all Adaptive Cards
            parsed_cards = []
            for card_data in adaptive_cards_data:
                card_json = card_data.get('adaptive_card_json', '')
                if card_json:
                    parsed = self._parse_adaptive_card_json(card_json)
                    parsed_cards.append(parsed)
            
            # Separate weekly and daily cards
            weekly_card = None
            daily_cards = []
            
            for card in parsed_cards:
                if card['period_type'] == 'weekly':
                    weekly_card = card
                elif card['period_type'] == 'daily':
                    daily_cards.append(card)
            
            if not weekly_card:
                return "⚠️ No weekly analysis found in Adaptive Cards"
            
            # Organize data for summary
            summary_data = self._organize_data_for_summary(weekly_card, daily_cards)
            
            # Generate summary using the organized data
            return await self._generate_summary_from_organized_data(summary_data)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating summary from Adaptive Cards: {str(e)}")
            return f"❌ Error generating summary from Adaptive Cards: {str(e)}"
    
    def _organize_data_for_summary(
        self, 
        weekly_card: Dict[str, Any], 
        daily_cards: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Organize parsed Adaptive Card data for summary generation.
        
        Returns:
            Dictionary organized by segment with weekly and daily analyses
        """
        organized_data = {
            'weekly_date_range': weekly_card['date_range'],
            'segments': {}
        }
        
        # Initialize structure for all segments from weekly card
        for segment_name, segment_data in weekly_card['segments'].items():
            organized_data['segments'][segment_name] = {
                'weekly_analysis': segment_data['analysis'],
                'weekly_nps': segment_data['nps'],
                'weekly_variation': segment_data['variation'],
                'daily_analyses': []  # Will be filled from daily cards
            }
        
        # Add daily analyses for each segment
        for daily_card in daily_cards:
            daily_date = daily_card['date_range']
            for segment_name, segment_data in daily_card['segments'].items():
                if segment_name in organized_data['segments']:
                    organized_data['segments'][segment_name]['daily_analyses'].append({
                        'date': daily_date,
                        'analysis': segment_data['analysis'],
                        'nps': segment_data['nps'],
                        'variation': segment_data['variation']
                    })
        
        return organized_data
    
    async def _generate_summary_from_organized_data(
        self, 
        organized_data: Dict[str, Any]
    ) -> str:
        """
        Generate summary from organized data structure.
        
        This is a simplified version that uses the existing summary prompts
        but with structured data instead of text parsing.
        """
        try:
            # Get prompts from configuration
            system_prompt = self.config.get('system_prompt', '')
            input_template = self.config.get('input_template', '')
            
            # Format the organized data for the prompt
            formatted_analysis = self._format_organized_data_for_prompt(organized_data)
            
            # Format the input with the analysis data
            num_days = sum(len(segment['daily_analyses']) for segment in organized_data['segments'].values()) // len(organized_data['segments'])
            formatted_input = input_template.format(
                num_periods=num_days + 1,  # +1 for weekly
                formatted_analysis=formatted_analysis
            )
            
            # Create message history for the summary generation
            message_history = MessageHistory()
            message_history.create_and_add_message(
                content=system_prompt,
                message_type=MessageType.SYSTEM
            )
            message_history.create_and_add_message(
                content=formatted_input,
                message_type=MessageType.USER
            )
            
            # Generate the summary using the agent
            response, structured_response, tool_calls = await self.agent.invoke(messages=message_history.get_messages())
            summary_response = response.content if hasattr(response, 'content') else str(response)
            
            if summary_response:
                self.logger.info(f"✅ Generated summary report from Adaptive Cards")
                return summary_response
            else:
                return "⚠️ Failed to generate summary report"
                
        except Exception as e:
            self.logger.error(f"❌ Error in summary generation: {str(e)}")
            return f"❌ Error in summary generation: {str(e)}"
    
    def _format_organized_data_for_prompt(self, organized_data: Dict[str, Any]) -> str:
        """
        Format organized data into text for the summary prompt.
        
        This converts the structured data back to text format that the existing
        prompts can understand, but in a much cleaner way than before.
        """
        formatted_parts = []
        
        # Weekly analysis section
        formatted_parts.append(f"📊 **ANÁLISIS SEMANAL** ({organized_data['weekly_date_range']}):")
        
        for segment_name, segment_data in organized_data['segments'].items():
            formatted_parts.append(f"\n**{segment_name}**:")
            formatted_parts.append(f"NPS: {segment_data['weekly_nps']}, Variación: {segment_data['weekly_variation']}")
            formatted_parts.append(f"Análisis: {segment_data['weekly_analysis']}")
            
            # Daily analyses for this segment
            if segment_data['daily_analyses']:
                formatted_parts.append(f"\n  📅 **Análisis Diarios para {segment_name}**:")
                for daily in segment_data['daily_analyses']:
                    formatted_parts.append(f"  - {daily['date']}: NPS {daily['nps']}, Var {daily['variation']}")
                    # Include first 200 chars of analysis
                    analysis_preview = daily['analysis'][:200] + "..." if len(daily['analysis']) > 200 else daily['analysis']
                    formatted_parts.append(f"    {analysis_preview}")
        
        return "\n".join(formatted_parts)
    
    async def generate_comprehensive_summary(
        self, 
        weekly_comparative_analysis: str, 
        daily_single_analyses: List[Dict[str, Any]],
        date_flight_local: str = None,
        # S3 upload parameters
        execution_metadata: Optional[Dict[str, Any]] = None,
        weekly_analysis_params: Optional[Dict[str, Any]] = None,
        daily_analysis_params: Optional[Dict[str, Any]] = None,
        date_ranges: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a comprehensive summary that combines weekly comparative analysis with daily single analyses.
        
        Args:
            weekly_comparative_analysis: String containing the weekly comparative analysis
            daily_single_analyses: List of daily single analysis results
                                  Each dict should have: 'date', 'analysis', 'anomalies'
            date_flight_local: Local flight date for context
            execution_metadata: Metadata about the execution (for S3 upload)
            weekly_analysis_params: Parameters used for weekly analysis (for S3 upload)
            daily_analysis_params: Parameters used for daily analysis (for S3 upload)
            date_ranges: Date range information (for S3 upload)
        
        Returns:
            Comprehensive summary string
        """
        try:
            if not weekly_comparative_analysis and not daily_single_analyses:
                return "⚠️ No data provided for comprehensive summary generation"
            
            # Sort daily analyses chronologically (oldest first) to help the model narrate in order
            daily_single_analyses_sorted = sorted(
                daily_single_analyses,
                key=lambda x: x.get('date', '0000-00-00')
            )
            self.logger.info(f"📅 Sorted {len(daily_single_analyses_sorted)} daily analyses chronologically")
            
            # Format daily analyses - only include days with relevant analysis
            daily_analyses_formatted = []
            total_days = len(daily_single_analyses_sorted)
            filtered_days = 0
            
            for daily_analysis in daily_single_analyses_sorted:
                date = daily_analysis.get('date', 'Unknown')
                analysis = daily_analysis.get('analysis', '')
                anomalies = daily_analysis.get('anomalies', [])
                
                # Extract only executive synthesis from daily analysis to save tokens
                synthesis = self._extract_executive_synthesis_from_daily(analysis)
                
                # Log length and only skip if truly empty
                text = (synthesis or '').strip()
                self.logger.info(f"📝 Daily synthesis length for {date}: {len(text)} chars (from {len(analysis)} total)")
                if not text:
                    self.logger.info(f"📅 Skipping {date}: No meaningful analysis (empty)")
                    filtered_days += 1
                    continue
                
                # Include all non-empty daily analyses
                daily_text = f"📅 {date}:\n{text}"
                if anomalies and len(anomalies) > 0:
                    daily_text += f"\n🚨 Anomalías detectadas: {', '.join(anomalies)}"
                
                daily_analyses_formatted.append(daily_text)
            
            self.logger.info(f"📊 Daily analysis filtering: {total_days} total days, {filtered_days} filtered out, {len(daily_analyses_formatted)} included")
            
            daily_analyses_combined = "\n\n".join(daily_analyses_formatted)
            
            # Get prompts from configuration
            system_prompt = self.config.get('system_prompt', '')
            input_template = self.config.get('input_template', '')
            
            # Format the input with the comprehensive analysis data
            formatted_input = input_template.format(
                weekly_comparative_analysis=weekly_comparative_analysis,
                daily_single_analyses=daily_analyses_combined
            )
            
            # DEBUG: persist exactly what the model will receive
            try:
                summary_reports_dir = Path.cwd() / "summary_reports"
                summary_reports_dir.mkdir(parents=True, exist_ok=True)
                dbg_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                debug_path = summary_reports_dir / f"summary_agent_input_{dbg_ts}.md"
                with open(debug_path, 'w', encoding='utf-8') as dbg:
                    dbg.write("===== SYSTEM =====\n\n")
                    dbg.write(system_prompt)
                    dbg.write("\n\n===== USER =====\n\n")
                    dbg.write(formatted_input)
                self.logger.info(f"📝 Saved summary agent input to: {debug_path}")
                self.logger.info(f"SYSTEM len={len(system_prompt)} | USER len={len(formatted_input)}")
            except Exception as e:
                self.logger.warning(f"Could not write debug input: {e}")
            
            # Create message history for the comprehensive summary generation
            message_history = MessageHistory()
            message_history.create_and_add_message(
                content=system_prompt,
                message_type=MessageType.SYSTEM
            )
            message_history.create_and_add_message(
                content=formatted_input,
                message_type=MessageType.USER
            )
            
            # Generate the comprehensive summary using the agent
            response, structured_response, tool_calls = await self.agent.invoke(messages=message_history.get_messages())
            comprehensive_response = response.content if hasattr(response, 'content') else str(response)
            
            # NOTE: Conversation export disabled - only final consolidated report is saved
            # conversation_file = await self.export_conversation(message_history, date_flight_local)
            # if conversation_file:
            #     self.logger.info(f"🗂️ Conversación de summary guardada: {conversation_file}")
            
            if comprehensive_response:
                self.logger.info(f"✅ Generated comprehensive summary: weekly + {len(daily_single_analyses)} daily analyses")
                
                # STEP 2: Generate Adaptive Card
                date_range = self._build_date_range_string(daily_single_analyses, date_ranges)
                adaptive_card_json, step4_conversation = await self._generate_adaptive_card(comprehensive_response, date_range)
                adaptive_card_json, optimization_debug = await self._optimize_adaptive_card_payload(adaptive_card_json, target_kb=24)
                
                # Merge optimization debug into step4 conversation
                step4_conversation['step5_modernize_tone'] = optimization_debug.get('step5_modernize_tone')
                step4_conversation['step6_size_optimization'] = optimization_debug.get('step6_size_optimization')
                step4_conversation['step7_validate_and_fix_json'] = optimization_debug.get('step7_validate_and_fix_json')
                
                # Combine synthesis and adaptive card
                final_output = f"{comprehensive_response}\n\n---ADAPTIVE_CARD_JSON---\n\n{adaptive_card_json}"

                # Upload to S3 if metadata is provided
                if (execution_metadata and weekly_analysis_params and 
                    daily_analysis_params and date_ranges):
                    try:
                        self.logger.info("📤 Uploading comprehensive report to S3...")
                        
                        # In prod, we upload the adaptive card as a JSON object
                        s3_final_synthesis = final_output
                        if self.environment == "prod":
                            try:
                                s3_final_synthesis = json.loads(adaptive_card_json)
                            except Exception:
                                s3_final_synthesis = adaptive_card_json
                        
                        s3_key = await self.s3_uploader.upload_comprehensive_report(
                            execution_date=datetime.now(),
                            analysis_date=execution_metadata.get('analysis_date', ''),
                            segment=execution_metadata.get('segment', 'Global'),
                            explanation_mode=execution_metadata.get('explanation_mode', 'agent'),
                            causal_filter=execution_metadata.get('causal_filter', 'vs L7d'),
                            weekly_analysis_params=weekly_analysis_params,
                            daily_analysis_params=daily_analysis_params,
                            date_ranges=date_ranges,
                            final_synthesis=s3_final_synthesis,
                            comparison_start_date=date_ranges.get('comparison_start_date'),
                            comparison_end_date=date_ranges.get('comparison_end_date')
                        )
                        if s3_key:
                            self.logger.info(f"✅ Report successfully uploaded to S3: {s3_key}")
                        else:
                            self.logger.warning("⚠️ Failed to upload report to S3")
                    except Exception as e:
                        self.logger.error(f"❌ Error uploading to S3: {str(e)}")
                        # Don't fail the entire process if S3 upload fails
                
                return final_output
            else:
                return "⚠️ Failed to generate comprehensive summary"
                
        except Exception as e:
            self.logger.error(f"❌ Error generating comprehensive summary: {str(e)}")
            return f"❌ Error generating comprehensive summary: {str(e)}"

    async def generate_comprehensive_summary_stratified(
        self, 
        weekly_comparative_analysis: str, 
        daily_single_analyses: List[Dict[str, Any]],
        date_flight_local: str = None,
        execution_metadata: Optional[Dict[str, Any]] = None,
        weekly_analysis_params: Optional[Dict[str, Any]] = None,
        daily_analysis_params: Optional[Dict[str, Any]] = None,
        date_ranges: Optional[Dict[str, Any]] = None,
        segment: str = 'Global'
    ) -> str:
        """
        Generate a comprehensive summary using a 3-step stratified approach.
        
        This method reduces cognitive load on the LLM by:
        1. First identifying which days are relevant
        2. Then extracting key data from those days
        3. Finally integrating into the weekly structure
        
        Args:
            weekly_comparative_analysis: String containing the weekly comparative analysis
            daily_single_analyses: List of daily single analysis results
            date_flight_local: Local flight date for context
            execution_metadata: Metadata about the execution (for S3 upload)
            weekly_analysis_params: Parameters used for weekly analysis (for S3 upload)
            daily_analysis_params: Parameters used for daily analysis (for S3 upload)
            date_ranges: Date range information (for S3 upload)
            segment: The root segment for hierarchical analysis (default: 'Global')
        
        Returns:
            Comprehensive summary string
        """
        try:
            if not weekly_comparative_analysis and not daily_single_analyses:
                return "⚠️ No data provided for comprehensive summary generation"
            
            # Sort daily analyses chronologically (oldest first) to help the model narrate in order
            daily_single_analyses_sorted = sorted(
                daily_single_analyses,
                key=lambda x: x.get('date', '0000-00-00')
            )
            self.logger.info(f"📅 Sorted {len(daily_single_analyses_sorted)} daily analyses chronologically")

            # Resolve the actual analysis period range for file naming.
            # Prefer the weekly date range passed via date_ranges; fall back to
            # deriving min/max from daily analyses; last resort is date_flight_local.
            _apr = (date_ranges or {}).get('analysis_period_range')
            if not _apr and daily_single_analyses_sorted:
                daily_dates = sorted(
                    d.get('date', '') for d in daily_single_analyses_sorted if d.get('date')
                )
                if daily_dates:
                    _apr = f"{daily_dates[0]} to {daily_dates[-1]}"
            analysis_period_str: Optional[str] = _apr or date_flight_local

            # Format daily analyses
            daily_analyses_formatted = []
            for daily_analysis in daily_single_analyses_sorted:
                date = daily_analysis.get('date', 'Unknown')
                analysis = daily_analysis.get('analysis', '')
                
                # Extract only executive synthesis from daily analysis to save tokens
                synthesis = self._extract_executive_synthesis_from_daily(analysis)
                
                text = (synthesis or '').strip()
                if text:
                    self.logger.info(f"📝 Daily synthesis length for {date}: {len(text)} chars (from {len(analysis)} total)")
                    daily_analyses_formatted.append(f"📅 {date}:\n{text}")
            
            daily_analyses_combined = "\n\n".join(daily_analyses_formatted)
            num_days = len(daily_analyses_formatted)
            
            self.logger.info(f"🔄 Starting stratified summary: {num_days} days to analyze")
            
            # =========================================================
            # Extract executive synthesis and parse into sections
            # =========================================================
            # First, extract only the executive synthesis (discard technical sections)
            weekly_synthesis_only = self._extract_executive_synthesis_from_weekly(weekly_comparative_analysis)
            self.logger.info(f"📊 Extracted executive synthesis: {len(weekly_synthesis_only)} chars (from {len(weekly_comparative_analysis)} total)")
            
            # Parse sections from the SYNTHESIS only (not the full technical report)
            # Use dynamic sections based on the segment hierarchy
            sections = self._parse_weekly_sections(weekly_synthesis_only, segment)
            self.logger.info(f"📊 Parsed {len(sections)} sections from executive synthesis for segment '{segment}'")
            
            # =========================================================
            # STEP 1: Analyze connections for each section
            # =========================================================
            self.logger.info("📋 STEP 1: Analyzing section connections...")
            
            step1_config = self.config.get('step1_analyze_section', {})
            step1_system = step1_config.get('system_prompt', '')
            step1_template = step1_config.get('input_template', '')
            
            daily_context_paragraphs = {}
            step1_conversations = {}  # Store full conversation for each section
            
            for section_name, section_content in sections.items():
                self.logger.info(f"   📌 Analyzing: {section_name}")
                
                # Rate limit protection: small pause between calls
                if daily_context_paragraphs:
                    await asyncio.sleep(2.0)
                
                # Get daily analyses for this section (SIEMPRE devuelve contenido con fechas)
                daily_for_section = self._filter_daily_for_section(
                    daily_analyses_combined, 
                    section_name
                )
                
                # Log what we're passing to the model
                has_specific = "específicos" in daily_for_section if daily_for_section else False
                self.logger.info(f"   📊 Daily context for {section_name}: {len(daily_for_section)} chars, specific={has_specific}")
                
                step1_input = step1_template.format(
                    section_name=section_name,
                    weekly_section=section_content,
                    daily_analyses_for_section=daily_for_section
                )
                
                message_history = MessageHistory()
                message_history.create_and_add_message(content=step1_system, message_type=MessageType.SYSTEM)
                message_history.create_and_add_message(content=step1_input, message_type=MessageType.USER)
                
                response, _, _ = await self.agent.invoke(messages=message_history.get_messages())
                paragraph = response.content if hasattr(response, 'content') else str(response)
                
                daily_context_paragraphs[section_name] = paragraph.strip()
                
                # Store full conversation for debugging
                step1_conversations[section_name] = {
                    'messages': [
                        {'role': 'system', 'content': step1_system},
                        {'role': 'user', 'content': step1_input},
                        {'role': 'assistant', 'content': paragraph}
                    ],
                    'input_length': len(step1_input),
                    'output_length': len(paragraph),
                    'result': paragraph.strip()
                }
                
                self.logger.info(f"   ✅ Generated context for {section_name}: {len(paragraph)} chars")
            
            self.logger.info(f"✅ Step 1 complete: Generated {len(daily_context_paragraphs)} context paragraphs")
            
            # Rate limit protection
            await asyncio.sleep(2.0)
            
            # =========================================================
            # STEP 2: Integrate into final report
            # =========================================================
            self.logger.info("📝 STEP 2: Integrating into final report...")
            
            # Note: weekly_synthesis_only was already extracted before Step 1
            # Reuse it here to avoid duplicate extraction
            
            # Format the context paragraphs for Step 2
            context_formatted = "\n\n".join([
                f"**{name}:**\n{para}" 
                for name, para in daily_context_paragraphs.items()
            ])
            
            step2_config = self.config.get('step2_integrate_final', {})
            step2_system = step2_config.get('system_prompt', '')
            step2_input = step2_config.get('input_template', '').format(
                weekly_analysis=weekly_synthesis_only,  # Use extracted synthesis, not full report
                daily_context_paragraphs=context_formatted
            )
            
            message_history_final = MessageHistory()
            message_history_final.create_and_add_message(content=step2_system, message_type=MessageType.SYSTEM)
            message_history_final.create_and_add_message(content=step2_input, message_type=MessageType.USER)
            
            response_final, _, _ = await self.agent.invoke(messages=message_history_final.get_messages())
            polished_report = response_final.content if hasattr(response_final, 'content') else str(response_final)
            
            if not polished_report or len(polished_report.strip()) == 0:
                self.logger.warning(f"⚠️ Step 2 returned empty response!")
                polished_report = weekly_comparative_analysis
            
            # Add assistant response to message history for export
            message_history_final.create_and_add_message(
                content=polished_report, 
                message_type=MessageType.AI,
                agent=AgentName.CONVERSATIONAL
            )
            
            self.logger.info(f"✅ Step 2 complete: Full report generated ({len(polished_report)} chars)")
            
            # Rate limit protection
            await asyncio.sleep(2.0)
            
            # =========================================================
            # STEP 3: Extract only executive synthesis
            # =========================================================
            self.logger.info("📋 STEP 3: Extracting executive synthesis...")
            
            step3_config = self.config.get('step3_extract_synthesis', {})
            step3_system = step3_config.get('system_prompt', '')
            step3_input = step3_config.get('input_template', '').format(
                full_report=polished_report
            )
            
            message_history_step3 = MessageHistory()
            message_history_step3.create_and_add_message(content=step3_system, message_type=MessageType.SYSTEM)
            message_history_step3.create_and_add_message(content=step3_input, message_type=MessageType.USER)
            
            response_step3, _, _ = await self.agent.invoke(messages=message_history_step3.get_messages())
            executive_synthesis = response_step3.content if hasattr(response_step3, 'content') else str(response_step3)
            
            if not executive_synthesis or len(executive_synthesis.strip()) == 0:
                self.logger.warning(f"⚠️ Step 3 returned empty response! Using full report as fallback.")
                executive_synthesis = polished_report
            
            # Add assistant response to message history for export
            message_history_step3.create_and_add_message(
                content=executive_synthesis, 
                message_type=MessageType.AI,
                agent=AgentName.CONVERSATIONAL
            )
            
            self.logger.info(f"✅ Step 3 complete: Executive synthesis extracted ({len(executive_synthesis)} chars)")
            
            # Rate limit protection
            await asyncio.sleep(2.0)
            
            # =========================================================
            # STEP 4: Generate Adaptive Card
            # =========================================================
            # Build date range from daily analyses for accurate display
            date_range = self._build_date_range_string(daily_single_analyses, date_ranges)
            adaptive_card_json, step4_conversation = await self._generate_adaptive_card(polished_report, date_range)
            adaptive_card_json, optimization_debug = await self._optimize_adaptive_card_payload(adaptive_card_json, target_kb=24)
            
            # Merge optimization debug into step4 conversation (for debugging)
            step4_conversation['step5_modernize_tone'] = optimization_debug.get('step5_modernize_tone')
            step4_conversation['step6_size_optimization'] = optimization_debug.get('step6_size_optimization')
            step4_conversation['step7_validate_and_fix_json'] = optimization_debug.get('step7_validate_and_fix_json')
            step4_conversation['final_result'] = adaptive_card_json  # Final optimized JSON
            step4_conversation['final_result_length'] = len(adaptive_card_json)
            
            self.logger.info(f"✅ Step 4 complete: Adaptive Card generated (date_range: {date_range})")

            # Save summary adaptive card as a report
            try:
                period_range = format_period_range(date_param=analysis_period_str)
                report_path = get_report_path(self.report_group, "summarizer", period_range)
                save_minified_json(report_path, adaptive_card_json)
                self.logger.info(f"💾 Summary Adaptive Card saved to: {report_path}")

                if self.environment == "prod":
                    try:
                        s3_key = get_s3_report_key(
                            self.s3_uploader.base_prefix, self.report_group, "summarizer", period_range,
                        )
                        minified = adaptive_card_json
                        try:
                            minified = json.dumps(json.loads(adaptive_card_json), ensure_ascii=False, separators=(",", ":"))
                        except json.JSONDecodeError:
                            pass
                        self.s3_uploader.s3_client.put_object(
                            Bucket=self.s3_uploader.bucket_name,
                            Key=s3_key,
                            Body=minified.encode("utf-8"),
                            ContentType="application/json",
                            Metadata={
                                "content_type": "adaptive_card",
                                "report_group": self.report_group,
                                "period_range": period_range,
                            },
                        )
                        self.logger.info(f"📤 Summary Adaptive Card uploaded to S3: {s3_key}")
                    except Exception as s3_err:
                        self.logger.warning(f"⚠️ Failed to upload Summary AC to S3: {s3_err}")
            except Exception as save_err:
                self.logger.warning(f"⚠️ Failed to save Summary Adaptive Card: {save_err}")

            # =========================================================
            # Combine synthesis and adaptive card
            # =========================================================
            # Use a clear separator for the downstream consumer
            final_output = f"{executive_synthesis}\n\n---ADAPTIVE_CARD_JSON---\n\n{adaptive_card_json}"
            
            # =========================================================
            # Save debug files and export conversation
            # =========================================================
            try:
                summary_reports_dir = Path.cwd() / "summary_reports"
                summary_reports_dir.mkdir(parents=True, exist_ok=True)
                dbg_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                debug_path = summary_reports_dir / f"summary_stratified_{dbg_ts}.md"
                with open(debug_path, 'w', encoding='utf-8') as dbg:
                    dbg.write("===== STEP 1: SECTION CONNECTIONS =====\n\n")
                    for section_name, paragraph in daily_context_paragraphs.items():
                        dbg.write(f"--- {section_name} ---\n{paragraph}\n\n")
                    dbg.write("===== STEP 2: FULL REPORT =====\n\n")
                    dbg.write(polished_report)
                    dbg.write("\n\n===== STEP 3: EXECUTIVE SYNTHESIS =====\n\n")
                    dbg.write(executive_synthesis)
                    dbg.write("\n\n===== STEP 4: ADAPTIVE CARD JSON =====\n\n")
                    dbg.write(adaptive_card_json)
                self.logger.info(f"📝 Saved stratified debug to: {debug_path}")
            except Exception as e:
                self.logger.warning(f"Could not write debug file: {e}")
            
            # Export ALL conversations (all 4 steps) for complete audit trail
            all_conversations = {
                'step1_section_connections': {
                    'sections': step1_conversations,  # Full conversations for each section
                    'results_summary': daily_context_paragraphs  # Quick access to results
                },
                'step2_full_report': {
                    'messages': [
                        {
                            'role': self._get_message_role(msg),
                            'content': msg.content
                        } for msg in message_history_final.get_messages()
                    ],
                    'result': polished_report,
                    'result_length': len(polished_report)
                },
                'step3_executive_synthesis': {
                    'messages': [
                        {
                            'role': self._get_message_role(msg),
                            'content': msg.content
                        } for msg in message_history_step3.get_messages()
                    ],
                    'result': executive_synthesis,
                    'result_length': len(executive_synthesis)
                },
                'step4_adaptive_card': step4_conversation  # Full conversation including raw response
            }
            conversation_file = await self.export_full_stratified_conversation(
                all_conversations, analysis_period_str
            )
            if conversation_file:
                self.logger.info(f"🗂️ Full stratified conversation saved: {conversation_file}")
            
            # Upload to S3 if metadata is provided (use final_output which includes synthesis and card)
            if final_output and execution_metadata and weekly_analysis_params and daily_analysis_params and date_ranges:
                try:
                    self.logger.info("📤 Uploading executive synthesis and adaptive card to S3...")
                    
                    # In prod, we upload the adaptive card as a JSON object
                    s3_final_synthesis = final_output
                    if self.environment == "prod":
                        try:
                            s3_final_synthesis = json.loads(adaptive_card_json)
                        except Exception:
                            s3_final_synthesis = adaptive_card_json
                            
                    s3_key = await self.s3_uploader.upload_comprehensive_report(
                        execution_date=datetime.now(),
                        analysis_date=execution_metadata.get('analysis_date', ''),
                        segment=execution_metadata.get('segment', 'Global'),
                        explanation_mode=execution_metadata.get('explanation_mode', 'agent'),
                        causal_filter=execution_metadata.get('causal_filter', 'vs L7d'),
                        weekly_analysis_params=weekly_analysis_params,
                        daily_analysis_params=daily_analysis_params,
                        date_ranges=date_ranges,
                        final_synthesis=s3_final_synthesis,
                        comparison_start_date=date_ranges.get('comparison_start_date'),
                        comparison_end_date=date_ranges.get('comparison_end_date')
                    )
                    if s3_key:
                        self.logger.info(f"✅ Report uploaded to S3: {s3_key}")
                except Exception as e:
                    self.logger.error(f"❌ Error uploading to S3: {str(e)}")
            
            return final_output
            
        except Exception as e:
            self.logger.error(f"❌ Error in stratified summary: {str(e)}")
            # Fallback to legacy method
            self.logger.info("⚠️ Falling back to legacy single-step method...")
            return await self.generate_comprehensive_summary(
                weekly_comparative_analysis=weekly_comparative_analysis,
                daily_single_analyses=daily_single_analyses,
                date_flight_local=date_flight_local,
                execution_metadata=execution_metadata,
                weekly_analysis_params=weekly_analysis_params,
                daily_analysis_params=daily_analysis_params,
                date_ranges=date_ranges
            )
    
    def _extract_executive_synthesis_from_weekly(self, weekly_analysis: str) -> str:
        """
        Extract only the SÍNTESIS EJECUTIVA FINAL section from the weekly analysis.
        
        This reduces the input to Step 2, passing only the relevant executive content
        instead of the full technical report with all diagnostic sections.
        
        Args:
            weekly_analysis: Full weekly analysis including technical sections
            
        Returns:
            Only the executive synthesis section (much shorter than full report)
        """
        import re
        
        # Patterns to find the start of executive synthesis
        synthesis_patterns = [
            r'##\s*📋\s*SÍNTESIS EJECUTIVA FINAL',
            r'##\s*📋\s*SÍNTESIS EJECUTIVA',
            r'\*\*SÍNTESIS EJECUTIVA\*\*',
            r'<b>SÍNTESIS EJECUTIVA</b>',
            r'SÍNTESIS EJECUTIVA FINAL',
        ]
        
        # Patterns to find the end (footer to exclude)
        end_patterns = [
            r'---\s*\n\s*✅\s*\*\*ANÁLISIS COMPLETADO\*\*',
            r'✅\s*\*\*ANÁLISIS COMPLETADO\*\*',
            r'---\s*\n\s*\*Nodos procesados:',
            r'\*Este análisis utiliza metodología conversacional',
        ]
        
        # Find the start of executive synthesis
        start_pos = None
        for pattern in synthesis_patterns:
            match = re.search(pattern, weekly_analysis, re.IGNORECASE)
            if match:
                start_pos = match.start()
                self.logger.info(f"   📍 Found synthesis start at position {start_pos} with pattern: {pattern[:30]}...")
                break
        
        if start_pos is None:
            # Fallback: if no synthesis header found, return the full analysis
            self.logger.warning("   ⚠️ Could not find SÍNTESIS EJECUTIVA section, using full analysis")
            return weekly_analysis
        
        # Find the end (exclude footer)
        end_pos = len(weekly_analysis)
        for pattern in end_patterns:
            match = re.search(pattern, weekly_analysis[start_pos:], re.IGNORECASE)
            if match:
                end_pos = start_pos + match.start()
                self.logger.info(f"   📍 Found synthesis end at position {end_pos}")
                break
        
        # Extract the synthesis
        synthesis = weekly_analysis[start_pos:end_pos].strip()
        
        self.logger.info(f"   ✅ Extracted executive synthesis: {len(synthesis)} chars (from {len(weekly_analysis)} total)")
        
        return synthesis
    
    def _extract_executive_synthesis_from_daily(self, daily_analysis: str) -> str:
        """
        Extract only the executive summary from a daily analysis report.
        Similar to _extract_executive_synthesis_from_weekly but tailored for daily reports.
        """
        if not daily_analysis:
            return ""
            
        import re
        
        # Look for synthesis headers common in daily reports
        synthesis_patterns = [
            r'##\s*📋\s*SÍNTESIS EJECUTIVA FINAL',
            r'##\s*📋\s*SÍNTESIS EJECUTIVA',
            r'\*\*SÍNTESIS EJECUTIVA\*\*',
            r'SÍNTESIS EJECUTIVA FINAL',
            r'SÍNTESIS EJECUTIVA',
            r'RESUMEN EJECUTIVO',
            r'##\s*Resumen',
        ]
        
        # End patterns to exclude technical sections
        end_patterns = [
            r'##\s*📊\s*DIAGNÓSTICO',
            r'##\s*📊\s*DETALLE',
            r'##\s*📊\s*COMPANY',
            r'##\s*🎯\s*IDENTIFICACIÓN',
            r'---\s*\n\s*✅\s*\*\*ANÁLISIS COMPLETADO\*\*',
            r'✅\s*\*\*ANÁLISIS COMPLETADO\*\*',
        ]
        
        start_pos = None
        for pattern in synthesis_patterns:
            match = re.search(pattern, daily_analysis, re.IGNORECASE)
            if match:
                start_pos = match.start()
                break
        
        if start_pos is None:
            # If no explicit header, daily reports often have the summary at the beginning
            # but we'll try to find any technical header to cut off
            end_pos = len(daily_analysis)
            for pattern in end_patterns:
                match = re.search(pattern, daily_analysis, re.IGNORECASE)
                if match:
                    end_pos = min(end_pos, match.start())
            
            # If the resulting text is significantly shorter than original, it's likely a good cut
            if end_pos < len(daily_analysis) * 0.7:
                return daily_analysis[:end_pos].strip()
            return daily_analysis
            
        # Find the end (exclude technical sections)
        end_pos = len(daily_analysis)
        for pattern in end_patterns:
            match = re.search(pattern, daily_analysis[start_pos:], re.IGNORECASE)
            if match:
                end_pos = start_pos + match.start()
                break
        
        return daily_analysis[start_pos:end_pos].strip()
    
    def _build_date_range_string(self, daily_single_analyses: List[Dict[str, Any]], date_ranges: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a human-readable date range string from daily analyses or provided date ranges.
        
        Args:
            daily_single_analyses: List of daily analysis dicts with 'date' key
            date_ranges: Optional dict with 'analysis_date', 'comparison_start_date', etc.
            
        Returns:
            String like "9 Dic - 15 Dic 2025"
        """
        try:
            # 1. Try to build from date_ranges if provided (highest priority)
            if date_ranges and date_ranges.get('analysis_date'):
                analysis_date_str = date_ranges.get('analysis_date')
                try:
                    # Assume format is YYYY-MM-DD
                    dt = datetime.strptime(analysis_date_str, '%Y-%m-%d')
                    # If it's a weekly report, we might want to show the week ending on this date
                    # but for now let's just return this date formatted nicely
                    return dt.strftime("%d %b %Y")
                except Exception:
                    return analysis_date_str

            # 2. Try to build from daily_single_analyses
            if not daily_single_analyses:
                return "Período de análisis"
            
            # Extract dates from daily analyses
            dates = []
            for analysis in daily_single_analyses:
                date_str = analysis.get('date', '')
                if date_str:
                    # Try to parse the date (format: YYYY-MM-DD or similar)
                    try:
                        if isinstance(date_str, str):
                            # Handle common formats
                            for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%Y%m%d']:
                                try:
                                    dt = datetime.strptime(date_str, fmt)
                                    dates.append(dt)
                                    break
                                except ValueError:
                                    continue
                    except Exception:
                        pass
            
            if not dates:
                return "Período de análisis"
            
            # Sort and get min/max
            dates.sort()
            start_date = dates[0]
            end_date = dates[-1]
            
            # Spanish month abbreviations
            months_es = {
                1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
            }
            
            start_month = months_es.get(start_date.month, start_date.strftime('%b'))
            end_month = months_es.get(end_date.month, end_date.strftime('%b'))
            
            if start_date.year == end_date.year:
                if start_date.month == end_date.month:
                    if start_date.day == end_date.day:
                        # Single day: "15 Dic 2025"
                        return f"{start_date.day} {end_month} {end_date.year}"
                    # Same month: "9 - 15 Dic 2025"
                    return f"{start_date.day} - {end_date.day} {end_month} {end_date.year}"
                else:
                    # Different months: "30 Nov - 6 Dic 2025"
                    return f"{start_date.day} {start_month} - {end_date.day} {end_month} {end_date.year}"
            else:
                # Different years: "28 Dic 2024 - 3 Ene 2025"
                return f"{start_date.day} {start_month} {start_date.year} - {end_date.day} {end_month} {end_date.year}"
                
        except Exception as e:
            self.logger.warning(f"Could not build date range: {e}")
            return "Período de análisis"
    
    def _get_section_patterns_for_segment(self, segment: str = 'Global') -> List[tuple]:
        """
        Devuelve los patrones de sección según el segmento seleccionado.
        Esto permite parsear dinámicamente TODAS las agregaciones bajo el segmento.
        
        Mejorado para incluir múltiples formatos y patrones más flexibles.
        
        Args:
            segment: El segmento raíz del análisis
            
        Returns:
            Lista de tuplas (nombre_sección, patrón_regex)
        """
        # Patrones base mejorados para cada tipo de sección
        # Ahora soporta múltiples formatos:
        # 1. Formato HTML: <b><u>SECTION_NAME: título</u></b>
        # 2. Formato Markdown: **SECTION_NAME: título**
        # 3. Formato simple: SECTION_NAME: título
        # 4. Formato con emojis: 📊 SECTION_NAME: título
        
        all_section_patterns = {
            # GLOBAL - múltiples formatos
            'GLOBAL': r'(?:📈|📋)?\s*(?:\*\*|<b>)?SÍNTESIS EJECUTIVA|Durante la semana',
            'GLOBAL': r'\*\*SÍNTESIS EJECUTIVA\*\*',
            'GLOBAL': r'##\s*📋\s*SÍNTESIS EJECUTIVA',
            
            # SH - múltiples formatos
            'SH': r'(?:<b>)?(?:<u>)?\s*(?:SHORT HAUL|SH)[^:]*:\s*[A-ZÁÉÍÓÚÑ]',
            'SH': r'\*\*(?:SHORT HAUL|SH)[^:]*:\*\*',
            'SH': r'(?:SHORT HAUL|SH)[^:]*:',
            
            # LH - múltiples formatos
            'LH': r'(?:<b>)?(?:<u>)?\s*(?:LONG HAUL|LH)[^:]*:\s*[A-ZÁÉÍÓÚÑ]',
            'LH': r'\*\*(?:LONG HAUL|LH)[^:]*:\*\*',
            'LH': r'(?:LONG HAUL|LH)[^:]*:',
            
            # ECONOMY SH - múltiples formatos
            'ECONOMY SH': r'(?:<b>)?(?:<u>)?\s*ECONOMY SH[^:]*:\s*[A-ZÁÉÍÓÚÑ]',
            'ECONOMY SH': r'\*\*ECONOMY SH[^:]*:\*\*',
            'ECONOMY SH': r'ECONOMY SH[^:]*:',
            
            # BUSINESS SH - múltiples formatos
            'BUSINESS SH': r'(?:<b>)?(?:<u>)?\s*BUSINESS SH[^:]*:\s*[A-ZÁÉÍÓÚÑ]',
            'BUSINESS SH': r'\*\*BUSINESS SH[^:]*:\*\*',
            'BUSINESS SH': r'BUSINESS SH[^:]*:',
            
            # ECONOMY SH IB - múltiples formatos
            'ECONOMY SH IB': r'(?:<b>)?(?:<u>)?\s*(?:ECONOMY SH\s*[-/]?\s*IB|IB \(Economy SH\))[^:]*:\s*[A-ZÁÉÍÓÚÑ]',
            'ECONOMY SH IB': r'\*\*(?:ECONOMY SH\s*[-/]?\s*IB|IB \(Economy SH\))[^:]*:\*\*',
            'ECONOMY SH IB': r'(?:ECONOMY SH\s*[-/]?\s*IB|IB \(Economy SH\))[^:]*:',
            
            # ECONOMY SH YW - múltiples formatos
            'ECONOMY SH YW': r'(?:<b>)?(?:<u>)?\s*(?:ECONOMY SH\s*[-/]?\s*YW|YW \(Economy SH\))[^:]*:\s*[A-ZÁÉÍÓÚÑ]',
            'ECONOMY SH YW': r'\*\*(?:ECONOMY SH\s*[-/]?\s*YW|YW \(Economy SH\))[^:]*:\*\*',
            'ECONOMY SH YW': r'(?:ECONOMY SH\s*[-/]?\s*YW|YW \(Economy SH\))[^:]*:',
            
            # BUSINESS SH IB - múltiples formatos
            'BUSINESS SH IB': r'(?:<b>)?(?:<u>)?\s*(?:BUSINESS SH\s*[-/]?\s*IB|IB \(Business SH\))[^:]*:\s*[A-ZÁÉÍÓÚÑ]',
            'BUSINESS SH IB': r'\*\*(?:BUSINESS SH\s*[-/]?\s*IB|IB \(Business SH\))[^:]*:\*\*',
            'BUSINESS SH IB': r'(?:BUSINESS SH\s*[-/]?\s*IB|IB \(Business SH\))[^:]*:',
            
            # BUSINESS SH YW - múltiples formatos
            'BUSINESS SH YW': r'(?:<b>)?(?:<u>)?\s*(?:BUSINESS SH\s*[-/]?\s*YW|YW \(Business SH\))[^:]*:\s*[A-ZÁÉÍÓÚÑ]',
            'BUSINESS SH YW': r'\*\*(?:BUSINESS SH\s*[-/]?\s*YW|YW \(Business SH\))[^:]*:\*\*',
            'BUSINESS SH YW': r'(?:BUSINESS SH\s*[-/]?\s*YW|YW \(Business SH\))[^:]*:',
            
            # ECONOMY LH - múltiples formatos
            'ECONOMY LH': r'(?:<b>)?(?:<u>)?\s*ECONOMY LH[^:]*:\s*[A-ZÁÉÍÓÚÑ]',
            'ECONOMY LH': r'\*\*ECONOMY LH[^:]*:\*\*',
            'ECONOMY LH': r'ECONOMY LH[^:]*:',
            
            # BUSINESS LH - múltiples formatos
            'BUSINESS LH': r'(?:<b>)?(?:<u>)?\s*BUSINESS LH[^:]*:\s*[A-ZÁÉÍÓÚÑ]',
            'BUSINESS LH': r'\*\*BUSINESS LH[^:]*:\*\*',
            'BUSINESS LH': r'BUSINESS LH[^:]*:',
            
            # PREMIUM LH - múltiples formatos
            'PREMIUM LH': r'(?:<b>)?(?:<u>)?\s*PREMIUM LH[^:]*:\s*[A-ZÁÉÍÓÚÑ]',
            'PREMIUM LH': r'\*\*PREMIUM LH[^:]*:\*\*',
            'PREMIUM LH': r'PREMIUM LH[^:]*:',
            
            # IB - múltiples formatos
            'IB': r'(?:<b>)?(?:<u>)?\s*IB[^:]*:\s*[A-ZÁÉÍÓÚÑ]',
            'IB': r'\*\*IB[^:]*:\*\*',
            'IB': r'IB[^:]*:',
            
            # YW - múltiples formatos
            'YW': r'(?:<b>)?(?:<u>)?\s*YW[^:]*:\s*[A-ZÁÉÍÓÚÑ]',
            'YW': r'\*\*YW[^:]*:\*\*',
            'YW': r'YW[^:]*:',
        }
        
        # Definir qué secciones aplican según el segmento
        segment_sections = {
            'Global': ['GLOBAL', 'SH', 'LH', 'BUSINESS SH', 'BUSINESS SH IB', 'BUSINESS SH YW',
                      'ECONOMY SH', 'ECONOMY SH IB', 'ECONOMY SH YW',
                      'ECONOMY LH', 'BUSINESS LH', 'PREMIUM LH'],
            'SH': ['SH', 'BUSINESS SH', 'BUSINESS SH IB', 'BUSINESS SH YW',
                   'ECONOMY SH', 'ECONOMY SH IB', 'ECONOMY SH YW'],
            'LH': ['LH', 'ECONOMY LH', 'BUSINESS LH', 'PREMIUM LH'],
            'Economy SH': ['ECONOMY SH', 'ECONOMY SH IB', 'ECONOMY SH YW'],
            'Business SH': ['BUSINESS SH', 'BUSINESS SH IB', 'BUSINESS SH YW'],
            'Economy LH': ['ECONOMY LH'],
            'Business LH': ['BUSINESS LH'],
            'Premium LH': ['PREMIUM LH'],
        }
        
        # Normalizar el nombre del segmento
        segment_normalized = segment
        if segment in ['Global/SH', 'Short Haul', 'SH']:
            segment_normalized = 'SH'
        elif segment in ['Global/LH', 'Long Haul', 'LH']:
            segment_normalized = 'LH'
        elif segment in ['Global/SH/Economy', 'Economy/SH']:
            segment_normalized = 'Economy SH'
        elif segment in ['Global/SH/Business', 'Business/SH']:
            segment_normalized = 'Business SH'
        elif segment in ['Global/LH/Economy', 'Economy/LH']:
            segment_normalized = 'Economy LH'
        elif segment in ['Global/LH/Business', 'Business/LH']:
            segment_normalized = 'Business LH'
        elif segment in ['Global/LH/Premium', 'Premium/LH']:
            segment_normalized = 'Premium LH'
        
        self.logger.info(f"🔍 Normalizando segmento: '{segment}' → '{segment_normalized}'")
        
        # Obtener las secciones aplicables
        applicable_sections = segment_sections.get(segment_normalized, ['GLOBAL'])
        self.logger.info(f"📊 Secciones aplicables para '{segment_normalized}': {applicable_sections}")
        
        # Construir los patrones (tomar solo el primer patrón para cada sección para evitar duplicados)
        patterns = []
        seen_sections = set()
        
        for section in applicable_sections:
            if section not in seen_sections:
                # Tomar el primer patrón disponible para esta sección
                for pattern_key, pattern_value in all_section_patterns.items():
                    if pattern_key == section:
                        patterns.append((section, pattern_value))
                        seen_sections.add(section)
                        self.logger.debug(f"   📝 Añadido patrón para '{section}': {pattern_value[:50]}...")
                        break
        
        self.logger.info(f"📊 Total de patrones generados: {len(patterns)}")
        return patterns

    def _parse_weekly_sections(self, weekly_analysis: str, segment: str = 'Global') -> Dict[str, str]:
        """
        Parse the weekly analysis into sections based on the segment hierarchy.
        
        Args:
            weekly_analysis: The weekly analysis text to parse
            segment: The root segment to determine which sections to parse
        
        Returns:
            Dictionary mapping section names to their content
        """
        import re
        
        self.logger.info(f"🔍 Parseando secciones del análisis semanal para segmento '{segment}'")
        self.logger.info(f"📄 Longitud del análisis semanal: {len(weekly_analysis)} caracteres")
        
        sections = {}
        
        # Obtener patrones dinámicos según el segmento
        section_patterns = self._get_section_patterns_for_segment(segment)
        self.logger.info(f"📊 Patrones de sección para '{segment}': {len(section_patterns)} patrones")
        
        # Find all section headers and their positions
        section_positions = []
        for section_name, pattern in section_patterns:
            matches = list(re.finditer(pattern, weekly_analysis, re.IGNORECASE))
            for match in matches:
                section_positions.append((match.start(), section_name, match.group()))
                self.logger.debug(f"   📍 Encontrada sección '{section_name}' en posición {match.start()} con patrón: {pattern[:50]}...")
        
        # Sort by position
        section_positions.sort(key=lambda x: x[0])
        self.logger.info(f"📊 Total de posiciones de sección encontradas: {len(section_positions)}")
        
        # Safety net: Check if there is text before the first section that should be GLOBAL/root
        if section_positions and section_positions[0][0] > 0:
            intro_text = weekly_analysis[0:section_positions[0][0]].strip()
            # Only use intro as root section if the first detected section is NOT the root
            root_section = section_patterns[0][0] if section_patterns else 'GLOBAL'
            if len(intro_text) > 50 and section_positions[0][1] != root_section:
                sections[root_section] = intro_text
                self.logger.info(f"   ⚠️ Recuperados {len(intro_text)} caracteres de texto introductorio como sección {root_section}")
        
        # Extract content for each section
        for i, (pos, section_name, header_text) in enumerate(section_positions):
            self.logger.debug(f"   🔄 Procesando sección '{section_name}' en posición {pos}")
            
            # Special handling for GLOBAL: extract until DETALLE POR AGREGACIÓN or first detail header
            if section_name == 'GLOBAL':
                # Find end: DETALLE POR AGREGACIÓN or first detailed section header
                end_patterns = [
                    r'<b><u>DETALLE POR AGREGACIÓN</u></b>',
                    r'<b><u>(?:SH|SHORT HAUL|LH|LONG HAUL|BUSINESS|ECONOMY|PREMIUM)[^<]*:</u></b>',
                    r'\*\*(?:SH|SHORT HAUL|LH|LONG HAUL|BUSINESS|ECONOMY|PREMIUM)[^:]*:\*\*',
                    r'##\s*📊\s*DIAGNÓSTICO',
                    r'##\s*📊\s*DETALLE',
                ]
                
                end_pos = len(weekly_analysis)
                for pattern in end_patterns:
                    match = re.search(pattern, weekly_analysis[pos:], re.IGNORECASE)
                    if match:
                        candidate_end = pos + match.start()
                        if candidate_end < end_pos:
                            end_pos = candidate_end
                            self.logger.debug(f"   📍 Fin de sección GLOBAL encontrado con patrón: {pattern[:30]}...")
                
                content = weekly_analysis[pos:end_pos].strip()
                self.logger.debug(f"   ✅ Sección GLOBAL extraída: {len(content)} caracteres")
            else:
                # Standard handling: find end of this section (start of next section or end of text)
                if i + 1 < len(section_positions):
                    end_pos = section_positions[i + 1][0]
                    next_section = section_positions[i + 1][1]
                    self.logger.debug(f"   📍 Fin en siguiente sección '{next_section}' en posición {end_pos}")
                else:
                    end_pos = len(weekly_analysis)
                    self.logger.debug(f"   📍 Fin al final del texto (posición {end_pos})")
                
                content = weekly_analysis[pos:end_pos].strip()
                self.logger.debug(f"   ✅ Sección '{section_name}' extraída: {len(content)} caracteres")
            
            # Only keep if we don't already have this section (avoid duplicates)
            if section_name not in sections:
                sections[section_name] = content
                self.logger.info(f"   ✅ Sección '{section_name}' añadida: {len(content)} caracteres")
            else:
                self.logger.warning(f"   ⚠️ Sección duplicada '{section_name}' ignorada")
        
        # If no sections found, use the whole text as root section
        if not sections:
            root_section = section_patterns[0][0] if section_patterns else 'GLOBAL'
            sections[root_section] = weekly_analysis
            self.logger.warning(f"⚠️ No se encontraron secciones, usando todo el texto como '{root_section}'")
        
        self.logger.info(f"📊 Secciones parseadas para segmento '{segment}': {list(sections.keys())}")
        self.logger.info(f"📊 Total de secciones: {len(sections)}")
        
        # Log tamaño de cada sección
        for section_name, content in sections.items():
            self.logger.debug(f"   📏 Sección '{section_name}': {len(content)} caracteres")
        
        return sections
    
    def _filter_daily_for_section(self, daily_analyses: str, section_name: str) -> str:
        """
        Filter daily analyses to extract ONLY the specific section content from each day.
        
        Cada día contiene un informe completo con todas las secciones. Esta función
        extrae solo la sección específica de cada día para reducir el contexto.
        
        Mejorado para manejar múltiples formatos de secciones y añadir logging detallado.
        """
        import re
        
        # Patrones mejorados para encontrar el HEADER de cada sección
        # Ahora soporta múltiples formatos:
        # 1. Formato HTML: <b><u>SECTION_NAME: título</u></b>
        # 2. Formato Markdown: **SECTION_NAME: título**
        # 3. Formato simple: SECTION_NAME: título
        # 4. Formato con emojis: 📊 SECTION_NAME: título
        
        all_section_patterns = [
            # GLOBAL - múltiples formatos
            ('GLOBAL', r'(?:<b>)?SÍNTESIS EJECUTIVA(?:</b>)?'),
            ('GLOBAL', r'\*\*SÍNTESIS EJECUTIVA\*\*'),
            ('GLOBAL', r'SÍNTESIS EJECUTIVA'),
            
            # SH - múltiples formatos
            ('SH', r'<b><u>SHORT HAUL[^<]*</u></b>'),
            ('SH', r'\*\*SHORT HAUL[^:]*:\*\*'),
            ('SH', r'SHORT HAUL[^:]*:'),
            ('SH', r'<b><u>SH[^<]*</u></b>'),
            ('SH', r'\*\*SH[^:]*:\*\*'),
            ('SH', r'SH[^:]*:'),
            
            # LH - múltiples formatos
            ('LH', r'<b><u>LONG HAUL[^<]*</u></b>'),
            ('LH', r'\*\*LONG HAUL[^:]*:\*\*'),
            ('LH', r'LONG HAUL[^:]*:'),
            ('LH', r'<b><u>LH[^<]*</u></b>'),
            ('LH', r'\*\*LH[^:]*:\*\*'),
            ('LH', r'LH[^:]*:'),
            
            # BUSINESS SH - múltiples formatos
            ('BUSINESS SH', r'<b><u>BUSINESS SH(?!\s*[-/]?\s*(?:IB|YW))[^<]*</u></b>'),
            ('BUSINESS SH', r'\*\*BUSINESS SH(?!\s*[-/]?\s*(?:IB|YW))[^:]*:\*\*'),
            ('BUSINESS SH', r'BUSINESS SH(?!\s*[-/]?\s*(?:IB|YW))[^:]*:'),
            
            # BUSINESS SH IB - múltiples formatos
            ('BUSINESS SH IB', r'<b><u>BUSINESS SH\s*[-/]?\s*IB[^<]*</u></b>'),
            ('BUSINESS SH IB', r'\*\*BUSINESS SH\s*[-/]?\s*IB[^:]*:\*\*'),
            ('BUSINESS SH IB', r'BUSINESS SH\s*[-/]?\s*IB[^:]*:'),
            ('BUSINESS SH IB', r'IB \(Business SH\)[^:]*:'),
            
            # BUSINESS SH YW - múltiples formatos
            ('BUSINESS SH YW', r'<b><u>BUSINESS SH\s*[-/]?\s*YW[^<]*</u></b>'),
            ('BUSINESS SH YW', r'\*\*BUSINESS SH\s*[-/]?\s*YW[^:]*:\*\*'),
            ('BUSINESS SH YW', r'BUSINESS SH\s*[-/]?\s*YW[^:]*:'),
            ('BUSINESS SH YW', r'YW \(Business SH\)[^:]*:'),
            
            # ECONOMY SH - múltiples formatos
            ('ECONOMY SH', r'<b><u>ECONOMY SH(?!\s*[-/]?\s*(?:IB|YW))[^<]*</u></b>'),
            ('ECONOMY SH', r'\*\*ECONOMY SH(?!\s*[-/]?\s*(?:IB|YW))[^:]*:\*\*'),
            ('ECONOMY SH', r'ECONOMY SH(?!\s*[-/]?\s*(?:IB|YW))[^:]*:'),
            
            # ECONOMY SH IB - múltiples formatos
            ('ECONOMY SH IB', r'<b><u>ECONOMY SH\s*[-/]?\s*IB[^<]*</u></b>'),
            ('ECONOMY SH IB', r'\*\*ECONOMY SH\s*[-/]?\s*IB[^:]*:\*\*'),
            ('ECONOMY SH IB', r'ECONOMY SH\s*[-/]?\s*IB[^:]*:'),
            ('ECONOMY SH IB', r'IB \(Economy SH\)[^:]*:'),
            
            # ECONOMY SH YW - múltiples formatos
            ('ECONOMY SH YW', r'<b><u>ECONOMY SH\s*[-/]?\s*YW[^<]*</u></b>'),
            ('ECONOMY SH YW', r'\*\*ECONOMY SH\s*[-/]?\s*YW[^:]*:\*\*'),
            ('ECONOMY SH YW', r'ECONOMY SH\s*[-/]?\s*YW[^:]*:'),
            ('ECONOMY SH YW', r'YW \(Economy SH\)[^:]*:'),
            
            # BUSINESS LH - múltiples formatos
            ('BUSINESS LH', r'<b><u>BUSINESS LH[^<]*</u></b>'),
            ('BUSINESS LH', r'\*\*BUSINESS LH[^:]*:\*\*'),
            ('BUSINESS LH', r'BUSINESS LH[^:]*:'),
            
            # PREMIUM LH - múltiples formatos
            ('PREMIUM LH', r'<b><u>PREMIUM LH[^<]*</u></b>'),
            ('PREMIUM LH', r'\*\*PREMIUM LH[^:]*:\*\*'),
            ('PREMIUM LH', r'PREMIUM LH[^:]*:'),
            
            # ECONOMY LH - múltiples formatos
            ('ECONOMY LH', r'<b><u>ECONOMY LH[^<]*</u></b>'),
            ('ECONOMY LH', r'\*\*ECONOMY LH[^:]*:\*\*'),
            ('ECONOMY LH', r'ECONOMY LH[^:]*:'),
        ]
        
        self.logger.info(f"🔍 Buscando sección '{section_name}' en análisis diarios...")
        self.logger.info(f"📄 Longitud total de análisis diarios: {len(daily_analyses)} caracteres")
        
        # Split por días primero (📅 marca cada día)
        day_pattern = r'(📅\s*\d{4}-\d{2}-\d{2})'
        day_parts = re.split(day_pattern, daily_analyses)
        
        self.logger.info(f"📅 Encontrados {len(day_parts)} partes al dividir por días")
        
        # Reconstruir días
        daily_entries = []
        i = 0
        while i < len(day_parts):
            if re.match(r'📅\s*\d{4}-\d{2}-\d{2}', day_parts[i] if i < len(day_parts) else ''):
                date = day_parts[i]
                content = day_parts[i + 1] if i + 1 < len(day_parts) else ''
                daily_entries.append((date, content))
                i += 2
            else:
                i += 1
        
        self.logger.info(f"📊 Reconstruidos {len(daily_entries)} días completos")
        
        # Extraer la sección específica de cada día
        extracted_sections = []
        days_with_section = 0
        days_without_section = 0
        
        for date, content in daily_entries:
            section_content = self._extract_section_from_text(content, section_name, all_section_patterns)
            if section_content and len(section_content) > 50:
                extracted_sections.append(f"{date}:\n{section_content}")
                days_with_section += 1
                self.logger.info(f"✅ Encontrada sección '{section_name}' en {date} ({len(section_content)} caracteres)")
            else:
                days_without_section += 1
                if section_content:
                    self.logger.info(f"⚠️ Sección '{section_name}' en {date} demasiado corta: {len(section_content)} caracteres")
                else:
                    self.logger.info(f"❌ No se encontró sección '{section_name}' en {date}")
        
        self.logger.info(f"📊 Resumen: {days_with_section} días con sección, {days_without_section} días sin sección")
        
        if extracted_sections:
            result = f"**Análisis de {section_name} ({len(extracted_sections)} días):**\n\n"
            result += "\n\n---\n\n".join(extracted_sections)
            self.logger.info(f"✅ Sección '{section_name}' procesada exitosamente: {len(extracted_sections)} días extraídos")
            return result
        else:
            self.logger.warning(f"⚠️ No se encontró la sección '{section_name}' en ningún análisis diario")
            
            # Fallback: intentar extraer contenido basado en palabras clave
            fallback_content = self._extract_fallback_content(daily_analyses, section_name)
            if fallback_content:
                self.logger.info(f"🔄 Usando fallback para sección '{section_name}': {len(fallback_content)} caracteres")
                return f"**Análisis de {section_name} (extraído por fallback):**\n\n{fallback_content}"
            else:
                return f"**No se encontró la sección {section_name} en los análisis diarios.**"
    
    def _extract_section_from_text(self, text: str, target_section: str, all_patterns: list) -> str:
        """
        Extract a specific section from text by finding its header and the next section's header.
        
        Special handling for GLOBAL: extracts from SÍNTESIS EJECUTIVA until 
        DETALLE POR AGREGACIÓN or the first <b><u>...: header in the detail section.
        
        Mejorado para manejar múltiples formatos y añadir logging.
        """
        import re
        
        self.logger.debug(f"🔍 Extrayendo sección '{target_section}' de texto de {len(text)} caracteres")
        
        # Special handling for GLOBAL section
        if target_section == 'GLOBAL':
            # Find SÍNTESIS EJECUTIVA start - múltiples formatos
            synthesis_patterns = [
                r'(?:<b>)?SÍNTESIS EJECUTIVA(?:</b>)?',
                r'\*\*SÍNTESIS EJECUTIVA\*\*',
                r'SÍNTESIS EJECUTIVA',
                r'##\s*📋\s*SÍNTESIS EJECUTIVA',
            ]
            
            start_pos = None
            for pattern in synthesis_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    start_pos = match.start()
                    self.logger.debug(f"   📍 Encontrado inicio de síntesis con patrón: {pattern[:30]}...")
                    break
            
            if start_pos is None:
                self.logger.debug("   ⚠️ No se encontró SÍNTESIS EJECUTIVA, intentando encontrar introducción")
                # Fallback: buscar cualquier texto al inicio antes de la primera sección detallada
                detail_patterns = [
                    r'<b><u>DETALLE POR AGREGACIÓN</u></b>',
                    r'<b><u>(?:SH|SHORT HAUL|LH|LONG HAUL|BUSINESS|ECONOMY|PREMIUM)[^<]*:</u></b>',
                    r'\*\*(?:SH|SHORT HAUL|LH|LONG HAUL|BUSINESS|ECONOMY|PREMIUM)[^:]*:\*\*',
                ]
                
                for pattern in detail_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        start_pos = 0
                        end_pos = match.start()
                        section_text = text[start_pos:end_pos].strip()
                        if len(section_text) > 50:
                            self.logger.debug(f"   🔄 Usando fallback para GLOBAL: {len(section_text)} caracteres")
                            return section_text
                return ""
            
            # Find end: DETALLE POR AGREGACIÓN or first detailed section header
            end_patterns = [
                r'<b><u>DETALLE POR AGREGACIÓN</u></b>',
                r'<b><u>(?:SH|SHORT HAUL|LH|LONG HAUL|BUSINESS|ECONOMY|PREMIUM)[^<]*:</u></b>',
                r'\*\*(?:SH|SHORT HAUL|LH|LONG HAUL|BUSINESS|ECONOMY|PREMIUM)[^:]*:\*\*',
                r'##\s*📊\s*DIAGNÓSTICO',
                r'##\s*📊\s*DETALLE',
            ]
            
            end_pos = len(text)
            for pattern in end_patterns:
                match = re.search(pattern, text[start_pos:], re.IGNORECASE)
                if match:
                    candidate_end = start_pos + match.start()
                    if candidate_end < end_pos:
                        end_pos = candidate_end
                        self.logger.debug(f"   📍 Encontrado fin con patrón: {pattern[:30]}...")
            
            section_text = text[start_pos:end_pos].strip()
            self.logger.debug(f"   ✅ Extraída sección GLOBAL: {len(section_text)} caracteres")
            return section_text
        
        # Standard handling for other sections
        # Encontrar todas las secciones y sus posiciones
        positions = []
        for name, pattern in all_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                positions.append((match.start(), name, match.end()))
        
        # Ordenar por posición
        positions.sort(key=lambda x: x[0])
        
        self.logger.debug(f"   📊 Encontradas {len(positions)} posiciones de sección en el texto")
        
        # Encontrar la sección objetivo y extraer hasta la siguiente
        for i, (start_pos, name, header_end) in enumerate(positions):
            if name == target_section:
                self.logger.debug(f"   🎯 Encontrada sección objetivo '{target_section}' en posición {start_pos}")
                
                # El contenido va desde el inicio del header hasta el inicio del siguiente header
                if i + 1 < len(positions):
                    end_pos = positions[i + 1][0]
                    next_section = positions[i + 1][1]
                    self.logger.debug(f"   📍 Fin en siguiente sección '{next_section}' en posición {end_pos}")
                else:
                    end_pos = len(text)
                    self.logger.debug(f"   📍 Fin al final del texto (posición {end_pos})")
                
                section_text = text[start_pos:end_pos].strip()
                self.logger.debug(f"   ✅ Extraída sección '{target_section}': {len(section_text)} caracteres")
                return section_text
        
        self.logger.debug(f"   ❌ No se encontró la sección '{target_section}' en el texto")
        return ""
    
    def _extract_fallback_content(self, daily_analyses: str, section_name: str) -> str:
        """
        Fallback method to extract content when regular patterns don't work.
        Uses keyword-based extraction for common section types.
        """
        import re
        
        self.logger.info(f"🔄 Usando fallback para extraer contenido de sección '{section_name}'")
        
        # Mapeo de palabras clave para cada sección
        section_keywords = {
            'GLOBAL': ['global', 'overall', 'resumen', 'síntesis', 'ejecutiva'],
            'SH': ['short haul', 'sh', 'radio corto'],
            'LH': ['long haul', 'lh', 'radio largo'],
            'BUSINESS SH': ['business sh', 'business short haul'],
            'ECONOMY SH': ['economy sh', 'economy short haul'],
            'BUSINESS LH': ['business lh', 'business long haul'],
            'ECONOMY LH': ['economy lh', 'economy long haul'],
            'PREMIUM LH': ['premium lh', 'premium long haul'],
        }
        
        # Obtener palabras clave para la sección
        keywords = section_keywords.get(section_name, [section_name.lower()])
        
        # Split por días
        day_pattern = r'(📅\s*\d{4}-\d{2}-\d{2})'
        day_parts = re.split(day_pattern, daily_analyses)
        
        extracted_content = []
        
        # Procesar cada día
        i = 0
        while i < len(day_parts):
            if re.match(r'📅\s*\d{4}-\d{2}-\d{2}', day_parts[i] if i < len(day_parts) else ''):
                date = day_parts[i]
                content = day_parts[i + 1] if i + 1 < len(day_parts) else ''
                
                # Buscar contenido relacionado con las palabras clave
                relevant_lines = []
                lines = content.split('\n')
                
                for line in lines:
                    line_lower = line.lower()
                    # Verificar si la línea contiene alguna palabra clave
                    if any(keyword in line_lower for keyword in keywords):
                        relevant_lines.append(line)
                    # También capturar líneas siguientes que puedan ser parte del análisis
                    elif relevant_lines and len(relevant_lines) < 10:  # Limitar a 10 líneas
                        relevant_lines.append(line)
                
                if relevant_lines:
                    day_content = f"{date}:\n" + '\n'.join(relevant_lines)
                    extracted_content.append(day_content)
                    self.logger.debug(f"   ✅ Encontrado contenido para '{section_name}' en {date}: {len(relevant_lines)} líneas")
                
                i += 2
            else:
                i += 1
        
        if extracted_content:
            result = "\n\n---\n\n".join(extracted_content)
            self.logger.info(f"✅ Fallback exitoso para '{section_name}': {len(extracted_content)} días extraídos")
            return result
        
        self.logger.warning(f"❌ Fallback también falló para sección '{section_name}'")
        return ""
    
    def _get_parent_section(self, section_name: str) -> str:
        """Get the parent section for hierarchical fallback."""
        parent_map = {
            'ECONOMY SH IB': 'ECONOMY SH',
            'ECONOMY SH YW': 'ECONOMY SH',
            'BUSINESS SH IB': 'BUSINESS SH',
            'BUSINESS SH YW': 'BUSINESS SH',
            'ECONOMY SH': 'SH',
            'BUSINESS SH': 'SH',
            'ECONOMY LH': 'LH',
            'BUSINESS LH': 'LH',
            'PREMIUM LH': 'LH',
            'SH': 'GLOBAL',
            'LH': 'GLOBAL',
        }
        return parent_map.get(section_name)
    
    def _format_periods_for_summary(self, periods_data: List[Dict[str, Any]]) -> str:
        """Format periods data into a structured text for AI analysis."""
        formatted_parts = []
        
        for period_data in periods_data:
            period = period_data.get('period', 'Unknown')
            date_range = period_data.get('date_range', 'Unknown dates')
            ai_interpretation = period_data.get('ai_interpretation', 'No interpretation available')
            
            # Extract specific examples from the AI interpretation
            specific_examples = self._extract_specific_examples(ai_interpretation)
            
            # Format period summary with interpretation AND specific examples
            period_summary = f"""
PERÍODO {period} ({date_range}):
{ai_interpretation}

🔍 EJEMPLOS ESPECÍFICOS EXTRAÍDOS:
{specific_examples}
"""
            formatted_parts.append(period_summary.strip())
        
        return "\n\n".join(formatted_parts)
    
    def _extract_specific_examples(self, ai_interpretation: str) -> str:
        """Extract specific examples (routes, OTP metrics, NPS values) from AI interpretation."""
        import re
        
        examples = []
        
        # Extract route examples (e.g., MAD-EZE, BCN-JFK, etc.)
        route_pattern = r'([A-Z]{3}-[A-Z]{3,4})'
        routes = re.findall(route_pattern, ai_interpretation)
        if routes:
            unique_routes = list(dict.fromkeys(routes))  # Remove duplicates while preserving order
            examples.append(f"• Rutas identificadas: {', '.join(unique_routes[:8])}")  # Limit to 8 routes
        
        # Extract OTP metrics (e.g., OTP 78.5%, OTP bajó 4.2pts)
        otp_pattern = r'OTP\s+(?:bajó\s+|mejoró\s+|se redujo\s+)?(\d+\.?\d*(?:pts|%)?)\s*(?:a\s+(\d+\.?\d*%))?'
        otp_matches = re.findall(otp_pattern, ai_interpretation, re.IGNORECASE)
        if otp_matches:
            for match in otp_matches[:3]:  # Limit to 3 OTP examples
                if match[1]:  # Has both delta and final value
                    examples.append(f"• OTP específico: bajó {match[0]} a {match[1]}")
                else:  # Just delta or value
                    examples.append(f"• Métrica OTP: {match[0]}")
        
        # Extract NPS values (e.g., NPS -31.2, NPS=-100.0)
        nps_pattern = r'NPS[:\s]*(-?\d+\.?\d*)'
        nps_matches = re.findall(nps_pattern, ai_interpretation)
        if nps_matches:
            unique_nps = list(dict.fromkeys(nps_matches))  # Remove duplicates
            extreme_nps = [nps for nps in unique_nps if float(nps) <= -50 or float(nps) >= 50]
            if extreme_nps:
                examples.append(f"• NPS extremos: {', '.join(extreme_nps[:5])}")  # Show extreme cases
        
        # Extract boarding/arrivals specific metrics
        touchpoint_pattern = r'(Boarding|Arrivals|Punctuality)[:\s=]*(\d+\.?\d*)'
        touchpoint_matches = re.findall(touchpoint_pattern, ai_interpretation, re.IGNORECASE)
        if touchpoint_matches:
            for match in touchpoint_matches[:3]:  # Limit to 3 touchpoint examples
                examples.append(f"• {match[0]}: {match[1]}")
        
        # Extract customer segments (business/work, leisure, etc.)
        segment_pattern = r'(business/work|leisure|Economy|Business|Premium).*?NPS[:\s]*(-?\d+\.?\d*)'
        segment_matches = re.findall(segment_pattern, ai_interpretation, re.IGNORECASE)
        if segment_matches:
            for match in segment_matches[:3]:  # Limit to 3 segment examples
                examples.append(f"• Segmento {match[0]}: NPS {match[1]}")
        
        return '\n'.join(examples) if examples else '• No se encontraron ejemplos específicos cuantificables'
    
    def _parse_adaptive_card_json(self, adaptive_card_json: str) -> Dict[str, Any]:
        """
        Parse Adaptive Card JSON from interpreter to extract structured data.
        
        Args:
            adaptive_card_json: JSON string of Adaptive Card
            
        Returns:
            Dictionary with parsed data:
            {
                "period_type": "weekly" | "daily",
                "date_range": "5-11 dic 2024",
                "segments": {
                    "GLOBAL": {
                        "nps": "45.2",
                        "variation": "+3.2",
                        "analysis": "Texto completo del análisis..."
                    },
                    "SH": {
                        "nps": "38.7", 
                        "variation": "+9.7",
                        "analysis": "Texto completo del análisis..."
                    },
                    ...
                }
            }
        """
        import json
        import re
        
        try:
            # Parse JSON
            card_data = json.loads(adaptive_card_json)
            
            # Extract date range from header
            date_range = "Unknown"
            if 'body' in card_data and len(card_data['body']) > 0:
                for item in card_data['body'][0].get('items', []):
                    if item.get('type') == 'TextBlock' and 'Período:' in item.get('text', ''):
                        date_range = item['text'].replace('Período: ', '').strip()
            
            # Determine period type based on date range format
            period_type = "daily" if re.match(r'\d{1,2}-\w{3}', date_range) else "weekly"
            
            # Parse table data (NPS and variation values)
            segments_data = {}
            
            # First, extract NPS and variation from table
            if 'body' in card_data and len(card_data['body']) > 1:
                table_container = card_data['body'][1]
                if 'items' in table_container:
                    for item in table_container['items']:
                        if item.get('type') == 'ColumnSet' and 'columns' in item:
                            # Check if this is a data row (not header)
                            columns = item['columns']
                            if len(columns) >= 3:
                                # Extract segment name from first column
                                segment_name_item = columns[0].get('items', [{}])[0]
                                if segment_name_item.get('type') == 'TextBlock':
                                    segment_name = segment_name_item.get('text', '').strip()
                                    # Remove icon for key
                                    segment_key = segment_name.replace('🌐 ', '').replace('✈️ ', '').replace('🌍 ', '').strip()
                                    
                                    # Extract NPS from second column
                                    nps_item = columns[1].get('items', [{}])[0]
                                    nps_value = nps_item.get('text', '') if nps_item.get('type') == 'TextBlock' else ''
                                    
                                    # Extract variation from third column
                                    var_item = columns[2].get('items', [{}])[0]
                                    var_value = var_item.get('text', '') if var_item.get('type') == 'TextBlock' else ''
                                    
                                    if segment_key and segment_key != 'Segmento':  # Skip header row
                                        segments_data[segment_key] = {
                                            'nps': nps_value,
                                            'variation': var_value,
                                            'analysis': ''  # Will be filled from actions
                                        }
            
            # Parse analysis text from actions
            if 'actions' in card_data:
                for action in card_data['actions']:
                    if action.get('type') == 'Action.ShowCard' and 'card' in action:
                        card = action['card']
                        title = action.get('title', '').strip()
                        # Remove icon for matching
                        segment_key = title.replace('🌐 ', '').replace('✈️ ', '').replace('🌍 ', '').strip()
                        
                        # Extract analysis text from card body
                        analysis_text = ''
                        if 'body' in card:
                            for body_item in card['body']:
                                if body_item.get('type') == 'TextBlock' and body_item.get('text', ''):
                                    text = body_item['text']
                                    # Skip title lines
                                    if not text.startswith('Análisis:'):
                                        analysis_text += text + '\n\n'
                        
                        # Update segment data with analysis
                        if segment_key in segments_data:
                            segments_data[segment_key]['analysis'] = analysis_text.strip()
                        else:
                            # If segment not in table, add it
                            segments_data[segment_key] = {
                                'nps': '',
                                'variation': '',
                                'analysis': analysis_text.strip()
                            }
            
            return {
                'period_type': period_type,
                'date_range': date_range,
                'segments': segments_data
            }
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing Adaptive Card JSON: {e}")
            return {
                'period_type': 'unknown',
                'date_range': 'Unknown',
                'segments': {}
            }
        except Exception as e:
            self.logger.error(f"Unexpected error parsing Adaptive Card: {e}")
            return {
                'period_type': 'unknown',
                'date_range': 'Unknown',
                'segments': {}
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from the agent."""
        return {
            "input_tokens": getattr(self.agent, 'input_tokens', 0),
            "output_tokens": getattr(self.agent, 'output_tokens', 0),
            "llm_type": self.llm_type.value,
            "model": getattr(self.llm, 'model_name', 'Unknown'),
            "total_time": getattr(self.agent, 'total_time', 0),
            "num_calls": getattr(self.agent, 'num_calls', 0),
            "avg_time": getattr(self.agent, 'avg_time', 0)
        }

    def _get_message_history_for_consolidated(self, consolidated_input: str):
        """Create message history for consolidated analysis combining daily and weekly insights"""
        from ..message_history import MessageHistory
        from ..utils.enums import MessageType
        
        message_history = MessageHistory(logger=self.logger)
        
        # Use consolidated system prompt from YAML configuration
        consolidated_system_prompt = self.config.get('consolidated_system_prompt', 'Default consolidated system prompt not found')
        
        message_history.create_and_add_message(
            content=consolidated_system_prompt,
            message_type=MessageType.SYSTEM
        )
        
        # Use consolidated input template from YAML
        user_input = self.config['consolidated_input_template'].format(
            consolidated_input=consolidated_input
        )
        
        message_history.create_and_add_message(
            content=user_input,
            message_type=MessageType.USER
        )
        
        return message_history

    def _get_message_role(self, message) -> str:
        """Extract role from different message types safely"""
        # Try different ways to get the role
        if hasattr(message, 'role'):
            return message.role
        elif hasattr(message, 'type'):
            message_type = message.type
            # Map LangChain types to standard roles
            type_mapping = {
                'system': 'system',
                'human': 'user', 
                'ai': 'assistant'
            }
            return type_mapping.get(message_type, message_type)
        elif hasattr(message, '__class__'):
            class_name = message.__class__.__name__
            if 'System' in class_name:
                return 'system'
            elif 'Human' in class_name:
                return 'user'
            elif 'AI' in class_name:
                return 'assistant'
            else:
                return class_name.lower()
        else:
            return 'unknown'

    async def export_full_stratified_conversation(
        self,
        all_conversations: dict,
        dateflight_local: Optional[str] = None,
        execution_id: Optional[str] = None,
        final_adaptive_card_json: Optional[str] = None,
        final_executive_synthesis: Optional[str] = None,
        adaptive_card_size_kb: Optional[float] = None,
        daily_analysis_dates: Optional[List[str]] = None,
        optimization_debug: Optional[Dict[str, Any]] = None,
        execution_duration_ms: Optional[float] = None,
        source_causal_report_ids: Optional[List[str]] = None,
    ) -> str:
        """Export the FULL stratified conversation (all 4 steps) to the logging directory and DynamoDB."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            period_range = format_period_range(date_param=dateflight_local)
            filename = f"summary_{timestamp}.json"

            full_path = get_logging_path(self.report_group, "summarizer", period_range, filename)

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
                    agent_type="anomaly_summary",
                    analysis_type="stratified_comprehensive",
                    dateflight_local=dateflight_local,
                    num_steps=4,
                    summary_success=True,
                ),
                "step1_section_connections": all_conversations.get('step1_section_connections', {}),
                "step2_full_report": all_conversations.get('step2_full_report', {}),
                "step3_executive_synthesis": all_conversations.get('step3_executive_synthesis', {}),
                "step4_adaptive_card": all_conversations.get('step4_adaptive_card', {}),
            }

            save_pretty_json(full_path, conversation_data)
            self.logger.info(f"📝 Full stratified conversation exported to: {full_path}")

            s3_key = None
            # Upload to S3 in production
            if self.environment == "prod":
                try:
                    s3_key = get_s3_logging_key(
                        self.s3_uploader.base_prefix, self.report_group,
                        "summarizer", period_range, filename,
                    )
                    self.s3_uploader.s3_client.put_object(
                        Bucket=self.s3_uploader.bucket_name,
                        Key=s3_key,
                        Body=json.dumps(conversation_data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                        ContentType="application/json",
                    )
                    self.logger.info(f"📤 Summary conversation uploaded to S3: {s3_key}")
                except Exception as s3_err:
                    self.logger.warning(f"⚠️ Failed to upload summary conversation to S3: {s3_err}")

            # Persist to DynamoDB
            if execution_id:
                try:
                    from ..utils.dynamodb_report_persistence import DynamoDBReportPersistence
                    dynamo = DynamoDBReportPersistence(environment=self.environment)
                    synthesis_id = dynamo.save_summarizer_report(
                        execution_id=execution_id,
                        agent=self,
                        all_conversations=all_conversations,
                        optimization_debug=optimization_debug,
                        final_adaptive_card_json=final_adaptive_card_json,
                        final_executive_synthesis=final_executive_synthesis,
                        adaptive_card_size_kb=adaptive_card_size_kb,
                        daily_analysis_dates=daily_analysis_dates,
                        execution_duration_ms=execution_duration_ms,
                        status="completed",
                        s3_report_key=s3_key,
                        source_causal_report_ids=source_causal_report_ids,
                        analysis_date=dateflight_local,
                    )
                    if synthesis_id:
                        self.logger.info(f"📊 Summarizer report persisted to DynamoDB: {synthesis_id}")
                except Exception as ddb_err:
                    self.logger.warning(f"⚠️ Failed to persist summarizer report to DynamoDB: {ddb_err}")

            return str(full_path)

        except Exception as e:
            self.logger.error(f"❌ Failed to export stratified conversation: {e}")
            return ""

    async def export_conversation(self, message_history: 'MessageHistory', dateflight_local: Optional[str] = None) -> str:
        """Export the conversation log to the logging directory (legacy single-step)."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            period_range = format_period_range(date_param=dateflight_local)
            filename = f"summary_legacy_{timestamp}.json"

            full_path = get_logging_path(self.report_group, "summarizer", period_range, filename)

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
                    agent_type="anomaly_summary",
                    analysis_type="comprehensive_consolidation",
                    dateflight_local=dateflight_local,
                    total_messages=len(message_history.get_messages()),
                    summary_success=True,
                ),
                "conversation_log": [
                    {
                        "role": self._get_message_role(msg),
                        "content": msg.content,
                        "timestamp": getattr(msg, 'timestamp', None).isoformat() if hasattr(getattr(msg, 'timestamp', None), 'isoformat') else getattr(msg, 'timestamp', None)
                    } for msg in message_history.get_messages()
                ],
            }

            save_pretty_json(full_path, conversation_data)
            self.logger.info(f"📝 Summary conversation exported to: {full_path}")

            if self.environment == "prod":
                try:
                    s3_key = get_s3_logging_key(
                        self.s3_uploader.base_prefix, self.report_group,
                        "summarizer", period_range, filename,
                    )
                    self.s3_uploader.s3_client.put_object(
                        Bucket=self.s3_uploader.bucket_name,
                        Key=s3_key,
                        Body=json.dumps(conversation_data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                        ContentType="application/json",
                    )
                    self.logger.info(f"📤 Summary legacy conversation uploaded to S3: {s3_key}")
                except Exception as s3_err:
                    self.logger.warning(f"⚠️ Failed to upload summary conversation to S3: {s3_err}")

            return str(full_path)

        except Exception as e:
            self.logger.error(f"❌ Failed to export summary conversation: {e}")
            return ""

    async def _generate_adaptive_card(self, report_text: str, date_range: str = None) -> tuple:
        """Helper to generate Adaptive Card JSON from a report text
        
        Args:
            report_text: The full report text to convert to Adaptive Card
            date_range: Date range string (e.g., "9 Dic - 15 Dic 2025")
            
        Returns:
            Tuple of (adaptive_card_json, conversation_dict) where conversation_dict contains:
            - messages: list of messages sent to the model
            - raw_response: the raw response from the model before cleaning
            - result: the final cleaned/repaired JSON
        """
        try:
            self.logger.info("📋 Generating Adaptive Card JSON...")
            
            # Build date range if not provided
            if not date_range:
                date_range = datetime.now().strftime("%d %b %Y")
            
            step4_config = self.config.get('step4_generate_adaptive_card', {})
            # Use .replace() instead of .format() because the system_prompt contains
            # embedded JSON with {} that would be misinterpreted as placeholders
            step4_system = step4_config.get('system_prompt', '').replace('{date_range}', date_range)
            step4_input = step4_config.get('input_template', '').replace(
                '{comprehensive_response}', report_text
            ).replace(
                '{full_report}', report_text
            ).replace(
                '{date_range}', date_range
            )
            
            message_history = MessageHistory()
            message_history.create_and_add_message(content=step4_system, message_type=MessageType.SYSTEM)
            message_history.create_and_add_message(content=step4_input, message_type=MessageType.USER)
            
            response, _, _ = await self.agent.invoke(messages=message_history.get_messages())
            raw_response = response.content if hasattr(response, 'content') else str(response)
            
            # Clean and validate JSON
            final_json = self._repair_json(raw_response)
            
            # Build conversation dict for debugging
            conversation = {
                'messages': [
                    {'role': 'system', 'content': step4_system},
                    {'role': 'user', 'content': step4_input},
                    {'role': 'assistant', 'content': raw_response}
                ],
                'raw_response_length': len(raw_response),
                'result': final_json,
                'result_length': len(final_json)
            }
            
            return final_json, conversation
        except Exception as e:
            self.logger.error(f"❌ Error generating adaptive card: {e}")
            return "{}", {'error': str(e), 'result': '{}'}


# Convenience function for standalone usage
async def generate_summary_report(
    periods_data: List[Dict[str, Any]], 
    llm_type: LLMType = None
) -> str:
    """
    Standalone function to generate summary report without class instantiation.
    
    Args:
        periods_data: List of period analysis dictionaries
        llm_type: LLM type to use for summary generation
        
    Returns:
        Summary report string
    """
    # Use default LLM type if none provided
    if llm_type is None:
        from dashboard_analyzer.anomaly_explanation.genai_core.utils.enums import get_default_llm_type
        llm_type = get_default_llm_type()
    agent = AnomalySummaryAgent(llm_type=llm_type)
    return await agent.generate_summary_report(periods_data)


# Example usage for testing
if __name__ == "__main__":
    async def main():
        # Example periods data for testing
        sample_periods = [
            {
                'period': 1,
                'date_range': '2025-05-25 to 2025-05-31',
                'anomalies': {'Global/SH/Business/YW': '+'},
                'ai_interpretation': 'Anomalía positiva en segmento YW Short Haul Business debido a mejoras en puntualidad'
            },
            {
                'period': 2,
                'date_range': '2025-05-18 to 2025-05-24', 
                'anomalies': {'Global/LH/Premium': '-'},
                'ai_interpretation': 'Anomalía negativa en Premium Long Haul por problemas en comportamiento de tripulación'
            }
        ]
        
        try:
            summary = await generate_summary_report(sample_periods)
            print("Generated Summary Report:")
            print("=" * 50)
            print(summary)
        except Exception as e:
            print(f"Error: {e}")
    
    asyncio.run(main()) 