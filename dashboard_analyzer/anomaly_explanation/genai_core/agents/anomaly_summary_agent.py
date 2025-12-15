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

# Import GenAI Core components (adjusted for new location)
from ..agents.agent import Agent
from ..llms.openai_llm import OpenAiLLM
from ..llms.aws_llm import AWSLLM
from ..utils.enums import LLMType, MessageType, AgentName, get_default_llm_type
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
        environment: str = "prod"
    ):
        """
        Initialize the Anomaly Summary Agent.
        
        Args:
            llm_type: Type of LLM to use (supports OpenAI and AWS Bedrock models)
            config_path: Path to YAML configuration file with prompts
            logger: Optional logger instance
            environment: Environment type ("local" or "prod")
        """
        # Use default LLM type if none provided
        if llm_type is None:
            llm_type = get_default_llm_type()
        self.llm_type = llm_type
        self.config_path = config_path
        self.logger = logger or self._setup_logger()
        self.silent_mode = False
        self.environment = environment
        
        # Initialize S3 uploader with environment
        self.s3_uploader = S3ReportUploader(environment=environment)
        
        # Load environment variables from .devcontainer/.env only if not in prod
        if self.environment != "prod":
            dotenv_path = Path(__file__).parent.parent.parent.parent.parent / '.devcontainer' / '.env'
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
        """Load prompt configuration from YAML file."""
        try:
            # Get the directory of this file to resolve relative paths
            current_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(current_dir, config_path)
            
            with open(full_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            self.logger.debug(f"Loaded summary prompt configuration from {full_path}")
            return config
            
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def _create_llm(self, llm_type: LLMType):
        """Factory method to create appropriate LLM instance based on type."""
        
        # OpenAI/Azure OpenAI models
        if llm_type in [LLMType.GPT3_5, LLMType.GPT4, LLMType.GPT4o, LLMType.GPT4o_MINI, 
                       LLMType.O1_MINI, LLMType.O3_MINI, LLMType.O3, LLMType.O4_MINI]:
            return self._create_openai_llm(llm_type)
        
        # AWS Bedrock models
        elif llm_type in [
            LLMType.CLAUDE_3_HAIKU, LLMType.CLAUDE_3_5_HAIKU, LLMType.CLAUDE_3_OPUS,
            LLMType.CLAUDE_3_5_SONNET, LLMType.CLAUDE_3_5_SONNET_V2, LLMType.CLAUDE_3_7_SONNET,
            LLMType.CLAUDE_SONNET_4, LLMType.LLAMA3_70, LLMType.LLAMA3_1_70, LLMType.LLAMA3_1_405
        ]:
            return self._create_aws_llm(llm_type)
        
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
    
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
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            api_dep_gpt=deployment_name,
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
            
            # Export conversation for debugging (use first period date as identifier)
            first_period_date = periods_data[0].get('period', 'unknown') if periods_data else None
            conversation_file = await self.export_conversation(message_history, first_period_date)
            if conversation_file:
                self.logger.info(f"🗂️ Conversación de summary guardada: {conversation_file}")
            
            if summary_response:
                self.logger.info(f"✅ Generated summary report for {num_periods} periods")
                return summary_response
            else:
                return "⚠️ Failed to generate summary report"
                
        except Exception as e:
            self.logger.error(f"❌ Error generating summary report: {str(e)}")
            return f"❌ Error generating summary: {str(e)}"
    
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
            
            # Format daily analyses - only include days with relevant analysis
            daily_analyses_formatted = []
            total_days = len(daily_single_analyses)
            filtered_days = 0
            
            for daily_analysis in daily_single_analyses:
                date = daily_analysis.get('date', 'Unknown')
                analysis = daily_analysis.get('analysis', '')
                anomalies = daily_analysis.get('anomalies', [])
                
                # Log length and only skip if truly empty
                text = (analysis or '').strip()
                self.logger.info(f"📝 Daily analysis length for {date}: {len(text)} chars")
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
                os.makedirs("dashboard_analyzer/summary_reports", exist_ok=True)
                dbg_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                debug_path = f"dashboard_analyzer/summary_reports/summary_agent_input_{dbg_ts}.md"
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
            
            # Export conversation for debugging
            conversation_file = await self.export_conversation(message_history, date_flight_local)
            if conversation_file:
                self.logger.info(f"🗂️ Conversación de summary guardada: {conversation_file}")
            
            if comprehensive_response:
                self.logger.info(f"✅ Generated comprehensive summary: weekly + {len(daily_single_analyses)} daily analyses")
                
                # Upload to S3 if metadata is provided
                if (execution_metadata and weekly_analysis_params and 
                    daily_analysis_params and date_ranges):
                    try:
                        self.logger.info("📤 Uploading comprehensive report to S3...")
                        s3_key = await self.s3_uploader.upload_comprehensive_report(
                            execution_date=datetime.now(),
                            analysis_date=execution_metadata.get('analysis_date', ''),
                            segment=execution_metadata.get('segment', 'Global'),
                            explanation_mode=execution_metadata.get('explanation_mode', 'agent'),
                            causal_filter=execution_metadata.get('causal_filter', 'vs L7d'),
                            weekly_analysis_params=weekly_analysis_params,
                            daily_analysis_params=daily_analysis_params,
                            date_ranges=date_ranges,
                            final_synthesis=comprehensive_response,
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
                
                return comprehensive_response
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
        date_ranges: Optional[Dict[str, Any]] = None
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
        
        Returns:
            Comprehensive summary string
        """
        try:
            if not weekly_comparative_analysis and not daily_single_analyses:
                return "⚠️ No data provided for comprehensive summary generation"
            
            # Format daily analyses
            daily_analyses_formatted = []
            for daily_analysis in daily_single_analyses:
                date = daily_analysis.get('date', 'Unknown')
                analysis = daily_analysis.get('analysis', '')
                text = (analysis or '').strip()
                if text:
                    daily_analyses_formatted.append(f"📅 {date}:\n{text}")
            
            daily_analyses_combined = "\n\n---\n\n".join(daily_analyses_formatted)
            num_days = len(daily_analyses_formatted)
            
            self.logger.info(f"🔄 Starting stratified summary: {num_days} days to analyze")
            
            # =========================================================
            # Parse weekly analysis into sections
            # =========================================================
            sections = self._parse_weekly_sections(weekly_comparative_analysis)
            self.logger.info(f"📊 Parsed {len(sections)} sections from weekly analysis")
            
            # =========================================================
            # STEP 1: Analyze connections for each section
            # =========================================================
            self.logger.info("📋 STEP 1: Analyzing section connections...")
            
            step1_config = self.config.get('step1_analyze_section', {})
            step1_system = step1_config.get('system_prompt', '')
            step1_template = step1_config.get('input_template', '')
            
            daily_context_paragraphs = {}
            
            for section_name, section_content in sections.items():
                self.logger.info(f"   📌 Analyzing: {section_name}")
                
                # Filter daily analyses for this section
                daily_for_section = self._filter_daily_for_section(
                    daily_analyses_combined, 
                    section_name
                )
                
                # Skip sections without relevant daily data (except GLOBAL which always has data)
                if not daily_for_section and section_name != 'GLOBAL':
                    self.logger.info(f"   ⏭️ Skipping {section_name}: no specific daily data found")
                    continue
                
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
                self.logger.info(f"   ✅ Generated context for {section_name}: {len(paragraph)} chars")
            
            self.logger.info(f"✅ Step 1 complete: Generated {len(daily_context_paragraphs)} context paragraphs")
            
            # =========================================================
            # STEP 2: Integrate into final report
            # =========================================================
            self.logger.info("📝 STEP 2: Integrating into final report...")
            
            # Format the context paragraphs for Step 2
            context_formatted = "\n\n".join([
                f"**{name}:**\n{para}" 
                for name, para in daily_context_paragraphs.items()
            ])
            
            step2_config = self.config.get('step2_integrate_final', {})
            step2_system = step2_config.get('system_prompt', '')
            step2_input = step2_config.get('input_template', '').format(
                weekly_analysis=weekly_comparative_analysis,
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
            
            # =========================================================
            # Save debug files and export conversation
            # =========================================================
            try:
                os.makedirs("dashboard_analyzer/summary_reports", exist_ok=True)
                dbg_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                debug_path = f"dashboard_analyzer/summary_reports/summary_stratified_{dbg_ts}.md"
                with open(debug_path, 'w', encoding='utf-8') as dbg:
                    dbg.write("===== STEP 1: SECTION CONNECTIONS =====\n\n")
                    for section_name, paragraph in daily_context_paragraphs.items():
                        dbg.write(f"--- {section_name} ---\n{paragraph}\n\n")
                    dbg.write("===== STEP 2: FULL REPORT =====\n\n")
                    dbg.write(polished_report)
                    dbg.write("\n\n===== STEP 3: EXECUTIVE SYNTHESIS =====\n\n")
                    dbg.write(executive_synthesis)
                self.logger.info(f"📝 Saved stratified debug to: {debug_path}")
            except Exception as e:
                self.logger.warning(f"Could not write debug file: {e}")
            
            # Export final conversation (step 3)
            conversation_file = await self.export_conversation(message_history_step3, date_flight_local)
            if conversation_file:
                self.logger.info(f"🗂️ Summary conversation saved: {conversation_file}")
            
            # Upload to S3 if metadata is provided (use executive_synthesis, not full report)
            if executive_synthesis and execution_metadata and weekly_analysis_params and daily_analysis_params and date_ranges:
                try:
                    self.logger.info("📤 Uploading executive synthesis to S3...")
                    s3_key = await self.s3_uploader.upload_comprehensive_report(
                        execution_date=datetime.now(),
                        analysis_date=execution_metadata.get('analysis_date', ''),
                        segment=execution_metadata.get('segment', 'Global'),
                        explanation_mode=execution_metadata.get('explanation_mode', 'agent'),
                        causal_filter=execution_metadata.get('causal_filter', 'vs L7d'),
                        weekly_analysis_params=weekly_analysis_params,
                        daily_analysis_params=daily_analysis_params,
                        date_ranges=date_ranges,
                        final_synthesis=executive_synthesis,
                        comparison_start_date=date_ranges.get('comparison_start_date'),
                        comparison_end_date=date_ranges.get('comparison_end_date')
                    )
                    if s3_key:
                        self.logger.info(f"✅ Report uploaded to S3: {s3_key}")
                except Exception as e:
                    self.logger.error(f"❌ Error uploading to S3: {str(e)}")
            
            return executive_synthesis
            
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
    
    def _parse_weekly_sections(self, weekly_analysis: str) -> Dict[str, str]:
        """
        Parse the weekly analysis into sections (Global, Economy SH, Business SH, etc.)
        
        Returns:
            Dictionary mapping section names to their content
        """
        import re
        
        sections = {}
        
        # Define section patterns to look for
        # Patterns match both "**ECONOMY SH:" and "ECONOMY SH:" formats
        section_patterns = [
            ('GLOBAL', r'(?:📈\s*\*\*SÍNTESIS EJECUTIVA|Durante la semana)'),
            ('ECONOMY SH', r'(?:\*\*)?ECONOMY SH[:\s]'),
            ('BUSINESS SH', r'(?:\*\*)?BUSINESS SH[:\s]'),
            ('ECONOMY LH', r'(?:\*\*)?ECONOMY LH[:\s]'),
            ('BUSINESS LH', r'(?:\*\*)?BUSINESS LH[:\s]'),
            ('PREMIUM LH', r'(?:\*\*)?PREMIUM LH[:\s]'),
        ]
        
        # Find all section headers and their positions
        section_positions = []
        for section_name, pattern in section_patterns:
            matches = list(re.finditer(pattern, weekly_analysis, re.IGNORECASE))
            for match in matches:
                section_positions.append((match.start(), section_name, match.group()))
        
        # Sort by position
        section_positions.sort(key=lambda x: x[0])
        
        # Extract content for each section
        for i, (pos, section_name, _) in enumerate(section_positions):
            # Find end of this section (start of next section or end of text)
            if i + 1 < len(section_positions):
                end_pos = section_positions[i + 1][0]
            else:
                end_pos = len(weekly_analysis)
            
            content = weekly_analysis[pos:end_pos].strip()
            
            # Only keep if we don't already have this section (avoid duplicates)
            if section_name not in sections:
                sections[section_name] = content
        
        # If no sections found, use the whole text as GLOBAL
        if not sections:
            sections['GLOBAL'] = weekly_analysis
        
        self.logger.info(f"📊 Parsed sections: {list(sections.keys())}")
        return sections
    
    def _filter_daily_for_section(self, daily_analyses: str, section_name: str) -> str:
        """
        Filter daily analyses to show only content relevant to a specific section.
        
        For GLOBAL, returns all daily analyses.
        For specific cabins (Economy SH, Business LH, etc.), filters to show
        only paragraphs/sections mentioning that cabin.
        """
        if section_name == 'GLOBAL':
            # For global, return all daily analyses
            return daily_analyses
        
        # Map section names to keywords to search for
        keyword_map = {
            'ECONOMY SH': ['economy sh', 'economy de sh', 'sh economy', 'short haul economy'],
            'BUSINESS SH': ['business sh', 'business de sh', 'sh business', 'short haul business'],
            'ECONOMY LH': ['economy lh', 'economy de lh', 'lh economy', 'long haul economy'],
            'BUSINESS LH': ['business lh', 'business de lh', 'lh business', 'long haul business'],
            'PREMIUM LH': ['premium lh', 'premium de lh', 'lh premium', 'long haul premium'],
        }
        
        keywords = keyword_map.get(section_name, [section_name.lower()])
        
        # Split into daily entries
        daily_entries = daily_analyses.split('---')
        
        # Filter entries that mention this section
        relevant_entries = []
        for entry in daily_entries:
            entry_lower = entry.lower()
            if any(kw in entry_lower for kw in keywords):
                relevant_entries.append(entry.strip())
        
        if relevant_entries:
            return "\n\n---\n\n".join(relevant_entries)
        else:
            # If no specific entries found for this cabin, return empty
            # This prevents copying the GLOBAL paragraph to all sections
            return ""
    
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

    async def export_conversation(self, message_history: 'MessageHistory', dateflight_local: Optional[str] = None) -> str:
        """Export the conversation log to JSON file and upload to S3"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Use dateflight_local if provided, otherwise use current date
            period_identifier = dateflight_local if dateflight_local else timestamp[:8]  # Extract YYYYMMDD from timestamp
            filename = f"summary_{period_identifier}_{timestamp}.json"
            
            # Create agent_conversations directory structure
            base_dir = Path(__file__).parent.parent.parent.parent.parent / 'dashboard_analyzer' / 'agent_conversations' / 'anomaly_summary'
            base_dir.mkdir(parents=True, exist_ok=True)
            
            full_path = base_dir / filename
            
            conversation_data = {
                "metadata": {
                    "agent_type": "anomaly_summary",
                    "analysis_type": "comprehensive_consolidation",
                    "export_timestamp": datetime.now().isoformat(),
                    "dateflight_local": dateflight_local,
                    "llm_type": self.llm_type.value,
                    "total_messages": len(message_history.get_messages()),
                    "summary_success": True
                },
                "conversation_log": [
                    {
                        "role": self._get_message_role(msg),
                        "content": msg.content,
                        "timestamp": getattr(msg, 'timestamp', None).isoformat() if hasattr(getattr(msg, 'timestamp', None), 'isoformat') else getattr(msg, 'timestamp', None)
                    } for msg in message_history.get_messages()
                ]
            }
            
            # Save locally
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📝 Summary conversation exported to: {full_path}")
            
            # Upload to S3 in production
            try:
                s3_key = await self.s3_uploader.upload_summary_conversation(conversation_data, filename)
                if s3_key:
                    self.logger.info(f"📤 Summary conversation uploaded to S3: {s3_key}")
                else:
                    self.logger.info("🔧 S3 upload skipped (local environment or failed)")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to upload to S3: {e}")
            
            return str(full_path)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export summary conversation: {e}")
            return ""


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