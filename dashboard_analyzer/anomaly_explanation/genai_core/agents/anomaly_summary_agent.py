"""
Anomaly Summary Agent

A specialized AI agent for summarizing multiple periods of NPS anomaly analysis
and generating executive-level insights about trends, patterns, and strategic priorities.
"""

import asyncio
import os
import yaml
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Import GenAI Core components (adjusted for new location)
from ..agents.agent import Agent
from ..llms.openai_llm import OpenAiLLM
from ..llms.aws_llm import AWSLLM
from ..utils.enums import LLMType, MessageType
from ..message_history import MessageHistory


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
        llm_type: LLMType = LLMType.O3,
        config_path: str = "../../config/prompts/anomaly_summary.yaml",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Anomaly Summary Agent.
        
        Args:
            llm_type: Type of LLM to use (supports OpenAI and AWS Bedrock models)
            config_path: Path to YAML configuration file with prompts
            logger: Optional logger instance
        """
        self.logger = logger or self._setup_logger()
        self.llm_type = llm_type
        
        # Load environment variables from .devcontainer/.env
        dotenv_path = Path(__file__).parent.parent.parent.parent.parent / '.devcontainer' / '.env'
        load_dotenv(dotenv_path)
        
        # Load prompt configuration
        self.config = self._load_prompt_config(config_path)
        
        # Initialize LLM based on type
        self.llm = self._create_llm(llm_type)
        
        # Initialize base agent
        self.agent = Agent(llm=self.llm, logger=self.logger)
        
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
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        if not all([api_key, api_base, deployment_name]):
            raise ValueError("Missing required OpenAI/Azure OpenAI environment variables")
        
        return OpenAiLLM(
            llm_type=llm_type,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            api_dep_gpt=deployment_name,
            temperature=0.2  # Slightly higher for more creative summarization
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
            periods_data: List of dictionaries containing period analysis data.
                         Each dict should have: period, anomalies, interpretations, date_range
            
        Returns:
            Executive summary report identifying trends, patterns, and strategic priorities
        """
        start_time = datetime.now()
        
        try:
            # Format the periods data for AI consumption
            formatted_analysis = self._format_periods_for_summary(periods_data)
            
            # Create message history
            message_history = MessageHistory(logger=self.logger)
            
            # Add system prompt
            system_prompt = self.config['system_prompt']
            message_history.create_and_add_message(
                content=system_prompt,
                message_type=MessageType.SYSTEM
            )
            
            # Format user input using template
            user_input = self.config['input_template'].format(
                num_periods=len(periods_data),
                periods_analysis=formatted_analysis
            )
            
            message_history.create_and_add_message(
                content=user_input,
                message_type=MessageType.USER
            )
            
            # Get summary from LLM
            self.logger.debug(f"Generating summary report for {len(periods_data)} periods")
            
            response, structured_response, tool_calls = await self.agent.invoke(
                messages=message_history.get_messages()
            )
            
            summary_report = response.content.strip()
            
            # Log performance metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Summary report generated in {execution_time:.2f}s "
                f"(Input: {self.agent.input_tokens}, Output: {self.agent.output_tokens} tokens)"
            )
            
            return summary_report
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {str(e)}")
            raise
    
    def _format_periods_for_summary(self, periods_data: List[Dict[str, Any]]) -> str:
        """Format periods data into a structured text for AI analysis."""
        formatted_parts = []
        
        for period_data in periods_data:
            period = period_data.get('period', 'Unknown')
            date_range = period_data.get('date_range', 'Unknown dates')
            ai_interpretation = period_data.get('ai_interpretation', 'No interpretation available')
            
            # Format period summary with just the AI interpretation
            period_summary = f"""
PERÍODO {period} ({date_range}):
{ai_interpretation}
"""
            formatted_parts.append(period_summary.strip())
        
        return "\n\n".join(formatted_parts)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from the agent."""
        return {
            "input_tokens": getattr(self.agent, 'input_tokens', 0),
            "output_tokens": getattr(self.agent, 'output_tokens', 0),
            "llm_type": self.llm_type.value,
            "model": getattr(self.llm, 'model_name', 'Unknown')
        }

    def _get_message_history_for_consolidated(self, consolidated_input: str):
        """Create message history for consolidated analysis combining daily and weekly insights"""
        from ..message_history import MessageHistory
        from ..utils.enums import MessageType
        
        message_history = MessageHistory(logger=self.logger)
        
        # Modified system prompt for consolidated analysis
        consolidated_system_prompt = """
Eres un experto analista ejecutivo especializado en consolidar análisis de NPS de múltiples escalas temporales (diario y semanal).

Se te proporcionarán:
- Resúmenes de análisis DIARIO (últimos 7 días)
- Resúmenes de análisis SEMANAL (últimas 5 semanas) para contexto

Tu objetivo es crear un RESUMEN EJECUTIVO CONSOLIDADO que:
- Identifique si las anomalías diarias son parte de tendencias semanales más amplias
- Distinga entre fluctuaciones diarias normales vs patrones preocupantes
- Proporcione contexto temporal: ¿son los problemas diarios nuevos o parte de tendencias semanales?
- Identifique prioridades: ¿qué requiere atención inmediata vs seguimiento?

Restricciones:
- Máximo 300 palabras
- Tono ejecutivo y estratégico
- Enfócate en la relación entre análisis diario y semanal
- No repitas información, sintetiza patrones cruzados
"""
        
        message_history.create_and_add_message(
            content=consolidated_system_prompt,
            message_type=MessageType.SYSTEM
        )
        
        # User input with consolidated data
        user_input = f"""
Analiza los siguientes análisis DIARIOS y SEMANALES y genera un resumen ejecutivo consolidado que identifique la relación entre ambas escalas temporales:

{consolidated_input}

Genera únicamente el RESUMEN EJECUTIVO CONSOLIDADO enfocándote en la relación entre patrones diarios y semanales.
"""
        
        message_history.create_and_add_message(
            content=user_input,
            message_type=MessageType.USER
        )
        
        return message_history


# Convenience function for standalone usage
async def generate_summary_report(
    periods_data: List[Dict[str, Any]], 
    llm_type: LLMType = LLMType.O3
) -> str:
    """
    Standalone function to generate summary report without class instantiation.
    
    Args:
        periods_data: List of period analysis dictionaries
        llm_type: LLM type to use for summary generation
        
    Returns:
        Summary report string
    """
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