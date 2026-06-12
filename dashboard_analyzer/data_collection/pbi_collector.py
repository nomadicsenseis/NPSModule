import os
import msal
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import asyncio
import aiohttp
import re # Added for regex replacement
import logging
import time
import random

# Retry configuration for PBI API calls
PBI_MAX_RETRIES = 3  # Maximum number of retry attempts
PBI_BASE_DELAY = 2.0  # Base delay in seconds for exponential backoff
PBI_MAX_DELAY = 30.0  # Maximum delay between retries
PBI_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}  # HTTP codes that trigger retry

# Mapping from filtered_name (TouchPoint_Master) to display_name (Issue_touchpoint_Dict).
# Needed because the two tables use different naming conventions.
# Extend this dict to support additional touchpoints without code changes.
TOUCHPOINT_DISPLAY_NAME_MAP: Dict[str, str] = {
    # Keys: filtered_name values from TouchPoint_Master[filtered_name]
    "ifl_100_cabin_crew_satisfaction": "Cabin Crew",
    "ifl_200_food_satisfaction": "Food & Beverage",
    "ifl_300_seat_satisfaction": "Seat",
    "ifl_400_entertainment_satisfaction": "Entertainment",
    "ifl_500_wifi_satisfaction": "Wi-Fi",
    "ifl_600_checkin_satisfaction": "Check-in",
    "ifl_700_boarding_satisfaction": "Boarding",
    "ifl_800_baggage_satisfaction": "Baggage",
    "ifl_900_lounge_satisfaction": "Lounge",
    # Keys: display_name values (TouchPoint_Master[filtered_name] confirmed values)
    # Allows passing the display name directly via CLI (e.g. --focus-touchpoint "Cabin Crew")
    "Cabin Crew": "Cabin Crew",
    "Check-in": "Check-in",
    "Lounge": "Lounge",
    "Boarding": "Boarding",
    "Aircraft interior": "Aircraft interior",
    "Wi-Fi": "Wi-Fi",
    "IFE": "IFE",
    "In flight food and beverage": "Food & Beverage",
    "Arrivals experience": "Arrivals experience",
    "Connections experience": "Connections experience",
    "Punctuality": "Punctuality",
}


# Mapping from display_name to Issue_touchpoint_Dict[issue_type_3] value.
# The issues table uses a different (longer) naming convention than TouchPoint_Master.
TOUCHPOINT_ISSUE_TYPE_MAP: Dict[str, str] = {
    "Cabin Crew": "issue with a crew member of staff on-board",
    "Food & Beverage": "issue with food and beverage on-board",
    "Seat": "issue with seat on-board",
    "Entertainment": "issue with entertainment on-board",
    "Wi-Fi": "issue with wi-fi on-board",
    "Check-in": "issue with check-in",
    "Boarding": "issue with boarding",
    "Baggage": "issue with baggage",
    "Lounge": "issue with lounge",
    "Aircraft interior": "issue with aircraft interior",
    "IFE": "issue with entertainment on-board",
    "Arrivals experience": "issue with arrivals experience",
    "Connections experience": "issue with connections experience",
    "Punctuality": "issue with punctuality",
}


# Mapping from display_name to verbatims_sentiment[topic] value.
# Used to filter verbatims by touchpoint topic.
TOUCHPOINT_TOPIC_MAP: Dict[str, str] = {
    "Cabin Crew": "Comportamiento Tripulación",
    "Food & Beverage": "Comida y Bebida",
    "Seat": "Asiento",
    "Entertainment": "Entretenimiento",
    "Wi-Fi": "Wi-Fi",
    "Check-in": "Check-in",
    "Boarding": "Embarque",
    "Baggage": "Equipaje",
    "Lounge": "Sala VIP",
    "Aircraft interior": "Interior del Avión",
    "IFE": "Entretenimiento",
    "Arrivals experience": "Desembarque",
    "Connections experience": "Experiencia de Conexión",
    "Punctuality": "Puntualidad",
}


def get_touchpoint_display_name(filtered_name: str) -> Optional[str]:
    """Return the display_name for a touchpoint filtered_name, or None if no mapping exists."""
    return TOUCHPOINT_DISPLAY_NAME_MAP.get(filtered_name)


def get_touchpoint_issue_type(display_name: str) -> Optional[str]:
    """Return the Issue_touchpoint_Dict[issue_type_3] value for a display_name, or None."""
    return TOUCHPOINT_ISSUE_TYPE_MAP.get(display_name)


def get_touchpoint_topic(display_name: str) -> Optional[str]:
    """Return the verbatims_sentiment[topic] value for a display_name, or None."""
    return TOUCHPOINT_TOPIC_MAP.get(display_name)


class PBIDataCollector:
    """Collects data from Power BI API for each node in the NPS tree hierarchy"""
    
    def __init__(self, environment: str = "prod"):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.environment = environment
        
        # Load environment variables from .env in current working directory ONLY if not in prod
        if self.environment != "prod":
            dotenv_path = Path.cwd() / '.env'
            if dotenv_path.exists():
                print(f"🔍 DEBUG: Loading .env from {dotenv_path}")
                load_dotenv(dotenv_path, override=True)
            else:
                print(f"⚠️ WARNING: .env file not found at {dotenv_path}")
        
        # Get credentials from environment
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET") 
        self.tenant_id = os.getenv("TENANT_ID")
        
        # Strip whitespace from credentials if they exist
        if self.client_id:
            self.client_id = self.client_id.strip()
        if self.client_secret:
            self.client_secret = self.client_secret.strip()
        if self.tenant_id:
            self.tenant_id = self.tenant_id.strip()
        
        # DEBUG CREDENTIALS (safe print)
        if self.client_id:
            print(f"🔍 DEBUG CREDENTIALS: CLIENT_ID starts with: {self.client_id[:5]}...")
        if self.client_secret:
            # Check for potentially problematic characters in secret length/content without revealing it
            has_special = any(c in self.client_secret for c in "$%#")
            print(f"🔍 DEBUG CREDENTIALS: CLIENT_SECRET len={len(self.client_secret)}, has_special_chars=${has_special}")
        if self.tenant_id:
            print(f"🔍 DEBUG CREDENTIALS: TENANT_ID starts with: {self.tenant_id[:5]}...")
        self.group_id = os.getenv("GROUP_ID")
        self.dataset_id = os.getenv("DATASET_ID")
        
        missing_vars = []
        if not self.client_id: missing_vars.append("CLIENT_ID")
        if not self.client_secret: missing_vars.append("CLIENT_SECRET")
        if not self.tenant_id: missing_vars.append("TENANT_ID")
        if not self.group_id: missing_vars.append("GROUP_ID")
        if not self.dataset_id: missing_vars.append("DATASET_ID")
        
        if missing_vars:
            error_msg = f"❌ CRITICAL: Missing required environment variables for Power BI API: {', '.join(missing_vars)}"
            if self.environment == "prod":
                error_msg += "\n   Running in 'prod' mode: System environment variables are expected but not found."
            else:
                error_msg += "\n   Running in 'local' mode: Checked .env in current directory but variables are missing."
            print(error_msg)
            raise ValueError(error_msg)
        
        self.access_token = None
        
        # Configure API timeout from environment variable (default 120 seconds)
        self.api_timeout = int(os.getenv("PBI_API_TIMEOUT", "120"))
        self.logger.info(f"🔧 PBI API timeout configured: {self.api_timeout}s")
        
        # Get access token on initialization
        self.access_token = self._get_access_token()
        
        if not self.access_token:
            print("❌ Failed to get access token")
        else:
            print("✅ Successfully authenticated with Power BI API")
        
        # Define the tree structure based on the hierarchy
        self.tree_structure = {
            'Global': {
                'LH': {
                    'Economy': ['IB', 'YW'],
                    'Business': ['IB', 'YW'], 
                    'Premium': ['IB', 'YW']
                },
                'SH': {
                    'Economy': ['IB', 'YW'],
                    'Business': ['IB', 'YW']
                }
            }
        }
        
        # Path to query files
        self.queries_path = Path(__file__).parent / 'queries'
        
    def _load_query_template(self, query_file: str) -> str:
        """Load DAX query template from file"""
        query_path = self.queries_path / query_file
        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_path}")
        
        with open(query_path, 'r', encoding='utf-8') as f:
            return f.read()
        
    def _get_access_token(self) -> str:
        """Get access token for Power BI API"""
        authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        scope = ["https://analysis.windows.net/powerbi/api/.default"]
        
        app = msal.ConfidentialClientApplication(
            client_id=self.client_id,
            client_credential=self.client_secret,
            authority=authority
        )
        
        result = app.acquire_token_for_client(scopes=scope)
        
        if "access_token" not in result:
            raise Exception("Error getting token: " + str(result))
            
        return result["access_token"]
    
    
    def _get_operative_query(self, cabins: List[str], companies: List[str], hauls: List[str], target_date: datetime = None, comparison_days: int = 7) -> str:
        """Generate DAX query for operative data using template"""
        template = self._load_query_template("Operativa.txt")
        
        # Replace placeholders with actual values
        cabins_str = '", "'.join(cabins)
        companies_str = '", "'.join(companies)
        hauls_str = '", "'.join(hauls)
        
        # Replace the template placeholders
        query = template.replace(
            'TREATAS({"Business", "Economy", "Premium EC"}, \'Cabin_Master\'[Cabin_Show])',
            f'TREATAS({{"{cabins_str}"}}, \'Cabin_Master\'[Cabin_Show])'
        ).replace(
            'TREATAS({"IB","YW"}, \'Company_Master\'[Company])',
            f'TREATAS({{"{companies_str}"}}, \'Company_Master\'[Company])'
        ).replace(
            'TREATAS({"SH","LH"}, \'Haul_Master\'[Haul_Aggr])',
            f'TREATAS({{"{hauls_str}"}}, \'Haul_Master\'[Haul_Aggr])'
        )
        
        # If target_date is provided, modify the date filter to look for a specific date range
        if target_date:
            # Original filter: 'Date_Master'[Date] > TODAY()-21
            # New filter: target date and previous (comparison_days-1) days for anomaly comparison
            start_date = target_date - timedelta(days=comparison_days-1)  # Include target date in the count
            end_date = target_date  # Include target date
            
            old_date_filter = "'Date_Master'[Date] > TODAY()-21"
            new_date_filter = f"'Date_Master'[Date] >= DATE({start_date.year},{start_date.month},{start_date.day}) && 'Date_Master'[Date] <= DATE({end_date.year},{end_date.month},{end_date.day})"
            
            query = query.replace(old_date_filter, new_date_filter)
            print(f"  📅 Operative date filter: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({comparison_days} days for anomaly comparison)")
        
        return query
    
    def _get_nps_vs_sel_period_query(self, cabins: List[str], companies: List[str], hauls: List[str], 
                                    current_start_date: datetime, current_end_date: datetime,
                                    comparison_start_date: datetime, comparison_end_date: datetime) -> str:
        """Generate DAX query for NPS comparison between two specific periods using NPS_ED_adjusted"""
        template = self._load_query_template("NPS_vs_sel_period.txt")
        
        # Create filter strings for the new format
        cabin_filter = f"Cabin_Master[Cabin_Show] IN {{{', '.join([f'"{cabin}"' for cabin in cabins])}}}"
        company_filter = f"Company_Master[Company] IN {{{', '.join([f'"{company}"' for company in companies])}}}"
        haul_filter = f"Haul_Master[Haul_Aggr] IN {{{', '.join([f'"{haul}"' for haul in hauls])}}}"
        
        # Replace filter placeholders
        query = template.replace('{CABIN_FILTER}', cabin_filter)
        query = query.replace('{COMPANY_FILTER}', company_filter)
        query = query.replace('{HAUL_FILTER}', haul_filter)
        
        # Replace current period dates
        query = query.replace('{CURRENT_START_YEAR}', str(current_start_date.year))
        query = query.replace('{CURRENT_START_MONTH}', str(current_start_date.month))
        query = query.replace('{CURRENT_START_DAY}', str(current_start_date.day))
        query = query.replace('{CURRENT_END_YEAR}', str(current_end_date.year))
        query = query.replace('{CURRENT_END_MONTH}', str(current_end_date.month))
        query = query.replace('{CURRENT_END_DAY}', str(current_end_date.day))
        
        # Replace comparison period dates
        query = query.replace('{COMPARISON_START_YEAR}', str(comparison_start_date.year))
        query = query.replace('{COMPARISON_START_MONTH}', str(comparison_start_date.month))
        query = query.replace('{COMPARISON_START_DAY}', str(comparison_start_date.day))
        query = query.replace('{COMPARISON_END_YEAR}', str(comparison_end_date.year))
        query = query.replace('{COMPARISON_END_MONTH}', str(comparison_end_date.month))
        query = query.replace('{COMPARISON_END_DAY}', str(comparison_end_date.day))
        
        print(f"  📊 Using NPS vs Sel. Period query with NPS_ED_adjusted")
        print(f"  📅 Current period: {current_start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')}")
        print(f"  📅 Comparison period: {comparison_start_date.strftime('%Y-%m-%d')} to {comparison_end_date.strftime('%Y-%m-%d')}")
        print(f"  🎯 Filters: {cabins} | {companies} | {hauls}")
        
        return query
    
    def _get_flexible_nps_query(self, aggregation_days: int, cabins: List[str], companies: List[str], hauls: List[str], analysis_date: datetime = None) -> str:
        """Generate DAX query for flexible NPS aggregation using template"""
        template = self._load_query_template("NPS_flex_agg.txt")
        
        # Replace placeholders with actual values
        cabins_str = '", "'.join(cabins)
        companies_str = '", "'.join(companies)
        hauls_str = '", "'.join(hauls)
        
        # If analysis_date is provided, handle date-specific replacements FIRST
        if analysis_date:
            print(f"  📅 Using analysis date: {analysis_date.strftime('%Y-%m-%d')} for NPS aggregation")
            # Replace the date filter to end on analysis_date instead of TODAY()
            old_date_filter = "'Date_Master'[Date] >= DATE(2024,01,01) && 'Date_Master'[Date] <= TODAY()"
            new_date_filter = f"'Date_Master'[Date] >= DATE(2024,01,01) && 'Date_Master'[Date] <= DATE({analysis_date.year},{analysis_date.month},{analysis_date.day})"
            print(f"  📝 NPS date filter BEFORE: {old_date_filter}")
            template = template.replace(old_date_filter, new_date_filter)
            print(f"  📝 NPS date filter AFTER: {new_date_filter}")
            
            # Replace the period calculation to use analysis_date as reference (BEFORE replacing AGGREGATION_DAYS)
            old_period_calc = "INT(DATEDIFF( 'Date_Master'[Date],max('Date_Master'[Date]), DAY) / {AGGREGATION_DAYS}) + 1)"
            new_period_calc = f"INT(DATEDIFF( 'Date_Master'[Date],DATE({analysis_date.year},{analysis_date.month},{analysis_date.day}), DAY) / {{AGGREGATION_DAYS}}) + 1)"
            print(f"  📝 NPS period calc BEFORE: {old_period_calc}")
            template = template.replace(old_period_calc, new_period_calc)
            print(f"  📝 NPS period calc AFTER: {new_period_calc}")
        else:
            print(f"  ⚠️ No analysis_date provided, using default TODAY() and max('Date_Master'[Date])")
        
        # Now replace the template placeholders
        query = template.replace(
            '{AGGREGATION_DAYS}', str(aggregation_days)
        ).replace(
            'TREATAS({"Business", "Economy", "Premium EC"}, \'Cabin_Master\'[Cabin_Show])',
            f'TREATAS({{"{cabins_str}"}}, \'Cabin_Master\'[Cabin_Show])'
        ).replace(
            'TREATAS({"IB","YW"}, \'Company_Master\'[Company])',
            f'TREATAS({{"{companies_str}"}}, \'Company_Master\'[Company])'
        ).replace(
            'TREATAS({"SH","LH"}, \'Haul_Master\'[Haul_Aggr])',
            f'TREATAS({{"{hauls_str}"}}, \'Haul_Master\'[Haul_Aggr])'
        )
        
        print(f"  📋 Final query preview: {query[:200]}...")
        return query

    def _get_operative_vs_sel_period_query(self, cabins: List[str], companies: List[str], hauls: List[str], 
                                          current_start_date: datetime, current_end_date: datetime,
                                          comparison_start_date: datetime, comparison_end_date: datetime) -> str:
        """Generate simplified DAX query for operative comparison between two specific periods"""
        template = self._load_query_template("Operativa_vs_sel_period.txt")
        
        # Replace cabin filters
        cabins_str = '", "'.join(cabins)
        template = template.replace(
            'TREATAS({"Business", "Economy", "Premium EC"}, \'Cabin_Master\'[Cabin_Show])',
            f'TREATAS({{"{cabins_str}"}}, \'Cabin_Master\'[Cabin_Show])'
        )
        
        # Replace company filters
        companies_str = '", "'.join(companies)
        template = template.replace(
            'TREATAS({"IB","YW"}, \'Company_Master\'[Company])',
            f'TREATAS({{"{companies_str}"}}, \'Company_Master\'[Company])'
        )
        
        # Replace haul filters
        hauls_str = '", "'.join(hauls)
        template = template.replace(
            'TREATAS({"SH","LH"}, \'Haul_Master\'[Haul_Aggr])',
            f'TREATAS({{"{hauls_str}"}}, \'Haul_Master\'[Haul_Aggr])'
        )
        
        # Replace current period dates
        query = template.replace('{CURRENT_START_YEAR}', str(current_start_date.year))
        query = query.replace('{CURRENT_START_MONTH}', str(current_start_date.month))
        query = query.replace('{CURRENT_START_DAY}', str(current_start_date.day))
        query = query.replace('{CURRENT_END_YEAR}', str(current_end_date.year))
        query = query.replace('{CURRENT_END_MONTH}', str(current_end_date.month))
        query = query.replace('{CURRENT_END_DAY}', str(current_end_date.day))
        
        # Replace comparison period dates
        query = query.replace('{COMPARISON_START_YEAR}', str(comparison_start_date.year))
        query = query.replace('{COMPARISON_START_MONTH}', str(comparison_start_date.month))
        query = query.replace('{COMPARISON_START_DAY}', str(comparison_start_date.day))
        query = query.replace('{COMPARISON_END_YEAR}', str(comparison_end_date.year))
        query = query.replace('{COMPARISON_END_MONTH}', str(comparison_end_date.month))
        query = query.replace('{COMPARISON_END_DAY}', str(comparison_end_date.day))
        
        print(f"  📊 Using simplified vs Sel. Period operative query")
        print(f"  📅 Current period: {current_start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')}")
        print(f"  📅 Comparison period: {comparison_start_date.strftime('%Y-%m-%d')} to {comparison_end_date.strftime('%Y-%m-%d')}")
        
        return query

    def _get_flexible_operative_query(self, aggregation_days: int, cabins: List[str], companies: List[str], hauls: List[str], analysis_date: datetime = None, comparison_start_date: datetime = None) -> str:
        """Generate DAX query for flexible operative aggregation using template - MATCHES NPS aggregation logic exactly
        
        Args:
            aggregation_days: Days per aggregation period
            cabins, companies, hauls: Node filters
            analysis_date: End date for analysis (default: TODAY())
            comparison_start_date: Start date to include comparison period data (for "vs Sel. Period")
        """
        template = self._load_query_template("Operativa_flex_agg.txt")
        
        # Replace placeholders with actual values
        cabins_str = '", "'.join(cabins)
        companies_str = '", "'.join(companies)
        hauls_str = '", "'.join(hauls)
        
        # Replace the template placeholders - replace ALL occurrences
        query = template.replace(
            '{AGGREGATION_DAYS}', str(aggregation_days)
        )
        
        # Replace ALL occurrences of cabin filters
        query = query.replace(
            'TREATAS({"Business", "Economy", "Premium EC"}, \'Cabin_Master\'[Cabin_Show])',
            f'TREATAS({{"{cabins_str}"}}, \'Cabin_Master\'[Cabin_Show])'
        )
        
        # Replace ALL occurrences of company filters
        query = query.replace(
            'TREATAS({"IB","YW"}, \'Company_Master\'[Company])',
            f'TREATAS({{"{companies_str}"}}, \'Company_Master\'[Company])'
        )
        
        # Replace ALL occurrences of haul filters
        query = query.replace(
            'TREATAS({"SH","LH"}, \'Haul_Master\'[Haul_Aggr])',
            f'TREATAS({{"{hauls_str}"}}, \'Haul_Master\'[Haul_Aggr])'
        )
        
        # Handle date filtering - extend range if comparison_start_date is provided (for "vs Sel. Period")
        if analysis_date or comparison_start_date:
            # Determine the actual start date (earliest of comparison_start_date or default)
            if comparison_start_date:
                # Use comparison_start_date as the start to include comparison period data
                start_date = comparison_start_date
                print(f"  📅 Including comparison period from: {comparison_start_date.strftime('%Y-%m-%d')}")
            else:
                # Use default start date
                start_date = datetime(2024, 1, 1)
            
            # Determine end date
            if analysis_date:
                end_date = analysis_date
                print(f"  📅 Using analysis date: {analysis_date.strftime('%Y-%m-%d')} for operative aggregation")
            else:
                # Keep TODAY() if no analysis_date provided
                new_date_filter = f"'Date_Master'[Date] >= DATE({start_date.year},{start_date.month},{start_date.day}) && 'Date_Master'[Date] <= TODAY()"
                old_date_filter = "'Date_Master'[Date] >= DATE(2024,01,01) && 'Date_Master'[Date] <= TODAY()"
                query = query.replace(old_date_filter, new_date_filter)
                print(f"  📝 Operative date filter: {old_date_filter} → {new_date_filter}")
                return query
            
            # Replace the date filter with specific start and end dates
            old_date_filter = "'Date_Master'[Date] >= DATE(2024,01,01) && 'Date_Master'[Date] <= TODAY()"
            new_date_filter = f"'Date_Master'[Date] >= DATE({start_date.year},{start_date.month},{start_date.day}) && 'Date_Master'[Date] <= DATE({end_date.year},{end_date.month},{end_date.day})"
            query = query.replace(old_date_filter, new_date_filter)
            print(f"  📝 Operative date filter: {old_date_filter} → {new_date_filter}")
            
            # Replace the period calculation to use analysis_date as reference
            # Match the exact pattern in the template (including the variable assignment)
            old_period_calc = f"INT(DATEDIFF( 'Date_Master'[Date],max('Date_Master'[Date]), DAY) / {{AGGREGATION_DAYS}}) + 1)"
            new_period_calc = f"INT(DATEDIFF( 'Date_Master'[Date],DATE({analysis_date.year},{analysis_date.month},{analysis_date.day}), DAY) / {aggregation_days}) + 1)"
            
            # Replace with the specific aggregation_days value
            old_period_calc_with_days = old_period_calc.replace("{AGGREGATION_DAYS}", str(aggregation_days))
            
            query = query.replace(old_period_calc_with_days, new_period_calc)
            print(f"  📝 Operative period calc: {old_period_calc_with_days} → {new_period_calc}")
        
        print(f"  📋 Operative aggregated query preview: {query[:200]}...")
        return query
    
    def _get_verbatims_range_query(self, cabins: List[str], companies: List[str], hauls: List[str], start_date: datetime, end_date: datetime) -> str:
        """Generate DAX query for verbatims data using date range template.
        
        NOTE: companies is always ["IB", "YW"] for LH (no distinction) or can be ["IB"] or ["YW"] for SH.
        """
        template = self._load_query_template("Verbatims.txt")
        
        # Replace placeholders with actual values
        cabins_str = '", "'.join(cabins)
        hauls_str = '", "'.join(hauls)
        companies_str = '", "'.join(companies)
        
        # Replace the template placeholders - replace ALL occurrences
        query = template.replace(
            'TREATAS({"Business", "Economy", "Premium EC"}, \'Cabin_Master\'[Cabin_Show])',
            f'TREATAS({{"{cabins_str}"}}, \'Cabin_Master\'[Cabin_Show])'
        ).replace(
            'TREATAS({"SH","LH"}, \'Haul_Master\'[Haul_Aggr])',
            f'TREATAS({{"{hauls_str}"}}, \'Haul_Master\'[Haul_Aggr])'
        ).replace(
            'TREATAS({"IB","YW"}, \'Company_Master\'[Company])',
            f'TREATAS({{"{companies_str}"}}, \'Company_Master\'[Company])'
        ).replace(
            '\'Date_Master\'[Date] =date(2025,05,12)',
            f'\'Date_Master\'[Date] >= date({start_date.year},{start_date.month},{start_date.day}) && \'Date_Master\'[Date] <= date({end_date.year},{end_date.month},{end_date.day})'
        )
        
        return query

    
    def _get_smart_verbatims_query(self, cabins: List[str], companies: List[str], hauls: List[str], start_date: datetime, end_date: datetime, anomaly_type: str = "neutral", route_filter: Optional[str] = None, touchpoint_filter: Optional[str] = None) -> str:
        """
        Generate optimized DAX query for verbatims data using Verbatims_Smart.txt template.
        Filters by NPS class based on anomaly type and limits to top 5 relevant comments.
        
        NOTE: companies is always ["IB", "YW"] for LH (no distinction) or can be ["IB"] or ["YW"] for SH.
        """
        template = self._load_query_template("Verbatims_Smart.txt")
        
        # Replace list placeholders using standard TREATAS replacement strategy
        cabins_str = '", "'.join(cabins)
        hauls_str = '", "'.join(hauls)
        companies_str = '", "'.join(companies)
        
        query = template.replace(
            'TREATAS({"Business", "Economy", "Premium EC"}, \'Cabin_Master\'[Cabin_Show])',
            f'TREATAS({{"{cabins_str}"}}, \'Cabin_Master\'[Cabin_Show])'
        ).replace(
            'TREATAS({"SH","LH"}, \'Haul_Master\'[Haul_Aggr])',
            f'TREATAS({{"{hauls_str}"}}, \'Haul_Master\'[Haul_Aggr])'
        ).replace(
            'TREATAS({"IB","YW"}, \'Company_Master\'[Company])',
            f'TREATAS({{"{companies_str}"}}, \'Company_Master\'[Company])'
        )
        
        # Replace date placeholders
        query = query.replace('__START_YEAR__', str(start_date.year))
        query = query.replace('__START_MONTH__', str(start_date.month))
        query = query.replace('__START_DAY__', str(start_date.day))
        query = query.replace('__END_YEAR__', str(end_date.year))
        query = query.replace('__END_MONTH__', str(end_date.month))
        query = query.replace('__END_DAY__', str(end_date.day))
        
        # Determine NPS condition based on anomaly type
        if anomaly_type == "positive":
            nps_condition = "'surveys_maritz'[nps_category] = \"Promoter\""
        elif anomaly_type == "negative":
            nps_condition = "'surveys_maritz'[nps_category] = \"Detractor\""
        else:
            # Neutral/All: Get all valid NPS
            nps_condition = "NOT(ISBLANK('surveys_maritz'[nps_category]))"
            
        query = query.replace('__NPS_CLASS_FILTER__', nps_condition)
        
        # Route and touchpoint filters — only injected when present (DAX doesn't accept TRUE() as table filter)
        # TouchPoint_Master[filtered_name] works because verbatims_sentiment has a relationship with TouchPoint_Master
        optional_filters = ""
        if route_filter:
            optional_filters += f',\n                TREATAS({{"{route_filter}"}}, \'surveys_maritz\'[route])'
        if touchpoint_filter:
            optional_filters += f',\n                TouchPoint_Master[filtered_name] = "{touchpoint_filter}"'
        
        query = query.replace('__OPTIONAL_FILTERS__', optional_filters)
        
        return query

    # NOTE: collect_smart_verbatims is defined later in the file (line ~646) using _get_node_filters
    # This duplicate definition was removed to avoid confusion and the missing _parse_node_filters error

    def _execute_query(self, query: str, timeout_seconds: int = None, max_retries: int = None) -> pd.DataFrame:
        """Execute a DAX query against Power BI API with retry logic
        
        Args:
            query: DAX query string
            timeout_seconds: Timeout for the API request (uses self.api_timeout if not specified)
            max_retries: Maximum retry attempts (uses PBI_MAX_RETRIES if not specified)
            
        Returns:
            DataFrame with query results or empty DataFrame on error
        """
        timeout = timeout_seconds if timeout_seconds is not None else self.api_timeout
        retries = max_retries if max_retries is not None else PBI_MAX_RETRIES
        
        dax_query = {
            "queries": [{"query": query}],
            "serializerSettings": {"includeNulls": True}
        }
        
        url = f"https://api.powerbi.com/v1.0/myorg/groups/{self.group_id}/datasets/{self.dataset_id}/executeQueries"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        last_error = None
        for attempt in range(retries + 1):
            try:
                response = requests.post(url, headers=headers, json=dax_query, timeout=timeout)
                
                # Check for retryable HTTP errors
                if response.status_code in PBI_RETRYABLE_STATUS_CODES:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    if attempt < retries:
                        delay = min(PBI_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), PBI_MAX_DELAY)
                        self.logger.warning(f"⚠️ PBI API returned {response.status_code}, retrying in {delay:.1f}s (attempt {attempt + 1}/{retries + 1})")
                        time.sleep(delay)
                        continue
                    else:
                        self.logger.error(f"❌ PBI API Error {response.status_code} after {retries + 1} attempts: {response.text}")
                        return pd.DataFrame()
                
                # Non-retryable HTTP errors (e.g., 400, 401, 403, 404)
                if response.status_code != 200:
                    self.logger.error(f"❌ PBI API Error {response.status_code}: {response.text}")
                    return pd.DataFrame()
                
                results = response.json()
                
                # Check for empty results - this is also retryable (server might be overloaded)
                if not results.get('results') or not results['results'][0].get('tables'):
                    last_error = "Empty results from PBI API"
                    if attempt < retries:
                        delay = min(PBI_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), PBI_MAX_DELAY)
                        self.logger.warning(f"⚠️ PBI returned empty results, retrying in {delay:.1f}s (attempt {attempt + 1}/{retries + 1})")
                        time.sleep(delay)
                        continue
                    else:
                        self.logger.warning(f"⚠️ No data returned from PBI query after {retries + 1} attempts")
                        return pd.DataFrame()
                
                rows = results['results'][0]['tables'][0].get('rows', [])
                
                # Also retry if we got a table but it's empty (possible transient issue)
                if len(rows) == 0:
                    last_error = "Empty rows from PBI API"
                    if attempt < retries:
                        delay = min(PBI_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), PBI_MAX_DELAY)
                        self.logger.warning(f"⚠️ PBI returned 0 rows, retrying in {delay:.1f}s (attempt {attempt + 1}/{retries + 1})")
                        time.sleep(delay)
                        continue
                
                # Success - log if we had to retry
                if attempt > 0:
                    self.logger.info(f"✅ PBI query succeeded on attempt {attempt + 1}")
                
                return pd.DataFrame(rows)
            
            except requests.exceptions.Timeout:
                last_error = f"Timeout after {timeout}s"
                if attempt < retries:
                    delay = min(PBI_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), PBI_MAX_DELAY)
                    self.logger.warning(f"⏰ PBI API timeout, retrying in {delay:.1f}s (attempt {attempt + 1}/{retries + 1})")
                    time.sleep(delay)
                    continue
                else:
                    self.logger.error(f"⏰ PBI API timeout after {retries + 1} attempts - query may be too complex or API is overloaded")
                    return pd.DataFrame()
            
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {str(e)}"
                if attempt < retries:
                    delay = min(PBI_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), PBI_MAX_DELAY)
                    self.logger.warning(f"❌ PBI connection error, retrying in {delay:.1f}s (attempt {attempt + 1}/{retries + 1})")
                    time.sleep(delay)
                    continue
                else:
                    self.logger.error(f"❌ PBI API connection error after {retries + 1} attempts: {type(e).__name__}: {str(e)}")
                    return pd.DataFrame()
            
            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                if attempt < retries:
                    delay = min(PBI_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), PBI_MAX_DELAY)
                    self.logger.warning(f"❌ PBI error, retrying in {delay:.1f}s (attempt {attempt + 1}/{retries + 1}): {last_error}")
                    time.sleep(delay)
                    continue
                else:
                    self.logger.error(f"❌ Error executing query after {retries + 1} attempts: {last_error}")
                    return pd.DataFrame()
        
        # Should not reach here, but just in case
        self.logger.error(f"❌ All {retries + 1} attempts failed. Last error: {last_error}")
        return pd.DataFrame()
    
    async def _execute_query_async(self, query: str, timeout_seconds: int = None, max_retries: int = None) -> pd.DataFrame:
        """Execute a DAX query against Power BI API asynchronously with retry logic
        
        Args:
            query: DAX query string
            timeout_seconds: Timeout for the API request (uses self.api_timeout if not specified)
            max_retries: Maximum retry attempts (uses PBI_MAX_RETRIES if not specified)
            
        Returns:
            DataFrame with query results or empty DataFrame on error
        """
        effective_timeout = timeout_seconds if timeout_seconds is not None else self.api_timeout
        retries = max_retries if max_retries is not None else PBI_MAX_RETRIES
        
        dax_query = {
            "queries": [{"query": query}],
            "serializerSettings": {"includeNulls": True}
        }
        
        url = f"https://api.powerbi.com/v1.0/myorg/groups/{self.group_id}/datasets/{self.dataset_id}/executeQueries"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        # Configure timeout for the aiohttp request
        timeout = aiohttp.ClientTimeout(total=effective_timeout)
        
        last_error = None
        for attempt in range(retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, headers=headers, json=dax_query) as response:
                        # Check for retryable HTTP errors
                        if response.status in PBI_RETRYABLE_STATUS_CODES:
                            response_text = await response.text()
                            last_error = f"HTTP {response.status}: {response_text[:200]}"
                            if attempt < retries:
                                delay = min(PBI_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), PBI_MAX_DELAY)
                                self.logger.warning(f"⚠️ PBI API returned {response.status}, retrying in {delay:.1f}s (attempt {attempt + 1}/{retries + 1})")
                                await asyncio.sleep(delay)
                                continue
                            else:
                                self.logger.error(f"❌ PBI API Error {response.status} after {retries + 1} attempts: {response_text}")
                                return pd.DataFrame()
                        
                        # Non-retryable HTTP errors
                        if response.status != 200:
                            response_text = await response.text()
                            self.logger.error(f"❌ PBI API Error {response.status}: {response_text}")
                            return pd.DataFrame()
                        
                        results = await response.json()
                        
                        # Check for empty results - this is also retryable
                        if not results.get('results') or not results['results'][0].get('tables'):
                            last_error = "Empty results from PBI API"
                            if attempt < retries:
                                delay = min(PBI_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), PBI_MAX_DELAY)
                                self.logger.warning(f"⚠️ PBI returned empty results, retrying in {delay:.1f}s (attempt {attempt + 1}/{retries + 1})")
                                await asyncio.sleep(delay)
                                continue
                            else:
                                self.logger.warning(f"⚠️ No data returned from PBI query after {retries + 1} attempts")
                                return pd.DataFrame()
                        
                        rows = results['results'][0]['tables'][0].get('rows', [])
                        
                        # Also retry if we got a table but it's empty
                        if len(rows) == 0:
                            last_error = "Empty rows from PBI API"
                            if attempt < retries:
                                delay = min(PBI_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), PBI_MAX_DELAY)
                                self.logger.warning(f"⚠️ PBI returned 0 rows, retrying in {delay:.1f}s (attempt {attempt + 1}/{retries + 1})")
                                await asyncio.sleep(delay)
                                continue
                        
                        # Success - log if we had to retry
                        if attempt > 0:
                            self.logger.info(f"✅ PBI async query succeeded on attempt {attempt + 1}")
                        
                        return pd.DataFrame(rows)
            
            except asyncio.TimeoutError:
                last_error = f"Timeout after {effective_timeout}s"
                if attempt < retries:
                    delay = min(PBI_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), PBI_MAX_DELAY)
                    self.logger.warning(f"⏰ PBI API timeout, retrying in {delay:.1f}s (attempt {attempt + 1}/{retries + 1})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    self.logger.error(f"⏰ PBI API timeout after {retries + 1} attempts - query may be too complex or API is overloaded")
                    return pd.DataFrame()
            
            except aiohttp.ClientError as e:
                last_error = f"Client error: {str(e)}"
                if attempt < retries:
                    delay = min(PBI_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), PBI_MAX_DELAY)
                    self.logger.warning(f"❌ PBI connection error, retrying in {delay:.1f}s (attempt {attempt + 1}/{retries + 1})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    self.logger.error(f"❌ PBI API connection error after {retries + 1} attempts: {type(e).__name__}: {str(e)}")
                    return pd.DataFrame()
            
            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                if attempt < retries:
                    delay = min(PBI_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), PBI_MAX_DELAY)
                    self.logger.warning(f"❌ PBI error, retrying in {delay:.1f}s (attempt {attempt + 1}/{retries + 1}): {last_error}")
                    await asyncio.sleep(delay)
                    continue
                else:
                    self.logger.error(f"❌ Error executing async query after {retries + 1} attempts: {last_error}")
                    return pd.DataFrame()
        
        # Should not reach here, but just in case
        self.logger.error(f"❌ All {retries + 1} async attempts failed. Last error: {last_error}")
        return pd.DataFrame()
    
    def _get_node_filters(self, node_path: str) -> Tuple[List[str], List[str], List[str]]:
        """Get the filter values for cabins, companies, and hauls based on node path.
        
        IMPORTANT: 
        - LH (Long Haul): Always uses BOTH companies ["IB", "YW"] - no distinction in LH
        - SH (Short Haul): Can filter by specific company (IB or YW)
        """
        path_parts = node_path.split('/')
        
        # Default to all values
        cabins = ["Business", "Economy", "Premium EC"]
        companies = ["IB", "YW"]  # Always include both - this is the default
        hauls = ["SH", "LH"]
        
        # Apply filters based on path
        if len(path_parts) >= 2 and path_parts[1] in ['LH', 'SH']:
            hauls = [path_parts[1]]
            
        if len(path_parts) >= 3 and path_parts[2] in ['Economy', 'Business', 'Premium']:
            if path_parts[2] == 'Premium':
                cabins = ["Premium EC"]
            else:
                cabins = [path_parts[2]]
                
        # Company filter: 
        # - LH: Always ["IB", "YW"] (both, no distinction)
        # - SH: Can be filtered to specific company if in path
        if 'LH' not in hauls and len(path_parts) >= 4 and path_parts[3] in ['IB', 'YW']:
            # Only filter by specific company for SH segments
            companies = [path_parts[3]]
        # For LH or SH without company in path, keep default ["IB", "YW"]
            
        return cabins, companies, hauls
    

    
    def collect_verbatims_for_date_range(self, node_path: str, start_date: datetime, end_date: datetime, output_dir: Path = None) -> pd.DataFrame:
        """Collect verbatims for a date range and segment - much more efficient than daily collection"""
        print(f"🔍 Collecting verbatims for {node_path} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get filters for this node
        cabins, companies, hauls = self._get_node_filters(node_path)
        
        # Use the date range template system for verbatims query
        query = self._get_verbatims_range_query(cabins, companies, hauls, start_date, end_date)
        
        try:
            print(f"  📝 Collecting verbatims with filters: Cabins={cabins}, Companies={companies}, Hauls={hauls}")
            df = self._execute_query(query)
            
            if not df.empty:
                print(f"  ✅ Found {len(df)} verbatims for {node_path} in date range")
                
                # Save to structured directory if output_dir is provided
                if output_dir:
                    # Create directory structure: output_dir/node_path/
                    range_str = f"{start_date.strftime('%Y_%m_%d')}_to_{end_date.strftime('%Y_%m_%d')}"
                    segment_dir = output_dir / node_path / range_str
                    segment_dir.mkdir(parents=True, exist_ok=True)
                    
                    filename = 'verbatims_range.csv'
                    filepath = segment_dir / filename
                    df.to_csv(filepath, index=False)
                    print(f"  💾 Saved to {filepath}")
                
                return df
            else:
                print(f"  ❌ No verbatims found for {node_path} in date range")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"  ❌ Error collecting verbatims for {node_path} in date range: {str(e)}")
            return pd.DataFrame()

    def collect_smart_verbatims(self, node_path: str, start_date: datetime, end_date: datetime, anomaly_type: str = "neutral", route_filter: Optional[str] = None, touchpoint_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Collect smart verbatims data from Power BI using the optimized query strategy.
        Fetches top 5 relevant comments filtered by anomaly type (promoters vs detractors).
        Optionally filters by route (IATA code) and/or touchpoint name.
        """
        filter_desc = ""
        if route_filter:
            filter_desc += f", route={route_filter}"
        if touchpoint_filter:
            filter_desc += f", touchpoint={touchpoint_filter}"
        print(f"🔍 Collecting SMART verbatims for {node_path} ({anomaly_type}{filter_desc}) from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get filters for this node
        cabins, companies, hauls = self._get_node_filters(node_path)
        
        # Generate smart query with optional route/touchpoint filters
        query = self._get_smart_verbatims_query(cabins, companies, hauls, start_date, end_date, anomaly_type, route_filter=route_filter, touchpoint_filter=touchpoint_filter)
        
        try:
            print(f"  📝 Executing smart query with filters: Cabins={cabins}, Companies={companies}, Hauls={hauls}, Type={anomaly_type}{filter_desc}")
            df = self._execute_query(query)
            
            if not df.empty:
                print(f"  ✅ Found {len(df)} smart verbatims for {node_path}")
                return df
            else:
                print(f"  ⚠️ No smart verbatims found for {node_path}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"  ❌ Error collecting smart verbatims: {e}")
            return pd.DataFrame()

    async def collect_verbatims_by_topic(
        self,
        node_path: str,
        start_date,
        end_date,
        touchpoint_display_name: str,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Collect verbatims filtered by touchpoint topic from verbatims_sentiment table.
        
        Args:
            node_path: Node path like "Global/SH/Business"
            start_date: Start date (datetime or YYYY-MM-DD str)
            end_date: End date
            touchpoint_display_name: Display name like "Cabin Crew"
            top_n: Number of verbatims to return (default 10)
            
        Returns:
            DataFrame with verbatims filtered by topic, or empty DataFrame if no data/error.
        """
        try:
            # Get topic from display name
            topic = get_touchpoint_topic(touchpoint_display_name)
            if not topic:
                self.logger.warning(
                    f"⚠️ No topic mapping found for touchpoint '{touchpoint_display_name}'"
                )
                return pd.DataFrame()
            
            # Convert dates if needed
            def _to_dt(d):
                if isinstance(d, str):
                    from datetime import datetime
                    return datetime.strptime(d, "%Y-%m-%d")
                return d
            
            sd = _to_dt(start_date)
            ed = _to_dt(end_date)
            
            # Get node filters
            cabins, companies, hauls = self._get_node_filters(node_path)
            
            # Build filter strings
            cabin_values = ", ".join(f'"{c}"' for c in cabins) if cabins else '"All"'
            company_values = ", ".join(f'"{c}"' for c in companies) if companies else '"All"'
            haul_values = ", ".join(f'"{h}"' for h in hauls) if hauls else '"All"'
            
            # Escape topic for DAX
            safe_topic = topic.replace('"', '\\"')
            
            # Build DAX query
            query = f"""
DEFINE
    VAR __DS0FilterTable =
        TREATAS({{{cabin_values}}}, 'Cabin_Master'[Cabin_Show])

    VAR __DS0FilterTable2 =
        TREATAS({{{company_values}}}, 'Company_Master'[Company])

    VAR __DS0FilterTable3 =
        TREATAS({{{haul_values}}}, 'Haul_Master'[Haul_Aggr])

    VAR __DS0FilterTable4 =
        FILTER(
            KEEPFILTERS(VALUES('Date_Master'[Date])),
            AND(
                'Date_Master'[Date] >= DATE({sd.year}, {sd.month}, {sd.day}),
                'Date_Master'[Date] <= DATE({ed.year}, {ed.month}, {ed.day})
            )
        )

    VAR __TopicFilter =
        FILTER(
            ALL(verbatims_sentiment[topic]),
            verbatims_sentiment[topic] = "{safe_topic}"
        )

    VAR __VerbatimsWithTopic =
        TOPN(
            {top_n},
            ADDCOLUMNS(
                CALCULATETABLE(
                    verbatims_sentiment,
                    __DS0FilterTable,
                    __DS0FilterTable2,
                    __DS0FilterTable3,
                    __DS0FilterTable4,
                    __TopicFilter
                ),
                "Verbatim", CALCULATE(MIN(surveys_maritz[nps_all_t])),
                "Route", CALCULATE(MIN(surveys_maritz[route])),
                "NPS_Score", CALCULATE(MIN(surveys_maritz[nps_100])),
                "NPS_Category", CALCULATE(MIN(surveys_maritz[nps_category])),
                "Date", CALCULATE(MIN(surveys_maritz[date_flight_local]))
            ),
            LEN([Verbatim]), DESC
        )

EVALUATE
    __VerbatimsWithTopic
"""
            
            self.logger.info(
                f"🔍 Collecting verbatims by topic for '{touchpoint_display_name}' "
                f"(topic='{topic}') on '{node_path}'"
            )
            
            df = await self._execute_query_async(query)
            
            if df.empty:
                self.logger.warning(
                    f"⚠️ No verbatims found for topic '{topic}' on '{node_path}'"
                )
                return pd.DataFrame()
            
            df = self._safe_clean_columns(df)
            self.logger.info(
                f"✅ Found {len(df)} verbatims for topic '{topic}' on '{node_path}'"
            )
            
            return df
            
        except Exception as e:
            self.logger.warning(
                f"⚠️ Error collecting verbatims by topic for '{touchpoint_display_name}' "
                f"on '{node_path}': {e}"
            )
            return pd.DataFrame()
    


    async def collect_flexible_data_for_node(self, node_path: str, aggregation_days: int, target_folder: str, analysis_date: datetime = None) -> Dict[str, bool]:
        """Collect flexible aggregated data for a specific node"""
        results = {}
        
        # Parse node path to get filters
        cabins, companies, hauls = self._parse_node_path(node_path)
        
        # Create node directory
        node_dir = Path(target_folder) / node_path.replace('/', '_')
        node_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Collecting flexible data for node: {node_path}")
        print(f"  Aggregation: {aggregation_days} days")
        if analysis_date:
            print(f"  Analysis date: {analysis_date.strftime('%Y-%m-%d')}")
        print(f"  Filters - Cabins: {cabins}, Companies: {companies}, Hauls: {hauls}")
        
        # Collect flexible NPS data
        try:
            query = self._get_flexible_nps_query(aggregation_days, cabins, companies, hauls, analysis_date)
            df = await self._execute_query_async(query)
            if not df.empty:
                # Clean column names safely
                df = self._safe_clean_columns(df)
                df.to_csv(node_dir / f'flexible_NPS_{aggregation_days}d.csv', index=False)
                results['flexible_NPS'] = True
                print(f"  ✓ flexible_NPS_{aggregation_days}d.csv saved ({len(df)} periods)")
            else:
                results['flexible_NPS'] = False
                print(f"  ✗ flexible_NPS_{aggregation_days}d.csv - no data")
        except Exception as e:
            results['flexible_NPS'] = False
            print(f"  ✗ flexible_NPS_{aggregation_days}d.csv - error: {str(e)}")
        
        # Collect operative data - ALWAYS use flexible query with appropriate aggregation_days
        try:
            # Use flexible aggregation for ALL cases (daily = 1 day, weekly = 7 days, etc.)
            query = self._get_flexible_operative_query(aggregation_days, cabins, companies, hauls, analysis_date)
            filename = f'flexible_operative_{aggregation_days}d.csv'
            
            if aggregation_days == 1:
                print(f"  📊 Using flexible operative query (1d = daily periods) - ✅ ALIGNED WITH NPS")
            else:
                print(f"  📊 Using flexible operative query ({aggregation_days}d periods) - ✅ ALIGNED WITH NPS")
            
            df = await self._execute_query_async(query)
            if not df.empty:
                # Clean column names safely
                df = self._safe_clean_columns(df)
                df.to_csv(node_dir / filename, index=False)
                
                results['flexible_operative'] = True
                if aggregation_days == 1:
                    print(f"  ✓ {filename} saved ({len(df)} daily periods) - ✅ SAME LOGIC AS NPS")
                else:
                    print(f"  ✓ {filename} saved ({len(df)} periods) - ✅ SAME LOGIC AS NPS")
            else:
                results['flexible_operative'] = False
                print(f"  ✗ {filename} - no data")
        except Exception as e:
            results['flexible_operative'] = False
            print(f"  ✗ flexible_operative_{aggregation_days}d.csv - error: {str(e)}")
        
        return results

    def _clean_routes_dictionary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean column names specifically for routes dictionary
        Removes Power BI table prefixes like 'Route_Master[column_name]' -> 'column_name'
        """
        if df.empty:
            return df
            
        cleaned_columns = []
        for col in df.columns:
            if col is not None and isinstance(col, str):
                # Remove Power BI table prefixes: 'Route_Master[column_name]' -> 'column_name'
                if '[' in col and ']' in col:
                    # Extract the part between brackets
                    start_bracket = col.find('[')
                    end_bracket = col.find(']')
                    if start_bracket != -1 and end_bracket != -1:
                        cleaned_col = col[start_bracket + 1:end_bracket]
                        cleaned_columns.append(cleaned_col)
                    else:
                        cleaned_columns.append(col)
                else:
                    cleaned_columns.append(col)
            else:
                # Log problematic columns and use a safe default
                self.logger.warning(f"⚠️ Found None/invalid column name: {col}, using default")
                cleaned_columns.append(f"Column_{len(cleaned_columns)}")
        
        df.columns = cleaned_columns
        return df

    def _safe_clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Safely clean column names by removing brackets, handling None columns
        """
        if df.empty:
            return df
            
        cleaned_columns = []
        for col in df.columns:
            if col is not None and isinstance(col, str):
                cleaned_columns.append(col.strip('[]'))
            else:
                # Log problematic columns and use a safe default
                print(f"         ⚠️ Found None/invalid column name: {col}, using default")
                cleaned_columns.append(f"Column_{len(cleaned_columns)}")
        df.columns = cleaned_columns
        return df

    def _parse_node_path(self, node_path: str) -> Tuple[List[str], List[str], List[str]]:
        """Parse node path to extract cabins, companies, and hauls.
        
        IMPORTANT: 
        - LH (Long Haul): Always uses BOTH companies ["IB", "YW"] - no distinction in LH
        - SH (Short Haul): Can filter by specific company (IB or YW)
        """
        
        # Default values for Global
        cabins = ['Business', 'Economy', 'Premium EC']
        companies = ['IB', 'YW']  # Always include both by default
        hauls = ['SH', 'LH']
        
        # Parse path segments
        segments = node_path.split('/')
        
        # Extract haul information
        if 'LH' in segments:
            hauls = ['LH']
            # LH: Keep companies = ['IB', 'YW'] - no distinction, but need both
        elif 'SH' in segments:
            hauls = ['SH']
        
        # Extract cabin information
        if 'Economy' in segments:
            cabins = ['Economy']
        elif 'Business' in segments:
            cabins = ['Business']
        elif 'Premium' in segments:
            cabins = ['Premium EC']
        
        # Extract company information - only filter specific company for SH
        if 'SH' in hauls:  # Only apply specific company filter for Short Haul
            if 'IB' in segments:
                companies = ['IB']
            elif 'YW' in segments:
                companies = ['YW']
        
        return cabins, companies, hauls

    async def collect_operative_data_for_date(self, node_path: str, target_date: datetime, comparison_days: int = 7, use_flexible: bool = True) -> pd.DataFrame:
        """
        Collect operational data for a specific date and the preceding days for comparison
        
        Args:
            node_path: Node path like "Global/LH/Business"
            target_date: The specific date we're analyzing (e.g., 2025-01-20)
            comparison_days: Number of days to include for comparison (default: 7)
            use_flexible: Whether to use flexible aggregated query (default: True)
            
        Returns:
            DataFrame with operational data including target date and previous comparison_days-1 dates
        """
        try:
            # Get filters for this node
            cabins, companies, hauls = self._get_node_filters(node_path)
            
            # Generate the appropriate operative query
            if use_flexible:
                # Use flexible aggregated query that matches NPS logic
                aggregation_days = comparison_days if comparison_days > 1 else 1
                query = self._get_flexible_operative_query(aggregation_days, cabins, companies, hauls, target_date)
                print(f"📊 Using FLEXIBLE operative query ({aggregation_days}d) for agent analysis - ✅ ALIGNED WITH NPS")
            else:
                # Use legacy daily query
                query = self._get_operative_query(cabins, companies, hauls, target_date, comparison_days)
                print(f"📁 Using LEGACY operative query for backward compatibility")
            
            # Execute the query
            df = self._execute_query(query)
            
            if not df.empty:
                # Clean column names
                df.columns = [col.strip('[]') for col in df.columns]
                
                # Convert date column to datetime
                if 'Date_Master[Date' in df.columns:
                    df.rename(columns={'Date_Master[Date': 'Date_Master'}, inplace=True)
                
                if 'Date_Master' in df.columns:
                    df['Date_Master'] = pd.to_datetime(df['Date_Master']).dt.date
                
                print(f"         ✅ Collected {len(df)} days of operational data for analysis")
                
                # Show date range for verification
                if not df.empty and 'Date_Master' in df.columns:
                    min_date = df['Date_Master'].min()
                    max_date = df['Date_Master'].max()
                    print(f"         📅 Date range: {min_date} to {max_date}")
            
            return df
            
        except Exception as e:
            print(f"         ❌ Error collecting operational data: {str(e)}")
            return pd.DataFrame()

    async def collect_explanatory_drivers_for_date_range(self, node_path: str, start_date: datetime, end_date: datetime, comparison_filter: str = "vs L7d", comparison_start_date: datetime = None, comparison_end_date: datetime = None) -> pd.DataFrame:
        """
        Collect explanatory drivers data for a specific node and date range with configurable comparison filter
        
        Args:
            node_path: Node path like "Global/LH/Business"
            start_date: Start date for the range
            end_date: End date for the range
            comparison_filter: Comparison filter to use (vs L7d, vsL14d, vs LM, vs LY, vs Target, vs Sel. Period)
            comparison_start_date: Start date for comparison period (required when comparison_filter is "vs Sel. Period")
            comparison_end_date: End date for comparison period (required when comparison_filter is "vs Sel. Period")
            
        Returns:
            DataFrame with explanatory drivers data
        """
        try:
            # Get filters for this node
            cabins, companies, hauls = self._get_node_filters(node_path)
            
            # Generate the explanatory drivers query for the date range
            query = self._get_explanatory_drivers_range_query(
                start_date, end_date, comparison_filter, 
                comparison_start_date, comparison_end_date,
                cabins, companies, hauls
            )
            
            # Execute the query
            df = self._execute_query(query)
            
            if not df.empty:
                # Clean column names safely
                df = self._safe_clean_columns(df)
                print(f"\t\t 🔍 DEBUG: PBI collector received comparison_filter: '{comparison_filter}'")
                print(f"\t\t ✅ Collected {len(df)} explanatory drivers for analysis (filter: {comparison_filter})")
            
            return df
            
        except Exception as e:
            print(f"\t\t ❌ Error collecting explanatory drivers: {str(e)}")
            return pd.DataFrame()

    async def collect_routes_for_date_range(self, node_path: str, start_date: datetime, end_date: datetime, comparison_filter: str = "vs L7d", comparison_start_date: datetime = None, comparison_end_date: datetime = None) -> pd.DataFrame:
        """
        Collect routes data for a specific node and date range with configurable comparison filter
        
        Args:
            node_path: Node path like "Global/LH/Business"
            start_date: Start date for the range
            end_date: End date for the range
            comparison_filter: Comparison filter to use (vs L7d, vsL14d, vs LM, vs LY, vs Target, vs Sel. Period)
            
        Returns:
            DataFrame with routes data
        """
        try:
            # Get filters for this node
            cabins, companies, hauls = self._get_node_filters(node_path)
            
            # Generate the routes query for the date range
            query = self._get_routes_range_query(cabins, companies, hauls, start_date, end_date, comparison_filter, comparison_start_date, comparison_end_date)
            
            # Execute the query
            df = self._execute_query(query)
            
            if not df.empty:
                # Clean column names safely
                df = self._safe_clean_columns(df)
                print(f"         ✅ Collected {len(df)} routes for analysis (filter: {comparison_filter})")
            
            return df
            
        except Exception as e:
            print(f"         ❌ Error collecting routes data: {str(e)}")
            return pd.DataFrame()

    def _get_explanatory_drivers_range_query(self, start_date: datetime, end_date: datetime, comparison_filter: str = "vs L7d", comparison_start_date: datetime = None, comparison_end_date: datetime = None, cabins: List[str] = None, companies: List[str] = None, hauls: List[str] = None) -> str:
        """Build explanatory drivers DAX query for the given date range and comparison filter."""
        try:
            # Read the template
            template_path = os.path.join(self.queries_path, "Exp. Drivers.txt")
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            # Replace analysis period placeholders (cross-platform date format without leading zeros)
            start_date_str = f"{start_date.year}, {start_date.month}, {start_date.day}"
            end_date_str = f"{end_date.year}, {end_date.month}, {end_date.day}"
            query = template.replace('__START_DATE__', start_date_str)
            query = query.replace('__END_DATE__', end_date_str)
            
            # Replace segment filters
            cabin_filter = "{" + ", ".join([f'"{c}"' for c in cabins]) + "}" if cabins else ""
            company_filter = "{" + ", ".join([f'"{c}"' for c in companies]) + "}" if companies else ""
            haul_filter = "{" + ", ".join([f'"{h}"' for h in hauls]) + "}" if hauls else ""

            query = query.replace('__CABIN_FILTER__', cabin_filter)
            query = query.replace('__COMPANY_FILTER__', company_filter)
            query = query.replace('__HAUL_FILTER__', haul_filter)
            
            # Replace comparison filter placeholder
            query = query.replace('__COMPARISON_FILTER__', comparison_filter or "")

            # Handle "vs Sel. Period" dynamic filter block
            if comparison_filter == "vs Sel. Period" and comparison_start_date and comparison_end_date:
                comp_start_year, comp_start_month, comp_start_day = comparison_start_date.year, comparison_start_date.month, comparison_start_date.day
                comp_end_year, comp_end_month, comp_end_day = comparison_end_date.year, comparison_end_date.month, comparison_end_date.day
                
                # Build the selected period VAR block
                selected_period_block = (
                    "\n    VAR __DS0FilterTableSelPeriod =\n"
                    "        FILTER(\n"
                    "            KEEPFILTERS(VALUES('Aux_Date_Master_Selected_Period'[Date_aux])),\n"
                    "            AND(\n"
                    f"                'Aux_Date_Master_Selected_Period'[Date_aux] >= DATE({comp_start_year}, {comp_start_month}, {comp_start_day}),\n"
                    f"                'Aux_Date_Master_Selected_Period'[Date_aux] < DATE({comp_end_year}, {comp_end_month}, {comp_end_day})\n"
                    "            ))\n"
                )
                # Insert before VAR __DS0Core definition
                query = query.replace("\n    VAR __DS0Core =", selected_period_block + "\n    VAR __DS0Core =")
                # Add the filter table in SUMMARIZECOLUMNS argument list right after __DS0FilterTable7,
                query = query.replace("__DS0FilterTable7,\n", "__DS0FilterTable7,\n            __DS0FilterTableSelPeriod,\n")
        
            return query
            
        except Exception as e:
            print(f"\t\t ❌ Error getting explanatory drivers range query: {e}")
            return ""

    def _get_routes_range_query(self, cabins: List[str], companies: List[str], hauls: List[str], 
                               start_date: datetime, end_date: datetime, comparison_filter: str = "vs L7d", comparison_start_date: datetime = None, comparison_end_date: datetime = None) -> str:
        """
        Generate DAX query for routes data using date range with configurable comparison filter
        """
        # Load the routes template
        template = self._load_query_template("Rutas.txt")
        
        # Replace placeholders with actual values
        cabins_str = '", "'.join(cabins)
        companies_str = '", "'.join(companies)
        hauls_str = '", "'.join(hauls)
        
        # Replace the template placeholders
        query = template.replace(
            'TREATAS({"Business", "Economy", "Premium EC"}, \'Cabin_Master\'[Cabin_Show])',
            f'TREATAS({{"{cabins_str}"}}, \'Cabin_Master\'[Cabin_Show])'
        ).replace(
            'TREATAS({"IB","YW"}, \'Company_Master\'[Company])',
            f'TREATAS({{"{companies_str}"}}, \'Company_Master\'[Company])'
        ).replace(
            'TREATAS({"SH","LH"}, \'Haul_Master\'[Haul_Aggr])',
            f'TREATAS({{"{hauls_str}"}}, \'Haul_Master\'[Haul_Aggr])'
        ).replace(
            '__COMPARISON_FILTER__',
            comparison_filter or ""
        ).replace(
            '__START_YEAR__', str(start_date.year)
        ).replace(
            '__START_MONTH__', str(start_date.month)
        ).replace(
            '__START_DAY__', str(start_date.day)
        ).replace(
            '__END_YEAR__', str(end_date.year)
        ).replace(
            '__END_MONTH__', str(end_date.month)
        ).replace(
            '__END_DAY__', str(end_date.day)
        )
        
        # Handle "vs Sel. Period" dynamic filter block (using same simple logic as exp_drivers_tool)
        if comparison_filter == "vs Sel. Period" and comparison_start_date and comparison_end_date:
            # Convert to datetime if they are strings
            if isinstance(comparison_start_date, str):
                comparison_start_date = datetime.strptime(comparison_start_date, '%Y-%m-%d')
            if isinstance(comparison_end_date, str):
                comparison_end_date = datetime.strptime(comparison_end_date, '%Y-%m-%d')
                
            comp_start_year, comp_start_month, comp_start_day = comparison_start_date.year, comparison_start_date.month, comparison_start_date.day
            comp_end_year, comp_end_month, comp_end_day = comparison_end_date.year, comparison_end_date.month, comparison_end_date.day
            
            # Build the selected period VAR block (same as exp_drivers_tool)
            selected_period_block = (
                f"VAR __DS0FilterTableSelPeriod =\n"
                f"    FILTER(\n"
                f"        KEEPFILTERS(VALUES('Aux_Date_Master_Selected_Period'[Date_aux])),\n"
                f"        AND(\n"
                f"            'Aux_Date_Master_Selected_Period'[Date_aux] >= DATE({comp_start_year}, {comp_start_month}, {comp_start_day}),\n"
                f"            'Aux_Date_Master_Selected_Period'[Date_aux] < DATE({comp_end_year}, {comp_end_month}, {comp_end_day})\n"
                f"        ))\n"
            )
            
            # Check if template already has the variable defined
            if '__DS0FilterTableSelPeriod' not in query:
                # Template doesn't have the variable, add it after DEFINE
                query = query.replace("DEFINE", f"DEFINE\n{selected_period_block}")
        else:
            # For all other comparison filters, remove any existing __DS0FilterTableSelPeriod references
            import re
            query = re.sub(
                r'\s*VAR __DS0FilterTableSelPeriod =.*?(?=\s*VAR|\s*EVALUATE)',
                '\n\n', query, flags=re.DOTALL
            )
            # Clean up any remaining fragments and references
            query = re.sub(
                r',\s*AND\(\s*\'Aux_Date_Master_Selected_Period\'.*?\)\)',
                '', query, flags=re.DOTALL
            )
            # Remove references to __DS0FilterTableSelPeriod in SUMMARIZECOLUMNS more precisely
            # Handle the case where it's the last filter before measures (with comma after)
            query = re.sub(
                r',\s*__DS0FilterTableSelPeriod,\s*"',
                ',\n                    "', query
            )
            # Handle other cases
            query = re.sub(
                r',?\s*__DS0FilterTableSelPeriod,?',
                '', query
            )
        
        # Replace comparison date placeholders if they exist
        if comparison_start_date and comparison_end_date:
            query = query.replace(
                '__START_YEAR__', str(comparison_start_date.year)
            ).replace(
                '__START_MONTH__', str(comparison_start_date.month)
            ).replace(
                '__START_DAY__', str(comparison_start_date.day)
            ).replace(
                '__END_YEAR__', str(comparison_end_date.year)
            ).replace(
                '__END_MONTH__', str(comparison_end_date.month)
            ).replace(
                '__END_DAY__', str(comparison_end_date.day)
            )
        else:
            # Use default values for comparison placeholders to avoid DAX errors
            query = query.replace(
                '__START_YEAR__', '2024'
            ).replace(
                '__START_MONTH__', '1'
            ).replace(
                '__START_DAY__', '1'
            ).replace(
                '__END_YEAR__', '2024'
            ).replace(
                '__END_MONTH__', '1'
            ).replace(
                '__END_DAY__', '2'
            )
        
        # Handle "vs Sel. Period" dynamic filter block (exactly like exp_drivers_tool)
        if comparison_filter == "vs Sel. Period" and comparison_start_date and comparison_end_date:
            # Convert to datetime if they are strings
            if isinstance(comparison_start_date, str):
                comparison_start_date = datetime.strptime(comparison_start_date, '%Y-%m-%d')
            if isinstance(comparison_end_date, str):
                comparison_end_date = datetime.strptime(comparison_end_date, '%Y-%m-%d')
                
            comp_start_year, comp_start_month, comp_start_day = comparison_start_date.year, comparison_start_date.month, comparison_start_date.day
            comp_end_year, comp_end_month, comp_end_day = comparison_end_date.year, comparison_end_date.month, comparison_end_date.day
            
            # Check if template already has the variable defined
            if '__DS0FilterTableSelPeriod' not in query:
                # Template doesn't have the variable, add it (same as exp_drivers_tool)
                selected_period_block = (
                    "\n    VAR __DS0FilterTableSelPeriod =\n"
                    "        FILTER(\n"
                    "            KEEPFILTERS(VALUES('Aux_Date_Master_Selected_Period'[Date_aux])),\n"
                    "            AND(\n"
                    f"                'Aux_Date_Master_Selected_Period'[Date_aux] >= DATE({comp_start_year}, {comp_start_month}, {comp_start_day}),\n"
                    f"                'Aux_Date_Master_Selected_Period'[Date_aux] < DATE({comp_end_year}, {comp_end_month}, {comp_end_day})\n"
                    "            ))\n"
                )
                # Insert before VAR __DS0Core definition
                query = query.replace("\n    VAR __DS0Core =", selected_period_block + "\n    VAR __DS0Core =")
                # Add the filter table in SUMMARIZECOLUMNS argument list right after __DS0FilterTable7,
                query = query.replace("__DS0FilterTable7,\n", "__DS0FilterTable7,\n            __DS0FilterTableSelPeriod,\n")
            # If template already has the variable, the dates should already be handled by the placeholders
        
        return query

    def _get_customer_profile_range_query(self, cabins: List[str], companies: List[str], hauls: List[str], 
                                        start_date: datetime, end_date: datetime, profile_dimension: str = "Channel", 
                                        route_filter: List[str] = None, comparison_filter: str = "vs L7d", comparison_start_date: datetime = None, comparison_end_date: datetime = None) -> str:
        """Generate DAX query for customer profile data using date range and optional route filter"""
        template = self._load_query_template("Customer Profile.txt")
        
        # Replace placeholders with actual values
        cabins_str = '", "'.join(cabins)
        companies_str = '", "'.join(companies)
        hauls_str = '", "'.join(hauls)
        
        # Convert dates to datetime if they are strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Replace the template placeholders for basic filters
        query = template.replace(
            '__CABINS__', cabins_str
        ).replace(
            '__COMPANIES__', companies_str
        ).replace(
            '__HAULS__', hauls_str
        ).replace(
            '__DIMENSION_NAME__', profile_dimension
        ).replace(
            '__COMPARISON_FILTER__', comparison_filter or ""
        )
        
        # Replace analysis period placeholders (cross-platform date format without leading zeros)
        start_date_str = f"{start_date.year}, {start_date.month}, {start_date.day}"
        end_date_str = f"{end_date.year}, {end_date.month}, {end_date.day}"
        query = query.replace('__START_DATE__', start_date_str)
        query = query.replace('__END_DATE__', end_date_str)
        
        # Replace comparison date placeholders for the selected period filter
        if comparison_filter == "vs Sel. Period" and comparison_start_date and comparison_end_date:
            # Convert to datetime if they are strings
            if isinstance(comparison_start_date, str):
                comparison_start_date = datetime.strptime(comparison_start_date, '%Y-%m-%d')
            if isinstance(comparison_end_date, str):
                comparison_end_date = datetime.strptime(comparison_end_date, '%Y-%m-%d')
                
            query = query.replace(
                '__START_YEAR__', str(comparison_start_date.year)
            ).replace(
                '__START_MONTH__', str(comparison_start_date.month)
            ).replace(
                '__START_DAY__', str(comparison_start_date.day)
            ).replace(
                '__END_YEAR__', str(comparison_end_date.year)
            ).replace(
                '__END_MONTH__', str(comparison_end_date.month)
            ).replace(
                '__END_DAY__', str(comparison_end_date.day)
            )
        else:
            # Use default values for comparison placeholders to avoid DAX errors
            query = query.replace(
                '__START_YEAR__', '2024'
            ).replace(
                '__START_MONTH__', '1'
            ).replace(
                '__START_DAY__', '1'
            ).replace(
                '__END_YEAR__', '2024'
            ).replace(
                '__END_MONTH__', '1'
            ).replace(
                '__END_DAY__', '2'
            )
        if comparison_filter != "vs Sel. Period":
            # For all other comparison filters, remove any existing __DS0FilterTableSelPeriod references
            import re
            query = re.sub(
                r'\s*VAR __DS0FilterTableSelPeriod =.*?(?=\s*VAR|\s*EVALUATE)',
                '\n\n', query, flags=re.DOTALL
            )
            # Remove references to __DS0FilterTableSelPeriod in SUMMARIZECOLUMNS and CALCULATETABLE
            query = re.sub(
                r',\s*__DS0FilterTableSelPeriod',
                '', query
            )
        
        # Add route filter if specified
        if route_filter and len(route_filter) > 0:
            routes_str = '", "'.join(route_filter)
            # Add route filter to the query - need to add this as an additional filter table
            route_filter_def = f'''
    VAR __DS0FilterTable_Routes =
        TREATAS({{"{routes_str}"}}, 'Route_Master'[route])
'''
            
            # Insert the route filter definition after the existing filter definitions
            query = query.replace(
                'VAR __DS0FilterTable8 =',
                route_filter_def + '    VAR __DS0FilterTable8 ='
            )
            
            # Add the route filter to all CALCULATETABLE calls
            query = query.replace(
                '__DS0FilterTable9,',
                '__DS0FilterTable9,\n                    __DS0FilterTable_Routes,'
            )
        
        # DEBUG: Log the final query
        self.logger.info(f"🔍 DEBUG CUSTOMER_PROFILE QUERY: Generated query length: {len(query)}")
        self.logger.info(f"🔍 DEBUG CUSTOMER_PROFILE QUERY PREVIEW: {query[:500]}...")
        if len(query) > 500:
            self.logger.info(f"🔍 DEBUG CUSTOMER_PROFILE QUERY END: ...{query[-200:]}")
        
        return query

    async def collect_routes_dictionary(self) -> pd.DataFrame:
        """
        Colecta el diccionario simple de rutas para filtrado NCS
        
        Returns:
            DataFrame con columnas: route, country_name, gr_region, haul_aggr
        """
        try:
            self.logger.info("🗺️ Collecting routes dictionary for NCS filtering...")
            
            # Load the simple routes dictionary template
            template = self._load_query_template("Rutas Diccionario.txt")
            
            # Execute query without any filters - we want the complete dictionary
            result = await self._execute_query_async(template)
            
            if result is not None and not result.empty:
                # Clean column names specifically for routes dictionary
                result = self._clean_routes_dictionary_columns(result)
                
                self.logger.info(f"✅ Collected routes dictionary with {len(result)} routes")
                self.logger.debug(f"Routes dictionary columns: {list(result.columns)}")
                
                # Log some sample routes for debugging
                if len(result) > 0:
                    sample_routes = result.head(3)
                    for _, route in sample_routes.iterrows():
                        route_info = route.get('route', 'N/A')
                        haul_info = route.get('haul_aggr', 'N/A')
                        self.logger.debug(f"Sample route: {route_info} -> {haul_info}")
                
                return result
            else:
                self.logger.warning("❌ Routes dictionary query returned empty result")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"❌ Error collecting routes dictionary: {e}")
            return pd.DataFrame()

    async def collect_customer_profile_for_date_range(
        self, 
        node_path: str, 
        start_date: datetime, 
        end_date: datetime, 
        profile_dimension: str = "Channel",
        comparison_filter: str = "vs L7d",
        comparison_start_date: datetime = None,
        comparison_end_date: datetime = None,
        route_filter: List[str] = None
    ) -> pd.DataFrame:
        """
        Collect customer profile data for a specific date range and dimension
        
        Args:
            node_path: Node path for filtering
            start_date: Start date for analysis
            end_date: End date for analysis
            profile_dimension: Dimension to analyze (e.g., "Channel", "Business/Leisure", "Fleet")
            comparison_filter: Comparison filter (e.g., "vs L7d", "vs LM")
            comparison_start_date: Start date for comparison period
            comparison_end_date: End date for comparison period
            route_filter: Optional list of routes to filter by
            
        Returns:
            DataFrame with customer profile data
        """
        try:
            # Get filters for this node
            cabins, companies, hauls = self._get_node_filters(node_path)
            
            # Generate the customer profile query (same pattern as exp_drivers_tool)
            self.logger.info(f"🔍 DEBUG CUSTOMER_PROFILE: About to call _get_customer_profile_range_query")
            query = self._get_customer_profile_range_query(
                cabins, companies, hauls, start_date, end_date, profile_dimension,
                route_filter, comparison_filter, comparison_start_date, comparison_end_date
            )
            self.logger.info(f"🔍 DEBUG CUSTOMER_PROFILE: Query generated, length: {len(query) if query else 0}")
            
            # Debug log
            self.logger.info(f"🔍 DEBUG CUSTOMER_PROFILE: dimension={profile_dimension}, cabins={cabins}, companies={companies}, hauls={hauls}")
            self.logger.info(f"🔍 DEBUG CUSTOMER_PROFILE: start_date={start_date}, end_date={end_date}, comparison_filter={comparison_filter}")
            
            # Execute the query
            df = await self._execute_query_async(query)
            
            if not df.empty:
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"❌ Error collecting customer profile data: {e}")
            return pd.DataFrame()

    def _get_focus_touchpoint_csat_query(
        self,
        cabins: List[str],
        companies: List[str],
        hauls: List[str],
        start_date,
        end_date,
        touchpoint_name: str,
        comparison_filter: Optional[str] = None,
        comparison_start_date=None,
        comparison_end_date=None,
    ) -> str:
        """Build DAX query for focus touchpoint CSAT vs target.

        Uses TREATAS for segment filters and FILTER(KEEPFILTERS(VALUES(...)))
        for dates, matching the pattern from Exp. Drivers / Rutas queries
        so that [Monthly_Satisfaction] and [Target_Satisfaction_filtered]
        correctly respond to cabin/haul/company filter context.
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        start_date_str = f"{start_date.year}, {start_date.month}, {start_date.day}"
        end_date_str = f"{end_date.year}, {end_date.month}, {end_date.day}"

        safe_touchpoint = touchpoint_name.replace('"', '""')

        def to_dax_set(items):
            return "{" + ", ".join([f'"{i}"' for i in items]) + "}"

        cabin_set = to_dax_set(cabins)
        company_set = to_dax_set(companies)
        haul_set = to_dax_set(hauls)

        comp_filter = comparison_filter or "vs L7d"

        query = (
            "DEFINE\n"
            f"    VAR __DS0FilterTable  = TREATAS({cabin_set}, 'Cabin_Master'[Cabin_Show])\n"
            f"    VAR __DS0FilterTable2 = TREATAS({company_set}, 'Company_Master'[Company])\n"
            f"    VAR __DS0FilterTable3 = TREATAS({haul_set}, 'Haul_Master'[Haul_Aggr])\n"
            "    VAR __DS0FilterTable4 =\n"
            "        FILTER(\n"
            "            KEEPFILTERS(VALUES('Date_Master'[Date])),\n"
            "            AND(\n"
            f"                'Date_Master'[Date] >= DATE({start_date_str}),\n"
            f"                'Date_Master'[Date] <= DATE({end_date_str})\n"
            "            ))\n"
            "    VAR __DS0FilterTable5 = TREATAS({1}, 'TouchPoint_Master'[explanatory_drivers])\n"
            f"    VAR __DS0FilterTable6 = TREATAS({{\"{safe_touchpoint}\"}}, 'TouchPoint_Master'[filtered_name])\n"
            f"    VAR __DS0FilterTable7 = TREATAS({{\"{comp_filter}\"}}, 'Filtro_Comparativa'[Filtro_Comparativa])\n"
            "    VAR __DS0Core =\n"
            "        SUMMARIZECOLUMNS(\n"
            "            'TouchPoint_Master'[filtered_name],\n"
            "            __DS0FilterTable,\n"
            "            __DS0FilterTable2,\n"
            "            __DS0FilterTable3,\n"
            "            __DS0FilterTable4,\n"
            "            __DS0FilterTable5,\n"
            "            __DS0FilterTable6,\n"
            "            __DS0FilterTable7,\n"
            '            "CSAT", [Monthly_Satisfaction],\n'
            '            "Target_CSAT", [Target_Satisfaction_filtered]\n'
            "        )\n"
            "EVALUATE __DS0Core\n"
        )

        return query

    async def collect_focus_touchpoint_csat_vs_target(
        self,
        node_path: str,
        start_date,
        end_date,
        touchpoint_name: str,
        comparison_filter: Optional[str] = None,
        comparison_start_date=None,
        comparison_end_date=None,
    ) -> Optional[Dict[str, float]]:
        """
        Collect CSAT, Target_Satisfaction_filtered, and Satisfaction_Diff for a specific touchpoint.

        Based on Exp. Drivers query but without the explanatory_drivers=1 filter,
        filtering instead by filtered_name = touchpoint_name.

        Args:
            node_path: Node path like "Global/LH/Business"
            start_date: Start date (datetime or str YYYY-MM-DD)
            end_date: End date (datetime or str YYYY-MM-DD)
            touchpoint_name: filtered_name of the touchpoint to query
            comparison_filter: Comparison filter string (e.g. "vs L7d"). None for single mode.
            comparison_start_date: Start date for comparison period (optional)
            comparison_end_date: End date for comparison period (optional)

        Returns:
            Dict with keys: 'csat', 'target', 'gap', 'satisfaction_diff'
            or None if no data or error.
        """
        try:
            cabins, companies, hauls = self._get_node_filters(node_path)

            query = self._get_focus_touchpoint_csat_query(
                cabins=cabins,
                companies=companies,
                hauls=hauls,
                start_date=start_date,
                end_date=end_date,
                touchpoint_name=touchpoint_name,
                comparison_filter=comparison_filter,
                comparison_start_date=comparison_start_date,
                comparison_end_date=comparison_end_date,
            )

            df = await self._execute_query_async(query)

            if df.empty:
                self.logger.warning(
                    f"⚠️ No data returned for focus touchpoint '{touchpoint_name}' "
                    f"on node '{node_path}'"
                )
                return None

            df = self._safe_clean_columns(df)

            # Locate the first row (there should be exactly one for the specific touchpoint)
            row = df.iloc[0]

            def _to_float(val) -> Optional[float]:
                """Convert a value to float, returning None if not possible."""
                try:
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        return None
                    return float(val)
                except (TypeError, ValueError):
                    return None

            csat = _to_float(row.get("CSAT"))
            target = _to_float(row.get("Target_CSAT"))
            
            satisfaction_diff = None
            if comparison_filter and comparison_start_date and comparison_end_date:
                safe_touchpoint = touchpoint_name.replace('"', '""')

                csd = comparison_start_date if isinstance(comparison_start_date, datetime) else datetime.strptime(comparison_start_date, '%Y-%m-%d')
                ced = comparison_end_date if isinstance(comparison_end_date, datetime) else datetime.strptime(comparison_end_date, '%Y-%m-%d')

                comp_start_str = f"{csd.year}, {csd.month}, {csd.day}"
                comp_end_str = f"{ced.year}, {ced.month}, {ced.day}"

                def to_dax_set(items):
                    return "{" + ", ".join([f'"{i}"' for i in items]) + "}"

                cabin_set = to_dax_set(cabins)
                company_set = to_dax_set(companies)
                haul_set = to_dax_set(hauls)

                comparison_query = (
                    "DEFINE\n"
                    f"    VAR __DS0FilterTable  = TREATAS({cabin_set}, 'Cabin_Master'[Cabin_Show])\n"
                    f"    VAR __DS0FilterTable2 = TREATAS({company_set}, 'Company_Master'[Company])\n"
                    f"    VAR __DS0FilterTable3 = TREATAS({haul_set}, 'Haul_Master'[Haul_Aggr])\n"
                    "    VAR __DS0FilterTable4 =\n"
                    "        FILTER(\n"
                    "            KEEPFILTERS(VALUES('Date_Master'[Date])),\n"
                    "            AND(\n"
                    f"                'Date_Master'[Date] >= DATE({comp_start_str}),\n"
                    f"                'Date_Master'[Date] <= DATE({comp_end_str})\n"
                    "            ))\n"
                    "    VAR __DS0FilterTable5 = TREATAS({1}, 'TouchPoint_Master'[explanatory_drivers])\n"
                    f"    VAR __DS0FilterTable6 = TREATAS({{\"{safe_touchpoint}\"}}, 'TouchPoint_Master'[filtered_name])\n"
                    "    VAR __DS0Core =\n"
                    "        SUMMARIZECOLUMNS(\n"
                    "            'TouchPoint_Master'[filtered_name],\n"
                    "            __DS0FilterTable,\n"
                    "            __DS0FilterTable2,\n"
                    "            __DS0FilterTable3,\n"
                    "            __DS0FilterTable4,\n"
                    "            __DS0FilterTable5,\n"
                    "            __DS0FilterTable6,\n"
                    '            "CSAT", [Monthly_Satisfaction]\n'
                    "        )\n"
                    "EVALUATE __DS0Core\n"
                )

                try:
                    df_comparison = await self._execute_query_async(comparison_query)
                    if not df_comparison.empty:
                        df_comparison = self._safe_clean_columns(df_comparison)
                        csat_comparison = _to_float(df_comparison.iloc[0].get("CSAT"))
                        if csat_comparison is not None:
                            satisfaction_diff = csat - csat_comparison
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not calculate satisfaction diff for focus touchpoint '{touchpoint_name}': {e}")

            if csat is None or target is None:
                self.logger.warning(
                    f"⚠️ Missing CSAT or Target for focus touchpoint '{touchpoint_name}': "
                    f"csat={csat}, target={target}"
                )
                return None

            gap = csat - target

            return {
                "csat": csat,
                "target": target,
                "gap": gap,
                "satisfaction_diff": satisfaction_diff,
            }

        except Exception as e:
            self.logger.warning(
                f"⚠️ Error collecting focus touchpoint CSAT vs target for "
                f"'{touchpoint_name}' on '{node_path}': {e}"
            )
            return None

    def _get_focus_touchpoint_issues_pct_query(
        self,
        cabins: List[str],
        companies: List[str],
        hauls: List[str],
        start_date,
        end_date,
        comparison_start_date,
        comparison_end_date,
        touchpoint_display_name: str,
    ) -> str:
        """Build the DAX query for % issues of a focus touchpoint vs L7D."""
        def _to_dt(d):
            if isinstance(d, str):
                return datetime.strptime(d, "%Y-%m-%d")
            return d

        sd = _to_dt(start_date)
        ed = _to_dt(end_date)
        csd = _to_dt(comparison_start_date)
        ced = _to_dt(comparison_end_date)

        def _date_dax(dt: datetime) -> str:
            return f"DATE({dt.year}, {dt.month}, {dt.day})"

        # Build segment filters
        cabin_values = ", ".join(f'"{c}"' for c in cabins) if cabins else '"All"'
        company_values = ", ".join(f'"{c}"' for c in companies) if companies else '"All"'
        haul_values = ", ".join(f'"{h}"' for h in hauls) if hauls else '"All"'

        cabin_filter_l7d = (
            f"FILTER(ALL(Cabin_Master), Cabin_Master[Cabin_Show] IN {{{cabin_values}}})"
            if cabins else ""
        )
        company_filter_l7d = (
            f"FILTER(ALL(Company_Master), Company_Master[Company] IN {{{company_values}}})"
            if companies else ""
        )
        haul_filter_l7d = (
            f"FILTER(ALL(Haul_Master), Haul_Master[Haul_Aggr] IN {{{haul_values}}})"
            if hauls else ""
        )

        seg_filters = ", ".join(
            f for f in [cabin_filter_l7d, company_filter_l7d, haul_filter_l7d] if f
        )
        seg_filters_prefix = (", " + seg_filters) if seg_filters else ""

        # Filter by Touchpoint (high-level display name in Issue_touchpoint_Dict).
        # The [Touchpoint] column matches display_name values like "Check-in", "Boarding",
        # "Arrivals experience" — verified against PBI. The [issue_type_3] values are
        # granular sub-issues and don't correspond to the touchpoint-level names.
        safe_touchpoint = touchpoint_display_name.replace('"', '\\"')

        query = (
            "EVALUATE\n"
            f"VAR _end        = {_date_dax(ed)}\n"
            f"VAR _start      = {_date_dax(sd)}\n"
            f"VAR _start_prev = {_date_dax(csd)}\n"
            f"VAR _end_prev   = {_date_dax(ced)}\n"
            "VAR _dateL7D   = FILTER(ALL(Date_Master), Date_Master[Date] >= _start      && Date_Master[Date] <= _end)\n"
            "VAR _datePrev  = FILTER(ALL(Date_Master), Date_Master[Date] >= _start_prev && Date_Master[Date] <= _end_prev)\n"
            "RETURN\n"
            "UNION(\n"
            '    ROW("Period", "L7D",\n'
            '        "Pct_Issues", CALCULATE([Switch_%_Affected_D&G],\n'
            f"            _dateL7D{seg_filters_prefix},\n"
            f'            FILTER(ALL(Issue_touchpoint_Dict), Issue_touchpoint_Dict[Touchpoint] = "{safe_touchpoint}")\n'
            "        )\n"
            "    ),\n"
            '    ROW("Period", "L7D_prev",\n'
            '        "Pct_Issues", CALCULATE([Switch_%_Affected_D&G],\n'
            f"            _datePrev{seg_filters_prefix},\n"
            f'            FILTER(ALL(Issue_touchpoint_Dict), Issue_touchpoint_Dict[Touchpoint] = "{safe_touchpoint}")\n'
            "        )\n"
            "    )\n"
            ")\n"
        )
        return query

    async def collect_focus_touchpoint_issues_pct(
        self,
        node_path: str,
        start_date,
        end_date,
        comparison_start_date,
        comparison_end_date,
        touchpoint_display_name: str,
    ) -> Optional[Dict[str, float]]:
        """
        Collect % of passengers affected by issues for a focus touchpoint.

        Args:
            node_path: Node path like "Global/LH/Business"
            start_date: Start of current period (datetime or YYYY-MM-DD str)
            end_date: End of current period
            comparison_start_date: Start of comparison period
            comparison_end_date: End of comparison period
            touchpoint_display_name: Name in Issue_touchpoint_Dict (e.g. "Cabin Crew")

        Returns:
            Dict with keys: 'pct_issues_current', 'pct_issues_prev', 'diff'
            or None on error / no data.
        """
        try:
            cabins, companies, hauls = self._get_node_filters(node_path)

            query = self._get_focus_touchpoint_issues_pct_query(
                cabins=cabins,
                companies=companies,
                hauls=hauls,
                start_date=start_date,
                end_date=end_date,
                comparison_start_date=comparison_start_date,
                comparison_end_date=comparison_end_date,
                touchpoint_display_name=touchpoint_display_name,
            )

            df = await self._execute_query_async(query)

            if df.empty:
                self.logger.warning(
                    f"⚠️ No issues data (empty DataFrame) for touchpoint '{touchpoint_display_name}' on '{node_path}'"
                )
                return None

            df = self._safe_clean_columns(df)
            self.logger.info(f"🔍 issues_pct columns for '{node_path}': {list(df.columns)}, rows: {len(df)}")
            self.logger.info(f"🔍 issues_pct data:\n{df.to_string()}")

            def _to_float(val) -> Optional[float]:
                try:
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        return None
                    return float(val)
                except (TypeError, ValueError):
                    return None

            # Locate L7D and L7D_prev rows
            period_col = next(
                (c for c in df.columns if "period" in c.lower()), None
            )
            pct_col = next(
                (c for c in df.columns if "pct" in c.lower() or "issues" in c.lower()), None
            )

            if period_col is None or pct_col is None:
                self.logger.warning(
                    f"⚠️ Unexpected columns in issues query result: {list(df.columns)}"
                )
                return None

            l7d_row = df[df[period_col] == "L7D"]
            prev_row = df[df[period_col] == "L7D_prev"]

            pct_current = _to_float(l7d_row.iloc[0][pct_col]) if not l7d_row.empty else None
            pct_prev = _to_float(prev_row.iloc[0][pct_col]) if not prev_row.empty else None

            if pct_current is None and pct_prev is None:
                return None

            diff = (
                (pct_current - pct_prev)
                if pct_current is not None and pct_prev is not None
                else None
            )

            return {
                "pct_issues_current": pct_current,
                "pct_issues_prev": pct_prev,
                "diff": diff,
            }

        except Exception as e:
            self.logger.warning(
                f"⚠️ Error collecting focus touchpoint issues pct for "
                f"'{touchpoint_display_name}' on '{node_path}': {e}"
            )
            return None
