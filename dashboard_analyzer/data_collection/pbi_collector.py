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

class PBIDataCollector:
    """Collects data from Power BI API for each node in the NPS tree hierarchy"""
    
    def __init__(self, environment: str = "prod"):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.environment = environment
        
        # Load environment variables from .devcontainer/.env ONLY if not in prod
        if self.environment != "prod":
            dotenv_path = Path(__file__).parent.parent.parent / '.devcontainer' / '.env'
            if dotenv_path.exists():
                load_dotenv(dotenv_path)
        
        # Get credentials from environment
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET") 
        self.tenant_id = os.getenv("TENANT_ID")
        self.group_id = os.getenv("GROUP_ID")
        self.dataset_id = os.getenv("DATASET_ID")
        
        if not all([self.client_id, self.client_secret, self.tenant_id, self.group_id, self.dataset_id]):
            print("⚠️ Warning: Missing required environment variables for Power BI API")
            print("Required: CLIENT_ID, CLIENT_SECRET, TENANT_ID, GROUP_ID, DATASET_ID")
        
        self.access_token = None
        
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
        """Generate DAX query for verbatims data using date range template"""
        template = self._load_query_template("Verbatims.txt")
        
        # Replace placeholders with actual values
        cabins_str = '", "'.join(cabins)
        companies_str = '", "'.join(companies)
        hauls_str = '", "'.join(hauls)
        
        # Replace the template placeholders - replace ALL occurrences
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
            '\'Date_Master\'[Date] =date(2025,05,12)',
            f'\'Date_Master\'[Date] >= date({start_date.year},{start_date.month},{start_date.day}) && \'Date_Master\'[Date] <= date({end_date.year},{end_date.month},{end_date.day})'
        )
        
        return query
    
    def _execute_query(self, query: str) -> pd.DataFrame:
        """Execute a DAX query against Power BI API"""
        dax_query = {
            "queries": [{"query": query}],
            "serializerSettings": {"includeNulls": True}
        }
        
        url = f"https://api.powerbi.com/v1.0/myorg/groups/{self.group_id}/datasets/{self.dataset_id}/executeQueries"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, headers=headers, json=dax_query)
            
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                return pd.DataFrame()
                
            results = response.json()
            
            if not results.get('results') or not results['results'][0].get('tables'):
                print("No data returned from query")
                return pd.DataFrame()
                
            rows = results['results'][0]['tables'][0].get('rows', [])
            return pd.DataFrame(rows)
            
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            return pd.DataFrame()
    
    async def _execute_query_async(self, query: str) -> pd.DataFrame:
        """Execute a DAX query against Power BI API asynchronously"""
        dax_query = {
            "queries": [{"query": query}],
            "serializerSettings": {"includeNulls": True}
        }
        
        url = f"https://api.powerbi.com/v1.0/myorg/groups/{self.group_id}/datasets/{self.dataset_id}/executeQueries"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=dax_query) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        print(f"Error {response.status}: {response_text}")
                        return pd.DataFrame()
                        
                    results = await response.json()
                    
                    if not results.get('results') or not results['results'][0].get('tables'):
                        print("No data returned from query")
                        return pd.DataFrame()
                        
                    rows = results['results'][0]['tables'][0].get('rows', [])
                    return pd.DataFrame(rows)
                    
        except Exception as e:
            print(f"Error executing async query: {str(e)}")
            return pd.DataFrame()
    
    def _get_node_filters(self, node_path: str) -> Tuple[List[str], List[str], List[str]]:
        """Get the filter values for cabins, companies, and hauls based on node path"""
        path_parts = node_path.split('/')
        
        # Default to all values
        cabins = ["Business", "Economy", "Premium EC"]
        companies = ["IB", "YW"]
        hauls = ["SH", "LH"]
        
        # Apply filters based on path
        if len(path_parts) >= 2 and path_parts[1] in ['LH', 'SH']:
            hauls = [path_parts[1]]
            
        if len(path_parts) >= 3 and path_parts[2] in ['Economy', 'Business', 'Premium']:
            if path_parts[2] == 'Premium':
                cabins = ["Premium EC"]
            else:
                cabins = [path_parts[2]]
                
        if len(path_parts) >= 4 and path_parts[3] in ['IB', 'YW']:
            companies = [path_parts[3]]
            
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
        """Parse node path to extract cabins, companies, and hauls"""
        
        # Default values for Global
        cabins = ['Business', 'Economy', 'Premium EC']
        companies = ['IB', 'YW'] 
        hauls = ['SH', 'LH']
        
        # Parse path segments
        segments = node_path.split('/')
        
        # Extract haul information
        if 'LH' in segments:
            hauls = ['LH']
        elif 'SH' in segments:
            hauls = ['SH']
        
        # Extract cabin information
        if 'Economy' in segments:
            cabins = ['Economy']
        elif 'Business' in segments:
            cabins = ['Business']
        elif 'Premium' in segments:
            cabins = ['Premium EC']
        
        # Extract company information
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
            
            # Replace analysis period placeholders
            query = template.replace('__START_DATE__', start_date.strftime('%Y, %-m, %-d'))
            query = query.replace('__END_DATE__', end_date.strftime('%Y, %-m, %-d'))
            
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
        
        # Replace analysis period placeholders (same as exp_drivers_tool)
        query = query.replace('__START_DATE__', start_date.strftime('%Y, %-m, %-d'))
        query = query.replace('__END_DATE__', end_date.strftime('%Y, %-m, %-d'))
        
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
