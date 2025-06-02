import os
import msal
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

class PBIDataCollector:
    """Collects data from Power BI API for each node in the NPS tree hierarchy"""
    
    def __init__(self):
        # Load environment variables from .devcontainer/.env
        dotenv_path = Path(__file__).parent.parent.parent / '.devcontainer' / '.env'
        load_dotenv(dotenv_path)
        
        # Get credentials from environment
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET") 
        self.tenant_id = os.getenv("TENANT_ID")
        self.group_id = os.getenv("GROUP_ID")
        self.dataset_id = os.getenv("DATASET_ID")
        
        # Validate credentials
        if not all([self.client_id, self.client_secret, self.tenant_id, self.group_id, self.dataset_id]):
            raise ValueError("Missing required environment variables for Power BI API")
            
        # Get access token
        self.access_token = self._get_access_token()
        
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
    
    def _get_daily_nps_query(self, cabins: List[str], companies: List[str], hauls: List[str]) -> str:
        """Generate DAX query for daily NPS data using template"""
        template = self._load_query_template("Daily NPS.txt")
        
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
        
        return query
    
    def _get_operative_query(self, cabins: List[str], companies: List[str], hauls: List[str]) -> str:
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
        
        return query
    
    def _get_verbatims_query(self, cabins: List[str], companies: List[str], hauls: List[str], date: datetime) -> str:
        """Generate DAX query for verbatims data using template"""
        template = self._load_query_template("Verbatims.txt")
        
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
            'date(2025,05,12)',
            f'date({date.year},{date.month},{date.day})'
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
    
    def collect_node_data(self, node_path: str, output_dir: Path) -> Dict[str, bool]:
        """Collect all data types for a specific node"""
        print(f"Collecting data for node: {node_path}")
        
        # Create output directory
        node_dir = output_dir / node_path
        node_dir.mkdir(parents=True, exist_ok=True)
        
        # Get filters for this node
        cabins, companies, hauls = self._get_node_filters(node_path)
        
        print(f"  Filters - Cabins: {cabins}, Companies: {companies}, Hauls: {hauls}")
        
        results = {}
        
        # Collect daily NPS data
        try:
            query = self._get_daily_nps_query(cabins, companies, hauls)
            df = self._execute_query(query)
            if not df.empty:
                df.to_csv(node_dir / 'daily_NPS.csv', index=False)
                results['daily_NPS'] = True
                print(f"  ✓ daily_NPS.csv saved ({len(df)} rows)")
            else:
                results['daily_NPS'] = False
                print(f"  ✗ daily_NPS.csv - no data")
        except Exception as e:
            results['daily_NPS'] = False
            print(f"  ✗ daily_NPS.csv - error: {str(e)}")
        
        # Collect operative data
        try:
            query = self._get_operative_query(cabins, companies, hauls)
            df = self._execute_query(query)
            if not df.empty:
                df.to_csv(node_dir / 'daily_operative.csv', index=False)
                results['daily_operative'] = True
                print(f"  ✓ daily_operative.csv saved ({len(df)} rows)")
            else:
                results['daily_operative'] = False
                print(f"  ✗ daily_operative.csv - no data")
        except Exception as e:
            results['daily_operative'] = False
            print(f"  ✗ daily_operative.csv - error: {str(e)}")
            
        return results

    def collect_verbatims_for_date_and_segment(self, node_path: str, date: datetime, output_dir: Path = None) -> pd.DataFrame:
        """Collect verbatims for a specific date and segment with explanation needed flag"""
        print(f"🔍 Collecting verbatims for {node_path} on {date.strftime('%Y-%m-%d')}")
        
        # Get filters for this node
        cabins, companies, hauls = self._get_node_filters(node_path)
        
        # Use the template system for verbatims query
        query = self._get_verbatims_query(cabins, companies, hauls, date)
        
        try:
            print(f"  📝 Collecting verbatims with filters: Cabins={cabins}, Companies={companies}, Hauls={hauls}")
            df = self._execute_query(query)
            
            if not df.empty:
                print(f"  ✅ Found {len(df)} verbatims for {node_path} on {date.strftime('%Y-%m-%d')}")
                
                # Save to structured directory if output_dir is provided
                if output_dir:
                    # Create directory structure: output_dir/date/node_path/
                    date_str = date.strftime("%Y_%m_%d")
                    segment_dir = output_dir / date_str / node_path
                    segment_dir.mkdir(parents=True, exist_ok=True)
                    
                    filename = 'verbatims.csv'
                    filepath = segment_dir / filename
                    df.to_csv(filepath, index=False)
                    print(f"  💾 Saved to {filepath}")
                
                return df
            else:
                print(f"  ❌ No verbatims found for {node_path} on {date.strftime('%Y-%m-%d')}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"  ❌ Error collecting verbatims for {node_path} on {date.strftime('%Y-%m-%d')}: {str(e)}")
            return pd.DataFrame()
    
    def collect_all_nodes_data(self, output_base_dir: str = "tables") -> Dict[str, Dict[str, bool]]:
        """Collect data for all nodes in the tree structure"""
        base_path = Path(output_base_dir)
        today_str = datetime.now().strftime('%d_%m_%Y')
        output_dir = base_path / today_str
        
        print(f"Starting data collection to: {output_dir}")
        
        all_results = {}
        
        # Define all node paths based on the tree structure
        node_paths = [
            'Global',
            'Global/LH',
            'Global/LH/Economy',
            'Global/LH/Economy/IB',
            'Global/LH/Economy/YW', 
            'Global/LH/Business',
            'Global/LH/Business/IB',
            'Global/LH/Business/YW',
            'Global/LH/Premium',
            'Global/LH/Premium/IB',
            'Global/LH/Premium/YW',
            'Global/SH',
            'Global/SH/Economy',
            'Global/SH/Economy/IB',
            'Global/SH/Economy/YW',
            'Global/SH/Business',
            'Global/SH/Business/IB',
            'Global/SH/Business/YW'
        ]
        
        # Collect data for each node
        for node_path in node_paths:
            all_results[node_path] = self.collect_node_data(node_path, output_dir)
            
        return all_results

    async def collect_all_data(self) -> Tuple[int, int]:
        """
        Collect data for all nodes and return success/total counts.
        This method is called by main.py and wraps collect_all_nodes_data.
        """
        results = self.collect_all_nodes_data()
        
        # Count successful file collections
        success_count = 0
        total_count = 0
        
        for node_path, node_results in results.items():
            for file_type, success in node_results.items():
                total_count += 1
                if success:
                    success_count += 1
        
        print(f"\n📊 Data Collection Summary:")
        print(f"   Total files attempted: {total_count}")
        print(f"   Successful files: {success_count}")
        print(f"   Success rate: {success_count/total_count*100:.1f}%")
        
        return success_count, total_count
