import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import asyncio

class RoutesAnalyzer:
    """Analyzes route data to identify most affected routes for anomaly explanations"""
    
    def __init__(self, pbi_collector=None):
        self.pbi_collector = pbi_collector
        self.routes_data: Dict[str, pd.DataFrame] = {}  # date -> routes_df
        self.routes_dictionary: Optional[pd.DataFrame] = None
        
    async def load_routes_data(self, date: str, node_paths: List[str]):
        """Load route data for a specific date and relevant segments"""
        print(f"📍 Loading routes data for {date}...")
        
        if self.pbi_collector is None:
            print("❌ No PBI collector available for routes data")
            return
            
        try:
            # Load routes dictionary first (only once)
            if self.routes_dictionary is None:
                await self._load_routes_dictionary()
            
            # Load routes data for the specific date
            await self._load_routes_for_date(date, node_paths)
            
        except Exception as e:
            print(f"❌ Error loading routes data: {e}")
    
    async def _load_routes_dictionary(self):
        """Load the routes dictionary to map routes to segments"""
        print("📚 Loading routes dictionary...")
        
        try:
            # Read the dictionary query template
            dict_query_path = Path(__file__).parent.parent / "data_collection" / "queries" / "Rutas Diccionario.txt"
            
            if not dict_query_path.exists():
                print(f"❌ Routes dictionary query file not found: {dict_query_path}")
                return
                
            with open(dict_query_path, 'r', encoding='utf-8') as f:
                dict_query = f.read()
            
            # Execute the dictionary query
            df = await self.pbi_collector._execute_query_async(dict_query)
            
            if df is not None and not df.empty:
                # Clean column names
                df.columns = [col.strip('[]') for col in df.columns]
                self.routes_dictionary = df
                print(f"✅ Loaded {len(df)} routes in dictionary")
            else:
                print("❌ No routes dictionary data returned")
                
        except Exception as e:
            print(f"❌ Error loading routes dictionary: {e}")
    
    async def _load_routes_for_date(self, date: str, node_paths: List[str]):
        """Load route NPS data for a specific date"""
        try:
            # Read the routes query template
            routes_query_path = Path(__file__).parent.parent / "data_collection" / "queries" / "Rutas.txt"
            
            if not routes_query_path.exists():
                print(f"❌ Routes query file not found: {routes_query_path}")
                return
                
            with open(routes_query_path, 'r', encoding='utf-8') as f:
                routes_query_template = f.read()
            
            # Replace the date placeholder in the query
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            routes_query = routes_query_template.replace(
                "date(2025,05,12)", 
                f"date({date_obj.year},{date_obj.month:02d},{date_obj.day:02d})"
            )
            
            # Execute the routes query
            df = await self.pbi_collector._execute_query_async(routes_query)
            
            if df is not None and not df.empty:
                # Clean column names
                df.columns = [col.strip('[]') for col in df.columns]
                
                # Store the data
                self.routes_data[date] = df
                print(f"✅ Loaded routes data for {date}: {len(df)} routes")
            else:
                print(f"❌ No routes data returned for {date}")
                
        except Exception as e:
            print(f"❌ Error loading routes data for {date}: {e}")
    
    def get_most_affected_routes(self, date: str, node_path: str, anomaly_type: str, top_n: int = 3) -> List[Dict]:
        """Get the most affected routes for a specific segment and anomaly type"""
        if date not in self.routes_data or self.routes_dictionary is None:
            print(f"❌ Data not available: date={date in self.routes_data}, dictionary={self.routes_dictionary is not None}")
            return []
        
        try:
            # Get routes data for the date
            routes_df = self.routes_data[date].copy()
            
            if routes_df.empty:
                print(f"❌ No routes data for {date}")
                return []
            
            print(f"  📊 Processing {len(routes_df)} routes for {node_path}")
            
            # Filter routes by segment
            filtered_routes = self._filter_routes_by_segment(routes_df, node_path)
            
            if filtered_routes.empty:
                print(f"❌ No routes after filtering for {node_path}")
                return []
            
            # Find NPS column (try different variations)
            nps_column = None
            for col in filtered_routes.columns:
                if col.upper() == 'NPS' or 'nps' in col.lower():
                    nps_column = col
                    break
            
            if nps_column is None:
                print(f"❌ No NPS column found. Available columns: {list(filtered_routes.columns)}")
                return []
            
            # Find Pax column (try different variations)
            pax_column = None
            for col in filtered_routes.columns:
                if col.upper() == 'PAX' or 'pax' in col.lower():
                    pax_column = col
                    break
            
            # Find route column
            route_column = None
            for col in filtered_routes.columns:
                if 'route' in col.lower():
                    route_column = col
                    break
            
            print(f"  🎯 Using columns - Route: {route_column}, NPS: {nps_column}, Pax: {pax_column}")
            
            # Sort by NPS based on anomaly type
            if anomaly_type == "+":
                # For positive anomalies, show highest NPS routes first
                sorted_routes = filtered_routes.sort_values(nps_column, ascending=False)
            else:
                # For negative anomalies, show lowest NPS routes first
                sorted_routes = filtered_routes.sort_values(nps_column, ascending=True)
            
            # Get top N routes
            top_routes = sorted_routes.head(top_n)
            
            # Format the results
            result = []
            for _, route in top_routes.iterrows():
                route_name = route.get(route_column, '') if route_column else ''
                route_info = {
                    'route': route_name,
                    'nps': route.get(nps_column, 0),
                    'pax': route.get(pax_column, 0) if pax_column else 0,
                    'country': self._get_route_country(route_name)
                }
                result.append(route_info)
            
            print(f"  ✅ Found {len(result)} top routes for {node_path}")
            return result
            
        except Exception as e:
            print(f"❌ Error getting most affected routes: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _filter_routes_by_segment(self, routes_df: pd.DataFrame, node_path: str) -> pd.DataFrame:
        """Filter routes based on node path segment"""
        if self.routes_dictionary is None:
            print("❌ No routes dictionary available")
            return pd.DataFrame()
        
        try:
            print(f"  🔍 Filtering routes for segment: {node_path}")
            print(f"  📋 Routes data columns: {list(routes_df.columns)}")
            print(f"  📋 Dictionary columns: {list(self.routes_dictionary.columns)}")
            
            # Parse the node path to get segment filters
            filters = self._parse_node_path(node_path)
            print(f"  🎯 Parsed filters: {filters}")
            
            # Get all routes from dictionary
            valid_routes = self.routes_dictionary.copy()
            
            # Check what haul column is available
            haul_columns = [col for col in valid_routes.columns if 'haul' in col.lower()]
            
            # Apply haul filter if specified
            if filters.get('haul') and haul_columns:
                haul_col = haul_columns[0]  # Use the first haul column found
                valid_routes = valid_routes[valid_routes[haul_col] == filters['haul']]
                print(f"  🔍 Filtered by haul {filters['haul']}: {len(valid_routes)} routes remaining")
            
            # Get the route names that match our criteria
            matching_route_names = valid_routes['route'].tolist()
            print(f"  📝 Matching route names: {len(matching_route_names)} routes")
            
            # Find the route column in routes_df (try different variations)
            route_column = None
            for col in routes_df.columns:
                if 'route' in col.lower():
                    route_column = col
                    break
            
            if route_column is None:
                print(f"❌ No route column found in routes data. Available columns: {list(routes_df.columns)}")
                return pd.DataFrame()
            
            print(f"  🎯 Using route column: '{route_column}'")
            
            # Filter the routes data to only include matching routes
            filtered_routes = routes_df[routes_df[route_column].isin(matching_route_names)]
            
            print(f"  ✅ Final filtered routes: {len(filtered_routes)} routes")
            return filtered_routes
            
        except Exception as e:
            print(f"❌ Error filtering routes by segment: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _parse_node_path(self, node_path: str) -> Dict[str, str]:
        """Parse node path to extract segment filters"""
        filters = {}
        
        # Extract haul information
        if '/LH/' in node_path or node_path.endswith('/LH'):
            filters['haul'] = 'LH'
        elif '/SH/' in node_path or node_path.endswith('/SH'):
            filters['haul'] = 'SH'
        
        # Extract cabin information (could be extended)
        if '/Economy' in node_path:
            filters['cabin'] = 'Economy'
        elif '/Business' in node_path:
            filters['cabin'] = 'Business'
        elif '/Premium' in node_path:
            filters['cabin'] = 'Premium'
        
        # Extract company information (could be extended)
        if '/IB' in node_path:
            filters['company'] = 'IB'
        elif '/YW' in node_path:
            filters['company'] = 'YW'
        
        return filters
    
    def _get_route_country(self, route_name: str) -> str:
        """Get country information for a route from the dictionary"""
        if self.routes_dictionary is None:
            return ""
        
        try:
            route_info = self.routes_dictionary[self.routes_dictionary['route'] == route_name]
            if not route_info.empty:
                # Try different possible country column names
                country_columns = [col for col in route_info.columns if 'country' in col.lower()]
                if country_columns:
                    return route_info.iloc[0].get(country_columns[0], '')
                else:
                    return route_info.iloc[0].get('country_name', '')
        except Exception as e:
            print(f"❌ Error getting route country: {e}")
        
        return ""
    
    def format_routes_explanation(self, routes: List[Dict], anomaly_type: str) -> str:
        """Format routes information for explanation text with improved structure"""
        if not routes:
            return ""
        
        try:
            if anomaly_type == "+":
                intro = "Top routes"
            else:
                intro = "Problem routes"
            
            routes_text = []
            for route in routes:
                country = f" ({route['country']})" if route['country'] else ""
                nps_indicator = "🔴" if route['nps'] < 0 else "🟡" if route['nps'] < 50 else "🟢"
                
                routes_text.append(f"{route['route']}{country} {nps_indicator}{route['nps']:.1f}")
            
            return f"{intro}: {', '.join(routes_text)}"
            
        except Exception as e:
            print(f"❌ Error formatting routes explanation: {e}")
            return "" 