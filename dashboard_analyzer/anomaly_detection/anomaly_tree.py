import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

class AnomalyNode:
    """Represents a node in the anomaly detection tree"""
    
    def __init__(self, name: str, path: str, parent=None):
        self.name = name
        self.path = path
        self.parent = parent
        self.children: Dict[str, 'AnomalyNode'] = {}
        self.data: Optional[pd.DataFrame] = None
        self.moving_averages: Dict[str, float] = {}  # day_of_week -> moving_average
        self.daily_anomaly: Optional['Anomaly'] = None
        self.weekly_anomaly: Optional['Anomaly'] = None
        self.explanation: Optional[str] = None
        
    def add_child(self, child: 'AnomalyNode'):
        """Add a child node"""
        self.children[child.name] = child
        child.parent = self
        
    def get_path(self) -> str:
        """Get the full path of this node"""
        return self.path
        
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return len(self.children) == 0
        
    def get_all_descendants(self) -> List['AnomalyNode']:
        """Get all descendant nodes"""
        descendants = []
        for child in self.children.values():
            descendants.append(child)
            descendants.extend(child.get_all_descendants())
        return descendants

class Anomaly:
    """Represents an anomaly detection result"""
    
    def __init__(self, current_value: float, target_value: float, 
                 deviation: float, threshold: float, anomaly_type: str):
        self.current_value = current_value
        self.target_value = target_value
        self.deviation = deviation
        self.threshold = threshold
        self.anomaly_type = anomaly_type  # 'daily' or 'weekly'
        self.is_anomaly = abs(deviation) > threshold

class AnomalyTree:
    """Tree structure for anomaly detection across hierarchical nodes"""
    
    def __init__(self, data_base_path: str = "tables"):
        self.data_base_path = Path(data_base_path)
        self.root: Optional[AnomalyNode] = None
        self.nodes: Dict[str, AnomalyNode] = {}
        self.dates: List[str] = []
        
    def build_tree_structure(self):
        """Build the tree structure based on the README hierarchy"""
        # Create root node
        self.root = AnomalyNode("Global", "Global")
        self.nodes["Global"] = self.root
        
        # Create LH branch (only goes to cabin level)
        lh_node = AnomalyNode("LH", "Global/LH", self.root)
        self.root.add_child(lh_node)
        self.nodes["Global/LH"] = lh_node
        
        # LH cabin nodes (no company subdivision)
        for cabin in ["Economy", "Business", "Premium"]:
            cabin_node = AnomalyNode(cabin, f"Global/LH/{cabin}", lh_node)
            lh_node.add_child(cabin_node)
            self.nodes[f"Global/LH/{cabin}"] = cabin_node
            
        # Create SH branch (goes to cabin AND company level)
        sh_node = AnomalyNode("SH", "Global/SH", self.root)
        self.root.add_child(sh_node)
        self.nodes["Global/SH"] = sh_node
        
        # SH cabin nodes with company subdivisions
        for cabin in ["Economy", "Business"]:
            cabin_node = AnomalyNode(cabin, f"Global/SH/{cabin}", sh_node)
            sh_node.add_child(cabin_node)
            self.nodes[f"Global/SH/{cabin}"] = cabin_node
            
            # Add company subdivisions for SH
            for company in ["IB", "YW"]:
                company_node = AnomalyNode(company, f"Global/SH/{cabin}/{company}", cabin_node)
                cabin_node.add_child(company_node)
                self.nodes[f"Global/SH/{cabin}/{company}"] = company_node
                
    def load_data(self, date_folder: str = "22_05_2025"):
        """Load NPS data for all nodes from the specified date folder"""
        data_path = self.data_base_path / date_folder
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data folder not found: {data_path}")
            
        print(f"📊 Loading data from: {data_path}")
        
        # Load data for each node
        loaded_nodes = 0
        for node_path, node in self.nodes.items():
            nps_file = data_path / node_path / "daily_NPS.csv"
            
            if nps_file.exists():
                try:
                    df = pd.read_csv(nps_file)
                    if not df.empty:
                        # Fix column names to match actual CSV structure
                        df = df.rename(columns={
                            'Date_Master[Date]': 'Date',
                            '[NPS_Raw]': 'NPS',
                            '[Responses]': 'Responses',
                            '[Target]': 'Target'
                        })
                        
                        # Ensure Date column is datetime
                        df['Date'] = pd.to_datetime(df['Date'])
                        df = df.sort_values('Date')
                        node.data = df
                        loaded_nodes += 1
                        
                        # Update global dates list
                        if not self.dates:
                            self.dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
                            
                except Exception as e:
                    print(f"⚠️ Error loading data for {node_path}: {e}")
                    
        print(f"✅ Loaded data for {loaded_nodes}/{len(self.nodes)} nodes")
        print(f"📅 Date range: {len(self.dates)} days ({self.dates[0]} to {self.dates[-1]})")
        
    def calculate_moving_averages(self, window_days: int = 7):
        """Calculate mean of last 7 days for each node"""
        print(f"📈 Calculating mean of last {window_days} days...")
        
        for node_path, node in self.nodes.items():
            if node.data is not None and len(node.data) >= window_days:
                # Get the last 7 days of data
                last_7_days = node.data.tail(window_days)
                
                # Calculate the mean of these last 7 days
                mean_last_7_days = last_7_days['NPS'].mean()
                
                # Store this mean for later comparison
                node.data['mean_last_7_days'] = mean_last_7_days
                
    def detect_daily_anomalies(self, threshold: float = 10.0):
        """Detect anomalies for each of the last 7 days comparing with their mean"""
        print(f"🔍 Detecting anomalies in last 7 days (threshold: ±{threshold} points)...")
        
        # Dictionary to store anomaly states for each day
        self.daily_anomalies: Dict[str, Dict[str, str]] = {}  # date -> {node_path -> anomaly_state}
        
        # Only analyze the last 7 days
        last_7_dates = self.dates[-7:] if len(self.dates) >= 7 else self.dates
        
        for date in last_7_dates:
            self.daily_anomalies[date] = {}
            
            for node_path, node in self.nodes.items():
                if node.data is not None and hasattr(node.data, 'mean_last_7_days'):
                    # Find data for this specific date
                    day_data = node.data[node.data['Date'].dt.strftime('%Y-%m-%d') == date]
                    
                    if not day_data.empty:
                        current_nps = day_data.iloc[0]['NPS']
                        mean_last_7_days = node.data['mean_last_7_days'].iloc[0]  # This is the same for all rows
                        deviation = current_nps - mean_last_7_days
                        
                        # Determine anomaly state based on README criteria
                        if deviation >= threshold:
                            anomaly_state = "+"
                        elif deviation <= -threshold:
                            anomaly_state = "-"
                        else:
                            anomaly_state = "N"
                            
                        self.daily_anomalies[date][node_path] = anomaly_state
                    else:
                        # No data for this date
                        self.daily_anomalies[date][node_path] = "?"
                else:
                    # No data or insufficient history
                    self.daily_anomalies[date][node_path] = "?"
        
    def print_collapsed_tree(self, date: str):
        """Print collapsed tree view for a specific date"""
        if date not in self.daily_anomalies:
            print(f"❌ No anomaly data for date: {date}")
            return
            
        print(f"\n🌳 Anomaly Tree for {date}")
        print("=" * 40)
        
        anomalies = self.daily_anomalies[date]
        
        # Print in hierarchical format
        global_state = anomalies.get("Global", "?")
        print(f"Global [{global_state}]")
        
        # LH branch
        lh_state = anomalies.get("Global/LH", "?")
        print(f"├── LH [{lh_state}]")
        
        lh_cabins = ["Economy", "Business", "Premium"]
        for i, cabin in enumerate(lh_cabins):
            cabin_path = f"Global/LH/{cabin}"
            cabin_state = anomalies.get(cabin_path, "?")
            prefix = "    ├──" if i < len(lh_cabins) - 1 else "    └──"
            print(f"{prefix} {cabin} [{cabin_state}]")
            
        # SH branch  
        sh_state = anomalies.get("Global/SH", "?")
        print(f"└── SH [{sh_state}]")
        
        sh_cabins = ["Economy", "Business"]
        for i, cabin in enumerate(sh_cabins):
            cabin_path = f"Global/SH/{cabin}"
            cabin_state = anomalies.get(cabin_path, "?")
            prefix = "    ├──" if i < len(sh_cabins) - 1 else "    └──"
            print(f"{prefix} {cabin} [{cabin_state}]")
            
            # Company subdivisions for SH
            companies = ["IB", "YW"]
            for j, company in enumerate(companies):
                company_path = f"Global/SH/{cabin}/{company}"
                company_state = anomalies.get(company_path, "?")
                company_prefix = "        ├──" if j < len(companies) - 1 else "        └──"
                print(f"{company_prefix} {company} [{company_state}]")
                
    def print_all_days_summary(self):
        """Print anomaly summary for the last 7 days"""
        print(f"\n📊 ANOMALY DETECTION SUMMARY (Last 7 Days)")
        print("=" * 60)
        print("Legend: [+] Above 7-day mean +10pts | [-] Below 7-day mean -10pts | [N] Normal | [?] No data")
        print()
        
        # Only show dates that we analyzed (last 7 days)
        analyzed_dates = sorted(self.daily_anomalies.keys())
        
        for date in analyzed_dates:
            if date in self.daily_anomalies:
                # Count anomalies for this day
                anomalies = self.daily_anomalies[date]
                plus_count = sum(1 for state in anomalies.values() if state == "+")
                minus_count = sum(1 for state in anomalies.values() if state == "-")
                normal_count = sum(1 for state in anomalies.values() if state == "N")
                
                status_emoji = "🚨" if (plus_count + minus_count) > 0 else "✅"
                print(f"{status_emoji} {date}: +{plus_count} -{minus_count} N{normal_count}")
                
    def analyze_date(self, date: str):
        """Complete analysis for a specific date"""
        print(f"\n🔍 ANALYZING DATE: {date}")
        print("=" * 50)
        
        if date not in self.daily_anomalies:
            print(f"❌ No data available for {date}")
            return
            
        # Print tree view
        self.print_collapsed_tree(date)
        
        # Print detailed anomaly info
        anomalies = self.daily_anomalies[date]
        plus_nodes = [path for path, state in anomalies.items() if state == "+"]
        minus_nodes = [path for path, state in anomalies.items() if state == "-"]
        
        if plus_nodes:
            print(f"\n🔺 POSITIVE ANOMALIES (+10+ points above 7-day average):")
            for node in plus_nodes:
                print(f"   • {node}")
                
        if minus_nodes:
            print(f"\n🔻 NEGATIVE ANOMALIES (-10+ points below 7-day average):")
            for node in minus_nodes:
                print(f"   • {node}")
                
        if not plus_nodes and not minus_nodes:
            print(f"\n✅ No significant anomalies detected for {date}")
            
    def run_full_analysis(self, date_folder: str = "22_05_2025"):
        """Run complete anomaly detection analysis"""
        print("🚀 STARTING ANOMALY DETECTION ANALYSIS")
        print("=" * 60)
        
        # Step 1: Build tree structure
        self.build_tree_structure()
        print(f"✅ Built tree structure with {len(self.nodes)} nodes")
        
        # Step 2: Load data
        self.load_data(date_folder)
        
        # Step 3: Calculate moving averages
        self.calculate_moving_averages()
        
        # Step 4: Detect anomalies
        self.detect_daily_anomalies()
        
        # Step 5: Print summary
        self.print_all_days_summary()
        
        return self
