"""
Flexible Anomaly Detection System
Supports different temporal aggregation periods (7 days, 14 days, 30 days, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from .anomaly_tree import AnomalyTree, AnomalyNode

class FlexibleAnomalyDetector:
    """Flexible anomaly detection using configurable temporal aggregation"""
    
    def __init__(self, aggregation_days: int = 7, threshold: float = 10.0, min_sample_size: int = 5):
        """
        Initialize flexible anomaly detector
        
        Args:
            aggregation_days: Number of days to aggregate into each period (default: 7)
            threshold: NPS deviation threshold for anomaly detection (default: 10.0)
            min_sample_size: Minimum sample size for analysis (default: 5)
        """
        self.aggregation_days = aggregation_days
        self.threshold = threshold
        self.min_sample_size = min_sample_size
        self.tree: Optional[AnomalyTree] = None
        
    def analyze_flexible_anomalies(self, data_folder: str) -> Tuple[Dict[str, str], Dict[str, float], List[str]]:
        """
        Analyze anomalies using flexible temporal aggregation
        
        Returns:
            Tuple of (anomalies_dict, deviations_dict, periods_list)
        """
        print(f"🔍 FLEXIBLE ANOMALY DETECTION ({self.aggregation_days} days aggregation)")
        print("="*70)
        
        # Initialize tree
        self.tree = AnomalyTree(data_folder)
        self.tree.build_tree_structure()
        
        # Load flexible data for all nodes
        all_data = self._load_flexible_data(data_folder)
        
        if not all_data:
            print("❌ No flexible data found")
            return {}, {}, []
        
        # Get available periods
        periods = self._get_available_periods(all_data)
        print(f"📅 Found {len(periods)} periods with {self.aggregation_days}-day aggregation")
        
        if len(periods) < 3:
            print(f"⚠️ Need at least 3 periods for anomaly detection, found {len(periods)}")
            return {}, {}, periods
        
        # Detect anomalies for the most recent period
        latest_period = periods[0]  # Period 1 is the most recent (at the beginning of descending order)
        anomalies, deviations = self._detect_period_anomalies(all_data, latest_period, periods)
        
        print(f"\n🎯 Anomaly detection completed for period {latest_period}")
        print(f"📊 Found {sum(1 for a in anomalies.values() if a in ['+', '-'])} anomalies out of {len(anomalies)} nodes")
        
        return anomalies, deviations, periods
    
    def _load_flexible_data(self, data_folder: str) -> Dict[str, pd.DataFrame]:
        """Load flexible aggregation data for all nodes"""
        data_folder_path = Path(data_folder)
        all_data = {}
        
        # Define node paths mapping: logical_path -> folder_name
        node_mapping = {
            "Global": "Global",
            "Global/LH": "Global_LH", 
            "Global/LH/Economy": "Global_LH_Economy", 
            "Global/LH/Business": "Global_LH_Business", 
            "Global/LH/Premium": "Global_LH_Premium",
            "Global/SH": "Global_SH", 
            "Global/SH/Economy": "Global_SH_Economy", 
            "Global/SH/Business": "Global_SH_Business",
            "Global/SH/Economy/IB": "Global_SH_Economy_IB", 
            "Global/SH/Economy/YW": "Global_SH_Economy_YW",
            "Global/SH/Business/IB": "Global_SH_Business_IB", 
            "Global/SH/Business/YW": "Global_SH_Business_YW"
        }
        
        loaded_count = 0
        for logical_path, folder_name in node_mapping.items():
            node_folder = data_folder_path / folder_name
            flexible_file = node_folder / f'flexible_NPS_{self.aggregation_days}d.csv'
            
            if flexible_file.exists():
                try:
                    df = pd.read_csv(flexible_file)
                    if not df.empty:
                        all_data[logical_path] = df
                        loaded_count += 1
                except Exception as e:
                    print(f"❌ Error loading {logical_path}: {e}")
        
        print(f"📊 Loaded {loaded_count}/{len(node_mapping)} segments")
        return all_data
    
    def _get_available_periods(self, all_data: Dict[str, pd.DataFrame]) -> List[int]:
        """Get list of available periods that have valid NPS data"""
        all_periods = set()
        
        for node_path, df in all_data.items():
            if 'Period_Group' in df.columns:
                # Only include periods that have valid NPS data (2024 or 2025)
                valid_periods = df[
                    (df['NPS_2025'].notna()) | (df['NPS_2024'].notna())
                ]['Period_Group'].unique()
                all_periods.update(valid_periods)
        
        # Sort periods in ascending order (period 1 = most recent)
        return sorted(list(all_periods))
    
    def _detect_period_anomalies(self, all_data: Dict[str, pd.DataFrame], 
                                target_period: int, all_periods: List[int]) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Detect anomalies for a specific period using mean of all 7 periods as baseline"""
        anomalies = {}
        deviations = {}
        
        # Use the 7 most recent periods (1-7) for baseline calculation
        baseline_periods = [p for p in all_periods if p <= 7][:7]
        
        if len(baseline_periods) < 3:
            print(f"⚠️ Insufficient baseline data for period {target_period} (need at least 3 periods)")
            return anomalies, deviations
        
        # Only print baseline info once per period  
        print(f"📈 Period {target_period}: baseline mean of periods {baseline_periods}")
        
        for node_path, df in all_data.items():
            if 'Period_Group' not in df.columns:
                continue
            
            # Get target period data
            target_data = df[df['Period_Group'] == target_period]
            if target_data.empty:
                anomalies[node_path] = "?"
                continue
                
            target_responses = target_data.get('Responses', pd.Series([0])).iloc[0]
            
            # Check minimum sample size
            if target_responses < self.min_sample_size:
                anomalies[node_path] = "S"  # Insufficient sample
                continue
            
            # Strategy: Try NPS_2025 first, fallback to NPS_2024
            target_nps = None
            baseline_avg = None
            
            # Try NPS_2025 for target
            if 'NPS_2025' in df.columns and not pd.isna(target_data['NPS_2025'].iloc[0]):
                target_nps = target_data['NPS_2025'].iloc[0]
                
                # Calculate baseline as mean of all 7 periods using NPS_2025
                baseline_data = df[df['Period_Group'].isin(baseline_periods)]
                baseline_nps_values = baseline_data['NPS_2025'].dropna()
                
                if len(baseline_nps_values) >= 3:
                    baseline_avg = baseline_nps_values.mean()
                else:
                    # Fallback: Use NPS_2024 baseline for periods 23-74 (2024 data)
                    nps_2024_periods = df[(df['Period_Group'] >= 23) & (df['Period_Group'] <= 74)]
                    if len(nps_2024_periods) >= 7:
                        baseline_avg = nps_2024_periods['NPS_2024'].dropna().tail(7).mean()
            
            # Fallback to NPS_2024 if NPS_2025 not available
            if target_nps is None and 'NPS_2024' in df.columns and not pd.isna(target_data['NPS_2024'].iloc[0]):
                target_nps = target_data['NPS_2024'].iloc[0]
                
                # Calculate baseline as mean of all 7 periods using NPS_2024
                baseline_data = df[df['Period_Group'].isin(baseline_periods)]
                baseline_nps_values = baseline_data['NPS_2024'].dropna()
                
                if len(baseline_nps_values) >= 3:
                    baseline_avg = baseline_nps_values.mean()
            
            # Final fallback to NPS_2019
            if target_nps is None and 'NPS_2019' in df.columns and not pd.isna(target_data['NPS_2019'].iloc[0]):
                target_nps = target_data['NPS_2019'].iloc[0]
                baseline_data = df[df['Period_Group'].isin(baseline_periods)]
                baseline_nps_values = baseline_data['NPS_2019'].dropna()
                
                if len(baseline_nps_values) >= 3:
                    baseline_avg = baseline_nps_values.mean()
            
            # Check if we have valid data
            if target_nps is None or baseline_avg is None:
                anomalies[node_path] = "?"
                continue
            
            deviation = target_nps - baseline_avg
            deviations[node_path] = deviation
            
            # Classify anomaly
            if deviation > self.threshold:
                anomalies[node_path] = "+"
            elif deviation < -self.threshold:
                anomalies[node_path] = "-"
            else:
                anomalies[node_path] = "N"
        
        return anomalies, deviations
    
    def get_period_summary(self, data_folder: str, periods: List[int]) -> pd.DataFrame:
        """Generate summary table for multiple periods"""
        summary_data = []
        
        for period in periods[:7]:  # Show last 7 periods
            anomalies, _, _ = self.analyze_period(data_folder, period)
            
            positive = sum(1 for a in anomalies.values() if a == "+")
            negative = sum(1 for a in anomalies.values() if a == "-")
            normal = sum(1 for a in anomalies.values() if a == "N")
            total = positive + negative + normal
            
            status = "🚨 Alert" if (positive > 0 or negative > 0) else "✅ Normal"
            
            summary_data.append({
                'Period': f"Period {period}",
                'Status': status,
                '+': positive,
                '-': negative,
                'N': normal,
                'Total': total
            })
        
        return pd.DataFrame(summary_data)
    
    def analyze_period(self, data_folder: str, target_period: int) -> Tuple[Dict[str, str], Dict[str, float], List[str]]:
        """Analyze a specific period"""
        all_data = self._load_flexible_data(data_folder)
        periods = self._get_available_periods(all_data)
        anomalies, deviations = self._detect_period_anomalies(all_data, target_period, periods)
        return anomalies, deviations, periods 