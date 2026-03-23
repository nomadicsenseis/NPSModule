"""
Flexible Anomaly Detection System
Supports different temporal aggregation periods (7 days, 14 days, 30 days, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class FlexibleAnomalyDetector:
    """Enhanced flexible anomaly detector with target-based detection support"""
    
    def __init__(self, aggregation_days: int = 7, threshold: float = 5.0, min_sample_size: int = 5, detection_mode: str = "target", baseline_periods: int = 7, causal_filter: str = None, causal_comparison_dates: tuple = None, environment: str = "prod"):
        """
        Initialize flexible anomaly detector
        """
        self.aggregation_days = aggregation_days
        self.threshold = threshold
        self.min_sample_size = min_sample_size
        self.causal_filter = causal_filter
        self.causal_comparison_dates = causal_comparison_dates
        self.environment = environment
        
        # Determine the actual detection mode
        if detection_mode == "vslast" and causal_filter:
            from dashboard_analyzer.deep_research_period import determine_anomaly_mode_for_vslast
            comp_start_date = causal_comparison_dates[0] if causal_comparison_dates and len(causal_comparison_dates) >= 2 else None
            comp_end_date = causal_comparison_dates[1] if causal_comparison_dates and len(causal_comparison_dates) >= 2 else None
            self.detection_mode, _ = determine_anomaly_mode_for_vslast(causal_filter, comp_start_date, comp_end_date)
        else:
            self.detection_mode = detection_mode
            
        self.baseline_periods = baseline_periods
        
        # Initialize target-based detector if enabled
        if self.detection_mode == "target":
            from .target_based_detector import TargetBasedAnomalyDetector
            self.target_detector = TargetBasedAnomalyDetector(
                aggregation_days=aggregation_days,
                min_sample_size=min_sample_size
            )
        else:
            self.target_detector = None
        
        self.monthly_targets_cache = {}
        
    async def analyze_flexible_anomalies(self, data_folder: str, analysis_date=None, reference_period: int = None) -> Tuple[Dict[str, str], Dict[str, float], List[str], Dict[str, Dict[str, float]]]:
        """Analyze flexible anomalies for the most recent period"""
        if analysis_date is None:
            analysis_date = datetime.now()
            
        print(f"🔍 Analyzing flexible anomalies for {analysis_date.strftime('%Y-%m-%d')}")
        all_data = self._load_flexible_data(data_folder)
        if not all_data: return {}, {}, [], {}
            
        periods = self._get_available_periods(all_data)
        if not periods: return {}, {}, [], {}
            
        latest_period = periods[0]
        print(f"🎯 Analyzing period {latest_period}")
        
        return await self.analyze_period(data_folder, latest_period, analysis_date, reference_period)
    
    def _load_flexible_data(self, data_folder: str) -> Dict[str, pd.DataFrame]:
        """Load flexible aggregation data and unify NPS columns across years"""
        data_folder_path = Path(data_folder)
        all_data = {}
        node_mapping = {
            "Global": "Global", "Global/LH": "Global_LH", 
            "Global/LH/Economy": "Global_LH_Economy", "Global/LH/Business": "Global_LH_Business", 
            "Global/LH/Premium": "Global_LH_Premium", "Global/SH": "Global_SH", 
            "Global/SH/Economy": "Global_SH_Economy", "Global/SH/Business": "Global_SH_Business",
            "Global/SH/Economy/IB": "Global_SH_Economy_IB", "Global/SH/Economy/YW": "Global_SH_Economy_YW",
            "Global/SH/Business/IB": "Global_SH_Business_IB", "Global/SH/Business/YW": "Global_SH_Business_YW"
        }
        
        loaded_count = 0
        for logical_path, folder_name in node_mapping.items():
            node_folder = data_folder_path / folder_name
            flexible_file = node_folder / f'flexible_NPS_{self.aggregation_days}d.csv'
            
            if flexible_file.exists():
                try:
                    df = pd.read_csv(flexible_file)
                    if not df.empty:
                        # UNIFY NPS COLUMNS: This handles year boundaries (Dec 2025 -> Jan 2026) correctly
                        year_cols = ['NPS_2026', 'NPS_2025', 'NPS_2024', 'NPS_2019']
                        df['Unified_NPS'] = np.nan
                        for col in year_cols:
                            if col in df.columns:
                                df['Unified_NPS'] = df['Unified_NPS'].fillna(df[col])
                        
                        all_data[logical_path] = df
                        loaded_count += 1
                except Exception as e:
                    print(f"❌ Error loading {logical_path}: {e}")
        
        print(f"📊 Loaded {loaded_count}/{len(node_mapping)} segments")
        return all_data
    
    def _get_available_periods(self, all_data: Dict[str, pd.DataFrame]) -> List[int]:
        """Get list of available periods that have valid NPS data in the unified column"""
        all_periods = set()
        for df in all_data.values():
            if 'Period_Group' in df.columns:
                valid_periods = df[df['Unified_NPS'].notna()]['Period_Group'].unique()
                all_periods.update(valid_periods)
        return sorted(list(all_periods))
    
    async def analyze_period(self, data_folder: str, target_period: int, analysis_date=None, reference_period: int = None) -> Tuple[Dict[str, str], Dict[str, float], List[str], Dict[str, Dict[str, float]]]:
        """Analyze anomalies for a specific period"""
        if analysis_date is None: analysis_date = datetime.now()
        all_data = self._load_flexible_data(data_folder)
        if not all_data: return {}, {}, [], {}
        periods = self._get_available_periods(all_data)
        if target_period not in periods: return {}, {}, periods, {}
            
        if self.detection_mode == "target":
            anomalies, deviations = await self._detect_target_based_anomalies(all_data, target_period, analysis_date)
            nps_values = {}
        elif self.detection_mode == "mean":
            anomalies, deviations, nps_values = self._detect_legacy_anomalies(all_data, target_period, periods, reference_period)
        elif self.detection_mode == "vslast_dynamic":
            anomalies, deviations, nps_values = self._detect_vslast_dynamic_anomalies(all_data, target_period, periods)
        elif self.detection_mode == "vslast_vs_lm":
            anomalies, deviations, nps_values = self._detect_vslast_vs_lm_anomalies(all_data, target_period, periods)
        elif self.detection_mode == "vslast_vs_ly":
            anomalies, deviations, nps_values = self._detect_vslast_vs_ly_anomalies(all_data, target_period, periods)
        else:
            anomalies, deviations, nps_values = self._detect_vslast_anomalies(all_data, target_period, periods)
            
        return anomalies, deviations, periods, nps_values 

    def _detect_legacy_anomalies(self, all_data: Dict[str, pd.DataFrame], target_period: int, all_periods: List[int], reference_period: int = None) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, Dict[str, float]]]:
        """Detect anomalies using mean-based baseline (supports year crossing)"""
        anomalies, deviations, nps_values = {}, {}, {}
        latest_ref = reference_period if reference_period is not None else all_periods[0]
        baseline_periods = [latest_ref + i for i in range(0, self.baseline_periods) if (latest_ref + i) in all_periods]
        
        if len(baseline_periods) < 3: return anomalies, deviations, nps_values
        print(f"📈 Period {target_period}: baseline mean of periods {baseline_periods}")
        
        for node_path, df in all_data.items():
            target_data = df[df['Period_Group'] == target_period]
            if target_data.empty: continue
            
            if target_data.get('Responses', pd.Series([0])).iloc[0] < self.min_sample_size:
                anomalies[node_path] = "S"
                continue
            
            target_nps = target_data['Unified_NPS'].iloc[0]
            baseline_nps_values = df[df['Period_Group'].isin(baseline_periods)]['Unified_NPS'].dropna()
            
            if len(baseline_nps_values) >= 3:
                baseline_avg = baseline_nps_values.mean()
                deviation = target_nps - baseline_avg
                anomalies[node_path] = self._classify_anomaly_new_logic(deviation)
                deviations[node_path] = deviation
                nps_values[node_path] = {'current': target_nps, 'baseline': baseline_avg, 'deviation': deviation}
            else:
                anomalies[node_path] = "?"
                
        return anomalies, deviations, nps_values

    def _detect_vslast_anomalies(self, all_data: Dict[str, pd.DataFrame], target_period: int, all_periods: List[int]) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, Dict[str, float]]]:
        """Detect anomalies comparing against previous period (handles year boundary)"""
        anomalies, deviations, nps_values = {}, {}, {}
        prev = target_period + 1
        if prev not in all_periods: return anomalies, deviations, nps_values
        
        for node_path, df in all_data.items():
            t_data = df[df['Period_Group'] == target_period]
            p_data = df[df['Period_Group'] == prev]
            if t_data.empty or p_data.empty: continue
            if t_data.get('Responses', pd.Series([0])).iloc[0] < self.min_sample_size:
                anomalies[node_path] = "S"
                continue
            
            target_nps, prev_nps = t_data['Unified_NPS'].iloc[0], p_data['Unified_NPS'].iloc[0]
            if pd.isna(target_nps) or pd.isna(prev_nps):
                anomalies[node_path] = "?"
                continue
                
            deviation = target_nps - prev_nps
            anomalies[node_path] = self._classify_anomaly_new_logic(deviation)
            deviations[node_path] = deviation
            nps_values[node_path] = {'current': target_nps, 'baseline': prev_nps, 'deviation': deviation}
            
        return anomalies, deviations, nps_values

    def _detect_vslast_dynamic_anomalies(self, all_data: Dict[str, pd.DataFrame], target_period: int, periods: List[int]) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, Dict[str, float]]]:
        from dashboard_analyzer.deep_research_period import calculate_baseline_period_for_causal_filter
        baseline_period, baseline_description = calculate_baseline_period_for_causal_filter(target_period, self.causal_filter, self.aggregation_days)
        if baseline_period is None:
            if self.causal_filter == "vs Target": return self._detect_target_based_anomalies_sync(all_data, target_period)
            if self.causal_filter == "vs Sel. Period" and self.causal_comparison_dates:
                return self._detect_vslast_with_selected_period(all_data, target_period, self.causal_comparison_dates[0], self.causal_comparison_dates[1])
            baseline_period, baseline_description = target_period + 1, "período anterior"
        return self._detect_vslast_with_baseline_period(all_data, target_period, baseline_period, baseline_description)

    def _detect_vslast_vs_lm_anomalies(self, all_data, target_period, all_periods):
        return self._detect_vslast_with_baseline_period(all_data, target_period, target_period + 4, "mes natural anterior")

    def _detect_vslast_vs_ly_anomalies(self, all_data, target_period, all_periods):
        return self._detect_vslast_with_baseline_period(all_data, target_period, target_period + 52, "mismo período año anterior")

    def _detect_vslast_with_baseline_period(self, all_data, target_p, base_p, desc):
        anomalies, deviations, nps_values = {}, {}, {}
        for node_path, df in all_data.items():
            t_data, b_data = df[df['Period_Group'] == target_p], df[df['Period_Group'] == base_p]
            if t_data.empty or b_data.empty: continue
            if t_data.get('Responses', pd.Series([0])).iloc[0] < self.min_sample_size:
                anomalies[node_path] = "S"
                continue
            t_nps, b_nps = t_data['Unified_NPS'].iloc[0], b_data['Unified_NPS'].iloc[0]
            if pd.isna(t_nps) or pd.isna(b_nps): continue
            deviation = t_nps - b_nps
            anomalies[node_path] = self._classify_anomaly_new_logic(deviation)
            deviations[node_path] = deviation
            nps_values[node_path] = {'current': t_nps, 'baseline': b_nps, 'deviation': deviation, 'baseline_description': desc}
        return anomalies, deviations, nps_values

    async def _detect_target_based_anomalies(self, all_data, target_period, analysis_date):
        anomalies, deviations = {}, {}
        period_start, period_end = self.target_detector._calculate_period_dates(analysis_date, target_period)
        required_months = self.target_detector._get_required_months(period_start, period_end)
        
        monthly_targets = {}
        for month in required_months:
            if month not in self.monthly_targets_cache:
                try:
                    targets = await self.target_detector.get_monthly_targets("Global", [month])
                    self.monthly_targets_cache[month] = targets.get(month)
                except: self.monthly_targets_cache[month] = None
            monthly_targets[month] = self.monthly_targets_cache[month]
            
        for node_path, df in all_data.items():
            t_data = df[df['Period_Group'] == target_period]
            if t_data.empty or t_data.get('Responses', pd.Series([0])).iloc[0] < self.min_sample_size: continue
            actual_nps = t_data['Unified_NPS'].iloc[0]
            if pd.isna(actual_nps): continue
            try:
                ctx = self.target_detector.determine_target_context(period_start, period_end, node_path, monthly_targets)
                status, dev, _ = self.target_detector.classify_anomaly_vs_target(actual_nps, ctx)
                anomalies[node_path] = 'N' if status == 'Normal' else status
                deviations[node_path] = dev
            except: pass
        return anomalies, deviations

    def _classify_anomaly_new_logic(self, deviation: float) -> str:
        if deviation < 0: return "-"
        return "+" if deviation > 4 else "N"

    def get_period_summary(self, data_folder: str, periods: List[int]) -> pd.DataFrame:
        summary_data = []
        for period in periods[:7]:
            import asyncio
            anoms, _, _, _ = asyncio.run(self.analyze_period(data_folder, period))
            pos = sum(1 for a in anoms.values() if a == "+")
            neg = sum(1 for a in anoms.values() if a == "-")
            norm = sum(1 for a in anoms.values() if a == "N")
            summary_data.append({'Period': f"Period {period}", 'Status': "🚨 Alert" if (pos > 0 or neg > 0) else "✅ Normal", '+': pos, '-': neg, 'N': norm, 'Total': pos + neg + norm})
        return pd.DataFrame(summary_data)
