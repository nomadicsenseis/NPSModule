import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class OperationalDataAnalyzer:
    """
    Analyzes operational metrics (OTP, Load Factor, Misconex, Mishandling) 
    to provide explanations for NPS anomalies
    """
    
    def __init__(self, data_base_path: str = "tables"):
        self.data_base_path = Path(data_base_path)
        self.operative_data: Dict[str, pd.DataFrame] = {}
        self.verbatims_data: Dict[str, pd.DataFrame] = {}
        
    def load_operative_data(self, date_folder: str, node_paths: List[str]):
        """Load operative data for specified nodes"""
        print("📊 Loading operative data...")
        
        for node_path in node_paths:
            file_path = self.data_base_path / date_folder / node_path / "daily_operative.csv"
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    # Clean column names (remove brackets but keep the content)
                    new_columns = []
                    for col in df.columns:
                        if '[' in col and ']' in col:
                            # Extract content inside brackets
                            start = col.find('[')
                            end = col.find(']')
                            clean_name = col[start+1:end]
                            new_columns.append(clean_name)
                        else:
                            new_columns.append(col)
                    df.columns = new_columns
                    
                    # Convert date column
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date']).dt.date
                        df.rename(columns={'Date': 'Date_Master'}, inplace=True)
                    
                    self.operative_data[node_path] = df
                    print(f"✅ Loaded operative data for {node_path}: {len(df)} records")
                    
                except Exception as e:
                    print(f"❌ Error loading operative data for {node_path}: {e}")
            else:
                print(f"⚠️  No operative data found for {node_path}")
    
    def load_verbatims_data(self, date_folder: str, node_paths: List[str]):
        """Load verbatims data for specified nodes (if available)"""
        print("💬 Loading verbatims data...")
        
        for node_path in node_paths:
            file_path = self.data_base_path / date_folder / node_path / "daily_verbatims.csv"
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    self.verbatims_data[node_path] = df
                    print(f"✅ Loaded verbatims for {node_path}: {len(df)} records")
                except Exception as e:
                    print(f"❌ Error loading verbatims for {node_path}: {e}")
            else:
                print(f"⚠️  No verbatims data found for {node_path}")
    
    def analyze_operative_metrics(self, node_path: str, target_date: str) -> Dict[str, any]:
        """
        Analyze operative metrics for a specific node and date
        Returns metrics comparison and potential issues
        """
        if node_path not in self.operative_data:
            return {"error": "No operative data available"}
            
        df = self.operative_data[node_path]
        target_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
        
        # Filter data for the target date
        day_data = df[df['Date_Master'] == target_date_obj]
        
        if day_data.empty:
            return {"error": f"No data for date {target_date}"}
        
        # Calculate 7-day average for comparison (like NPS)
        date_range = pd.date_range(end=target_date_obj, periods=7).date
        week_data = df[df['Date_Master'].isin(date_range)]
        
        analysis = {
            "date": target_date,
            "node_path": node_path,
            "metrics": {}
        }
        
        # Analyze each metric
        metrics_to_analyze = ['Load_Factor', 'OTP15_adjusted', 'Misconex', 'Mishandling']
        
        for metric in metrics_to_analyze:
            if metric in df.columns:
                day_values = day_data[metric].dropna()
                week_values = week_data[metric].dropna()
                
                if not day_values.empty and not week_values.empty:
                    day_avg = day_values.mean()
                    week_avg = week_values.mean()
                    delta = day_avg - week_avg
                    
                    # Determine if significant deviation (using different thresholds per metric)
                    threshold = self._get_metric_threshold(metric)
                    is_significant = abs(delta) > threshold
                    
                    analysis["metrics"][metric] = {
                        "day_value": round(day_avg, 2),
                        "week_average": round(week_avg, 2),
                        "delta": round(delta, 2),
                        "is_significant": is_significant,
                        "direction": "higher" if delta > 0 else "lower",
                        "threshold": threshold
                    }
        
        return analysis
    
    def _get_metric_threshold(self, metric: str) -> float:
        """Get significance threshold for each metric (lowered for better sensitivity)"""
        thresholds = {
            'Load_Factor': 3.0,    # 3% points (reduced from 5)
            'OTP15_adjusted': 3.0,  # 3% points (reduced from 5)
            'Misconex': 1.0,       # 1% points (reduced from 2)
            'Mishandling': 0.5     # 0.5 per 1000 passengers (reduced from 1)
        }
        return thresholds.get(metric, 2.0)
    
    def _get_metric_impact(self, metric: str, direction: str) -> str:
        """Get the likely impact of metric deviation on NPS"""
        impacts = {
            'Load_Factor': {
                'higher': "Higher load factor may lead to crowded conditions, reduced service quality",
                'lower': "Lower load factor suggests underutilized capacity, possible route issues"
            },
            'OTP15_adjusted': {
                'higher': "Better on-time performance should positively impact customer satisfaction",
                'lower': "Poor punctuality likely causing customer dissatisfaction and negative NPS"
            },
            'Misconex': {
                'higher': "Increased connection issues causing customer frustration",
                'lower': "Improved connection performance supporting better customer experience"
            },
            'Mishandling': {
                'higher': "Increased baggage mishandling causing significant customer dissatisfaction",
                'lower': "Improved baggage handling supporting better customer experience"
            }
        }
        return impacts.get(metric, {}).get(direction, "Unknown impact")
    
    def generate_explanation(self, node_path: str, target_date: str, nps_anomaly_type: str) -> str:
        """
        Generate a comprehensive explanation for an NPS anomaly
        """
        analysis = self.analyze_operative_metrics(node_path, target_date)
        
        if "error" in analysis:
            return f"⚠️  Cannot analyze {node_path} for {target_date}: {analysis['error']}"
        
        explanation_parts = []
        explanation_parts.append(f"🔍 **Operational Analysis for {node_path} on {target_date}**")
        explanation_parts.append(f"NPS Anomaly: {nps_anomaly_type}")
        explanation_parts.append("")
        
        # Show all available metrics (even if not significant)
        explanation_parts.append("**Available Operational Metrics:**")
        
        all_metrics = []
        significant_metrics = []
        supporting_metrics = []
        contradicting_metrics = []
        
        for metric, data in analysis["metrics"].items():
            all_metrics.append((metric, data))
            
            # Show metric info
            direction_text = "📈 higher" if data["direction"] == "higher" else "📉 lower"
            significance_marker = "⚠️ **SIGNIFICANT**" if data["is_significant"] else ""
            
            explanation_parts.append(
                f"• **{metric}**: {data['day_value']} vs {data['week_average']} (7-day avg) "
                f"→ {direction_text} by {abs(data['delta'])} pts {significance_marker}"
            )
            
            if data["is_significant"]:
                significant_metrics.append((metric, data))
                
                # Determine if metric supports or contradicts the NPS anomaly
                if self._metric_supports_anomaly(metric, data["direction"], nps_anomaly_type):
                    supporting_metrics.append((metric, data))
                else:
                    contradicting_metrics.append((metric, data))
        
        explanation_parts.append("")
        
        # Generate explanation based on findings
        if significant_metrics:
            explanation_parts.append("**🎯 Analysis Results:**")
            
            if supporting_metrics:
                metric_names = [metric for metric, _ in supporting_metrics]
                explanation_parts.append(
                    f"✅ **Operational explanation found**: The {nps_anomaly_type} NPS anomaly is "
                    f"likely explained by significant changes in: **{', '.join(metric_names)}**"
                )
                
                for metric, data in supporting_metrics:
                    impact = self._get_metric_impact(metric, data["direction"])
                    explanation_parts.append(f"   • *{impact}*")
                
            if contradicting_metrics:
                explanation_parts.append("")
                metric_names = [metric for metric, _ in contradicting_metrics]
                explanation_parts.append(
                    f"🤔 **Contradictory signals**: {', '.join(metric_names)} show improvements "
                    "that don't align with the NPS anomaly"
                )
                explanation_parts.append("   • This suggests multiple factors may be at play")
        
        else:
            explanation_parts.append("**❌ No Operational Explanation Found**")
            explanation_parts.append(
                f"The {nps_anomaly_type} NPS anomaly cannot be explained by significant changes in "
                "available operational metrics"
            )
            
            # Check if we have limited data
            if len(analysis["metrics"]) < 4:
                missing_metrics = []
                if 'Load_Factor' not in analysis["metrics"]:
                    missing_metrics.append("Load_Factor")
                if 'Misconex' not in analysis["metrics"]:
                    missing_metrics.append("Misconex") 
                if 'Mishandling' not in analysis["metrics"]:
                    missing_metrics.append("Mishandling")
                    
                if missing_metrics:
                    explanation_parts.append(f"   ⚠️  **Limited data**: Missing metrics: {', '.join(missing_metrics)}")
            
            explanation_parts.append("")
            explanation_parts.append("**🔍 Possible alternative causes:**")
            explanation_parts.append("   • Service quality issues (staff, cleanliness, food)")
            explanation_parts.append("   • External factors (weather, strikes, IT issues)")
            explanation_parts.append("   • Route/schedule specific problems")
            explanation_parts.append("   • Customer perception or expectation changes")
        
        return "\n".join(explanation_parts)
    
    def _metric_supports_anomaly(self, metric: str, direction: str, nps_anomaly_type: str) -> bool:
        """Determine if a metric deviation supports the observed NPS anomaly"""
        
        # For negative NPS anomalies, these directions support the anomaly
        negative_supporting = {
            'Load_Factor': 'higher',      # Higher LF = worse service
            'OTP15_adjusted': 'lower',    # Lower OTP = worse experience  
            'Misconex': 'higher',         # Higher misconex = worse experience
            'Mishandling': 'higher'       # Higher mishandling = worse experience
        }
        
        # For positive NPS anomalies, opposite directions support
        positive_supporting = {
            'Load_Factor': 'lower',       # Lower LF = better service
            'OTP15_adjusted': 'higher',   # Higher OTP = better experience
            'Misconex': 'lower',          # Lower misconex = better experience  
            'Mishandling': 'lower'        # Lower mishandling = better experience
        }
        
        if nps_anomaly_type == "negative":
            return negative_supporting.get(metric) == direction
        elif nps_anomaly_type == "positive":
            return positive_supporting.get(metric) == direction
        else:
            return False
    
    def get_available_nodes(self, date_folder: str) -> List[str]:
        """Get list of nodes that have operative data available"""
        nodes = []
        base_folder = self.data_base_path / date_folder
        
        if base_folder.exists():
            for path in base_folder.rglob("daily_operative.csv"):
                # Extract node path relative to date folder
                relative_path = path.parent.relative_to(base_folder)
                node_path = str(relative_path).replace("\\", "/")
                if node_path != ".":
                    nodes.append(node_path)
                else:
                    nodes.append("Global")
        
        return sorted(nodes)

    def get_specific_explanations(self, node_path: str, target_date: str, nps_anomaly_type: str) -> Dict[str, str]:
        """
        Generate specific one-line explanations for OTP and Load Factor
        Returns dict with 'otp_explanation' and 'load_factor_explanation'
        """
        analysis = self.analyze_operative_metrics(node_path, target_date)
        
        explanations = {
            'otp_explanation': "No OTP data available",
            'load_factor_explanation': "No Load Factor data available"
        }
        
        if "error" in analysis:
            return explanations
        
        metrics = analysis.get("metrics", {})
        
        # OTP15_adjusted explanation
        if 'OTP15_adjusted' in metrics:
            otp_data = metrics['OTP15_adjusted']
            if otp_data['is_significant']:
                direction = "improved" if otp_data['direction'] == 'higher' else "worsened"
                supports = self._metric_supports_anomaly('OTP15_adjusted', otp_data['direction'], nps_anomaly_type)
                correlation = "explains" if supports else "contradicts"
                explanations['otp_explanation'] = (
                    f"OTP {direction} by {abs(otp_data['delta'])}pts ({otp_data['day_value']}% vs {otp_data['week_average']}%) "
                    f"- {correlation} NPS anomaly"
                )
            else:
                explanations['otp_explanation'] = (
                    f"OTP stable at {otp_data['day_value']}% (Δ{otp_data['delta']:+.1f}pts vs 7-day avg) "
                    f"- no significant impact"
                )
        
        # Load_Factor explanation
        if 'Load_Factor' in metrics:
            lf_data = metrics['Load_Factor']
            if lf_data['is_significant']:
                direction = "increased" if lf_data['direction'] == 'higher' else "decreased"
                supports = self._metric_supports_anomaly('Load_Factor', lf_data['direction'], nps_anomaly_type)
                correlation = "explains" if supports else "contradicts"
                explanations['load_factor_explanation'] = (
                    f"Load Factor {direction} by {abs(lf_data['delta'])}pts ({lf_data['day_value']}% vs {lf_data['week_average']}%) "
                    f"- {correlation} NPS anomaly"
                )
            else:
                explanations['load_factor_explanation'] = (
                    f"Load Factor stable at {lf_data['day_value']}% (Δ{lf_data['delta']:+.1f}pts vs 7-day avg) "
                    f"- no significant impact"
                )
        
        return explanations
