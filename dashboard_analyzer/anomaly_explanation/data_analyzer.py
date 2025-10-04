"""
Data Analyzer - Análisis de datos operativos y métricas complementarias
Proporciona contexto operativo para explicar anomalías NPS
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from pathlib import Path
import os

logger = logging.getLogger(__name__)



class OperationalDataAnalyzer:
    """
    Analizador especializado para datos operativos con soporte para múltiples modos de comparación
    Diseñado específicamente para el análisis de métricas operativas en contexto de anomalías NPS
    """

    def __init__(self, comparison_mode: str = "mean", comparison_start_date: datetime = None, comparison_end_date: datetime = None, aggregation_days: int = 1, baseline_periods: int = 4):
        """
        Inicializa el analizador operativo

        Args:
            comparison_mode: Modo de comparación - "mean", "vslast", "vslast_dynamic", "target"
            comparison_start_date: Fecha de inicio específica para comparación (usado en "vs Sel. Period")
            comparison_end_date: Fecha de fin específica para comparación (usado en "vs Sel. Period")
            aggregation_days: Días de agregación para análisis flexible
            baseline_periods: Número de períodos a usar para baseline en modo mean
        """
        self.comparison_mode = comparison_mode
        self.comparison_start_date = comparison_start_date
        self.comparison_end_date = comparison_end_date
        self.aggregation_days = aggregation_days
        self.baseline_periods = baseline_periods
        self.operative_data = {}  # Almacena datos operativos por nodo
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"OperationalDataAnalyzer initialized with comparison_mode: {comparison_mode}, aggregation_days: {aggregation_days}, baseline_periods: {baseline_periods}")

    def load_operative_data(self, data_folder: str = None, node_path: str = None, aggregation_days: int = 7) -> bool:
        """
        Carga datos operativos desde archivos CSV
        Soporta tanto formato LEGACY como FLEXIBLE
        
        Args:
            data_folder: Carpeta con datos CSV (si None, busca automáticamente)
            node_path: Ruta del nodo a cargar (si None, usa "Global/SH/Economy" por defecto)
            aggregation_days: Días de agregación para buscar archivos flexible
            
        Returns:
            True si se cargaron datos exitosamente
        """
        try:
            # Defaults
            if data_folder is None:
                # Buscar carpeta de datos más reciente
                from pathlib import Path
                tables_path = Path("/app/tables")
                if tables_path.exists():
                    # Buscar carpeta available_* más reciente
                    available_folders = sorted(tables_path.glob("available_*"))
                    if available_folders:
                        data_folder = str(available_folders[-1])
                        self.logger.info(f"Auto-detected data folder: {data_folder}")
                    else:
                        self.logger.error("No available_* folders found in /app/tables")
                        return False
                else:
                    self.logger.error("Tables path /app/tables not found")
                    return False
                    
            if node_path is None:
                node_path = "Global/SH/Economy"
                
            data_folder_path = Path(data_folder)
            
            # Intentar formato FLEXIBLE primero
            operative_files = list(data_folder_path.glob(f"flexible_operative_{aggregation_days}d*.csv"))
            if operative_files:
                # Tomar el archivo más reciente
                operative_file = sorted(operative_files)[-1]
                self.logger.info(f"Loading FLEXIBLE operative data from: {operative_file}")
                is_flexible = True
            else:
                # Intentar formato LEGACY
                operative_file = data_folder_path / "operative.csv"
                if operative_file.exists():
                    self.logger.info(f"Loading LEGACY operative data from: {operative_file}")
                    is_flexible = False
                else:
                    self.logger.warning(f"No operative data files found in {data_folder}")
                    return False
            
            # Cargar el archivo CSV
            operative_data = pd.read_csv(operative_file)
            
            if operative_data.empty:
                self.logger.warning(f"Empty operative data loaded from {operative_file}")
                return False
            
            # Filtrar por node_path si es necesario
            if 'node_path' in operative_data.columns:
                node_data = operative_data[operative_data['node_path'] == node_path]
                if node_data.empty:
                    self.logger.warning(f"No data found for node_path: {node_path}")
                    return False
                self.operative_data[node_path] = node_data
            else:
                # Asumir que todos los datos son para este nodo
                self.operative_data[node_path] = operative_data
            
            self.logger.info(f"Loaded {len(self.operative_data[node_path])} operative records for {node_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading operative data: {e}")
            return False

    def _clean_dax_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean DAX column names and convert to standard format.
        Similar to pbi_collector.py logic.
        """
        # Limpiar nombres de columnas DAX
        if 'Date_Master[Date' in data.columns:
            data.rename(columns={'Date_Master[Date': 'Date_Master'}, inplace=True)
        
        # Convertir Date_Master a Date para compatibilidad
        if 'Date_Master' in data.columns:
            data['Date'] = pd.to_datetime(data['Date_Master']).dt.date
        elif 'Date_Master[Date]' in data.columns:
            data['Date'] = pd.to_datetime(data['Date_Master[Date]']).dt.date
        
        # Para datos operativos flexibles, usar Min_Date como Date principal
        elif 'Min_Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Min_Date']).dt.date
            self.logger.info(f"📅 Using Min_Date as primary Date column for operative data")
            
        # Limpiar otros nombres de columnas comunes
        column_mapping = {
            'Date_Master[Date]': 'Date',
            'Company_Master[Company': 'Company',
            'Cabin_Master[Cabin_Show': 'Cabin',
            'Haul_Master[Haul_Aggr': 'Haul'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in data.columns:
                data.rename(columns={old_col: new_col}, inplace=True)
        
        return data

    def _analyze_precalculated_comparison(self, data: pd.DataFrame, node_path: str) -> Dict[str, Any]:
        """Analyze pre-calculated comparison data from simplified vs Sel. Period query"""
        try:
            self.logger.info(f"📊 Processing pre-calculated comparison data for {node_path}")
            
            metrics = {}
            summary_parts = []
            
            for _, row in data.iterrows():
                metric_name = row['Metric']
                current_val = pd.to_numeric(row['Current_Value'], errors='coerce')
                comparison_val = pd.to_numeric(row['Comparison_Value'], errors='coerce')
                difference = pd.to_numeric(row['Difference'], errors='coerce')
                change_pct = pd.to_numeric(row['Change_Pct'], errors='coerce')
                
                if pd.notna(current_val) and pd.notna(comparison_val):
                    # Determine direction and significance
                    direction = 'higher' if difference > 0 else 'lower'
                    is_significant = abs(change_pct) > 5  # Change > 5% is significant
                    
                    # Map metric names to standard format
                    standard_name = metric_name
                    if metric_name == "OTP15_adjusted":
                        standard_name = "OTP15_adjusted"
                    elif metric_name == "Load_Factor":
                        standard_name = "Load_Factor"
                    
                    metrics[standard_name] = {
                        'current': round(current_val, 2),
                        'comparison': round(comparison_val, 2),
                        'difference': round(difference, 2),
                        'change_pct': round(change_pct, 1),
                        # Keys expected by causal agent
                        'current_value': round(current_val, 2),
                        'previous_value': round(comparison_val, 2),
                        'day_value': round(current_val, 2),
                        'delta': round(difference, 2),
                        'direction': direction,
                        'is_significant': is_significant
                    }
                    
                    # Add to summary if significant
                    if is_significant:
                        direction_text = "aumentó" if difference > 0 else "disminuyó"
                        metric_display = self._get_metric_display_name(standard_name)
                        comparison_desc = f"período seleccionado ({self.comparison_start_date.strftime('%Y-%m-%d')} a {self.comparison_end_date.strftime('%Y-%m-%d')})"
                        summary_parts.append(f"{metric_display} {direction_text} {abs(change_pct):.1f}% vs {comparison_desc}")
                    
                    self.logger.info(f"📊 {metric_name}: {current_val:.2f} vs {comparison_val:.2f} (Δ{difference:+.2f}, {change_pct:+.1f}%)")
                else:
                    # Handle missing data
                    metrics[standard_name] = {
                        'current': current_val if pd.notna(current_val) else None,
                        'comparison': comparison_val if pd.notna(comparison_val) else None,
                        'difference': 0,
                        'change_pct': 0,
                        'current_value': current_val if pd.notna(current_val) else None,
                        'previous_value': comparison_val if pd.notna(comparison_val) else None,
                        'day_value': current_val if pd.notna(current_val) else None,
                        'delta': 0,
                        'direction': 'unchanged',
                        'is_significant': False,
                        'error': 'Missing data'
                    }
            
            # Create summary
            if summary_parts:
                comparison_desc = f"período seleccionado ({self.comparison_start_date.strftime('%Y-%m-%d')} a {self.comparison_end_date.strftime('%Y-%m-%d')})"
                summary = f"Comparación vs {comparison_desc}: " + "; ".join(summary_parts)
            else:
                summary = f"Sin cambios significativos vs período seleccionado"
            
            return {
                'metrics': metrics,
                'summary': summary,
                'comparison_info': {
                    'comparison_period': f"{self.comparison_start_date.strftime('%Y-%m-%d')} to {self.comparison_end_date.strftime('%Y-%m-%d')}",
                    'comparison_type': 'precalculated_dax'
                }
            }
            
        except Exception as e:
            return {"error": f"Error in precalculated comparison analysis: {str(e)}"}

    def analyze_operative_metrics(self, node_path: str, target_date: str) -> Dict[str, Any]:
        """
        Analiza métricas operativas para un nodo y fecha específicos

        Args:
            node_path: Ruta del nodo en el árbol jerárquico
            target_date: Fecha objetivo para el análisis (YYYY-MM-DD)

        Returns:
            Dict con análisis de métricas operativas
        """
        try:
            if node_path not in self.operative_data:
                return {"error": f"No operative data available for {node_path}"}

            data = self.operative_data[node_path]
            if data.empty:
                return {"error": f"Empty operative data for {node_path}"}

            # Limpiar columnas DAX y asegurar formato correcto
            data = self._clean_dax_columns(data)

            # Convertir target_date a datetime para comparaciones
            try:
                target_dt = pd.to_datetime(target_date)
            except:
                return {"error": f"Invalid target_date format: {target_date}"}

            # First check if this is pre-calculated vs Sel. Period data (from simplified query)
            # Handle both with and without brackets in column names
            metric_col = 'Metric' if 'Metric' in data.columns else '[Metric]' if '[Metric]' in data.columns else None
            current_col = 'Current_Value' if 'Current_Value' in data.columns else '[Current_Value]' if '[Current_Value]' in data.columns else None
            comparison_col = 'Comparison_Value' if 'Comparison_Value' in data.columns else '[Comparison_Value]' if '[Comparison_Value]' in data.columns else None
            diff_col = 'Difference' if 'Difference' in data.columns else '[Difference]' if '[Difference]' in data.columns else None
            
            if metric_col and current_col and comparison_col and diff_col:
                self.logger.info(f"🎯 Detected precalculated vs Sel. Period data format")
                # Clean column names if they have brackets
                if metric_col.startswith('['):
                    data = data.rename(columns={
                        '[Metric]': 'Metric',
                        '[Current_Value]': 'Current_Value',
                        '[Comparison_Value]': 'Comparison_Value', 
                        '[Difference]': 'Difference',
                        '[Change_Pct]': 'Change_Pct'
                    })
                    self.logger.info(f"✅ Cleaned column names (removed brackets)")
                return self._analyze_precalculated_comparison(data, node_path)

            # Otherwise, ensure date columns are datetime for regular analysis
            # Support both legacy format (Date column) and flexible format (Min_Date/Max_Date columns)
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
            elif 'Min_Date' in data.columns and 'Max_Date' in data.columns:
                # Flexible aggregated data - convert date columns
                data['Min_Date'] = pd.to_datetime(data['Min_Date'])
                data['Max_Date'] = pd.to_datetime(data['Max_Date'])
                self.logger.info(f"Using flexible aggregated data format with Period_Group")
            else:
                return {"error": "No 'Date' or 'Min_Date'/'Max_Date' columns found in operative data"}
            
            # Análisis según el modo de comparación
            if self.comparison_mode in ["vslast", "vslast_dynamic"]:
                return self._analyze_vslast(data, target_dt, node_path)
            elif self.comparison_mode == "mean":
                return self._analyze_mean(data, target_dt, node_path)
            elif self.comparison_mode == "target":
                return self._analyze_target(data, target_dt, node_path)
            else:
                return {"error": f"Unsupported comparison_mode: {self.comparison_mode}"}

        except Exception as e:
            self.logger.error(f"Error analyzing operative metrics: {e}")
            return {"error": str(e)}

    def get_specific_explanations(self, node_path: str, target_date: str, anomaly_type: str, aggregation_days: int = 7) -> Dict[str, str]:
        """
        Obtiene explicaciones específicas basadas en el tipo de anomalía
        Adaptado para datos agregados por período
        
        Returns:
            Dict con claves 'otp_explanation', 'load_factor_explanation', 'mishandling_explanation'

        Args:
            node_path: Ruta del nodo
            target_date: Fecha objetivo
            anomaly_type: Tipo de anomalía ('positive', 'negative', 'unknown')
            aggregation_days: Días de agregación para el análisis

        Returns:
            Diccionario con explicaciones específicas por métrica
        """
        try:
            result = {
                'otp_explanation': "No OTP data available",
                'load_factor_explanation': "No Load Factor data available", 
                'mishandling_explanation': "No Mishandling data available",
                'misconex_explanation': "No Misconex data available"
            }

            if node_path not in self.operative_data:
                return result

            data = self.operative_data[node_path]
            if data.empty:
                return result

            # Limpiar columnas DAX y asegurar formato correcto
            data = self._clean_dax_columns(data)
            target_dt = pd.to_datetime(target_date)

            # Para datos agregados por período, buscar el período que contiene la fecha objetivo
            target_row = None
            
            if 'Period_Group' in data.columns and 'Min_Date' in data.columns and 'Max_Date' in data.columns:
                # Convertir Min_Date y Max_Date a datetime para comparación
                data['Min_Date'] = pd.to_datetime(data['Min_Date'])
                data['Max_Date'] = pd.to_datetime(data['Max_Date'])
                
                # Encontrar el período que contiene la fecha objetivo
                target_period = data[
                    (data['Min_Date'] <= target_dt) & 
                    (data['Max_Date'] >= target_dt)
                ]
                
                if target_period.empty:
                    # Si no hay período exacto, buscar el más cercano
                    data['distance'] = abs((data['Min_Date'] + (data['Max_Date'] - data['Min_Date'])/2) - target_dt).dt.days
                    target_period = data.loc[data['distance'].idxmin():data['distance'].idxmin()]
                    self.logger.info(f"📅 No exact period found, using closest period for {target_dt.date()}")
                
                target_row = target_period.iloc[0]
                
            else:
                # Lógica original para datos por fecha exacta
                if 'Date' not in data.columns:
                    return result
                
                target_data = data[data['Date'] == target_dt]
                
                if target_data.empty:
                    return result
                
                target_row = target_data.iloc[0]

            # Obtener análisis completo para contexto de cambios
            full_analysis = self.analyze_operative_metrics(node_path, target_date)
            metrics_context = full_analysis.get('metrics', {}) if 'error' not in full_analysis else {}

            # Análisis específico por métrica con contexto de cambios
            load_factor_explanation = self._explain_load_factor(target_row, anomaly_type, metrics_context.get('Load_Factor', {}))
            if load_factor_explanation:
                result['load_factor_explanation'] = load_factor_explanation

            otp_explanation = self._explain_otp(target_row, anomaly_type, metrics_context.get('OTP15_adjusted', {}))
            if otp_explanation:
                result['otp_explanation'] = otp_explanation

            mishandling_explanation = self._explain_mishandling(target_row, anomaly_type, metrics_context.get('Mishandling', {}))
            if mishandling_explanation:
                result['mishandling_explanation'] = mishandling_explanation

            misconex_explanation = self._explain_misconex(target_row, anomaly_type, metrics_context.get('Misconex', {}))
            if misconex_explanation:
                result['misconex_explanation'] = misconex_explanation

            return result

        except Exception as e:
            self.logger.error(f"Error getting specific explanations: {e}")
            return {
                'otp_explanation': f"Error analyzing OTP: {str(e)}",
                'load_factor_explanation': f"Error analyzing Load Factor: {str(e)}",
                'mishandling_explanation': f"Error analyzing Mishandling: {str(e)}",
                'misconex_explanation': f"Error analyzing Misconex: {str(e)}"
            }

    def _analyze_vslast(self, data: pd.DataFrame, target_dt: pd.Timestamp, node_path: str) -> Dict[str, Any]:
        """Análisis vs período anterior - Adaptado para datos agregados por período"""
        try:
            # Check if we have specific comparison dates (for "vs Sel. Period" in vslast_dynamic mode)
            if (self.comparison_mode == "vslast_dynamic" and 
                self.comparison_start_date and self.comparison_end_date):
                self.logger.info(f"🎯 Using specific comparison period: {self.comparison_start_date.strftime('%Y-%m-%d')} to {self.comparison_end_date.strftime('%Y-%m-%d')}")
                return self._analyze_vslast_with_specific_dates(data, target_dt, node_path)
            
            # Para datos agregados por período, necesitamos encontrar el período que contiene la fecha objetivo
            if 'Period_Group' in data.columns and 'Min_Date' in data.columns and 'Max_Date' in data.columns:
                # Convertir Min_Date y Max_Date a datetime para comparación
                data['Min_Date'] = pd.to_datetime(data['Min_Date'])
                data['Max_Date'] = pd.to_datetime(data['Max_Date'])
                
                # Encontrar el período que contiene la fecha objetivo
                current_period = data[
                    (data['Min_Date'] <= target_dt) & 
                    (data['Max_Date'] >= target_dt)
                ]
                
                if current_period.empty:
                    # Si no hay período exacto, buscar el más cercano
                    data['distance'] = abs((data['Min_Date'] + (data['Max_Date'] - data['Min_Date'])/2) - target_dt).dt.days
                    current_period = data.loc[data['distance'].idxmin():data['distance'].idxmin()]
                    self.logger.info(f"📅 No exact period found, using closest period for {target_dt.date()}")
                
                current_row = current_period.iloc[0]
                
                # Buscar período anterior (Period_Group mayor = más reciente, menor = más anterior)
                current_period_group = current_row['Period_Group']
                previous_period = data[data['Period_Group'] > current_period_group].sort_values('Period_Group', ascending=True)
                
                if previous_period.empty:
                    return {"error": "No previous period available for comparison"}
                
                previous_row = previous_period.iloc[0]
                
                self.logger.info(f"📊 Comparing Period {current_period_group} vs Period {previous_row['Period_Group']}")
                
            else:
                # Lógica original para datos por fecha exacta
                current_data = data[data['Date'] == target_dt]
                
                if current_data.empty:
                    return {"error": f"No data for target date {target_dt.date()}"}
                
                current_row = current_data.iloc[0]
                
                # Obtener período anterior (día anterior con datos)
                previous_data = data[data['Date'] < target_dt].sort_values('Date', ascending=False)
                
                if previous_data.empty:
                    return {"error": "No previous data available for comparison"}
                
                previous_row = previous_data.iloc[0]

            # Calcular diferencias (misma lógica para ambos casos)
            metrics = {}
            for col in ['Load_Factor', 'OTP15_adjusted', 'Mishandling', 'Misconex']:
                if col in data.columns:
                    current_val = pd.to_numeric(current_row.get(col), errors='coerce')
                    previous_val = pd.to_numeric(previous_row.get(col), errors='coerce')

                    if pd.notna(current_val) and pd.notna(previous_val):
                        difference = current_val - previous_val
                        change_pct = round((difference / previous_val) * 100, 1) if previous_val != 0 else 0
                        
                        # Determinar dirección y significancia
                        direction = 'higher' if difference > 0 else 'lower'
                        is_significant = abs(change_pct) > 5  # Cambio > 5% es significativo
                        
                        metrics[col] = {
                            'current': round(current_val, 2),
                            'previous': round(previous_val, 2),
                            'difference': round(difference, 2),
                            'change_pct': change_pct,
                            # Claves que espera el causal agent
                            'current_value': round(current_val, 2),
                            'previous_value': round(previous_val, 2),
                            'day_value': round(current_val, 2),
                            'delta': round(difference, 2),
                            'direction': direction,
                            'is_significant': is_significant
                        }

            # Determinar fecha de referencia para el resultado
            if 'Period_Group' in data.columns:
                target_date_str = current_row.get('Min_Date', target_dt).strftime('%Y-%m-%d') if hasattr(current_row.get('Min_Date', target_dt), 'strftime') else str(current_row.get('Min_Date', target_dt))[:10]
                previous_date_str = previous_row.get('Min_Date', '').strftime('%Y-%m-%d') if hasattr(previous_row.get('Min_Date', ''), 'strftime') else str(previous_row.get('Min_Date', ''))[:10]
            else:
                target_date_str = target_dt.date().isoformat()
                previous_date_str = previous_row['Date'].date().isoformat()

            return {
                'mode': 'vslast',
                'target_date': target_date_str,
                'previous_date': previous_date_str,
                'metrics': metrics,
                'summary': self._create_vslast_summary(metrics)
            }

        except Exception as e:
            return {"error": f"Error in vslast analysis: {str(e)}"}

    def _analyze_vslast_with_specific_dates(self, data: pd.DataFrame, target_dt: pd.Timestamp, node_path: str) -> Dict[str, Any]:
        """Análisis vs período específico seleccionado - Para "vs Sel. Period" mode"""
        try:
            # Para datos agregados por período, necesitamos encontrar el período que contiene la fecha objetivo
            if 'Period_Group' in data.columns and 'Min_Date' in data.columns and 'Max_Date' in data.columns:
                # Convertir Min_Date y Max_Date a datetime para comparación
                data['Min_Date'] = pd.to_datetime(data['Min_Date'])
                data['Max_Date'] = pd.to_datetime(data['Max_Date'])
                
                # Encontrar el período que contiene la fecha objetivo (período actual)
                current_period = data[
                    (data['Min_Date'] <= target_dt) & 
                    (data['Max_Date'] >= target_dt)
                ]
                
                if current_period.empty:
                    # Si no hay período exacto, buscar el más cercano
                    data['distance'] = abs((data['Min_Date'] + (data['Max_Date'] - data['Min_Date'])/2) - target_dt).dt.days
                    current_period = data.loc[data['distance'].idxmin():data['distance'].idxmin()]
                    self.logger.info(f"📅 No exact period found, using closest period for {target_dt.date()}")
                
                current_row = current_period.iloc[0]
                current_period_group = current_row['Period_Group']
                
                # Encontrar el período de comparación que contiene las fechas especificadas
                comparison_start_dt = pd.to_datetime(self.comparison_start_date)
                comparison_end_dt = pd.to_datetime(self.comparison_end_date)
                
                # Buscar período que se solape con el rango de comparación
                comparison_period = data[
                    (data['Min_Date'] <= comparison_end_dt) & 
                    (data['Max_Date'] >= comparison_start_dt)
                ]
                
                if comparison_period.empty:
                    return {"error": f"No data available for comparison period {self.comparison_start_date} to {self.comparison_end_date}"}
                
                # Si hay múltiples períodos que se solapan, usar el que mejor se solapa
                if len(comparison_period) > 1:
                    # Calcular el solapamiento para cada período
                    comparison_period['overlap'] = comparison_period.apply(
                        lambda row: min(row['Max_Date'], comparison_end_dt) - max(row['Min_Date'], comparison_start_dt), 
                        axis=1
                    )
                    comparison_period = comparison_period[comparison_period['overlap'] == comparison_period['overlap'].max()]
                
                comparison_row = comparison_period.iloc[0]
                
                self.logger.info(f"📊 Comparing Period {current_period_group} vs Comparison Period {comparison_row['Period_Group']} ({self.comparison_start_date} to {self.comparison_end_date})")
                
            else:
                # Lógica original para datos por fecha exacta
                current_data = data[data['Date'] == target_dt]
                
                if current_data.empty:
                    return {"error": f"No data for target date {target_dt.date()}"}
                
                current_row = current_data.iloc[0]
                
                # Para fechas exactas, buscar datos dentro del rango de comparación
                comparison_start_dt = pd.to_datetime(self.comparison_start_date)
                comparison_end_dt = pd.to_datetime(self.comparison_end_date)
                
                comparison_data = data[
                    (data['Date'] >= comparison_start_dt) & 
                    (data['Date'] <= comparison_end_dt)
                ]
                
                if comparison_data.empty:
                    return {"error": f"No data available for comparison period {self.comparison_start_date} to {self.comparison_end_date}"}
                
                # Usar el promedio del período de comparación si hay múltiples fechas
                comparison_row = comparison_data.mean() if len(comparison_data) > 1 else comparison_data.iloc[0]

            # Calcular diferencias (misma lógica que _analyze_vslast)
            metrics = {}
            for col in ['Load_Factor', 'OTP15_adjusted', 'Mishandling', 'Misconex']:
                if col in data.columns:
                    current_val = pd.to_numeric(current_row.get(col), errors='coerce')
                    comparison_val = pd.to_numeric(comparison_row.get(col), errors='coerce')

                    if pd.notna(current_val) and pd.notna(comparison_val):
                        difference = current_val - comparison_val
                        change_pct = round((difference / comparison_val) * 100, 1) if comparison_val != 0 else 0
                        
                        # Determinar dirección y significancia
                        direction = 'higher' if difference > 0 else 'lower'
                        is_significant = abs(change_pct) > 5  # Cambio > 5% es significativo
                        
                        metrics[col] = {
                            'current': round(current_val, 2),
                            'comparison': round(comparison_val, 2),
                            'difference': round(difference, 2),
                            'change_pct': change_pct,
                            # Claves que espera el causal agent
                            'current_value': round(current_val, 2),
                            'previous_value': round(comparison_val, 2),  # Usar comparison_val como "previous"
                            'day_value': round(current_val, 2),
                            'delta': round(difference, 2),
                            'direction': direction,
                            'is_significant': is_significant
                        }
                        
                        self.logger.info(f"📊 {col}: {current_val:.2f} vs {comparison_val:.2f} (Δ{difference:+.2f}, {change_pct:+.1f}%)")
                    else:
                        metrics[col] = {
                            'current': current_val if pd.notna(current_val) else None,
                            'comparison': comparison_val if pd.notna(comparison_val) else None,
                            'difference': 0,
                            'change_pct': 0,
                            'current_value': current_val if pd.notna(current_val) else None,
                            'previous_value': comparison_val if pd.notna(comparison_val) else None,
                            'day_value': current_val if pd.notna(current_val) else None,
                            'delta': 0,
                            'direction': 'unchanged',
                            'is_significant': False,
                            'error': 'Missing data'
                        }

            # Crear resumen
            summary_parts = []
            comparison_period_desc = f"período seleccionado ({self.comparison_start_date} a {self.comparison_end_date})"
            
            for metric_name, metric_data in metrics.items():
                if 'error' not in metric_data and metric_data.get('is_significant'):
                    direction = "aumentó" if metric_data['difference'] > 0 else "disminuyó" 
                    summary_parts.append(f"{self._get_metric_display_name(metric_name)} {direction} {abs(metric_data['change_pct']):.1f}% vs {comparison_period_desc}")
            
            summary = f"Comparación vs {comparison_period_desc}: " + "; ".join(summary_parts) if summary_parts else f"Sin cambios significativos vs {comparison_period_desc}"
            
            return {
                'metrics': metrics,
                'summary': summary,
                'comparison_info': {
                    'comparison_period': f"{self.comparison_start_date} to {self.comparison_end_date}",
                    'comparison_type': 'specific_dates'
                }
            }
            
        except Exception as e:
            return {"error": f"Error in vslast analysis with specific dates: {str(e)}"}

    def _analyze_mean(self, data: pd.DataFrame, target_dt: pd.Timestamp, node_path: str) -> Dict[str, Any]:
        """Análisis vs media de período - Adaptado para datos agregados por período"""
        try:
            # Para datos agregados por período, necesitamos encontrar el período que contiene la fecha objetivo
            if 'Period_Group' in data.columns and 'Min_Date' in data.columns and 'Max_Date' in data.columns:
                # Convertir Min_Date y Max_Date a datetime para comparación
                data['Min_Date'] = pd.to_datetime(data['Min_Date'])
                data['Max_Date'] = pd.to_datetime(data['Max_Date'])
                
                # Encontrar el período que contiene la fecha objetivo
                current_period = data[
                    (data['Min_Date'] <= target_dt) & 
                    (data['Max_Date'] >= target_dt)
                ]
                
                if current_period.empty:
                    # Si no hay período exacto, buscar el más cercano
                    data['distance'] = abs((data['Min_Date'] + (data['Max_Date'] - data['Min_Date'])/2) - target_dt).dt.days
                    current_period = data.loc[data['distance'].idxmin():data['distance'].idxmin()]
                    self.logger.info(f"📅 No exact period found, using closest period for {target_dt.date()}")
                
                current_row = current_period.iloc[0]
                
                # Para el análisis de media, usar baseline_periods períodos incluyendo el actual
                current_period_group = current_row['Period_Group']
                
                # NUEVA LÓGICA: Siempre usar períodos 1 a baseline_periods para el cálculo de baseline
                # Esto significa que el baseline incluye el período que se está analizando
                baseline_data = data[
                    (data['Period_Group'] >= 1) & 
                    (data['Period_Group'] <= self.baseline_periods)
                ]
                
                if len(baseline_data) < self.baseline_periods:
                    # Si no hay suficientes períodos, usar todos los disponibles desde período 1
                    available_periods = len(data[data['Period_Group'] >= 1])
                    self.logger.warning(f"📊 Requested {self.baseline_periods} baseline periods, but only {available_periods} available")
                    baseline_data = data[data['Period_Group'] >= 1]
                
                if baseline_data.empty:
                    return {"error": "No baseline data available for mean comparison"}
                
                self.logger.info(f"📊 Analyzing Period {current_period_group} vs mean of {len(baseline_data)} baseline periods (1 to {self.baseline_periods}): {list(sorted(baseline_data['Period_Group'].unique()))}")
                
                # Para el cálculo de la media, usar todos los períodos baseline
                historical_data = baseline_data
                
            else:
                # Lógica original para datos por fecha exacta
                current_data = data[data['Date'] == target_dt]
                
                if current_data.empty:
                    return {"error": f"No data for target date {target_dt.date()}"}
                
                current_row = current_data.iloc[0]
                
                # Para análisis de media, usar datos históricos
                historical_data = data[data['Date'] < target_dt]
                
                if historical_data.empty:
                    return {"error": "No historical data available for mean comparison"}

            # Calcular métricas vs media histórica (misma lógica para ambos casos)
            metrics = {}
            for col in ['Load_Factor', 'OTP15_adjusted', 'Mishandling', 'Misconex']:
                if col in data.columns:
                    current_val = pd.to_numeric(current_row.get(col), errors='coerce')
                    historical_vals = pd.to_numeric(historical_data[col], errors='coerce').dropna()

                    if pd.notna(current_val) and len(historical_vals) > 0:
                        mean_val = historical_vals.mean()
                        difference = current_val - mean_val
                        change_pct = round((difference / mean_val) * 100, 1) if mean_val != 0 else 0
                        
                        # Determinar dirección y significancia
                        direction = 'higher' if difference > 0 else 'lower'
                        is_significant = abs(change_pct) > 5  # Cambio > 5% es significativo
                        
                        metrics[col] = {
                            'current': round(current_val, 2),
                            'baseline': round(mean_val, 2),  # Mostrar como "baseline" en lugar de "historical_mean"
                            'historical_mean': round(mean_val, 2),
                            'difference': round(difference, 2),
                            'change_pct': change_pct,
                            'historical_periods': len(historical_vals),
                            # Claves que espera el causal agent
                            'current_value': round(current_val, 2),
                            'day_value': round(current_val, 2),
                            'week_average': round(mean_val, 2),
                            'delta': round(difference, 2),
                            'direction': direction,
                            'is_significant': is_significant
                        }

            # Determinar fecha de referencia para el resultado
            if 'Period_Group' in data.columns:
                target_date_str = current_row.get('Min_Date', target_dt).strftime('%Y-%m-%d') if hasattr(current_row.get('Min_Date', target_dt), 'strftime') else str(current_row.get('Min_Date', target_dt))[:10]
            else:
                target_date_str = target_dt.date().isoformat()

            return {
                'mode': 'mean',
                'target_date': target_date_str,
                'metrics': metrics,
                'summary': self._create_mean_summary(metrics)
            }

        except Exception as e:
            return {"error": f"Error in mean analysis: {str(e)}"}

    def _analyze_target(self, data: pd.DataFrame, target_dt: pd.Timestamp, node_path: str) -> Dict[str, Any]:
        """Análisis vs targets predefinidos"""
        try:
            # Targets por defecto (estos podrían venir de configuración)
            targets = {
                'Load_Factor': 85.0,
                'OTP15_adjusted': 90.0,
                'Mishandling': 0.5,
                'Misconex': 0.3
            }

            current_data = data[data['Date'] == target_dt]

            if current_data.empty:
                return {"error": f"No data for target date {target_dt.date()}"}

            current_row = current_data.iloc[0]

            # Comparar con targets
            metrics = {}
            for col, target_val in targets.items():
                if col in data.columns:
                    current_val = pd.to_numeric(current_row.get(col), errors='coerce')

                    if pd.notna(current_val):
                        difference = current_val - target_val
                        metrics[col] = {
                            'current': round(current_val, 2),
                            'target': target_val,
                            'difference': round(difference, 2),
                            'performance': 'above_target' if difference > 0 else 'below_target' if difference < 0 else 'on_target'
                        }

            return {
                'mode': 'target',
                'target_date': target_dt.date().isoformat(),
                'metrics': metrics,
                'summary': self._create_target_summary(metrics)
            }

        except Exception as e:
            return {"error": f"Error in target analysis: {str(e)}"}

    def _create_vslast_summary(self, metrics: Dict) -> str:
        """Crea resumen para análisis vslast"""
        if not metrics:
            return "No se pudieron calcular comparaciones vs período anterior"

        summary_parts = []
        for metric, data in metrics.items():
            change = data['difference']
            metric_name = self._get_metric_display_name(metric)

            if abs(change) > 0.1:  # Solo mencionar cambios significativos
                direction = "aumentó" if change > 0 else "disminuyó"
                summary_parts.append(f"{metric_name} {direction} {abs(change)}")

        if summary_parts:
            return f"vs período anterior: {', '.join(summary_parts)}"
        else:
            return "Métricas operativas estables vs período anterior"

    def _create_mean_summary(self, metrics: Dict) -> str:
        """Crea resumen para análisis mean"""
        if not metrics:
            return "No se pudieron calcular comparaciones vs media histórica"

        summary_parts = []
        for metric, data in metrics.items():
            change = data['difference']
            metric_name = self._get_metric_display_name(metric)

            if abs(change) > 0.1:
                direction = "superior" if change > 0 else "inferior"
                summary_parts.append(f"{metric_name} {direction} a la media en {abs(change)}")

        if summary_parts:
            return f"vs media baseline (incluyendo período actual): {', '.join(summary_parts)}"
        else:
            return "Métricas operativas en línea con la media baseline"

    def _create_target_summary(self, metrics: Dict) -> str:
        """Crea resumen para análisis target"""
        if not metrics:
            return "No se pudieron comparar vs targets"

        on_target = sum(1 for data in metrics.values() if data['performance'] == 'on_target')
        above_target = sum(1 for data in metrics.values() if data['performance'] == 'above_target')
        below_target = sum(1 for data in metrics.values() if data['performance'] == 'below_target')

        total = len(metrics)
        return f"Performance vs targets: {on_target}/{total} en objetivo, {above_target} superando, {below_target} por debajo"

    def _get_metric_display_name(self, metric: str) -> str:
        """Convierte nombres técnicos a nombres de display"""
        display_names = {
            'Load_Factor': 'Load Factor',
            'OTP15_adjusted': 'OTP',
            'Mishandling': 'Mishandling',
            'Misconex': 'Conexiones Perdidas'
        }
        return display_names.get(metric, metric)

    def _explain_load_factor(self, row: pd.Series, anomaly_type: str, metrics_context: dict = None) -> Optional[str]:
        """Genera explicación específica para Load Factor"""
        try:
            lf = pd.to_numeric(row.get('Load_Factor'), errors='coerce')
            if pd.isna(lf):
                return None

            # Construir explicación base
            explanation_parts = []
            
            if lf > 90 and anomaly_type == 'negative':
                explanation_parts.append(f"Load Factor muy alto ({lf}%) puede haber impactado la experiencia del cliente")
            elif lf < 70 and anomaly_type == 'positive':
                explanation_parts.append(f"Load Factor bajo ({lf}%) puede haber mejorado la comodidad del vuelo")
            else:
                explanation_parts.append(f"Load Factor ({lf}%)")

            # Añadir contexto de cambio si está disponible
            if metrics_context and 'previous_value' in metrics_context:
                previous = metrics_context['previous_value']
                difference = metrics_context.get('difference', 0)
                if abs(difference) > 1:  # Solo si hay cambio significativo
                    direction = "aumentó" if difference > 0 else "disminuyó"
                    explanation_parts.append(f"- {direction} {abs(difference):.1f}% vs período anterior ({previous}%)")

            return " ".join(explanation_parts) if explanation_parts else None

        except Exception:
            return None

    def _explain_otp(self, row: pd.Series, anomaly_type: str, metrics_context: dict = None) -> Optional[str]:
        """Genera explicación específica para OTP"""
        try:
            otp = pd.to_numeric(row.get('OTP15_adjusted'), errors='coerce')
            if pd.isna(otp):
                return None

            # Construir explicación base
            explanation_parts = []
            
            if otp < 80 and anomaly_type == 'negative':
                explanation_parts.append(f"Puntualidad baja ({otp:.1f}%) puede haber generado insatisfacción")
            elif otp > 95 and anomaly_type == 'positive':
                explanation_parts.append(f"Excelente puntualidad ({otp:.1f}%) puede haber contribuido a la mejora")
            else:
                explanation_parts.append(f"Puntualidad ({otp:.1f}%)")

            # Añadir contexto de cambio si está disponible
            if metrics_context and 'previous_value' in metrics_context:
                previous = metrics_context['previous_value']
                difference = metrics_context.get('difference', 0)
                if abs(difference) > 0.5:  # Solo si hay cambio significativo
                    direction = "mejoró" if difference > 0 else "empeoró"
                    explanation_parts.append(f"- {direction} {abs(difference):.1f}% vs período anterior ({previous:.1f}%)")

            return " ".join(explanation_parts) if explanation_parts else None

        except Exception:
            return None

    def _explain_mishandling(self, row: pd.Series, anomaly_type: str, metrics_context: dict = None) -> Optional[str]:
        """Genera explicación específica para Mishandling"""
        try:
            mh = pd.to_numeric(row.get('Mishandling'), errors='coerce')
            if pd.isna(mh):
                return None

            # Construir explicación base
            explanation_parts = []
            
            if mh > 1.0 and anomaly_type == 'negative':
                explanation_parts.append(f"Tasa alta de mishandling ({mh:.1f}%) impactó negativamente la experiencia")
            elif mh < 0.3 and anomaly_type == 'positive':
                explanation_parts.append(f"Baja tasa de mishandling ({mh:.1f}%) contribuyó a la mejora del servicio")
            else:
                explanation_parts.append(f"Mishandling ({mh:.1f}%)")

            # Añadir contexto de cambio si está disponible
            if metrics_context and 'previous_value' in metrics_context:
                previous = metrics_context['previous_value']
                difference = metrics_context.get('difference', 0)
                if abs(difference) > 0.1:  # Solo si hay cambio significativo
                    direction = "aumentó" if difference > 0 else "disminuyó"
                    explanation_parts.append(f"- {direction} {abs(difference):.1f}% vs período anterior ({previous:.1f}%)")

            return " ".join(explanation_parts) if explanation_parts else None

        except Exception:
            return None

    def _explain_misconex(self, row: pd.Series, anomaly_type: str, metrics_context: dict = None) -> Optional[str]:
        """Genera explicación específica para Misconex (conexiones perdidas)"""
        try:
            mc = pd.to_numeric(row.get('Misconex'), errors='coerce')
            if pd.isna(mc):
                return None

            # Construir explicación base
            explanation_parts = []
            
            if mc > 0.5 and anomaly_type == 'negative':
                explanation_parts.append(f"Tasa alta de conexiones perdidas ({mc:.1f}%) impactó negativamente la experiencia")
            elif mc < 0.1 and anomaly_type == 'positive':
                explanation_parts.append(f"Baja tasa de conexiones perdidas ({mc:.1f}%) contribuyó a la mejora del servicio")
            else:
                explanation_parts.append(f"Conexiones perdidas ({mc:.1f}%)")

            # Añadir contexto de cambio si está disponible
            if metrics_context and 'previous_value' in metrics_context:
                previous = metrics_context['previous_value']
                difference = metrics_context.get('difference', 0)
                if abs(difference) > 0.1:  # Solo si hay cambio significativo
                    direction = "aumentó" if difference > 0 else "disminuyó"
                    explanation_parts.append(f"- {direction} {abs(difference):.1f}% vs período anterior ({previous:.1f}%)")

            return " ".join(explanation_parts) if explanation_parts else None

        except Exception:
            return None 