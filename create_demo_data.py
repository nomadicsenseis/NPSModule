#!/usr/bin/env python3
"""
Create demo data for flexible anomaly detection system
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_demo_data():
    """Create demo data for flexible system"""
    # Estructura de nodos
    nodes = [
        'Global',
        'Global_LH', 'Global_LH_Economy', 'Global_LH_Business', 'Global_LH_Premium',
        'Global_SH', 'Global_SH_Economy', 'Global_SH_Business',
        'Global_SH_Economy_IB', 'Global_SH_Economy_YW',
        'Global_SH_Business_IB', 'Global_SH_Business_YW'
    ]
    
    base_path = Path('tables/demo_flexible_7d')
    
    for node in nodes:
        node_dir = base_path / node
        node_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar datos de ejemplo para 5 períodos (7 días cada uno)
        data = []
        base_nps = np.random.normal(65, 10)  # NPS base alrededor de 65
        
        for period in range(1, 6):
            # Simular una anomalía en el período 1 (más reciente)
            if period == 1 and 'YW' in node:
                nps = base_nps - 15  # Anomalía negativa para YW
            elif period == 1 and node == 'Global_LH_Premium':
                nps = base_nps + 12  # Anomalía positiva para Premium LH
            else:
                nps = base_nps + np.random.normal(0, 3)
            
            data.append({
                'Period_Group': period,
                'NPS_2025': nps,
                'NPS_2024': nps - 2,
                'NPS_2019': nps - 5,
                'Target': 70,
                'Responses': np.random.randint(10, 50),
                'Min_Date': f'2025-05-{period*7:02d}',
                'Max_Date': f'2025-05-{period*7+6:02d}'
            })
        
        df = pd.DataFrame(data)
        df.to_csv(node_dir / 'flexible_NPS_7d.csv', index=False)
        print(f'✅ Created demo data for {node}')

if __name__ == "__main__":
    create_demo_data()
    print('🎯 Demo data creation completed!') 