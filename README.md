# NPS Anomaly Detection System

A comprehensive system for detecting and explaining Net Promoter Score (NPS) anomalies using Power BI data and operational metrics analysis.

## 🚀 Features

- **Automated Data Collection**: Retrieves NPS and operational data from Power BI using MSAL authentication
- **Anomaly Detection**: 7-day moving average analysis with ±10 point threshold for NPS anomalies
- **Tree-based Analysis**: Hierarchical anomaly interpretation following organizational structure
- **Operational Explanations**: Correlates NPS anomalies with operational metrics (OTP, Load Factor)
- **Smart Interpretation**: Bottom-up analysis with explanation flagging based on anomaly patterns

## 🏗️ System Architecture

### Tree Hierarchy Structure

```
Global
├── LH (Long Haul)
│   ├── Economy
│   ├── Business
│   └── Premium
└── SH (Short Haul)
        ├── Economy
    │   ├── IB
    │   └── YW
        └── Business
        ├── IB
        └── YW
```

### Module Structure

```
dashboard_analyzer/
├── main.py                     # Complete system pipeline
├── data_collection/
│   ├── pbi_collector.py        # Power BI data collection
│   └── queries/                # DAX query templates
├── anomaly_detection/
│   ├── anomaly_tree.py         # Tree structure and anomaly detection
│   └── anomaly_interpreter.py  # Bottom-up interpretation logic
└── anomaly_explanation/
    └── data_analyzer.py        # Operational metrics analysis
```

## 📊 Data Sources

### NPS Data
- **Source**: Power BI via DAX queries
- **Metrics**: Daily NPS scores by segment/date
- **Granularity**: All tree nodes (12 total)

### Operational Data
- **OTP15_adjusted**: On-time performance (15min threshold)
- **Load_Factor**: Cabin occupancy percentage
- **Misconex**: Connection issues (currently unavailable)
- **Mishandling**: Baggage mishandling (currently unavailable)

## 🔍 Anomaly Detection Logic

### Detection Rules
1. **7-day Moving Average**: Calculate trailing average for each node
2. **Threshold**: ±10 NPS points deviation from moving average
3. **States**: 
   - `[+]`: Above average +10 points
   - `[-]`: Below average -10 points  
   - `[N]`: Within ±10 points (normal)

### Explanation Requirements
Based on sibling node patterns:

- **Isolated Anomalies**: Single child with anomaly → needs explanation
- **Homogeneous Anomalies**: 2+ children with same anomaly → needs explanation
- **Mixed Anomalies**: Different anomaly types → each needs explanation

## 🧠 Interpretation Logic

### Bottom-up Analysis
The system analyzes parent-child relationships:

1. **Consistent**: All children have same state as parent
2. **Diluted**: Anomalous children's impact reduced by normal children
3. **Significant**: Anomalous children's impact overcomes dilution
4. **Cancelled**: Positive and negative children cancel each other
5. **Contradictory**: Parent-child relationship doesn't follow expected pattern

### Operational Correlation
For nodes marked `[Explanation needed]`, the system:

1. **Compares operational metrics** vs 7-day average
2. **Determines correlation** with NPS anomaly direction
3. **Generates explanations**:
   - "explains" when metric change supports NPS anomaly
   - "contradicts" when metric change opposes NPS anomaly
   - "no significant impact" when change below threshold

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Power BI access with MSAL authentication
- Environment variables configured

### Environment Setup

Create `.devcontainer/.env` with:
```bash
TENANT_ID=your_tenant_id
CLIENT_ID=your_client_id
CLIENT_SECRET=your_client_secret
DATASET_ID=your_dataset_id
WORKSPACE_ID=your_workspace_id
```

### Installation
```bash
# Clone repository
git clone https://github.com/nomadicsenseis/NPSModule.git
cd NPSModule

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Complete Analysis Pipeline
```bash
# Run full system (download data + analysis)
python -m dashboard_analyzer.main

# Skip download, analyze existing data
python -m dashboard_analyzer.main --skip-download

# Analyze specific date
python -m dashboard_analyzer.main --date 2025-05-24
```

### Output Format

The system provides two tree views for each day:

#### 1. Anomaly Tree with Explanation Flags
```
🌳 Anomaly Tree: 2025-05-19
Global [-] [Explanation needed]
  LH [N]
    Economy [N]
    Business [-] [Explanation needed]
    Premium [-] [Explanation needed]
  SH [-] [Explanation needed]
    ...
```

#### 2. Tree with Operational Explanations
```
🌳 Anomaly Tree with Operational Explanations: 2025-05-19
Global [-]
    • OTP stable at 90.57% (Δ-1.9pts vs 7-day avg) - no significant impact
    • Load Factor increased by 3.08pts (85.94% vs 82.86%) - explains NPS anomaly
  LH [N]
    ...
```

## 📈 System Workflow

1. **Data Collection**: Download NPS and operational data from Power BI
2. **Data Organization**: Save in date-structured folders (`DD_MM_YYYY`)
3. **Anomaly Detection**: Calculate 7-day moving averages and detect deviations
4. **Tree Analysis**: Apply bottom-up interpretation logic
5. **Explanation Flagging**: Mark nodes requiring operational analysis
6. **Operational Analysis**: Correlate operational metrics with NPS anomalies
7. **Report Generation**: Display interpreted trees with explanations

## 🔧 Configuration

### Anomaly Thresholds
   ```python
# NPS anomaly threshold
NPS_THRESHOLD = 10.0  # points

# Operational metric thresholds
THRESHOLDS = {
    'Load_Factor': 3.0,      # percentage points
    'OTP15_adjusted': 3.0,   # percentage points
    'Misconex': 1.0,         # percentage points
    'Mishandling': 0.5       # per 1000 passengers
}
```

### Data Structure
```
tables/
└── DD_MM_YYYY/
    ├── Global/
    │   ├── daily_NPS.csv
    │   └── daily_operative.csv
    ├── Global/LH/
    │   ├── daily_NPS.csv
    │   └── daily_operative.csv
    └── ...
```

## 📊 Sample Output

### Week Summary
```
📈 WEEK SUMMARY TABLE
Date         Status   +   -   N   Total 
────────────────────────────────────────
2025-05-18   🚨 Alert  1   5   6   12    
2025-05-19   🚨 Alert  0   8   4   12    
2025-05-20   ✅ Normal 0   0   12  12    
2025-05-21   🚨 Alert  1   2   9   12    
```

### Operational Insights
- **High Load Factor** correlations with negative NPS (crowded conditions)
- **Poor OTP performance** directly explains NPS drops
- **Contradictory signals** highlight complex multi-factor scenarios

## 🔮 Future Enhancements

- **Verbatims Analysis**: Text sentiment analysis for deeper insights
- **Misconex/Mishandling**: Integration when data becomes available
- **Predictive Modeling**: Forecast anomalies based on operational trends
- **Dashboard Interface**: Web-based visualization and interaction
- **Alert System**: Automated notifications for significant anomalies

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏷️ Version

**Current Version**: 1.0.0 - Complete NPS Anomaly Detection and Explanation System

**Last Updated**: December 2024

---

*Built with ❤️ for intelligent NPS analysis and operational insights*