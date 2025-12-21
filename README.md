# 🚀 Iberia NPS Anomaly Detection & Explanation System

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![AI Powered](https://img.shields.io/badge/AI-Powered-green.svg)](https://openai.com/)
[![Power BI](https://img.shields.io/badge/Power%20BI-API-yellow.svg)](https://docs.microsoft.com/en-us/power-bi/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

An enterprise-grade **AI-powered anomaly detection and explanation system** for Net Promoter Score (NPS) analysis in the airline industry. The system automatically detects NPS anomalies across hierarchical customer segments, investigates root causes using multiple data sources, and generates executive-level insights through intelligent AI agents.

---

## 📋 **Table of Contents**

- [🎯 Overview](#-overview)
- [🏗️ System Architecture](#️-system-architecture)
- [📊 Data Sources](#-data-sources)
- [🔧 Core Modules](#-core-modules)
- [🤖 AI Agents](#-ai-agents)
- [⚙️ Installation & Setup](#️-installation--setup)
- [🚀 Usage](#-usage)
- [📈 Analysis Modes](#-analysis-modes)
- [📝 Configuration](#-configuration)
- [🐳 Docker Deployment](#-docker-deployment)
- [📁 Output Structure](#-output-structure)

---

## 🎯 **Overview**

The **Iberia NPS Anomaly Detection & Explanation System** provides:

- **🔍 Automatic anomaly detection** across hierarchical customer segments (Global → Radio → Cabin → Company)
- **📊 Multi-source root cause investigation** using operational data, customer feedback, and incident reports
- **🤖 AI-powered explanations** via OpenAI (GPT-4, o4-mini) or AWS Bedrock (Claude Sonnet 4)
- **📈 Dual analysis modes**: Weekly comparative + Daily single period analysis
- **🎯 Executive-level reporting** with actionable insights and trend identification
- **☁️ Automated S3 upload** for report persistence and distribution

### **Key Capabilities**

| Feature | Description |
|---------|-------------|
| **Multi-source Integration** | Power BI, NCS incidents (S3), Customer verbatims |
| **Hierarchical Analysis** | Global → LH/SH → Economy/Business/Premium → IB/YW |
| **Flexible Detection Modes** | Target-based, Mean-based, or VS-Last comparison |
| **Multi-LLM Support** | OpenAI (o4-mini, GPT-4) + AWS Bedrock (Claude Sonnet 4) |
| **Automated Scheduling** | Docker-ready for CI/CD pipeline execution |

---

## 🏗️ **System Architecture**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ENTRY POINTS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  weekly_deep_research.py     │ Automated weekly analysis (7d + daily)       │
│  deep_research_period.py     │ Custom/flexible period analysis              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA COLLECTION LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  pbi_collector.py          │ Power BI DAX queries (NPS, Drivers, Routes)    │
│  ncs_collector.py          │ AWS S3 incident data (NCS system)              │
│  chatbot_verbatims.py      │ Customer feedback text analysis                │
│  s3_report_uploader.py     │ Report upload to S3 bucket                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ANOMALY DETECTION LAYER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  flexible_detector.py      │ Multi-mode detection (target/mean/vslast)      │
│  target_based_detector.py  │ Monthly target comparison                      │
│  flexible_interpreter.py   │ Orchestrates AI explanation generation         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AI EXPLANATION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  causal_explanation_agent  │ Root cause investigation per segment           │
│  anomaly_interpreter_agent │ Hierarchical tree pattern analysis             │
│  anomaly_summary_agent     │ Executive summary consolidation                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Executive Reports (JSON)  │ Consolidated weekly + daily insights           │
│  S3 Upload                 │ Automated report persistence                   │
│  Conversation Logs         │ Debug & audit trail                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 **Data Sources**

### **1. 🔷 Power BI Dashboard**

| Aspect | Details |
|--------|---------|
| **Purpose** | Primary source for NPS metrics and operational data |
| **Authentication** | Microsoft Azure AD with service principal (MSAL) |
| **Query Language** | DAX (Data Analysis Expressions) |
| **Data Types** | NPS scores, SHAP drivers, routes, customer profiles, operational KPIs |

**DAX Query Templates** (located in `data_collection/queries/`):
- `NPS_flex_agg.txt` - Flexible NPS aggregation
- `Exp. Drivers.txt` - SHAP-based explanatory drivers
- `Rutas.txt` - Route-specific NPS performance
- `Customer Profile.txt` - Customer segmentation
- `Operativa_flex_agg.txt` - Operational metrics (OTP, Load Factor)

### **2. 🟠 NCS (Non-Compliance System)**

| Aspect | Details |
|--------|---------|
| **Source** | AWS S3 bucket (`ibdata-prod-ew1-s3-customer`) |
| **Content** | Operational incident reports, flight disruptions |
| **Format** | TXT/CSV files with structured incident data |
| **Path** | `customer/catia/ncs/raw/attatchments/` |

### **3. 🟢 Customer Verbatims**

| Aspect | Details |
|--------|---------|
| **Source** | Chatbot API + Survey feedback |
| **Processing** | NLP-based theme extraction and sentiment analysis |
| **Categories** | Baggage, punctuality, onboard service, reservations |

---

## 🔧 **Core Modules**

### **📁 Entry Points**

#### **`weekly_deep_research.py`** - Automated Weekly Analysis ⭐ Primary
The main script for production use. Executes a comprehensive weekly analysis:

1. **Weekly Comparative** (7-day aggregation, vs L7d comparison)
2. **Daily Single** (1-day aggregation x 7 days, mean-based detection)
3. **Consolidated Summary** (AI-generated executive report)
4. **S3 Upload** (automated report persistence)

```bash
# Run weekly analysis
python dashboard_analyzer/weekly_deep_research.py [options]
```

#### **`deep_research_period.py`** - Custom Period Analysis
Flexible analysis with full parameter control for ad-hoc investigations:

- Custom aggregation periods (1d, 7d, 14d, 30d)
- Multiple detection modes (target, mean, vslast)
- Configurable comparison filters
- Single or comparative study modes

```bash
# Run custom analysis
python dashboard_analyzer/deep_research_period.py [options]
```

### **📁 Data Collection (`data_collection/`)**

| Module | Purpose |
|--------|---------|
| `pbi_collector.py` | Power BI API integration, DAX query execution |
| `ncs_collector.py` | AWS S3 incident data retrieval |
| `chatbot_verbatims_collector.py` | Customer feedback analysis |
| `s3_report_uploader.py` | Report upload to S3 bucket |

### **📁 Anomaly Detection (`anomaly_detection/`)**

| Module | Purpose |
|--------|---------|
| `flexible_detector.py` | Multi-mode anomaly detection engine |
| `target_based_detector.py` | Monthly target comparison logic |
| `flexible_anomaly_interpreter.py` | AI agent orchestration |

### **📁 AI Explanation (`anomaly_explanation/`)**

| Module | Purpose |
|--------|---------|
| `genai_core/agents/` | AI agent implementations |
| `config/prompts/` | YAML-based prompt configuration |
| `data_analyzer.py` | Operational data analysis |
| `routes_analyzer.py` | Route-specific analysis |

---

## 🤖 **AI Agents**

### **🔍 Causal Explanation Agent**
Investigates root causes of individual segment anomalies.

**Available Tools:**
| Tool | Comparative Mode | Single Mode | Purpose |
|------|------------------|-------------|---------|
| `explanatory_drivers_tool` | ✅ SHAP changes | ❌ N/A | Main satisfaction drivers |
| `operative_data_tool` | ✅ Metric changes | ✅ Correlations | OTP, Load Factor analysis |
| `ncs_tool` | ✅ Incident changes | ✅ Absolute counts | Operational disruptions |
| `routes_tool` | ✅ Route NPS changes | ✅ Absolute NPS | Geographic impact |
| `verbatims_tool` | ✅ Theme changes | ✅ Absolute themes | Customer voice |
| `customer_profile_tool` | ✅ Segment reactivity | ✅ Absolute NPS | Customer sensitivity |

### **🌳 Anomaly Interpreter Agent**
Analyzes patterns across the hierarchical segment tree using a multi-step conversational methodology.

**Hierarchical Bubbling Logic:**
The interpreter uses a sophisticated "bubbling" algorithm to determine how anomalies propagate through the tree:

| Scenario | Pattern | Interpretation |
|----------|---------|----------------|
| **SINERGIA** | `(+,+ \| +)` or `(-,- \| -)` | Both children push same direction → Use parent's explanation |
| **CANCELACIÓN** | `(+,- \| N)` or `(-,+ \| N)` | Opposite effects cancel out → Report both causes separately |
| **DOMINANCIA** | `(+,- \| +)` or `(-,+ \| -)` | One child wins → Use dominant child's explanation |
| **DILUCIÓN** | `(+,N \| N)` or `(-,N \| N)` | Anomaly absorbed by Normal → Report anomalous child |
| **TRANSFERENCIA** | `(+,N \| +)` or `(-,N \| -)` | One child infects parent → Use anomalous child's explanation |

**Multi-Step Analysis Flow:**
```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1-3: BUBBLING LOGIC                                        │
│   Determine aggregation dynamics at each tree level             │
│   (Company → Cabin → Radio → Global)                            │
├─────────────────────────────────────────────────────────────────┤
│ STEP 4: NMA IDENTIFICATION                                      │
│   Identify "Nodo Máximo Afectado" (highest affected node)       │
│   for each cause based on bubbling rules                        │
├─────────────────────────────────────────────────────────────────┤
│ STEP 4B: EVIDENCE EXTRACTION ⭐ Key Innovation                  │
│   Forces the model to RE-READ the initial context and           │
│   extract ALL evidence data textually for each NMA              │
│   (Solves the "data loss in multi-turn" problem)                │
├─────────────────────────────────────────────────────────────────┤
│ STEP 5: EXECUTIVE SYNTHESIS                                     │
│   Generate narrative summary with fresh evidence from 4B        │
└─────────────────────────────────────────────────────────────────┘
```

**Evidence Sources Extracted:**
- 📈 **Explanatory Drivers** (SHAP values)
- 📊 **Operative Data** (OTP, Load Factor, Mishandling)
- 🚨 **NCS Incidents** (Cancellations, Delays, Aircraft limitations)
- 💬 **Customer Verbatims** (Feedback themes)
- ✈️ **Routes** (Top affected routes with NPS)
- 👥 **Customer Profiles** (Reactive segments with spread)

### **📋 Anomaly Summary Agent**
Consolidates multi-period analysis into executive reports using a **3-step stratified approach**.

**Features:**
- Weekly + Daily integration with narrative flow
- Trend analysis across time periods
- Strategic insights with actionable recommendations
- Spanish language executive reporting

**Stratified Summary Flow:**
```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: weekly_comparative_analysis (full report ~20K)      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  _extract_executive_synthesis_from_weekly()                 │
│  → Extracts only SÍNTESIS EJECUTIVA (~3K chars)             │
│  → Discards: Diagnósticos, Nodos, Evidencias                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              weekly_synthesis_only (~3K chars)
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
          ▼                                       ▼
┌─────────────────────────┐           ┌─────────────────────────┐
│  STEP 1: Parse sections │           │  STEP 2: Integrate      │
│  from SYNTHESIS only    │           │  synthesis + daily      │
│                         │           │  context paragraphs     │
│  For each section:      │           │                         │
│  → GLOBAL               │           │  Receives:              │
│  → ECONOMY SH           │──────────▶│  - weekly_synthesis_only│
│  → BUSINESS SH          │  daily    │  - daily_context_paragraphs
│  → ECONOMY LH           │  context  │                         │
│  → BUSINESS LH          │  (from    │  Outputs integrated     │
│  → PREMIUM LH           │  Step 1)  │  report with daily      │
│                         │           │  insights per section   │
│  Generates 1 daily      │           └─────────────────────────┘
│  context paragraph      │                       │
│  per section            │                       ▼
└─────────────────────────┘           ┌─────────────────────────┐
                                      │  STEP 3: Extract final  │
                                      │  executive synthesis    │
                                      │  in HTML for Teams      │
                                      └─────────────────────────┘
```

**Key Optimization:** By extracting only the executive synthesis (~3K chars) instead of the full technical report (~20K chars), we reduce LLM cognitive load by ~85%, enabling more accurate integration of daily context into each cabin section.

---

## ⚙️ **Installation & Setup**

### **Prerequisites**
- **Python 3.12+**
- **Docker** (optional, for containerized deployment)
- **Power BI Premium** workspace access
- **AWS S3** access for NCS data and report uploads
- **Azure OpenAI** or **AWS Bedrock** access

### **1. 📦 Clone & Install**

```bash
git clone <repository-url>
cd dashboard_analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### **2. 🔑 Environment Configuration**

Create `.devcontainer/.env` for local development:

```env
# Power BI Authentication
CLIENT_ID=your_power_bi_client_id
CLIENT_SECRET=your_power_bi_client_secret
TENANT_ID=your_azure_tenant_id
GROUP_ID=your_power_bi_workspace_id
DATASET_ID=your_power_bi_dataset_id

# Azure OpenAI Configuration
AZURE_API_KEY=your_azure_openai_api_key
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name

# AWS Configuration (for NCS data and S3 uploads)
# In production, use IAM roles instead of keys
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=eu-west-1

# Performance Tuning
PBI_API_TIMEOUT=120
```

### **3. 🔧 LLM Configuration**

The default LLM is configured in `anomaly_explanation/genai_core/utils/enums.py`:

```python
# Change this to switch all agents at once
DEFAULT_LLM_TYPE = "O4_MINI"  # Options: O4_MINI, CLAUDE_SONNET_4, GPT4o, etc.
```

**Supported LLM Types:**

| Provider | Models |
|----------|--------|
| **Azure OpenAI** | GPT-4, GPT-4o, GPT-4o-mini, o1-mini, o3-mini, o3, o4-mini |
| **AWS Bedrock** | Claude 3 Haiku, Claude 3.5 Sonnet, Claude Sonnet 4, Llama 3 |

---

## 🚀 **Usage**

### **🎯 Weekly Analysis (Recommended)**

The primary script for production use - runs comprehensive weekly + daily analysis:

```bash
# Standard weekly analysis (today - 4 days lag)
python dashboard_analyzer/weekly_deep_research.py

# Specific date analysis
python dashboard_analyzer/weekly_deep_research.py \
    --date-flight-local "2025-01-15"

# Custom segment focus
python dashboard_analyzer/weekly_deep_research.py \
    --segment "Global/LH" \
    --date-flight-local "2025-01-15"

# Different comparison filter
python dashboard_analyzer/weekly_deep_research.py \
    --causal-filter-comparison "vs LM" \
    --date-flight-local "2025-01-15"

# Local development (reads from .devcontainer/.env)
python dashboard_analyzer/weekly_deep_research.py \
    --environment local \
    --date-flight-local "2025-01-15"
```

### **⚡ Custom Period Analysis**

For flexible, parameter-controlled analysis and ad-hoc investigations:

```bash
# Single period analysis (no comparison)
python dashboard_analyzer/deep_research_period.py \
    --study-mode single \
    --aggregation-days 1 \
    --periods 7 \
    --segment "Global"

# Comparative analysis with custom baseline
python dashboard_analyzer/deep_research_period.py \
    --study-mode comparative \
    --aggregation-days 7 \
    --periods 4 \
    --causal-filter-comparison "vs LM" \
    --segment "Global/SH/Economy"

# Custom date range comparison
python dashboard_analyzer/deep_research_period.py \
    --causal-filter-comparison "vs Sel. Period" \
    --comparison-start-date "2024-12-01" \
    --comparison-end-date "2024-12-31" \
    --date-flight-local "2025-01-15"
```

### **🎛️ Command Line Parameters**

#### **`weekly_deep_research.py` Parameters**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--date-flight-local` | Analysis date (YYYY-MM-DD) | Today - 4 days |
| `--insert-date-ci` | Simulate CI run date | None |
| `--segment` | Root segment to analyze | `Global` |
| `--causal-filter-comparison` | Comparison filter | `vs L7d` |
| `--comparison-start-date` | Custom period start | None |
| `--comparison-end-date` | Custom period end | None |
| `--daily-anomaly-detection-mode` | Daily detection mode | `mean` |
| `--daily-baseline-periods` | Daily baseline periods | `7` |
| `--environment` | `local` or `prod` | `prod` |

#### **`deep_research_period.py` Parameters**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--study-mode` | `single` or `comparative` | `comparative` |
| `--aggregation-days` | Days per period (1, 7, 14, 30) | `1` |
| `--periods` | Number of periods to analyze | `74` |
| `--anomaly-detection-mode` | `target`, `mean`, or `vslast` | `target` |
| `--baseline-periods` | Periods for mean baseline | `7` |
| `--date-flight-local` | Analysis date (YYYY-MM-DD) | Today - 4 days |
| `--segment` | Root segment to analyze | `Global` |
| `--causal-filter-comparison` | Comparison filter | None |
| `--environment` | `local` or `prod` | `prod` |

### **🎯 Segment Targeting**

```bash
# Full tree analysis
--segment "Global"

# Radio-specific
--segment "Global/LH"    # Long Haul only
--segment "Global/SH"    # Short Haul only

# Cabin-specific
--segment "Global/LH/Business"
--segment "Global/SH/Economy"

# Company-specific
--segment "Global/LH/Business/IB"
--segment "Global/SH/Economy/YW"
```

---

## 📈 **Analysis Modes**

### **🔄 Study Modes**

| Mode | Description | Use Case |
|------|-------------|----------|
| **Comparative** | Compares current vs reference period | Understanding changes and trends |
| **Single** | Analyzes absolute values + correlations | Understanding specific period performance |

### **⚙️ Detection Modes**

| Mode | Algorithm | Best For |
|------|-----------|----------|
| **Target** | Compare against monthly/quarterly targets | Goal-oriented tracking |
| **Mean** | Compare against rolling average | Trend analysis |
| **VSLast** | Compare against previous period | Short-term change detection |

### **🎯 Comparison Filters**

| Filter | Description |
|--------|-------------|
| `vs L7d` | Last 7 days |
| `vs L14D` | Last 14 days |
| `vs L30D` | Last 30 days |
| `vs LM` | Last month |
| `vs LY` | Last year |
| `vs Target` | Business targets |
| `vs Sel. Period` | Custom date range |

---

## 📝 **Configuration**

### **🎯 Prompt Configuration**

AI agent prompts are configured via YAML files in `config/prompts/`:

| File | Agent | Purpose |
|------|-------|---------|
| `causal_explanation.yaml` | Causal Agent | Root cause investigation prompts (comparative & single modes) |
| `anomaly_interpreter.yaml` | Interpreter Agent | 6-step hierarchical analysis (bubbling + evidence extraction) |
| `anomaly_summary.yaml` | Summary Agent | Executive summary prompts |

**Interpreter Steps (in `anomaly_interpreter.yaml`):**
| Step | Name | Purpose |
|------|------|---------|
| 1 | `step1_company_level_diagnosis` | Analyze IB/YW dynamics in SH cabins |
| 2 | `step2_cabin_level_diagnosis` | Analyze cabin interactions within radios |
| 3 | `step3_radio_global_diagnosis` | Analyze LH/SH dynamics at Global level |
| 4 | `step4_nma_identification` | Identify highest affected nodes per cause |
| 4B | `step4b_evidence_extraction` | **Re-read context & extract all evidence** |
| 5 | `step5_executive_synthesis` | Generate executive narrative with evidence |

### **🗃️ DAX Query Templates**

Located in `data_collection/queries/`:

| Template | Purpose |
|----------|---------|
| `NPS_flex_agg.txt` | Flexible NPS aggregation |
| `Exp. Drivers.txt` | SHAP explanatory drivers |
| `Rutas.txt` | Route NPS performance |
| `Customer Profile.txt` | Segment analysis |
| `Operativa_flex_agg.txt` | Operational metrics |

---

## 🐳 **Docker Deployment**

### **Build & Run**

```bash
# Build image
docker build -t nps-anomaly-system .

# Run weekly analysis (default entrypoint)
docker run --rm \
    -e CLIENT_ID=xxx \
    -e CLIENT_SECRET=xxx \
    -e TENANT_ID=xxx \
    -e GROUP_ID=xxx \
    -e DATASET_ID=xxx \
    -e AZURE_API_KEY=xxx \
    -e AZURE_ENDPOINT=xxx \
    -e AZURE_OPENAI_DEPLOYMENT_NAME=xxx \
    nps-anomaly-system

# Run with custom parameters
docker run --rm \
    -e CLIENT_ID=xxx \
    [... other env vars ...] \
    nps-anomaly-system \
    python dashboard_analyzer/weekly_deep_research.py --segment "Global/LH"
```

### **Entrypoint**

The default entrypoint (`script/entrypoint.sh`) runs the weekly analysis:
```bash
python -u dashboard_analyzer/weekly_deep_research.py
```

---

## 📁 **Output Structure**

```
dashboard_analyzer/
├── {LLM_TYPE}_agent_conversations/  # AI agent conversation logs (e.g., O4_MINI_agent_conversations/)
│   ├── causal_explanation/          # Individual segment investigations
│   ├── anomaly_interpreter/         # Tree-wide analysis results
│   ├── anomaly_summary/             # Executive summary conversations
│   └── interpreter_debug/           # Debug data for interpreter
├── summary_reports/                 # Consolidated executive reports
└── tables/                          # Raw data downloads
    └── [date]_[mode]_[aggregation]/
        └── [segment]/
            ├── flexible_NPS_Xd.csv
            ├── flexible_operative_Xd.csv
            └── [other_data].csv
```

> **Note:** The agent conversations folder is prefixed with the LLM type (e.g., `O4_MINI_agent_conversations`, `CLAUDE_SONNET_4_agent_conversations`) to easily distinguish outputs from different models.

### **S3 Report Upload**

Reports are automatically uploaded to:
```
s3://ibdata-sbx-ew1-s3-customer/customer/catia/reports/raw/
```

---

## 🔒 **Security & Environment**

### **Environment Modes**

| Mode | Credential Source | Use Case |
|------|------------------|----------|
| `--environment local` | `.devcontainer/.env` file | Development |
| `--environment prod` | System environment variables / IAM roles | Production |

### **Credential Requirements**

| Service | Required Variables |
|---------|-------------------|
| **Power BI** | `CLIENT_ID`, `CLIENT_SECRET`, `TENANT_ID`, `GROUP_ID`, `DATASET_ID` |
| **Azure OpenAI** | `AZURE_API_KEY`, `AZURE_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME` |
| **AWS S3** | IAM role (prod) or `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (local) |

---

## 🎉 **Quick Start**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .devcontainer/.env.example .devcontainer/.env
# Edit .env with your credentials

# 3. Run weekly analysis
python dashboard_analyzer/weekly_deep_research.py \
    --date-flight-local "2025-01-15" \
    --environment local

# 4. Check results
ls dashboard_analyzer/summary_reports/
ls dashboard_analyzer/agent_conversations/interpreter_outputs/
```

---

## 📚 **Hierarchy Structure**

```
Global
├── Long Haul (LH)
│   ├── Economy [IB, YW]
│   ├── Business [IB, YW]
│   └── Premium [IB, YW]
└── Short Haul (SH)
    ├── Economy [IB, YW]
    └── Business [IB, YW]
```

**Companies:**
- **IB**: Iberia
- **YW**: Air Europa

---

**🏆 Built for intelligent airline customer experience analytics**

For support or contributions, please contact the development team or create an issue in the repository.
