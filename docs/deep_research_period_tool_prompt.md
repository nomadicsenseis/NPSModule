# deep_research_period - Tool Reference

## Tool Description

`deep_research_period` is a CLI tool that analyzes NPS (Net Promoter Score) anomalies across customer segments. It downloads data from Power BI, detects statistical anomalies, and generates AI-powered causal explanations.

**Command:** `python -m dashboard_analyzer.deep_research_period [OPTIONS]`

---

## Arguments

### `--study-mode` (string)
Analysis mode that determines comparison behavior.

| Value | Description |
|-------|-------------|
| `single` | Analyzes periods individually. Compares each period against historical mean. No causal comparison. |
| `comparative` | Compares current period against a reference period. Uses causal filter for context. **Default.** |

**When to use:**
- Use `single` for daily trend analysis (detecting deviations from normal)
- Use `comparative` for weekly reports comparing vs previous week/month/year

---

### `--aggregation-days` (integer)
Number of days per analysis period. Defines temporal granularity.

| Value | Analysis Type |
|-------|---------------|
| `1` | Daily analysis. Each period = 1 day. **Default.** |
| `7` | Weekly analysis. Each period = 7 days. |
| `14` | Biweekly analysis. |
| `30` | Monthly analysis. |

**Impact:** Lower values = more granular but noisier. Higher values = more stable but less responsive.

---

### `--periods` (integer)
Number of periods to analyze backwards from the analysis date. **Default: 74**

**Examples:**
- `--periods 1 --aggregation-days 7` → Analyze only the last week
- `--periods 7 --aggregation-days 1` → Analyze the last 7 days individually
- `--periods 4 --aggregation-days 7` → Analyze the last 4 weeks

**Performance note:** More periods = longer execution time. Recommend 1-7 for routine analysis.

---

### `--insert-date-ci` (string, YYYY-MM-DD)
Simulate running the analysis as if today were this date. The system applies a 4-day lag automatically (PBI data availability).

**Example:** `--insert-date-ci 2025-01-25` → Analyzes data up to 2025-01-21

**Use case:** Backfilling historical analyses, testing, CI/CD pipelines.

**Mutually exclusive with:** `--date-flight-local`

---

### `--date-flight-local` (string, YYYY-MM-DD)
Use this date directly as the analysis end date. No lag applied.

**Example:** `--date-flight-local 2025-01-20` → Analyzes data up to 2025-01-20

**Use case:** When you know the exact date available in the dashboard.

**Mutually exclusive with:** `--insert-date-ci`

---

### `--segment` (string)
Root segment of the NPS hierarchy to analyze. **Default: Global**

**Valid values:**

| Segment | Nodes Analyzed | Description |
|---------|----------------|-------------|
| `Global` | 12 | Full tree (all segments) |
| `Global/LH` | 4 | Long Haul only |
| `Global/SH` | 7 | Short Haul only |
| `Global/LH/Economy` | 1 | LH Economy only |
| `Global/LH/Business` | 1 | LH Business only |
| `Global/LH/Premium` | 1 | LH Premium only |
| `Global/SH/Economy` | 3 | SH Economy + IB + YW |
| `Global/SH/Business` | 3 | SH Business + IB + YW |

**Shortcuts accepted:** `SH`, `LH`, `Economy/SH`, `Business/SH`, `Economy/LH`, `Business/LH`, `Premium/LH`

**Performance note:** Smaller segments = faster execution.

---

### `--anomaly-detection-mode` (string)
Algorithm for detecting anomalies. **Default: target**

| Mode | Baseline | Use Case |
|------|----------|----------|
| `target` | Monthly target values | Compare vs business objectives |
| `mean` | Mean of last N periods | Detect deviations from trend |
| `vslast` | Previous period | Week-over-week comparison |

**Note:** When `--study-mode single`, behavior defaults to `mean` regardless of this setting.

---

### `--baseline-periods` (integer)
Number of historical periods for baseline calculation in `mean` mode. **Default: 7**

**Only applies when:** `--anomaly-detection-mode mean` or `--study-mode single`

**Example:** `--baseline-periods 7 --aggregation-days 1` → Baseline = average of last 7 days

---

### `--causal-filter-comparison` (string)
Reference period for causal analysis (operational metrics, verbatims, routes). **Default: vs L7d**

| Value | Comparison Period |
|-------|-------------------|
| `vs L7d` | Last 7 days |
| `vs L14d` | Last 14 days |
| `vs LM` | Last month |
| `vs LY` | Same period last year |
| `vs Target` | Target values |
| `vs Sel. Period` | Custom period (requires `--comparison-start-date` and `--comparison-end-date`) |

**Note:** Ignored when `--study-mode single`.

---

### `--comparison-start-date` (string, YYYY-MM-DD)
Start date for custom comparison period.

**Required when:** `--causal-filter-comparison "vs Sel. Period"`

---

### `--comparison-end-date` (string, YYYY-MM-DD)
End date for custom comparison period.

**Required when:** `--causal-filter-comparison "vs Sel. Period"`

---

### `--environment` (string)
Execution environment for credentials. **Default: prod**

| Value | Behavior |
|-------|----------|
| `prod` | Read credentials from system environment variables |
| `local` | Read credentials from `.env` file |

---

## Common Configurations

### Weekly Comparative Report
Analyze last week vs previous week. Standard weekly report.
```bash
python -m dashboard_analyzer.deep_research_period \
  --study-mode comparative \
  --aggregation-days 7 \
  --periods 1 \
  --causal-filter-comparison "vs L7d" \
  --date-flight-local 2025-01-20 \
  --segment Global
```

### Daily Trend Analysis
Analyze last 7 days individually, detect anomalies vs historical mean.
```bash
python -m dashboard_analyzer.deep_research_period \
  --study-mode single \
  --aggregation-days 1 \
  --periods 7 \
  --anomaly-detection-mode mean \
  --baseline-periods 7 \
  --date-flight-local 2025-01-20 \
  --segment Global
```

### Segment-Specific Analysis
Analyze only Short Haul Economy for faster results.
```bash
python -m dashboard_analyzer.deep_research_period \
  --study-mode comparative \
  --aggregation-days 7 \
  --periods 1 \
  --causal-filter-comparison "vs L7d" \
  --date-flight-local 2025-01-20 \
  --segment "Global/SH/Economy"
```

### Year-over-Year Comparison
Compare current week vs same week last year.
```bash
python -m dashboard_analyzer.deep_research_period \
  --study-mode comparative \
  --aggregation-days 7 \
  --periods 1 \
  --causal-filter-comparison "vs LY" \
  --date-flight-local 2025-01-20 \
  --segment Global
```

### Custom Period Comparison
Compare current week vs a specific historical period.
```bash
python -m dashboard_analyzer.deep_research_period \
  --study-mode comparative \
  --aggregation-days 7 \
  --periods 1 \
  --causal-filter-comparison "vs Sel. Period" \
  --comparison-start-date 2024-12-01 \
  --comparison-end-date 2024-12-31 \
  --date-flight-local 2025-01-20 \
  --segment Global
```

### Backfill Historical Analysis
Run analysis as if today were a past date.
```bash
python -m dashboard_analyzer.deep_research_period \
  --study-mode comparative \
  --aggregation-days 7 \
  --periods 1 \
  --causal-filter-comparison "vs L7d" \
  --insert-date-ci 2025-01-25 \
  --segment Global
```

---

## Output

The tool outputs:
1. **Anomaly detection results** - Which segments have anomalies (+/-/Normal)
2. **NPS values** - Current vs baseline values per segment
3. **Causal explanations** - AI-generated analysis of root causes
4. **AI interpretation** - Executive summary of findings

Output is printed to stdout. No files are saved by default.

---

## Important Notes

1. **PBI Data Lag:** Power BI data has a 4-day lag. When using `--insert-date-ci`, the system automatically subtracts 4 days.

2. **Date Selection:** Use `--date-flight-local` when you know the exact available date. Use `--insert-date-ci` when simulating a specific "today".

3. **Study Mode Impact:** `--study-mode single` forces `--causal-filter-comparison` to be ignored and uses mean-based anomaly detection.

4. **Execution Time:** Full `Global` analysis with multiple periods can take 10-30 minutes. Use specific segments for faster results.

5. **Anomaly Threshold:** Default threshold is 5 NPS points. Values deviating more than 5 points from baseline are flagged as anomalies.

6. **Return Behavior:** If no anomalies are detected, the tool reports "No anomalies found" - this is normal, not an error.
