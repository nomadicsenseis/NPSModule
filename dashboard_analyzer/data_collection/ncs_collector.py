"""
NCS Data Collector
==================
Collects Net Customer Satisfaction (NCS) incident data from HTML emails stored in AWS S3.

Source: s3://ibdata-prod-ew1-s3-customer/customer/catia/ncs/raw/attatchments/
Files:  .txt files containing HTML email content with operational incident tables.
"""

import os
import re
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from bs4 import BeautifulSoup

from dashboard_analyzer.anomaly_explanation.genai_core.utils.aws_session import get_aws_session


# ── Known section headers in NCS emails ──
SECTION_HEADERS = {
    "cancelaciones": "Cancelaciones",
    "desvíos": "Desvíos",
    "desvios": "Desvíos",
    "retrasos": "Retrasos",
    "otras incidencias": "Otras incidencias",
    "equipaje": "Equipaje",
    "limitación de la aeronave": "Limitación de la aeronave",
    "incidencias con sistemas": "Incidencias con sistemas",
}

# Regex for flight numbers: IB0362, I21514, LL2601, YW1234
_FLIGHT_RE = re.compile(r"^(?:IB|I2|LL|YW)\d{3,5}$")

# Regex for 6-char concatenated route: IADMAD, MADJFK
_ROUTE_CONCAT_RE = re.compile(r"^[A-Z]{6}$")

# Regex for pax breakdown: J8W0Y66, J14W0Y109, J32W24Y303
_PAX_RE = re.compile(r"J(\d+)W(\d+)Y(\d+)")

# Regex for delay: "60minutos", "141minutos"
_DELAY_MINUTES_RE = re.compile(r"(\d+)\s*minutos?")

# Regex for delay: "2h 35min"
_DELAY_HM_RE = re.compile(r"(\d+)h\s*(\d+)\s*min")


class NCSDataCollector:
    """Collects and parses NCS operational incident data from AWS S3."""

    BUCKET = "ibdata-prod-ew1-s3-customer"
    PREFIX = "customer/catia/ncs/raw/attatchments/"

    def __init__(self, environment: str = "prod"):
        """
        Args:
            environment: "local" or "prod"
        """
        self.logger = logging.getLogger(__name__)
        self.environment = environment
        session = get_aws_session(environment=self.environment, use_sandbox=False)
        self.s3_client = session.client("s3")

    # ── S3 Access ──

    def list_available_files(self, date_prefix: str = None) -> List[str]:
        """List .txt NCS files in S3, optionally filtered by date prefix (YYYY-MM-DD)."""
        try:
            prefix = self.PREFIX
            if date_prefix:
                prefix += f"ndc-{date_prefix}"
            resp = self.s3_client.list_objects_v2(Bucket=self.BUCKET, Prefix=prefix)
            files = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".txt")]
            self.logger.info(f"Found {len(files)} NCS files with prefix '{prefix}'")
            return files
        except Exception as e:
            self.logger.error(f"Error listing S3 files: {e}")
            return []

    def read_ncs_file(self, file_key: str) -> pd.DataFrame:
        """Download and parse a single NCS email file from S3."""
        try:
            self.logger.info(f"Reading NCS file: s3://{self.BUCKET}/{file_key}")
            body = self.s3_client.get_object(Bucket=self.BUCKET, Key=file_key)["Body"].read().decode("utf-8")
            df = self._parse_html(body, file_key)
            if not df.empty:
                self.logger.info(f"Parsed {len(df)} incidents from {file_key}")
            else:
                self.logger.warning(f"No data extracted from {file_key}")
            return df
        except Exception as e:
            self.logger.error(f"Error reading {file_key}: {e}")
            return pd.DataFrame()

    # ══════════════════════════════════════════════════════════════════
    # HTML PARSING — structured incident extraction
    # ══════════════════════════════════════════════════════════════════

    def _parse_html(self, content: str, file_key: str) -> pd.DataFrame:
        """Parse HTML email content into a structured DataFrame of incidents.

        Each row = one incident with columns:
            incident_type, flight, origin, destination, route, date,
            pax_total, pax_j, pax_w, pax_y, description, delay_minutes,
            source_file, email_date, email_subject, email_summary
        """
        try:
            metadata = self._extract_metadata(content)
            soup = BeautifulSoup(content, "html.parser")
            summary_text = self._extract_summary(soup)

            incidents: List[Dict[str, Any]] = []
            for table in soup.find_all("table"):
                incidents.extend(self._parse_section_table(table))

            if not incidents:
                incidents = self._extract_text_fallback(soup.get_text())

            if not incidents:
                return pd.DataFrame()

            df = pd.DataFrame(incidents)
            df["source_file"] = file_key
            df["email_date"] = metadata.get("date", "")
            df["email_subject"] = metadata.get("subject", "")
            if summary_text:
                df["email_summary"] = summary_text
            return df

        except Exception as e:
            self.logger.error(f"Error parsing HTML from {file_key}: {e}")
            return pd.DataFrame()

    def _parse_section_table(self, table) -> List[Dict[str, Any]]:
        """Parse a single HTML table representing one incident section.

        Expected structure:
        - ROW 0: Header ['', 'Cancelaciones', 'Total24', ...]
        - Subsequent rows alternate between flight data rows and description rows.
        - For Retrasos: flight row, delay row, description row (3 rows per incident).
        """
        rows = table.find_all("tr", recursive=False)
        tbody = table.find("tbody")
        if tbody:
            rows = tbody.find_all("tr", recursive=False)

        if len(rows) < 2:
            return []

        header_cells = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
        section_type = self._identify_section(header_cells)
        if not section_type:
            return []

        incidents: List[Dict[str, Any]] = []
        i = 1
        while i < len(rows):
            cells = [c.get_text(strip=True) for c in rows[i].find_all(["th", "td"])]
            if not any(cells):
                i += 1
                continue

            flight_info = self._parse_flight_row(cells)
            if not flight_info:
                i += 1
                continue

            flight_info["incident_type"] = section_type
            description = ""
            delay_minutes = None

            j = i + 1
            while j < len(rows):
                next_cells = [c.get_text(strip=True) for c in rows[j].find_all(["th", "td"])]
                if not any(next_cells):
                    j += 1
                    continue

                delay = self._parse_delay_row(next_cells)
                if delay is not None:
                    delay_minutes = delay
                    j += 1
                    continue

                if not self._parse_flight_row(next_cells):
                    desc_text = " ".join(c for c in next_cells if c).strip()
                    if desc_text and len(desc_text) > 5:
                        description = desc_text
                    j += 1
                    break
                else:
                    break

            flight_info["description"] = description
            flight_info["delay_minutes"] = delay_minutes

            # For Retrasos, try to extract delay from description if not found
            if section_type == "Retrasos" and delay_minutes is None and description:
                extracted = self._extract_delay_from_text(description)
                if extracted:
                    flight_info["delay_minutes"] = extracted

            incidents.append(flight_info)
            i = j

        return incidents

    @staticmethod
    def _identify_section(header_cells: List[str]) -> Optional[str]:
        for cell in header_cells:
            cell_lower = cell.lower().strip()
            if cell_lower in SECTION_HEADERS:
                return SECTION_HEADERS[cell_lower]
            for key, canonical in SECTION_HEADERS.items():
                if key in cell_lower:
                    return canonical
        return None

    @staticmethod
    def _extract_section_total(header_cells: List[str]) -> Optional[int]:
        for cell in header_cells:
            m = re.match(r"Total\s*(\d+)", cell)
            if m:
                return int(m.group(1))
            if cell.isdigit():
                return int(cell)
        return None

    @staticmethod
    def _parse_flight_row(cells: List[str]) -> Optional[Dict[str, Any]]:
        """Parse a flight data row into structured fields.

        Expected patterns:
            ['', 'IB0362', 'IADMAD', 'IAD', '', 'MAD', '17 feb 2026', '74', '', '74', 'J8W0Y66', ...]
            ['', 'IB2435', 'IBZVLC', 'IBZ', '', 'VLC', '18 feb 2026']  (no pax data)
        """
        flight = None
        flight_idx = -1
        for idx, cell in enumerate(cells):
            if _FLIGHT_RE.match(cell):
                flight = cell
                flight_idx = idx
                break

        if not flight:
            return None

        result: Dict[str, Any] = {"flight": flight}

        # Route: next cell after flight should be concatenated route (IADMAD)
        origin = ""
        destination = ""
        route_concat = ""
        if flight_idx + 1 < len(cells):
            candidate = cells[flight_idx + 1]
            if _ROUTE_CONCAT_RE.match(candidate):
                route_concat = candidate
                origin = candidate[:3]
                destination = candidate[3:]

        if not origin and flight_idx + 2 < len(cells):
            candidate = cells[flight_idx + 2]
            if re.match(r"^[A-Z]{3}$", candidate):
                origin = candidate

        if not destination:
            for k in range(flight_idx + 3, min(flight_idx + 6, len(cells))):
                candidate = cells[k]
                if re.match(r"^[A-Z]{3}$", candidate) and candidate != origin:
                    destination = candidate
                    break

        result["origin"] = origin
        result["destination"] = destination
        result["route"] = f"{origin}-{destination}" if origin and destination else route_concat

        # Date: look for pattern like "17 feb 2026"
        date_str = ""
        for cell in cells[flight_idx:]:
            if re.match(r"\d{1,2}\s+\w{3,}\s+\d{4}", cell):
                date_str = cell
                break
        result["date"] = date_str

        # Pax: look for breakdown (J8W0Y66) or standalone numeric total
        pax_total = pax_j = pax_w = pax_y = None
        for cell in cells[flight_idx:]:
            pax_match = _PAX_RE.match(cell)
            if pax_match:
                pax_j = int(pax_match.group(1))
                pax_w = int(pax_match.group(2))
                pax_y = int(pax_match.group(3))
                pax_total = pax_j + pax_w + pax_y
                break

        if pax_total is None:
            for cell in cells[flight_idx + 5:]:
                if cell.isdigit() and int(cell) > 0:
                    pax_total = int(cell)
                    break

        result["pax_total"] = pax_total
        result["pax_j"] = pax_j
        result["pax_w"] = pax_w
        result["pax_y"] = pax_y
        return result

    @staticmethod
    def _parse_delay_row(cells: List[str]) -> Optional[int]:
        """Parse a delay row like ['', '60minutos', '', '60', 'minutos'].
        Returns delay in minutes or None if not a delay row.
        """
        text = " ".join(c for c in cells if c).strip()
        m = _DELAY_MINUTES_RE.search(text)
        if m and len(text) < 30:
            return int(m.group(1))
        m = _DELAY_HM_RE.search(text)
        if m and len(text) < 30:
            return int(m.group(1)) * 60 + int(m.group(2))
        return None

    @staticmethod
    def _extract_delay_from_text(text: str) -> Optional[int]:
        """Extract delay minutes from description text like '2h 35min. por causas técnicas.'"""
        m = _DELAY_HM_RE.search(text)
        if m:
            return int(m.group(1)) * 60 + int(m.group(2))
        m = _DELAY_MINUTES_RE.search(text)
        if m:
            return int(m.group(1))
        return None

    @staticmethod
    def _extract_metadata(content: str) -> Dict[str, str]:
        meta: Dict[str, str] = {}
        for tag, key in [("Asunto", "subject"), ("Enviados", "date"), ("De", "sender")]:
            m = re.search(rf"<b>{tag}:</b>\s*([^<]+)", content)
            if m:
                meta[key] = m.group(1).strip()
        return meta

    @staticmethod
    def _extract_summary(soup) -> str:
        """Extract the summary paragraph from the email body."""
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                text = row.get_text(strip=True)
                if text and len(text) > 50 and "hemos gestionado" in text.lower():
                    return text
        return ""

    @staticmethod
    def _extract_text_fallback(text: str) -> List[Dict[str, Any]]:
        """Fallback: extract incidents from plain text using regex patterns."""
        incidents: List[Dict[str, Any]] = []
        current: Dict[str, Any] = {}
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            fm = re.search(r"((?:IB|I2|LL|YW)\d{3,5})", line, re.IGNORECASE)
            if fm:
                if current:
                    incidents.append(current)
                current = {"flight": fm.group(1), "incident_type": "Desconocido"}
            rm = re.search(r"([A-Z]{3})-?([A-Z]{3})", line)
            if rm and current:
                current["origin"] = rm.group(1)
                current["destination"] = rm.group(2)
                current["route"] = f"{rm.group(1)}-{rm.group(2)}"
            if any(kw in line.lower() for kw in ("cancel", "retraso", "delay", "desvío", "equipaje", "técnic")):
                if current:
                    current["description"] = line
        if current:
            incidents.append(current)
        return incidents

    # ── Data Collection ──

    def collect_ncs_data_for_date_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Collect NCS data for a date range (day by day)."""
        self.logger.info(f"Collecting NCS data {start_date:%Y-%m-%d} → {end_date:%Y-%m-%d}")
        frames: List[pd.DataFrame] = []
        cur = start_date
        while cur <= end_date:
            for key in self.list_available_files(cur.strftime("%Y-%m-%d")):
                df = self.read_ncs_file(key)
                if not df.empty:
                    df["collection_date"] = cur
                    frames.append(df)
            cur += timedelta(days=1)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            self.logger.info(f"Collected {len(combined)} NCS incidents")
            return combined

        self.logger.warning("No NCS data found for date range")
        return pd.DataFrame()

    # ── Analysis ──

    def analyze_ncs_incidents_for_period(self, df: pd.DataFrame, analysis_focus: str = "all") -> Dict[str, Any]:
        """Analyze NCS incidents to provide operational insights.

        Args:
            df: DataFrame with NCS incident data (output of collect_ncs_data_for_date_range)
            analysis_focus: "flights", "routes", "incidents", or "all"

        Returns:
            Dict with analysis results including incident_counts expected by causal agent.
        """
        if df.empty:
            return {
                "total_incidents": 0,
                "analysis": "No incidents found",
                "incident_counts": {},
                "detailed_incidents": {"count": 0, "sample_incidents": []},
                "route_analysis": {},
                "flight_analysis": {},
            }

        analysis: Dict[str, Any] = {
            "total_incidents": len(df),
            "date_range": {
                "start": df["period_start_date"].iloc[0] if "period_start_date" in df.columns else None,
                "end": df["period_end_date"].iloc[0] if "period_end_date" in df.columns else None,
            },
            "period_info": {
                "period_number": df["period_number"].iloc[0] if "period_number" in df.columns else None,
                "aggregation_days": df["aggregation_days"].iloc[0] if "aggregation_days" in df.columns else None,
            },
        }

        try:
            analysis["incident_counts"] = self._build_incident_counts(df)

            if analysis_focus in ("flights", "all"):
                analysis["flight_analysis"] = self._build_flight_analysis(df)

            if analysis_focus in ("incidents", "all"):
                analysis["detailed_incidents"] = self._build_detailed_incidents(df)
                analysis["incident_categories"] = self._build_incident_categories(df)

            if analysis_focus in ("routes", "all"):
                analysis["route_analysis"] = self._build_route_analysis(df)

            analysis["summary_insights"] = self._generate_summary_insights(analysis)

        except Exception as e:
            analysis["error"] = f"Error during analysis: {e}"

        return analysis

    def _build_incident_counts(self, df: pd.DataFrame) -> Dict[str, int]:
        """Count incidents by type using the structured incident_type column when available."""
        if "incident_type" in df.columns:
            counts = df["incident_type"].value_counts().to_dict()
            return {k: int(v) for k, v in counts.items() if k}

        # Fallback: keyword search across all text
        all_text = " ".join(df.astype(str).values.flatten()).lower()
        mapping = {
            "Cancelaciones": ["cancel", "anulad"],
            "Retrasos": ["retraso", "delay"],
            "Desvíos": ["desvio", "desvío", "divert"],
            "Limitación de la aeronave": ["aircraft", "aeronave", "técnic"],
            "Equipaje": ["equipaje", "baggage"],
            "Otras incidencias": ["other", "otro"],
        }
        return {k: sum(all_text.count(kw) for kw in kws) for k, kws in mapping.items() if sum(all_text.count(kw) for kw in kws) > 0}

    def _build_flight_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        if "flight" not in df.columns:
            return {"total_flights_affected": 0, "most_affected_flights": {}, "flight_impact_summary": "No flight data"}
        counts = df["flight"].dropna().value_counts()
        return {
            "total_flights_affected": len(counts),
            "most_affected_flights": counts.head(5).to_dict(),
            "flight_impact_summary": f"{len(counts)} flights affected",
        }

    def _build_detailed_incidents(self, df: pd.DataFrame) -> Dict[str, Any]:
        if "description" not in df.columns:
            return {"count": 0, "sample_incidents": [], "incident_themes": []}
        detailed = df[df["description"].str.len() > 50] if "description" in df.columns else pd.DataFrame()
        texts = detailed["description"].tolist() if not detailed.empty else []
        return {
            "count": len(texts),
            "sample_incidents": texts[:3],
            "incident_themes": self._extract_incident_themes(texts),
        }

    def _build_incident_categories(self, df: pd.DataFrame) -> Dict[str, Any]:
        if "incident_type" not in df.columns:
            return {"categories_found": [], "category_summary": "No category data"}
        cats = df["incident_type"].dropna().unique().tolist()
        return {"categories_found": cats, "category_summary": f"{len(cats)} incident categories"}

    def _build_route_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        if "route" in df.columns:
            routes = df["route"].dropna()
            routes = routes[routes.str.match(r"^[A-Z]{3}-[A-Z]{3}$")]
        else:
            # Fallback: scan all text columns
            route_pattern = r"[A-Z]{3}-[A-Z]{3}"
            all_routes: List[str] = []
            for col in df.columns:
                for val in df[col].astype(str):
                    all_routes.extend(re.findall(route_pattern, val))
            routes = pd.Series(all_routes)

        if routes.empty:
            return {"total_routes_affected": 0, "most_affected_routes": {}, "route_impact_summary": "No routes found", "all_routes": []}

        counts = routes.value_counts()
        return {
            "total_routes_affected": len(counts),
            "most_affected_routes": counts.head(5).to_dict(),
            "route_impact_summary": f"{len(counts)} routes affected",
            "all_routes": list(counts.index),
        }

    @staticmethod
    def _extract_incident_themes(texts: List[str]) -> List[str]:
        theme_keywords = {
            "technical_issues": ["técnica", "technical", "avería", "breakdown"],
            "weather": ["weather", "tiempo", "meteorológica", "tormenta"],
            "bird_strike": ["aves", "bird", "impacto"],
            "delays": ["retraso", "delay"],
            "cancellations": ["cancel", "anulado"],
            "baggage": ["equipaje", "baggage"],
            "crew": ["tripulación", "crew", "piloto"],
        }
        counts = {}
        for theme, keywords in theme_keywords.items():
            n = sum(1 for t in texts if any(kw in t.lower() for kw in keywords))
            if n:
                counts[theme] = n
        return [f"{t}: {c} incidents" for t, c in sorted(counts.items(), key=lambda x: x[1], reverse=True)][:5]

    @staticmethod
    def _generate_summary_insights(analysis: Dict) -> List[str]:
        insights = [f"Total of {analysis.get('total_incidents', 0)} operational incidents detected"]
        if fa := analysis.get("flight_analysis"):
            insights.append(f"Flight impact: {fa['total_flights_affected']} flights affected")
        if ra := analysis.get("route_analysis"):
            insights.append(f"Route impact: {ra['total_routes_affected']} routes affected")
        if di := analysis.get("detailed_incidents"):
            insights.append(f"Detailed incidents: {di['count']} with descriptions")
        return insights
