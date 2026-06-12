#!/usr/bin/env python3
"""
Cliente standalone para consultar Power BI (REST API + DAX).

Requisitos:
    pip install msal requests pandas python-dotenv

Variables en .env (raíz del proyecto) o en el entorno:
    CLIENT_ID, CLIENT_SECRET, TENANT_ID, GROUP_ID, DATASET_ID
    PBI_API_TIMEOUT=120   (opcional)

Uso:
    # Consulta mínima de prueba (última fecha disponible)
    python scripts/pbi_query_client.py

    # DAX desde archivo (plantillas del proyecto)
    python scripts/pbi_query_client.py --query-file dashboard_analyzer/data_collection/queries/NPS_flex_agg.txt

    # DAX inline
    python scripts/pbi_query_client.py --query "EVALUATE ROW(\"ok\", 1)"

    # Guardar CSV
    python scripts/pbi_query_client.py --query-file ... --output resultado.csv
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import msal
import pandas as pd
import requests
from dotenv import load_dotenv

# --- Configuración de reintentos (igual que pbi_collector.py) ---
PBI_MAX_RETRIES = 3
PBI_BASE_DELAY = 2.0
PBI_MAX_DELAY = 30.0
PBI_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Consulta de prueba: última fecha en Date_Master (ajusta si tu modelo usa otro nombre)
DEFAULT_TEST_QUERY = """
EVALUATE
TOPN(
    5,
    VALUES('Date_Master'[Date]),
    'Date_Master'[Date],
    DESC
)
""".strip()


class PowerBIClient:
    """Autenticación Azure AD (service principal) + executeQueries sobre un dataset."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        group_id: str,
        dataset_id: str,
        api_timeout: int = 120,
    ):
        self.client_id = client_id.strip()
        self.client_secret = client_secret.strip()
        self.tenant_id = tenant_id.strip()
        self.group_id = group_id.strip()
        self.dataset_id = dataset_id.strip()
        self.api_timeout = api_timeout
        self.access_token = self._get_access_token()

    def _get_access_token(self) -> str:
        authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        scope = ["https://analysis.windows.net/powerbi/api/.default"]

        app = msal.ConfidentialClientApplication(
            client_id=self.client_id,
            client_credential=self.client_secret,
            authority=authority,
        )
        result = app.acquire_token_for_client(scopes=scope)

        if "access_token" not in result:
            error = result.get("error_description") or result.get("error") or result
            raise RuntimeError(f"No se pudo obtener token de Azure AD: {error}")

        return result["access_token"]

    @property
    def execute_queries_url(self) -> str:
        return (
            f"https://api.powerbi.com/v1.0/myorg/groups/{self.group_id}"
            f"/datasets/{self.dataset_id}/executeQueries"
        )

    def execute_dax(
        self,
        query: str,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> pd.DataFrame:
        """Ejecuta una consulta DAX y devuelve un DataFrame."""
        timeout = timeout_seconds if timeout_seconds is not None else self.api_timeout
        retries = max_retries if max_retries is not None else PBI_MAX_RETRIES

        payload = {
            "queries": [{"query": query}],
            "serializerSettings": {"includeNulls": True},
        }
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        last_error = None
        for attempt in range(retries + 1):
            try:
                response = requests.post(
                    self.execute_queries_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )

                if response.status_code in PBI_RETRYABLE_STATUS_CODES:
                    last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                    if attempt < retries:
                        delay = min(
                            PBI_BASE_DELAY * (2**attempt) + random.uniform(0, 1),
                            PBI_MAX_DELAY,
                        )
                        print(f"Reintento en {delay:.1f}s ({attempt + 1}/{retries + 1})...")
                        time.sleep(delay)
                        continue
                    raise RuntimeError(last_error)

                if response.status_code != 200:
                    raise RuntimeError(
                        f"PBI API {response.status_code}: {response.text[:500]}"
                    )

                results = response.json()
                if not results.get("results") or not results["results"][0].get("tables"):
                    last_error = "Respuesta vacía (sin tablas)"
                    if attempt < retries:
                        delay = min(
                            PBI_BASE_DELAY * (2**attempt) + random.uniform(0, 1),
                            PBI_MAX_DELAY,
                        )
                        print(f"Sin datos, reintento en {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    return pd.DataFrame()

                rows = results["results"][0]["tables"][0].get("rows", [])
                return pd.DataFrame(rows)

            except requests.exceptions.Timeout as e:
                last_error = str(e)
                if attempt < retries:
                    delay = min(
                        PBI_BASE_DELAY * (2**attempt) + random.uniform(0, 1),
                        PBI_MAX_DELAY,
                    )
                    time.sleep(delay)
                    continue
                raise

            except requests.exceptions.ConnectionError as e:
                last_error = str(e)
                if attempt < retries:
                    delay = min(
                        PBI_BASE_DELAY * (2**attempt) + random.uniform(0, 1),
                        PBI_MAX_DELAY,
                    )
                    time.sleep(delay)
                    continue
                raise

        raise RuntimeError(f"Consulta fallida tras reintentos: {last_error}")


def load_env(project_root: Optional[Path] = None) -> Path:
    """Carga .env desde la raíz del repo."""
    root = project_root or Path(__file__).resolve().parents[1]
    dotenv_path = root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=True)
    return root


def get_client_from_env() -> PowerBIClient:
    """Construye el cliente leyendo variables de entorno."""
    required = [
        "CLIENT_ID",
        "CLIENT_SECRET",
        "TENANT_ID",
        "GROUP_ID",
        "DATASET_ID",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise ValueError(
            "Faltan variables de entorno: "
            + ", ".join(missing)
            + "\nAñádelas al .env en la raíz del proyecto."
        )

    timeout = int(os.getenv("PBI_API_TIMEOUT", "120"))
    return PowerBIClient(
        client_id=os.environ["CLIENT_ID"],
        client_secret=os.environ["CLIENT_SECRET"],
        tenant_id=os.environ["TENANT_ID"],
        group_id=os.environ["GROUP_ID"],
        dataset_id=os.environ["DATASET_ID"],
        api_timeout=timeout,
    )


def load_query_from_file(path: Path, aggregation_days: int = 7) -> str:
    """
    Lee un archivo .txt con DAX.
    Si es NPS_flex_agg.txt, sustituye {AGGREGATION_DAYS} como en pbi_collector.
    """
    text = path.read_text(encoding="utf-8")
    if "{AGGREGATION_DAYS}" in text:
        text = text.replace("{AGGREGATION_DAYS}", str(aggregation_days))
    return text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ejecutar consultas DAX contra el dataset de Power BI (Iberia NPS)."
    )
    parser.add_argument(
        "--query",
        help="Consulta DAX como string (entre comillas en la shell)",
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        help="Ruta a archivo .txt con DAX (p. ej. plantillas en data_collection/queries/)",
    )
    parser.add_argument(
        "--aggregation-days",
        type=int,
        default=7,
        help="Sustituye {AGGREGATION_DAYS} en plantillas flex (default: 7)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Guardar resultado en CSV",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=20,
        help="Filas a mostrar por consola (default: 20)",
    )
    args = parser.parse_args()

    project_root = load_env()
    print(f"Proyecto: {project_root}")
    print("Autenticando con Azure AD (service principal)...")

    try:
        client = get_client_from_env()
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print("OK: token obtenido")

    if args.query_file:
        qpath = args.query_file
        if not qpath.is_absolute():
            qpath = project_root / qpath
        if not qpath.exists():
            print(f"ERROR: no existe {qpath}", file=sys.stderr)
            return 1
        query = load_query_from_file(qpath, args.aggregation_days)
        print(f"Consulta cargada desde: {qpath}")
    elif args.query:
        query = args.query
    else:
        query = DEFAULT_TEST_QUERY
        print("Usando consulta de prueba (últimas fechas en Date_Master).")
        print("Usa --query o --query-file para otra consulta.")

    print("Ejecutando DAX en Power BI...")
    try:
        df = client.execute_dax(query)
    except Exception as e:
        print(f"ERROR al ejecutar consulta: {e}", file=sys.stderr)
        return 1

    print(f"Filas: {len(df)} | Columnas: {list(df.columns)}")
    if df.empty:
        print("(sin datos)")
    else:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        print(df.head(args.head).to_string(index=False))

    if args.output:
        out = args.output
        if not out.is_absolute():
            out = project_root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Guardado: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
