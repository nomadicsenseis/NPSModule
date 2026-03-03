"""
Test script para verificar la query de focus touchpoint CSAT vs target.

Ejecutar:
    python test_focus_touchpoint_query.py

Hace dos cosas:
1. Discovery: lista todos los touchpoints disponibles con [Monthly_Satisfaction] para
   el nodo Global_SH_Economy en enero 2026, para encontrar el filtered_name correcto.
2. Test directo: llama a collect_focus_touchpoint_csat_vs_target con "Cabin Crew"
   para confirmar que devuelve datos.
"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector

# Parámetros de prueba
NODE_PATH = "Global_SH_Economy"
START_DATE = datetime(2026, 1, 20)
END_DATE = datetime(2026, 1, 26)
TOUCHPOINT_CANDIDATE = "Cabin Crew"


async def run_discovery(collector: PBIDataCollector):
    """Ejecuta query de discovery para listar todos los touchpoints disponibles."""
    print("\n" + "="*60)
    print("DISCOVERY: Todos los touchpoints con Monthly_Satisfaction")
    print(f"Nodo: {NODE_PATH} | {START_DATE.date()} → {END_DATE.date()}")
    print("="*60)

    cabins, companies, hauls = collector._get_node_filters(NODE_PATH)
    cabin_values = " || ".join([f'Cabin_Master[Cabin_Show] = "{c}"' for c in cabins])
    haul_values = " || ".join([f'Haul_Master[Haul_Aggr] = "{h}"' for h in hauls])
    company_values = " || ".join([f'Company_Master[Company] = "{c}"' for c in companies])

    start_str = f"{START_DATE.year}, {START_DATE.month}, {START_DATE.day}"
    end_str = f"{END_DATE.year}, {END_DATE.month}, {END_DATE.day}"

    discovery_query = (
        "EVALUATE\n"
        f"VAR _start = DATE({start_str})\n"
        f"VAR _end   = DATE({end_str})\n"
        "VAR _tabla =\n"
        "    SUMMARIZECOLUMNS(\n"
        "        TouchPoint_Master[filtered_name],\n"
        "        TREATAS({1}, TouchPoint_Master[explanatory_drivers]),\n"
        f"        FILTER(ALL(Date_Master), Date_Master[Date] >= _start && Date_Master[Date] <= _end),\n"
        f"        FILTER(ALL(Cabin_Master), {cabin_values}),\n"
        f"        FILTER(ALL(Haul_Master), {haul_values}),\n"
        f"        FILTER(ALL(Company_Master), {company_values}),\n"
        '        "CSAT", [Monthly_Satisfaction],\n'
        '        "Target_CSAT", [Target_Satisfaction_filtered]\n'
        "    )\n"
        "RETURN _tabla\n"
        "ORDER BY [CSAT] ASC\n"
    )

    print("\nQuery DAX:")
    print(discovery_query)

    df = await collector._execute_query_async(discovery_query)

    if df.empty:
        print("❌ Sin resultados — revisar fechas o filtros")
    else:
        df = collector._safe_clean_columns(df)
        print(f"\n✅ {len(df)} touchpoints encontrados:\n")
        print(df.to_string(index=False))

    return df


async def run_direct_test(collector: PBIDataCollector):
    """Test directo con el touchpoint candidato."""
    print("\n" + "="*60)
    print(f"TEST DIRECTO: '{TOUCHPOINT_CANDIDATE}' en {NODE_PATH}")
    print("="*60)

    result = await collector.collect_focus_touchpoint_csat_vs_target(
        node_path=NODE_PATH,
        start_date=START_DATE,
        end_date=END_DATE,
        touchpoint_name=TOUCHPOINT_CANDIDATE,
    )

    if result:
        print(f"\n✅ Resultado:")
        print(f"   CSAT:   {result['csat']:.1f}")
        print(f"   Target: {result['target']:.1f}")
        print(f"   Gap:    {result['gap']:+.1f} pts")
    else:
        print(f"\n❌ Sin datos para '{TOUCHPOINT_CANDIDATE}'")
        print("   → Revisa el filtered_name en la tabla de discovery de arriba")


async def main():
    collector = PBIDataCollector(environment="local")
    await run_discovery(collector)
    await run_direct_test(collector)


if __name__ == "__main__":
    asyncio.run(main())
