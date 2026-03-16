import asyncio
import pandas as pd
import json
from datetime import datetime
from dashboard_analyzer.data_collection.pbi_collector import PBIDataCollector

async def compare_queries():
    collector = PBIDataCollector(environment="dev")
    
    start_date = datetime(2026, 2, 1) # Use Feb just in case March is too fresh
    end_date = datetime(2026, 2, 28)
    touchpoint_name = "Punctuality"
    
    # Test cases
    scenarios = [
        {"name": "Global/SH", "node_path": "Global/SH"},
        {"name": "Global/LH", "node_path": "Global/LH"}
    ]
    
    results_data = []
    for scenario in scenarios:
        # ... (same logic as before) ...
        # (I will just rewrite the loop slightly to collect results)
        cabins, companies, hauls = collector._get_node_filters(scenario["node_path"])
        
        # 1. ACTUAL (Current) Query
        start_date_str = f"{start_date.year}, {start_date.month}, {start_date.day}"
        end_date_str = f"{end_date.year}, {end_date.month}, {end_date.day}"
        cabin_values = " || ".join([f"Cabin_Master[Cabin_Show] = \"{c}\"" for c in cabins])
        haul_values = " || ".join([f"Haul_Master[Haul_Aggr] = \"{h}\"" for h in hauls])
        company_values = " || ".join([f"Company_Master[Company] = \"{c}\"" for c in companies])
        
        current_query = (
            "EVALUATE\n"
            f"VAR _start = DATE({start_date_str})\n"
            f"VAR _end   = DATE({end_date_str})\n"
            "VAR _tabla =\n"
            "    SUMMARIZECOLUMNS(\n"
            "        TouchPoint_Master[filtered_name],\n"
            f"        FILTER(ALL(Date_Master), Date_Master[Date] >= _start && Date_Master[Date] <= _end),\n"
            f"        FILTER(ALL(Cabin_Master), {cabin_values}),\n"
            f"        FILTER(ALL(Haul_Master), {haul_values}),\n"
            f"        FILTER(ALL(Company_Master), {company_values}),\n"
            f"        FILTER(ALL(TouchPoint_Master), TouchPoint_Master[filtered_name] = \"{touchpoint_name}\"),\n"
            '        "CSAT", [Monthly_Satisfaction],\n'
            '        "Target_CSAT", [Target_Satisfaction_filtered]\n'
            "    )\n"
            "RETURN _tabla\n"
        )
        
        def to_dax_list(items):
            return "{" + ", ".join([f"\"{i}\"" for i in items]) + "}"
            
        proposed_query = (
            "EVALUATE\n"
            f"VAR _start = DATE({start_date_str})\n"
            f"VAR _end   = DATE({end_date_str})\n"
            "VAR _tabla =\n"
            "    SUMMARIZECOLUMNS(\n"
            "        TouchPoint_Master[filtered_name],\n"
            f"        FILTER(KEEPFILTERS(VALUES(Date_Master[Date])), Date_Master[Date] >= _start && Date_Master[Date] <= _end),\n"
            f"        TREATAS({to_dax_list(cabins)}, Cabin_Master[Cabin_Show]),\n"
            f"        TREATAS({to_dax_list(hauls)}, Haul_Master[Haul_Aggr]),\n"
            f"        TREATAS({to_dax_list(companies)}, Company_Master[Company]),\n"
            f"        TREATAS({{\"{touchpoint_name}\"}}, TouchPoint_Master[filtered_name]),\n"
            '        "CSAT", [Monthly_Satisfaction],\n'
            '        "Target_CSAT", [Target_Satisfaction_filtered]\n'
            "    )\n"
            "RETURN _tabla\n"
        )
        
        df_current = await collector._execute_query_async(current_query)
        if not df_current.empty: df_current = collector._safe_clean_columns(df_current)
        
        df_proposed = await collector._execute_query_async(proposed_query)
        if not df_proposed.empty: df_proposed = collector._safe_clean_columns(df_proposed)
        
        results_data.append({
            "scenario": scenario["name"],
            "current": df_current.to_dict(orient="records") if not df_current.empty else [],
            "proposed": df_proposed.to_dict(orient="records") if not df_proposed.empty else []
        })

    with open("compare_results.json", "w") as f:
        json.dump(results_data, f, indent=4)
    print("Results saved to compare_results.json")

if __name__ == "__main__":
    asyncio.run(compare_queries())
