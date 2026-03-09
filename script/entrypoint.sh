#!/usr/bin/env bash

echo -e "Generating Catia report...\n"

weekly_args=()
if [[ -n "${INSERT_DATE_CI:-}" ]]; then
  weekly_args+=(--insert-date-ci "${INSERT_DATE_CI}")
fi
python -u dashboard_analyzer/weekly_deep_research.py --focus-touchpoint "Cabin Crew" "${weekly_args[@]}"

echo -e "\nCatia report generation completed."
