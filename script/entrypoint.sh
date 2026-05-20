#!/usr/bin/env bash

echo -e "Generating Catia report...\n"

weekly_args=()
if [[ -n "${INSERT_DATE_CI:-}" ]]; then
  weekly_args+=(--insert-date-ci "${INSERT_DATE_CI}")
fi
if [[ -n "${FOCUS_TOUCHPOINT:-}" ]]; then
  # FOCUS_TOUCHPOINT can be a single value or comma-separated list
  # e.g. "Cabin Crew" or "Cabin Crew,Punctuality,Check-in"
  IFS=',' read -ra _tp_array <<< "${FOCUS_TOUCHPOINT}"
  weekly_args+=(--focus-touchpoint "${_tp_array[@]}")
fi

python -u dashboard_analyzer/weekly_deep_research.py "${weekly_args[@]}"

echo -e "\nCatia report generation completed."
