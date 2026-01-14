#!/usr/bin/env bash

echo -e "Generating Catia report...\n"

python -u dashboard_analyzer/weekly_deep_research.py --insert-date-ci "${INSERT_DATE_CI}"

echo -e "\nCatia report generation completed."
