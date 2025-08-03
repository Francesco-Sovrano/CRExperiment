#!/bin/bash

source .env/bin/activate

python3 parse_magix_logs.py --input experiment_results/ --output formatted_results/
python3 reliance_analysis.py --input formatted_results/ --output visualizations_and_statistics/ --min-seconds 30
python3 reliance_analysis.py --input formatted_results/ --output visualizations_and_statistics/ --min-seconds 30 --keep_only_who_changed_mind