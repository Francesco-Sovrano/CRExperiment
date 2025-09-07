#!/bin/bash

source .env/bin/activate

python3 parse_magix_logs.py --input experiment_results/image_based experiment_results/text_based --output formatted_results/all
python3 reliance_analysis.py --input formatted_results/all --output visualizations_and_statistics/all --keep_only_who_changed_decision
python3 reliance_analysis.py --input formatted_results/all --output visualizations_and_statistics/all
python3 reliance_analysis.py --input formatted_results/all --output visualizations_and_statistics/all --keep_only_who_easily_understood_explanation

python3 parse_magix_logs.py --input experiment_results/image_based --output formatted_results/image_based
python3 reliance_analysis.py --input formatted_results/image_based --output visualizations_and_statistics/image_based --keep_only_who_changed_decision
python3 reliance_analysis.py --input formatted_results/image_based --output visualizations_and_statistics/image_based
python3 reliance_analysis.py --input formatted_results/image_based --output visualizations_and_statistics/image_based --keep_only_who_easily_understood_explanation

python3 parse_magix_logs.py --input experiment_results/text_based --output formatted_results/text_based
python3 reliance_analysis.py --input formatted_results/text_based --output visualizations_and_statistics/text_based --keep_only_who_changed_decision
python3 reliance_analysis.py --input formatted_results/text_based --output visualizations_and_statistics/text_based
python3 reliance_analysis.py --input formatted_results/text_based --output visualizations_and_statistics/text_based --keep_only_who_easily_understood_explanation
