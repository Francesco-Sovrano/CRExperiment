import argparse
import csv
import datetime as dt
import os
from pathlib import Path
import re
import tempfile
import zipfile
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

extra_data_folders = {
	'2': (2, 'scenario_2_bis'),
}

questionnaire_labels_map = {
	"prolific_id": "Prolific ID",
	"question_pre1": "How would you rate your overall attitude toward Artificial Intelligence (AI)?",
	"question_pre2": "How much do you trust AI systems in general?",
	"question_pre3": "What is one word or phrase that describes how you feel about AI?",
	"post1": "What is your gender?",
	"question_age": "What is your Age?",
	"post2": "How familiar are you with Artificial Intelligence (AI)?",
	"question_post3": "Your profession or role:",
	"post9": "Do you have any feedback about the AI output, the explanation, or the task itself?",
}

scenario_labels_map = {
	"prolific_id": "Prolific ID",
	"time_spent_seconds": "Seconds",
	"changed_mind": "Explanation changed mind",
	"reason_que": "What made you change your decision?",
	"task": "Task file",
	"format": "Format",
	"expected_answer": "Expected answer",
	"is_MAGIX_explanation": "Explanation is MAGIX-defined",
	"user_response": "Response before explanation",
	# "time_to_user_response_seconds": "Seconds to Decision (without explanation)",
	"sq1": "How confident are you in the decision you made? (without explanation)",
	"user_response_XAI": "Response after explanation",
	# "time_sq1_to_user_response_XAI_seconds": "Seconds to Decision (with explanation)",
	"sq2": "How confident are you in the decision you made? (with explanation)",
	"sq3": "Did the explanation help you evaluate the AI's output?",
	"sq4": "How useful was the explanation provided?",
	"sq5": "How easy was it to understand the explanation?",
	"sq6": "How much effort did it take to understand and complete this task?",
	"user_error": "User error",
}


def parse_timestamp(ts):
	"""
	Parse timestamp like '2025-07-26 17:21:43.548237'
	"""
	# fromisoformat handles this format directly
	return dt.datetime.fromisoformat(ts)


def read_log_records(path):
	"""
	Read a .log file into a list of (timestamp, key, value).
	Each line has format: <timestamp>;<key>;<value>
	"""
	records = []
	with path.open("r", encoding="utf-8", errors="replace") as f:
		for line in f:
			line = line.rstrip("\n")
			if not line:
				continue
			parts = line.split(";", 2)
			if len(parts) < 3:
				# Skip malformed lines
				continue
			ts_str, key, value = parts[0], parts[1], parts[2]
			try:
				ts = parse_timestamp(ts_str)
			except Exception:
				# Skip lines with bad timestamps
				continue
			records.append((ts, key, value))
	return records


def find_latest_prolific_idx(records):
	"""
	Return index of the latest 'prolific_id' entry, or None if not present.
	"""
	idx = None
	for i, (_, key, _) in enumerate(records):
		if key == "prolific_id":
			idx = i
	return idx


def rename_scenario_field(field, n):
	"""
	Remove scenario index number from relevant fields:
	  task1 -> task
	  user_response_1 -> user_response
	  user_response_1_XAI -> user_response_XAI
	  s1q1 -> sq1
	  s1_reason -> s_reason
	Other fields are returned unchanged.
	"""
	

	# Exact tokens first
	if field == f"task{n}":
		return "task"
	if field == f"user_response_{n}" or field == f"user_response_{n if not isinstance(n, str) else n[0]}":
		return "user_response"
	if field == f"user_response_{n}_XAI" or field == f"user_response_{n if not isinstance(n, str) else n[0]}_XAI":
		return "user_response_XAI"

	# s{n}..... -> remove the {n} immediately following 's'
	# Examples: s1q1 -> sq1, s1_reason -> s_reason
	if field.startswith(f"s{n}"):
		field = "s" + field[len(f"s{n}") :]

	if field == "s_reason":
		return "reason_que"

	return field


def slice_scenario(records_after, n):
	"""
	Given records AFTER latest prolific_id, return the slice for scenario n:
	from after 'start;scenarioN' up to and including 'sNq6'.
	"""
	start_idx = None
	for i, (_, key, val) in enumerate(records_after):
		if key == f"task{n}":
			start_idx = i
			break
	if start_idx is None:
		return None

	# Find sNq6 at/after start
	end_idx = None
	target_end_key = f"s{n}q6"
	for j in range(start_idx + 1, len(records_after)):
		if records_after[j][1] == target_end_key:
			end_idx = j
			break
	if end_idx is None:
		# Scenario start found but no sNq6 -> treat as incomplete, skip
		return None

	return records_after[start_idx: end_idx + 1]


def build_questionnaire_row(records_after, prolific_id_value):
	"""
	Build questionnaire row from records_after.
	If multiple entries for a field exist, take the last occurrence.
	"""
	row = {"prolific_id": prolific_id_value}
	target = set(questionnaire_labels_map.keys()) - {"prolific_id"}
	last_values = {}
	for _, key, val in records_after:
		if key in target:
			last_values[key] = val
	row.update(last_values)
	return row


def build_scenario_row(slice_records, n, prolific_id_value, log_files, special_task_label=None):
	"""
	Build a row for scenario n from its slice (after start;scenarioN up to sNq6).
	Adds time_spent_seconds and derived fields.
	"""
	row = {"prolific_id": prolific_id_value}

	if not slice_records:
		return None

	# time spent from first entry in slice to sNq6 (the last in slice)
	first_ts = slice_records[0][0]
	last_ts = slice_records[-1][0]
	time_spent_seconds = int((last_ts - first_ts).total_seconds())
	row["time_spent_seconds"] = time_spent_seconds

	# # # NEW: time from scenario start to first user response
	# # for ts, key, _ in slice_records:
	# # 	print(key, key == "user_response")
	# ts_user_resp = next((ts for ts, key, _ in slice_records if key == f"user_response_{n}"), None)
	# if not ts_user_resp:
	# 	return None
	# row["time_to_user_response_seconds"] = int((ts_user_resp - first_ts).total_seconds())
	
	# # NEW: time from first confidence question to response after explanation
	# ts_sq1 = next((ts for ts, key, _ in slice_records if key == f"s{n}q1"), None)
	# ts_user_resp_xai = next((ts for ts, key, _ in slice_records if key == f"user_response_{n}_XAI"), None)
	# if not ts_user_resp_xai:
	# 	return None
	# row["time_sq1_to_user_response_XAI_seconds"] = int((ts_user_resp_xai - ts_sq1).total_seconds())
	
	# Collect fields in-order; keep last occurrence if duplicates
	tmp = {}
	for _, key, val in slice_records:
		new_key = rename_scenario_field(key, n)
		tmp[new_key] = val

	row.update(tmp)

	# Derived fields
	task_val = row.get("task", None)
	if not task_val:
		return None

	if special_task_label:
		task_val = row['task'] = row['task'].replace(f'task{n}', f'task{special_task_label}')

	task_format = row['format'] = 'image_based' if 'image_based' in str(log_files) else 'text_based'

	if task_format == 'image_based':
		if '2bis' in row['task']:
			task_val = row['task'] = row['task'].replace("2bis", "2hard")
		elif 'task2_' in row['task']:
			task_val = row['task'] = row['task'].replace("2", "2easy")
	elif task_format == 'text_based' and 'scenario_2_bis' not in str(log_files):
		if 'task2_' in row['task']:
			task_val = row['task'] = row['task'].replace("task2_", "task2bis_").replace("MAGIX", "NonMAGIX").replace("NonNonMAGIX", "MAGIX")

	expected_answer = "Accept" if "_corr_" in task_val else "Reject"
	row["expected_answer"] = expected_answer
	user_resp_xai = row["user_response_XAI"].strip().lower()
	row["user_error"] = (user_resp_xai != expected_answer.strip().lower())
	row["is_MAGIX_explanation"] = "_MAGIX.json" in task_val

	row["changed_mind"] = (user_resp_xai != row["user_response"].strip().lower())
	# if row["changed_mind"]:
	#     print(row["reason_que"])
	return row


def write_merged_csv(rows, out_path, preferred_order_map):
	if not rows:
		# Write an empty file with no rows but a tiny header for clarity
		out_path.write_text("", encoding="utf-8")
		return

	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=list(preferred_order_map.values()))
		writer.writeheader()
		for r in rows:
			writer.writerow({v: r.get(k,'') for k,v in preferred_order_map.items()})


def gather_log_files(input_path):
	"""
	Return a list of .log file paths.
	If input_path is a zip, extract it to a temp dir and return .log files within.
	If it's a directory, walk recursively for .log files.
	"""
	log_paths = []
	if input_path.is_file() and input_path.suffix.lower() == ".zip":
		tempdir = Path(tempfile.mkdtemp(prefix="magix_logs_"))
		with zipfile.ZipFile(input_path, "r") as z:
			z.extractall(tempdir)
		for p in tempdir.glob("*.log"):
			log_paths.append(p)
		return log_paths

	if input_path.is_dir():
		for p in input_path.glob("*.log"):
			log_paths.append(p)
	return log_paths

def get_scenario_rows(log_files, special_task_label=None):
	questionnaire_rows = []
	scenarios_rows = defaultdict(list)

	for path in sorted(log_files):
		records = read_log_records(path)
		if not records:
			continue

		latest_idx = find_latest_prolific_idx(records)
		if latest_idx is None:
			continue

		prolific_id_value = records[latest_idx][2]
		records_after = records[latest_idx + 1 :]

		# questionnaire
		questionnaire_row = build_questionnaire_row(records_after, prolific_id_value)
		questionnaire_rows.append(questionnaire_row)

		# scenarios 1..4
		for n in (1, 2, '2bis', 3, 4):
			slice_rec = slice_scenario(records_after, n)
			if slice_rec is None:
				continue
			row = build_scenario_row(slice_rec, n, prolific_id_value, log_files, special_task_label=special_task_label)
			if row:
				if row['format'] == 'image_based':
					if n == '2bis':
						n = '2hard'
					elif n == 2:
						n = '2easy'
				elif row['format'] == 'text_based' and 'scenario_2_bis' not in str(log_files):
					if n == 2:
						n = '2bis'
				scenarios_rows[n].append(row)
	return questionnaire_rows, scenarios_rows

def main():
	parser = argparse.ArgumentParser(description="Parse MAGIX logs and produce merged CSV tables.")
	parser.add_argument(
		"--inputs", 
		required=True, 
		nargs="+", 
		help="One or more directories containing .log files, or .zip files of them."
	)
	parser.add_argument("--output", required=True, help="Directory to write CSV outputs.")
	args = parser.parse_args()

	output_dir = Path(args.output)
	output_dir.mkdir(parents=True, exist_ok=True)

	questionnaire_rows_all = []
	scenarios_rows_all = defaultdict(list)

	for input_path_str in args.inputs:
		input_path = Path(input_path_str)
		log_files = gather_log_files(input_path)
		if not log_files:
			print(f"No .log files found in {input_path}.")
			continue

		questionnaire_rows, scenarios_rows = get_scenario_rows(log_files)
		# for n, v in scenarios_rows.items():
		# 	print(0, n, len(v), input_path)

		# Add extra experiments (per input path)
		for label, (n, extra_path) in extra_data_folders.items():
			full_path = input_path / Path(extra_path)
			if not full_path.exists():
				continue
			_questionnaire_rows, _scenarios_rows = get_scenario_rows(
				gather_log_files(full_path), 
				special_task_label=label
			)
			scenarios_rows[label] = _scenarios_rows[n]
			# print(0, label, len(_scenarios_rows[n]), full_path)
			questionnaire_rows += _questionnaire_rows

		questionnaire_rows_all.extend(questionnaire_rows)
		for n, v in scenarios_rows.items():
			scenarios_rows_all[str(n)] += v

	for n, v in scenarios_rows_all.items():
		print('Dataset size:', n, len(v))

	# Write outputs
	write_merged_csv(
		questionnaire_rows_all,
		output_dir / "questionnaire.csv",
		preferred_order_map=questionnaire_labels_map,
	)
	print(f"Wrote: {output_dir / 'questionnaire.csv'}")

	for n, v in scenarios_rows_all.items():
		write_merged_csv(
			v,
			output_dir / f"scenario_{n}.csv",
			preferred_order_map=scenario_labels_map,
		)
		print(f"Wrote: {output_dir / f'scenario_{n}.csv'}")


if __name__ == "__main__":
	main()
