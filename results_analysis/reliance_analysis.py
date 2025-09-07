import argparse, os, zipfile, glob, tempfile, shutil
import pandas as pd
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter, MaxNLocator
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import TTestIndPower, NormalIndPower, GofChisquarePower
import statsmodels.stats.api as sms
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, spearmanr
import numpy as np
from matplotlib.lines import Line2D

SHOW_ALL_FIGURES = False

# ---- Force vector text embedding ----
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

plt.rcParams.update({
	"figure.dpi": 150,
	"axes.titlesize": 15,
	"axes.labelsize": 12,
	"xtick.labelsize": 10,
	"ytick.labelsize": 10,
	"legend.fontsize": 10,
	"axes.grid": True,
	"grid.alpha": 0.3,
})

RELIANCE_ORDER = ["Appropriate accept", "Appropriate reject", "Over-reliance", "Under-reliance"]
HATCHES = {
	"Appropriate accept": "",
	"Appropriate reject": "..",
	"Over-reliance": "//..",
	"Under-reliance": "//",
}

scenario_name_map = {
	'Scenario 2easy': {'all':'Q2','image_based':'Q2'},
	'Scenario 2hard': {'all':'Q2','image_based':'Q2-Hard'},
	'Scenario 2bis': {'all':'Q4','text_based':'Q2→Q4'},
	# 'Scenario 2bis': 'Q4',
	'Scenario 2': 'Q2',
	'Scenario 1': 'Q1',
	'Scenario 3': 'Q3',
	'Scenario 4': 'Q4',
}

def ensure_dir(p):
	os.makedirs(p, exist_ok=True)

def annotate_bars(ax, fmt="{:.0f}", y_is_pct=False):
	for c in ax.containers:
		for b in c:
			h = b.get_height()
			if h <= 0:
				continue
			y = b.get_y() + h
			ax.annotate(fmt.format(h*100 if y_is_pct else h),
						(b.get_x()+b.get_width()/2, y), xytext=(0,3),
						textcoords="offset points", ha="center", va="bottom", fontsize=9)

def load_frames(path, output_dir):
	tmp_dir = None
	try:
		base = path
		if path.lower().endswith(".zip"):
			tmp_dir = tempfile.mkdtemp()
			with zipfile.ZipFile(path, "r") as zf:
				zf.extractall(tmp_dir)
			base = tmp_dir
		questionnaire_pattern = os.path.join(base, "**", "questionnaire.csv")
		questionnaire_path = glob.glob(questionnaire_pattern, recursive=True)[0]  # Assumes at least one match
		questionnaire_df = pd.read_csv(questionnaire_path)
		
		pattern = os.path.join(base, "**", "scenario_*.csv")
		frames = [
			pd.merge(questionnaire_df, pd.read_csv(f), on="Prolific ID", how="right") 
			for f in glob.glob(pattern, recursive=True)
		]
		if not frames:
			raise FileNotFoundError("No scenario_*.csv files found.")
		all_data = pd.concat(frames, ignore_index=True)
		all_data.to_csv(os.path.join(output_dir,"all_data_combined.csv"), index=False)
		return all_data
	finally:
		if tmp_dir and os.path.isdir(tmp_dir):
			shutil.rmtree(tmp_dir)

def label_reliance(r):
	if r["User error"]:
		if r["Expected answer"] == "Reject":
			return "Over-reliance"
		elif r["Expected answer"] == "Accept":
			return "Under-reliance"
	else: # if not r["User error"]:
		if r["Expected answer"] == "Reject":
			return "Appropriate reject"
		elif r["Expected answer"] == "Accept":
			return "Appropriate accept"
		# return "Appropriate"
	return "Other"

def tidy_task(val,_format):
	# print(val)
	v = str(val).split("_")[0].replace('task','scenario ').capitalize()
	t = scenario_name_map.get(v,v)
	return t.get(_format,v) if isinstance(t, dict) else t

def filter_invalid_rows(df, _input):
	df = df.copy()
	# Keep only valid Prolific IDs
	df = df[df["Prolific ID"].str.len() >= 23]
	# Keep only valid changes of mind
	df = df[~df["What made you change your decision?"].str.contains(r"(n't| not) change", case=False, na=False)]
	# For any rows sharing both the same Prolific ID and the same Scenario, keep only the last occurrence.
	df["Scenario"] = df.apply(lambda row: tidy_task(row["Task file"], _input.split('/')[-1]), axis=1)
	df["Reliance category"] = df.apply(label_reliance, axis=1)
	df = df.drop_duplicates(subset=["Prolific ID", "Task file", "Format"], keep="first")
	# # Keep only those IDs that appear in 4 scenarios
	# df = df[df.groupby("Prolific ID")["Scenario"].transform("nunique").ge(1)]
	return df

# Apply filtering row-by-row
def within_quantiles(row, min_seconds_per_scenario=None, max_seconds_per_scenario=None):
	scenario = row["Scenario"]
	seconds = row["Seconds"]
	min_sec = min_seconds_per_scenario.get(scenario, None) if isinstance(min_seconds_per_scenario, dict) else min_seconds_per_scenario
	max_sec = max_seconds_per_scenario.get(scenario, None) if isinstance(max_seconds_per_scenario, dict) else max_seconds_per_scenario
	if not min_seconds_per_scenario:
		min_sec = float('-inf')
	if not max_seconds_per_scenario:
		max_sec = float('inf')
	return min_sec <= seconds <= max_sec

def analyse(df, min_seconds=None, max_seconds=None, keep_only_who_changed_decision=True, do_balance_treatments=False, keep_only_who_easily_understood_explanation=False, expected_answer=None):
	df = df.copy()
	# Keep only who spent enough time
	
	old_len = len(df)
	df = df[df.apply(lambda x: within_quantiles(x, min_seconds, max_seconds), axis=1)]
	if old_len-len(df):
		print(f'<analyse::time_filter> Dropped entries: {old_len-len(df)}/{old_len} ({100*(old_len-len(df))/old_len:.2f}%) entries were removed because produced in LESS than {min_seconds} and MORE than {max_seconds}')
	if expected_answer:
		df = df[df["Expected answer"] == expected_answer]

	# df = df[(df["How much do you trust AI systems in general?"] <= 3)]
	# df = df[(df["How would you rate your overall attitude toward Artificial Intelligence (AI)?"] <= 3)]
	# df = df[(df['How familiar are you with Artificial Intelligence (AI)?'] >= 1)]

	if keep_only_who_easily_understood_explanation:
		old_len = len(df)
		df = df[(df["How easy was it to understand the explanation?"] >= 3)] # Keep only who easily understood the explanation
		if old_len-len(df):
			print(f'<analyse::ease_filter> Dropped entries: {old_len-len(df)}/{old_len} ({100*(old_len-len(df))/old_len:.2f}%)')

	if keep_only_who_changed_decision: # Keep only who actually updated their decision after receiving the explanation
		old_len = len(df)
		df = df[(df["Explanation changed mind"] == True)]
		if old_len-len(df):
			print(f'<analyse::changed_mind_only> Dropped entries: {old_len-len(df)}/{old_len} ({100*(old_len-len(df))/old_len:.2f}%)')
	else: # Keep only who actually used the explanations, updating their mental model
		old_len = len(df)
		df = df[
			(
				(df["Explanation changed mind"] == True) |
				(
					(df["How confident are you in the decision you made? (without explanation)"] != df["How confident are you in the decision you made? (with explanation)"]) &
					(df["How useful was the explanation provided?"] >= 1) &
					(df["Did the explanation help you evaluate the AI's output?"] >= 1)
				)
			)
		]
		if old_len-len(df):
			print(f'<analyse::measurable_effect_filter> Dropped entries: {old_len-len(df)}/{old_len} ({100*(old_len-len(df))/old_len:.2f}%)')

	# df = df[(df["How confident are you in the decision you made? (without explanation)"] < df["How confident are you in the decision you made? (with explanation)"])]
	# df = df[df["Explanation changed mind"]]
	
	# df = df[df["How easy was it to understand the explanation?"] > 3]
	
	if do_balance_treatments:
		df = balance_treatments(df)

	counts = (df.groupby(["Explanation is MAGIX-defined","Reliance category"])
				.size().unstack(fill_value=0)
				.reindex(RELIANCE_ORDER, axis=1).sort_index(axis=0))
	chi2, p, dof, _ = chi2_contingency(counts.values)
	print(f"Overall χ²={chi2:.3f}, dof={dof}, p={p:.4f}  ({max_seconds} ≥ Seconds ≥ {min_seconds})")
	return df, counts

def fmt(x, digits=2):
	return 'n/a' if pd.isna(x) else f"{x:.{digits}f}"

def star(p):
	if pd.isna(p): return ''
	return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

def plot_reliance_counts(counts, out_dir, min_seconds=None, max_seconds=None, keep_only_who_changed_decision=False, do_balance_treatments=False, keep_only_who_easily_understood_explanation=False):
	ax = counts.plot(kind="bar", figsize=(9,5))
	ax.set_title("Reliance counts by explanation type")
	ax.set_ylabel("Number of judgements")
	ax.set_xlabel("Explanation is MAGIX-defined")
	ax.yaxis.set_major_locator(MaxNLocator(integer=True))
	# Hatches by category
	for container, cat in zip(ax.containers, counts.columns.tolist()):
		for bar in container:
			bar.set_hatch(HATCHES.get(cat, ""))
	annotate_bars(ax, fmt="{:.0f}")
	leg = ax.legend(title="Reliance category", ncols=3, frameon=True)
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, f"reliance_counts-s={min_seconds}_{max_seconds}{'-explanation_clarity' if keep_only_who_easily_understood_explanation else ''}{'-balanced' if do_balance_treatments else ''}{'-changed_decision' if keep_only_who_changed_decision else ''}.pdf"))
	if SHOW_ALL_FIGURES: plt.show()

def plot_reliance_proportions(counts, out_dir, min_seconds=None, max_seconds=None, keep_only_who_changed_decision=True, do_balance_treatments=False, keep_only_who_easily_understood_explanation=False):
	props = counts.div(counts.sum(axis=1), axis=0)
	ax = props.plot(kind="bar", figsize=(9,5))
	ax.set_title("Reliance proportions by explanation type")
	ax.set_ylabel("Proportion")
	ax.set_xlabel("Explanation is MAGIX-defined")
	ax.yaxis.set_major_formatter(PercentFormatter(1.0))
	# Hatches by category
	for container, cat in zip(ax.containers, props.columns.tolist()):
		for bar in container:
			bar.set_hatch(HATCHES.get(cat, ""))
	annotate_bars(ax, fmt="{:.0f}%", y_is_pct=True)
	ax.legend(title="Reliance category", ncols=3, frameon=True)
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, f"reliance_props-s={min_seconds}_{max_seconds}{'-explanation_clarity' if keep_only_who_easily_understood_explanation else ''}{'-balanced' if do_balance_treatments else ''}{'-changed_decision' if keep_only_who_changed_decision else ''}.pdf"))
	if SHOW_ALL_FIGURES: plt.show()

def plot_changes(df, out_dir, min_seconds=None, max_seconds=None, keep_only_who_changed_decision=False, do_balance_treatments=False, keep_only_who_easily_understood_explanation=False):
	df = df.copy()
	df["Change type"] = df.apply(
		lambda r: f"{r['Response before explanation']}→{r['Response after explanation']}" if r["Explanation changed mind"] else "No change",
		axis=1,
	)
	# Sorted order if present
	order = ["Accept→Reject", "No change", "Reject→Accept"]
	ch = (df.groupby(["Explanation is MAGIX-defined","Change type"]).size().unstack(fill_value=0))
	ch = ch.reindex(columns=[c for c in order if c in ch.columns], fill_value=0)
	ax = ch.plot(kind="bar", figsize=(9,5))
	ax.set_title("Response‑change patterns by explanation type")
	ax.set_ylabel("Number of judgements")
	ax.set_xlabel("Explanation is MAGIX-defined")
	ax.yaxis.set_major_locator(MaxNLocator(integer=True))
	# Hatches per change-type to avoid relying on color
	for container, col in zip(ax.containers, ch.columns.tolist()):
		hatch = {"Accept→Reject":"//","No change":"","Reject→Accept":"xx"}.get(col, "")
		for bar in container:
			bar.set_hatch(hatch)
	annotate_bars(ax, fmt="{:.0f}")
	ax.legend(title="Change type", ncols=3, frameon=True)
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, f"response_changes-s={min_seconds}_{max_seconds}{'-explanation_clarity' if keep_only_who_easily_understood_explanation else ''}{'-balanced' if do_balance_treatments else ''}{'-changed_decision' if keep_only_who_changed_decision else ''}.pdf"))
	if SHOW_ALL_FIGURES: plt.show()

def plot_per_scenario_multi(df, out_dir, min_seconds=None, max_seconds=None, keep_only_who_changed_decision=False, do_balance_treatments=False, keep_only_who_easily_understood_explanation=False):
	"""
	Create a 1x3 subplot figure showing per-scenario reliance composition for:
	- All experiments (expected_answer=None)
	- Only Accept-correct (expected_answer='Accept')
	- Only Reject-incorrect (expected_answer='Reject')
	"""
	# Define the three analyses
	results = []
	scenarios = sorted(df["Scenario"].unique())
	expl = df["Explanation is MAGIX-defined"].unique()
	reliance = df["Reliance category"].unique()
	for label, expected in [("All", None), ("Accept", "Accept"), ("Reject", "Reject")]:
		df_sub, counts = analyse(df, 
			min_seconds=min_seconds,
			max_seconds=max_seconds, 
			keep_only_who_changed_decision=keep_only_who_changed_decision, 
			do_balance_treatments=do_balance_treatments, 
			keep_only_who_easily_understood_explanation=keep_only_who_easily_understood_explanation,
			expected_answer=expected
		)
		if label == 'All':
			df_sub = balance_treatments(df_sub)
		# all combinations of grouping keys
		full_index = pd.MultiIndex.from_product(
			[scenarios, expl, reliance],
			names=["Scenario", "Explanation is MAGIX-defined", "Reliance category"]
		)
		# count occurrences
		base = (
			df_sub.groupby(["Scenario", "Explanation is MAGIX-defined", "Reliance category"])
			.size()
			.rename("n")
			.reindex(full_index, fill_value=0)   # fill missing with 0
			.reset_index()
		)
		# compute totals per scenario × explanation
		totals = base.groupby(["Scenario", "Explanation is MAGIX-defined"])["n"].transform("sum")

		base["prop"] = base["n"] / totals
		results.append((label, base, df_sub))

	plt.rcParams.update({
		# … other params …
		"axes.grid": False,          # turn off all grids
		# "grid.alpha": 0.3,         # no longer needed
	})

	x = np.arange(len(scenarios))
	width = 0.3
	width_per_scenario = 2.5
	fig, axes = plt.subplots(1, 3, figsize=(len(scenarios) * width_per_scenario, 4), sharey=True)

	expl_colors = {False: 'C0', True: 'C1'}
	score_map = {"Under-reliance": -1, "Appropriate accept": 1, "Appropriate reject": 1, "Over-reliance": -1}

	for ax, (label, base, df_sub) in zip(axes, results):
		for idx, expl in enumerate([False, True]):
			subset = base[base['Explanation is MAGIX-defined'] == expl]
			pivot = subset.pivot(index='Scenario', columns='Reliance category', values='prop') \
							 .reindex(scenarios, fill_value=0)
			bottom = np.zeros(len(scenarios))
			for cat in RELIANCE_ORDER:
				vals = pivot.get(cat, np.zeros(len(scenarios)))
				bars = ax.bar(
					x + (idx - 0.5)*width,
					vals,
					width,
					bottom=bottom,
					color=expl_colors[expl] if 'Appropriate' in cat else (*mcolors.to_rgb(expl_colors[expl]), 0.1),
					edgecolor='black',
					hatch=HATCHES.get(cat, '')
				)
				# annotate counts and percentages
				# compute raw counts
				count_pivot = subset.pivot(index='Scenario', columns='Reliance category', values='n')\
									 .reindex(scenarios, fill_value=0)
				for i, v in enumerate(vals):
					if v > 0.02:
						c = int(count_pivot.loc[scenarios[i], cat])
						if c > 1:
							ax.annotate(
								f"{int(round(v*100)):.0f}%\n({c})",
								(x[i] + (idx - 0.5)*width, bottom[i] + (v/2 if v > 0.05 else 0)),
								xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=7,
								bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9)
							)
				bottom += vals
		# p-value annotation via Mann–Whitney U
		p_vals = {}
		_tt_power = TTestIndPower()
		alpha = 0.05  # or whatever you're using
		for scen in scenarios:
			sub = df_sub[df_sub['Scenario'] == scen]
			scores_non = sub[sub['Explanation is MAGIX-defined'] == False]['Reliance category'].map(score_map)
			scores_mag = sub[sub['Explanation is MAGIX-defined'] == True]['Reliance category'].map(score_map)
			if len(scores_non)>0 and len(scores_mag)>0:
				# Mann-Whitney U test
				U, p = mannwhitneyu(scores_non, scores_mag, alternative='greater' if np.mean(scores_non) > np.mean(scores_mag) else 'less')
				# # Cliff's delta effect size: (2*U)/(n1*n2) - 1
				# n1, n2 = len(scores_non), len(scores_mag)
				# delta = (2 * U) / (n1 * n2) - 1
				# Cohen's d
				diff = np.mean(scores_mag) - np.mean(scores_non)
				pooled_var = ((scores_non.var(ddof=1) + scores_mag.var(ddof=1)) / 2)
				d = diff / np.sqrt(pooled_var) if pooled_var > 0 else np.nan
				# ---- post-hoc power (approx via two-sample t-test) ----
				n_non, n_mag = len(scores_non), len(scores_mag)
				power = np.nan
				if np.isfinite(d) and n_non > 1 and n_mag > 1:
					if _tt_power is not None:
						# match direction: d = mean_mag - mean_non
						ratio = n_non / n_mag
						alt = 'larger' if d >= 0 else 'smaller'
						power = _tt_power.power(
							effect_size=float(d),  # sign matters with 'larger'/'smaller'
							nobs1=n_mag,           # group1 = MAGIX group (to match d)
							ratio=ratio,
							alpha=alpha,
							alternative=alt
						)
			else:
				p = np.nan
				# delta = np.nan
				d = np.nan
				# w = np.nan
				power = np.nan
			p_vals[scen] = (p,d, power)
		for i, scen in enumerate(scenarios):
			p_val, d_val, pw_val = p_vals[scen]

			# Bold markers
			if p_val < 0.01:
				p_str = r"$\mathbf{p\!<\!0.01}$"
			else:
				# format with 3 decimals and drop leading "0"
				p_fmt = f"{p_val:.3f}"
				if p_val < 0.05:
					p_str = rf"$\mathbf{{p\!=\!{p_fmt}}}$"
				else:
					p_str = f"p={p_fmt}"
			pw_str = rf"$\mathbf{{pw\!=\!{pw_val:.2f}}}$" if pw_val >= 0.795 else f"pw={pw_val:.2f}"
			d_str = f"(d={d_val:.2f})"

			# Combine with newlines
			text_str = f"{p_str}\n{pw_str}\n{d_str}".replace(' ','')

			ax.text(
				x[i], 1.1, text_str,
				ha='center', va='bottom', fontsize=8
			)

		ax.set_xticks(x)
		ax.set_xticklabels(list(map(lambda x: x.replace('Scenario ','S'), scenarios)), rotation=0, ha='center', fontsize=9)
		if label == 'All':
			label += ' (balanced within-scenario)'
		ax.set_title(f"Expected: {label}", fontsize=9)
		if ax is axes[0]:
			ax.set_ylabel('Proportion within explanation type', fontsize=9)
			ax.yaxis.set_major_formatter(PercentFormatter(1.0))
		ax.set_ylim(0, 1.3)
		# ax.yaxis.set_major_locator(MaxNLocator(5))
		ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0]) # Force ticks only up to 1.0
		ax.tick_params(axis='y', labelsize=9)

	# Figure-level legend for Explanation type on top-left
	type_handles = [
		Patch(facecolor=expl_colors[False], edgecolor='black', label='Non-MAGIX'),
		Patch(facecolor=expl_colors[True], edgecolor='black', label='MAGIX')
	]
	fig.legend(
		handles=type_handles,
		title='Explanation type',
		loc='upper left',
		bbox_to_anchor=(0.1, 1),
		ncol=len(type_handles),
		frameon=True,
		fontsize=8,           # label font size
		title_fontsize=8     # title font size
	)

	# Figure-level legend for Reliance categories at top center
	cat_handles = [
		Patch(facecolor='white', edgecolor='black', hatch=HATCHES[c], label=c) for c in RELIANCE_ORDER
	]
	fig.legend(
		handles=cat_handles,
		title='Reliance category',
		loc='upper center',
		bbox_to_anchor=(0.6, 1),
		ncol=len(RELIANCE_ORDER),
		frameon=True,
		fontsize=8,           # label font size
		title_fontsize=10     # title font size
	)

	plt.tight_layout(rect=[0, 0, 1, 0.9])
	plt.savefig(os.path.join(out_dir, f"per_scenario_reliance_props_multi-s={min_seconds}_{max_seconds}{'-explanation_clarity' if keep_only_who_easily_understood_explanation else ''}{'-balanced' if do_balance_treatments else ''}{'-changed_decision' if keep_only_who_changed_decision else ''}.pdf"))
	if SHOW_ALL_FIGURES: plt.show()

def plot_corrections(df, output_dir, min_seconds=None, max_seconds=None):
	df = df.copy()
	# Time filter only; keep full range of 'ease' values
	# df = df[df["Seconds"] >= seconds]

	df = df[df.apply(lambda x: within_quantiles(x, min_seconds, max_seconds), axis=1)]
	# # Ensure reliance labels exist
	# if "Reliance category" not in df.columns:
	# 	df["Reliance category"] = df.apply(label_reliance, axis=1)

	# 1) filter to only “corrections” in the two appropriate categories
	corr = df[
		(df["Explanation changed mind"] == True) &
		(df["Reliance category"].isin(["Appropriate reject", "Appropriate accept"]))
	].copy()
	# 2) label the direction of the correction
	corr["Correction type"] = corr["Reliance category"].map({
		"Appropriate reject": "Accept → Reject (on AI Incorrect)",
		"Appropriate accept": "Reject → Accept (on AI Correct)"
	})
	# 3) count by scenario, MAGIX-flag, and correction type
	counts = (
		corr
		.groupby(["Scenario", "Explanation is MAGIX-defined", "Correction type"])
		.size()
		.reset_index(name="Count")
	)
	# 4) pivot so we can plot side by side
	pivot = counts.pivot_table(
		index="Scenario",
		columns=["Correction type", "Explanation is MAGIX-defined"],
		values="Count",
		fill_value=0
	)
	# 5) plot
	fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
	for ax, ctype in zip(axes, ["Accept → Reject (on AI Incorrect)", "Reject → Accept (on AI Correct)"]):
		# get two bars per scenario: MAGIX True vs False
		data = pivot[ctype]
		data.plot(
			kind="bar",
			ax=ax,
			width=0.8,
			legend=True,
			title=ctype,
			ylabel="Number of corrections" if ax is axes[0] else ""
		)
		ax.set_xlabel("")
		ax.tick_params(axis='x', labelrotation=0)
		ax.yaxis.set_major_locator(MaxNLocator(integer=True))
		ax.ticklabel_format(axis='y', style='plain')  # avoid scientific notation
		# annotate bars
		for container in ax.containers:
			for bar in container:
				h = bar.get_height()
				if h > 0:
					ax.annotate(f"{h:.0f}", 
								(bar.get_x() + bar.get_width() / 2, h),
								textcoords="offset points", xytext=(0,3),
								ha="center", va="bottom", fontsize=9)
	fig.tight_layout()
	# save to output directory
	plt.savefig(os.path.join(output_dir, f"corrections_by_scenario-s={min_seconds}_{max_seconds}.pdf"))
	if SHOW_ALL_FIGURES: plt.show()

def plot_mitigation_by_driver(df, out_dir, min_seconds=None, max_seconds=None, keep_only_who_changed_decision=False, do_balance_treatments=False):
	d = df.copy()

	# Time filter only; keep full range of driver values
	d = d[d.apply(lambda x: within_quantiles(x, min_seconds, max_seconds), axis=1)]

	# Ensure reliance labels exist
	if "Reliance category" not in d.columns:
		d["Reliance category"] = d.apply(label_reliance, axis=1)

	if keep_only_who_changed_decision: # Keep only who actually updated their decision after receiving the explanation
		old_len = len(d)
		d = d[(d["Explanation changed mind"] == True)]
		if old_len-len(d):
			print(f'<analyse::changed_mind_only> Dropped entries: {old_len-len(d)}/{old_len} ({100*(old_len-len(d))/old_len:.2f}%)')
	else: # Keep only who actually used the explanations, updating their mental model
		old_len = len(d)
		d = d[
			(
				(d["Explanation changed mind"] == True) |
				(
					(d["How confident are you in the decision you made? (without explanation)"] != d["How confident are you in the decision you made? (with explanation)"]) &
					(d["How useful was the explanation provided?"] >= 1) &
					(d["Did the explanation help you evaluate the AI's output?"] >= 1)
				)
			)
		]
		if old_len-len(d):
			print(f'<analyse::measurable_effect_filter> Dropped entries: {old_len-len(d)}/{old_len} ({100*(old_len-len(d))/old_len:.2f}%)')

	d = balance_treatments(d)

	# ---------------------------
	# Drivers to plot (column, pretty x-label, stub)
	# ---------------------------
	drivers = [
		("How easy was it to understand the explanation?", "Expl. Clarity", "Ease"),
		("How confident are you in the decision you made? (with explanation)", "Confidence after Expl.", "Confidence_after"),
		# ("How confident are you in the decision you made? (without explanation)", "Confidence before Expl.", "Confidence_before"),
		("Did the explanation help you evaluate the AI's output?", "Expl. Helpfulness", "Helpfulness"),
		("How useful was the explanation provided?", "Expl. Usefulness", "Usefulness"),
		("How much effort did it take to understand and complete this task?", "Effort", "Effort"),
	]

	# Coerce any Likert-like column to 1–5 integers; if it's 0–4 shift to 1–5
	def to_one_to_five(series):
		s = pd.to_numeric(series, errors="coerce")
		return (s + 1).round().astype("Int64")

	# Helper: compute mitigation tables/rates
	def mitigation_series(data, expected, appropriate_label, error_label):
		sub = data[data["Expected answer"] == expected]
		tab = (
			sub.groupby(["Explanation is MAGIX-defined", "Scale", "Reliance category"])
			   .size()
			   .unstack(fill_value=0)
		)
		for col in (appropriate_label, error_label):
			if col not in tab.columns:
				tab[col] = 0
		# den = tab[appropriate_label] + tab[error_label]
		# rate = (tab[appropriate_label] / den).replace([np.inf, np.nan], np.nan)
		rate = tab[appropriate_label].replace([np.inf, np.nan], np.nan)
		return rate, tab  # also return counts

	# Keep only drivers that actually have data
	valid_drivers = []
	for col, x_label, stub in drivers:
		if col in d.columns and d[col].notna().any():
			valid_drivers.append((col, x_label, stub))
	if not valid_drivers:
		raise ValueError("None of the specified driver columns contain data to plot.")

	ncols = len(valid_drivers)
	fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(3.4 * ncols, 3.7), sharex=True, sharey=True)
	axes = np.atleast_2d(axes)

	_norm_power = NormalIndPower()
	alpha = 0.05  # or whatever you're using
	for c, (col, x_label, stub) in enumerate(valid_drivers):
		d_sub = d[pd.notna(d[col])].copy()
		d_sub["Scale"] = to_one_to_five(d_sub[col])
		d_sub = d_sub[pd.notna(d_sub["Scale"])]

		# Build rates and counts
		over_rate, over_counts = mitigation_series(
			d_sub, expected="Reject",
			appropriate_label="Appropriate reject",
			error_label="Over-reliance"
		)
		under_rate, under_counts = mitigation_series(
			d_sub, expected="Accept",
			appropriate_label="Appropriate accept",
			error_label="Under-reliance"
		)

		levels = sorted([int(v) for v in d_sub["Scale"].dropna().unique()])
		panels = [
			(axes[0, c], over_rate, over_counts, "Appropriate reject", "Over-reliance"),
			(axes[1, c], under_rate, under_counts, "Appropriate accept", "Under-reliance"),
		]

		for ax, rate, counts, appr_label, err_label in panels:
			# Compute per-level p-values
			p_values = {}
			for e in levels:
				try:
					cnt0 = counts.loc[(False, e)][appr_label]
					tot0 = counts.loc[(False, e)][appr_label] + counts.loc[(False, e)][err_label]
				except KeyError:
					cnt0, tot0 = 0, 0
				try:
					cnt1 = counts.loc[(True, e)][appr_label]
					tot1 = counts.loc[(True, e)][appr_label] + counts.loc[(True, e)][err_label]
				except KeyError:
					cnt1, tot1 = 0, 0
				if tot0 > 0 and tot1 > 0:
					_, pval = proportions_ztest([cnt0, cnt1], [tot0, tot1])
					effect = sms.proportion_effectsize(cnt1 / tot1, cnt0 / tot0)
					# Total sample size
					alpha = 0.05
					power = _norm_power.power(effect_size=effect, nobs1=tot0, alpha=alpha, ratio=tot1/tot0)
				else:
					pval = np.nan
					effect = np.nan
					power = np.nan
				p_values[e] = (pval, power, effect)

			# Plot lines and annotate counts and p-values
			y_values = {}
			# Keep track of previous annotation positions at each x
			annot_positions = {}
			for magix_flag, label, marker in [(False, "Non-MAGIX", "o"), (True, "MAGIX", "s")]:
				series = rate.loc[magix_flag] if (magix_flag in rate.index.get_level_values(0)) else pd.Series(dtype=float)
				y = [series.get(e, np.nan) for e in levels]
				y_values[magix_flag] = y
				ln, = ax.plot(levels, y, marker=marker, label=label)
				
				# N annotations
				for idx, e in enumerate(levels):
					N = 0
					if (magix_flag, e) in counts.index:
						N = int(counts.loc[(magix_flag, e)][appr_label] + counts.loc[(magix_flag, e)][err_label])
					yv = y[idx]
					if np.isfinite(yv):
						# Default offset
						y_offset = 4

						# If we already annotated something close at this x, move it higher
						if e in annot_positions:
							if 0 <= yv - annot_positions[e] < 0.05:
								y_offset += 10   # push higher if overlap
							elif -0.05 < yv - annot_positions[e] < 0:
								y_offset -= 10   # push higher if overlap

						# Store position
						annot_positions[e] = yv

						ax.annotate(
							f"N={N}", (e, yv), xytext=(0, y_offset), textcoords="offset points",
							ha="center", va="bottom", fontsize=7,
							color=ln.get_color(),   # match line color
							bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9)
						)

			# P-value annotations
			for e in levels:
				pval, power, effect = p_values.get(e, (np.nan, np.nan, np.nan))
				if np.isfinite(pval) and pval < 0.05:
					y0 = y_values.get(False, [np.nan] * len(levels))[levels.index(e)]
					y1 = y_values.get(True,  [np.nan] * len(levels))[levels.index(e)]
					y_max = max([yv for yv in (y0, y1) if np.isfinite(yv)] + [0])
					# Bold markers
					if pval < 0.01:
						p_str = r"$\mathbf{p\!<\!0.01}$"
					else:
						# format with 3 decimals and drop leading "0"
						p_fmt = f"{pval:.3f}"
						if pval < 0.05:
							p_str = rf"$\mathbf{{p\!=\!{p_fmt}}}$"
						else:
							p_str = f"p={p_fmt}"
					pw_str = rf"$\mathbf{{pw\!=\!{power:.2f}}}$" if power >= 0.795 else f"pw={power:.2f}"
					d_str = f"(h={effect:.2f})"

					# Combine with newlines
					text_str = f"{p_str}\n{pw_str}\n{d_str}".replace(' ','')
					ax.annotate(
						text_str, (e, y_max + 0.01),
						xytext=(0, 12), textcoords="offset points",
						ha="center", va="bottom", fontsize=8,
						bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9)
					)

			# ax.yaxis.set_major_formatter(PercentFormatter(1.0))
			# ax.set_ylim(0, 1.1)
			ax.xaxis.set_major_locator(MaxNLocator(integer=True))

		# Only bottom row shows x-axis labels
		# axes[0, c].set_xlabel(x_label)
		axes[1, c].set_xlabel(x_label)

		# Only left column shows y-axis label
		if c == 0:
			axes[0, 0].set_ylabel("Appropriate\nReject", rotation=90)
			axes[1, 0].set_ylabel("Appropriate\nAccept", rotation=90)

	# One shared legend (top-right outside the grid)
	axes[0,-1].legend(
		title="Explanation type",
		loc='upper right',
		frameon=True,
		fontsize=8,          # legend labels
		title_fontsize=8     # legend title
	)

	plt.tight_layout()  # leave room for the legend
	os.makedirs(out_dir, exist_ok=True)
	fname = (
		f"mitigation_by_ALL_DRIVERS-s={min_seconds}_{max_seconds}"
		f"{'-balanced' if do_balance_treatments else ''}"
		f"{'-changed_decision' if keep_only_who_changed_decision else ''}.pdf"
	)
	plt.savefig(os.path.join(out_dir, fname), bbox_inches='tight')
	if SHOW_ALL_FIGURES: plt.show()
	plt.close(fig)

def plot_reliance_vs_trust_attitude_effort(df, out_dir, min_seconds=None, max_seconds=None, annotate_n=True, show_stats=True, per_cell_w=3, per_cell_h=3, base_font=8):
	questions = {
		'Effort': 'How much effort did it take to understand and complete this task?',
		'Attitude': 'How would you rate your overall attitude toward Artificial Intelligence (AI)?',
		'Trust': 'How much do you trust AI systems in general?'
	}
	q_keys = list(questions.keys())

	d = df.copy()
	d = d[d.apply(lambda x: within_quantiles(x, min_seconds, max_seconds), axis=1)]
	
	if 'Reliance category' not in d.columns:
		d['Reliance category'] = d.apply(label_reliance, axis=1)

	d['is_over']  = ((d['Response before explanation'] == 'Accept') & (d['Expected answer'] == 'Reject')).astype(int)
	d['is_under'] = ((d['Response before explanation'] == 'Reject')  & (d['Expected answer'] == 'Accept')).astype(int)

	for key, col in questions.items():
		d[key] = pd.to_numeric(d[col], errors='coerce')
		d = d.dropna(subset=[key])
		d[key] = (d[key] + 1).astype(int)

	n_rows, n_cols = 1, len(q_keys)  # single row, all questions

	rc = {
		'font.size': base_font,
		'axes.titlesize': base_font,
		'axes.labelsize': base_font,
		'xtick.labelsize': max(base_font - 1, 6),
		'ytick.labelsize': max(base_font - 1, 6),
		'legend.fontsize': max(base_font - 1, 6),
		'axes.titlepad': 2.0,
		'axes.labelpad': 2.0
	}

	with plt.rc_context(rc):
		fig, axes = plt.subplots(
			n_rows, n_cols,
			figsize=(per_cell_w * n_cols, per_cell_h * n_rows),
			sharex=True, sharey=True, squeeze=False
		)

		color_non, color_mag = 'C0', 'C1'
		any_ax_with_lines = None

		for c, key in enumerate(q_keys):
			ax = axes[0, c]
			ds_valid = d.dropna(subset=[key])
			if ds_valid.empty:
				ax.set_title(f"Reliance vs {key}")
				ax.set_ylabel("Proportion")
				ax.yaxis.set_major_formatter(PercentFormatter(1.0))
				ax.set_xlabel(key)
				ax.grid(alpha=0.2, linewidth=0.4)
				ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
				ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
				ax.margins(x=0.05, y=0.05)
				continue

			grp = ds_valid.groupby([key])
			rates = grp.agg(
				over_rate=('is_over', 'mean'),
				under_rate=('is_under', 'mean'),
				n_over=('is_over', 'sum'),
				n_under=('is_under', 'sum')
			).reset_index()

			subset = rates.sort_values(key)
			if subset.empty: continue
			xvals = subset[key].astype(float)
			ax.plot(xvals, subset['over_rate'], linestyle='-', marker='o', markersize=3, color=color_mag,
					label=f'Over-reliance', linewidth=0.9)
			ax.plot(xvals, subset['under_rate'], linestyle='--', marker='x', markersize=3, color=color_non,
					label=f'Under-reliance', linewidth=0.9)

			if annotate_n:
				ann_fs = max(base_font - 2, 6)
				for _, row in subset.iterrows():
					x = float(row[key])
					ax.annotate(f"{int(row['n_over'])}",
								(x, row['over_rate']),
								xytext=(0, 5), textcoords='offset points',
								ha='center', va='top',
								fontsize=ann_fs, color=color_mag, bbox=dict(facecolor='white', alpha=0.8, pad=1, edgecolor='none'))
					ax.annotate(f"{int(row['n_under'])}",
								(x, row['under_rate']),
								xytext=(0, -5), textcoords='offset points',
								ha='center', va='bottom',
								fontsize=ann_fs, color=color_non, bbox=dict(facecolor='white', alpha=0.8, pad=1, edgecolor='none'))

			ax.set_title(f"Reliance vs {key}")
			if c == 0:
				ax.set_ylabel("Proportion")
				ax.yaxis.set_major_formatter(PercentFormatter(1.0))

			ax.set_xlabel(key)
			ax.grid(alpha=0.2, linewidth=0.4)
			ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
			ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
			ax.margins(x=0.05, y=0.05)

			if show_stats:
				sub = ds_valid
				rho_o, p_o = spearmanr(sub[key], sub['is_over'], nan_policy='omit')
				rho_u, p_u = spearmanr(sub[key], sub['is_under'], nan_policy='omit')
				stats = {
					'rho_over': rho_o, 'p_over': p_o,
					'rho_under': rho_u, 'p_under': p_u
				}

				proxy = Line2D([], [], linestyle='')
				handles = [proxy, proxy]
				labels = []
				s = stats
				labels.append(
					f"$\\rho_{{\\mathrm{{over}}}}$={fmt(s['rho_over'])}{star(s['p_over'])} (p={fmt(s['p_over'], 2)})\n"
					f"$\\rho_{{\\mathrm{{under}}}}$={fmt(s['rho_under'])}{star(s['p_under'])} (p={fmt(s['p_under'], 2)})"
				)

				leg_stats = ax.legend(
					handles, labels,
					loc='upper left',
					bbox_to_anchor=(0.01, 0.99),
					borderaxespad=0.2,
					handlelength=0,
					handletextpad=0,
					frameon=True, framealpha=0.33, fancybox=True,
					fontsize=6
				)
				ax.add_artist(leg_stats)

			if any_ax_with_lines is None and ax.lines:
				any_ax_with_lines = ax

		if any_ax_with_lines is not None:
			handles, labels = any_ax_with_lines.get_legend_handles_labels()
			fig.legend(handles, labels, loc='upper center', ncol=4, frameon=True)

		fig.tight_layout(rect=(0, 0, 1, 0.92))
		os.makedirs(out_dir, exist_ok=True)
		out_path = os.path.join(out_dir, f"reliance_vs_trust_attitude_effort-s={min_seconds}_{max_seconds}.pdf")
		plt.savefig(out_path, bbox_inches='tight')
		if SHOW_ALL_FIGURES: plt.show()
		print(f"Saved figure to: {out_path}")

def plot_reliance_vs_trust_attitude_effort_by_scenario(df, out_dir, min_seconds=None, max_seconds=None, keep_only_who_changed_decision=False, do_balance_treatments=False, keep_only_who_easily_understood_explanation=False, annotate_n=True, show_stats=True, per_cell_w=3, per_cell_h=2.1, base_font=8):
	def natural_key(s):
		s = str(s)
		return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

	questions = {
		'Effort': 'How much effort did it take to understand and complete this task?',
		'Attitude': 'How would you rate your overall attitude toward Artificial Intelligence (AI)?',
		'Trust': 'How much do you trust AI systems in general?'
	}
	q_keys = list(questions.keys())

	d = df.copy()
	d = d[d.apply(lambda x: within_quantiles(x, min_seconds, max_seconds), axis=1)]

	if 'Reliance category' not in d.columns:
		d['Reliance category'] = d.apply(label_reliance, axis=1)

	d['is_over']  = ((d['Response before explanation'] == 'Accept') & (d['Expected answer'] == 'Reject')).astype(int)
	d['is_under'] = ((d['Response before explanation'] == 'Reject')  & (d['Expected answer'] == 'Accept')).astype(int)

	# Convert each question to numeric scale starting at 1
	for key, col in questions.items():
		d[key] = pd.to_numeric(d[col], errors='coerce')
		d = d.dropna(subset=[key, "Scenario"])
		d[key] = (d[key] + 1).astype(int)

	scenarios = list(pd.Series(d['Scenario'].dropna()).unique())
	if len(scenarios) == 0:
		print("No scenarios found after filtering. Nothing to plot.")
		return
	scenarios = sorted(scenarios, key=natural_key)

	n_rows, n_cols = len(scenarios), len(q_keys)

	rc = {
		'font.size': base_font,
		'axes.titlesize': base_font,
		'axes.labelsize': base_font,
		'xtick.labelsize': max(base_font - 1, 6),
		'ytick.labelsize': max(base_font - 1, 6),
		'legend.fontsize': max(base_font - 1, 6),
		'axes.titlepad': 2.0,
		'axes.labelpad': 2.0
	}

	with plt.rc_context(rc):
		fig, axes = plt.subplots(
			n_rows, n_cols,
			figsize=(per_cell_w * n_cols, per_cell_h * n_rows),
			sharex=True, sharey=True, squeeze=False
		)

		color_non, color_mag = 'C0', 'C1'
		any_ax_with_lines = None

		for r, sc in enumerate(scenarios):
			d_s = d[d['Scenario'] == sc]
			for c, key in enumerate(q_keys):
				ax = axes[r, c]

				ds_valid = d_s.dropna(subset=[key])
				if ds_valid.empty:
					if r == 0: ax.set_title(f"Reliance vs {key}")
					if c == 0:
						ax.set_ylabel(f"{sc}\nProportion")
						ax.yaxis.set_major_formatter(PercentFormatter(1.0))
					ax.set_xlabel(key if r == n_rows - 1 else '')
					ax.tick_params(axis='x', labelbottom=(r == n_rows - 1))
					ax.grid(alpha=0.2, linewidth=0.4)
					ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
					ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
					ax.margins(x=0.05, y=0.05)
					continue

				grp = ds_valid.groupby(['Explanation is MAGIX-defined', key])
				rates = grp.agg(
					over_rate=('is_over', 'mean'),
					under_rate=('is_under', 'mean'),
					n_over=('is_over', 'sum'),     # count of over-reliance
					n_under=('is_under', 'sum')    # count of under-reliance
				).reset_index()

				for magix_flag, color, label_prefix in [
					(False, color_non, 'Non-MAGIX'),
					(True,  color_mag, 'MAGIX')
				]:
					subset = rates[rates['Explanation is MAGIX-defined'] == magix_flag].sort_values(key)
					if subset.empty: continue
					xvals = subset[key].astype(float)
					ax.plot(xvals, subset['over_rate'],  linestyle='-',  marker='o', markersize=3, color=color,
							label=f'Over {label_prefix}', linewidth=0.9)
					ax.plot(xvals, subset['under_rate'], linestyle='--', marker='x', markersize=3, color=color,
							label=f'Under {label_prefix}', linewidth=0.9)

					if annotate_n:
						ann_fs = max(base_font - 2, 6)
						# annotate Over line with the OVER count
						for _, row in subset.iterrows():
							x = float(row[key])
							ax.annotate(f"{int(row['n_over'])}",
										(x, row['over_rate']),
										xytext=(0, 5), textcoords='offset points',
										ha='center', va='top',
										fontsize=ann_fs, color=color, bbox=dict(facecolor='white', alpha=0.8, pad=1, edgecolor='none'))

						# annotate Under line with the UNDER count
						for _, row in subset.iterrows():
							x = float(row[key])
							ax.annotate(f"{int(row['n_under'])}",
										(x, row['under_rate']),
										xytext=(0, -5), textcoords='offset points',
										ha='center', va='bottom',
										fontsize=ann_fs, color=color, bbox=dict(facecolor='white', alpha=0.8, pad=1, edgecolor='none'))

				if r == 0:
					ax.set_title(f"Reliance vs {key}")
				if c == 0:
					ax.set_ylabel(f"{sc}\nProportion")
					ax.yaxis.set_major_formatter(PercentFormatter(1.0))

				ax.set_xlabel(key if r == n_rows - 1 else '')
				ax.tick_params(axis='x', labelbottom=(r == n_rows - 1))

				ax.grid(alpha=0.2, linewidth=0.4)
				ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
				ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
				ax.margins(x=0.05, y=0.05)

				# --- Stats as a per-axes legend (replaces the text box) ---
				if show_stats:
					stats = {}
					for flag, label in [(False, 'Non-MAGIX'), (True, 'MAGIX')]:
						# use the same NA-filtered data as the lines
						sub = ds_valid[ds_valid['Explanation is MAGIX-defined'] == flag]

						rho_o, p_o = spearmanr(sub[key], sub['is_over'], nan_policy='omit')
						rho_u, p_u = spearmanr(sub[key], sub['is_under'], nan_policy='omit')
						
						stats[label] = {
							'rho_over': rho_o, 'p_over': p_o,
							'rho_under': rho_u, 'p_under': p_u
						}

					# Build text-only legend entries using invisible proxy handles
					proxy = Line2D([], [], linestyle='')  # invisible
					handles = [proxy, proxy]
					labels = []
					for label in ['Non-MAGIX', 'MAGIX']:
						s = stats[label]
						labels.append(
							f"{label}:\n"
							f"\t$\\rho_{{\\mathrm{{over}}}}$={fmt(s['rho_over'])}{star(s['p_over'])} (p={fmt(s['p_over'], 2)})\n"
							f"\t$\\rho_{{\\mathrm{{under}}}}$={fmt(s['rho_under'])}{star(s['p_under'])} (p={fmt(s['p_under'], 2)})"
						)

					leg_stats = ax.legend(
						handles, labels,
						loc='upper left',
						bbox_to_anchor=(0.01, 0.99),
						borderaxespad=0.2,
						handlelength=0,
						handletextpad=0,
						frameon=True, framealpha=0.33, fancybox=True,
						fontsize=max(base_font - 2, 6),
						title=""
					)
					ax.add_artist(leg_stats)
				# --- end stats legend ---

				if any_ax_with_lines is None and ax.lines:
					any_ax_with_lines = ax

		if any_ax_with_lines is not None:
			handles, labels = any_ax_with_lines.get_legend_handles_labels()
			fig.legend(handles, labels, loc='upper center', ncol=4, frameon=True,  bbox_to_anchor=(0.5, 0.99),
				columnspacing=0.8, handlelength=1.2, handletextpad=0.4, borderaxespad=0.2
			)

		fig.tight_layout(pad=0.6, w_pad=0.6, h_pad=0.6, rect=(0, 0, 1, 0.965))


		os.makedirs(out_dir, exist_ok=True)
		out_path = os.path.join(out_dir, f"reliance_vs_trust_attitude_effort_by_scenario-s={min_seconds}_{max_seconds}{'-explanation_clarity' if keep_only_who_easily_understood_explanation else ''}{'-balanced' if do_balance_treatments else ''}{'-changed_decision' if keep_only_who_changed_decision else ''}.pdf")
		plt.savefig(out_path, bbox_inches='tight')
		if SHOW_ALL_FIGURES: plt.show()
		print(f"Saved figure to: {out_path}")

def plot_effort_reliance_by_scenario(df, out_dir, min_seconds=None, max_seconds=None):
	effort_col = "How much effort did it take to understand and complete this task?"
	scenario_col = "Scenario"
	d = df.copy()

	d = d[d.apply(lambda x: within_quantiles(x, min_seconds, max_seconds), axis=1)]

	if "Reliance category" not in d.columns:
		d["Reliance category"] = d.apply(label_reliance, axis=1)
	d["Effort"] = pd.to_numeric(d[effort_col], errors="coerce")
	d = d.dropna(subset=["Effort", scenario_col])
	d["Effort"] = (d["Effort"] + 1).astype(int)
	d["is_over"] = ((d["Response before explanation"] == "Accept") & (d["Expected answer"] == "Reject")).astype(int)
	d["is_under"] = ((d["Response before explanation"] == "Reject") & (d["Expected answer"] == "Accept")).astype(int)

	scenarios = sorted(d[scenario_col].unique())
	cmap = plt.get_cmap('tab10')

	fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey='row', facecolor='white')
	metrics = [('is_over', 'Over-reliance'), ('is_under', 'Under-reliance')]
	magix_flags = [False, True]

	marker_styles = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
	line_width = 2.0
	marker_size = 8
	title_fs = 14
	label_fs = 12
	tick_fs = 10
	legend_fs = 10

	# placeholder for the first‐quadrant stats legend
	first_stats_leg = None

	for i, (metric, mlabel) in enumerate(metrics):
		for j, flag in enumerate(magix_flags):
			ax = axes[i, j]
			stats_lines = []

			for idx, scen in enumerate(scenarios):
				sub = d[(d[scenario_col] == scen) & (d['Explanation is MAGIX-defined'] == flag)]
				grp = sub.groupby('Effort')[metric].agg(mean='mean', n='sum').reset_index()
				if grp.empty:
					continue

				ax.plot(
					grp['Effort'], grp['mean'],
					marker=marker_styles[idx % len(marker_styles)],
					markersize=marker_size,
					linewidth=line_width,
					label=scen,
					color=cmap(idx)
				)

				for _, r in grp.iterrows():
					ax.text(
						r['Effort'], r['mean'], f"{int(r['n'])}",
						fontsize=8, va='bottom', ha='center', alpha=0.7,
						bbox=dict(facecolor='white', alpha=0.8, pad=1, edgecolor='none')
					)

				if sub['Effort'].nunique() >= 2:
					rho, p = spearmanr(sub['Effort'], sub[metric])
					stats_lines.append(f"{scen.replace('Scenario ','S')}: ρ={rho:.2f} (p={p:.3f}), n={len(sub)}")

			# draw stats legend in each subplot
			if stats_lines:
				stats_handles = [Line2D([], [], linestyle='') for _ in stats_lines]
				stats_leg = ax.legend(
					stats_handles, stats_lines,
					loc='lower center',
					# bbox_to_anchor=(0.5, 0.05),
					frameon=True,
					fontsize=6,
					ncol=1,
					borderaxespad=0,
					handlelength=0,       # no handle line at all
					handletextpad=0.2,    # tiny gap between (non-existent) handle and text
					labelspacing=0.1,     # less vertical space if you stack them
					columnspacing=0,       # no extra space between columns (if ncol>1)
				)
				# remember the first quadrant’s stats legend
				if i == 0 and j == 0:
					first_stats_leg = stats_leg

			ax.set_title(f"{mlabel} — {'MAGIX' if flag else 'Non-MAGIX'}", fontsize=title_fs)
			if i == 1:
				ax.set_xlabel('Effort (1–5)', fontsize=label_fs)
			if j == 0:
				ax.set_ylabel('Proportion', fontsize=label_fs)
				ax.yaxis.set_major_formatter(PercentFormatter(1.0))
			ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
			ax.set_xticks(sorted(d['Effort'].unique()))
			ax.tick_params(axis='both', labelsize=tick_fs)

	# add scenario legend only to first quadrant, **without** removing stats
	first_ax = axes[0, 0]
	sc_handles, sc_labels = first_ax.get_legend_handles_labels()
	scen_leg = first_ax.legend(
		sc_handles, sc_labels,
		title='Scenario',
		fontsize=legend_fs,
		title_fontsize=legend_fs,
		loc='upper left'
	)
	# re‐add the stats legend on top of it
	if first_stats_leg is not None:
		first_ax.add_artist(first_stats_leg)

	# fig.suptitle(f"Effort vs Reliance across scenarios ({max_seconds} ≥ seconds ≥ {min_seconds})", fontsize=18, y=1)
	# plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12, hspace=0.1, wspace=0.05)

	out_path = os.path.join(out_dir, f"effort_reliance_comparison_by_scenario-s={min_seconds}_{max_seconds}.pdf")
	plt.savefig(out_path, dpi=300, bbox_inches='tight')
	if SHOW_ALL_FIGURES: plt.show()
	print(f"Saved figure to: {out_path}")

def plot_effort_distribution(df, out_dir, min_seconds=None, max_seconds=None, cmap_name='Set3'):
	"""
	Improved boxplot of 'How much effort...' Likert responses by scenario,
	using a light pastel colormap and readable annotation backgrounds.
	"""
	# Filter and prepare data
	effort_col = "How much effort did it take to understand and complete this task?"
	d = df.copy()
	d = d[d.apply(lambda x: within_quantiles(x, min_seconds, max_seconds), axis=1)]
	if "Reliance category" not in d.columns:
		d["Reliance category"] = d.apply(label_reliance, axis=1)
	d["Effort"] = pd.to_numeric(d[effort_col], errors="coerce") + 1
	d = d.dropna(subset=["Effort"])

	# Gather per-scenario
	scenarios = sorted(d["Scenario"].unique())
	data = [d[d["Scenario"] == sc]["Effort"].values for sc in scenarios]

	# Compute stats for annotation
	stats = {}
	for sc, vals in zip(scenarios, data):
		q1, med, q3 = np.percentile(vals, [25, 50, 75])
		mean = np.mean(vals)
		stats[sc] = {"q1": q1, "med": med, "q3": q3, "mean": mean}

	# Set up colormap (light pastel)
	cmap = plt.get_cmap(cmap_name, len(scenarios))
	colors = [cmap(i) for i in range(len(scenarios))]

	# Plot
	fig, ax = plt.subplots(figsize=(8, 4))
	bp = ax.boxplot(
		data,
		labels=scenarios, #[sc.replace("Scenario", "S") for sc in scenarios],
		showmeans=True,
		patch_artist=True,
		boxprops=dict(linewidth=1.5),
		whiskerprops=dict(color='gray', linewidth=1),
		capprops=dict(color='gray', linewidth=1),
		medianprops=dict(color='black', linewidth=2),
		meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='white'),
		flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='gray', markersize=5, alpha=0.6)
	)

	# Color each box with pastel
	for patch, color in zip(bp['boxes'], colors):
		patch.set_facecolor(color)
		patch.set_alpha(0.8)

	# Axes labels and title styling
	ax.set_title('Effort Distribution by Scenario', fontsize=16, fontweight='bold')
	ax.set_xlabel('')
	ax.set_ylabel('Effort (1-5)', fontsize=14)
	ax.tick_params(axis='x', labelrotation=0, labelsize=10)
	ax.tick_params(axis='y', labelsize=10)
	ax.yaxis.set_major_locator(MaxNLocator(integer=True))
	ax.grid(axis='y', linestyle='--', alpha=0.5)

	# Annotate stats with white background boxes
	ymin, ymax = ax.get_ylim()
	span = ymax - ymin
	offsets = {'q1': span * 0.02, 'med': span * 0.05, 'q3': span * 0.08, 'mean': span * 0.11}
	for i, sc in enumerate(scenarios, start=1):
		s = stats[sc]
		for key, style in zip(['q1','med','q3','mean'], ['Q1','Med','Q3','Mean']):
			y_val = s[key] + offsets[key]
			ax.text(
				i,
				y_val,
				f"{style}={s[key]:.2f}",
				ha='center', va='bottom', fontsize=8,
				color='black',
				bbox=dict(facecolor='white', alpha=0.8, pad=1, edgecolor='none')
			)

	plt.tight_layout()

	# Save and show
	out_path = os.path.join(out_dir, f"effort_distribution-s={min_seconds}_{max_seconds}.pdf")
	plt.savefig(out_path)
	if SHOW_ALL_FIGURES: plt.show()
	print(f"Saved improved effort distribution plot to {out_path}")

def visualize_distribution(df, out_dir, min_seconds=None, max_seconds=None, keep_only_who_changed_decision=False, do_balance_treatments=False, keep_only_who_easily_understood_explanation=False, figsize=(8, 5)):
	"""
	Compute and plot the distribution of participants across:
	  - Scenario
	  - AI correctness (Accept → AI Correct, Reject → AI Incorrect)
	  - Explanation is MAGIX-defined (True/False)

	Parameters
	----------
	df : pandas.DataFrame
		DataFrame must contain columns
		"Scenario", "Expected answer", and "Explanation is MAGIX-defined".
	figsize : tuple, default (8, 6)
		Figure size.

	Returns
	-------
	table_counts : pandas.DataFrame
		Multi‐indexed table of raw counts with index=(Scenario, AI correctness)
		and columns=[False, True].
	"""
	# 0) Filter and copy
	df = df.copy()
	df = df[df.apply(lambda x: within_quantiles(x, min_seconds, max_seconds), axis=1)]

	# 1) Map Expected answer to AI correctness labels
	df['AI correctness'] = df['Expected answer'].map({
		'Accept': 'AI Correct',
		'Reject': 'AI Incorrect'
	})

	# 2) Build raw counts table
	table_counts = pd.crosstab(
		index=[df["Scenario"], df["AI correctness"]],
		columns=df["Explanation is MAGIX-defined"],
		dropna=False
	).sort_index()

	# 3) Compute proportions table (row-wise)
	table_props = table_counts.div(table_counts.sum(axis=1), axis=0)

	# 4) Choose table for plotting
	table_to_plot = table_counts

	# 5) Plot
	fig, ax = plt.subplots(figsize=figsize)
	table_to_plot.plot(
		kind="bar",
		stacked=True,
		ax=ax,
		width=0.8
	)
	ax.set_ylabel("Count")
	ax.set_title("Participants by Scenario / AI correctness / Explanation")
	ax.set_xlabel("")

	# Tidy up legend
	ax.legend(
		title="Explanation",
		loc="upper right",
		labels=["Not MAGIX-defined", "MAGIX-defined"]
	)

	# 6) Annotate with both counts and proportions (with white semi-transparent background)
	for i, container in enumerate(ax.containers):
		raw_vals = table_counts.values[:, i]
		prop_vals = table_props.values[:, i]
		labels = []
		for raw, prop in zip(raw_vals, prop_vals):
			labels.append(f"{int(raw)}\n({prop*100:.1f}%)")
		ax.bar_label(
			container,
			labels=labels,
			label_type='center',
			bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.2)
		)

	# 7) Improve readability of x-axis tick labels
	combined_labels = [f"{scenario}\n{ai_label}" for scenario, ai_label in table_counts.index]
	ax.set_xticks(range(len(combined_labels)))
	ax.set_xticklabels(combined_labels, rotation=45, ha='right', fontsize=10)

	plt.tight_layout()

	# 8) Save and show
	out_path = os.path.join(out_dir, f"participants_distribution-s={min_seconds}_{max_seconds}{'-explanation_clarity' if keep_only_who_easily_understood_explanation else ''}{'-balanced' if do_balance_treatments else ''}{'-changed_decision' if keep_only_who_changed_decision else ''}.pdf")
	plt.savefig(out_path)
	if SHOW_ALL_FIGURES: plt.show()

	return table_counts

def balance_treatments(df, seed=42):
	"""
	For each (Scenario, AI correctness) group, randomly down‐sample
	the larger of the MAGIX vs non-MAGIX subsets so both have equal size.
	"""
	np.random.seed(seed)
	# ensure AI correctness column exists
	if 'AI correctness' not in df.columns:
		df = df.copy()
		df['AI correctness'] = df['Expected answer'].map({
			'Accept': 'AI Correct',
			'Reject': 'AI Incorrect'
		})
	parts = []
	grouped = df.groupby(['Scenario', 'AI correctness'])
	dropped_entries = 0
	all_entries = 0
	for (_, _), grp in grouped:
		t = grp[grp['Explanation is MAGIX-defined'] == True]
		f = grp[grp['Explanation is MAGIX-defined'] == False]
		n = min(len(t), len(f))
		if n == 0:
			# skip if one side is empty
			continue
		dropped_entries += abs(len(t)-len(f))
		all_entries += len(t) + len(f)
		parts.append(t.sample(n=n, random_state=seed))
		parts.append(f.sample(n=n, random_state=seed))
	print(f'<balance_treatments> Dropped entries: {dropped_entries}/{all_entries} ({100*dropped_entries/all_entries:.2f}%)')
	return pd.concat(parts, ignore_index=True)

def plot_gender_distribution(df, out_dir):
	"""
	Plot the distribution of participants' gender.
	Assumes encoding: 0=Male, 1=Female, 2=Others/Prefer not to say.
	"""
	gender_col = "What is your gender?"
	if gender_col not in df.columns:
		print(f"Column '{gender_col}' not found in DataFrame.")
		return

	mapping = {0: "Male", 1: "Female", 2: "Others/Prefer not to say"}
	d = df.copy()
	d = d.drop_duplicates(subset=["Prolific ID"]) # Keep only one row per participant
	d = d.dropna(subset=[gender_col])

	# Map numeric codes to labels
	d[gender_col] = d[gender_col].map(mapping).fillna("Unknown")

	counts = d[gender_col].value_counts().reindex(mapping.values(), fill_value=0)
	props = counts / counts.sum()

	fig, ax = plt.subplots(figsize=(5, 4))
	bars = ax.bar(counts.index, counts.values, color="skyblue", edgecolor="black")

	# Annotate with count and percentage
	for bar, pct in zip(bars, props.values):
		h = bar.get_height()
		ax.annotate(
			f"{h}\n({pct*100:.1f}%)",
			xy=(bar.get_x() + bar.get_width() / 2, h//2),
			xytext=(0, 3),
			textcoords="offset points",
			ha="center",
			va="bottom",
			fontsize=9,
			bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.2)
		)

	# ax.set_title("Participants' Gender Distribution")
	ax.set_ylabel("Count")
	ax.set_xlabel("Gender")
	ax.yaxis.set_major_locator(MaxNLocator(integer=True))
	ax.grid(axis="y", linestyle="--", alpha=0.5)

	plt.tight_layout()
	out_path = os.path.join(out_dir, "gender_distribution.pdf")
	plt.savefig(out_path)
	if SHOW_ALL_FIGURES: plt.show()
	print(f"Saved gender distribution plot to {out_path}")

def main():
	parser = argparse.ArgumentParser(description="Analyse reliance patterns in scenario CSVs.")
	parser.add_argument("--input", required=True, help="Directory containing scenario_*.csv files, or a .zip of them.")
	parser.add_argument("--output", required=True, help="Directory to write results.")
	parser.add_argument("--min-seconds", type=int, default=None, help="Minimum 'Seconds' to include (default: 1% quantile).")
	parser.add_argument("--max-seconds", type=int, default=None, help="Maximum 'Seconds' to include (default: 99% quantile).")
	parser.add_argument("--keep_only_who_changed_decision", action="store_true")
	parser.add_argument("--balance_treatments", action="store_true")
	parser.add_argument("--keep_only_who_easily_understood_explanation", action="store_true")
	args = parser.parse_args()

	ensure_dir(args.output)

	raw_df = load_frames(args.input, args.output)
	raw_df = filter_invalid_rows(raw_df, args.input)

	if args.min_seconds is None:
		args.min_seconds = 10 # 30 seconds
	# 	# args.min_seconds = raw_df.groupby("Scenario")["Seconds"].quantile(0.01).clip(upper=30).astype(int).to_dict()
	# 	# print("Min seconds (1st percentile) per scenario:\n", args.min_seconds)

	# if args.max_seconds is None:
	# 	args.max_seconds = 360 # 6 minutes
	# 	# args.max_seconds = raw_df.groupby("Scenario")["Seconds"].quantile(0.99).clip(lower=360).astype(int).to_dict()
	# 	# print("Max seconds (99th percentile) per scenario:\n", args.max_seconds)

	plot_gender_distribution(raw_df, args.output)
	visualize_distribution(raw_df, args.output, args.min_seconds, args.max_seconds)
	plot_effort_distribution(raw_df, args.output, args.min_seconds, args.max_seconds)
	plot_reliance_vs_trust_attitude_effort(raw_df, args.output, args.min_seconds, args.max_seconds)
	# plot_reliance_vs_trust_attitude_effort_by_scenario(raw_df, args.output, args.min_seconds, args.max_seconds)
	# plot_effort_reliance_by_scenario(raw_df, args.output, args.min_seconds, args.max_seconds)
	# plot_corrections(raw_df, args.output, args.min_seconds, args.max_seconds)

	df, counts = analyse(raw_df, min_seconds=args.min_seconds, max_seconds=args.max_seconds, keep_only_who_changed_decision=args.keep_only_who_changed_decision, do_balance_treatments=args.balance_treatments, keep_only_who_easily_understood_explanation=args.keep_only_who_easily_understood_explanation)
	visualize_distribution(df, args.output, args.min_seconds, args.max_seconds, args.keep_only_who_changed_decision, args.balance_treatments, keep_only_who_easily_understood_explanation=args.keep_only_who_easily_understood_explanation)
	plot_mitigation_by_driver(raw_df, args.output, args.min_seconds, args.max_seconds, args.keep_only_who_changed_decision, args.balance_treatments)
	plot_per_scenario_multi(df, args.output, args.min_seconds, args.max_seconds, args.keep_only_who_changed_decision, args.balance_treatments, keep_only_who_easily_understood_explanation=args.keep_only_who_easily_understood_explanation)
	# plot_reliance_proportions(counts, args.output, args.min_seconds, args.max_seconds, args.keep_only_who_changed_decision, args.balance_treatments, keep_only_who_easily_understood_explanation=args.keep_only_who_easily_understood_explanation)

if __name__ == "__main__":
	main()
