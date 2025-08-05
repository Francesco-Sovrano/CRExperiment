import argparse, os, zipfile, glob, tempfile, shutil
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter, MaxNLocator
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, spearmanr
import numpy as np
from matplotlib.lines import Line2D

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

def load_frames(path):
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
		frames = [pd.merge(questionnaire_df, pd.read_csv(f), on="Prolific ID", how="outer") for f in glob.glob(pattern, recursive=True)]
		if not frames:
			raise FileNotFoundError("No scenario_*.csv files found.")
		return pd.concat(frames, ignore_index=True)
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

def tidy_task(val):
	return str(val).split("_")[0].replace('task','scenario ').capitalize()

def filter_invalid_rows(df):
	# Keep only valid Prolific IDs
	df = df[df["Prolific ID"].str.len() == 24]
	# For any rows sharing both the same Prolific ID and the same Scenario, keep only the last occurrence.
	df["Scenario"] = df["Task file"].apply(tidy_task)
	df = df.drop_duplicates(subset=["Prolific ID", "Scenario"], keep="last")
	# Keep only those IDs that appear in 4 scenarios
	df = df[df.groupby("Prolific ID")["Scenario"].transform("nunique").eq(4)]
	return df

def analyse(df, min_seconds, keep_only_who_changed_mind, expected_answer=None):
	df = df.copy()
	# Keep only who spent enough time
	old_len = len(df)
	df = df[df["Seconds"] >= min_seconds]
	print(f'{old_len-len(df)} entries were removed because produce in less than {min_seconds} seconds')
	if expected_answer:
		df = df[df["Expected answer"] == expected_answer]

	# df = df[df["How much do you trust AI systems in general?"] <= 3]
	# df = df[df["How would you rate your overall attitude toward Artificial Intelligence (AI)?"] <= 3]
	# df = df[df['How much effort did it take to understand and complete this task?'] >= 3]
	df = df[
		(
			# Keep only who understood the explanations
			(df["How easy was it to understand the explanation?"] >= 2) # not difficult (2 = neutral)
			# # Keep only who understood the task
			# & (
			# 	(df["How confident are you in the decision you made? (with explanation)"] >= 1)
			# 	| (df["How confident are you in the decision you made? (without explanation)"] >= 1)
			# )
			# Keep only who actually used the explanations, updating their mental model
			& (df["Did the explanation help you evaluate the AI's output?"] >= 1)
		)
	]
	if keep_only_who_changed_mind:
		df = df[
			(df["How useful was the explanation provided?"] >= 1)
			& (
				(df["How confident are you in the decision you made? (without explanation)"] != df["How confident are you in the decision you made? (with explanation)"])
				| (df["Explanation changed mind"] == True)
			)
		]

	# df = df[(df["How confident are you in the decision you made? (without explanation)"] < df["How confident are you in the decision you made? (with explanation)"])]
	# df = df[df["How much effort did it take to understand and complete this task?"] <= 3]
	# df = df[df["Explanation changed mind"]]
	
	# df = df[df["How easy was it to understand the explanation?"] > 3]
	df["Reliance category"] = df.apply(label_reliance, axis=1)
	counts = (df.groupby(["Explanation is MAGIX-defined","Reliance category"])
				.size().unstack(fill_value=0)
				.reindex(RELIANCE_ORDER, axis=1).sort_index(axis=0))
	chi2, p, dof, _ = chi2_contingency(counts.values)
	print(f"Overall χ²={chi2:.3f}, dof={dof}, p={p:.4f}  (Seconds ≥ {min_seconds})")
	return df, counts

def plot_counts(counts, out_dir, seconds, keep_only_who_changed_mind):
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
	plt.savefig(os.path.join(out_dir, f"reliance_counts-s={seconds}{'-changed_mind' if keep_only_who_changed_mind else ''}.pdf"))
	plt.show()

def plot_props(counts, out_dir, seconds, keep_only_who_changed_mind):
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
	plt.savefig(os.path.join(out_dir, f"reliance_props-s={seconds}{'-changed_mind' if keep_only_who_changed_mind else ''}.pdf"))
	plt.show()

def plot_changes(df, out_dir, seconds, keep_only_who_changed_mind):
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
	plt.savefig(os.path.join(out_dir, f"response_changes-s={seconds}{'-changed_mind' if keep_only_who_changed_mind else ''}.pdf"))
	plt.show()

def plot_per_scenario_multi(df, out_dir, min_seconds, keep_only_who_changed_mind):
	"""
	Create a 1x3 subplot figure showing per-scenario reliance composition for:
	- All experiments (expected_answer=None)
	- Only Accept-correct (expected_answer='Accept')
	- Only Reject-incorrect (expected_answer='Reject')
	"""
	# Define the three analyses
	scenarios = None
	results = []
	for label, expected in [("All", None), ("Accept", "Accept"), ("Reject", "Reject")]:
		df_sub, counts = analyse(df, min_seconds, keep_only_who_changed_mind, expected_answer=expected)
		# prepare proportions per scenario
		base = (df_sub.groupby(["Scenario", "Explanation is MAGIX-defined", "Reliance category"])  
					.size().rename("n").reset_index())
		totals = base.groupby(["Scenario", "Explanation is MAGIX-defined"])["n"].transform("sum")
		base['prop'] = base['n'] / totals
		if scenarios is None:
			scenarios = sorted(base['Scenario'].unique())
		results.append((label, base))

	plt.rcParams.update({
		# … other params …
		"axes.grid": False,          # turn off all grids
		# "grid.alpha": 0.3,         # no longer needed
	})

	x = np.arange(len(scenarios))
	width = 0.3
	fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

	expl_colors = {False: 'C0', True: 'C1'}
	score_map = {"Under-reliance": 0, "Appropriate accept": 1, "Appropriate reject": 1, "Over-reliance": 0}

	for ax, (label, base) in zip(axes, results):
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
						ax.annotate(
							f"{c}({int(round(v*100)):.0f}%)",
							(x[i] + (idx - 0.5)*width, bottom[i] + v/2),
							xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=7,
							bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9)
						)
				bottom += vals
		# p-value annotation via Mann–Whitney U
		df_label = df[df['Seconds'] >= min_seconds].copy()
		if label != 'All':
			df_label = df_label[df_label['Expected answer'] == label]
		# compute p-values per scenario
		p_vals = {}
		for scen in scenarios:
			sub = df_label[df_label['Scenario'] == scen]
			scores_non = sub[sub['Explanation is MAGIX-defined'] == False]['Reliance category'].map(score_map)
			scores_mag = sub[sub['Explanation is MAGIX-defined'] == True]['Reliance category'].map(score_map)
			if len(scores_non)>0 and len(scores_mag)>0:
				# Run Mann-Whitney U test
				_, p = mannwhitneyu(scores_non, scores_mag, alternative='greater' if np.mean(scores_non) > np.mean(scores_mag) else 'less')
				# # Run Chi-squared test
				# non_counts = scores_non.value_counts().reindex([0, 1], fill_value=0)
				# mag_counts = scores_mag.value_counts().reindex([0, 1], fill_value=0)
				# contingency = [
				# 	[non_counts.loc[0], non_counts.loc[1]],
				# 	[mag_counts.loc[0], mag_counts.loc[1]]
				# ]
				# chi2, p, dof, expected = chi2_contingency(contingency)
			else:
				p = np.nan
			p_vals[scen] = p
		for i, scen in enumerate(scenarios):
			ax.text(x[i], 1.05, f"p={p_vals[scen]:.3f}", weight ='bold' if p_vals[scen] < 0.05 else 'normal', ha='center', va='bottom', fontsize=9)

		ax.set_xticks(x)
		ax.set_xticklabels(list(map(lambda x: x.replace('Scenario','Scen.'), scenarios)), rotation=0, ha='center', fontsize=9)
		ax.set_title(f"Expected: {label}", fontsize=9)
		if ax is axes[0]:
			ax.set_ylabel('Proportion within explanation type', fontsize=9)
			ax.yaxis.set_major_formatter(PercentFormatter(1.0))
		ax.set_ylim(0, 1.1)
		ax.yaxis.set_major_locator(MaxNLocator(5))
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
	plt.savefig(os.path.join(out_dir, f"per_scenario_reliance_props_multi-s={min_seconds}{'-changed_mind' if keep_only_who_changed_mind else ''}.pdf"))
	plt.show()

def plot_corrections(df, output_dir, seconds):
	df = df.copy()
	# Time filter only; keep full range of 'ease' values
	df = df[df["Seconds"] >= seconds]
	# Ensure reliance labels exist
	if "Reliance category" not in df.columns:
		df["Reliance category"] = df.apply(label_reliance, axis=1)

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
	plt.savefig(os.path.join(output_dir, f"corrections_by_scenario-s={seconds}.pdf"))
	plt.show()

def plot_mitigation_by_ease(df, out_dir, seconds, keep_only_who_changed_mind):
	"""
	Plot how 'How easy was it to understand the explanation?' relates to
	over- and under-reliance mitigation, split by MAGIX vs non-MAGIX,
	with statistical tests showing p-values for each ease level.

	Definitions:
	- Over-reliance mitigation rate (Expected=Reject):
	  Appropriate reject / (Appropriate reject + Over-reliance)
	- Under-reliance mitigation rate (Expected=Accept):
	  Appropriate accept / (Appropriate accept + Under-reliance)
	"""
	d = df.copy()
	# Time filter only; keep full range of 'ease' values
	d = d[d["Seconds"] >= seconds]
	ease_col = "How easy was it to understand the explanation?"
	d = d[pd.notna(d[ease_col])]
	d["Ease"] = d[ease_col].astype(int) + 1

	# Ensure reliance labels exist
	if "Reliance category" not in d.columns:
		d["Reliance category"] = d.apply(label_reliance, axis=1)

	# Filter for valid explanation feedback
	d = d[
		(d["Did the explanation help you evaluate the AI's output?"] >= 1)
	]

	if keep_only_who_changed_mind:
		d = d[
			(d["How useful was the explanation provided?"] >= 1)
			& (
				(df["How confident are you in the decision you made? (without explanation)"] != df["How confident are you in the decision you made? (with explanation)"])
				| (df["Explanation changed mind"] == True)
			)
		]

	def mitigation_series(data, expected, appropriate_label, error_label):
		sub = data[data["Expected answer"] == expected]
		tab = (sub.groupby(["Explanation is MAGIX-defined", "Ease", "Reliance category"])   
				   .size()
				   .unstack(fill_value=0))
		for col in (appropriate_label, error_label):
			if col not in tab.columns:
				tab[col] = 0
		den = tab[appropriate_label] + tab[error_label]
		rate = (tab[appropriate_label] / den).replace([np.inf, np.nan], np.nan)
		return rate, tab  # also return counts

	# Build rates and counts
	over_rate, over_counts = mitigation_series(
		d, expected="Reject",
		appropriate_label="Appropriate reject",
		error_label="Over-reliance"
	)
	under_rate, under_counts = mitigation_series(
		d, expected="Accept",
		appropriate_label="Appropriate accept",
		error_label="Under-reliance"
	)

	eases = sorted(d["Ease"].unique())
	fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
	titles = ["Over-reliance mitigation (Expected = Reject)",
			  "Under-reliance mitigation (Expected = Accept)"]
	rates_counts = [(over_rate, over_counts, "Appropriate reject", "Over-reliance"),
					(under_rate, under_counts, "Appropriate accept", "Under-reliance")]

	for ax, title, (rate, counts, appr_label, err_label) in zip(axes, titles, rates_counts):
		# Compute p-values for each ease level
		p_values = {}
		for e in eases:
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
			else:
				pval = np.nan
			p_values[e] = pval

		# Plot lines and annotate counts and p-values
		y_values = {}
		for magix_flag, label, marker in [(False, "Non-MAGIX", "o"), (True, "MAGIX", "s")]:
			series = rate.loc[magix_flag] if (magix_flag in rate.index.get_level_values(0)) else pd.Series(dtype=float)
			y = [series.get(e, np.nan) for e in eases]
			y_values[magix_flag] = y
			ax.plot(eases, y, marker=marker, label=label)
			# N annotations (slightly lower offset)
			for idx, e in enumerate(eases):
				N = 0
				if (magix_flag, e) in counts.index:
					N = int(counts.loc[(magix_flag, e)][appr_label] + counts.loc[(magix_flag, e)][err_label])
				yv = y[idx]
				if np.isfinite(yv):
					ax.annotate(f"N={N}", (e, yv), xytext=(0, 4), textcoords="offset points",
								ha="center", va="bottom", fontsize=7, bbox=dict(
						boxstyle="round,pad=0.2",
						facecolor="white",
						edgecolor="none",
						alpha=0.9
					))

		# P-value annotations higher and bold when <0.05
		for e in eases:
			pval = p_values.get(e, np.nan)
			if np.isfinite(pval):
				y0 = y_values.get(False)[eases.index(e)]
				y1 = y_values.get(True)[eases.index(e)]
				y_max = max([yv for yv in (y0, y1) if np.isfinite(yv)] + [0])
				weight = 'bold' if pval < 0.05 else 'normal'
				ax.annotate(f"p={pval:.3f}", (e, y_max), xytext=(0, 12), textcoords="offset points",
							ha="center", va="bottom", fontsize=8, fontweight=weight, bbox=dict(
						boxstyle="round,pad=0.2",
						facecolor="white",
						edgecolor="none",
						alpha=0.9
					))

		ax.set_title(title, fontsize=12)
		ax.set_xlabel("Ease of understanding (1–5)")
		ax.yaxis.set_major_formatter(PercentFormatter(1.0))
		ax.set_ylim(0, 1.1)
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))

	axes[0].set_ylabel("Mitigation rate")
	axes[0].legend(title="Explanation type", frameon=True)
	plt.tight_layout()
	os.makedirs(out_dir, exist_ok=True)
	plt.savefig(os.path.join(out_dir, f"mitigation_by_ease-s={seconds}{'-changed_mind' if keep_only_who_changed_mind else ''}.pdf"))
	plt.show()

def plot_reliance_vs_trust_attitude_effort(df, out_dir, seconds=0):
	"""
	Generate side-by-side plots showing over- and under-reliance rates against user-reported Effort, Attitude, and Trust,
	using consistent colors for MAGIX vs non-MAGIX groups.

	Parameters:
	- df: pd.DataFrame containing the survey data
	- out_dir: directory to save the output figures
	- seconds: minimum threshold for df['Seconds'] filtering
	"""
	# Define question columns and labels
	questions = {
		'Effort': 'How much effort did it take to understand and complete this task?',
		'Attitude': 'How would you rate your overall attitude toward Artificial Intelligence (AI)?',
		'Trust': 'How much do you trust AI systems in general?'
	}
	# Prepare data
	d = df.copy()
	d = d[d.get('Seconds', 0) >= seconds]
	if 'Reliance category' not in d.columns:
		d['Reliance category'] = d.apply(label_reliance, axis=1)
	d['is_over'] = ((d['Response before explanation']=='Accept') & (d['Expected answer']=='Reject')).astype(int)
	d['is_under'] = ((d['Response before explanation']=='Reject') & (d['Expected answer']=='Accept')).astype(int)

	# Convert each question to numeric scale starting at 1
	for key, col in questions.items():
		d[key] = pd.to_numeric(d[col], errors='coerce')
		d = d.dropna(subset=[key, "Scenario"])
		d[key] = (d[key] + 1).astype(int)

	# Setup subplots
	n_q = len(questions)
	fig, axes = plt.subplots(1, n_q, figsize=(5.5*n_q, 4), sharey=True)

	for key, _ in questions.items():
		print(f"\n=== {key} ===")
		print(" overall non-null count:", d[key].notna().sum())
		print(" unique values     :", d[key].dropna().unique())
		for flag, label in [(False,'Non-MAGIX'), (True,'MAGIX')]:
			sub = d[d['Explanation is MAGIX-defined']==flag]
			print(f"  {label}: count={sub[key].notna().sum()}, unique={sub[key].nunique()}")
	# assert False

	# Compute stats and plot
	stats = {}
	for idx, (key, _) in enumerate(questions.items()):
		ax = axes[idx]
		# Aggregate rates by question value and MAGIX flag
		grp = d.groupby(['Explanation is MAGIX-defined', key])
		rates = grp.agg(
			over_rate=('is_over', 'mean'),
			under_rate=('is_under', 'mean'),
			n=('is_over', 'size')
		).reset_index()
		# Plot data lines with consistent colors
		for magix_flag in [False, True]:
			subset = rates[rates['Explanation is MAGIX-defined']==magix_flag].sort_values(key)
			if subset.empty:
				continue
			# assign a single color per group
			color = 'C0' if not magix_flag else 'C1'
			label_prefix = 'Non-MAGIX' if not magix_flag else 'MAGIX'
			# solid line for over, dashed for under
			ax.plot(subset[key], subset['over_rate'], linestyle='-', marker='o',
					color=color, label=f'Over {label_prefix}', linewidth=1.2)
			ax.plot(subset[key], subset['under_rate'], linestyle='--', marker='x',
					color=color, label=f'Under {label_prefix}', linewidth=1.2)
			# Annotate sample sizes at over points
			for _, row in subset.iterrows():
				ax.annotate(f"n={int(row['n'])}", (row[key], row['over_rate']),
							xytext=(0, 6), textcoords='offset points', ha='center', va='top', fontsize=6)
		# Axis formatting
		ax.set_title(f"Reliance vs {key}")
		ax.set_xlabel(key)
		if idx == 0:
			ax.set_ylabel('Proportion')
			ax.yaxis.set_major_formatter(PercentFormatter(1.0))
		ax.grid(alpha=0.3)
		# Force x-ticks to be integers
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		# Compute Spearman correlations
		stats[key] = {}
		for flag, label in [(False, 'Non-MAGIX'), (True, 'MAGIX')]:
			sub = d[d['Explanation is MAGIX-defined']==flag]
			if sub[key].nunique() >= 2:
				rho_o, p_o = spearmanr(sub[key], sub['is_over'])
				rho_u, p_u = spearmanr(sub[key], sub['is_under'])
			else:
				rho_o = p_o = rho_u = p_u = np.nan
			stats[key][label] = {'rho_over': rho_o, 'p_over': p_o,
								  'rho_under': rho_u, 'p_under': p_u,
								  'n': len(sub)}
		# Build stats text for this panel
		panel_text = []
		for label in ['Non-MAGIX', 'MAGIX']:
			s = stats[key][label]
			panel_text.append(
				f"{label}: ρ_over={s['rho_over']:.2f} (p={s['p_over']:.3f}),"
				f" ρ_under={s['rho_under']:.2f} (p={s['p_under']:.3f})"
			)
		# Place text inside subplot
		ax.text(0, 1.2, '\n'.join(panel_text), transform=ax.transAxes,
				ha='left', va='top', fontsize=7,
				bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

	# Combined legend at bottom
	handles, labels = axes[-1].get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper center', ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.96))

	# Overall title & layout
	fig.suptitle('Reliance rates vs Effort, Attitude, and Trust (Before Explanations)', y=1)
	fig.tight_layout()

	# Save output
	os.makedirs(out_dir, exist_ok=True)
	out_path = os.path.join(out_dir, f"reliance_vs_trust_attitude_effort-s={seconds}.pdf")
	plt.savefig(out_path, bbox_inches='tight')
	plt.show()
	print(f"Saved figure to: {out_path}")

def plot_effort_reliance_by_scenario(df, out_dir, seconds):
	effort_col = "How much effort did it take to understand and complete this task?"
	scenario_col = "Scenario"

	# Prepare data
	d = df[df["Seconds"] >= seconds].copy()
	if "Reliance category" not in d.columns:
		d["Reliance category"] = d.apply(label_reliance, axis=1)
	d["Effort"] = pd.to_numeric(d[effort_col], errors="coerce")
	d = d.dropna(subset=["Effort", scenario_col])
	d["Effort"] = (d["Effort"] + 1).astype(int)
	d["is_over"] = ((d["Response before explanation"] == "Accept") & (d["Expected answer"] == "Reject")).astype(int)
	d["is_under"] = ((d["Response before explanation"] == "Reject") & (d["Expected answer"] == "Accept")).astype(int)

	scenarios = sorted(d[scenario_col].unique())
	cmap = plt.get_cmap('tab10')

	fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey='row', facecolor='white')
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
				grp = sub.groupby('Effort')[metric].agg(mean='mean', n='size').reset_index()
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
					stats_lines.append(f"{scen}: ρ={rho:.2f} (p={p:.3f}), n={len(sub)}")

			# draw stats legend in each subplot
			if stats_lines:
				stats_handles = [Line2D([], [], linestyle='') for _ in stats_lines]
				stats_leg = ax.legend(
					stats_handles, stats_lines,
					loc='lower center',
					bbox_to_anchor=(0.5, 0.05),
					frameon=True,
					fontsize=9,
					borderaxespad=0,
					ncol=1
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

	fig.suptitle(f"Effort vs Reliance across scenarios (seconds ≥ {seconds})", fontsize=18, y=0.98)
	plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12, hspace=0.1, wspace=0.05)

	out_path = os.path.join(out_dir, f"effort_reliance_comparison_by_scenario-s={seconds}.pdf")
	plt.savefig(out_path, dpi=300, bbox_inches='tight')
	plt.show()
	print(f"Saved figure to: {out_path}")

def plot_effort_distribution(df, out_dir, seconds, cmap_name='Set3'):
	"""
	Improved boxplot of 'How much effort...' Likert responses by scenario,
	using a light pastel colormap and readable annotation backgrounds.
	"""
	# Filter and prepare data
	effort_col = "How much effort did it take to understand and complete this task?"
	d = df.copy()
	d = d[d["Seconds"] >= seconds]
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
	fig, ax = plt.subplots(figsize=(10, 6))
	bp = ax.boxplot(
		data,
		labels=[sc.replace("Scenario", "Scen.") for sc in scenarios],
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
	ax.set_xlabel('Scenario', fontsize=14)
	ax.set_ylabel('Effort (1-5)', fontsize=14)
	ax.tick_params(axis='x', labelrotation=45, labelsize=12)
	ax.tick_params(axis='y', labelsize=12)
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
	out_path = os.path.join(out_dir, f"effort_distribution-s={seconds}.pdf")
	plt.savefig(out_path)
	plt.show()
	print(f"Saved improved effort distribution plot to {out_path}")

def visualize_distribution(df, out_dir, seconds=0, figsize=(8, 5)):
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
	df = df[df.get('Seconds', 0) >= seconds]

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
	out_path = os.path.join(out_dir, f"participants_distribution-s={seconds}.pdf")
	plt.savefig(out_path)
	plt.show()

	return table_counts

def main():
	parser = argparse.ArgumentParser(description="Analyse reliance patterns in scenario CSVs.")
	parser.add_argument("--input", required=True, help="Directory containing scenario_*.csv files, or a .zip of them.")
	parser.add_argument("--output", required=True, help="Directory to write results.")
	parser.add_argument("--min-seconds", type=int, default=10, help="Minimum 'Seconds' to include (default: 120).")
	parser.add_argument("--keep_only_who_changed_mind", action="store_true")
	args = parser.parse_args()

	ensure_dir(args.output)

	raw_df = load_frames(args.input)
	raw_df = filter_invalid_rows(raw_df)
	visualize_distribution(raw_df, args.output, args.min_seconds)

	plot_effort_distribution(raw_df, args.output, args.min_seconds)
	plot_reliance_vs_trust_attitude_effort(raw_df, args.output, args.min_seconds)
	plot_effort_reliance_by_scenario(raw_df, args.output, args.min_seconds)
	plot_corrections(raw_df, args.output, args.min_seconds)
	plot_mitigation_by_ease(raw_df, args.output, args.min_seconds, args.keep_only_who_changed_mind)

	df, counts = analyse(raw_df, args.min_seconds, args.keep_only_who_changed_mind)
	plot_per_scenario_multi(df, args.output, args.min_seconds, args.keep_only_who_changed_mind)
	plot_changes(df, args.output, args.min_seconds, args.keep_only_who_changed_mind)
	plot_counts(counts, args.output, args.min_seconds, args.keep_only_who_changed_mind)
	plot_props(counts, args.output, args.min_seconds, args.keep_only_who_changed_mind)

if __name__ == "__main__":
	main()
