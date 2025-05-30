import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
import os
import re

# Load new-format CSV
file_path = r"D:\apoptosis_code\P10.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Rename and compute columns
df["Raw"] = df["Total manual V1 count"]
df["AvgScaled"] = pd.to_numeric(df["Total manual V1 count, scaled up by average growth factor for long incubation pups"], errors="coerce")
df["IndivScaled"] = pd.to_numeric(df["Total manual V1 count, scaled up by individualized growth factor for long incubation pups"], errors="coerce")

# Create normalization metrics
df["Area"] = df["Total V1 area"]
df["ExclArea"] = df["Total area excluding sections with no cells"]
df["Injection"] = df["Injection site count"]

# Basic normalized values
df["Raw_per_Inj"] = df["Raw"] / df["Injection"]
df["Raw_per_Area"] = df["Raw"] / df["Area"]
df["Raw_per_ExclArea"] = df["Raw"] / df["ExclArea"]
df["Raw_Area_Inj"] = df["Raw"] / df["Area"] / df["Injection"]
df["Raw_ExclArea_Inj"] = df["Raw"] / df["ExclArea"] / df["Injection"]

# Growth-adjusted density for long group (pups only)
df["AdjDensity_1.17"] = df["Raw"] / df["Area"] / df["Injection"]
df["ExclAdjDensity_1.17"] = df["Raw"] / df["ExclArea"] / df["Injection"]
df.loc[(df["Cohort"] == "pup") & (df["Incubation"] == "long"), "AdjDensity_1.17"] *= 1.17
df.loc[(df["Cohort"] == "pup") & (df["Incubation"] == "long"), "ExclAdjDensity_1.17"] *= 1.17

# Function to compute stats and generate plots
def analyze_combined(df, column, label):
    groups = [
        ("pup", "short"), ("pup", "long"),
        ("adult", "short"), ("adult", "long")
    ]
    data = []
    for cohort, inc in groups:
        vals = df[(df["Cohort"] == cohort) & (df["Incubation"] == inc)][column].dropna()
        data.append(vals if len(vals) > 0 else pd.Series([np.nan]))

    means = [np.mean(vals) for vals in data]
    sems = [np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0 for vals in data]

    # Statistical comparison: pup short vs pup long, adult short vs adult long
    stats = []
    for i, j in [(0, 1), (2, 3)]:
        if not (data[i].isna().all() or data[j].isna().all()):
            if len(data[i].dropna()) >= 3 and len(data[j].dropna()) >= 3:
                if shapiro(data[i].dropna()).pvalue > 0.05 and shapiro(data[j].dropna()).pvalue > 0.05:
                    stat, pval = ttest_ind(data[i].dropna(), data[j].dropna(), equal_var=(levene(data[i].dropna(), data[j].dropna()).pvalue > 0.05))
                else:
                    stat, pval = mannwhitneyu(data[i].dropna(), data[j].dropna())
            else:
                pval = np.nan
        else:
            pval = np.nan
        stats.append(pval)

    stars = lambda p: "*" if p < 0.05 else "ns"

    plt.figure(figsize=(8, 5))
    plt.bar(range(4), means, yerr=sems, capsize=5, alpha=0.7)
    colors = ['blue', 'blue', 'orange', 'orange']
    for i in range(4):
        if not data[i].isna().all():
            plt.scatter([i]*len(data[i]), data[i], color=colors[i], zorder=10)

    for idx, (i, j) in enumerate([(0, 1), (2, 3)]):
        if not (data[i].isna().all() or data[j].isna().all()):
            y = max(max(data[i].dropna()), max(data[j].dropna())) * 1.1
            plt.plot([i, j], [y, y], color='black')
            plt.text((i + j) / 2, y * 1.01, stars(stats[idx]), ha='center')

    plt.xticks(range(4), ['Pup Short', 'Pup Long', 'Adult Short', 'Adult Long'])
    plt.ylabel(column)
    plt.title(label)
    plt.tight_layout()
    os.makedirs("plots_combined", exist_ok=True)
    # Ensure safe filenames by removing characters invalid in Windows
    safe_filename = re.sub(r'[\\/*?:"<>|]', "_", label.replace(' ', '_'))
    plt.savefig(f"plots_combined/{safe_filename}.png")
    plt.close()

# Comparisons to run
comparisons = {
    "Raw Manual V1 Count": "Raw",
    "Normalized by Area (cells/area)": "Raw_per_Area",
    "Normalized by Injection Site Count (cells/starter)": "Raw_per_Inj",
    "Normalized by Area and Injection Count (density/starter)": "Raw_Area_Inj",
    "Normalized by Excl Area": "Raw_per_ExclArea",
    "Normalized by Excl Area and Injection": "Raw_ExclArea_Inj",
    "Normalized by ((tot_area/count)/starter) long density adjusted by *1.17": "AdjDensity_1.17",
    "Normalized by ((excl_area/count)/starter) long density adjusted by *1.17": "ExclAdjDensity_1.17"
}

# Run and plot all comparisons
results = []
for label, col in comparisons.items():
    if "1.17" in label:
        filtered_df = df.copy()
        filtered_df = filtered_df[~((filtered_df["Cohort"] == "adult") & (filtered_df["Incubation"] == "long"))]
        filtered_df = filtered_df[~((filtered_df["Cohort"] == "adult") & (filtered_df["Incubation"] == "short"))]
    else:
        filtered_df = df
    analyze_combined(filtered_df, col, label)

# Export updated dataframe with all calculated columns
df.to_csv("Updated_Animal_Normalization_Values.csv", index=False)

print("All combined plots with significance stars saved to ./plots_combined")
print("Updated dataframe with new normalization columns saved to Updated_Animal_Normalization_Values.csv")
