
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from rdkit import Chem
from rdkit.Chem import Draw
from scipy.stats import ttest_ind


# EDIT THESE FILENAMES to match your CSVs.
# These paths are relative to the notebook's current working directory.
chembl_file = Path("chembl_sample_300_mw_200_400.csv")
pubchem_file = Path("Escerichia Coli_pubchem.csv")
cleaned_file = Path("chembl_pubchem_cleaned.csv")
best_molecules_file = Path("best_10_molecules.png")

# Load and inspect both datasets.
chembl_df = pd.read_csv(chembl_file)
pubchem_df = pd.read_csv(pubchem_file)

print("ChEMBL shape:", chembl_df.shape)
display(chembl_df.head())
chembl_df.info()

print("\nPubChem shape:", pubchem_df.shape)
display(pubchem_df.head())
pubchem_df.info()


# Confirm that the columns needed below are present.
required_chembl_columns = {
    "molecule_chembl_id",
    "smiles",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "assay_chembl_id",
}
required_pubchem_columns = {"smiles", "XLogP"}

missing_chembl = required_chembl_columns - set(chembl_df.columns)
missing_pubchem = required_pubchem_columns - set(pubchem_df.columns)

if missing_chembl:
    raise ValueError(f"Missing ChEMBL columns: {sorted(missing_chembl)}")
if missing_pubchem:
    raise ValueError(f"Missing PubChem columns: {sorted(missing_pubchem)}")


# Normalize the join key so whitespace does not prevent an otherwise exact match.
chembl_df["smiles"] = chembl_df["smiles"].astype("string").str.strip()
pubchem_df["smiles"] = pubchem_df["smiles"].astype("string").str.strip()

# PubChem descriptors should have one row per SMILES. Keep ChEMBL assay rows:
# the same molecule can have separate measurements for different assays.
print("Duplicate PubChem SMILES:", pubchem_df["smiles"].duplicated().sum())
pubchem_df = pubchem_df.dropna(subset=["smiles"]).drop_duplicates(
    subset="smiles",
    keep="first",
).copy()

# Merge each ChEMBL measurement with at most one PubChem descriptor row.
df = chembl_df.merge(
    pubchem_df,
    on="smiles",
    how="inner",
    validate="many_to_one",
)

print("\nMerged shape:", df.shape)
print("Merged columns:", df.columns.tolist())
display(df.head())


# Keep MIC measurements and convert supported mass/volume units to ug/mL.
# Surface-area units (for example ug cm**-2) are not concentrations, so they
# are excluded rather than incorrectly converted.
df["standard_type"] = df["standard_type"].astype("string").str.strip().str.upper()
df["standard_relation"] = (
    df["standard_relation"]
    .astype("string")
    .str.strip()
    .str.strip("'\"")
)
df["standard_units"] = (
    df["standard_units"]
    .astype("string")
    .str.strip()
    .str.lower()
    .str.replace("μ", "u", regex=False)
    .str.replace("µ", "u", regex=False)
    .str.replace(" ", "", regex=False)
)

unit_to_ug_per_ml = {
    "ug.ml-1": 1.0,
    "ug/ml": 1.0,
    "mg.l-1": 1.0,
    "mg/l": 1.0,
    "ng.ml-1": 0.001,
    "ng/ml": 0.001,
    "mg.ml-1": 1000.0,
    "mg/ml": 1000.0,
    "ug.l-1": 0.001,
    "ug/l": 0.001,
}

df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
df["unit_factor"] = df["standard_units"].map(unit_to_ug_per_ml)

# Remove non-MIC records, censored MIC values (such as > or <), unsupported
# units, and missing/non-positive measurements. Censored values are limits,
# not exact MIC measurements, so they are excluded from this t-test workflow.
df = df[
    (df["standard_type"] == "MIC")
    & (df["standard_relation"] == "=")
    & df["unit_factor"].notna()
    & df["standard_value"].notna()
    & (df["standard_value"] > 0)
].copy()

df["mic_ug_ml"] = df["standard_value"] * df["unit_factor"]

# Remove rows lacking the descriptor used in this example.
df["XLogP"] = pd.to_numeric(df["XLogP"], errors="coerce")
df = df.dropna(subset=["smiles", "XLogP", "mic_ug_ml"]).copy()

# The logarithm is useful for visualizing/testing MIC values, which often span
# several orders of magnitude. Keep mic_ug_ml in the dataset in original units.
df["log10_mic_ug_ml"] = np.log10(df["mic_ug_ml"])

print("\nCleaned shape:", df.shape)
print("\nMIC units retained:")
print(df["standard_units"].value_counts())
print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False).head(15))
print("\nSummary statistics:")
display(df[["mic_ug_ml", "XLogP", "MolecularWeight"]]
        .describe(include="all")
        if "MolecularWeight" in df.columns
        else df[["mic_ug_ml", "XLogP"]].describe())

# Save the cleaned, merged dataset.
df.to_csv(cleaned_file, index=False)
print(f"Saved cleaned dataset to: {cleaned_file}")


# Basic EDA: MIC and descriptor distributions, followed by their relationship.
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.histplot(df["mic_ug_ml"], bins=30, ax=axes[0])
axes[0].set_xscale("log")
axes[0].set_title("MIC distribution")
axes[0].set_xlabel("MIC (ug/mL; log scale)")

sns.histplot(df["XLogP"], bins=30, ax=axes[1])
axes[1].set_title("XLogP distribution")
axes[1].set_xlabel("PubChem XLogP")

sns.scatterplot(data=df, x="XLogP", y="mic_ug_ml", alpha=0.7, ax=axes[2])
axes[2].set_yscale("log")
axes[2].set_title("MIC vs XLogP")
axes[2].set_ylabel("MIC (ug/mL; log scale)")

plt.tight_layout()
plt.show()


# Show the ten best unique molecules, where lower exact MIC is better.
# A molecule with multiple assay measurements is represented by its lowest
# retained MIC for this visualization only.
best = (
    df.sort_values("mic_ug_ml")
    .drop_duplicates(subset="smiles")
    .head(10)
    .copy()
)

molecules = []
legends = []

for _, row in best.iterrows():
    mol = Chem.MolFromSmiles(row["smiles"])
    if mol is not None:
        molecules.append(mol)
        legends.append(
            f"{row['molecule_chembl_id']} | MIC={row['mic_ug_ml']:g} ug/mL"
        )

if molecules:
    image = Draw.MolsToGridImage(
        molecules,
        molsPerRow=5,
        subImgSize=(260, 200),
        legends=legends,
    )
    image.save(best_molecules_file)
    display(image)
    print(f"Saved molecule image to: {best_molecules_file}")
else:
    print("No valid SMILES were available to draw.")


# Inspect the scatterplot and choose a scientifically reasonable XLogP split.
# Change 2.0 after looking at the plot; it is only an example threshold.
XLOGP_THRESHOLD = 2.0


def compare_xlogp_groups(data, label):
    """Compare log10 MIC between lower- and higher-XLogP groups."""
    low_group = data.loc[
        data["XLogP"] <= XLOGP_THRESHOLD, "log10_mic_ug_ml"
    ].dropna()
    high_group = data.loc[
        data["XLogP"] > XLOGP_THRESHOLD, "log10_mic_ug_ml"
    ].dropna()

    print(f"\n{label}")
    print(f"XLogP threshold: {XLOGP_THRESHOLD}")
    print(f"Lower/equal XLogP group n={len(low_group)}")
    print(f"Higher XLogP group n={len(high_group)}")

    if len(low_group) < 2 or len(high_group) < 2:
        print("Not enough observations in both groups to run a t-test.")
        return

    # Welch's two-sample t-test does not assume equal group variances.
    # This two-sided test checks for a difference, not a particular direction.
    result = ttest_ind(low_group, high_group, equal_var=False)

    print(f"Mean log10(MIC), lower/equal XLogP: {low_group.mean():.3f}")
    print(f"Mean log10(MIC), higher XLogP: {high_group.mean():.3f}")
    print(f"Welch t-statistic: {result.statistic:.3f}")
    print(f"p-value: {result.pvalue:.4g}")


compare_xlogp_groups(df, "All retained MIC measurements")


# Repeat the comparison among entries with MIC below 10 ug/mL.
# The assignment wording says ug/L; this example interprets its cutoff as
# 10 ug/mL because the source data shown in the assignment uses ug/mL.
# If you literally mean 10 ug/L, use 0.01 ug/mL here instead.
MIC_CUTOFF_UG_ML = 10.0
low_mic_df = df[df["mic_ug_ml"] < MIC_CUTOFF_UG_ML].copy()

print(f"\nRows with MIC < {MIC_CUTOFF_UG_ML} ug/mL:", len(low_mic_df))
compare_xlogp_groups(low_mic_df, f"MIC < {MIC_CUTOFF_UG_ML} ug/mL subset")