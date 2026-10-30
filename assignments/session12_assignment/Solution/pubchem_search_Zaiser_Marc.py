import pubchempy as pcp
import pandas as pd
from tqdm import tqdm
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display
from rdkit import Chem
from rdkit.Chem import Draw
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu


# Define ROOT directory as parent folder
ROOT = Path(__file__).resolve().parent
input_file = ROOT / "Escerichia Coli.csv"
output_file = ROOT / "Escerichia Coli_pubchem.csv"
chembl_df = pd.read_csv(input_file, sep=";")
smiles_list = (
    chembl_df["Smiles"]
    .dropna()
    .astype(str)
    .str.strip()
)
smiles_list = smiles_list[smiles_list != ""].tolist()

DATA = ROOT/"Solution"



# Load your ChEMBL dataset
chembl_df = pd.read_csv(DATA / "/Users/marczaiser/Desktop/Studium /Masterstudium/2. Semester/DSA103_Advanced Chemical Data Science/ChemicalDataScience103_Fork_Zaiser_Marc/assignments/session12_assignment/Solution/Escerichia Coli.csv", sep=";") 
smiles_list = chembl_df["Smiles"].dropna().tolist() # use the column tag with the "smiles"

records = []

for smi in tqdm(smiles_list[:200]):  # limit to 200 compounds for testing, tqdm just adds progress bar
    try:
        compounds = pcp.get_compounds(smi, namespace='smiles') # look for each smiles-code in the list sliced from the ChEMBL data
        if not compounds:
            continue
        
        c = compounds[0]
        records.append({
            "Smiles": smi, # make sure that the column name for the smiles code matches the chembl dataframe!
            "CID": c.cid,
            "MolecularWeight": c.molecular_weight,
            "XLogP": c.xlogp,
            "TPSA": c.tpsa,
            "HBondDonorCount": c.h_bond_donor_count,
            "HBondAcceptorCount": c.h_bond_acceptor_count,
            "RotatableBondCount": c.rotatable_bond_count,
            "HeavyAtomCount": c.heavy_atom_count
        })
        
        # delay to avoid flooding PubChem servers
        time.sleep(0.2)
        
    except Exception as e:
        print(f"Failed for {smi[:20]}...: {e}")

pubchem_df = pd.DataFrame(records)
pubchem_df.to_csv(output_file, index=False, sep=";")
print(f"Saved {len(pubchem_df)} entries.")


# Combine ChEMBL measurements with PubChem descriptors

# Inspect both datasets before combining them.
print("ChEMBL shape:", chembl_df.shape)
display(chembl_df.head())
chembl_df.info()

print("\nPubChem shape:", pubchem_df.shape)
display(pubchem_df.head())
pubchem_df.info()

# Normalize the join key so leading/trailing whitespace does not prevent matches.
chembl_df["Smiles"] = chembl_df["Smiles"].astype("string").str.strip()
pubchem_df["Smiles"] = pubchem_df["Smiles"].astype("string").str.strip()

# PubChem should contribute only one descriptor row per SMILES.
# ChEMBL may have multiple rows for a molecule because it can have results from different assays, so those measurement rows are deliberately retained.
print("\nDuplicate PubChem SMILES:", pubchem_df["Smiles"].duplicated().sum())
pubchem_df = (
    pubchem_df.dropna(subset=["Smiles"])
    .drop_duplicates(subset="Smiles", keep="first")
    .copy()
)

# Validate that each ChEMBL measurement matches no more than one PubChem row.
df = chembl_df.merge(
    pubchem_df,
    on="Smiles",
    how="inner",
    validate="many_to_one",
)

print("\nMerged shape:", df.shape)
display(df.head())


# Clean MIC values and standardize units

# Keep MIC measurements only.
df["Standard Type"] = (
    df["Standard Type"].astype("string").str.strip().str.upper()
)
df["Standard Relation"] = (
    df["Standard Relation"]
    .astype("string")
    .str.strip()
    .str.strip("'\"")
)
df["Standard Units"] = (
    df["Standard Units"]
    .astype("string")
    .str.strip()
    .str.lower()
    .str.replace("μ", "u", regex=False)
    .str.replace("µ", "u", regex=False)
    .str.replace(" ", "", regex=False)
)

# Convert supported concentration units to ug/mL.
# Surface-area units (for example ug cm**-2) are not concentrations and
# cannot be converted to ug/mL, so those rows will be excluded below.
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

df["Standard Value"] = pd.to_numeric(df["Standard Value"], errors="coerce")
df["unit_factor"] = df["Standard Units"].map(unit_to_ug_per_ml)

# Keep exact, positive MIC values with recognized concentration units.
# Censored values (for example, > or <) are limits rather than exact values,
# so they are excluded from this comparison.
df = df[
    (df["Standard Type"] == "MIC")
    & (df["Standard Relation"] == "=")
    & df["unit_factor"].notna()
    & df["Standard Value"].notna()
    & (df["Standard Value"] > 0)
].copy()

df["mic_ug_ml"] = df["Standard Value"] * df["unit_factor"]

# Convert descriptor columns to numeric values and remove rows missing
# the MIC, XLogP, or SMILES needed for the analysis.
df["XLogP"] = pd.to_numeric(df["XLogP"], errors="coerce")
df = df.dropna(subset=["Smiles", "XLogP", "mic_ug_ml"]).copy()

# Log-transform MIC for analysis because MIC values commonly span
# several orders of magnitude. Lower log10(MIC) still means better activity.
df["log10_mic_ug_ml"] = np.log10(df["mic_ug_ml"])

print("\nCleaned dataset shape:", df.shape)
print("\nRetained source units:")
print(df["Standard Units"].value_counts())
print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False).head(15))
print("\nSummary statistics:")
display(df[["mic_ug_ml", "XLogP"]].describe())


# Exploratory data analysis

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.histplot(df["mic_ug_ml"], bins=30, ax=axes[0])
axes[0].set_xscale("log")
axes[0].set_title("MIC distribution")
axes[0].set_xlabel("MIC (ug/mL, log scale)")

sns.histplot(df["XLogP"], bins=30, ax=axes[1])
axes[1].set_title("XLogP distribution")
axes[1].set_xlabel("XLogP")

sns.scatterplot(data=df, x="XLogP", y="mic_ug_ml", alpha=0.7, ax=axes[2])
axes[2].set_yscale("log")
axes[2].set_title("MIC vs XLogP")
axes[2].set_ylabel("MIC (ug/mL, log scale)")

plt.tight_layout()
plt.show()


# Visualize the 10 best unique molecules with RDKit

# Sort by MIC (lower is better), then keep one row per molecule.
best = (
    df.sort_values("mic_ug_ml")
    .drop_duplicates(subset="Smiles")
    .head(10)
)

molecules = []
legends = []

for _, row in best.iterrows():
    mol = Chem.MolFromSmiles(row["Smiles"])
    if mol is not None:
        molecules.append(mol)
        legends.append(
            f"{row['Molecule ChEMBL ID']} | MIC={row['mic_ug_ml']:g} ug/mL"
        )

if molecules:
    image = Draw.MolsToGridImage(
        molecules,
        molsPerRow=5,
        subImgSize=(260, 200),
        legends=legends,
    )
    image.save(ROOT / "best_10_molecules.png")
    display(image)
else:
    print("No valid SMILES were available to draw.")


# Save the cleaned merged dataset.
cleaned_file = ROOT / "chembl_pubchem_cleaned.csv"
df.to_csv(cleaned_file, index=False, sep=";")
print(f"Saved cleaned dataset to: {cleaned_file}")



# Hypothesis and Welch's t-tests

# First inspect the scatterplot above, then adjust this example threshold
# to a scientifically reasonable split for your dataset.
XLOGP_THRESHOLD = 2.0


def compare_xlogp_groups(data, label):
    """Compare log10 MIC values for lower- and higher-XLogP groups."""
    low_xlogp = data.loc[
        data["XLogP"] <= XLOGP_THRESHOLD, "log10_mic_ug_ml"
    ].dropna()
    high_xlogp = data.loc[
        data["XLogP"] > XLOGP_THRESHOLD, "log10_mic_ug_ml"
    ].dropna()

    print(f"\n--- Welch's t-test: {label} ---")
    print(f"XLogP threshold: {XLOGP_THRESHOLD}")
    print(f"Lower/equal XLogP group: n={len(low_xlogp)}")
    print(f"Higher XLogP group: n={len(high_xlogp)}")

    if len(low_xlogp) < 2 or len(high_xlogp) < 2:
        print("Test not run: each group needs at least two measurements.")
        return

    result = ttest_ind(low_xlogp, high_xlogp, equal_var=False)

    print(f"Mean log10(MIC), lower/equal XLogP: {low_xlogp.mean():.3f}")
    print(f"Mean log10(MIC), higher XLogP: {high_xlogp.mean():.3f}")
    print(f"t-statistic: {result.statistic:.4f}")
    print(f"p-value: {result.pvalue:.6g}")

    if result.pvalue < 0.05:
        print("Conclusion: the group means differ significantly at alpha = 0.05.")
    else:
        print("Conclusion: no statistically significant difference at alpha = 0.05.")

# Non-parametric test
df["logMIC"] = np.log10(df["mic_ug_ml"])
low  = df.loc[df["XLogP"] <= 2, "logMIC"]
high = df.loc[df["XLogP"] > 2,  "logMIC"]

stat, p = mannwhitneyu(low, high, alternative="less")
print(f"n_low={len(low)}, n_high={len(high)}, p={p:.3g}")
print("Median MIC (ug/mL):", 10**low.median(), "vs", 10**high.median())

# Example hypothesis: MIC differs between compounds above and below
# the selected XLogP threshold. Lower MIC indicates stronger activity.
compare_xlogp_groups(df, "All retained MIC measurements")

MIC_CUTOFF_UG_L = 10.0
MIC_CUTOFF_UG_ML = MIC_CUTOFF_UG_L / 1000

low_mic_df = df[df["mic_ug_ml"] < MIC_CUTOFF_UG_ML].copy()
print(
    f"\nMeasurements with MIC < {MIC_CUTOFF_UG_L:g} ug/L "
    f"({MIC_CUTOFF_UG_ML:g} ug/mL): {len(low_mic_df)}"
)
compare_xlogp_groups(low_mic_df, "Subset with MIC below 10 ug/L")