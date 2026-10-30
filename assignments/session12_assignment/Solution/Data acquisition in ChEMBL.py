import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# 1) Load the ChEMBL CSV
# Replace this with your actual file name/path
df = pd.read_csv("/Users/marczaiser/Desktop/Studium /Masterstudium/2. Semester/DSA103_Advanced Chemical Data Science/ChemicalDataScience103_Fork_Zaiser_Marc/assignments/session12_assignment/Solution/Escerichia Coli.csv", sep=";")


# 2) Select only the needed columns
df = df[
    [
        "Molecule ChEMBL ID",
        "Smiles",
        "Standard Type",
        "Standard Relation",
        "Standard Value",
        "Standard Units",
        "Assay ChEMBL ID",
    ]
].copy()


# 3) Rename columns to cleaner names
df.columns = [
    "molecule_chembl_id",
    "smiles",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "assay_chembl_id",
]


# 4) Drop rows with no SMILES

# Remove rows without a SMILES string because they cannot be used
# for downstream analysis (e.g. molecular weight calculation).
df = df.dropna(subset=["smiles"]).copy()
df["smiles"] = df["smiles"].str.strip()
df = df[df["smiles"] != ""]


# 5) Filter by molecular weight range

def calculate_mw(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.MolWt(mol)

df["mw"] = df["smiles"].apply(calculate_mw)

# Example range: molecular weight between 200 and 400
min_mw = 200
max_mw = 400
df = df[(df["mw"] >= min_mw) & (df["mw"] <= max_mw)].copy()


# 6) Randomly sample a few hundred entries
# Keep it moderate so PubChem queries do not take too long
n_sample = min(300, len(df))
df_sample = df.sample(n=n_sample, random_state=42)


# 7) Export sample DataFrame to CSV
output_file = "/Users/marczaiser/Desktop/Studium /Masterstudium/2. Semester/DSA103_Advanced Chemical Data Science/ChemicalDataScience103_Fork_Zaiser_Marc/assignments/session12_assignment/Solution/chembl_sample_300_mw_200_400.csv"

df_sample.to_csv(output_file, index=False)

print(f"Saved {len(df_sample)} entries to: {output_file}")