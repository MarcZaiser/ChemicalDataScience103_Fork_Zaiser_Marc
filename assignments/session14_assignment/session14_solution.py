"""Reusable RDKit feature generation and random-forest comparison for Session 14."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, PandasTools, rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


DATA_DIR = Path(__file__).parent / "data"
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
MOTIF_SMARTS = {
    "benzene_ring": "c1ccccc1",
    "ketone": "[CX3](=O)[#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "amine": "[NX3;!$(NC=O)]",
    "halogen": "[F,Cl,Br,I]",
    "alkene": "[CX3]=[CX3]",
    "ether": "[OD2]([#6])[#6]",
    "four_carbon_path": "[C;!a]~[C;!a]~[C;!a]~[C;!a]",
    "ester": "[CX3](=O)[OX2H0][#6]",
}
MOTIF_QUERIES = {
    name: Chem.MolFromSmarts(smarts) for name, smarts in MOTIF_SMARTS.items()
}

if any(query is None for query in MOTIF_QUERIES.values()):
    raise ValueError("At least one custom fingerprint SMARTS pattern is invalid.")


def custom_fingerprint(mol: Chem.Mol) -> list[int]:
    """Return one binary feature for each SMARTS motif in MOTIF_SMARTS."""
    if mol is None:
        raise ValueError("Cannot fingerprint an invalid or missing molecule.")
    return [
        int(mol.HasSubstructMatch(query))
        for query in MOTIF_QUERIES.values()
        if query is not None
    ]


def load_dataset(
    path: Path,
    smiles_column: str,
    target_column: str,
    target_transform: str | None = None,
) -> pd.DataFrame:
    """Load, clean, and add RDKit molecule objects to one dataset."""
    frame = pd.read_csv(path)
    required_columns = {smiles_column, target_column}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")

    frame = frame.dropna(subset=[smiles_column, target_column]).copy()
    frame[smiles_column] = frame[smiles_column].astype(str).str.strip()
    frame = frame.loc[frame[smiles_column].ne("")].copy()
    frame["target"] = pd.to_numeric(frame[target_column], errors="coerce")
    frame = frame.dropna(subset=["target"]).copy()

    if target_transform == "log10":
        frame = frame.loc[frame["target"] > 0].copy()
        frame["target"] = np.log10(frame["target"])
    elif target_transform is not None:
        raise ValueError(f"Unsupported target transform: {target_transform!r}")

    PandasTools.AddMoleculeColumnToFrame(
        frame,
        smilesCol=smiles_column,
        molCol="ROMol",
        includeFingerprints=False,
    )
    invalid_molecules = frame["ROMol"].isna()
    if invalid_molecules.any():
        warnings.warn(
            f"Dropping {int(invalid_molecules.sum())} rows with invalid SMILES from {path}.",
            stacklevel=2,
        )
        frame = frame.loc[~invalid_molecules].copy()

    return frame.reset_index(drop=True)


def add_molecular_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add custom and Morgan fingerprints and return a descriptor table."""
    frame = frame.copy()
    molecules = frame["ROMol"]
    frame["CustomFP"] = molecules.map(custom_fingerprint)
    frame["MorganFP"] = molecules.map(MORGAN_GENERATOR.GetFingerprintAsNumPy)

    descriptor_rows: list[dict[str, Any]] = [
        Descriptors.CalcMolDescriptors(mol) for mol in molecules
    ]
    descriptor_frame = pd.DataFrame(descriptor_rows, index=frame.index)
    descriptor_frame = descriptor_frame.replace([np.inf, -np.inf], np.nan)
    return frame, descriptor_frame


def _feature_matrix(frame: pd.DataFrame, descriptors: pd.DataFrame, name: str) -> np.ndarray:
    """Convert a dataframe feature set into a two-dimensional numeric matrix."""
    if name == "CustomFP":
        return np.asarray(frame["CustomFP"].tolist(), dtype=np.uint8)
    if name == "MorganFP":
        return np.vstack(frame["MorganFP"].to_numpy())
    if name == "Descriptors":
        return descriptors.to_numpy(dtype=float)
    raise ValueError(f"Unknown feature set: {name}")


def compare_feature_sets(
    frame: pd.DataFrame,
    descriptors: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
) -> pd.DataFrame:
    """Fit the same random-forest setup to each feature set and report metrics."""
    y = frame["target"].to_numpy(dtype=float)
    indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
    )
    results: list[dict[str, float | str]] = []

    for feature_name in ("CustomFP", "MorganFP", "Descriptors"):
        features = _feature_matrix(frame, descriptors, feature_name)
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=n_estimators, random_state=random_state),
        )
        model.fit(features[train_indices], y[train_indices])
        train_predictions = model.predict(features[train_indices])
        test_predictions = model.predict(features[test_indices])

        results.append(
            {
                "Feature set": feature_name,
                "Train R2": r2_score(y[train_indices], train_predictions),
                "Test R2": r2_score(y[test_indices], test_predictions),
                "Test RMSE": np.sqrt(mean_squared_error(y[test_indices], test_predictions)),
                "Test MAE": mean_absolute_error(y[test_indices], test_predictions),
            }
        )

    return pd.DataFrame(results).set_index("Feature set")


def display_structure_preview(frame: pd.DataFrame, rows: int = 10) -> None:
    """Display the RDKit molecule column as images in a Jupyter notebook."""
    from IPython.display import display

    display(frame.head(rows))


def run_dataset(
    path: Path,
    smiles_column: str,
    target_column: str,
    target_transform: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare a dataset and return its rows, descriptors, and model metrics."""
    frame = load_dataset(path, smiles_column, target_column, target_transform)
    frame, descriptors = add_molecular_features(frame)
    metrics = compare_feature_sets(frame, descriptors)
    return frame, descriptors, metrics


def main() -> None:
    """Run the feature comparison for both assignment datasets."""
    datasets = (
        ("Delaney solubility", DATA_DIR / "delaney.csv", "logS", None),
        ("E. coli MIC (log10)", DATA_DIR / "ecoli_mics_sample.csv", "mic", "log10"),
    )
    for label, path, target_column, transform in datasets:
        frame, descriptors, metrics = run_dataset(
            path,
            smiles_column="smiles",
            target_column=target_column,
            target_transform=transform,
        )
        print(f"\n{label}: {len(frame)} usable molecules")
        print(f"Descriptor matrix: {descriptors.shape[0]} rows x {descriptors.shape[1]} columns")
        print(metrics.round(3).to_string())


if __name__ == "__main__":
    main()
