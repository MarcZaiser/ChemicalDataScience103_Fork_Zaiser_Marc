# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



# Main analysis function
def main():
    # --- 1. Load data ---
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns = wine.feature_names)
    
    # --- 1.1 Add target cumn ---
    df["target"] = wine.target                                         
    
    # --- 1.2 First output ---
    print("Shape:", df.shape)
    print("Info:", df.info())
    print("First 10 rows:\n", df.head(10)) 
    
    # --- 1.3 Basic summary statistics
    print("\nValue counts:\n", df["wine"].value_counts())
    print("\nMaximum:", df["wine"].max())
    print("Minimum:", df["wine"].min())
    print("\nSummary statistics:\n", df.describe())
    print("\n5 largest rows:\n", df.nlargest(5, "wine"))
    print("\n5 smallest rows:\n", df.nsmallest(5, "wine"))
    print("\nUnique values:", df["wine"].nunique())
    
    
    # --- 2. Basic exploration --- 
    # --- 2.1 Number of samples in each class
    print("Class distribution:\n", df["wine"].value_counts())
    
    # --- 2.2 Check for missing values and duplicates ---
    
    
    
    
main()
    
    
    
  