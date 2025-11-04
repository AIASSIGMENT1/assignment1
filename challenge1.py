import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Load transactions data
transactions = pd.read_csv("transactions.csv", parse_dates=['Date'])

# Load website analytics data
analytics = pd.read_csv("analytics_data.csv")

# Load trip budget data
budget = pd.read_excel("budget_units.xlsx")

# Load satisfaction score files for multiple years
# Automatically detects all files that start with "scores_"
score_files = glob.glob("scores_*.csv")

# Combine all yearly score files into one DataFrame
scores = pd.concat([pd.read_csv(f) for f in score_files], ignore_index=True)

# --- DATA CLEANING ---
# --- OUTLIER REMOVAL ---
# We delete values outside the range of 1st and 99th percentile.


# --- DATA EXPLORATION ---


# --- FEATURE PREPARATION ---


# --- CONCLUSION --- (poner tablas y comentarlas...)
