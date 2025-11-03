import pandas as pd
import glob

# --- 2. Load transactions data ---
# This file contains the sales funnel information (customer status history)
transactions = pd.read_csv("transactions.csv", parse_dates=['Date'])

# --- 3. Load website analytics data ---
# Contains page views, visitors, bounce rate, and conversion rate
analytics = pd.read_csv("analytics_data.csv")

# --- 4. Load trip budget data ---
# Contains budget category (A–D) per trip and year
budget = pd.read_excel("budget_units.xlsx")

# --- 5. Load satisfaction score files for multiple years ---
# Automatically detects all files that start with "scores_"
score_files = glob.glob("scores_*.csv")

# Combine all yearly score files into one DataFrame
scores = pd.concat([pd.read_csv(f) for f in score_files], ignore_index=True)
