import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from pathlib import Path
from pathlib import Path

BASE = Path(__file__).resolve().parent
print("BASE:", BASE)

# --- transactions (funnel) ---
transactions_path = BASE / "transactions.csv"
transactions = pd.read_csv(transactions_path,
                           parse_dates=[c for c in ["Date","date"] if c in pd.read_csv(transactions_path, nrows=0).columns])
# Normalizo nombre de fecha
transactions.rename(columns={"date":"Date"}, inplace=True)

# --- analytics ---
analytics_path = BASE / "analytics_data.csv"
analytics = pd.read_csv(analytics_path)

# --- budget ---
budget_path = BASE / "budget_units.xlsx"
try:
    budget = pd.read_excel(budget_path, engine="openpyxl")
except Exception as e:
    raise RuntimeError(f"No pude leer {budget_path}: {e}")

# --- scores (varios años) ---
score_files = sorted((BASE).glob("scores_*.csv"))
if not score_files:
    raise FileNotFoundError("No encontré archivos 'scores_*.csv' en la carpeta del script.")
scores = pd.concat([pd.read_csv(f) for f in score_files], ignore_index=True)

# ---- prints útiles ----
print("\nCARGA OK")
print("transactions:", transactions.shape, "— columnas:", list(transactions.columns))
print("analytics   :", analytics.shape, "— columnas:", list(analytics.columns))
print("budget      :", budget.shape, "— columnas:", list(budget.columns))
print("scores      :", scores.shape, "— columnas:", list(scores.columns))

print("\nRango de fechas en transactions:", transactions["Date"].min(), "→", transactions["Date"].max())
print("\nPREVIEW")
print("transactions.head():\n", transactions.head().to_string(index=False))
print("\nanalytics.head():\n", analytics.head().to_string(index=False))
print("\nbudget.head():\n", budget.head().to_string(index=False))
print("\nscores.head():\n", scores.head().to_string(index=False))



# --- DATA CLEANING ---
def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df

transactions = clean_columns(transactions)
analytics = clean_columns(analytics)
budget = clean_columns(budget)
scores = clean_columns(scores)


#nulos y duplicados
for name, df in zip(
    ["transactions", "analytics", "budget", "scores"],
    [transactions, analytics, budget, scores]
):
    print(f"\n{name.upper()} nulls:")
    print(df.isna().sum())
#eliminar duplicados
# eliminar duplicados
transactions = transactions.drop_duplicates()
analytics = analytics.drop_duplicates()
budget = budget.drop_duplicates()
scores = scores.drop_duplicates()

# rellenar valores faltantes si hace falta
scores = scores.fillna({'organization': scores['organization'].median(),
                        'global_satisfaction': scores['global_satisfaction'].median()})
budget = budget.fillna('unknown')


#asegurar que las fechas y categorías estén en el formato correcto (datetime y category)
transactions['date'] = pd.to_datetime(transactions['date'], errors='coerce')
transactions['status'] = transactions['status'].astype('category')
transactions['operator_id'] = transactions['operator_id'].astype('category')


#estandarizar nombres de trips (quitar espacios, pasar a minúsculas)
for df in [analytics, budget, scores]:
    if 'trip' in df.columns:
        df['trip'] = df['trip'].str.strip().str.lower()
    if 'trip_name' in df.columns:
        df['trip_name'] = df['trip_name'].str.strip().str.lower()

# renombramos para tener todos igual
budget.rename(columns={'trip': 'trip_name'}, inplace=True)
scores.rename(columns={'trip': 'trip_name'}, inplace=True)


# --- OUTLIER REMOVAL ---
# detectar outliers y limpiar valores extremos.
def cap_outliers(df, cols, low=0.01, high=0.99):
    for c in cols:
        q_low, q_high = df[c].quantile([low, high])
        df[c] = df[c].clip(q_low, q_high)
    return df

# aplicamos a métricas numéricas
analytics = cap_outliers(analytics, ['page_views', 'unique_visitors'])
scores = cap_outliers(scores, ['organization', 'global_satisfaction'])


# --- DATA EXPLORATION ---


# --- FEATURE PREPARATION ---


# --- CONCLUSION --- (poner tablas y comentarlas...)
