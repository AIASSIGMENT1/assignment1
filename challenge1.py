import pandas as pd
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------
transactions = pd.read_csv("transactions.csv")

scores_2020 = pd.read_csv("scores_20.csv")
scores_2021 = pd.read_csv("scores_21.csv")
scores_2022 = pd.read_csv("scores_22.csv")
scores_2023 = pd.read_csv("scores_23.csv")

budget = pd.read_excel("budget_units.xlsx")
analytics = pd.read_csv("analytics_data.csv")

# --------------------------------------------------------------------
# TRANSACTION DATA CLEANING
# --------------------------------------------------------------------
print("\n--- TRANSACTION DATA CLEANING ---")
print("Shape:", transactions.shape)
print(transactions.info())

# Clean column names (consistent formatting)
transactions.columns = transactions.columns.str.strip().str.lower()

# Convert 'date' column to datetime
transactions['date'] = pd.to_datetime(transactions['date'], errors='coerce')

# Drop rows missing essential columns
transactions = transactions.dropna(subset=['date', 'customer id', 'status'])

# Fill optional fields
if 'operator id' in transactions.columns:
    transactions['operator id'].fillna('unknown', inplace=True)

# Add a year column
transactions['year'] = transactions['date'].dt.year

# Clean text columns
transactions['status'] = transactions['status'].str.strip().str.lower()

# Check for outliers or unusual years
print("\nTransaction years distribution:")
print(transactions['year'].value_counts().sort_index())

# --------------------------------------------------------------------
# SCORES DATA CLEANING
# --------------------------------------------------------------------
print("\n--- SCORES DATA CLEANING ---")
def clean_scores(filepath, year):
    df = pd.read_csv(filepath)
    print(f"\n--- Cleaning {filepath} ---")
    print("Before cleaning:", df.shape)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Check for required columns first (avoids KeyError)
    expected_cols = ['trip', 'organization', 'global_satisfaction']
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column '{col}' in {filepath}")

    # Drop rows missing key data
    df.dropna(subset=expected_cols, inplace=True)

    # Convert numeric columns
    numeric_cols = ['organization', 'global_satisfaction']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)

    # Outlier removal (IQR)
    def remove_outliers_iqr(data, col):
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return data[(data[col] >= lower) & (data[col] <= upper)]

    for col in numeric_cols:
        df = remove_outliers_iqr(df, col)

    # Add year column
    df['year'] = year

    print("After cleaning:", df.shape)
    return df

# Clean each scores dataset separately
scores_2020 = clean_scores("scores_20.csv", 2020)
scores_2021 = clean_scores("scores_21.csv", 2021)
scores_2022 = clean_scores("scores_22.csv", 2022)
scores_2023 = clean_scores("scores_23.csv", 2023)

# --------------------------------------------------------------------
# BUDGET DATA CLEANING
# --------------------------------------------------------------------
print("\n--- BUDGET DATA CLEANING ---")
print("Shape:", budget.shape)
print(budget.info())

# Clean column names
budget.columns = budget.columns.str.strip().str.lower()

# Drop duplicates and missing trip names
budget = budget.drop_duplicates().dropna(subset=['trip'])

# Transform from wide to long format
budget_long = budget.melt(
    id_vars=['trip'],
    var_name='period_year',
    value_name='budget_category'
)

# Drop missing categories and convert to categorical
budget_long = budget_long.dropna(subset=['budget_category'])
budget_long['budget_category'] = budget_long['budget_category'].astype('category')

# Rename for clarity
budget_clean = budget_long.rename(columns={'trip': 'trip_name'})

print("\nFinal Cleaned Budget Data Info:")
print(budget_clean.info())
print(budget_clean.head())

# --------------------------------------------------------------------
# ANALYTICS DATA CLEANING
# --------------------------------------------------------------------
print("\n--- ANALYTICS DATA CLEANING ---")

analytics.columns = analytics.columns.str.strip().str.lower()
print("Shape:", analytics.shape)
print(analytics.info())

# Drop rows missing essential ID
analytics.dropna(subset=['trip_name'], inplace=True)

# Define numeric columns
numeric_cols = ['page_views', 'unique_visitors', 'avg_session_duration', 'bounce_rate', 'conversion_rate']

# Clean and convert percentage & duration columns
for col in ['bounce_rate', 'conversion_rate']:
    if analytics[col].dtype == 'object':
        analytics[col] = analytics[col].str.replace('%', '', regex=False)
if 'avg_session_duration' in analytics.columns:
    analytics['avg_session_duration'] = analytics['avg_session_duration'].str.replace('s', '', regex=False)

# Convert to numeric
analytics[numeric_cols] = analytics[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Drop missing numeric data
analytics.dropna(subset=numeric_cols, inplace=True)

# Outlier removal
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

for col in numeric_cols:
    analytics = remove_outliers_iqr(analytics, col)

# Standardize numeric variables
scaler = StandardScaler()
analytics[numeric_cols] = scaler.fit_transform(analytics[numeric_cols])

print("\nClean Analytics Data Shape:", analytics.shape)
print(analytics.head())
