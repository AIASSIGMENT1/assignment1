import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
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
# Clean column names (consistent formatting)
transactions.columns = transactions.columns.str.strip().str.lower()

# Convert 'date' column to datetime
transactions['date'] = pd.to_datetime(transactions['date'], errors='coerce')

# Drop rows missing essential columns
transactions = transactions.dropna(subset=['date', 'customer id', 'status'])

# Rename columns for consistency
transactions.rename(columns={'customer id':'customer_id','operator id':'operator_id'}, inplace=True)

# then fillna with assignment
if 'operator_id' in transactions.columns:
    transactions['operator_id'] = transactions['operator_id'].fillna('unknown')

# Add a year column
transactions['year'] = transactions['date'].dt.year

# Clean text columns
transactions['status'] = transactions['status'].str.strip().str.lower()

 #Standardize status values to a known set and check unexpected values
transactions['status'] = transactions['status'].str.strip().str.lower().replace({
    'paid diposit': 'paid deposit'
})

allowed = {'filled in form','not reachable','paid deposit','sales','cancelled'}
unexpected = set(transactions['status'].unique()) - allowed
if unexpected:
    print("Unexpected status values:", unexpected)


def summarize_transactions(df):
    print("\nTRANSACTIONS (clean)")
    print(f"rows={len(df)}, cols={len(df.columns)}")
    print("date range:", df['date'].min().date(), "→", df['date'].max().date())
    print("status counts (top):")
    print(df['status'].value_counts().to_string())
    print("year counts:")
    print(df['year'].value_counts().sort_index().to_string())

# --------------------------------------------------------------------
# SCORES DATA CLEANING
# --------------------------------------------------------------------
def clean_scores(filepath, year):
    df = pd.read_csv(filepath)
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
    return df

    
# Clean each scores dataset separately
scores_2020 = clean_scores("scores_20.csv", 2020)
scores_2021 = clean_scores("scores_21.csv", 2021)
scores_2022 = clean_scores("scores_22.csv", 2022)
scores_2023 = clean_scores("scores_23.csv", 2023)

#Combine the yearly scores
scores_all = pd.concat(
    [scores_2020, scores_2021, scores_2022, scores_2023],
    ignore_index=True
)
def summarize_scores(df_all):
    print("\nSCORES (clean, all years)")
    print(f"rows={len(df_all)}, cols={len(df_all.columns)}")
    print("rows per year:")
    print(df_all['year'].value_counts().sort_index().to_string())
    # estadísticas básicas de satisfacción
    stats = df_all[['organization','global_satisfaction']].describe().loc[['mean','std','min','max']]
    print("\nsatisfaction stats (org & global):")
    print(stats.to_string())


# --------------------------------------------------------------------
# BUDGET DATA CLEANING
# --------------------------------------------------------------------
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

# to convert 'period_20' → 2020, 'period_21' → 2021, etc.
budget_long['period_year'] = (
    budget_long['period_year']
    .str.extract(r'(\d+)$')     
    .astype(int) + 2000         
)

# Drop missing categories and convert to categorical
budget_long = budget_long.dropna(subset=['budget_category'])
budget_long['budget_category'] = budget_long['budget_category'].astype('category')

# Rename for clarity
budget_clean = budget_long.rename(columns={'trip': 'trip_name'})

#Ensure budget letter consistency and ordering
budget_clean['budget_category'] = (
    budget_clean['budget_category'].astype(str).str.upper().str.strip()
)
budget_clean['budget_category'] = pd.Categorical(
    budget_clean['budget_category'], categories=list("ABCD"), ordered=True
)
def summarize_budget(df):
    print("\nBUDGET (clean, long format)")
    print(f"rows={len(df)}, cols={len(df.columns)}")
    print("year range:", int(df['period_year'].min()), "→", int(df['period_year'].max()))
    print("budget_category counts (A–D):")
    print(df['budget_category'].value_counts().to_string())



# --------------------------------------------------------------------
# ANALYTICS DATA CLEANING
# --------------------------------------------------------------------
analytics.columns = analytics.columns.str.strip().str.lower()

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

# Convert percent columns to proportions (so 1.5% → 0.015)
analytics['bounce_rate'] = analytics['bounce_rate'] / 100.0
analytics['conversion_rate'] = analytics['conversion_rate'] / 100.0

# Outlier imputation with median
def impute_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    median_val = df[col].median()
    df[col] = df[col].mask((df[col] < lower) | (df[col] > upper), median_val)
    return df

for col in numeric_cols:
    analytics = impute_outliers_iqr(analytics, col)


# Standardize numeric variables (create z-score columns without overwriting originals)
scaled = analytics[numeric_cols].apply(lambda s: (s - s.mean()) / s.std(ddof=0))
analytics[[c + '_z' for c in numeric_cols]] = scaled

def summarize_analytics(df):
    print("\nANALYTICS (clean)")
    print(f"rows={len(df)}, cols={len(df.columns)}")
    # columnas originales clave
    base_cols = ['page_views','unique_visitors','avg_session_duration','bounce_rate','conversion_rate']
    existing = [c for c in base_cols if c in df.columns]
    print("columns:", existing, "+ z-scores added")
    # percentiles compactos
    desc = df[existing].quantile([0,0.5,1]).rename(index={0:'min',0.5:'median',1:'max'})
    print(desc.to_string())

# --------------------------------------------------------------------
# BEFORE MERGING (APPLIES TO ALL DATAFRAMES)
# --------------------------------------------------------------------
# Normalize trip name columns for merging
def norm_trip_cols(df):
    if 'trip' in df.columns:
        df['trip'] = df['trip'].str.strip().str.lower()
    if 'trip_name' in df.columns:
        df['trip_name'] = df['trip_name'].str.strip().str.lower()
    return df

analytics = norm_trip_cols(analytics)
scores_all = norm_trip_cols(scores_all)
budget_clean = norm_trip_cols(budget_clean)

summarize_transactions(transactions)
summarize_scores(scores_all)
summarize_budget(budget_clean)
summarize_analytics(analytics)


# --------------------------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# --------------------------------------------------------------------
plt.style.use('seaborn-v0_8') #makes plots look nicer and more clear.

#(a) Sales trend. It counts how many unique customers actually bought something each year and then plots a line chart to show if sales went up or down over time.
#Count the number of completed sales per year: (shows volume--> how many sales each year.)
sales_trend = (
    transactions[transactions['status'] == 'sales']
    .groupby('year')['customer_id']
    .nunique()
    .reset_index(name='num_sales')
)

print(sales_trend)

sales_trend.plot(x='year', y='num_sales', kind='line', marker='o',
                 title='Yearly Sales Trend', ylabel='Number of Sales')
plt.show() #This shows whether sales are increasing or decreasing over time.

#(b) Funnel conversion: Shows what % of leads who filled the form actually made a purchase — your conversion rate.
transactions_summary = (
    transactions.groupby(['year', 'status'])['customer_id']
    .nunique()
    .unstack(fill_value=0)
)

transactions_summary['conversion_rate'] = (
    transactions_summary['sales'] / transactions_summary['filled in form']
)

print(transactions_summary)

transactions_summary['conversion_rate'].plot(kind='bar', title='Conversion Rate by Year')
plt.show()


#a/b. 1 Quantitative evidence: year-over-year % changes
# Year-over-year % change in sales
sales_trend['pct_change_sales'] = sales_trend['num_sales'].pct_change() * 100
print("\nYear-over-year % change in sales:")
print(sales_trend[['year', 'num_sales', 'pct_change_sales']])

# Year-over-year % change in conversion rate
transactions_summary['conversion_rate_pct_change'] = (
    transactions_summary['conversion_rate'].pct_change() * 100
)
print("\nYear-over-year % change in conversion rate:")
print(transactions_summary[['conversion_rate', 'conversion_rate_pct_change']])

# Optional visual: bar chart of % sales change
sales_trend.plot(x='year', y='pct_change_sales', kind='bar',
                 title='% Change in Sales per Year', ylabel='% Change from Previous Year')
plt.show()




#(c) Satisfaction trends: Are customers happier or less satisfied compared to previous years?
satisfaction_trend = (
    scores_all.groupby('year')[['organization', 'global_satisfaction']].mean()
)

print(satisfaction_trend)

satisfaction_trend.plot(y=['organization', 'global_satisfaction'],
                        marker='o', title='Average Satisfaction by Year')
plt.ylabel('Satisfaction Score (1–10)')
plt.show()

#(d) Budget distribution: Shows the mix of trip price levels (A = luxury, D = budget).
latest_budget = (
    budget_clean.sort_values('period_year')
    .drop_duplicates(subset='trip_name', keep='last')
)

latest_budget['budget_category'].value_counts().plot(kind='bar',
                                                     title='Trips by Budget Category (Latest Year)')
plt.ylabel('Number of Trips')
plt.show()


#(e) Web analytics overview: Helps see which trips attract more visitors or have better conversion.
print(analytics.describe()[['page_views','unique_visitors','bounce_rate','conversion_rate']])

analytics[['page_views','unique_visitors','bounce_rate','conversion_rate']].hist(figsize=(10,6))
plt.suptitle('Distribution of Website Metrics', fontsize=14)
plt.show()

# --------------------------------------------------------------------
# BUILD YEARLY PANEL FOR CORRELATIONS
# --------------------------------------------------------------------
yearly_funnel = (
    transactions.groupby(['year', 'status'])['customer_id']
    .nunique()
    .unstack(fill_value=0)
    .rename_axis(None, axis=1)
    .reset_index()
)

# Ensure consistent column names
yearly_funnel = yearly_funnel.rename(columns={'filled in form': 'leads'})
yearly_funnel['conversion_rate'] = yearly_funnel['sales'] / yearly_funnel['leads']

yearly_satisfaction = (
    scores_all.groupby('year')[['organization', 'global_satisfaction']]
    .mean()
    .reset_index()
)

# Merge everything into a single panel
yearly = yearly_funnel.merge(yearly_satisfaction, on='year', how='left')

# --------------------------------------------------------------------
# CORRELATIONS
# --------------------------------------------------------------------
corr_to_sales = yearly[['sales', 'leads', 'conversion_rate',
                        'organization', 'global_satisfaction']].corr()['sales']

print("\nPearson correlations with SALES:")
print(corr_to_sales)

# --------------------------------------------------------------------
# SCATTERPLOTS WITH TREND LINES
# --------------------------------------------------------------------

def scatter_with_trend(x, y, xlab, ylab, title):
    # Build a clean (x,y) DataFrame
    s = pd.DataFrame({'x': x, 'y': y}).replace([np.inf, -np.inf], np.nan).dropna()

    # Always show the scatter (even if we can't fit a line)
    plt.figure()
    plt.scatter(s['x'], s['y'])
    plt.title(title)
    plt.xlabel(xlab); plt.ylabel(ylab)

    # Compute r on the cleaned data
    r = s['x'].corr(s['y'])
    if pd.notna(r):
        plt.text(0.02, 0.95, f"r = {r:.2f}", transform=plt.gca().transAxes)

    # Only fit a line if we have >= 2 points and x varies
    if len(s) >= 2 and s['x'].nunique() >= 2:
        m, b = np.polyfit(s['x'].to_numpy(), s['y'].to_numpy(), 1)
        xs = np.linspace(s['x'].min(), s['x'].max(), 100)
        plt.plot(xs, m*xs + b)
    else:
        plt.text(0.02, 0.90, "(not enough data to fit a line)", transform=plt.gca().transAxes)

    plt.show()

# Plots
scatter_with_trend(yearly['leads'], yearly['sales'],
                   "Leads", "Sales", "Sales vs Leads")

scatter_with_trend(yearly['conversion_rate'], yearly['sales'],
                   "Conversion rate", "Sales", "Sales vs Conversion rate")

scatter_with_trend(yearly['global_satisfaction'], yearly['sales'],
                   "Global satisfaction", "Sales", "Sales vs Satisfaction")

# --------------------------------------------------------------------
# RELATE DATASETS (to explain sales changes)
# --------------------------------------------------------------------
#(a) Merge satisfaction with budget
# Here you can check whether high-budget trips get lower satisfaction.
scores_budget = pd.merge(
    scores_all, budget_clean,
    left_on=['trip', 'year'], right_on=['trip_name', 'period_year'],
    how='left'
)

avg_sat_by_budget = (
    scores_budget.groupby('budget_category')[['organization', 'global_satisfaction']]
    .mean()
)
print(avg_sat_by_budget)
avg_sat_by_budget.plot(kind='bar', title='Average Satisfaction by Budget Category')
plt.ylabel('Satisfaction Score')
plt.show() #Does trip satisfaction vary by budget? See if luxury trips (A/B) get higher or lower satisfaction than budget ones (C/D).

#(b) Merge analytics with satisfaction or budget
# Helps you see if bounce rate or conversion rate correlate with trip cost or satisfaction.
analytics_budget = pd.merge(
    analytics, budget_clean,
    on='trip_name', how='left'
)

analytics_budget.groupby('budget_category')[['conversion_rate','bounce_rate']].mean().plot(
    kind='bar', title='Web Metrics by Budget Category'
)
plt.show() #Does website performance differ by budget? Check if cheaper or premium trips have better engagement online.
