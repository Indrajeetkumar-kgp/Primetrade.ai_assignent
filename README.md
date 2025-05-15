# Primetrade.ai_assignent
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
fear_greed = pd.read_csv('/fear_greed_index (1).csv')
historical_data = pd.read_csv('/historical_data.csv')

# Check columns for each
print("Fear and Greed Index columns:", fear_greed.columns)
print("Historical Data columns:", historical_data.columns)

# Preprocessing
# Rename columns for easier handling
fear_greed = fear_greed.rename(columns={'date': 'Date', 'classification': 'Classification'})
historical_data = historical_data.rename(columns={'Timestamp IST': 'Time', 'Closed PnL': 'ClosedPnL'})

# Convert 'Date' columns to datetime
fear_greed['Date'] = pd.to_datetime(fear_greed['Date'])
historical_data['Time'] = pd.to_datetime(historical_data['Time'], format='%d-%m-%Y %H:%M')  # Specify the correct format


# Extract date only
historical_data['Date'] = historical_data['Time'].dt.date
fear_greed['Date'] = fear_greed['Date'].dt.date

# Merge datasets on Date
merged_df = pd.merge(historical_data, fear_greed[['Date', 'Classification']], on='Date', how='inner')

# Check merge success
print(f"Merged Data shape: {merged_df.shape}")
print(merged_df[['Date', 'Classification', 'ClosedPnL']].head())

# Clean ClosedPnL (remove any NaN or invalid entries if any)
merged_df = merged_df[merged_df['ClosedPnL'].notnull()]
merged_df['ClosedPnL'] = pd.to_numeric(merged_df['ClosedPnL'], errors='coerce')
merged_df = merged_df.dropna(subset=['ClosedPnL'])

# Aggregate ClosedPnL by Classification (Fear/Greed)
agg_pnl = merged_df.groupby('Classification')['ClosedPnL'].agg(['mean', 'median', 'sum', 'count']).reset_index()

print("\n--- Trader Performance by Market Sentiment ---")
print(agg_pnl)

# Visualization
plt.figure(figsize=(8,5))
sns.boxplot(x='Classification', y='ClosedPnL', data=merged_df)
plt.title('Distribution of Trader PnL by Market Sentiment')
plt.xlabel('Market Sentiment')
plt.ylabel('Closed PnL')
plt.grid(True)
plt.show()

# Also, visualize average PnL
plt.figure(figsize=(8,5))
sns.barplot(x='Classification', y='ClosedPnL', data=merged_df, estimator='mean', ci=None)
plt.title('Average Trader PnL by Market Sentiment')
plt.xlabel('Market Sentiment')
plt.ylabel('Average Closed PnL')
plt.grid(True)
plt.show()
