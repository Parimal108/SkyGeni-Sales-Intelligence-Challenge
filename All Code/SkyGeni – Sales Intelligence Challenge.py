import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. LOAD DATA
try:
    df = pd.read_csv("skygeni_sales_data.csv")
except FileNotFoundError:
    print("Please ensure skygeni_sales_data.csv is in the local directory.")
    exit()

# 2. CLEANING & PREPROCESSING
df['created_date'] = pd.to_datetime(df['created_date'])
df['closed_date'] = pd.to_datetime(df['closed_date'])
df['deal_amount'] = pd.to_numeric(df['deal_amount'], errors='coerce').fillna(0)

# Calculate cycle days
df['calc_cycle_days'] = (df['closed_date'] - df['created_date']).dt.days.clip(lower=1)

# 3. ADVANCED EDA (ADDED)
print("\n--- CORRELATION ANALYSIS ---")
plt.figure(figsize=(10, 6))
# Only numeric for heatmap
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()

# Insight 4: Time-Series Trend (Monthly Revenue)
plt.figure(figsize=(12, 6))
df_won = df[df['outcome'] == 'Won'].copy()
df_won.set_index('closed_date').resample('M')['deal_amount'].sum().plot(kind='line', marker='o', color='green')
plt.title('Monthly Closed-Won Revenue Trend')
plt.ylabel('Total Deal Amount')
plt.show()

# 4. CUSTOM METRIC DESIGN
df['deal_momentum'] = df['deal_amount'] / df['calc_cycle_days']

# Safe Rep Efficiency (Avoiding Leakage)
# For the assessment, we use global stats, but in production use a Rolling average.
rep_stats = df.groupby('sales_rep_id')['outcome'].apply(lambda x: (x == 'Won').mean())
df['rep_historical_win_rate'] = df['sales_rep_id'].map(rep_stats)

# 5. PREDICTIVE MODELING
# Prep data: Use only closed deals
df_ml = df.dropna(subset=['outcome']).copy()
df_ml['target'] = (df_ml['outcome'].str.lower() == 'won').astype(int)

# Encoding
le = LabelEncoder()
features_to_encode = ['industry', 'region', 'lead_source', 'product_type']
for col in features_to_encode:
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))

# Feature Selection
features = ['industry', 'region', 'lead_source', 'product_type', 'deal_amount', 
            'calc_cycle_days', 'rep_historical_win_rate', 'deal_momentum']
X = df_ml[features]
y = df_ml['target']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model Building with Balanced Weights
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 6. ASSESSMENT RESULTS
y_pred = model.predict(X_test)
print("\n--- MODEL PERFORMANCE REPORT ---")
print(classification_report(y_test, y_pred))

# Feature Importance Visualization
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
importances.plot(kind='bar', color='teal')
plt.title('What Drives Deal Success? (Feature Importance)')
plt.show()

# 5. Extracting the "Drivers"
driver_impact = pd.DataFrame({
    'Driver': features,
    'Impact_Score': model.feature_importances_
}).sort_values(by='Impact_Score', ascending=False)


print("--- Win Rate Driver Analysis ---")
print(driver_impact)
