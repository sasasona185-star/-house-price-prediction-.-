import os
import json
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
base_dir = r"C:\Users\Mostafa\.gemini\antigravity-ide\scratch\house-price-project"
data_dir = os.path.join(base_dir, "data")
models_dir = os.path.join(base_dir, "models")
notebooks_dir = os.path.join(base_dir, "notebooks")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(notebooks_dir, exist_ok=True)

# Copy/read source csv
source_csv = r"C:\Users\Mostafa\.gemini\antigravity-ide\scratch\house_prices.csv"
target_csv = os.path.join(data_dir, "house_prices.csv")

print(f"Loading data from {source_csv}...")
df_raw = pd.read_csv(source_csv)
print(f"Loaded {df_raw.shape[0]} rows and {df_raw.shape[1]} columns.")

# If target_csv does not exist, save copy
if not os.path.exists(target_csv):
    df_raw.to_csv(target_csv, index=False)
    print(f"Saved dataset to {target_csv}")

# --- Data Cleaning Functions ---
def clean_amount(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if "Call for Price" in val or "Price on Request" in val:
        return np.nan
    
    # Check for Cr
    cr_match = re.search(r"([\d\.]+)\s*Cr", val, re.IGNORECASE)
    if cr_match:
        try:
            return float(cr_match.group(1)) * 10000000.0
        except:
            return np.nan
            
    # Check for Lac / Lakh
    lac_match = re.search(r"([\d\.]+)\s*(?:Lac|Lakh|L)", val, re.IGNORECASE)
    if lac_match:
        try:
            return float(lac_match.group(1)) * 100000.0
        except:
            return np.nan
            
    # Numeric fallback
    clean_num = re.sub(r"[^\d\.]", "", val)
    if clean_num:
        try:
            return float(clean_num)
        except:
            return np.nan
    return np.nan

def clean_area(area_str):
    if pd.isna(area_str):
        return np.nan
    area_str = str(area_str).strip().lower()
    
    # Extract number
    num_match = re.search(r"([\d\.]+)", area_str)
    if not num_match:
        return np.nan
    val = float(num_match.group(1))
    
    if "sqm" in area_str or "sq. meter" in area_str or "sq meter" in area_str:
        return val * 10.764
    elif "sqyrd" in area_str or "sq. yard" in area_str or "sq yard" in area_str:
        return val * 9.0
    elif "acre" in area_str:
        return val * 43560.0
    else: # default sqft
        return val

def clean_floor(floor_str):
    if pd.isna(floor_str):
        return 1 # default floor
    floor_str = str(floor_str).strip().lower()
    if "ground" in floor_str or "lower" in floor_str:
        return 0
    if "basement" in floor_str:
        return -1
    num_match = re.search(r"^(\d+)", floor_str)
    if num_match:
        return int(num_match.group(1))
    return 1

def clean_numeric(val, default=1):
    if pd.isna(val):
        return default
    num_match = re.search(r"(\d+)", str(val))
    if num_match:
        return int(num_match.group(1))
    return default

print("Cleaning data...")
df = df_raw.copy()

# 1. Target Price
df['price_rupees'] = df['Amount(in rupees)'].apply(clean_amount)
df = df.dropna(subset=['price_rupees'])
df = df[df['price_rupees'] > 100000] # filter out unrealistic prices < 1 Lac

# 2. Area
df['carpet_area_sqft'] = df['Carpet Area'].apply(clean_area)
# Fallback to Super Area if Carpet Area missing
df['super_area_sqft'] = df['Super Area'].apply(clean_area)
df['carpet_area_sqft'] = df['carpet_area_sqft'].fillna(df['super_area_sqft'])
df = df.dropna(subset=['carpet_area_sqft'])
df = df[(df['carpet_area_sqft'] >= 200) & (df['carpet_area_sqft'] <= 10000)]

# 3. Floor
df['floor_num'] = df['Floor'].apply(clean_floor)

# 4. Bathrooms & Balconies
df['bathrooms'] = df['Bathroom'].apply(lambda x: clean_numeric(x, default=2))
df['balconies'] = df['Balcony'].apply(lambda x: clean_numeric(x, default=1))

# 5. Location Cleaning
df['location_clean'] = df['location'].astype(str).str.strip().str.lower()
top_locations = df['location_clean'].value_counts().head(50).index.tolist()
df['location_clean'] = df['location_clean'].apply(lambda x: x if x in top_locations else 'other')

# 6. Categorical features
df['furnishing'] = df['Furnishing'].fillna('Unfurnished').astype(str).str.strip()
df['transaction'] = df['Transaction'].fillna('Resale').astype(str).str.strip()

# 7. Outlier Removal on Price Per Sqft
df['price_per_sqft'] = df['price_rupees'] / df['carpet_area_sqft']
q_low = df['price_per_sqft'].quantile(0.01)
q_high = df['price_per_sqft'].quantile(0.99)
df = df[(df['price_per_sqft'] >= q_low) & (df['price_per_sqft'] <= q_high)]

print(f"Cleaned dataset shape: {df.shape}")

# Features & Target
feature_cols = ['location_clean', 'carpet_area_sqft', 'floor_num', 'furnishing', 'transaction', 'bathrooms', 'balconies']
X = df[feature_cols].rename(columns={'location_clean': 'location'})
y = df['price_rupees']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ColumnTransformer
numeric_features = ['carpet_area_sqft', 'floor_num', 'bathrooms', 'balconies']
categorical_features = ['location', 'furnishing', 'transaction']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ]
)

# Train on log1p(y)
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Model 1: Linear Regression Pipeline
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
print("Training Linear Regression baseline...")
lr_pipeline.fit(X_train, y_train_log)
lr_preds_log = lr_pipeline.predict(X_test)
lr_preds = np.expm1(lr_preds_log)

lr_mae = mean_absolute_error(y_test, lr_preds)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_r2 = r2_score(y_test, lr_preds)
print(f"Linear Regression - MAE: {lr_mae:,.2f}, RMSE: {lr_rmse:,.2f}, R2: {lr_r2:.4f}")

# Model 2: Random Forest Pipeline
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1))
])
print("Training Random Forest Regressor...")
rf_pipeline.fit(X_train, y_train_log)
rf_preds_log = rf_pipeline.predict(X_test)
rf_preds = np.expm1(rf_preds_log)

rf_mae = mean_absolute_error(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_r2 = r2_score(y_test, rf_preds)
print(f"Random Forest - MAE: {rf_mae:,.2f}, RMSE: {rf_rmse:,.2f}, R2: {rf_r2:.4f}")

# Best Model
best_pipeline = rf_pipeline if rf_r2 > lr_r2 else lr_pipeline

# Export Model
model_path = os.path.join(models_dir, "house_price.pkl")
joblib.dump(best_pipeline, model_path)
print(f"Saved model pipeline to {model_path}")

# Export Locations
locations_list = sorted(list(set(top_locations + ['other'])))
locations_path = os.path.join(models_dir, "locations.json")
with open(locations_path, 'w', encoding='utf-8') as f:
    json.dump(locations_list, f, indent=2, ensure_ascii=False)
print(f"Saved {len(locations_list)} locations to {locations_path}")

# Summary Metrics dictionary for reporting
metrics = {
    "LinearRegression": {"MAE": float(lr_mae), "RMSE": float(lr_rmse), "R2": float(lr_r2)},
    "RandomForest": {"MAE": float(rf_mae), "RMSE": float(rf_rmse), "R2": float(rf_r2)},
    "BestModel": "RandomForestRegressor" if rf_r2 > lr_r2 else "LinearRegression"
}
with open(os.path.join(models_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("Model training & artifact exports complete!")
