import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import warnings

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('car data.csv')

print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Exploratory Data Analysis
print("\n=== EXPLORATORY DATA ANALYSIS ===")

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check unique values for categorical columns
print("\nUnique Values in Categorical Columns:")
categorical_cols = ['Car_Name', 'Fuel_Type', 'Selling_type', 'Transmission', 'Owner']
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique values")

# Data Visualization
plt.figure(figsize=(15, 10))

# 1. Distribution of Selling Price
plt.subplot(2, 3, 1)
plt.hist(df['Selling_Price'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
plt.title('Distribution of Selling Price')
plt.xlabel('Selling Price (Lakhs)')
plt.ylabel('Frequency')

# 2. Distribution of Present Price
plt.subplot(2, 3, 2)
plt.hist(df['Present_Price'], bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
plt.title('Distribution of Present Price')
plt.xlabel('Present Price (Lakhs)')
plt.ylabel('Frequency')

# 3. Fuel Type Distribution
plt.subplot(2, 3, 3)
df['Fuel_Type'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Fuel Type Distribution')
plt.xlabel('Fuel Type')
plt.ylabel('Count')

# 4. Transmission Type Distribution
plt.subplot(2, 3, 4)
df['Transmission'].value_counts().plot(kind='bar', color='gold')
plt.title('Transmission Type Distribution')
plt.xlabel('Transmission')
plt.ylabel('Count')

# 5. Year vs Selling Price
plt.subplot(2, 3, 5)
plt.scatter(df['Year'], df['Selling_Price'], alpha=0.6, color='purple')
plt.title('Year vs Selling Price')
plt.xlabel('Year')
plt.ylabel('Selling Price')

# 6. Driven Kilometers vs Selling Price
plt.subplot(2, 3, 6)
plt.scatter(df['Driven_kms'], df['Selling_Price'], alpha=0.6, color='orange')
plt.title('Driven Kilometers vs Selling Price')
plt.xlabel('Kilometers')
plt.ylabel('Selling Price')

plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

print("\n=== DATA PREPROCESSING AND FEATURE ENGINEERING ===")

# Create a copy of the original dataframe
df_processed = df.copy()

# Feature Engineering
# 1. Car Age (instead of Year)
current_year = 2024
df_processed['Car_Age'] = current_year - df_processed['Year']

# 2. Brand extraction from Car_Name
df_processed['Brand'] = df_processed['Car_Name'].apply(lambda x: x.split()[0].lower())

# 3. Price depreciation ratio (handle division by zero)
df_processed['Depreciation_Ratio'] = df_processed['Selling_Price'] / np.where(
    df_processed['Present_Price'] == 0, 1, df_processed['Present_Price']
)

# 4. Kilometers per year (handle division by zero)
df_processed['Km_Per_Year'] = df_processed['Driven_kms'] / np.where(
    df_processed['Car_Age'] == 0, 1, df_processed['Car_Age']
)

# 5. Log transformation for skewed features
df_processed['Driven_kms_log'] = np.log1p(df_processed['Driven_kms'])
df_processed['Present_Price_log'] = np.log1p(df_processed['Present_Price'])

# 6. Create car categories based on brand
brand_categories = {
    'premium': ['fortuner', 'innova', 'corolla', 'camry', 'land', 'elantra', 'creta', 'city'],
    'mid_range': ['ciaz', 'sx4', 'ertiga', 'verna', 'i20', 'jazz', 'amaze', 'brio', 'baleno'],
    'economy': ['ritz', 'wagon', 'swift', 'alto', 'i10', 'eon', 'bajaj', 'hero', 'honda', 'yamaha', 'tvs', 'royal']
}


def categorize_brand(brand):
    brand_lower = brand.lower()
    for category, brands in brand_categories.items():
        for b in brands:
            if b in brand_lower:
                return category
    return 'economy'


df_processed['Brand_Category'] = df_processed['Brand'].apply(categorize_brand)

print("Feature Engineering Completed!")
print(
    "New features created: Car_Age, Brand, Depreciation_Ratio, Km_Per_Year, Driven_kms_log, Present_Price_log, Brand_Category")


# Remove outliers based on IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


# Remove outliers from key numerical columns
original_size = len(df_processed)
df_processed = remove_outliers(df_processed, 'Selling_Price')
df_processed = remove_outliers(df_processed, 'Present_Price')
df_processed = remove_outliers(df_processed, 'Driven_kms')
print(f"Removed {original_size - len(df_processed)} outliers")

# Encode categorical variables
label_encoders = {}
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission', 'Brand_Category']

for feature in categorical_features:
    le = LabelEncoder()
    df_processed[feature + '_encoded'] = le.fit_transform(df_processed[feature])
    label_encoders[feature] = le

print("\nCategorical variables encoded successfully!")

# Select features for modeling
feature_columns = [
    'Present_Price_log', 'Driven_kms_log', 'Owner', 'Car_Age',
    'Depreciation_Ratio', 'Km_Per_Year',
    'Fuel_Type_encoded', 'Selling_type_encoded', 'Transmission_encoded',
    'Brand_Category_encoded'
]

X = df_processed[feature_columns]
y = df_processed['Selling_Price']

print(f"Feature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")

# Feature Selection
selector = SelectKBest(score_func=f_regression, k='all')
X_selected = selector.fit_transform(X, y)

# Get feature scores
feature_scores = pd.DataFrame({
    'Feature': feature_columns,
    'Score': selector.scores_
}).sort_values('Score', ascending=False)

print("\nFeature Importance Scores:")
print(feature_scores)

# Select top 6 features
top_features = feature_scores.head(6)['Feature'].values
X_selected = X[top_features]

print(f"\nSelected Top 6 Features: {list(top_features)}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== MODEL TRAINING AND EVALUATION ===")

# Initialize models with better parameters
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.01, max_iter=1000),
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        min_samples_split=5
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
}

# Train and evaluate models
results = {}

for name, model in models.items():
    # Use scaled features for linear models, original for tree-based
    if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R2 Score': r2,
        'Predictions': y_pred,
        'Model': model
    }

    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R2 Score: {r2:.4f}")

# Find the best model
best_model_name = max(results.keys(), key=lambda x: results[x]['R2 Score'])
best_result = results[best_model_name]

print(f"\n=== BEST MODEL: {best_model_name} ===")
print(f"R2 Score: {best_result['R2 Score']:.4f}")
print(f"RMSE: {best_result['RMSE']:.4f}")
print(f"MAE: {best_result['MAE']:.4f}")

# Cross-validation for the best model
best_model = results[best_model_name]['Model']
if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
    cv_scores = cross_val_score(best_model, scaler.transform(X_selected), y, cv=5, scoring='r2')
else:
    cv_scores = cross_val_score(best_model, X_selected, y, cv=5, scoring='r2')

print(f"\nCross-validation R2 Scores: {cv_scores}")
print(f"Mean CV R2 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature Importance for tree-based models
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'feature': top_features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=True)

    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.title(f'Feature Importance - {best_model_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

# Visualization of predictions vs actual values
plt.figure(figsize=(12, 5))

# Actual vs Predicted values
plt.subplot(1, 2, 1)
plt.scatter(y_test, best_result['Predictions'], alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Prices (Lakhs)')
plt.ylabel('Predicted Prices (Lakhs)')
plt.title(f'Actual vs Predicted Prices\n{best_model_name}')

# Residual plot
plt.subplot(1, 2, 2)
residuals = y_test - best_result['Predictions']
plt.scatter(best_result['Predictions'], residuals, alpha=0.6, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Prices (Lakhs)')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.tight_layout()
plt.show()

# Model comparison visualization
plt.figure(figsize=(12, 5))

models_list = list(results.keys())
r2_scores = [results[model]['R2 Score'] for model in models_list]
rmse_scores = [results[model]['RMSE'] for model in models_list]

x_pos = np.arange(len(models_list))

plt.subplot(1, 2, 1)
bars = plt.bar(x_pos, r2_scores, color='skyblue', alpha=0.7)
plt.xlabel('Models')
plt.ylabel('R2 Score')
plt.title('Model Comparison - R2 Scores')
plt.xticks(x_pos, models_list, rotation=45)
plt.ylim(0, 1)

# Add value labels on bars
for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom')

plt.subplot(1, 2, 2)
bars = plt.bar(x_pos, rmse_scores, color='lightcoral', alpha=0.7)
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('Model Comparison - RMSE Scores')
plt.xticks(x_pos, models_list, rotation=45)

# Add value labels on bars
for bar, score in zip(bars, rmse_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# FIXED: Real-world prediction function
def predict_car_price(car_features):
    """Predict car price based on input features"""

    # Create DataFrame from input features
    sample_df = pd.DataFrame([car_features])

    # Apply the same feature engineering - FIXED: Use .iloc[0] to get scalar values
    sample_df['Car_Age'] = current_year - sample_df['Year']

    # FIXED: Use .iloc[0] to avoid Series ambiguity
    car_age = max(sample_df['Car_Age'].iloc[0], 1)
    sample_df['Km_Per_Year'] = sample_df['Driven_kms'] / car_age

    sample_df['Depreciation_Ratio'] = 0.7  # Typical depreciation
    sample_df['Driven_kms_log'] = np.log1p(sample_df['Driven_kms'])
    sample_df['Present_Price_log'] = np.log1p(sample_df['Present_Price'])
    sample_df['Brand_Category'] = sample_df['Brand'].apply(categorize_brand)

    # Encode categorical variables
    for feature in categorical_features:
        if feature in sample_df.columns:
            le = label_encoders[feature]
            # Handle unseen labels
            if sample_df[feature].iloc[0] not in le.classes_:
                # Use default encoding
                sample_df[feature + '_encoded'] = 0
            else:
                sample_df[feature + '_encoded'] = le.transform(sample_df[feature])

    # Select the same features
    sample_features = sample_df[[col for col in feature_columns if col in sample_df.columns]]

    # Ensure we have all required columns (fill missing with 0)
    for col in top_features:
        if col not in sample_features.columns:
            sample_features[col] = 0

    sample_features = sample_features[top_features]

    # Scale if using linear model
    if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
        sample_scaled = scaler.transform(sample_features)
        predicted_price = best_model.predict(sample_scaled)[0]
    else:
        predicted_price = best_model.predict(sample_features)[0]

    return max(0.1, predicted_price)  # Ensure positive price


# Real-world prediction example
print("\n=== REAL-WORLD PREDICTION EXAMPLE ===")

# Test predictions with different car types
test_cars = [
    {
        'Present_Price': 8.5, 'Year': 2021, 'Driven_kms': 45000, 'Owner': 0,
        'Fuel_Type': 'Petrol', 'Selling_type': 'Dealer', 'Transmission': 'Manual', 'Brand': 'city'
    },
    {
        'Present_Price': 4.2, 'Year': 2018, 'Driven_kms': 60000, 'Owner': 0,
        'Fuel_Type': 'Diesel', 'Selling_type': 'Individual', 'Transmission': 'Manual', 'Brand': 'swift'
    },
    {
        'Present_Price': 25.0, 'Year': 2020, 'Driven_kms': 25000, 'Owner': 0,
        'Fuel_Type': 'Diesel', 'Selling_type': 'Dealer', 'Transmission': 'Automatic', 'Brand': 'fortuner'
    }
]

print("\nCar Price Predictions:")
print("-" * 50)
for i, car in enumerate(test_cars, 1):
    predicted_price = predict_car_price(car)  # FIXED: Removed unnecessary parameter
    print(f"Car {i}:")
    print(f"  Brand: {car['Brand']}, Year: {car['Year']}, KM: {car['Driven_kms']:,}")
    print(f"  Fuel: {car['Fuel_Type']}, Transmission: {car['Transmission']}")
    print(f"  Predicted Selling Price: {predicted_price:.2f} Lakhs")
    print()

print("\n=== REAL-WORLD APPLICATIONS ===")
applications = [
    "1. Used Car Valuation: Accurately price used cars for dealerships and individuals",
    "2. Insurance Premium Calculation: Determine appropriate insurance coverage amounts",
    "3. Loan Assessment: Help financial institutions assess vehicle collateral value",
    "4. Market Analysis: Understand price trends and depreciation patterns",
    "5. Inventory Management: Optimize pricing strategies for car dealerships",
    "6. Customer Advisory: Provide price guidance to buyers and sellers",
    "7. Auction Pricing: Set realistic price expectations for vehicle auctions"
]

for app in applications:
    print(app)

print("\n=== KEY INSIGHTS ===")
print(f"* Dataset contains {len(df)} car listings with {len(df.columns)} features")
print(f"* Price range: {df['Selling_Price'].min():.2f}L - {df['Selling_Price'].max():.2f}L")
print(f"* Most common fuel type: {df['Fuel_Type'].mode().values[0]}")
print(f"* Average car age: {(current_year - df['Year']).mean():.1f} years")
print(f"* Best performing model: {best_model_name} with R2 = {best_result['R2 Score']:.4f}")
print(f"* Average prediction error: {best_result['RMSE']:.2f} Lakhs")

# Save the trained model and preprocessing objects
import joblib

model_artifacts = {
    'model': best_model,
    'scaler': scaler,
    'feature_columns': top_features.tolist(),
    'label_encoders': label_encoders,
    'current_year': current_year,
    'predict_function': predict_car_price  # Include the prediction function
}

joblib.dump(model_artifacts, 'car_price_predictor.pkl')
print("\nModel artifacts saved as 'car_price_predictor.pkl'")

print("\n=== MODEL DEPLOYMENT READY ===")
print("The car price prediction model is ready for real-world use!")
print("Key improvements in this version:")
print("- Fixed pandas Series ambiguity error")
print("- Improved feature engineering with log transformations")
print("- Better outlier handling")
print("- Enhanced error handling in prediction function")

# Additional: Show some actual vs predicted comparisons
print("\n=== SAMPLE PREDICTIONS COMPARISON ===")
print("Actual vs Predicted Prices for Test Set (Sample of 10):")
print("-" * 50)
sample_indices = np.random.choice(len(y_test), min(10, len(y_test)), replace=False)
for i, idx in enumerate(sample_indices):
    actual = y_test.iloc[idx]
    predicted = best_result['Predictions'][idx]
    error = abs(actual - predicted)
    print(f"Car {i + 1}: Actual = {actual:.2f}L, Predicted = {predicted:.2f}L, Error = {error:.2f}L")