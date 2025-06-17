# First, install required packages in your new environment:
# pip install xgboost pandas scikit-learn matplotlib graphviz
# For Mac: brew install graphviz
# For Ubuntu: sudo apt-get install graphviz

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# Load the data
df = pd.read_csv('model_data.csv')
print(f"Data shape: {df.shape}")
print(f"Data columns: {df.columns.tolist()}")

# Define features and target
features = [
    'floor_price', 'total_volume', 'num_owners', 'tweet_count',
    'avg_sentiment',
    'positive_tweets', 'negative_tweets', 'neutral_tweets',
    'sentiment_range_min', 'sentiment_range_max',
    'rarity_rank'
]
target = 'sale_price_eth'

# Check if all features exist in the dataset
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print(f"WARNING: Missing features: {missing_features}")
    features = [f for f in features if f in df.columns]
    print(f"Using available features: {features}")

# Check if target exists
if target not in df.columns:
    print(f"ERROR: Target column '{target}' not found in data")
    print(f"Available columns: {df.columns.tolist()}")
    exit()

# Prepare the data
X = df[features]
y = df[target]

print(f"Features being used: {features}")
print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Check for missing values
print(f"Missing values in features: {X.isnull().sum().sum()}")
print(f"Missing values in target: {y.isnull().sum()}")

# Handle missing values if any
if X.isnull().sum().sum() > 0:
    print("Filling missing values with median...")
    X = X.fillna(X.median())
    
if y.isnull().sum() > 0:
    print("Dropping rows with missing target values...")
    mask = ~y.isnull()
    X = X[mask]
    y = y[mask]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create XGBoost regressor
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)

# Train the model
print("Training XGBoost model...")
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
feature_importance = xgb_model.feature_importances_
feature_names = X.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
indices = np.argsort(feature_importance)[::-1]
plt.bar(range(len(feature_importance)), feature_importance[indices])
plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# ===== TREE VISUALIZATION =====
print("\nVisualizing trees...")

# Option 1: Simple matplotlib tree visualization (no graphviz needed)
try:
    # Save plots to files since GUI might not display properly
    print("Saving tree visualizations...")
    
    # Plot the first few trees and save them
    for i in range(min(3, xgb_model.n_estimators)):
        plt.figure(figsize=(20, 12))
        # Use tree_idx instead of num_trees (updated parameter name)
        xgb.plot_tree(xgb_model, tree_idx=i, rankdir='LR', 
                     fmap='', ax=plt.gca())
        plt.title(f'XGBoost Tree #{i}')
        filename = f'xgboost_tree_{i}_simple.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Tree {i} saved as {filename}")
        plt.close()  # Close to prevent display issues
        
except Exception as e:
    print(f"Simple tree plotting failed: {e}")

# Option 2: Text representation of tree structure (always works)
print("\nTree structure (first tree as text):")
try:
    booster = xgb_model.get_booster()
    tree_dump = booster.get_dump(dump_format='text')
    print("First few lines of tree structure:")
    print('\n'.join(tree_dump[0].split('\n')[:20]))  # Show first 20 lines
    
    # Save full tree structure to file
    with open('tree_structure.txt', 'w') as f:
        for i, tree in enumerate(tree_dump[:5]):  # Save first 5 trees
            f.write(f"=== TREE {i} ===\n")
            f.write(tree)
            f.write("\n\n")
    print("Full tree structures saved to 'tree_structure.txt'")
    
except Exception as e:
    print(f"Text tree dump failed: {e}")

# Option 3: Graphviz visualization (only if system graphviz is installed)
try:
    import graphviz
    
    # Test if graphviz system command is available
    import subprocess
    subprocess.run(['dot', '-V'], capture_output=True, check=True)
    
    print("Creating detailed graphviz visualizations...")
    for i in range(min(2, xgb_model.n_estimators)):
        graph = xgb.to_graphviz(xgb_model, tree_idx=i, rankdir='LR')
        filename = f'xgboost_tree_{i}_detailed'
        graph.render(filename, format='png', cleanup=True)
        print(f"Detailed tree {i} saved as '{filename}.png'")
        
except (ImportError, subprocess.CalledProcessError, FileNotFoundError):
    print("Graphviz not available. Install with: brew install graphviz (Mac)")
    print("Then restart your terminal and try again.")
except Exception as e:
    print(f"Graphviz visualization failed: {e}")

# ===== ADDITIONAL ANALYSIS =====

# Plot predictions vs actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Sale Price (ETH)')
plt.ylabel('Predicted Sale Price (ETH)')
plt.title('Predictions vs Actual Values')
plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
print("Predictions vs actual plot saved as 'predictions_vs_actual.png'")
plt.close()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Sale Price (ETH)')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('residual_plot.png', dpi=300, bbox_inches='tight')
print("Residual plot saved as 'residual_plot.png'")
plt.close()

print(f"\nModel training and visualization complete!")
print(f"All files saved in: {os.getcwd()}")

# Save the model
model_filename = 'xgboost_model.json'
xgb_model.save_model(model_filename)
print(f"Model saved as '{model_filename}' in {os.getcwd()}")

# List all generated files
print("\nGenerated files:")
for file in os.listdir('.'):
    if any(file.endswith(ext) for ext in ['.png', '.txt', '.json']):
        print(f"  - {file}")

# To load the model later:
# loaded_model = xgb.XGBRegressor()
# loaded_model.load_model('xgboost_model.json')