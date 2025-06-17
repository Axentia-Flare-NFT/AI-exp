import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better visualizations
plt.style.use('default')  # Using default style instead of seaborn
sns.set_theme()  # This will set a nice seaborn theme without requiring the style file

try:
    # Read the data
    df = pd.read_csv('data/nft_features.csv')
    
    # Select relevant features and target
    features = ['floor_price', 'total_volume', 'num_owners', 'avg_sentiment',
                'positive_tweets', 'negative_tweets', 'neutral_tweets']
    target = 'sale_price_eth'
    
    # Create a figure for correlation matrix
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[features + [target]].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of Features and Target')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # Create scatter plots for each feature vs target
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    
    for idx, feature in enumerate(features):
        sns.scatterplot(data=df, x=feature, y=target, ax=axes[idx], alpha=0.6)
        axes[idx].set_title(f'{feature} vs {target}')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel(target)
    
    plt.tight_layout()
    plt.savefig('feature_vs_target_scatter.png')
    plt.close()
    
    # Create box plots for each feature grouped by target quartiles
    df['price_quartile'] = pd.qcut(df[target], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    
    for idx, feature in enumerate(features):
        sns.boxplot(data=df, x='price_quartile', y=feature, ax=axes[idx])
        axes[idx].set_title(f'{feature} by Price Quartile')
        axes[idx].set_xlabel('Price Quartile')
        axes[idx].set_ylabel(feature)
    
    plt.tight_layout()
    plt.savefig('feature_boxplots.png')
    plt.close()
    
    # Print statistical summaries
    print("\n=== Feature Statistics ===")
    print("\nCorrelation with sale_price_eth:")
    correlations = df[features + [target]].corr()[target].sort_values(ascending=False)
    print(correlations.round(3))
    
    print("\nFeature Statistics:")
    print(df[features].describe().round(3))
    
    # Calculate and print feature importance based on correlation
    print("\nFeature Importance (based on absolute correlation):")
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Correlation': [abs(correlations[feature]) for feature in features]
    })
    feature_importance = feature_importance.sort_values('Correlation', ascending=False)
    print(feature_importance.round(3))
    
    # Create a bar plot of feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Correlation', y='Feature')
    plt.title('Feature Importance (Absolute Correlation with sale_price_eth)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("\nEDA completed! Check the generated plots in the current directory.")
    
except FileNotFoundError:
    print("Error: Could not find the data file. Please make sure 'data/nft_features.csv' exists.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
