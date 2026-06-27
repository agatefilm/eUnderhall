#!/usr/bin/env python3
"""
PCA Analysis on machine_data.csv
This script performs Principal Component Analysis on the machine data
and creates pedagogical diagrams to explain the results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the data
df = pd.read_csv('machine_data.csv')

# Clean column names
df.columns = df.columns.str.strip()
df = df.rename(columns={'manufacturef': 'manufacturer'})

print("=" * 80)
print("PCA ANALYSIS ON MACHINE DATA")
print("=" * 80)
print()

# Prepare data for PCA
# We'll use time and load as features
X = df[['time', 'load']].values
feature_names = ['time', 'load']

print("Original Data Shape:", X.shape)
print("Features:", feature_names)
print()

# Standardize the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data has been standardized (mean=0, std=1)")
print()

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with PCA results
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['manufacturer'] = df['manufacturer'].values

print("PCA Results:")
print("-" * 80)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
print()

# Print PCA components
print("Principal Components (Eigenvectors):")
print("-" * 80)
components = pd.DataFrame(
    pca.components_,
    columns=feature_names,
    index=['PC1', 'PC2']
)
print(components)
print()

# Interpret the components
print("Interpretation of Principal Components:")
print("-" * 80)
print("PC1 (Principal Component 1):")
print(f"  - Explains {pca.explained_variance_ratio_[0]*100:.2f}% of the variance")
print(f"  - Loadings: time = {pca.components_[0,0]:.4f}, load = {pca.components_[0,1]:.4f}")
if abs(pca.components_[0,0]) > abs(pca.components_[0,1]):
    print(f"  - Most influenced by: time")
else:
    print(f"  - Most influenced by: load")
print()

print("PC2 (Principal Component 2):")
print(f"  - Explains {pca.explained_variance_ratio_[1]*100:.2f}% of the variance")
print(f"  - Loadings: time = {pca.components_[1,0]:.4f}, load = {pca.components_[1,1]:.4f}")
if abs(pca.components_[1,0]) > abs(pca.components_[1,1]):
    print(f"  - Most influenced by: time")
else:
    print(f"  - Most influenced by: load")
print()

# Create pedagogical diagrams
print("Creating pedagogical diagrams...")
print()

# Create a directory for plots
os.makedirs('pca_plots', exist_ok=True)

# ============================================================================
# DIAGRAM 1: Original Data with PCA Vectors
# ============================================================================
plt.figure(figsize=(14, 10))

# Plot original data
plt.subplot(2, 2, 1)
for manufacturer in df['manufacturer'].unique():
    subset = df[df['manufacturer'] == manufacturer]
    plt.scatter(subset['time'], subset['load'], 
                label=f'Manufacturer {manufacturer}', 
                alpha=0.6, s=50)
plt.xlabel('Time (standardized)')
plt.ylabel('Load (standardized)')
plt.title('Original Data (Standardized)')
plt.legend()
plt.grid(True, alpha=0.3)

# Add PCA vectors
scale_factor = 5  # Scale for visualization
for i, feature in enumerate(feature_names):
    plt.arrow(0, 0, pca.components_[0,i] * scale_factor, 
              pca.components_[1,i] * scale_factor, 
              color='red', alpha=0.5, head_width=0.1)
    plt.text(pca.components_[0,i] * scale_factor * 1.15, 
             pca.components_[1,i] * scale_factor * 1.15, 
             f'{feature}', color='red', ha='center', va='center')

# ============================================================================
# DIAGRAM 2: PCA Scatter Plot
# ============================================================================
plt.subplot(2, 2, 2)
for manufacturer in df['manufacturer'].unique():
    subset = pca_df[pca_df['manufacturer'] == manufacturer]
    plt.scatter(subset['PC1'], subset['PC2'], 
                label=f'Manufacturer {manufacturer}', 
                alpha=0.6, s=50)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('PCA Scatter Plot')
plt.legend()
plt.grid(True, alpha=0.3)

# Add explanation text
plt.figtext(0.5, 0.01, 
            "PCA transforms the original features into uncorrelated components.\n" +
            "PC1 captures the most variance, PC2 the second most.",
            ha='center', fontsize=10, bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})

plt.tight_layout()
plt.savefig('pca_plots/pca_overview.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# DIAGRAM 3: Explained Variance (Scree Plot)
# ============================================================================
plt.figure(figsize=(10, 6))

# Bar plot of explained variance
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
        pca.explained_variance_ratio_, 
        alpha=0.7, color=['skyblue', 'salmon'])

# Line plot of cumulative variance
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), 
         marker='o', color='red', linestyle='--', label='Cumulative')

plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot: Explained Variance by Principal Component')
plt.xticks([1, 2])
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels
for i, v in enumerate(pca.explained_variance_ratio_):
    plt.text(i + 1, v + 0.01, f'{v*100:.1f}%', ha='center')

# Add explanation
plt.figtext(0.5, 0.01, 
            "The scree plot shows how much variance each principal component captures.\n" +
            "PC1 captures most of the information in the data.",
            ha='center', fontsize=10, bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})

plt.tight_layout()
plt.savefig('pca_plots/scree_plot.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# DIAGRAM 4: Component Loadings (Biplot)
# ============================================================================
plt.figure(figsize=(12, 8))

# Plot PCA scores
for manufacturer in df['manufacturer'].unique():
    subset = pca_df[pca_df['manufacturer'] == manufacturer]
    plt.scatter(subset['PC1'], subset['PC2'], 
                label=f'Manufacturer {manufacturer}', 
                alpha=0.6, s=50)

# Add feature vectors (loadings)
for i, feature in enumerate(feature_names):
    # Scale the loadings for better visualization
    plt.arrow(0, 0, pca.components_[0,i] * max(X_pca[:,0]), 
              pca.components_[1,i] * max(X_pca[:,1]), 
              color='red', alpha=0.8, head_width=0.05, length_includes_head=True)
    plt.text(pca.components_[0,i] * max(X_pca[:,0]) * 1.15, 
             pca.components_[1,i] * max(X_pca[:,1]) * 1.15, 
             feature, color='red', ha='center', va='center', fontsize=12)

# Draw circles to show correlation
for i, feature in enumerate(feature_names):
    x = pca.components_[0,i] * max(X_pca[:,0])
    y = pca.components_[1,i] * max(X_pca[:,1])
    # Draw a circle with radius equal to the loading
    circle = plt.Circle((0, 0), np.sqrt(x**2 + y**2), 
                        color='gray', alpha=0.1, fill=False)
    plt.gca().add_patch(circle)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('PCA Biplot: Scores and Loadings')
plt.legend()
plt.grid(True, alpha=0.3)

# Add explanation
plt.figtext(0.5, 0.01, 
            "Biplot shows both data points (scores) and feature contributions (loadings).\n" +
            "The length and direction of arrows indicate how much each feature contributes to each PC.\n" +
            "Points close together are similar in the original feature space.",
            ha='center', fontsize=10, bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})

plt.tight_layout()
plt.savefig('pca_plots/biplot.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# DIAGRAM 5: PCA by Manufacturer (Separate Plots)
# ============================================================================
plt.figure(figsize=(15, 5))

for i, manufacturer in enumerate(df['manufacturer'].unique()):
    plt.subplot(1, 3, i + 1)
    subset = pca_df[pca_df['manufacturer'] == manufacturer]
    plt.scatter(subset['PC1'], subset['PC2'], 
                color=sns.color_palette("husl", 3)[i], 
                alpha=0.6, s=50)
    plt.xlabel(f'PC1')
    plt.ylabel(f'PC2')
    plt.title(f'Manufacturer {manufacturer}')
    plt.grid(True, alpha=0.3)
    
    # Add mean point
    mean_pc1 = subset['PC1'].mean()
    mean_pc2 = subset['PC2'].mean()
    plt.scatter(mean_pc1, mean_pc2, color='black', s=100, marker='X', label='Mean')
    plt.legend()

plt.suptitle('PCA Distribution by Manufacturer', fontsize=16)
plt.tight_layout()
plt.savefig('pca_plots/pca_by_manufacturer.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# DIAGRAM 6: Variance Explained - Pie Chart
# ============================================================================
plt.figure(figsize=(8, 8))

# Create pie chart
labels = [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))]
sizes = pca.explained_variance_ratio_ * 100
colors = ['skyblue', 'salmon']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
        startangle=90, explode=(0.1, 0), shadow=True)
plt.title('Variance Explained by Each Principal Component')

# Add explanation
plt.figtext(0.5, 0.01, 
            "This pie chart shows the proportion of total variance captured by each PC.\n" +
            "PC1 captures the majority of the information in this 2D dataset.",
            ha='center', fontsize=10, bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})

plt.tight_layout()
plt.savefig('pca_plots/variance_pie_chart.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# DIAGRAM 7: Correlation Circle
# ============================================================================
plt.figure(figsize=(10, 10))

# Create unit circle
circle = plt.Circle((0, 0), 1, color='gray', alpha=0.2, fill=True)
plt.gca().add_patch(circle)

# Plot feature loadings on correlation circle
for i, feature in enumerate(feature_names):
    x = pca.components_[0,i]
    y = pca.components_[1,i]
    
    # Plot arrow from origin to loading
    plt.arrow(0, 0, x, y, color='blue', alpha=0.7, head_width=0.05)
    
    # Plot point at loading
    plt.scatter(x, y, color='red', s=100)
    
    # Add label
    plt.text(x * 1.15, y * 1.15, feature, 
             ha='center', va='center', fontsize=12)

# Draw axes
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)

plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Correlation Circle')
plt.grid(True, alpha=0.3)

# Add explanation
plt.figtext(0.5, 0.01, 
            "The correlation circle shows how original features correlate with principal components.\n" +
            "Features close to the circle edge have strong correlation with that PC direction.\n" +
            "The angle between feature vectors shows their correlation in the original space.",
            ha='center', fontsize=10, bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})

plt.tight_layout()
plt.savefig('pca_plots/correlation_circle.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# DIAGRAM 8: PCA Transformation Visualization
# ============================================================================
plt.figure(figsize=(16, 6))

# Original coordinate system
plt.subplot(1, 3, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.3, s=30)
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Standardized Time')
plt.ylabel('Standardized Load')
plt.title('Original Space')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid(True, alpha=0.3)

# Draw original axes
plt.arrow(0, 0, 3, 0, head_width=0.1, fc='red', ec='red', linestyle='-', alpha=0.5)
plt.arrow(0, 0, 0, 3, head_width=0.1, fc='red', ec='red', linestyle='-', alpha=0.5)
plt.text(3.2, 0, 'Time', color='red', ha='left', va='center')
plt.text(0, 3.2, 'Load', color='red', ha='center', va='bottom')

# PCA coordinate system
plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, s=30)
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Space')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid(True, alpha=0.3)

# Draw PCA axes
plt.arrow(0, 0, 3, 0, head_width=0.1, fc='blue', ec='blue', linestyle='-', alpha=0.5)
plt.arrow(0, 0, 0, 3, head_width=0.1, fc='blue', ec='blue', linestyle='-', alpha=0.5)
plt.text(3.2, 0, 'PC1', color='blue', ha='left', va='center')
plt.text(0, 3.2, 'PC2', color='blue', ha='center', va='bottom')

# Transformation visualization
plt.subplot(1, 3, 3)
# Plot both coordinate systems
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.3, s=30, color='gray')

# Original axes
plt.arrow(0, 0, 2, 0, head_width=0.1, fc='red', ec='red', linestyle='-', alpha=0.5)
plt.arrow(0, 0, 0, 2, head_width=0.1, fc='red', ec='red', linestyle='-', alpha=0.5)
plt.text(2.2, 0, 'Time', color='red', ha='left', va='center')
plt.text(0, 2.2, 'Load', color='red', ha='center', va='bottom')

# PCA axes (scaled for visualization)
pca_scale = 2
plt.arrow(0, 0, pca.components_[0,0] * pca_scale, pca.components_[0,1] * pca_scale, 
          head_width=0.1, fc='blue', ec='blue', linestyle='-', alpha=0.7)
plt.arrow(0, 0, pca.components_[1,0] * pca_scale, pca.components_[1,1] * pca_scale, 
          head_width=0.1, fc='green', ec='green', linestyle='-', alpha=0.7)
plt.text(pca.components_[0,0] * pca_scale * 1.15, pca.components_[0,1] * pca_scale * 1.15, 
         'PC1', color='blue', ha='center', va='center')
plt.text(pca.components_[1,0] * pca_scale * 1.15, pca.components_[1,1] * pca_scale * 1.15, 
         'PC2', color='green', ha='center', va='center')

plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Feature Space')
plt.ylabel('Feature Space')
plt.title('Coordinate Transformation')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid(True, alpha=0.3)

plt.suptitle('PCA Transformation: From Original Space to PCA Space', fontsize=16)
plt.tight_layout()
plt.savefig('pca_plots/transformation.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# DIAGRAM 9: Eigenvalue Analysis
# ============================================================================
plt.figure(figsize=(12, 6))

# Eigenvalues
plt.subplot(1, 2, 1)
eigenvalues = pca.explained_variance_
plt.bar(range(1, len(eigenvalues) + 1), eigenvalues, color=['skyblue', 'salmon'])
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues of Covariance Matrix')
plt.xticks([1, 2])
plt.grid(True, alpha=0.3)

# Add value labels
for i, v in enumerate(eigenvalues):
    plt.text(i + 1, v + 0.1, f'{v:.4f}', ha='center')

# Explained variance
plt.subplot(1, 2, 2)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
        pca.explained_variance_ratio_ * 100, 
        color=['skyblue', 'salmon'])
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance (%)')
plt.title('Percentage of Variance Explained')
plt.xticks([1, 2])
plt.grid(True, alpha=0.3)

# Add value labels
for i, v in enumerate(pca.explained_variance_ratio_):
    plt.text(i + 1, v * 100 + 1, f'{v*100:.1f}%', ha='center')

plt.suptitle('Eigenvalue Analysis', fontsize=16)
plt.tight_layout()
plt.savefig('pca_plots/eigenvalue_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# DIAGRAM 10: Manufacturer Separation Analysis
# ============================================================================
plt.figure(figsize=(12, 8))

# Calculate mean PCA scores for each manufacturer
manufacturer_means = pca_df.groupby('manufacturer')[['PC1', 'PC2']].mean()

# Plot all data points
for manufacturer in df['manufacturer'].unique():
    subset = pca_df[pca_df['manufacturer'] == manufacturer]
    plt.scatter(subset['PC1'], subset['PC2'], 
                label=f'Manufacturer {manufacturer}', 
                alpha=0.3, s=30)

# Plot mean points with error bars
for manufacturer in df['manufacturer'].unique():
    subset = pca_df[pca_df['manufacturer'] == manufacturer]
    mean_pc1 = subset['PC1'].mean()
    mean_pc2 = subset['PC2'].mean()
    std_pc1 = subset['PC1'].std()
    std_pc2 = subset['PC2'].std()
    
    plt.errorbar(mean_pc1, mean_pc2, 
                 xerr=std_pc1, yerr=std_pc2,
                 fmt='o', markersize=10, capsize=5, 
                 color='black', markeredgewidth=2)
    plt.text(mean_pc1, mean_pc2 + 0.5, 
             f'{manufacturer}', 
             ha='center', va='bottom', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('Manufacturer Separation in PCA Space')
plt.legend()
plt.grid(True, alpha=0.3)

# Add explanation
plt.figtext(0.5, 0.01, 
            "This plot shows how well manufacturers are separated in the PCA space.\n" +
            "Mean points with error bars indicate the central tendency and spread of each group.",
            ha='center', fontsize=10, bbox={"facecolor":"lightyellow", "alpha":0.5, "pad":5})

plt.tight_layout()
plt.savefig('pca_plots/manufacturer_separation.png', dpi=150, bbox_inches='tight')
plt.close()

print("Pedagogical diagrams saved to 'pca_plots/' directory:")
print("  1. pca_overview.png - Original data with PCA vectors")
print("  2. scree_plot.png - Explained variance by component")
print("  3. biplot.png - PCA scores and loadings")
print("  4. pca_by_manufacturer.png - PCA distribution by manufacturer")
print("  5. variance_pie_chart.png - Variance explained pie chart")
print("  6. correlation_circle.png - Feature correlation circle")
print("  7. transformation.png - Coordinate transformation visualization")
print("  8. eigenvalue_analysis.png - Eigenvalue and variance analysis")
print("  9. manufacturer_separation.png - Manufacturer separation in PCA space")
print()

# Print summary
print("=" * 80)
print("PCA ANALYSIS SUMMARY")
print("=" * 80)
print()
print("Principal Components:")
print(f"  PC1: Explains {pca.explained_variance_ratio_[0]*100:.2f}% of variance")
print(f"  PC2: Explains {pca.explained_variance_ratio_[1]*100:.2f}% of variance")
print(f"  Total: {np.sum(pca.explained_variance_ratio_)*100:.2f}% of variance explained")
print()
print("Component Loadings:")
print(components)
print()
print("Interpretation:")
print(f"  PC1 is primarily influenced by: {feature_names[np.argmax(np.abs(pca.components_[0]))]}")
print(f"  PC2 is primarily influenced by: {feature_names[np.argmax(np.abs(pca.components_[1]))]}")
print()

# Calculate and print manufacturer statistics in PCA space
print("Manufacturer Statistics in PCA Space:")
print("-" * 80)
manufacturer_pca_stats = pca_df.groupby('manufacturer')[['PC1', 'PC2']].agg(['mean', 'std', 'min', 'max'])
print(manufacturer_pca_stats)
print()

# Calculate distances between manufacturer means
print("Distances Between Manufacturer Means in PCA Space:")
print("-" * 80)
manufacturer_means = pca_df.groupby('manufacturer')[['PC1', 'PC2']].mean()
distances = {}
for m1 in manufacturer_means.index:
    for m2 in manufacturer_means.index:
        if m1 < m2:  # Only calculate upper triangle
            dist = np.sqrt((manufacturer_means.loc[m1, 'PC1'] - manufacturer_means.loc[m2, 'PC1'])**2 + 
                          (manufacturer_means.loc[m1, 'PC2'] - manufacturer_means.loc[m2, 'PC2'])**2)
            distances[(m1, m2)] = dist
            print(f"  Distance between {m1} and {m2}: {dist:.4f}")
print()

print("=" * 80)
print("PCA ANALYSIS COMPLETE")
print("=" * 80)
