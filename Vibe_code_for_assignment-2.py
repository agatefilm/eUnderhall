#!/usr/bin/env python3
"""
Analysis of machine_data.csv
This script answers the following questions:
1. What is the range of load and time during operation for each manufacturer?
2. What is the most expected load value?
3. How are the load and time related?
4. Which distribution best describes the load?
5. Which distribution best describes the time?
6. Which manufacturer has the best performance and why?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, uniform, expon, gamma, beta, lognorm
from sklearn.preprocessing import MinMaxScaler
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the data
df = pd.read_csv('machine_data.csv')

# Clean column names - remove any extra whitespace or special characters
df.columns = df.columns.str.strip()

# Rename columns for clarity
df = df.rename(columns={
    'manufacturef': 'manufacturer'
})

print("=" * 80)
print("MACHINE DATA ANALYSIS")
print("=" * 80)
print()

# Question 1: What is the range of load and time during operation for each manufacturer?
print("1. RANGE OF LOAD AND TIME FOR EACH MANUFACTURER")
print("-" * 80)

manufacturer_stats = df.groupby('manufacturer').agg({
    'time': ['min', 'max', 'mean', 'std'],
    'load': ['min', 'max', 'mean', 'std']
}).round(4)

print("Statistics by Manufacturer:")
print(manufacturer_stats)
print()

# Calculate range for each manufacturer
manufacturer_range = df.groupby('manufacturer').agg({
    'time': lambda x: x.max() - x.min(),
    'load': lambda x: x.max() - x.min()
}).rename(columns={'time': 'time_range', 'load': 'load_range'})

print("Range of values by Manufacturer:")
print(manufacturer_range)
print()

# Question 2: What is the most expected load value?
print("2. MOST EXPECTED LOAD VALUE")
print("-" * 80)

# Calculate mean, median, and mode
mean_load = df['load'].mean()
median_load = df['load'].median()

# For mode, we'll use the most frequent value in a binned distribution
# Create histogram to find the peak
load_hist, load_bins = np.histogram(df['load'], bins=50, density=True)
peak_bin_idx = np.argmax(load_hist)
most_expected_load = (load_bins[peak_bin_idx] + load_bins[peak_bin_idx + 1]) / 2

print(f"Mean load: {mean_load:.4f}")
print(f"Median load: {median_load:.4f}")
print(f"Most expected load (histogram peak): {most_expected_load:.4f}")
print()

# Question 3: How are the load and time related?
print("3. RELATIONSHIP BETWEEN LOAD AND TIME")
print("-" * 80)

# Calculate correlation
correlation = df['load'].corr(df['time'])
print(f"Pearson correlation coefficient: {correlation:.4f}")

# Interpret correlation
if abs(correlation) < 0.3:
    relationship = "Weak or no linear relationship"
elif abs(correlation) < 0.7:
    relationship = "Moderate linear relationship"
else:
    relationship = "Strong linear relationship"

if correlation > 0:
    direction = "positive"
elif correlation < 0:
    direction = "negative"
else:
    direction = "no"

print(f"Interpretation: {relationship} ({direction} correlation)")
print()

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(df['time'], df['load'])
print(f"Linear regression: load = {slope:.6f} * time + {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value:.6f}")
print()

# Question 4: Which distribution best describes the load?
print("4. BEST DISTRIBUTION FOR LOAD")
print("-" * 80)

# Test different distributions for load
def fit_distribution(data, distributions):
    """Fit data to multiple distributions and return best fit"""
    results = []
    for distribution in distributions:
        try:
            # Fit distribution
            if distribution == norm:
                # For normal distribution, use loc and scale
                loc, scale = distribution.fit(data)
                fitted_dist = distribution(loc=loc, scale=scale)
                params = (loc, scale)
            elif distribution == uniform:
                # For uniform, we need min and max
                data_min = data.min()
                data_max = data.max()
                fitted_dist = distribution(loc=data_min, scale=data_max - data_min)
                params = (data_min, data_max - data_min)
            elif distribution == expon:
                # For exponential, use scale parameter
                loc, scale = distribution.fit(data)
                fitted_dist = distribution(loc=loc, scale=scale)
                params = (loc, scale)
            elif distribution == gamma:
                # For gamma, fit shape, loc, scale
                shape, loc, scale = distribution.fit(data)
                fitted_dist = distribution(a=shape, loc=loc, scale=scale)
                params = (shape, loc, scale)
            elif distribution == beta:
                # For beta, fit a, b, loc, scale
                a, b, loc, scale = distribution.fit(data)
                fitted_dist = distribution(a=a, b=b, loc=loc, scale=scale)
                params = (a, b, loc, scale)
            elif distribution == lognorm:
                # For lognormal, fit s, loc, scale
                s, loc, scale = distribution.fit(data)
                fitted_dist = distribution(s=s, loc=loc, scale=scale)
                params = (s, loc, scale)
            else:
                params = distribution.fit(data)
                fitted_dist = distribution(*params)
            
            # Calculate goodness of fit (Kolmogorov-Smirnov test)
            ks_stat, ks_pvalue = stats.kstest(data, distribution.name, args=params)
            
            # Also calculate AIC for comparison
            log_likelihood = np.sum(fitted_dist.logpdf(data))
            k = len(params)  # number of parameters
            aic = 2 * k - 2 * log_likelihood
            
            results.append({
                'distribution': distribution.name,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'aic': aic,
                'params': params
            })
        except Exception as e:
            print(f"  Warning: Could not fit {distribution.name}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

# Test distributions for load
distributions_to_test = [norm, uniform, expon, gamma, beta, lognorm]
load_dist_results = fit_distribution(df['load'], distributions_to_test)

print("Distribution fit results for load (sorted by AIC):")
load_dist_results_sorted = load_dist_results.sort_values('aic')
print(load_dist_results_sorted[['distribution', 'aic', 'ks_statistic', 'ks_pvalue']].to_string(index=False))
print()

if len(load_dist_results_sorted) > 0:
    best_load_dist = load_dist_results_sorted.iloc[0]
    print(f"Best distribution for load: {best_load_dist['distribution']}")
    print(f"  AIC: {best_load_dist['aic']:.4f}")
    print(f"  KS statistic: {best_load_dist['ks_statistic']:.4f}")
    print(f"  KS p-value: {best_load_dist['ks_pvalue']:.4f}")
else:
    print("Could not determine best distribution for load")
print()

# Question 5: Which distribution best describes the time?
print("5. BEST DISTRIBUTION FOR TIME")
print("-" * 80)

# Test distributions for time
time_dist_results = fit_distribution(df['time'], distributions_to_test)

print("Distribution fit results for time (sorted by AIC):")
time_dist_results_sorted = time_dist_results.sort_values('aic')
print(time_dist_results_sorted[['distribution', 'aic', 'ks_statistic', 'ks_pvalue']].to_string(index=False))
print()

if len(time_dist_results_sorted) > 0:
    best_time_dist = time_dist_results_sorted.iloc[0]
    print(f"Best distribution for time: {best_time_dist['distribution']}")
    print(f"  AIC: {best_time_dist['aic']:.4f}")
    print(f"  KS statistic: {best_time_dist['ks_statistic']:.4f}")
    print(f"  KS p-value: {best_time_dist['ks_pvalue']:.4f}")
else:
    print("Could not determine best distribution for time")
print()

# Question 6: Which manufacturer has the best performance and why?
print("6. MANUFACTURER WITH BEST PERFORMANCE")
print("-" * 80)

# Performance can be evaluated based on multiple criteria:
# 1. Higher average load (more work done)
# 2. Lower time (faster operation)
# 3. Higher load-to-time ratio (efficiency)
# 4. Consistency (lower variance)

# Calculate performance metrics for each manufacturer
performance_metrics = df.groupby('manufacturer').agg({
    'load': ['mean', 'std', 'min', 'max'],
    'time': ['mean', 'std', 'min', 'max']
})

# Calculate efficiency (load/time ratio)
df['efficiency'] = df['load'] / df['time']
efficiency_stats = df.groupby('manufacturer')['efficiency'].agg(['mean', 'std', 'min', 'max'])

# Combine all metrics
performance_metrics['efficiency_mean'] = efficiency_stats['mean']
performance_metrics['efficiency_std'] = efficiency_stats['std']

print("Performance Metrics by Manufacturer:")
print(performance_metrics)
print()

# Determine best manufacturer based on different criteria
print("Best manufacturer by criterion:")

# Highest average load
highest_load_manufacturer = performance_metrics['load']['mean'].idxmax()
highest_load_value = performance_metrics['load']['mean'].max()
print(f"  Highest average load: {highest_load_manufacturer} ({highest_load_value:.4f})")

# Lowest average time
lowest_time_manufacturer = performance_metrics['time']['mean'].idxmin()
lowest_time_value = performance_metrics['time']['mean'].min()
print(f"  Lowest average time: {lowest_time_manufacturer} ({lowest_time_value:.4f})")

# Highest efficiency (load/time)
highest_efficiency_manufacturer = performance_metrics['efficiency_mean'].idxmax()
highest_efficiency_value = performance_metrics['efficiency_mean'].max()
print(f"  Highest efficiency (load/time): {highest_efficiency_manufacturer} ({highest_efficiency_value:.4f})")

# Most consistent (lowest std deviation of load)
most_consistent_manufacturer = performance_metrics['load']['std'].idxmin()
most_consistent_value = performance_metrics['load']['std'].min()
print(f"  Most consistent load: {most_consistent_manufacturer} ({most_consistent_value:.4f})")

print()

# Overall assessment
print("Overall Performance Assessment:")
print("The best manufacturer depends on the performance criteria:")
print(f"- For maximum load capacity: {highest_load_manufacturer}")
print(f"- For speed (lowest time): {lowest_time_manufacturer}")
print(f"- For efficiency (load per unit time): {highest_efficiency_manufacturer}")
print(f"- For consistency: {most_consistent_manufacturer}")

# Create a composite score (normalized)
# Normalize each metric (higher is better)
scaler = MinMaxScaler()

# Create a DataFrame with the metrics we want to combine
metrics_df = pd.DataFrame({
    'load_mean': performance_metrics['load']['mean'],
    'time_mean': -performance_metrics['time']['mean'],  # Negative because lower time is better
    'efficiency_mean': performance_metrics['efficiency_mean'],
    'load_std': -performance_metrics['load']['std']  # Negative because lower std is better
})

# Normalize
metrics_normalized = scaler.fit_transform(metrics_df)

# Calculate composite score (equal weights)
composite_scores = metrics_normalized.mean(axis=1)
# Convert to pandas Series to get the index (manufacturer names)
composite_series = pd.Series(composite_scores, index=metrics_df.index)
best_overall_manufacturer = composite_series.idxmax()
best_overall_score = composite_series.max()

print(f"\nOverall best manufacturer (composite score): {best_overall_manufacturer} (score: {best_overall_score:.4f})")
print()

# Create visualizations
print("Creating visualizations...")
print()

# Create a directory for plots
os.makedirs('plots', exist_ok=True)

# 1. Distribution of load and time
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['load'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Load')
plt.xlabel('Load')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(df['time'], bins=50, kde=True, color='salmon')
plt.title('Distribution of Time')
plt.xlabel('Time')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('plots/load_time_distributions.png')
plt.close()

# 2. Load vs Time scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(x='time', y='load', data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title(f'Load vs Time (Correlation: {correlation:.4f})')
plt.xlabel('Time')
plt.ylabel('Load')
plt.savefig('plots/load_vs_time.png')
plt.close()

# 3. Box plots for load and time by manufacturer
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x='manufacturer', y='load', data=df)
plt.title('Load Distribution by Manufacturer')
plt.xlabel('Manufacturer')
plt.ylabel('Load')

plt.subplot(1, 2, 2)
sns.boxplot(x='manufacturer', y='time', data=df)
plt.title('Time Distribution by Manufacturer')
plt.xlabel('Manufacturer')
plt.ylabel('Time')

plt.tight_layout()
plt.savefig('plots/manufacturer_comparison.png')
plt.close()

# 4. Efficiency by manufacturer
plt.figure(figsize=(10, 6))
sns.boxplot(x='manufacturer', y='efficiency', data=df)
plt.title('Efficiency (Load/Time) by Manufacturer')
plt.xlabel('Manufacturer')
plt.ylabel('Efficiency')
plt.savefig('plots/efficiency_by_manufacturer.png')
plt.close()

# 5. Q-Q plots for best distributions (if we have valid distributions)
if len(load_dist_results_sorted) > 0 and len(time_dist_results_sorted) > 0:
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    stats.probplot(df['load'], dist=best_load_dist['distribution'], 
                   sparams=best_load_dist['params'], plot=plt)
    plt.title(f'Q-Q Plot for Load ({best_load_dist["distribution"]})')
    
    plt.subplot(1, 2, 2)
    stats.probplot(df['time'], dist=best_time_dist['distribution'], 
                   sparams=best_time_dist['params'], plot=plt)
    plt.title(f'Q-Q Plot for Time ({best_time_dist["distribution"]})')
    
    plt.tight_layout()
    plt.savefig('plots/qq_plots.png')
    plt.close()

print("Visualizations saved to 'plots/' directory:")
print("  - load_time_distributions.png")
print("  - load_vs_time.png")
print("  - manufacturer_comparison.png")
print("  - efficiency_by_manufacturer.png")
if len(load_dist_results_sorted) > 0 and len(time_dist_results_sorted) > 0:
    print("  - qq_plots.png")
print()

# Summary
print("=" * 80)
print("SUMMARY OF FINDINGS")
print("=" * 80)
print()
print("1. Range of load and time for each manufacturer:")
print(manufacturer_range.to_string())
print()
print(f"2. Most expected load value: {most_expected_load:.4f}")
print(f"   (Mean: {mean_load:.4f}, Median: {median_load:.4f})")
print()
print(f"3. Load and time relationship: {relationship} ({direction} correlation)")
print(f"   Correlation coefficient: {correlation:.4f}")
print()
if len(load_dist_results_sorted) > 0:
    print(f"4. Best distribution for load: {best_load_dist['distribution']}")
else:
    print("4. Best distribution for load: Could not determine")
print()
if len(time_dist_results_sorted) > 0:
    print(f"5. Best distribution for time: {best_time_dist['distribution']}")
else:
    print("5. Best distribution for time: Could not determine")
print()
print(f"6. Best performing manufacturer: {best_overall_manufacturer}")
print(f"   (Based on composite score considering load, time, efficiency, and consistency)")
print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)