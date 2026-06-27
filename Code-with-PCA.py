#%%
# ============================================================
# Assignment 2 - Operations data analysis
# Each section is labelled with the question it answers, so the
# printed output and the plots map directly to the report.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

#%%
# ------------------------------------------------------------
# Read data
# NOTE: confirm the filename. The starting code used
# 'machine_data-1.csv'; the assignment text refers to
# 'machine_data.csv'. Change the line below to match your file.
# ------------------------------------------------------------
df = pd.read_csv('machine_data-1.csv')

# Print columns and a preview so you can confirm the exact names.
print("Columns:", list(df.columns))
print(df.head())
print("Shape:", df.shape)

# Drop the leftover row-index column if present (e.g. 'Unnamed: 0').
df = df.drop(columns=[c for c in df.columns if c.lower().startswith('unnamed')],
             errors='ignore')

# The column is literally named 'manufacturef' in the file (a typo
# baked into the data, not something to rename). Detect it instead of
# hard-coding it.
manu_col = next(c for c in df.columns if c.lower().startswith('manufact'))
print("Using manufacturer column:", manu_col)

# Manufacturer C is stored as lowercase 'c' while A and B are uppercase.
# Normalise to uppercase so tables and plots read A, B, C consistently.
df[manu_col] = df[manu_col].astype(str).str.upper()

# Assumes the value columns are named 'load' and 'time' (as in the
# starting code). Adjust here if the printed columns differ.
manufacturers = sorted(df[manu_col].unique())
print("Manufacturers found:", manufacturers)


#%%
# ============================================================
# Q1: Range of load and time for each manufacturer
# range = max - min
# ============================================================
q1 = df.groupby(manu_col).agg(
    load_min=('load', 'min'),
    load_max=('load', 'max'),
    time_min=('time', 'min'),
    time_max=('time', 'max'),
)
q1['load_range'] = q1['load_max'] - q1['load_min']
q1['time_range'] = q1['time_max'] - q1['time_min']

print("\n=== Q1: Range of load and time per manufacturer ===")
print(q1)


#%%
# ============================================================
# Q2: Most expected load value
# "Most expected" is ambiguous. It can mean the expected value
# (mean) or the most likely value (the peak of the distribution).
# Load is continuous, so the raw mode is meaningless (almost every
# value is unique). The most likely value is the centre of the
# tallest histogram bar (the modal bin). Both are reported; state
# in the report which one you use.
# ============================================================
load_all = df['load']
mean_load = load_all.mean()
median_load = load_all.median()

counts, edges = np.histogram(load_all, bins=20)
modal_bin = (edges[counts.argmax()] + edges[counts.argmax() + 1]) / 2

print("\n=== Q2: Most expected load value (all data) ===")
print(f"Mean (expected value):      {mean_load:.3f}")
print(f"Median:                     {median_load:.3f}")
print(f"Modal bin centre (peak):    {modal_bin:.3f}")

plt.figure()
plt.hist(load_all, bins=20, edgecolor='black')
plt.axvline(mean_load, color='red', linestyle='--', label=f'Mean {mean_load:.2f}')
plt.title("Q2: Distribution of load (all manufacturers)")
plt.xlabel("Load"); plt.ylabel("Frequency"); plt.legend()
plt.show()


#%%
# ============================================================
# Q3: Relationship between load and time
# Physically, a run-to-failure part usually fails sooner under
# higher load, so expect a negative correlation. Let the data
# confirm it rather than assuming.
# ============================================================
plt.figure()
for m in manufacturers:
    sub = df[df[manu_col] == m]
    plt.scatter(sub['load'], sub['time'], label=m, alpha=0.6)
plt.title("Q3: Load vs time by manufacturer")
plt.xlabel("Load"); plt.ylabel("Time"); plt.legend()
plt.show()

print("\n=== Q3: Correlation between load and time ===")
print(f"Pearson (overall):  {df['load'].corr(df['time']):.3f}")
print(f"Spearman (overall): {df['load'].corr(df['time'], method='spearman'):.3f}")
for m in manufacturers:
    sub = df[df[manu_col] == m]
    print(f"  {m}: Pearson = {sub['load'].corr(sub['time']):.3f}")

#%%
# ============================================================
# Q3 tillägg: PCA, linjär regression och residualplot
# PCA används här som referens mot de två kompletterande
# metoderna: regression (kvantifierar sambandet) och
# residualplot (verifierar att sambandet är linjärt).
# ============================================================
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as skPCA

# --- PCA (referensverktyg) ---
# Med bara två variabler sammanfattar PCA i princip samma
# information som korrelationen. PC1 visar den riktning längs
# vilken datan varierar mest, dvs. det negativa sambandet
# last/tid som vi redan sett.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['load', 'time']])
pca = skPCA(n_components=2)
pca.fit(X_scaled)

print("\n=== Q3 tillägg: PCA ===")
print(f"Förklarad varians per komponent: {pca.explained_variance_ratio_.round(3)}")
print(f"PC1 förklarar {pca.explained_variance_ratio_[0]*100:.1f}% av variansen")
print("PC1-riktning (load, time):", pca.components_[0].round(3))

plt.figure()
X_pca = pca.transform(X_scaled)
colors = {'A': 'steelblue', 'B': 'darkorange', 'C': 'seagreen'}
for m in manufacturers:
    idx = df[manu_col] == m
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1],
                label=m, alpha=0.6, color=colors[m])
plt.axhline(0, color='grey', lw=0.8, linestyle='--')
plt.axvline(0, color='grey', lw=0.8, linestyle='--')
plt.title("Q3 PCA: score-plot (PC1 vs PC2)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend(); plt.tight_layout(); plt.show()

# --- Linjär regression per tillverkare (nytt verktyg 1) ---
# Ger lutningskoefficienten: hur många tidsenheter kortare
# livslängd per enhet ökad last. Konkretiserar Q3 mer än
# korrelation och PCA gör.
print("\n=== Q3 tillägg: Linjär regression per tillverkare ===")
fig, axes = plt.subplots(1, len(manufacturers), figsize=(12, 4), sharey=True)
for ax, m in zip(axes, manufacturers):
    sub = df[df[manu_col] == m]
    slope, intercept, r, p, se = stats.linregress(sub['load'], sub['time'])
    x_line = np.linspace(sub['load'].min(), sub['load'].max(), 100)
    ax.scatter(sub['load'], sub['time'], alpha=0.4, color=colors[m], s=15)
    ax.plot(x_line, slope * x_line + intercept, color='black', lw=1.5)
    ax.set_title(f"{m}: slope={slope:.3f}\nr²={r**2:.3f}")
    ax.set_xlabel("Load")
    print(f"  {m}: slope={slope:.3f}, intercept={intercept:.2f}, "
          f"r²={r**2:.3f}, p={p:.2e}")
axes[0].set_ylabel("Time")
fig.suptitle("Q3: Linjär regression last → tid per tillverkare")
plt.tight_layout(); plt.show()

# --- Residualplot (nytt verktyg 2) ---
# Visar om det linjära sambandet räcker eller om det finns
# kvarvarande mönster i felet. Slumpmässigt spridda residualer
# bekräftar att en rät linje beskriver sambandet väl.
print("\n=== Q3 tillägg: Residualplot ===")
plt.figure()
for m in manufacturers:
    sub = df[df[manu_col] == m]
    slope, intercept, *_ = stats.linregress(sub['load'], sub['time'])
    fitted = slope * sub['load'] + intercept
    residuals = sub['time'] - fitted
    plt.scatter(fitted, residuals, label=m, alpha=0.5, color=colors[m], s=15)
plt.axhline(0, color='black', lw=1, linestyle='--')
plt.title("Q3: Residualer vs anpassat värde")
plt.xlabel("Anpassat värde (tid)"); plt.ylabel("Residual")
plt.legend(); plt.tight_layout(); plt.show()


#%%
# ============================================================
# Q4 & Q5: Which distribution best describes load / time?
# Fit candidate distributions and rank them by AIC (lower is
# better). AIC only ranks relative fit; it does not prove the
# true law. Read it together with the overlaid PDF plot and
# domain knowledge (time-to-failure data is classically Weibull).
# ============================================================
candidates = {
    'normal':      stats.norm,
    'uniform':     stats.uniform,
    'exponential': stats.expon,
    'weibull_min': stats.weibull_min,
}

def fit_and_rank(data, title):
    data = np.asarray(data, dtype=float)
    results = []
    fitted = {}
    for label, dist in candidates.items():
        try:
            params = dist.fit(data)
            ll = np.sum(dist.logpdf(data, *params))
            aic = 2 * len(params) - 2 * ll
            results.append((label, aic))
            fitted[label] = params
        except Exception:
            results.append((label, np.inf))

    results.sort(key=lambda r: r[1])
    print(f"\n=== {title}: distribution ranking by AIC (lower = better) ===")
    for label, aic in results:
        print(f"  {label:12s} AIC = {aic:.2f}")
    best = results[0][0]
    print(f"  -> Best fit: {best}")

    # Histogram with all fitted PDFs overlaid for visual comparison.
    plt.figure()
    plt.hist(data, bins=20, density=True, edgecolor='black', alpha=0.5)
    x = np.linspace(data.min(), data.max(), 200)
    for label, _ in results:
        if label in fitted:
            plt.plot(x, candidates[label].pdf(x, *fitted[label]), lw=2, label=label)
    plt.title(f"{title}: histogram with fitted distributions")
    plt.xlabel(title); plt.ylabel("Density"); plt.legend()
    plt.show()
    return best

best_load = fit_and_rank(df['load'], "Q4 Load")
best_time = fit_and_rank(df['time'], "Q5 Time")


#%%
# ============================================================
# Q6: Which manufacturer has the best performance?
# The assignment does not define "best performance", so you must
# state your criterion in the report. Reasonable options for a
# run-to-failure part:
#   - longest time to failure (durability)
#   - highest load tolerated
#   - lowest variability (most consistent / most reliable)
# Caveat: if the manufacturers were tested at different load
# levels, comparing raw mean time is unfair, because higher load
# shortens life. Check mean_load below; if it differs a lot
# between A/B/C, compare time at comparable loads instead of
# comparing raw averages.
# ============================================================
q6 = df.groupby(manu_col).agg(
    n=('time', 'size'),
    mean_time=('time', 'mean'),
    median_time=('time', 'median'),
    std_time=('time', 'std'),
    mean_load=('load', 'mean'),
    std_load=('load', 'std'),
)
print("\n=== Q6: Performance summary per manufacturer ===")
print(q6)

# Side-by-side boxplots to compare the spread of time and load.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df.boxplot(column='time', by=manu_col, ax=axes[0]); axes[0].set_title("Time by manufacturer")
df.boxplot(column='load', by=manu_col, ax=axes[1]); axes[1].set_title("Load by manufacturer")
plt.suptitle(""); plt.tight_layout(); plt.show()