import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# Load the .npy files   
synthetic = np.load("data/syn_seqs/...")
original = np.load("data/ori_seqs/...")

# Calculate moments across rows (i.e., across samples for each sequence)
def calculate_moments(data):
    return pd.DataFrame({
        'mu1': np.mean(data, axis=1),
        'mu2': np.std(data, axis=1),
        'mu3': skew(data, axis=1),
        'mu4': kurtosis(data, axis=1)
    })

df_synthetic = calculate_moments(synthetic)
df_original = calculate_moments(original)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
moment_names = ['mu1', 'mu2', 'mu3', 'mu4']
titles = [r'$\mu_1$', r'$\mu_2$', r'$\mu_3$', r'$\mu_4$']

colors = {
    'Original': '#70B2E4',   # soft blue
    'Synthetic': '#CC9633',   # soft orange
}

for i, ax in enumerate(axes.flat):
    moment = moment_names[i]
    
    # Shared bins for consistency
    data_combined = np.concatenate([df_original[moment], df_synthetic[moment]])
    bins = np.histogram_bin_edges(data_combined, bins=30)
    
    # Plot Original
    sns.histplot(df_original[moment], bins=bins, kde=True, label='Original', ax=ax,
                 color=colors['Original'], stat='count', element="bars",
                 edgecolor='black', alpha=0.5)

    # Plot Synthetic
    sns.histplot(df_synthetic[moment], bins=bins, kde=True, label='Synthetic', ax=ax,
                 color=colors['Synthetic'], stat='count', element="bars",
                 edgecolor='black', alpha=0.5)
    
    ax.set_xlabel(titles[i])
    ax.set_ylabel("Count")
    ax.legend()