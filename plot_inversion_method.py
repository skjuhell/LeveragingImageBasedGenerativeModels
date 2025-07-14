import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load scores
scores = pd.read_csv("results/backtest/scores.csv", sep=",") 

# Filter for model WGAN_GP and the desired representations
filtered = scores[
    (scores['model'] == 'WGAN_GP') &
    (scores['representation'].isin(['gasf', 'xirp', 'naive']))
]

# Map recovery_method to Inversion Method
recovery_map = {
    'columns_mean': 'IM',
    'columns_random': 'IRC'
}
filtered['Inversion Method'] = filtered['recovery_method'].map(recovery_map)

# Rename columns for consistency
filtered = filtered.rename(columns={
    'representation': 'Representation',
    'metric': 'S',
    'score': 'Metric Score'
})

# Group by to calculate mean score per Representation, Metric, and Inversion Method
df_plot = (
    filtered.groupby(['Representation', 'S', 'Inversion Method'], as_index=False)
    .agg({'Metric Score': 'mean'})
)
print(df_plot)
# Set the color palette and style
palette = sns.color_palette(palette=['#CC9633','#70B2E4'], n_colors=2)
sns.set(rc={'figure.figsize': (13, 13)}, font_scale=2)

# Plot for S_D
fig = sns.barplot(
    data=df_plot[df_plot.S == 'S_D'],
    x="Representation",
    y="Metric Score",
    hue="Inversion Method",
    palette=palette
)
fig.set(ylabel=r'$S_D$')
plt.savefig('s_d.png', dpi=200)
plt.show()

# Plot for S_P
sns.set(rc={'figure.figsize': (12, 12)}, font_scale=2)
fig = sns.barplot(
    data=df_plot[df_plot.S == 'S_P'],
    x="Representation",
    y="Metric Score",
    hue="Inversion Method",
    palette=palette
)
fig.set(ylabel=r'$S_P$')
plt.savefig('s_p.png', dpi=200)
plt.show()
