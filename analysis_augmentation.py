# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

df_raw = pd.read_csv('results/backtest/scores.csv',sep=';')
df_raw.rename(columns={'data_name':'dataset'},inplace=True)

df = df_raw.groupby(['model','dataset','metric','representation']).median(numeric_only=True)
df = df.reset_index()

# Combine model and representation
df_raw['model_representation'] = df_raw['model'] + '_' + df_raw['representation']
df_raw = df_raw.reset_index(drop=True)
df_raw['rank'] = None

# Define pretty name mapping
pretty_name_map = {
    'Diffusion_ts': 'TS-Diffusion',
    'TimeGAN_ts': 'TS-TimeGAN',
    'WGAN_GP_gasf': 'GASF',
    'WGAN_GP_xirp': 'XIRP',
    'WGAN_GP_naive': 'Naive'
}

# Rank pred_score (lower is better)
mask_pred = (df_raw['metric'] == 'pred_score')
df_pred = df_raw[mask_pred].copy()
df_pred['rank'] = df_pred.groupby(['dataset', 'column'])['score'].rank(ascending=True, method='min')

# Rank disc_score (higher is better)
mask_disc = (df_raw['metric'] == 'disc_score')
df_disc = df_raw[mask_disc].copy()
df_disc['rank'] = df_disc.groupby(['dataset', 'column'])['score'].rank(ascending=False, method='min')

# Combine ranked results
ranked_df = pd.concat([df_pred, df_disc], axis=0).reset_index(drop=True)

# Filter for best (rank == 1)
best_ranked = ranked_df[ranked_df['rank'] == 1]

# Count how often each model_representation was best per dataset
best_counts = (
    best_ranked.groupby(['dataset', 'model_representation'])
    .size()
    .reset_index(name='count')
)

# Apply the name mapping
best_counts['model_representation'] = best_counts['model_representation'].map(pretty_name_map)

# Pivot to your desired table format
pivot_counts = (
    best_counts.pivot(index='dataset', columns='model_representation', values='count')
    .fillna(0)
    .astype(int)
)

desired_order = ['TS-Diffusion', 'TS-TimeGAN', 'GASF', 'XIRP', 'Naive']
pivot_counts = pivot_counts.reindex(columns=desired_order)

df['rmse_ori_syn'] = df['rmse_ori_syn']/df['rmse_ori']
df['mae_ori_syn'] = df['mae_ori_syn']/df['mae_ori']
df['mape_ori_syn'] = df['mape_ori_syn']/df['mape_ori']

df['rmse_ori'] = df['rmse_ori']/df['rmse_ori']
df['mae_ori'] = df['mae_ori']/df['mae_ori']
df['mape_ori'] = df['mape_ori']/df['mape_ori']

df = df.reset_index()

df['model'] = df.model.str.replace('_WGAN_GP','WGAN-GP').replace('_TimeGAN','TimeGAN')
#df = df.reset_index()
df['dataset'] = df.dataset.str.replace('_',' ').str.title()
df

df_raw.groupby(['dataset','model','rep']).median()

columns = ['dataset','model','mae_ori','mae_ori_syn','mape_ori','mape_ori_syn','rmse_ori','rmse_ori_syn']
df = df[columns]
print(df)
latex_table = df.reset_index().to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.4f}".format)
latex_table



# Combine 'dataset' and 'model' columns with the sorted columns
result = df.pivot_table(index='dataset', columns='model', values=['rmse_ori','mae_ori','mape_ori','rmse_ori_syn','mae_ori_syn','mape_ori_syn'])
result

result.median().round(4)

df_filtered = df_raw.groupby(['dataset','model','rep']).median().reset_index()
df_filtered['model'] = df_raw.model.str.replace('_WGAN_GP','WGAN-GP').replace('_TimeGAN','TimeGAN')
df_wgan = df_filtered[df_filtered.model=='WGAN-GP']
df_timegan = df_filtered[df_filtered.model=='TimeGAN'].groupby(['dataset','model']).median().reset_index()
df_timegan

df_comb = pd.concat([df_wgan,df_timegan],axis=0).sort_values(['dataset','model'])
df_comb['rep'] = df_comb['rep'].fillna('time series')
df_comb

df_comb['rmse_ori_syn'] = df_comb['rmse_ori_syn']/df_comb['rmse_ori']
df_comb['mae_ori_syn'] = df_comb['mae_ori_syn']/df_comb['mae_ori']
df_comb['mape_ori_syn'] = df_comb['mape_ori_syn']/df_comb['mape_ori']

df_comb['rmse_ori'] = df_comb['rmse_ori']/df_comb['rmse_ori']
df_comb['mae_ori'] = df_comb['mae_ori']/df_comb['mae_ori']
df_comb['mape_ori'] = df_comb['mape_ori']/df_comb['mape_ori']

df_comb = df_comb.groupby('rep').median().reset_index().replace('unthresholded','naive')

df_comb = df_comb.drop(columns=['mape_ori_syn','rmse_ori','mae_ori','mape_ori'])
df_comb

latex_table = df_comb.to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.4f}".format)
latex_table