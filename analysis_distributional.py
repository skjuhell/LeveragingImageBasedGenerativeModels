# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import re
from scipy.stats import skew, kurtosis

def compute_moments(array):
    mean = array.mean(axis=1)
    var = array.var(axis=1)
    sk = skew(array, axis=1)
    kurt = kurtosis(array, axis=1)
    return np.mean(mean), np.mean(var), np.mean(sk), np.mean(kurt)

import os
import numpy as np
import pandas as pd

def analyze_matched_files(folder_ori, folder_syn):
    results = []

    for syn_file in os.listdir(folder_syn):
        try:
            if not syn_file.endswith('.npy'):
                continue

            filename = syn_file.replace('WGAN_GP', 'WGAN-GP').replace('.npy', '')
            parts = filename.split('_')

            if len(parts) < 6 or parts[0] != 'syn' or parts[1] != 'data' or parts[2] != 'seqs':
                print(f"Unexpected filename format: {syn_file}")
                continue

            model = parts[3].replace('WGAN-GP', 'WGAN_GP')
            data_name = parts[4]
            column = parts[5]

            if model in ["WGAN_GP", "Diffusion"]:
                recovery_method = None
                representation = 'ts'
                # syn_data_seqs_{model}_{data_name}_{column}.npy
            else:
                if len(parts) < 8:
                    print(f"Incomplete filename for model {model}: {syn_file}")
                    continue
                recovery_method = parts[6]
                representation = parts[7]

            ori_file = f"ori_data_seqs_{model}_{data_name}.npy"
            syn_path = os.path.join(folder_syn, syn_file)
            ori_path = os.path.join(folder_ori, ori_file)

            if not os.path.isfile(syn_path) or not os.path.isfile(ori_path):
                print(f"Missing file(s): {ori_path} or {syn_path}")
                continue

            arr1 = np.load(ori_path)
            arr2 = np.load(syn_path)

            if arr1.ndim != 2 or arr2.ndim != 2:
                print(f"Invalid shape for: {ori_file} or {syn_file}")
                continue

            # Compute moment differences
            m1_1, m2_1, m3_1, m4_1 = compute_moments(arr1)
            m1_2, m2_2, m3_2, m4_2 = compute_moments(arr2)

            results.append({
                'dataset': data_name,
                'column': column,
                'model': model,
                'recovery_method': recovery_method,
                'representation': representation,
                'moment_1_diff': abs(m1_1 - m1_2),
                'moment_2_diff': abs(m2_1 - m2_2),
                'moment_3_diff': abs(m3_1 - m3_2),
                'moment_4_diff': abs(m4_1 - m4_2)
            })

        except Exception as e:
            print(f"Error processing {syn_file}: {e}")
            continue

    return pd.DataFrame(results)



def rank_models_by_moments_sep_repr(df):
    df = df.copy()
    df['model_repr'] = df.apply(
        lambda row: row['model'] if row['representation'] == 'ts'
        else f"{row['model']}-{row['representation']}", axis=1
    )

    agg = df.groupby(['dataset', 'model_repr'])[
        ['moment_1_diff', 'moment_2_diff', 'moment_3_diff', 'moment_4_diff']
    ].mean().reset_index()

    ranked = []
    for dataset_name, group in agg.groupby('dataset'):
        group = group.copy()
        group['rank_m1'] = group['moment_1_diff'].rank(method='min')
        group['rank_m2'] = group['moment_2_diff'].rank(method='min')
        group['rank_m3'] = group['moment_3_diff'].rank(method='min')
        group['rank_m4'] = group['moment_4_diff'].rank(method='min')
        group['average_rank'] = group[['rank_m1', 'rank_m2', 'rank_m3', 'rank_m4']].mean(axis=1)
        ranked.append(group)

    summary = pd.concat(ranked).sort_values(['dataset', 'average_rank'])
    return summary

df_results = analyze_matched_files(
    folder_ori='data/ori_seqs',
    folder_syn='data/syn_seqs'
)

summary_ranking = rank_models_by_moments_sep_repr(df_results)

print("Summary ranking (top 5 per dataset):")
res = summary_ranking.groupby(['dataset','model_repr']).mean()

res = res.loc[:,'average_rank'].reset_index()

df_pivot = res.pivot_table(
    index='dataset',
    columns='model_repr',
    values='average_rank'
)
df_pivot

desired_order = ['TimeGAN', 'WGAN-GP-gasf', 'WGAN-GP-xirp', 'WGAN-GP-unthresholded', 'Diffusion']

# Apply column order (drop missing if some aren't in the pivot)
df_pivot = df_pivot.reindex(columns=[col for col in desired_order if col in df_pivot.columns])
df_pivot

df_pivot.to_latex()

latex_table = df_pivot.to_latex(
    index=True,
    float_format="%.2f",           
    na_rep="--",                   
    bold_rows=False,               
    column_format='l' + 'c' * len(df_pivot.columns),  # align left for row index, center for others
    caption='Model Ranking by Dataset and Representation',
    label='tab:model_ranking'
)

print(latex_table)