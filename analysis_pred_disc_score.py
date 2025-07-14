# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


df = pd.read_csv('/results/backtest/scores.csv', index_col=0)

# Keep only pred_scores with a matching disc_score
valid_keys = df[df['metric'] == 'disc_score'][['model', 'data_name', 'column', 'representation']].drop_duplicates()
df = df.merge(valid_keys, on=['model', 'data_name', 'column', 'representation'], how='inner')

# Clean 'Diffusion' recovery_method and drop duplicates
df.loc[df.model == 'Diffusion', 'recovery_method'] = np.nan


def summarize_scores(df_subset, metric_label, column_map, desired_order):
    """Generate summary matrix and average rank summary for a given metric subset."""
    best_counts_per_dataset = {}
    avg_ranks_all = []

    for dataset, df_group in df_subset.groupby('data_name'):
        df_pivot = df_group.pivot_table(
            index='column',
            columns=['model', 'representation'],
            values='score'
        ).sample(frac=1)  # Shuffle to break ties randomly

        df_ranks = df_pivot.rank(axis=1, method='first', ascending=True)

        best_counts = (df_ranks == 1).sum().reset_index()
        best_counts.columns = ['model', 'representation', 'n_best_scores']
        best_counts['data_name'] = dataset
        best_counts_per_dataset[dataset] = best_counts

        df_avg_ranks = df_ranks.mean().reset_index()
        df_avg_ranks.columns = ['model', 'representation', 'avg_rank']
        df_avg_ranks['data_name'] = dataset
        avg_ranks_all.append(df_avg_ranks)

    final_best_counts = pd.concat(best_counts_per_dataset.values(), ignore_index=True)
    final_avg_ranks = pd.concat(avg_ranks_all, ignore_index=True)

    summary_matrix = final_best_counts.pivot_table(
        index='data_name',
        columns=['model', 'representation'],
        values='n_best_scores',
        fill_value=0
    )
    summary_matrix.columns = ['_'.join(col).strip() for col in summary_matrix.columns.values]
    summary_matrix = summary_matrix.rename(columns=column_map)
    summary_matrix = summary_matrix.reindex(columns=desired_order)

    avg_rank_summary = final_avg_ranks.copy()
    avg_rank_summary['model_rep'] = avg_rank_summary['model'] + '_' + avg_rank_summary['representation']
    avg_rank_summary = avg_rank_summary.groupby('model_rep')['avg_rank'].mean().reset_index()
    avg_rank_summary['model_rep'] = avg_rank_summary['model_rep'].map(column_map)
    avg_rank_summary = avg_rank_summary.dropna().set_index('model_rep').loc[desired_order]

    # Output
    print(f"\n=== Best Score Counts for {metric_label} ===")
    print(summary_matrix)
    print("\nColumn Totals:")
    print(summary_matrix.sum())
    print("\nRow Totals:")
    print(summary_matrix.sum(axis=1))
    print("\nGrand Total:")
    print(summary_matrix.sum().sum())

    print(f"\n=== Average Ranks (lower is better) for {metric_label} ===")
    print(avg_rank_summary.round(2))


column_map = {
    'TimeGAN_ts': 'TS',
    'WGAN_GP_gasf': 'GASF',
    'WGAN_GP_xirp': 'XIRP',
    'WGAN_GP_naive': 'Naive',
    'Diffusion_ts': 'Diffusion'
}
desired_order = ['TS', 'GASF', 'XIRP', 'Naive', 'Diffusion']

# Summary for pred_score
df_pred = df[df.metric == 'pred_score']
summarize_scores(df_pred, metric_label='pred_score', column_map=column_map, desired_order=desired_order)

# Summary for disc_score
df_disc = df[df.metric == 'disc_score']
summarize_scores(df_disc, metric_label='disc_score', column_map=column_map, desired_order=desired_order)
