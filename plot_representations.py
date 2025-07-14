import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def irp(data: np.array, dim: int, eps: float):
    data = data.flatten()
    rec_mats = []
    for i in range(data.shape[0] + 1 - dim):
        ts = data[i:i + dim]
        rec_mat = np.zeros(shape=(dim, dim))
        for j in range(dim):
            for k in range(dim):
                rec_mat[j, k] = np.log(np.max([ts[j], eps]) / np.max([ts[k], eps]))

        if np.isfinite(rec_mat).all():
            rec_mats.append(rec_mat)

    rec_mats = np.squeeze(np.array(rec_mats))
    return rec_mats


def xirp(data: np.array, dim: int, eps: float):
    data = data.flatten()
    rec_mats = []
    for i in range(data.shape[0] + 1 - dim):
        ts = data[i:i + dim]
        rec_mat = np.zeros(shape=(dim, dim))
        for j in range(dim):
            for k in range(dim):
                if j == k:
                    rec_mat[j, k] = ts[j]
                else:
                    rec_mat[j, k] = np.log(np.max([ts[j], eps]) / np.max([ts[k], eps]))

        if np.isfinite(rec_mat).all():
            rec_mats.append(rec_mat)

    rec_mats = np.squeeze(np.array(rec_mats))
    return rec_mats


def naive(data: np.array, dim: int):
    data = data.flatten()
    rec_mats = []
    for i in range(data.shape[0] + 1 - dim):
        ts = data[i:i + dim]
        rec_mat = np.repeat(ts[:, np.newaxis], dim, axis=1)

        if np.isfinite(rec_mat).all():
            rec_mats.append(rec_mat)

    rec_mats = np.squeeze(np.array(rec_mats))
    return rec_mats


def gasf(data: np.array, dim: int):
    data = data.flatten()
    rec_mats = []
    for i in range(data.shape[0] + 1 - dim):
        ts = data[i:i + dim]
        rec_mat = np.zeros(shape=(dim, dim))
        for j in range(dim):
            for k in range(dim):
                rec_mat[j, k] = np.cos(np.arccos(ts[j]) + np.arccos(ts[k]))

        if np.isfinite(rec_mat).all():
            rec_mats.append(rec_mat)

    rec_mats = np.squeeze(np.array(rec_mats))
    return rec_mats


def thresholded(data: np.array, dim: int, eps: float):
    data = data.flatten()
    rec_mats = []
    for i in range(data.shape[0] + 1 - dim):
        ts = data[i:i + dim]
        rec_mat = np.zeros(shape=(dim, dim))
        for j in range(dim):
            for k in range(dim):
                rec_mat[j, k] = 1 if abs(ts[j] - ts[k]) < eps else 0

        if np.isfinite(rec_mat).all():
            rec_mats.append(rec_mat)

    rec_mats = np.squeeze(np.array(rec_mats))
    return rec_mats


# Prepare data
np.random.seed(1234)
scaler = MinMaxScaler()
n = 20
ts = np.array([1, 2, 4, 3, 4, 3, 5, 7, 6, 7, 5, 7, 8, 6, 9, 10, 11, 12, 10, 11]).reshape((-1, 1))
ts = scaler.fit_transform(ts).flatten()  # Flatten to 1D

# Generate recurrence images
img_naive = naive(ts, n)
img_gasf = gasf(ts, n)
img_irp = irp(ts, n, 0.0001)
img_xirp = xirp(ts, n, 0.0001)
img_rec = thresholded(ts, n, 0.05)

# Plot time series
data = pd.DataFrame()
data['ts'] = ts

sns.set(rc={'figure.figsize': (11.7, 11.7)})
g = sns.lineplot(data=data, legend=False)
g.set_xticks(range(n))
plt.savefig('ts.png', dpi=200)
plt.close()

# Plot recurrence images
sns.set(rc={'figure.figsize': (11.7, 11.7)})
for img, name in zip([img_gasf, img_irp, img_xirp, img_naive, img_rec],
                     ['img_gasf', 'img_irp', 'img_xirp', 'img_naive', 'img_rec']):
    ax = sns.heatmap(img, annot=False, cmap='Blues', square=True)
    ax.invert_yaxis()
    #plt.title(name)
    plt.savefig(name + '.png', dpi=200)
    plt.close()
