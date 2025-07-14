import umap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 
import numpy as np
import time
from sklearn.manifold import TSNE

def create_umap_plot(ori_seq,syn_seq,suffix,title,elements):
  ori_seq = np.squeeze(ori_seq)
  syn_seq = np.squeeze(syn_seq)
  print('UMAP Clustering...')
  reducer = umap.UMAP()

  idxs = np.random.randint(0,ori_seq.shape[0],elements)
  joint_array= np.concatenate([ori_seq[idxs],syn_seq[idxs]],axis=0)
  
  embedding = reducer.fit_transform(joint_array)
  classes = ['original','synthetic']
  values = ['green']*int(embedding.shape[0]/2)+['red']*int(embedding.shape[0]/2)

  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1, 1, 1)
  scatter = ax.scatter(
    x=embedding[:, 0],
    y=embedding[:, 1],
    c = [0]*int(embedding.shape[0]/2)+[1]*int(embedding.shape[0]/2),
    cmap=ListedColormap(['#CC9633','#70B2E4']),
    alpha=.6,
    s=80
)
  classes = ['original','synthetic']
  ax.legend(handles=scatter.legend_elements()[0], labels=classes)
  #ax.set_title(title, fontsize=12)
  plt.savefig('results/visualization/umap_'+suffix+'.png')


def create_tsne_plot(ori_seq,syn_seq,suffix,title):
  print('t-SNE Clustering...')
  time_start = time.time()
  joint_array = np.concatenate([ori_seq,syn_seq],axis=0)
  #idxs = np.random.randint(0,joint_array.shape[0],50)

  tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
  embedding = tsne.fit_transform(joint_array)
  classes = ['original','synthetic']
  #values = ['green']*int(embedding.shape[0]/2)+['red']*int(embedding.shape[0]/2)

  print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1, 1, 1)
  scatter = ax.scatter(
    x=embedding[:, 0],
    y=embedding[:, 1],
    c = [0]*int(embedding.shape[0]/2)+[1]*int(embedding.shape[0]/2),
    cmap=ListedColormap(['#CC9633','#70B2E4']),
    alpha=.6,
    s=80
)
  classes = ['original','synthetic']
  ax.legend(handles=scatter.legend_elements()[0], labels=classes)
  #ax.set_title(title, fontsize=12)
  suffix_path = suffix.replace(' ','_')
  plt.savefig('results/visualization/tsne_'+suffix+'.png')