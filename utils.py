import numpy as np
#from numpy.lib.index_tricks import diag_indices
#from pandas.io.common import stringify_path
from statsmodels.tsa.arima_process import ArmaProcess
import pandas as pd
#import matplotlib.pyplot as plt
from os.path import exists
from csv import writer
from datetime import datetime
#import torch
from torch.utils.data import Dataset
import os

# datetime object containing current date and time
def train_test_val_split_idxs(data,seq_len,train_frac,test_frac):
  if not (train_frac+test_frac)<1:
    raise AssertionError('Sum of Fractions must be <1')

  idxs = np.arange(seq_len,data.shape[0])
  np.random.shuffle(idxs)
  split1 = int(train_frac*idxs.shape[0])
  split2 = int((train_frac+test_frac)*idxs.shape[0])
  train_idxs = idxs[:split1]
  test_idxs = idxs[split1:split2]
  val_idxs = idxs[split2:]
  return train_idxs,test_idxs,val_idxs

def scaler(data,a,b):
    
    minimum = np.min(data)
    maximum = np.max(data)

    rec_mats = a+((data - minimum)*(b-a))/(maximum-minimum)
    
    return rec_mats,minimum,maximum

def descaler(data_scaled,mini,maxi,a,b):

    data = (maxi*(a-data_scaled)+mini*(data_scaled-b))/(a-b)

    return data

def create_ts_chunks(ori_data,args):
    ori_data = [ori_data[i:i+args.sequence_length] for i in range(ori_data.shape[0]-args.sequence_length)]
    return ori_data
        
def ar1_process(nsample, sigma,c):
    ar1 = np.array([1, -sigma])
    ma1 = np.array([1])
    AR_object1 = ArmaProcess(ar1, ma1)
    simulated_data = AR_object1.generate_sample(nsample=nsample).reshape((-1,1))+c
    return simulated_data

def read_stock_price_data(path):
    data = pd.read_csv(path, delimiter=",")['Adj_Close'] # only adjusted close
    # Flip to make chronological data
    data = data[::-1]
    data = np.expand_dims(data, axis=1)
    print('Stock Dataset is ready')
    return data

def write_results(model,data_name,column,representation,recovery_method,metric,score,std):
    now = datetime.now()
    date_time = now.strftime("%d/%m/%Y %H:%M:%S")
    headers = ['model','data_name','column','representation','recovery_method','metric','score','std','datetime']
    data = [model,data_name,column,representation,recovery_method,metric,score,std,date_time]
    file_exists = exists('results/backtest/scores.csv')
    os.makedirs('results/backtest', exist_ok=True)

    with open('results/backtest/scores.csv', 'a+',newline='') as f_object:
      # Pass this file object to csv.writer()
      # and get a writer object
      writer_object = writer(f_object)
      # Pass the list as an argument into
      # the writerow()
      if file_exists==False:
        writer_object.writerow(headers)
      
      writer_object.writerow(data)

      #Close the file object
      f_object.close()

def save_seqs(data_type, data, args):
    if data_type == "ori":
        path = 'data/{}_seqs/{}_data_seqs_{}_{}.npy'.format(
            data_type, data_type, args.model, args.data_name)
    elif data_type == "syn":
        if args.model in ["WGAN_GP", "Diffusion"]:
            path = 'data/{}_seqs/{}_data_seqs_{}_{}_{}.npy'.format(
                data_type, data_type, args.model, args.data_name, args.column)
        else:
            path = 'data/{}_seqs/{}_data_seqs_{}_{}_{}_{}.npy'.format(
                data_type, data_type, args.model, args.data_name, args.column, args.recovery_method, args.representation)
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")
    
    np.save(path, data)


def load_seqs(data_type, args):
    if data_type == "ori":
        path = 'data/{}_seqs/{}_data_seqs_{}_{}.npy'.format(
            data_type, data_type, args.model, args.data_name)
    elif data_type == "syn":
        if args.model in ["WGAN_GP", "Diffusion"]:
            path = 'data/{}_seqs/{}_data_seqs_{}_{}_{}.npy'.format(
                data_type, data_type, args.model, args.data_name, args.column)
        else:
            path = 'data/{}_seqs/{}_data_seqs_{}_{}_{}_{}.npy'.format(
                data_type, data_type, args.model, args.data_name, args.column, args.recovery_method, args.representation)
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")
    
    return np.load(path)


def irp(data:np.array,dim:int,eps:float):
    rec_mats = []
    for i in range(data.shape[0]+1 - dim):
      ts = data[i:i + dim]
      rec_mat = np.zeros(shape=(dim, dim))
      for j in range(dim):
        for k in range(dim):
          rec_mat[j,k] = np.log(np.max([ts[j],eps])/np.max([ts[k],eps]))
      
      if np.isfinite(rec_mat).all():
        rec_mats.append(rec_mat)
        
    rec_mats = np.expand_dims(np.squeeze(np.array(rec_mats)),axis=3)
    return rec_mats

def recover_irp(start:int,imgs:np.array,method:str):
    recovered_ts = []
    sequence_length = imgs.shape[1]
    for i in range(imgs.shape[0]):
      recovered = np.zeros((sequence_length-1,sequence_length))
      image = np.exp(imgs[i,:,:])

      np.fill_diagonal(image,1)
      initial_ts = image[:,0]*start

      for j in range(0,sequence_length-1):
        recovered[j,:] = initial_ts[j]*image[:,j]
      
      recovered = np.concatenate([np.expand_dims(initial_ts,axis=0),recovered],axis=0)
      if method=='columns_mean':
        recovered = np.mean(recovered,axis=0)
      elif method=='columns_random':
        recovered = recovered[np.random.randint(recovered.shape[0]),:]
      else:
        raise ValueError('IRP Recovery Method not accepted ')

      if np.isfinite(recovered).all():
        recovered_ts.append(recovered)
      
    recovered_ts = np.array(recovered_ts)

    return recovered_ts

def xirp(data:np.array,dim:int,eps:float):
    # 1 sequentialize the data
    rec_mats = []
    ts = []
    for i in range(data.shape[0]+1 - dim):
      ts_ = data[i:i + dim]
      rec_mat = np.zeros(shape=(dim, dim))
      for j in range(dim):
        for k in range(dim):
          if j==k:
            rec_mat[j,k] = float(ts_[j])
          else:
            rec_mat[j,k] = np.log(np.max([float(ts_[j]),eps])/np.max([float(ts_[k]),eps]))

      if np.isfinite(rec_mat).all():
        rec_mats.append(rec_mat)
        ts.append(ts_)

    rec_mats = np.expand_dims(np.squeeze(np.array(rec_mats)),axis=3)
    ts = np.array(ts)
    return rec_mats,ts

def recover_xirp(imgs:np.array,method:str):
    sequence_length = imgs.shape[1]
    recovered_ts = []
    for i in range(imgs.shape[0]):
      img = np.squeeze(imgs[i,:,:])
      if method=='diagonal':
        #extract the diagonal
        recovered_ts.append(np.diag(img))
      
      if method=='columns_random':
        start = np.diag(img)
        img = np.exp(img)
        np.fill_diagonal(img,1)
        rec_mat = img*start
        rec_mat = rec_mat[:,np.random.randint(img.shape[0])]
        recovered_ts.append(rec_mat)
      
      elif method=='columns_mean':
        start = np.diag(img)
        img = np.exp(img)
        np.fill_diagonal(img,1)
        rec_mat = img*start
        rec_mat = np.mean(rec_mat,axis=1)
        recovered_ts.append(rec_mat)

      else:
        raise ValueError('recover_xirp: Method not accepted')
      


    recovered_ts = np.array(recovered_ts)

    return recovered_ts
    
def naive(data:np.array,dim:int):
        # 1 sequentialize the data
    rec_mats = []
    for i in range(data.shape[0]+1 - dim):
      ts = data[i:i + dim]
      rec_mat = np.repeat(ts,dim,axis=2)

      if np.isfinite(rec_mat).all():
        rec_mats.append(rec_mat)

    rec_mats = np.expand_dims(np.squeeze(np.array(rec_mats)),axis=3)
    return rec_mats


def recover_naive(imgs:np.array,method:str):
    sequence_length = imgs.shape[1]
    recovered_ts = []
    for i in range(imgs.shape[0]):
      img = np.squeeze(imgs[i,:,:])
      
      if method=='columns_random':
        rec_mat = img[:,np.random.randint(img.shape[0])]
        recovered_ts.append(rec_mat)
      
      elif method=='columns_mean':
        rec_mat = np.mean(img,axis=1)
        recovered_ts.append(rec_mat)

      else:
        raise ValueError('recover_naive: Method not accepted')

    recovered_ts = np.array(recovered_ts)

    return recovered_ts

def gasf(data:np.array,dim:int):
    rec_mats = []
    for i in range(data.shape[0]+1 - dim):
      ts = data[i:i + dim]
      rec_mat = np.zeros(shape=(dim, dim))
      for j in range(dim):
        for k in range(dim):
          rec_mat[j,k] = np.cos(np.arccos(ts[j])+np.arccos(ts[k]))
      
      if np.isfinite(rec_mat).all():
        rec_mats.append(rec_mat)

    rec_mats = np.expand_dims(np.squeeze(np.array(rec_mats)),axis=3)
  
    return rec_mats

def recover_gasf(imgs:np.array,method:str):
    dim = imgs.shape[1]
    recovered_ts = []
    for i in range(imgs.shape[0]):
      img = imgs[i,:,:] 
      diag = np.cos(1/2*np.arccos(np.diag(img)))
      rec_mat = np.zeros(shape=(dim, dim))
      for j in range(img.shape[0]):
        for k in range(img.shape[0]):
          if j!=k:
            rec_mat[j,k] = np.cos(np.arccos(img[j,k])-np.arccos(diag[j]))
      np.fill_diagonal(rec_mat,diag)

      if method=='columns_random':
        
        rec_mat = rec_mat[np.random.randint(img.shape[0]),:]
        if np.isfinite(rec_mat).all():
          recovered_ts.append(rec_mat)
      
      elif method=='columns_mean':
        rec_mat = np.mean(rec_mat,axis=0)
        if np.isfinite(rec_mat).all():
          recovered_ts.append(rec_mat)
      
      else:
        raise ValueError('recover_gasf: Method not accepted')

    return np.array(recovered_ts)

def unthresholded(data:np.array,dim:int):
    rec_mats = []
    for i in range(data.shape[0]+1 - dim):
      ts = data[i:i + dim]
      rec_mat = np.zeros(shape=(dim, dim))
      for j in range(dim):
        for k in range(dim):
          rec_mat[j,k] = ts[j]-ts[k]
      
      if np.isfinite(rec_mat).all():
        rec_mats.append(rec_mat)
        
    rec_mats = np.expand_dims(np.squeeze(np.array(rec_mats)),axis=3)
    return rec_mats

def recover_unthresholded(start,imgs,method):
  recovered_ts = []
  sequence_length = imgs.shape[1]

  for i in range(imgs.shape[0]):
    recovered = np.zeros((sequence_length-1,sequence_length))
    image = np.squeeze(imgs[i,:,:])
    image = np.copy(image)
    np.fill_diagonal(image,0)
    initial_ts = image[:,0]+start

    for j in range(0,sequence_length-1):
      recovered[j,:] = initial_ts[j]+image[:,j]
    
    if method=='columns_mean':
      recovered = np.mean(recovered,axis=0)
    elif method=='columns_random':
      recovered = recovered[np.random.randint(recovered.shape[0]),:]
    else:
      raise ValueError('recover_zero_threshold: Method not accepted')

    if np.isfinite(recovered_ts).all():
      recovered_ts.append(recovered)
    
  recovered_ts = np.array(recovered_ts)

  return recovered_ts

def scale_image_rep(data:np.array,a:float,b:float,method:str):
  res = {}
  res['a'] = a
  res['b'] = b

  if method=='default':
    res['min_img'] = np.min(data)
    res['max_img'] = np.max(data)

    res['rec_mats'] = a+((data - res['min_img'])*(b-a))/(res['max_img']-res['min_img'])

  elif method=='diagonal_separately':

    res['diags'] = []
    for img in data[0]:
      if np.squeeze(img).ndim==2:
        res['diags'].append(np.diag(np.squeeze(img)))

    res['diags'] = np.array(res['diags'])

    print('res diags shape ', res['diags'].shape)

    res['imgs'] = []
    for img in data[0]:
      if np.squeeze(img).ndim==2:
        np.fill_diagonal(np.squeeze(img),0)
      res['imgs'].append(img)
    res['imgs'] = np.array(res['imgs'])

    res['min_img'] = np.min(res['imgs'])
    res['max_img'] = np.max(res['imgs'])

    res['imgs'] = a+((res['imgs'] -res['min_img'])*(b-a))/(res['max_img']-res['min_img'])

    res['min_diag'] = np.min(res['diags'])
    res['max_diag'] = np.max(res['diags'])

    res['diags'] = a+((res['diags'] - res['min_diag'])*(b-a))/(res['max_diag']-res['min_diag'])
    
    res['rec_mats'] = res['imgs']
    for i in range(res['imgs'].shape[0]):
      for j in range(res['imgs'].shape[1]):
        res['rec_mats'][i,j,j] = res['diags'][i,j]

  else:
    raise ValueError('Method not accepted')
  
  return res

def descale_img_rep(data_scaled:np.array,method:str,res:dict):
    diags = []
    # extract diagonals
    for img in data_scaled:
      img = np.copy(np.squeeze(img))
      diags.append(np.diag(img))
    diags = np.array(diags)

    # extract images
    imgs = []
    for img in data_scaled:
      img = np.copy(np.squeeze(img))
      np.fill_diagonal(img,0)
      imgs.append(img)
    imgs = np.array(imgs)
    
    if method=='default':
      res['rec_mats_descaled'] = (res['max_img']*(res['a']-data_scaled)+res['min_img']*(data_scaled-res['b']))/(res['a']-res['b'])
      imgs_descaled = res['rec_mats_descaled']

    elif method=='diagonal_separately':
      diags_descaled = (res['max_diag']*(res['a']-diags)+res['min_diag']*(diags-res['b']))/(res['a']-res['b'])
      imgs_descaled = (res['max_img']*(res['a']-imgs)+res['min_img']*(imgs-res['b']))/(res['a']-res['b'])

      for j in range(imgs_descaled.shape[0]):
        for i in range(imgs_descaled.shape[1]):
          imgs_descaled[j,i,i] = diags_descaled[j,i]
    
    else:
      raise ValueError('descale_img_rep: Method not accepted')

    return imgs_descaled


class Image_TS_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]