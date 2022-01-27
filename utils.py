import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
import pandas as pd
import matplotlib.pyplot as plt

def irps(data,dim):
    
    # 1 sequentialize the data
    rec_mats = []
    for i in range(data.shape[0]+1 - dim):
      ts = data[i:i + dim]

      rec_mat = np.zeros(shape=(dim, dim))
      for j in range(dim):
        
        for k in range(dim):
          rec_mat[j,k] = np.log(ts[j]/ts[k])

      rec_mats.append(rec_mat)
    
    rec_mats = np.array(rec_mats)
    if np.array(rec_mats).shape[0]!=1:
        rec_mats = np.expand_dims(np.squeeze(np.array(rec_mats)),axis=3)

    return rec_mats

def scaler(data,a,b):
    minimum = np.min(data)
    maximum = np.max(data)

    rec_mats = a+((data - minimum)*(b-a))/(maximum-minimum)
    
    return rec_mats

def recover_ts(imgs,start,args):
    recovered_container = []

    for i in range(imgs.shape[0]):
      recovered = np.zeros((args.sequence_length,args.sequence_length))
      image = imgs[i,:,:]

      initial_ts = np.repeat(start,image.shape[1])*(1+image[:,0])
      recovered[1,:]= initial_ts[1]*(1+image[:,1])

      for j in range(args.sequence_length):
        recovered[j,:]= initial_ts[j]*np.exp(image[:,j])

      recovered = np.concatenate([np.array(start).reshape((1,1)),np.expand_dims(np.mean(recovered,axis=0),axis=1)],axis=0)
      recovered_container.append(recovered)
    
    #drop the start value
    syn_seq = np.squeeze(np.array(recovered_container)[:,1:,:])
    return syn_seq
        
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