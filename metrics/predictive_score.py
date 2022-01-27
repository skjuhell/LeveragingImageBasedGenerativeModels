import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

##suppress scientific notation
np.set_printoptions(formatter={'float': '{: 0.8f}'.format})

def predictive_score(ori_seq,syn_seq,args):
    
    ori_seq_x = np.expand_dims(ori_seq[:,:-1].astype('float32'),axis=1)
    ori_seq_y = np.expand_dims(ori_seq[:,-1].astype('float32'),axis=1)

    syn_seq_x = np.expand_dims(syn_seq[:,:-1].astype('float32'),axis=1)
    syn_seq_y = np.expand_dims(syn_seq[:,-1].astype('float32'),axis=1)

    
    pred_score_syn = []

    pred_cbk = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.cbk_patience)

    for i in range(args.metric_iterations):
        print('Predictive Score Calculation '+str(i+1)+'/'+str(args.metric_iterations)+' ... ')
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.GRU(units=args.hidden_dim_predictive,activation='relu',return_sequences=True))
        model.add(tf.keras.layers.GRU(units=args.hidden_dim_predictive))
        model.add(tf.keras.layers.Dense(units=1,activation='relu'))
        model.compile(optimizer="Adam", loss="mae", metrics=["mse"])
        model.fit(x=syn_seq_x,y=syn_seq_y,validation_split=0.2,epochs=args.iterations_predictive,batch_size=args.batch_size_predictive,verbose=0,callbacks=[pred_cbk])
        result = model.evaluate(x=ori_seq_x,y=ori_seq_y)[0]
        pred_score_syn.append(result)
    
    syn_pred_score_mean = np.mean(pred_score_syn)
    syn_pred_score_std = np.std(pred_score_syn)
    
    return syn_pred_score_mean, syn_pred_score_std
