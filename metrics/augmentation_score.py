import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras import backend as K

def reset_weights(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
            reset_weights(layer) #apply function recursively
            continue

        #where are the initializers?
        if hasattr(layer, 'cell'):
            init_container = layer.cell
        else:
            init_container = layer

        for key, initializer in init_container.__dict__.items():
            if "initializer" not in key: #is this item an initializer?
                  continue #if no, skip it

            # find the corresponding variable, like the kernel or the bias
            if key == 'recurrent_initializer': #special case check
                var = getattr(init_container, 'recurrent_kernel')
            else:
                var = getattr(init_container, key.replace("_initializer", ""))

            var.assign(initializer(var.shape, var.dtype))
            #use the initializer

def augmentation_score(ori_seq,syn_seq,args):

    syn_fraction = np.arange(0.0,1.1,0.1)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GRU(units=args.hidden_dim_augmentation,activation='relu',return_sequences=True,dropout=.2))
    #model.add(tf.keras.layers.GRU(units=args.hidden_dim_augmentation,activation='relu',return_sequences=True,dropout=.2))
    model.add(tf.keras.layers.GRU(units=int(args.hidden_dim_augmentation/2),activation='relu',dropout=.2))
    model.add(tf.keras.layers.Dense(units=1,activation='relu'))
    model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    aug_cbk = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.cbk_patience)

    
    res = []
    for frac in syn_fraction:
        pred_score_rmse = []
        pred_score_mae = []
        for i in range(args.metric_iterations):
            #shuffle the original sequences
            np.random.shuffle(ori_seq)

            test_fraction = 0.2
            test_idx = int(ori_seq.shape[0]*(1-test_fraction))
            ori_train_data = ori_seq[:test_idx]
            syn_train_data = syn_seq[np.random.randint(0,ori_seq.shape[0],int(frac*ori_train_data.shape[0]))]
            test_data = ori_seq[test_idx:]
            no_test_data = int(test_data.shape[0]*0.5)
            joint_train_data = np.concatenate([ori_train_data,syn_train_data],axis=0)

            joint_seq_x = np.expand_dims(joint_train_data[:,:-1].astype('float32'),axis=1)
            joint_seq_y = np.expand_dims(joint_train_data[:,-1].astype('float32'),axis=1)
            
            #we have to split the original test data again to make sure we don't test on fake data in the callback that triggers the stopping
            ori_test_seq_x = np.expand_dims(test_data[:no_test_data,:-1].astype('float32'),axis=1)
            ori_test_seq_y = np.expand_dims(test_data[:no_test_data,-1].astype('float32'),axis=1)

            ori_val_seq_x = np.expand_dims(test_data[no_test_data:,:-1].astype('float32'),axis=1)
            ori_val_seq_y = np.expand_dims(test_data[no_test_data:,-1].astype('float32'),axis=1)
            
            if i==0:
                print(str(np.round(frac/2,2))+' -------------------- ')
                print('Train Data Set '+str(joint_seq_x.shape[0]))
                print('Test Data Set '+str(ori_test_seq_x.shape[0]))
                print('Validation Data Set '+str(ori_val_seq_x.shape[0]))

            model.fit(x=joint_seq_x,y=joint_seq_y,validation_data=(ori_val_seq_x,ori_val_seq_y),epochs=args.iterations_augmentation,batch_size=args.batch_size_augmentation,verbose=0,callbacks=[aug_cbk])
            result_rmse = np.sqrt(model.evaluate(x=ori_test_seq_x,y=ori_test_seq_y)[0])
            result_mae = model.evaluate(x=ori_test_seq_x,y=ori_test_seq_y)[1]
            pred_score_rmse.append(result_rmse)
            pred_score_mae.append(result_mae)

            reset_weights(model)

            print('Finished Iteration '+str(i+1)+' of '+str(args.metric_iterations))

        res.append([np.round(frac/2,2),np.round(np.mean(pred_score_rmse),4), np.round(np.mean(pred_score_mae),4),joint_seq_x.shape[0],ori_test_seq_x.shape[0],ori_val_seq_x.shape[0]])
        res_df = pd.DataFrame(res,columns=['frac','rmse','mae','train','test','validation'])

    model = args.model
    data_name=args.data_name
    column=args.column
    representation = args.representation
    recovery_method = args.recovery_method

    path_name = f'results/augmentation/{model}_{data_name}_{column}_{representation}_{recovery_method}_augmentation_benefit.csv'
    res_df.to_csv(path_name)

    return res_df
        









'''
        print('Predictive Score Calculation '+str(i+1)+'/'+str(args.metric_iterations)+' ... ')
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.GRU(units=args.hidden_dim_predictive,activation='relu',return_sequences=True))
        model.add(tf.keras.layers.GRU(units=args.hidden_dim_predictive))
        model.add(tf.keras.layers.Dense(units=1,activation='relu'))
        model.compile(optimizer="Adam", loss="mae", metrics=["mse"])
        model.fit(x=syn_seq_x,y=syn_seq_y,validation_split=0.2,epochs=args.iterations,batch_size=args.batch_size_predictive,verbose=0,callbacks=[pred_cbk])
        result = model.evaluate(x=ori_seq_x,y=ori_seq_y)[0]
        pred_score_syn.append(result)
    
    syn_pred_score_mean = np.mean(pred_score_syn)
    syn_pred_score_std = np.std(pred_score_syn)
    
    return syn_pred_score_mean, syn_pred_score_std
'''