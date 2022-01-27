import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

##suppress scientific notation
np.set_printoptions(formatter={'float': '{: 0.8f}'.format})
## Network parameters taken from yoon et al

def discriminative_score(ori_seq,syn_seq,args):

    ori_seq = np.expand_dims(ori_seq.astype('float32'),axis=1)
    syn_seq = np.expand_dims(syn_seq.astype('float32'),axis=1)

    disc_cbk = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.cbk_patience)

    x = np.concatenate([ori_seq,syn_seq],axis=0)

    #create labels
    ori_label = np.zeros(ori_seq.shape[0])
    syn_label = np.ones(syn_seq.shape[0])
    y = np.concatenate([ori_label,syn_label],axis=0)
    disc_score = []

    for i in range(args.metric_iterations):
        print('Discriminative Score Calculation '+str(i+1)+'/'+str(args.metric_iterations)+' ... ')
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.GRU(units=args.hidden_dim_discriminative,activation='relu',return_sequences=True,dropout=.4))
        model.add(tf.keras.layers.GRU(units=args.hidden_dim_discriminative,activation='sigmoid',return_sequences=False,dropout=.4))
        model.add(tf.keras.layers.Dense(units=1,activation='softmax'))
        model.compile(optimizer="Adam", loss="binary_crossentropy",metrics=['accuracy'])

        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1, shuffle=True)

        model.fit(x=x_train,y=y_train,validation_split=0.2,epochs=args.iterations_discriminative,batch_size=args.batch_size_discriminative,verbose=0,callbacks=[disc_cbk])
        result = model.evaluate(x=x_test,y=y_test)[0]
        #print(np.mean(y_test), np.mean(model.predict(x_test)))
        disc_score.append(result)

    disc_score_mean = np.mean(disc_score)
    disc_score_std = np.std(disc_score)
    
    return disc_score_mean,disc_score_std
