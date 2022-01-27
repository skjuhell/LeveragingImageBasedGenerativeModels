from models.wgan_gp_irp import train_WGAN
from metrics.visualization import create_umap_plot, create_tsne_plot
from metrics.predictive_score import predictive_score
from metrics.discriminative_score import discriminative_score

import numpy as np
import argparse
import utils
import pandas as pd


def main(args):
    
    parser = argparse.ArgumentParser()

    #1 Load data, Select model and generate data
    if args.data_name == 'ar1':
        ori_data = utils.ar1_process(nsample=1000,sigma=0.9,c=1000)
        
    elif args.data_name == 'stock':
        ori_data = utils.read_stock_price_data('data/stock_data.csv')
        print(ori_data.shape)

    elif args.data_name =='brownian_motion':
        ori_data = 100+np.cumsum(np.random.normal(loc=0.05, scale=1,size=1000))
    
    #calculate recurrence plots
    ori_data_imgs = utils.irps(ori_data,args.sequence_length)
    
    #scale the data fot the WGAN training between -1 and 1
    ori_data_imgs = utils.scaler(data=ori_data_imgs,a=-1,b=1)
    
    if args.mode=='generate':
        ori_data_imgs,syn_data_imgs = train_WGAN(ori_data_imgs=ori_data_imgs,args=args)
    elif args.mode=='benchmark':
        if args.data_name!='stock':
            raise ValueError('Benchmark only available for stock data')
        else:
            syn_data = np.load('data/synthesized_data_yoon.npy')
            syn_data_imgs = np.squeeze(np.array([utils.irps(np.expand_dims(x,axis=1),args.sequence_length) for x in syn_data]))
            ori_data_imgs = np.squeeze(ori_data_imgs)
    
    
    #scale between 0 and 1 (since IRP only supports positive values) and recover the time series
    ori_data_imgs = utils.scaler(data=ori_data_imgs,a=0,b=1)
    syn_data_imgs = utils.scaler(data=syn_data_imgs,a=0,b=1)

    ori_data_seqs = utils.recover_ts(imgs=ori_data_imgs,start=100,args=args)
    syn_data_seqs = utils.recover_ts(imgs=syn_data_imgs,start=100,args=args)
    
    #scale back again for the tests with the RNNs
    ori_data_seqs = utils.scaler(data=ori_data_seqs,a=-1,b=1)
    syn_data_seqs = utils.scaler(data=syn_data_seqs,a=-1,b=1)
    
    if args.visualization=='UMAP':
        create_umap_plot(ori_data_seqs,syn_data_seqs,suffix=args.data_name,title='UMAP for '+args.data_name,elements=args.elements_umap)
    elif args.visualization=='tSNE':
        create_tsne_plot(ori_data_seqs,syn_data_seqs,suffix=args.data_name,title='t-SNE for '+args.data_name,elements=args.elements_umap)
        
    #3 Run RNN backtests
    print(str(args.data_name)+' run predictive backtest ...')
    pred_score_mean,pred_score_std = predictive_score(ori_seq=ori_data_seqs,syn_seq=syn_data_seqs,args=args)
    pred_score_str = str(args.data_name)+ ' Predictive Score '+str(np.round(pred_score_mean,4))+' +/- '+str(np.round(pred_score_std,4))
    print(pred_score_str)

    print(str(args.data_name)+' run discriminative backtest ...')
    disc_score_mean,disc_score_std = discriminative_score(ori_seq=ori_data_seqs,syn_seq=syn_data_seqs,args=args)
    disc_score_str = str(args.data_name)+' Discriminative Score '+str(np.round(disc_score_mean,4))+' +/- '+str(np.round(disc_score_std,4))
    print(disc_score_str)

    return ori_data_seqs,syn_data_seqs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_name',
        choices=['ar1','stock','brownian_motion'],
        default='stock',
        type=str
    )

    parser.add_argument(
        '--mode',
        choices=['generate','benchmark'],
        default='generate',
        type=str
    )

    parser.add_argument(
        '--sequence_length',
        choices=[28],
        default=28,
        type=int
    )

    parser.add_argument(
        '--epochs',
        choices=[100,200,500,1000],
        default=200,
        type=int
    )
    parser.add_argument(
        '--discriminator_extra_steps',
        choices=[1,2,3,4,5],
        default=3,
        type=int
    )

    parser.add_argument(
        '--visualization',
        choices=['UMAP', 'tSNE'],
        default='UMAP',
        type=str
    )
    parser.add_argument(
        '--elements_umap',
        choices=[100,200,300,500],
        default=500,
        type=int
    )
    parser.add_argument(
        '--iterations',
        choices=[100,500,1000,1500],
        default=100,
        type=int
    )
    parser.add_argument(
        '--batch_size',
        choices=[2,4,8,16],
        default=8,
        type=int
    )
    parser.add_argument(
        '--noise_dim',
        choices=[128,256,512],
        default=512,
        type=int
    )
    parser.add_argument(
        '--batch_size_predictive',
        choices=[8,16,32,64,128],
        default=8,
        type=int
    )
    parser.add_argument(
        '--metric_iterations',
        choices=[5,10,15,20],
        default=10,
        type=int
    )
    parser.add_argument(
        '--hidden_dim_predictive',
        choices=[14,28,56],
        default=28,
        type=int
    )
    parser.add_argument(
        '--iterations_predictive',
        choices=[200,400,800],
        default=200,
        type=int
    )
    parser.add_argument(
        '--hidden_dim_discriminative',
        choices=[4,8,16,32],
        default=28,
        type=int
    )
    parser.add_argument(
        '--iterations_discriminative',
        choices=[200,400,800],
        default=200,
        type=int
    )
    parser.add_argument(
        '--batch_size_discriminative',
        choices=[2,4,8,16,32],
        default=8,
        type=int
    )
    parser.add_argument(
        '--cbk_patience',
        choices=[5,10,15],
        default=5,
        type=int
    )

    args = parser.parse_args()

    ori,syn = main(args)
