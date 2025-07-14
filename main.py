
from metrics.visualization import create_umap_plot, create_tsne_plot
from metrics.predictive_score import predictive_score
from metrics.discriminative_score import discriminative_score
from metrics.augmentation_score import augmentation_score

import numpy as np
import argparse
#from pandas.core.window.rolling import args_compat
import utils
import pandas as pd

def main(args):
    
    if args.model=='WGAN_GP':
      from models.wgan_gp_irp import train_WGAN
    elif (args.model=='TimeGAN')&(args.mode=='generate'):
      from models.timegan import timegan
    elif (args.model=='Diffusion')&(args.mode=='generate'):
      from models.diffusion_model import diffusion
    
    parser = argparse.ArgumentParser()

    np.random.seed(42)

    #1 Load data, Select model and generate data
    print('This column is being loaded: {}'.format(args.column))
    ori_data_all = pd.read_csv('data/raw_data/'+args.data_name+'.csv',usecols=[args.column]).dropna().head(2000)
    ori_data = ori_data_all[::-1]
    ori_data = np.expand_dims(ori_data, axis=1)

    #calculate image representation
    if args.representation=='irp':
      ori_data_prescale,mini_prescale,maxi_prescale = utils.scaler(data=ori_data,a=0,b=1) 
      ori_data_imgs = utils.irp(ori_data_prescale,args.sequence_length,0.001)
      ori_data_imgs,mini,maxi = utils.scaler(data=ori_data_imgs,a=-1,b=1)
    
    elif args.representation=='xirp':
      ori_data_prescale,mini_prescale,maxi_prescale = utils.scaler(data=ori_data,a=0,b=1) 
      ori_data_imgs = utils.xirp(ori_data_prescale,args.sequence_length,0.001)
      res = utils.scale_image_rep(data=ori_data_imgs,a=0,b=1,method='diagonal_separately')
      ori_data_imgs = res['rec_mats']
      #res = utils.scale_image_rep(data=ori_data_imgs,a=-1,b=1,method='diagonal_separately')

    elif args.representation=='naive':
      ori_data_prescale,mini_prescale,maxi_prescale = utils.scaler(data=ori_data,a=0,b=1) 
      ori_data_imgs = utils.naive(ori_data_prescale,args.sequence_length)
      res = utils.scale_image_rep(data=ori_data_imgs,a=0,b=1,method='default')
      ori_data_imgs = res['rec_mats']
      res = utils.scale_image_rep(data=ori_data_imgs,a=-1,b=1,method='default')

    elif args.representation=='gasf':
      # here we need to prescale the data between min 0 and max pi/2 for the gasf
      ori_data_prescale,mini_prescale,maxi_prescale = utils.scaler(data=ori_data,a=0,b=1) 
      ori_data_imgs = utils.gasf(ori_data_prescale,args.sequence_length)
      res = utils.scale_image_rep(data=ori_data_imgs,a=0,b=1,method='default')
      ori_data_imgs = res['rec_mats']
      res = utils.scale_image_rep(data=ori_data_imgs,a=-1,b=1,method='diagonal_separately')
      #ori_data_imgs,mini,maxi = utils.scaler(data=ori_data_imgs,a=-1,b=1)
    
    elif args.representation=='unthresholded':
      ori_data_prescale,mini_prescale,maxi_prescale = utils.scaler(data=ori_data,a=0,b=1) 
      ori_data_imgs = utils.unthresholded(ori_data,args.sequence_length)
      ori_data_imgs,mini,maxi = utils.scaler(data=ori_data_imgs,a=-1,b=1)

    elif args.representation=='ts':
      # sequentialize the data for use in TimeGAN
      ori_data_imgs = utils.create_ts_chunks(ori_data,args)

    #scale the data fot the WGAN training between -1 and 1, TimeGAN has built in scaler
    if args.model not in ['TimeGAN','Diffusion']:
      ori_data_imgs,mini,maxi = utils.scaler(data=ori_data_imgs,a=-1,b=1)
    
    if args.mode=='generate':
        print('Generate Data ... ')
        if args.model=='WGAN_GP':
          ori_data_imgs,syn_data_imgs = train_WGAN(ori_data_imgs=ori_data_imgs,args=args)        

          if args.representation=='irp':
            ori_data_imgs = utils.descaler(data_scaled=ori_data_imgs,mini=mini,maxi=maxi,a=-1,b=1)
            syn_data_imgs = utils.descaler(data_scaled=syn_data_imgs,mini=mini,maxi=maxi,a=-1,b=1)

            ori_data_seqs = utils.recover_irp(start=args.recovery_start_value,imgs=ori_data_imgs,method=args.recovery_method)
            syn_data_seqs = utils.recover_irp(start=args.recovery_start_value,imgs=syn_data_imgs,method=args.recovery_method)
            
            ori_data_seqs = utils.descaler(data_scaled=ori_data_seqs,mini=mini_prescale,maxi=maxi_prescale,a=0,b=1)
            syn_data_seqs = utils.descaler(data_scaled=syn_data_seqs,mini=mini_prescale,maxi=maxi_prescale,a=0,b=1)

          if args.representation=='xirp':

            ori_descaled = utils.descale_img_rep(data_scaled=ori_data_imgs,method='diagonal_separately',res=res)
            syn_descaled = utils.descale_img_rep(data_scaled=syn_data_imgs,method='diagonal_separately',res=res)

            ori_descaled = utils.descaler(data_scaled=ori_descaled,mini=mini,maxi=maxi,a=-1,b=1)
            syn_descaled = utils.descaler(data_scaled=syn_descaled,mini=mini,maxi=maxi,a=-1,b=1)

            ori_data_seqs = utils.recover_xirp(imgs=ori_descaled,method=args.recovery_method)
            syn_data_seqs = utils.recover_xirp(imgs=syn_descaled,method=args.recovery_method)

            ori_data_seqs = utils.descaler(data_scaled=ori_data_seqs,mini=mini_prescale,maxi=maxi_prescale,a=0,b=1)
            syn_data_seqs = utils.descaler(data_scaled=syn_data_seqs,mini=mini_prescale,maxi=maxi_prescale,a=0,b=1)


          if args.representation=='gasf':
            ori_descaled = utils.descale_img_rep(data_scaled=ori_data_imgs,method='default',res=res)
            syn_descaled = utils.descale_img_rep(data_scaled=syn_data_imgs,method='default',res=res)

            ori_descaled = utils.descaler(data_scaled=ori_descaled,mini=mini,maxi=maxi,a=-1,b=1)
            syn_descaled = utils.descaler(data_scaled=syn_descaled,mini=mini,maxi=maxi,a=-1,b=1)

            ori_data_seqs = utils.recover_gasf(imgs=ori_descaled,method=args.recovery_method)
            syn_data_seqs = utils.recover_gasf(imgs=syn_descaled,method=args.recovery_method)

            ori_data_seqs = utils.descaler(data_scaled=ori_data_seqs,mini=mini_prescale,maxi=maxi_prescale,a=0,b=1)
            syn_data_seqs = utils.descaler(data_scaled=syn_data_seqs,mini=mini_prescale,maxi=maxi_prescale,a=0,b=1)
          
          if args.representation=='unthresholded':
            ori_data_seqs = utils.recover_unthresholded(start=args.recovery_start_value,imgs=ori_data_imgs,method=args.recovery_method)
            syn_data_seqs = utils.recover_unthresholded(start=args.recovery_start_value,imgs=syn_data_imgs,method=args.recovery_method)
            
            ori_data_seqs = utils.descaler(data_scaled=ori_data_seqs,mini=mini_prescale,maxi=maxi_prescale,a=0,b=1)
            syn_data_seqs = utils.descaler(data_scaled=syn_data_seqs,mini=mini_prescale,maxi=maxi_prescale,a=0,b=1)

          if args.representation=='naive':
            ori_descaled = utils.descale_img_rep(data_scaled=ori_data_imgs,method='default',res=res)
            syn_descaled = utils.descale_img_rep(data_scaled=syn_data_imgs,method='default',res=res)

            ori_descaled = utils.descaler(data_scaled=ori_descaled,mini=mini,maxi=maxi,a=-1,b=1)
            syn_descaled = utils.descaler(data_scaled=syn_descaled,mini=mini,maxi=maxi,a=-1,b=1)

            ori_data_seqs = utils.recover_naive(imgs=ori_descaled,method=args.recovery_method)
            syn_data_seqs = utils.recover_naive(imgs=syn_descaled,method=args.recovery_method)

            ori_data_seqs = utils.descaler(data_scaled=ori_data_seqs,mini=mini_prescale,maxi=maxi_prescale,a=0,b=1)
            syn_data_seqs = utils.descaler(data_scaled=syn_data_seqs,mini=mini_prescale,maxi=maxi_prescale,a=0,b=1)

        if args.model=='TimeGAN':
          ori_data_seqs = ori_data_imgs
          ori_data_seqs,syn_data_seqs = timegan(ori_data_seqs,args)

        if args.model=='Diffusion':
          ori_data_seqs = ori_data_imgs
          syn_data_seqs, ori_data_seqs = diffusion(ori_data_seqs,args)  

        print('Ori Data Sequence: Min: {:.4f} Max:{:.4f}'.format(np.min(ori_data_seqs),np.max(ori_data_seqs)))
        print('Syn Data Sequence: Min: {:.4f} Max:{:.4f}'.format(np.min(syn_data_seqs),np.max(syn_data_seqs)))

        utils.save_seqs('syn',syn_data_seqs,args)
        utils.save_seqs('ori',ori_data_seqs,args)
    
    elif args.mode=='benchmark':

      ori_data_seqs = utils.load_seqs('ori',args)
      syn_data_seqs = utils.load_seqs('syn',args)

      #scale back again for the tests with the RNNs
      ori_data_seqs,min_ori,max_ori = utils.scaler(data=ori_data_seqs,a=-1,b=1)
      syn_data_seqs,min_syn,max_syn = utils.scaler(data=syn_data_seqs,a=-1,b=1)
      
      if args.visualization=='UMAP':
          create_umap_plot(ori_data_seqs,syn_data_seqs,suffix=args.data_name+'_'+args.model+'_'+args.column+'_'+args.representation+'_'+args.recovery_method,title='UMAP for '+args.data_name+'_'+args.column,elements=args.elements_umap)
      elif args.visualization=='tSNE':
          create_tsne_plot(ori_data_seqs,syn_data_seqs,suffix=args.data_name+'_'+args.model+'_'+args.column+'_'+args.representation+'_'+args.recovery_method,title='t-SNE for '+args.data_name+'_'+args.column,elements=args.elements_umap)
     
      #3 Run RNN backtests

      augmentation_score(ori_seq=ori_data_seqs,syn_seq=syn_data_seqs,args=args)

      print(str(args.data_name)+' run predictive backtest ...')
      pred_score_mean,pred_score_std = predictive_score(ori_seq=ori_data_seqs,syn_seq=syn_data_seqs,args=args)
      pred_score_str = str(args.data_name)+ ' Predictive Score '+str(np.round(pred_score_mean,4))+' +/- '+str(np.round(pred_score_std,4))
      
      print(pred_score_str)

      utils.write_results(
        model = args.model,
        data_name=args.data_name,
        column=args.column,
        representation = args.representation,
        recovery_method = args.recovery_method,
        metric='pred_score',
        score=pred_score_mean,
        std=pred_score_std,
        )

      print(str(args.data_name)+' run discriminative backtest ...')
      disc_score_mean,disc_score_std = discriminative_score(ori_seq=ori_data_seqs,syn_seq=syn_data_seqs,args=args)
      disc_score_str = str(args.data_name)+' Discriminative Score '+str(np.round(disc_score_mean,4))+' +/- '+str(np.round(disc_score_std,4))
      print(disc_score_str)

      utils.write_results(
        model = args.model,
        data_name=args.data_name,
        column=args.column,
        representation = args.representation,
        recovery_method = args.recovery_method,
        metric='disc_score',
        score=disc_score_mean,
        std=disc_score_std,
        )
      
      
    return ori_data_seqs,syn_data_seqs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_name',
        choices=['brownian_motion','sine','noisy_sine','energy_data','merton_process','power_law','stock_data','air_quality','bike_share'],
        default='stock_data',
        type=str
    )

    parser.add_argument(
        '--column',
        type=str
    )

    parser.add_argument(
        '--model',
        choices=['WGAN_GP','TimeGAN','Diffusion'],
        type=str
    )

    parser.add_argument(
      '--representation',
      choices=['irp','xirp','gasf','unthresholded','ts','naive'],
      default = 'xirp',
      type=str
    )

    parser.add_argument(
      '--recovery_method',
      choices=['columns_mean','columns_random',''],
      default = 'columns_mean',
      type=str
    )

    parser.add_argument(
      '--recovery_start_value',
      choices=[1,10,100],
      default = 100,
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
        choices=[1,10,50,100,200,500,1000],
        default=200,
        type=int
    )
    parser.add_argument(
        '--discriminator_extra_steps',
        choices=[1,2,3,4,5],
        default=5,
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
        choices=[100,150,500,1000,1500],
        default=150,
        type=int
    )
    parser.add_argument(
        '--batch_size',
        choices=[2,4,8,16,32,64,128,256],
        default=32,
        type=int
    )
    parser.add_argument(
        '--learning_rate',
        choices=[0.0001,0.0005],
        default=0.0005,
        type=float
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
        choices=[1,5,10,15,20],
        default=20,
        type=int
    )
    parser.add_argument(
        '--hidden_dim_predictive',
        choices=[14,28,56],
        default=28,
        type=int
    )
    parser.add_argument(
        '--hidden_dim_augmentation',
        choices=[8,14,28,56],
        default=8,
        type=int
    )
    parser.add_argument(
        '--batch_size_augmentation',
        choices=[2,4,8,16,32],
        default=16,
        type=int
    )
    parser.add_argument(
        '--iterations_predictive',
        choices=[200,400,800],
        default=200,
        type=int
    )
    parser.add_argument(
        '--iterations_augmentation',
        choices=[200,400,800],
        default=400,
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
        default=10,
        type=int
    )

    args = parser.parse_args()

    ori,syn = main(args)
