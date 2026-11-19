import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
import torch

import pandas as pd
import matplotlib.pyplot as plt

from .createdatamodule import CreateDataModule
from .argvamodel import ARGVAModel

class TrainingTestingClass():

    def __init__(self, args,year): 
        
        super(TrainingTestingClass, self).__init__()
        self.args = args
        self.year = year
        self.n_epochs = args['n_epochs']
        self.folder = args['folder']
        self.models_folder = args['models_folder']

    def training_testing_model(self):
        
        torch.manual_seed(1)  
        args = self.args
        year = self.year
        datamodule = CreateDataModule(args,year)
        datamodule.setup()
        
        model = ARGVAModel(args,year)
    
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',  
            patience=20,          
            mode='min',          
            verbose=False)

        csv_logger = pl.loggers.CSVLogger(save_dir=self.folder, 
                                            name=f'losses/loss_{year}', 
                                            version='version')
        
        checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",  
                dirpath=self.models_folder,
                filename=f"model_{year}_path",
                save_top_k=1,  
                mode="min"  ,
                )
        
        trainer = pl.Trainer(
            max_epochs=self.n_epochs,
            enable_checkpointing=True,  
            callbacks=[checkpoint_callback,early_stopping_callback],  
            logger=csv_logger,
            log_every_n_steps=1,
            accelerator = args['accelerator'],
            enable_progress_bar=False,
            inference_mode=False,
            enable_model_summary=True,
        )

        trainer.fit(model, datamodule = datamodule) 
        trainer.validate(model, datamodule = datamodule, verbose= False)
        
        trainer.test(model, datamodule = datamodule, verbose =False)

        log_file = f"{args['folder']}" +f"/losses/loss_{year}/version/metrics.csv"
        metrics = pd.read_csv(log_file) 
        
        train_loss = metrics['train_loss_epoch'].dropna().tolist()
        val_loss = metrics['val_loss'].dropna().tolist()
        
        mse = metrics['mse'].dropna().tolist()
        mae = metrics['mae'].dropna().tolist()

        auc = metrics['auc'].dropna().tolist()
        ap = metrics['ap'].dropna().tolist()
        
        accuracy = metrics['accuracy'].dropna().tolist()
        precision = metrics['precision'].dropna().tolist()
        recall = metrics['recall'].dropna().tolist()
        f1= metrics['f1'].dropna().tolist()
    
        """ Plotting measures """
        
        datas1 = [train_loss, val_loss, mse, mae] 
        names1 = ['train_loss', 'val_loss', 'mse', 'mae' ] 

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        for ax, dat, data_type in zip(axes, datas1, names1):
            ax.plot(dat)
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(data_type, fontsize=12)

        title_name = f'Reconstruction Measures: Year {year}'
        plt.suptitle(title_name, fontsize=18)
        plt.tight_layout()
        path_name = f'losses/plots/reconstruction_accuracy_measures_year_{year}.png'
        plt.savefig(args['folder'] + path_name)
        plt.tight_layout()
        plt.close()
    
        datas2 = [auc, ap, accuracy, precision, recall, f1]
        names2 = ['auc', 'ap', 'accuracy', 'precision', 'recall', 'f1']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for ax, dat, name in zip(axes, datas2, names2):
            ax.plot(dat)
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(name, fontsize=12)
            ax.set_title(f'{name}, Year {year}', fontsize=15)
        plt.tight_layout()
        path_name = f'losses/plots/validation_measures_{year}.png'
        plt.savefig(args['folder'] + path_name)
        plt.close()

        return 