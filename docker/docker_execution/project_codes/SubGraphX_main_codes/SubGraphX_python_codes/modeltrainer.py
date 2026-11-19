import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
import torch
import pandas as pd

from .graphsdatamodule import GraphsDataModule
from .runmodel import RunModel


class ModelTrainer:
    def __init__(self, args):
        self.args = args
        self.nepochs = args['nepochs']
        self.monitor = args['monitor']
        self.mode = args['mode']
        self.patience = args['patience']

    def results_visualization(self, model):
        args = self.args
       
        metrics_logs = args['folder_results'] + 'subgraphx_model_logs/version/metrics.csv'
        metrics = pd.read_csv(metrics_logs)
        train_metrics = metrics[['epoch','train_loss','train_acc']].dropna().reset_index(drop = True)
        val_metrics = metrics[['epoch','val_loss','val_acc']].dropna().reset_index(drop = True)
        metricsDict = {
            'epoch': train_metrics['epoch'].values,
            'train_loss': train_metrics['train_loss'].values,
            'train_acc': train_metrics['train_acc'].values,
            'val_loss': val_metrics['val_loss'].values,
            'val_acc': val_metrics['val_acc'].values,
        }
        epochs_range = range(args['nepochs'])
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        for ax , key in zip(axes,list(metricsDict.keys())[1:]):
            y = metricsDict[key]
            ax.plot(range(y.shape[0]),y, label=f'{key}')
            ax.set_xlabel("Epochs")
            ax.set_ylabel('')
            ax.legend()

        plt.suptitle("Training and Validation Metrics")
        plt.tight_layout()
        plt.savefig( args["folder_results"] + 'loss_accuracy.png')
        plt.close()

    def train_model(self):
        args = self.args
        datamodule = GraphsDataModule(self.args)
        datamodule.setup()
        
        model = RunModel(args)
        model.to(args['device'])
        
        csv_logger = CSVLogger(save_dir=args['folder_results'], 
                           name='subgraphx_model_logs',
                          version = 'version')

        checkpoint_callback = ModelCheckpoint(
            dirpath=args['folder_results'],
            filename="best_model",
            save_top_k=1,
            monitor=self.monitor, 
            mode=self.mode ,  
        )
        early_stopping = EarlyStopping(
            monitor = self.monitor, 
            # patience = self.patience,  
            mode = self.mode ,
        )

        callbacks = [checkpoint_callback] #,early_stopping]
        
        trainer = pl.Trainer(
            max_epochs=self.nepochs,
            enable_progress_bar=False, 
            enable_checkpointing=True,  
            log_every_n_steps=1,
            inference_mode=True,
            accelerator = args['accelerator'],
            devices=1,
            logger=csv_logger,
            enable_model_summary=True,
            callbacks=callbacks,
            gradient_clip_val=0.5,
            gradient_clip_algorithm="norm",
        )

        
        trainer.fit(model, datamodule = datamodule) 
        trainer.validate(model, datamodule = datamodule, verbose= False)
        
        print('Done training and validating --------------------------')
        
        # Visualize the results after training
        self.results_visualization(model)

