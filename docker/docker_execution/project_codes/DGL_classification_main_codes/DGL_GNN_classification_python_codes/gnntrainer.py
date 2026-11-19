import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping


from .rungnnmodel import RunGNNModel
from .graphsdatamodule import GraphsDataModule

class GNNTrainer():
    def __init__(self, args):

        self.args = args
        self.models_folder = args['models_folder']
        self.out_channels = args['out_channels']
        self.monitor = args['monitor']
        self.mode = args['mode']
        self.patience = args['patience']
        self.folder = args['folder'] + 'plots/'
        self.phases = ['train','val','test']
        self.datamodule = GraphsDataModule(args)

    def train_test(self):
        pl.seed_everything(42, workers=True)
        datamodule = self.datamodule

        run_model = RunGNNModel(self.args)

        csv_logger = pl.loggers.CSVLogger(save_dir=self.models_folder, name=None,
                                          version='version')

        checkpoint_callback = ModelCheckpoint(
            monitor=self.monitor,
            dirpath=self.models_folder,
            filename="saved_model",
            save_top_k=1,
            mode=self.mode,
        )

        early_stopping = EarlyStopping(
            monitor = self.monitor,
            patience = self.patience,
            mode = self.mode,
        )

        callbacks = [checkpoint_callback] #,early_stopping]

        trainer = pl.Trainer(
            default_root_dir=self.models_folder,
            enable_checkpointing=True,
            log_every_n_steps=1, 
            callbacks=callbacks,
            max_epochs=self.args['nepochs'],
            logger=csv_logger,
            devices=1,
            accelerator=self.args['accelerator'],
            enable_progress_bar=False,
            inference_mode=True,
            enable_model_summary=True,
            gradient_clip_val=0.5,
            gradient_clip_algorithm="norm",
        )
      

        trainer.fit(run_model, datamodule = datamodule)
        trainer.validate(run_model, datamodule = datamodule, verbose=True)
        trainer.test(run_model, datamodule=self.datamodule, verbose=True)
    
        return

  