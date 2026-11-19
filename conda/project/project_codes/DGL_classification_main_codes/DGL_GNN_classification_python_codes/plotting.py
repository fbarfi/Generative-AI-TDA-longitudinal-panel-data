import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .graphsdatamodule import GraphsDataModule

class PlottingClass():

    def __init__(self, args):
        self.args = args
        self.out_channels = args['out_channels']
        self.num_hidden_layers = args['num_hidden_layers']
        self.hidden_channels = args['hidden_channels']
        self.h_dim = args['h_dim']
        self.folder = args['folder'] + 'plots/'
        self.folder1 = args['folder'] + 'csv_files/'
        self.models_folder = args['folder'] + 'GNN_models/'
        
    def get_data(self):                                      
        data_module = GraphsDataModule(self.args)
        data_module.setup()
        train_data = data_module.train_data
        val_data = data_module.val_data
        test_data = data_module.test_data
        return train_data, val_data, test_data

    def loading_model(self):
        args = self.args
        checkpoint_path =args['models_folder']+ '/model_path.ckpt'
        model = RunGNNModel.load_from_checkpoint(checkpoint_path=checkpoint_path, 
                                             args=args)
        return model

    def get_metrics(self):
        log_file = f"{self.args['folder']}GNN_models/version/metrics.csv"
        metrics = pd.read_csv(log_file)
        metricsDict = {col: metrics[col].dropna().tolist() 
                       for col in metrics.columns}
        return metricsDict

    def get_true_and_pred(self, dataloader):
        model = self.loading_model()
        model.eval()
        true_labels = []
        predictions = []

        with torch.no_grad():  
            for batch in dataloader:
                true_labels.append(batch.label.detach().numpy())  
                
                outputs, _ = model(batch.x, batch.edge_index, 
                                   batch.edge_attr, 
                                   batch=batch.batch)
                probas = F.softmax(outputs, dim=1)  
                preds = torch.argmax(probas, dim=1)
                
                predictions.append(preds.detach().numpy())  
            
            true_labels = np.concatenate(true_labels)
            predictions = np.concatenate(predictions)
    
        return true_labels, predictions
        
        
    def plot_losses(self):
        h_dim = self.h_dim
        folder = self.folder

        log_file = f"{self.args['folder']}GNN_models/version/metrics.csv"
        metrics = pd.read_csv(log_file)

        metricsDict = {
        'train_loss':metrics[['epoch',
                    'train_loss']].dropna().reset_index(drop = True)['train_loss'].values,
         'val_loss':  metrics[['epoch',
                        'val_loss']].dropna().reset_index(drop=True)['val_loss'].values,
         'train_acc':metrics[['epoch',
                        'train_acc']].dropna().reset_index(drop = True)['train_acc'].values,
         'val_acc':  metrics[['epoch',
                        'val_acc']].dropna().reset_index(drop=True)['val_acc'].values,
        }

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()
        labels = metricsDict.keys()
        for ax , label in zip(axes,labels):
            ax.plot(metricsDict[label], label=label)
            ax.legend()
            ax.set_xlabel('Epoch')
           
        plt.suptitle('Losses and Accuracies')
        plt.tight_layout()

        plt.savefig(folder + f'train_val_losses.png')
        plt.close()

        return

    def plot_histograms(self):
        m_dict = self.get_metrics()
        colors = ['red', 'blue', 'green', 'orange'] 
    
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for ax, cl in zip(axes.flatten(), range(self.out_channels)):
            train_data = m_dict.get(f'train_pred_class_{cl}', [])
            ax.hist(train_data, bins=50, color='red')
            ax.set_title(f'Train Class {cl}', fontsize=12)
            ax.set_xlabel('Probability')
        
        plt.suptitle('Train Predictions Histogram', fontsize=15)
        plt.tight_layout()
        plt.savefig(self.folder + 'train_histograms_pred.png')
        plt.close()
    

    def comparing_actual_predicted_classes(self):
        h_dim = self.h_dim
        folder = self.folder
        out_channels = self.out_channels
        
        metrics = self.get_metrics()
        train_data, val_data, test_data = self.get_data()
    
        train_val_data = train_data + val_data
        
        train_true = np.array([label.cpu().numpy() 
                               for _,label in train_val_data]).reshape(-1)
        test_true = np.array([label.cpu().numpy() 
                              for _,label in test_data]).reshape(-1)
        
        class_labels = sorted(set(
            int(col.split('_pred_class_')[1].split('_')[0]) 
            for col in metrics.keys() if 'pred_class_' in col
        ))
        
        pred_data = {phase: {} for phase in ['train', 'test']}
        
        for phase in pred_data.keys():
            for class_label in class_labels:
                key = f'{phase}_pred_class_{class_label}'
                if key in metrics.keys():
                    pred_data[phase][f'class_{class_label}'] = metrics[key]
    
        true_data = {'train': train_true, 'test': test_true}
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()
        
        kinds = [
            ('train', 'true'), 
            ('train', 'pred'), 
            ('test', 'true'), 
            ('test', 'pred')
        ]
        
        for ax, (phase, value_type) in zip(axes, kinds):
            if value_type == 'true':
                data = true_data[phase]
                
                for cl in range(out_channels):
                    class_data = data[data == cl]
                    ax.hist(class_data, bins=50, label=f'class_{cl}', alpha=0.6)
            else:  # value_type == 'pred'
        
                for cl in range(out_channels):
                    pred_probs = pred_data[phase].get(f'class_{cl}', [])
                    ax.hist(pred_probs, bins=50, label=f'class_{cl}', alpha=0.6)
    
            ax.set_xlabel('Probability' if value_type == 'pred' else 'Count')
            ax.set_ylabel('Frequency')
            ax.set_title(f"{phase.capitalize()} {value_type.capitalize()} Classes")
            ax.legend()
        
        plt.tight_layout()
        plt.suptitle('Comparing True and Predicted Class Distributions')
        plt.savefig(folder + 'class_comparison_histograms.png')
        plt.close()
    
        return

        