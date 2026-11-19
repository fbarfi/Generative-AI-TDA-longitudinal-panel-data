import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric

from .edgegatconv_model import EdgeGATConv_Model
from .graphsdatamodule import GraphsDataModule

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction='mean'):
        """
        Focal Loss for multi-class classification.
        
        Args:
        - alpha (float or list): Weighting factor for class imbalance 
        (default 1). Can be a float or a list of class weights.
        - gamma (float): Focusing parameter to adjust the rate at 
        which easy examples are down-weighted (default 2).
        - reduction (str): Specifies the reduction to apply 
        to the output: 'none' | 'mean' | 'sum' (default 'mean').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=1)
        
        labels_one_hot = torch.eye(logits.size(1))[labels].to(logits.device)  
        probs_for_true_class = (probs * labels_one_hot).sum(dim=1)  
    
        log_probs_for_true_class = probs_for_true_class.log()
        
        focal_term = (1 - probs_for_true_class) ** self.gamma
    
        if isinstance(self.alpha, (float, int)):
            alpha_factor = self.alpha
        elif isinstance(self.alpha, list):
            alpha_factor = torch.tensor(self.alpha)[labels].to(logits.device)
        else:
            raise ValueError("Alpha must be a float or a list of class weights.")
    
        loss = -alpha_factor * focal_term * log_probs_for_true_class
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class RunModel(pl.LightningModule):
    def __init__(self, args):
        super(RunModel, self).__init__()
        self.save_hyperparameters(args)
        self.model = EdgeGATConv_Model(args)
        self.criterion = FocalLoss(alpha=0.25, gamma=2)
        self.lr = args['learning_rate']
        self.warmup_epochs = 5
        self.nepochs = args['nepochs']
        self.num_classes = args['num_classes']

    def forward(self, batch, batch_idx):
        batched_graph, labels = batch
        node_features = batched_graph.ndata["feat"]
        edge_features = batched_graph.edata["efeat"]
        return self.model(batched_graph, node_features, edge_features)

    def training_step(self, batch, batch_idx):
        
        batched_graph, labels = batch
        outputs = self(batch,batch_idx) 
        loss = self.criterion(outputs, labels)
       
        self.log('train_loss', loss, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        train_acc = correct / total

        self.log('train_acc', train_acc, 
                 on_step=False, on_epoch=True,
                 prog_bar=False,logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        
        batched_graph, labels = batch

        outputs = self(batch,batch_idx) 
        val_loss = self.criterion(outputs, labels)
        
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True)

        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        val_acc = correct / total
        
        self.log('val_acc', val_acc, 
                 on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True)

        return val_loss

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.warmup_epochs / self.nepochs,
                anneal_strategy='cos'
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

   