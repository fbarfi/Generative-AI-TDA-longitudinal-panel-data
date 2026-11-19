import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from .dgl_gnnmodel import GNNModel
from .graphsdatamodule import GraphsDataModule



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', weight_by_magnitude=False):
        """
        Focal Loss for multi-class classification with dynamic re-weighting.

        Args:
        - alpha (float or list): Weighting factor for class imbalance
        (default 1). Can be a float or a list of class weights.
        - gamma (float): Focusing parameter to adjust the rate at
        which easy examples are down-weighted (default 2).
        - reduction (str): Specifies the reduction to apply
        to the output: 'none' | 'mean' | 'sum' (default 'mean').
        - weight_by_magnitude (bool): If True, apply a dynamic reduction
        based on the magnitude of the loss (default False).
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight_by_magnitude = weight_by_magnitude

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=1)
        labels_one_hot = torch.eye(logits.size(1))[labels] #.to(logits.device)
        probs_for_true_class = (probs * labels_one_hot).sum(dim=1)
        log_probs_for_true_class = probs_for_true_class.log()
        focal_term = (1 - probs_for_true_class) ** self.gamma

        if isinstance(self.alpha, (float, int)):
            alpha_factor = self.alpha
        elif isinstance(self.alpha, list):
            alpha_factor = torch.tensor(self.alpha)[labels] #.to(logits.device)
        else:
            raise ValueError("Alpha must be a float or a list of class weights.")

        loss = -alpha_factor * focal_term * log_probs_for_true_class

        if self.weight_by_magnitude:
            loss_weight = loss / loss.max()
            loss = loss * loss_weight

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class RunGNNModel(pl.LightningModule):

    def __init__(self, args):
        super(RunGNNModel, self).__init__()

        self.args = args
        self.learning_rate = args['learning_rate']
        self.num_classes = args['num_classes']
        self.out_channels = args['out_channels']
        self.batch_size = args['batch_size']
        self.models_folder = args['models_folder']
        self.weight_decay = args['weight_decay']
        self.save_hyperparameters(args)
        self.criterion = nn.CrossEntropyLoss()
        self.model = GNNModel(args)
        self.data_module = GraphsDataModule(args)
        self.data_module.setup()

        self.outputs = {'train': [], 'val': [], 'test': []}
        self.att_weights = {'train': [], 'val': [], 'test': []}

    def forward(self, batch, batch_idx=None):
        g, labels = batch
        node_features = g.ndata["feat"]
        edge_features = g.edata["efeat"]
        out, att_w = self.model(g, node_features, edge_features)
        preds = F.softmax(out, dim=1)
        return preds, att_w


    def log_predictions(self, outputs, labels, phase):
        probas = F.softmax(outputs, dim=1)
        _, max_probas = torch.max(probas, dim=1)

        assert max_probas.max() < self.out_channels, "Predicted class index exceeds number of classes"

        class_predictions = F.one_hot(max_probas, self.out_channels).float()
        for i in range(self.out_channels):
            self.log(f'{phase}_pred_class_{i}', class_predictions[:, i].mean(),
                     prog_bar=False,
                     logger=True,
                     batch_size=self.batch_size,
                     on_step=False,
                     on_epoch=True)
        return

    def training_step(self, batch, batch_idx):
        g, labels  = batch

        outputs, att_w = self( batch, batch_idx)
        loss = self.criterion(outputs, labels)

        probas = F.softmax(outputs, dim=1)
        _, preds = torch.max(probas, dim=1)
        correct = torch.eq(preds, labels).sum().item()
        acc = correct / labels.size(0)

        self.log('train_loss', loss, prog_bar=False, logger=True,
                 batch_size=self.batch_size, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=False, logger=True,
                 batch_size=self.batch_size, on_step=False,  on_epoch=True)

        self.log_predictions(outputs, labels, 'train')

        self.outputs['train'].append(outputs.detach().cpu())
        self.att_weights['train'].append(att_w)

        return loss

    def validation_step(self, batch, batch_idx):
        g, labels  = batch

        outputs, att_w = self(batch, batch_idx)
        loss = self.criterion(outputs, labels)

        probas = F.softmax(outputs, dim=1)
        _, preds = torch.max(probas, dim=1)

        correct = torch.eq(preds, labels).sum().item()
        acc = correct / labels.size(0)

        self.log('val_loss', loss, prog_bar=False,
                 logger=True, batch_size=self.batch_size, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=False,
                 logger=True, batch_size=self.batch_size, on_step=False, on_epoch=True)

        self.log_predictions(outputs, labels, 'val')

        self.outputs['val'].append(outputs.detach().cpu())
        self.att_weights['val'].append(att_w)

        return loss


    def test_step(self, batch, batch_idx):
        g, labels = batch
        outputs, att_w = self(batch, batch_idx)
        self.outputs['test'].append(outputs.detach().cpu())
        self.att_weights['test'].append(att_w)

        # loss = self.criterion(outputs, labels)
        # probas = F.softmax(outputs, dim=1)
        # _, preds = torch.max(probas, dim=1)
        # correct = torch.eq(preds, labels).sum().item()
        # acc = correct / labels.size(0)

        # self.log('test_loss', loss, prog_bar=False,
        #         logger=True, batch_size=self.batch_size, on_step=False, on_epoch=True)
        # self.log('test_acc', acc, prog_bar=False,
        #         logger=True, batch_size=self.batch_size, on_step=False, on_epoch=True)

        # self.log_predictions(outputs, labels, 'test')

        return 


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        g, _ = batch
        node_features = g.ndata["feat"]
        edge_features = g.edata["efeat"]
        
        out,attw  = self.model(g, node_features, edge_features)

        if hasattr(g, "year"):
            num_graphs = out.size(0)
            year_tensor = torch.full((num_graphs,), g.year.item(), dtype=torch.long)
        else:
            year_tensor = torch.full((out.size(0),), -1, dtype=torch.long)

        return out, year_tensor


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        return optimizer

    def on_train_end(self):
        outputs, att_weights = self.get_epoch_outputs()
        for phase in ['train', 'val']:
            attw_path = f'{self.models_folder}/GNN_attention_weights_{phase}_last_epoch.ckpt'
            torch.save(att_weights[phase], attw_path)

            outputs_path = f'{self.models_folder}/GNN_outputs_{phase}_last_epoch.ckpt'
            torch.save(outputs[phase], outputs_path)

    def on_test_end(self):
        outputs, att_weights = self.get_epoch_outputs()
        for phase in ['test']:
            attw_path = f'{self.models_folder}/GNN_attention_weights_{phase}_last_epoch.ckpt'
            torch.save(att_weights[phase], attw_path)

            outputs_path = f'{self.models_folder}/GNN_outputs_{phase}_last_epoch.ckpt'
            torch.save(outputs[phase], outputs_path)

    def get_epoch_outputs(self):
        return self.outputs, self.att_weights
