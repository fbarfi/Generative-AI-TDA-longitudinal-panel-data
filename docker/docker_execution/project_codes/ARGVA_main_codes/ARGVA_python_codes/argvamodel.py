import pytorch_lightning as pl
from torch_geometric.nn import ARGVA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                                accuracy_score,precision_score,
                                recall_score, f1_score,roc_auc_score,
                                average_precision_score)
import numpy as np

from .encoder import Encoder
from .decoder import Decoder
from .discriminator import Discriminator

class ARGVAModel(pl.LightningModule):
    def __init__(self, args,year):
        super(ARGVAModel, self).__init__()
        
        self.EPS = 1e-15
        self.args = args
        self.year = year
        self.n_epochs = args['n_epochs']
        self.models_folder = args['folder'] + f'saved_ARGVA_models/'
        self.batch_size = args['batch_size']
        self.encoder = Encoder(args)
        self.discriminator = Discriminator(args)
        self.decoder = Decoder(args)
        self.classifier = nn.Linear(self.encoder.out_channels, 1)
        self.lr = args['learning_rate']
        self.model = ARGVA(self.encoder,self.decoder,self.discriminator)
        self.save_hyperparameters(args)

        self.phases = ['train','val']
        self.outputs = {'train': [], 'val': []}
        self.att_weights = {'train': [], 'val': []}
        
    def encode(self, x, edge_index, edge_attr):
        return self.encoder(x, edge_index, edge_attr)

    def decode(self, z, edge_index, edge_attr,tda):
        return self.decoder(z, edge_index, edge_attr,tda)

    def discriminate(self, x, edge_index, edge_attr):
        return self.discriminator(x, edge_index, edge_attr)

    def forward(self, x, edge_index, edge_attr,tda):
        mu, logstd,_ = self.encode(x, edge_index, edge_attr)
        z = mu + torch.exp(logstd) * torch.randn_like(logstd)
        out = self.decode(z, edge_index, edge_attr,tda)
        return out

    def forward_all(self, x, edge_index, edge_attr, tda,sigmoid=True):
        mu, logstd,_ = self.encode(x, edge_index, edge_attr)
        z = mu + torch.exp(logstd) * torch.randn_like(logstd)
        out = self.decoder.forward_all(z, edge_index, edge_attr, tda,
                                       sigmoid=sigmoid)
        return out

    def reg_loss(self, z, edge_index, edge_attr):
        real = torch.sigmoid(self.discriminator(z, edge_index, edge_attr))
        real_loss = -torch.log(real + self.EPS).mean()
        return real_loss

    def discriminator_loss(self, z, edge_index, edge_attr):
        real = torch.sigmoid(self.discriminator(torch.randn_like(z), edge_index, 
                                                edge_attr))
        fake = torch.sigmoid(self.discriminator(z.detach(), edge_index, edge_attr))
        real_loss = -torch.log(real + self.EPS).mean()
        fake_loss = -torch.log(1 - fake + self.EPS).mean()
        return real_loss + fake_loss

    def kl_loss(self, z, edge_index, edge_attr):
        mu, logstd,_= self.encode(z, edge_index, edge_attr)
        return -0.5 * (torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2)).mean()

    def recon_loss(self, z, x, edge_index, edge_attr,tda):
        node_reconstructions = self.decoder(z, edge_index, edge_attr,tda)
        loss = F.mse_loss(node_reconstructions, x)
        return loss

    def z_parameterize(self, model, x, edge_index, edge_attr):
        mu, logvar,att_w = model(x, edge_index, edge_attr)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z,att_w

    def masking_data(self, data, mask):
        # Convert boolean mask to indices if needed
        if mask.dtype == torch.bool:
            subset = mask.nonzero(as_tuple=False).view(-1)
        else:
            subset = mask

        edge_index_masked, edge_attr_masked = subgraph(
                                                        subset=subset,
                                                        edge_index=data.edge_index,
                                                        edge_attr=data.edge_attr,
                                                        relabel_nodes=True,
                                                        num_nodes=data.num_nodes 
                                                    )
        return edge_index_masked, edge_attr_masked


    def compute_classification_metrics(self, logits, labels):
        probabilities = torch.sigmoid(logits).cpu().detach().numpy()
        pred = (probabilities >= 0.5).astype(int)
        y_true = labels.flatten()
        y_pred = pred.flatten()
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return accuracy, precision, recall, f1

    def compute_regression_metrics(self, reconstructed, original):
        original = original.float()
        y_true = original.cpu().detach().numpy().flatten()
        y_pred = reconstructed.cpu().detach().numpy().flatten()
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return mse, mae

    def classify(self, x, edge_index, edge_attr):
        mu, logvar,_ = self.encoder(x, edge_index, edge_attr)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        logits = self.classifier(z)  
        return logits

    def compute_auc_ap(self, z, node_labels, edge_index, edge_attr,tda):
        node_predictions = self.decoder(z,
                                       edge_index,
                                       edge_attr,
                                       tda)
        pred_probs = F.softmax(node_predictions, dim=1)
        y = node_labels.view(-1)
        y, pred_probs = y.detach().cpu().numpy(), pred_probs.detach().cpu().numpy()
        if len(np.unique(y)) == 2:
            auc = roc_auc_score(y, pred_probs[:, 1])
            ap = average_precision_score(y, pred_probs[:, 1])
        else:
            auc = float('nan')  
            ap = float('nan')  
        return auc, ap

    def training_step(self, data, batch_idx):
        self.model.train()
        
        train_mask = data.train_mask
        edge_index_masked, edge_attr_masked = self.masking_data(data, train_mask)
        z,att_w = self.z_parameterize(self.encoder, 
                                data.x[train_mask],
                                edge_index_masked,
                                edge_attr_masked,
                                )
        discriminator_loss = 0
        for _ in range(10):
            idx = torch.randperm(data.x[train_mask].shape[0])
            d_loss = self.discriminator_loss(z[idx], 
                                            edge_index_masked,
                                            edge_attr_masked,
                                            )
            discriminator_loss += d_loss
        discriminator_loss /= 10

        reg_loss = self.reg_loss(z, 
                                edge_index_masked,
                                edge_attr_masked)
        recon_loss = self.recon_loss(z, 
                                    data.x[train_mask],
                                    edge_index_masked,
                                    edge_attr_masked,
                                    data.glabel)
        
        kl_loss = (1 / data.num_nodes) * self.kl_loss(z, 
                                                     edge_index_masked,
                                                     edge_attr_masked)
        loss = reg_loss + recon_loss + kl_loss + discriminator_loss
        self.log('train_loss', loss, 
                             prog_bar=False, 
                             logger=True, 
                             batch_size=self.batch_size, 
                             on_epoch = True)
        
        self.outputs['train'].append(z.detach().cpu())
        self.att_weights['train'].append(att_w)
        
        return loss

    def validation_step(self, data, batch_idx):

        self.model.eval()

        val_mask = data.val_mask
        edge_index_masked, edge_attr_masked = self.masking_data(data, val_mask)
        
        z, att_w = self.z_parameterize(self.encoder, 
                                data.x[val_mask], 
                                edge_index_masked,
                                edge_attr_masked)

        discriminator_loss = 0
        for _ in range(10):
            idx = torch.randperm(data.x[val_mask].shape[0])
            d_loss = self.discriminator_loss(z[idx], 
                                                   edge_index_masked,
                                                   edge_attr_masked)
            discriminator_loss += d_loss
        discriminator_loss /= 10
        
        reg_loss = self.reg_loss(z, 
                                 edge_index_masked,
                                 edge_attr_masked)
        
        recon_loss = self.recon_loss(z, 
                                     data.x[val_mask],
                                     edge_index_masked,
                                     edge_attr_masked,
                                    data.glabel)
        
        kl_loss = (1 / data.num_nodes) * self.kl_loss(z, 
                                                      edge_index_masked,
                                                      edge_attr_masked)
        loss = reg_loss + recon_loss + kl_loss + discriminator_loss
        self.log('val_loss', loss, 
                             prog_bar=False, 
                             logger=True, 
                             batch_size=self.batch_size, 
                             on_epoch = True)
        
        self.outputs['val'].append(z.detach().cpu())
        self.att_weights['val'].append(att_w)

        reconstructed = self.decoder(z, 
                                        edge_index_masked,
                                        edge_attr_masked,
                                        data.glabel)
        mse,mae = self.compute_regression_metrics(reconstructed, 
                                                  data.x[val_mask])
        self.log('mse', mse, 
                         prog_bar=False, 
                         logger=True, 
                         batch_size=self.batch_size, 
                         on_epoch = True)
        self.log('mae', mae,
                        prog_bar=False, 
                         logger=True, 
                         batch_size=self.batch_size, 
                         on_epoch = True)
        
        labels = data.y[val_mask].cpu().detach().numpy()
        logits = self.classify(data.x[val_mask], 
                                 edge_index_masked,
                                 edge_attr_masked)
        
        accuracy, precision, recall, f1 = self.compute_classification_metrics(logits, labels)
        self.log('accuracy', accuracy,
                            prog_bar=False, 
                         logger=True, 
                         batch_size=self.batch_size, 
                         on_epoch = True)
        self.log('precision', precision,
                            prog_bar=False, 
                         logger=True, 
                         batch_size=self.batch_size, 
                         on_epoch = True)
        self.log('recall', recall,
                         prog_bar=False, 
                         logger=True, 
                         batch_size=self.batch_size, 
                         on_epoch = True)
        self.log('f1', f1,
                        prog_bar=False, 
                         logger=True, 
                         batch_size=self.batch_size, 
                         on_epoch = True)
        
        auc,ap = self.compute_auc_ap(z, 
                            data.y[val_mask],
                            edge_index_masked,
                            edge_attr_masked,
                            data.glabel)      
        self.log('auc', auc, 
                         prog_bar=False, 
                         logger=True, 
                         batch_size=self.batch_size, 
                         on_epoch = True)
        self.log('ap', ap,
                        prog_bar=False, 
                         logger=True, 
                         batch_size=self.batch_size, 
                         on_epoch = True)
        
        return loss

    def test_step(self, batch, batch_idx):
        data = batch
        self.model.eval()
        test_mask = data.val_mask
        edge_index_masked, edge_attr_masked = self.masking_data(data, test_mask)
        
        z,_ = self.z_parameterize(self.encoder, 
                                data.x[test_mask], 
                                 edge_index_masked,
                                 edge_attr_masked)

        discriminator_loss = 0
        for _ in range(10):
            idx = torch.randperm(data.x[test_mask].shape[0])
            d_loss = self.discriminator_loss(z[idx], 
                                             edge_index_masked,
                                             edge_attr_masked)
            discriminator_loss += d_loss
        discriminator_loss /= 10
        
        reg_loss = self.reg_loss(z, 
                                 edge_index_masked,
                                 edge_attr_masked)
        recon_loss = self.recon_loss(z, 
                                     data.x[test_mask],
                                     edge_index_masked,
                                     edge_attr_masked,
                                    data.glabel)
        
        kl_loss = (1 / data.num_nodes) * self.kl_loss(z, 
                                                       edge_index_masked,
                                                       edge_attr_masked)
        loss = reg_loss + recon_loss + kl_loss + discriminator_loss
        self.log('test_loss', loss, 
                              prog_bar=False, 
                             logger=True, 
                             batch_size=self.batch_size, 
                             on_epoch = True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def on_train_end(self):
        for phase in self.phases:
            att_weights = self.get_epoch_outputs()[phase]
            attw_path = f'{self.models_folder}/GNN_attention_weights_{phase}_'
            attw_path += f'last_epoch_{self.year}.pth'
            torch.save(att_weights, attw_path)
    
    def get_epoch_outputs(self):
        out_attw = {}
        for phase in self.phases:
            wlist, elist = [], []
            for idx, attw in enumerate(self.att_weights[phase]):
                wlist.append(attw[1])
                elist.append(attw[0])
            out_attw[phase] = (torch.cat(elist, dim=1),torch.cat(wlist))
            
        return out_attw