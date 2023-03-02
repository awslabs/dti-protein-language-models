import os
import sys
import shutil
import re
import random
import time
import json
import glob
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Utility Imports
from tqdm import tqdm
from pprint import pprint
from datetime import datetime
import argparse
import time

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ProtBERT
from transformers import BertModel, BertTokenizer

# DGL
import dgl
from torch.utils.data._utils.collate import default_collate
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
from dgllife.model import GAT
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
from d2l import torch as d2l
from torch.nn.utils.rnn import pad_sequence

# Tranform Libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from scipy.stats import pearsonr

# PyTorch Lighning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint

# PyTDC
from tdc.multi_pred import DTI

class DTIDataset(Dataset):
    def __init__(self, drugs, targets, affinities, prot_model):
        self.drugs = drugs
        self.targets = targets
        self.affinities = affinities
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer()
        self.prot_model = prot_model

    def __len__(self):
        return len(self.affinities)

    def __getitem__(self, idx):
        smiles = self.drugs[idx]
        mol_graph = smiles_to_bigraph(smiles, 
                                      node_featurizer=self.atom_featurizer, 
                                      edge_featurizer=self.bond_featurizer,
                                      )
        mol_graph = dgl.add_self_loop(mol_graph)
        sequence = self.targets[idx]
        temp = [l for l in sequence]
        temp = " ".join(temp)
        temp = re.sub(r"[UZOB]", "X", temp)
        label = self.affinities[idx]
        return mol_graph, temp, label

def DTILitDataset(dataset_name, batch_size, target_scaler, prot_model, pin_memory=False, num_workers=0):

    dataset = DTI(name = dataset_name)
    if dataset_name == 'DAVIS' or dataset_name == 'BindingDB_IC50':
        dataset.convert_to_log(form = 'binding')
    split = dataset.get_split()

    train_data = split["train"]
    train_data.Y = target_scaler.fit_transform(train_data.Y.values.reshape(-1, 1))
    train_drugs = train_data.Drug.tolist()
    train_targets = train_data.Target.tolist()
    train_affinities = train_data.Y.tolist()
    train_dataset = DTIDataset(train_drugs, train_targets, train_affinities, prot_model)

    valid_data = split["valid"]
    valid_drugs = valid_data.Drug.tolist()
    valid_targets = valid_data.Target.tolist()
    valid_affinities = valid_data.Y.tolist()
    valid_dataset = DTIDataset(valid_drugs, valid_targets, valid_affinities, prot_model)


    test_data = split["test"]
    test_drugs = test_data.Drug.tolist()
    test_targets = test_data.Target.tolist()
    test_affinities = test_data.Y.tolist()
    test_dataset = DTIDataset(test_drugs, test_targets, test_affinities, prot_model)

    return train_dataset, valid_dataset, test_dataset

def collate_fn(batch):
        mol_graphs, protein_sequences, labels = tuple(zip(*batch))
        return dgl.batch(mol_graphs), default_collate(protein_sequences), default_collate(labels)

###############################################################################
# Model Initialization
# --------------------
class GATEmbedding(nn.Module):
    def __init__(self,
                 in_feats, 
                 hidden_feats,
                 num_heads,
                 dropouts):
        super(GATEmbedding, self).__init__()

        self.gnn = GAT(in_feats, 
                       hidden_feats=hidden_feats, 
                       num_heads=num_heads,
                       feat_drops=dropouts,
                       attn_drops=dropouts)
        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.gnn_out_feats = gnn_out_feats
        self.readout = WeightedSumAndMax(gnn_out_feats)

    def forward(self, g, in_feats, readout=False):
        node_feats = self.gnn(g, in_feats)
        if readout:
            return self.readout(g, node_feats)
        else:
            batch_num_nodes = g.batch_num_nodes().tolist()
            return pad_sequence(torch.split(node_feats, batch_num_nodes, dim=0), batch_first=True)

class DTILitNetwork(pl.LightningModule):
    def __init__(self,
                 prot_model,
                 criterion,
                 batch_size,
                 lr,
                 max_prot_len,
                 dataset,
                 target_scaler,
                 pin_memory=False,
                 num_workers=0,
                 in_feats=74,
                 graph_hidden_feats=[74, 128], # GraphDTA: [74, 128], Original: [32, 32]
                 graph_num_heads=[10, 1], # GraphDTA: [10, 1], Original: [4, 4]
                 use_cross_attention=False,
                 cross_hidden_feats=128, # Original: 128
                 cross_num_heads=4, # Original: 4
                 dense_hidden_feats=[1024, 256], # GraphDTA: [1024, 256], Original: 64
                 dropout=0.2, # GraphDTA: 0.2,
                 verbose=False):
        super(DTILitNetwork, self).__init__()
        self.verbose = verbose
        self.batch_size = batch_size
        self.lr = lr
        self.max_prot_len = max_prot_len
        self.train_dataset, self.valid_dataset, self.test_dataset = dataset
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.target_scaler = target_scaler

        self.use_cross_attention = use_cross_attention

        # Protein Language Model
        self.prot_model = BertModel.from_pretrained(prot_model)
        prot_dim = self.prot_model.pooler.dense.out_features

        # Protein Language Model Tokenizer
        self.prot_tokenizer = BertTokenizer.from_pretrained(prot_model, do_lower_case=False)

        # Loss Function
        self.criterion = criterion

        self.mol_model = GATEmbedding(in_feats=in_feats, 
                                      hidden_feats=graph_hidden_feats,
                                      num_heads=graph_num_heads,
                                      dropouts=[dropout]*len(graph_hidden_feats))
        self.dense = nn.ModuleList()
        if self.use_cross_attention:
            self.residue_attention = d2l.MultiHeadAttention(
                                                         query_size=self.mol_model.gnn_out_feats,
                                                         key_size=prot_dim, 
                                                         value_size=prot_dim, 
                                                         num_hiddens=cross_hidden_feats,
                                                         num_heads=cross_num_heads,
                                                         dropout=dropout)
            self.molecule_attention = d2l.MultiHeadAttention(
                                                         query_size=prot_dim,
                                                         key_size=self.mol_model.gnn_out_feats, 
                                                         value_size=self.mol_model.gnn_out_feats, 
                                                         num_hiddens=cross_hidden_feats,
                                                         num_heads=cross_num_heads,
                                                         dropout=dropout)
            self.dense.append(nn.Linear(2*cross_hidden_feats, dense_hidden_feats[0]))
        else:
            self.prot_fc = nn.Linear(prot_dim, 2*self.mol_model.gnn_out_feats)
            self.dense.append(nn.Linear(2*2*self.mol_model.gnn_out_feats, dense_hidden_feats[0]))
        for i in range(1, len(dense_hidden_feats)):
            self.dense.append(nn.Linear(dense_hidden_feats[i-1], dense_hidden_feats[i]))
        self.output = nn.Linear(dense_hidden_feats[-1], 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, mol_graphs, atom_mask, encoded_proteins, protein_mask):
        if self.use_cross_attention:
            x_mol = self.mol_model(mol_graphs, in_feats=mol_graphs.ndata['h'], readout=False)
            if self.verbose: print("Molecule tensor shape:", x_mol.shape)
            x_prot = self.prot_model(**encoded_proteins).last_hidden_state[:, 1:-1, :] # Don't use <CLS> and <SEP> tokens
            if self.verbose: print("Protein tensor shape:", x_prot.shape)
                
            x1 = self.residue_attention(queries=x_mol,
                                       keys=x_prot, 
                                       values=x_prot,
                                       valid_lens=atom_mask)
            x1 = torch.bmm(atom_mask.unsqueeze(1), x1).squeeze(1)
            
            x2 = self.molecule_attention(queries=x_prot,
                                       keys=x_mol, 
                                       values=x_mol,
                                       valid_lens=protein_mask)
            x2 = torch.bmm(protein_mask.unsqueeze(1), x2).squeeze(1)
            
            x = torch.cat((x1, x2), axis=1)
            for layer in self.dense:
                x = self.dropout(self.activation(layer(x)))
            return self.output(x)
        else:
            x_mol = self.mol_model(mol_graphs, in_feats=mol_graphs.ndata['h'].half(), readout=True)
            if self.verbose: print("Molecule tensor shape:", x_mol.shape)
            x_prot = self.prot_model(**encoded_proteins).last_hidden_state[:, 0, :]
            if self.verbose: print("Protein tensor shape:", x_prot.shape)
            x_prot = self.prot_fc(x_prot)
            x = torch.cat((x_prot, x_mol), axis=1)
            for layer in self.dense:
                x = self.dropout(self.activation(layer(x)))
            return self.output(x)

    def training_step(self, batch, batch_idx):
        mol_graphs, protein_sequences, labels = batch
        y_true = labels.unsqueeze(1).float()

        encoded_proteins = self.prot_tokenizer(protein_sequences, 
                                  return_tensors='pt', 
                                  max_length=self.max_prot_len, 
                                  truncation=True, 
                                  padding=True, 
                                  return_length=True)
        atoms_per_mol = mol_graphs.batch_num_nodes().tolist()
        atom_mask = [torch.ones(x) for x in atoms_per_mol]
        atom_mask = pad_sequence(atom_mask, batch_first=True)
        protein_lens = encoded_proteins.pop(key="length")
        max_len = torch.max(protein_lens)
        protein_mask = [torch.ones(x) if x < (max_len-2) else torch.ones(max_len-2) for x in protein_lens.tolist()]
        protein_mask = pad_sequence(protein_mask, batch_first=True)
        y_pred = self.forward(mol_graphs.to(self.device), atom_mask.to(self.device), encoded_proteins.to(self.device), protein_mask.to(self.device)).float()
        loss = self.criterion(y_pred, y_true)
        self.log('train_loss', loss)
        
        return {'loss': loss}

    def training_epoch_end(self, outputs, **kwargs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        print('Epoch: {}, Training MSE: {:.4f}q'.format(self.current_epoch, avg_loss))
        # return {'loss': avg_loss}

    def validation_step(self, batch, batch_idx):
        mol_graphs, protein_sequences, labels = batch
        y_true = labels.unsqueeze(1)

        encoded_proteins = self.prot_tokenizer(protein_sequences, 
                                  return_tensors='pt', 
                                  max_length=self.max_prot_len, 
                                  truncation=True, 
                                  padding=True, 
                                  return_length=True)
        atoms_per_mol = mol_graphs.batch_num_nodes().tolist()
        atom_mask = [torch.ones(x) for x in atoms_per_mol]
        atom_mask = pad_sequence(atom_mask, batch_first=True)
        protein_lens = encoded_proteins.pop(key="length")
        max_len = torch.max(protein_lens)
        protein_mask = [torch.ones(x) if x < (max_len-2) else torch.ones(max_len-2) for x in protein_lens.tolist()]
        protein_mask = pad_sequence(protein_mask, batch_first=True)
        y_pred = self(mol_graphs.to(self.device), atom_mask.to(self.device), encoded_proteins.to(self.device), protein_mask.to(self.device))

        return {'true': y_true.squeeze(1).detach().cpu().numpy(), 'pred': y_pred.squeeze(1).detach().cpu().numpy()}

    def validation_epoch_end(self, outputs, **kwargs):
        y_true_final = np.concatenate([x['true'] for x in outputs])
        y_pred_final = self.target_scaler.inverse_transform(np.concatenate([x['pred'] for x in outputs]).reshape(-1, 1)).flatten()
        
        mse = (np.square(y_pred_final - y_true_final)).mean()
        pcc = pearsonr(y_true_final, y_pred_final)[0]

        print('Epoch: {}, Validation MSE: {:.4f}, Validation PCC: {:.4f}'.format(self.current_epoch, mse, pcc))
        # return mse, pcc

    def test_step(self, batch, batch_idx):
        mol_graphs, protein_sequences, labels = batch
        y_true = labels.unsqueeze(1)

        encoded_proteins = self.prot_tokenizer(protein_sequences, 
                                  return_tensors='pt', 
                                  max_length=self.max_prot_len, 
                                  truncation=True, 
                                  padding=True, 
                                  return_length=True)
        atoms_per_mol = mol_graphs.batch_num_nodes().tolist()
        atom_mask = [torch.ones(x) for x in atoms_per_mol]
        atom_mask = pad_sequence(atom_mask, batch_first=True)
        protein_lens = encoded_proteins.pop(key="length")
        max_len = torch.max(protein_lens)
        protein_mask = [torch.ones(x) if x < (max_len-2) else torch.ones(max_len-2) for x in protein_lens.tolist()]
        protein_mask = pad_sequence(protein_mask, batch_first=True)
        y_pred = self(mol_graphs.to(self.device), atom_mask.to(self.device), encoded_proteins.to(self.device), protein_mask.to(self.device))

        return {'true': y_true.squeeze(1).detach().cpu().numpy(), 'pred': y_pred.squeeze(1).detach().cpu().numpy()}
    
    def test_epoch_end(self, outputs, **kwargs):
        y_true_final = np.concatenate([x['true'] for x in outputs])
        y_pred_final = self.target_scaler.inverse_transform(np.concatenate([x['pred'] for x in outputs]).reshape(-1, 1)).flatten()
        
        mse = (np.square(y_pred_final - y_true_final)).mean()
        pcc = pearsonr(y_true_final, y_pred_final)[0]

        print('Test MSE: {:.4f}, Test PCC: {:.4f}'.format(mse, pcc))
        # return mse, pcc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr) # Optimizer
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=self.num_workers, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=self.num_workers, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=self.pin_memory, num_workers=self.num_workers, collate_fn=collate_fn)

# CLI
def parse_args():
    # Argument Parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs' ,type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--MAX_PROT_LEN', type=int, default=1024)
    parser.add_argument('--scale_target', type=bool, default=True)
    parser.add_argument('--model_choice', type=int, default=1)
    parser.add_argument('--dataset_choice', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default="checkpoints/")

    opt = parser.parse_args()
    return opt

def main():
    args = parse_args()
    print("Batch Size: " + str(args.batch_size))

    prot_models = ["Rostlab/prot_bert", "yarongef/DistilProtBert"]
    PROT_MODEL = prot_models[args.model_choice]

    datasets = ["DAVIS", "KIBA", "BindingDB_Kd", "BindingDB_Ki", "BindingDB_IC50"]
    DATASET = datasets[args.dataset_choice]

    criterion = nn.MSELoss() # Loss function
    
    if args.scale_target:
        target_scaler = StandardScaler() # Standard scaling of target
    else:
        target_scaler = FunctionTransformer(lambda x: x) # Do not scale target

    dataset = DTILitDataset(dataset_name = DATASET, 
                        batch_size = args.batch_size, 
                        target_scaler = target_scaler,
                        prot_model = PROT_MODEL,
                        pin_memory=True, 
                        num_workers=args.num_workers,
                        )

    checkpoint_callback = ModelCheckpoint(dirpath=args.save_dir, 
    every_n_epochs=10, monitor='train_loss', mode='min', filename="dti-{epoch:02d}-{train_loss:.2f}")

    model = DTILitNetwork(prot_model=PROT_MODEL, 
                        criterion=criterion,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        max_prot_len=args.MAX_PROT_LEN,
                        dataset=dataset,
                        target_scaler=target_scaler,
                        pin_memory=True,
                        num_workers=args.num_workers,
                        )
    if args.model_choice == 0 or args.model_choice == 1:
        for param in model.prot_model.embeddings.parameters():
            param.requires_grad = False

    # Enable CPU Offloading, and offload parameters to CPU
    trainer = Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        max_epochs=args.epochs,
        strategy=DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
        ),
        precision=16,
        callbacks=[checkpoint_callback]
    )
    
    start = time.time()
    trainer.fit(model)
    end = time.time()
    print("Training Time: " + str(end - start))
    trainer.test(model)

if __name__ == '__main__':
    main()
