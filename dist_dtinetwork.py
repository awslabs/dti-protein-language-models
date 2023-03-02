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

from tqdm import tqdm
from pprint import pprint
from datetime import datetime

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import models, transforms

import dgl
from torch.utils.data._utils.collate import default_collate
from transformers import BertModel, BertTokenizer
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer, smiles_to_bigraph
from dgllife.model import AttentiveFPGNN, AttentiveFPReadout
from d2l import torch as d2l
from torch.nn.utils.rnn import pad_sequence

import torch.distributed as dist

from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer

def init_process_group(world_size, rank):
    dist.init_process_group(
        # backend='gloo',     # change to 'nccl' for multiple GPUs
        backend='nccl',
        init_method='tcp://127.0.0.1:12345',
        world_size=world_size,
        rank=rank)

###############################################################################
# Data Loader Preparation
# -----------------------

from dgl.data import split_dataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.distributed import DistributedSampler

class DTIDataset(Dataset):
    def __init__(self, drugs, targets, affinities):
        self.drugs = drugs
        self.targets = targets
        self.affinities = affinities
        self.atom_featurizer = AttentiveFPAtomFeaturizer()
        self.bond_featurizer = AttentiveFPBondFeaturizer()

    def __len__(self):
        return len(self.affinities)

    def __getitem__(self, idx):
        smiles = self.drugs[idx]
        mol_graph = smiles_to_bigraph(smiles, 
                                      node_featurizer=self.atom_featurizer, 
                                      edge_featurizer=self.bond_featurizer,
                                      )
        sequence = self.targets[idx]
        temp = [l for l in sequence]
        temp = " ".join(temp)
        temp = re.sub(r"[UZOB]", "X", temp)
        label = self.affinities[idx]
        return mol_graph, temp, label
    
    
def collate_fn(batch):
    mol_graphs, protein_sequences, labels = tuple(zip(*batch))
    return dgl.batch(mol_graphs), default_collate(protein_sequences), default_collate(labels) 


def get_dataloaders(dataset, seed, batch_size, world_size, rank, pin_memory=False, num_workers=0):
    split = dataset.get_split()
    train_data = split["train"]
    valid_data = split["valid"]
    test_data = split["test"]
    
    target_scaler = MinMaxScaler()
    # target_scaler = StandardScaler()
    # target_scaler = FunctionTransformer(lambda x: x)
    train_data.Y = target_scaler.fit_transform(train_data.Y.values.reshape(-1, 1))
    
    # Create dataset and dataloader from PyTDC 
    train_drugs = train_data.Drug.tolist()
    train_targets = train_data.Target.tolist()
    train_affinities = train_data.Y.tolist()
    train_dataset = DTIDataset(train_drugs, train_targets, train_affinities)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, sampler=train_sampler, collate_fn=collate_fn)
    
    # Create dataset and dataloader from PyTDC 
    valid_drugs = valid_data.Drug.tolist()
    valid_targets = valid_data.Target.tolist()
    valid_affinities = valid_data.Y.tolist()
    valid_dataset = DTIDataset(valid_drugs, valid_targets, valid_affinities)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=128, pin_memory=pin_memory, num_workers=num_workers, sampler=valid_sampler, collate_fn=collate_fn)
    
    # Create dataset and dataloader from PyTDC 
    test_drugs = test_data.Drug.tolist()
    test_targets = test_data.Target.tolist()
    test_affinities = test_data.Y.tolist()
    test_dataset = DTIDataset(test_drugs, test_targets, test_affinities)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=128, pin_memory=pin_memory, num_workers=num_workers, sampler=test_sampler, collate_fn=collate_fn)

    return train_dataloader, valid_dataloader, test_dataloader, target_scaler

###############################################################################
# Model Initialization
# --------------------

class AttentiveFPEmbedding(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 num_timesteps=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(AttentiveFPEmbedding, self).__init__()

        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)

    def forward(self, g, node_feats, edge_feats, get_node_weight=False):
        node_feats = self.gnn(g, node_feats, edge_feats)
        batch_num_nodes = g.batch_num_nodes().tolist()
        return pad_sequence(torch.split(node_feats, batch_num_nodes, dim=0), batch_first=True)

    
class DTINetwork(nn.Module):
    def __init__(self,
                 prot_model,
                 mol_node_dim=39, 
                 mol_edge_dim=10, 
                 num_layers=2,
                 num_timesteps=2,
                 graph_feat_size=200,
                 hidden_dim=64, 
                 dropout=0.2,
                 use_residue_attention=True,
                 num_heads=4,
                 verbose=False):
        super(DTINetwork, self).__init__()
        self.verbose = verbose
        self.prot_model = prot_model
        prot_dim = prot_model.pooler.dense.out_features
        self.mol_model = AttentiveFPEmbedding(node_feat_size=mol_node_dim, 
                                              edge_feat_size=mol_edge_dim,
                                              graph_feat_size=graph_feat_size,
                                              num_layers=num_layers,
                                              num_timesteps=num_timesteps,
                                              dropout=dropout)
        self.use_residue_attention = use_residue_attention
        if self.use_residue_attention:
            self.residue_attention = d2l.MultiHeadAttention(
                                                         query_size=graph_feat_size,
                                                         key_size=prot_dim, 
                                                         value_size=prot_dim, 
                                                         num_hiddens=hidden_dim,
                                                         num_heads=num_heads,
                                                         dropout=dropout)
            self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.linear1 = nn.Linear(prot_dim+graph_feat_size, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, mol_graphs, atom_mask, encoded_proteins, protein_lens):
        x_mol = self.mol_model(mol_graphs, 
                               node_feats=mol_graphs.ndata['h'], 
                               edge_feats=mol_graphs.edata['e'])
        if self.verbose: print("Molecule tensor shape:", x_mol.shape)
        if self.use_residue_attention:
            x_prot = self.prot_model(**encoded_proteins).last_hidden_state[:, 1:-1, :] # Don't use <CLS> and <SEP> tokens
            if self.verbose: print("Protein tensor shape:", x_prot.shape)
            # x_mol = x_mol.unsqueeze(1)
            x = self.residue_attention(queries=x_mol,
                                       keys=x_prot, 
                                       values=x_prot,
                                       valid_lens=atom_mask)
            if self.verbose: print("Attention output shape:", x.shape)
            # x = x.squeeze(1)
        else:
            x_prot = self.prot_model(**encoded_proteins).last_hidden_state[:, 0, :]
            x = torch.cat((x_prot, x_mol), axis=1)
        x = self.dropout(self.activation(self.linear1(x)))
        x = self.linear2(x)
        return torch.sum(torch.mul(atom_mask, x.squeeze(2)), dim=1).unsqueeze(1)

###############################################################################
# To ensure same initial model parameters across processes, we need to set the
# same random seed before model initialization. Once we construct a model
# instance, we wrap it with :func:`~torch.nn.parallel.DistributedDataParallel`.
#

import torch
from torch.nn.parallel import DistributedDataParallel

def init_model(seed, device):
    torch.manual_seed(seed)
    prot_model = BertModel.from_pretrained("Rostlab/prot_bert")
    model = DTINetwork(prot_model).to(device)
    
    for param in model.prot_model.parameters():
        param.requires_grad = False
    
    if device.type == 'cpu':
        model = DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)

    return model

###############################################################################
# Main Function for Each Process
# -----------------------------

from scipy.stats import pearsonr


def evaluate(model, dataloader, loss_fn, target_scaler, device):
    prot_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model.eval()
    y_true_list = []
    y_pred_list = []
    total_loss = 0.0
    total_samples = 0
    for i, batch in enumerate(dataloader):
        mol_graphs, protein_sequences, labels = batch
        y_true = labels.unsqueeze(1).to(device)
        total_samples += y_true.shape[0]
        
        encoded_proteins = prot_tokenizer(protein_sequences, 
                                  return_tensors='pt', 
                                  max_length=512, 
                                  truncation=True, 
                                  padding=True, 
                                  return_length=True)

        mol_graphs = mol_graphs.to(device)
        atoms_per_mol = mol_graphs.batch_num_nodes().tolist()
        # print(atoms_per_mol)
        atom_mask = [torch.ones(x) for x in atoms_per_mol]
        atom_mask = pad_sequence(atom_mask, batch_first=True).to(device)
        encoded_proteins = encoded_proteins.to(device)
        protein_lens = encoded_proteins.pop(key="length")
        y_pred = model(mol_graphs, atom_mask, encoded_proteins, protein_lens)
#         loss = loss_fn(y_pred, y_true)
#         total_loss += loss.cpu().item()
        y_true_list.append(y_true.squeeze(1).detach().cpu().numpy())
        y_pred_list.append(y_pred.squeeze(1).detach().cpu().numpy())
#     total_loss /= total_samples

    y_true_final = np.concatenate(y_true_list)
    y_pred_final = target_scaler.inverse_transform(np.concatenate(y_pred_list).reshape(-1, 1)).flatten()
    mse = (np.square(y_pred_final - y_true_final)).mean()
    pcc = pearsonr(y_true_final, y_pred_final)[0]
    return mse, pcc


def train(model, optimizer, loss_fn, train_loader, val_loader, target_scaler, epochs=20, max_batches_per_epoch=1000, device="cpu"):
    prot_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    for epoch in range(1, epochs+1):
        model.train()
        training_loss = 0.0
        train_samples = 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            mol_graphs, protein_sequences, labels = batch
            y_true = labels.unsqueeze(1).to(device)
            train_samples += y_true.shape[0]
            encoded_proteins = prot_tokenizer(protein_sequences, 
                                  return_tensors='pt', 
                                  max_length=512, 
                                  truncation=True, 
                                  padding=True, 
                                  return_length=True)

            mol_graphs = mol_graphs.to(device)
            atoms_per_mol = mol_graphs.batch_num_nodes().tolist()
            # print(atoms_per_mol)
            atom_mask = [torch.ones(x) for x in atoms_per_mol]
            atom_mask = pad_sequence(atom_mask, batch_first=True).to(device)
            encoded_proteins = encoded_proteins.to(device)
            protein_lens = encoded_proteins.pop(key="length")
            y_pred = model(mol_graphs, atom_mask, encoded_proteins, protein_lens).to(torch.float64)
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            optimizer.step()
            training_loss += loss.cpu().item()
            if (i+1) >= max_batches_per_epoch:
                break
            elif (i+1)%100 == 0:
                print('Device: {}, Batch: {}, MSE: {:.4f}'.format(device, i+1, training_loss/train_samples))
            else:
                continue
        # training_loss /= len(train_loader.dataset)
        training_loss /= train_samples
        val_mse, val_pcc = evaluate(model, val_loader, loss_fn, target_scaler, device)
        print('Device: {}, Epoch: {}, Training MSE: {:.4f}, Validation MSE: {:.4f}, Validation PCC: {:.4f}'.format(device, epoch, training_loss, val_mse, val_pcc))

###############################################################################
# Define the main function for each process.
#

def main(rank, world_size, dataset, seed=0):
    init_process_group(world_size, rank)
    if torch.cuda.is_available():
        device = torch.device('cuda:{:d}'.format(rank))
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    print("Spawning process on device", 'cuda:{:d}'.format(rank))

    model = init_model(seed, device)
    
    lr = 0.001 # Learning rate
    criterion = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    batch_size = 16 # Per-GPU batch size for training (max ~128)
    train_loader, val_loader, test_loader, target_scaler = get_dataloaders(dataset,
                                                            seed,
                                                            batch_size=batch_size,
                                                            world_size=world_size,
                                                            rank=rank)

    test_mse, test_pcc = evaluate(model, test_loader, criterion, target_scaler, device)
    print('Device: {:}, Test MSE: {:.4f}, Test PCC: {:.4f}'.format(device, test_mse, test_pcc))
    
    epochs = 100 # Number of epochs to train the model
    max_batches_per_epoch = 500 # Maximum number of batches per GPU per epoch during training
    train(model=model, 
          loss_fn=criterion, 
          optimizer=optimizer,
          train_loader=train_loader,
          val_loader=val_loader,
          target_scaler=target_scaler,
          epochs=epochs,
          max_batches_per_epoch=max_batches_per_epoch,
          device=device)

    test_mse, test_pcc = evaluate(model, test_loader, criterion, target_scaler, device)
    print('Device: {:}, Test MSE: {:.4f}, Test PCC: {:.4f}'.format(device, test_mse, test_pcc))
    dist.destroy_process_group()

###############################################################################
# Finally we load the dataset and launch the processes.
#
# .. code:: python
#
if __name__ == '__main__':
    import torch.multiprocessing as mp

    from tdc.multi_pred import DTI
    
    torch.cuda.empty_cache()
   
    num_gpus = 8
    procs = []
    
#     dataset = DTI(name = 'BindingDB_Kd')
#     dataset = DTI(name = 'BindingDB_Ki')
#     dataset = DTI(name = 'BindingDB_IC50')
#     dataset.harmonize_affinities(mode = 'max_affinity')
#     dataset.convert_to_log(form = 'binding')
    
    dataset = DTI(name = 'DAVIS')
    dataset.convert_to_log(form = 'binding')
    
#     dataset = DTI(name = 'KIBA')
    
    mp.spawn(main, args=(num_gpus, dataset), nprocs=num_gpus)
