import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from sbi import utils
from sbi import analysis
from sbi import inference
from sbi.inference import SNPE, simulate_for_sbi, prepare_for_sbi
import numpy as np


class InvariantModule(nn.Module):
    """Implements an invariant module with keras."""
    
    def __init__(self, meta, input_dim):
        super(InvariantModule, self).__init__()
        

        self.input_dim = input_dim
        
        self.s1 = nn.ModuleList(nn.Sequential(nn.Linear(in_features = self.input_dim, out_features=meta['s1']['features']), meta['s1']['activation']))
        for i in range(meta['n_dense_s1'] - 1):
            self.s1.add_module('s1layer{i}', nn.Linear(in_features = meta['s1']['features'], out_features = meta['s1']['features']))
            self.s1.add_module('activation{i}', nn.ReLU())
            
        self.s2 = nn.ModuleList(nn.Sequential(nn.Linear(in_features = meta['s1']['features'], out_features=meta['s2']['features']), meta['s2']['activation']))
        for i in range(meta['n_dense_s2'] - 1):
            self.s2.add_module(f's2layer{i}', nn.Linear(in_features = meta['s2']['features'], out_features = meta['s2']['features']))
            self.s2.add_module(f'activation{i}', nn.ReLU())
        
        #self.s2 = nn.Sequential(nn.Linear(in_features=32, out_features=64), nn.ReLU(), nn.Linear(in_features=64, out_features=64), nn.ReLU())
    def call(self, x):
        """Performs the forward pass of a learnable invariant transform.
        
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch_size, N, x_dim)
        
        Returns
        -------
        x_reduced : torch.Tensor
            Output of shape (batch_size, out_dim)
        """
        #Because of the ModuleList structure we'll need to manually call on each layer of said ModuleList
        for layer in self.s1:
            tensor = layer(x)
            x = tensor
        #After this call x should be the output of s1
        #Note that the data structure is [batchsize, n_obs, data_dim] so here we are taking the mean over n_obs
        #x_reduced.shape = [batch_size, s1_feature]
        x_reduced = torch.mean(x, axis=1)
        
        for layer in self.s2:
            tensor = layer(x_reduced)
            x_reduced = tensor
        
        return x_reduced

class EquivariantModule(nn.Module):
    """Implements an equivariant module with keras."""
    
    def __init__(self, meta, input_dim):
        super(EquivariantModule, self).__init__()
        self.input_dim=input_dim
        self.invariant_module = InvariantModule(meta, input_dim = self.input_dim)
        self.s3 = nn.ModuleList(nn.Sequential(nn.Linear(in_features = self.input_dim + meta['s2']['features'] \
                                                     ,out_features=meta['s3']['features']), meta['s3']['activation']))
        for i in range(meta['n_dense_s3'] - 1):
            self.s3.add_module(f's3layer{i}', nn.Linear(in_features = meta['s3']['features'], out_features = meta['s3']['features']))
            self.s3.add_module(f'activation{i}', nn.ReLU())
        #self.s3 = nn.Sequential(nn.Linear(in_features = self.input_dim + 64, out_features=32), nn.ReLU(), nn.Linear(in_features= 32, out_features=32), nn.ReLU())
                    
    def call(self, x):
        """Performs the forward pass of a learnable equivariant transform.
        
        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, N, x_dim)
        
        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, N, equiv_dim)
        """
        
        # Store N
        N = int(x.shape[1])
        
        # Output dim is (batch_size, inv_dim) - > (batch_size, N, inv_dim)
        out_inv = self.invariant_module.call(x)
        out_inv_rep = torch.stack([out_inv] * N, axis=1)
        
        # Concatenate each x with the repeated invariant embedding (batch_size, N, inv_dim) -> (batch_size, N, inv_dim + data_dim)
        out_c = torch.cat([x, out_inv_rep], axis=-1)
        
        # Pass through equivariant func
        for layer in self.s3:
            tensor = layer(out_c)
            out_c = tensor
  
        return out_c

class InvariantNetwork(nn.Module):
    """Implements an invariant network with keras.
    """

    def __init__(self, meta, input_dim, gpu = True):
        super(InvariantNetwork, self).__init__()
        self.meta=meta
        self.gpu=gpu
        self.input_dim = input_dim
        #Make a sequential based on the number of n_equiv
        #equiv_seq1 is the layer that directly receives input_dim, the rest follows information in dictionary.
        self.equiv = nn.ModuleList()
        self.equiv.add_module(f'inputLayer', EquivariantModule(meta = self.meta, input_dim = self.input_dim))
        for i in range(meta['n_equiv'] - 1):
             self.equiv.add_module(f'equivModule{i}', EquivariantModule(meta = self.meta, input_dim = self.meta['s3']['features']))

        self.inv = InvariantModule(meta = self.meta, input_dim=self.meta['s3']['features'])
    
    def __call__(self, x):
        """ Performs the forward pass of a learnable deep invariant transformation consisting of
        a sequence of equivariant transforms followed by an invariant transform.
        
        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, n_obs, data_dim)
        
        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, out_dim + 1)
        """
        
        # Extract n_obs and create sqrt(N) vector
        N = int(x.shape[1])
        if self.gpu:
            N_rep = torch.sqrt(N * torch.ones((x.shape[0], 1))).cuda()
        else:
            N_rep = torch.sqrt(N * torch.ones((x.shape[0], 1)))

        # Pass through series of augmented equivariant transforms
        for layer in self.equiv:
            tensor = layer.call(x)
            x = tensor


        out_equiv = x

        # Pass through final invariant layer and concatenate with N_rep
        if self.gpu:
            out_inv = self.inv.call(out_equiv).cuda()
        else:
            out_inv = self.inv.call(out_equiv)
        out = torch.cat((out_inv, N_rep), axis=-1)

        return out