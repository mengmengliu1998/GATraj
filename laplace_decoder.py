""""
This is the Laplace decoder
Author: Mengmeng Liu
Date: 2022/09/24
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import time
from torch import nn, Tensor
from typing import Dict, List, Tuple, NamedTuple, Any
import matplotlib.pyplot as plt
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np


class GRUDecoder(nn.Module):

    def __init__(self, args) -> None:
        super(GRUDecoder, self).__init__()
        min_scale: float = 1e-3
        self.args = args
        self.input_size = self.args.hidden_size
        self.hidden_size = self.args.hidden_size
        self.future_steps = args.pred_length
        self.num_modes = args.final_mode
        self.min_scale = min_scale
        self.args = args
        self.lstm = nn.LSTM(input_size=self.hidden_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=False,
                          dropout=0,
                          bidirectional=False)
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        self.scale = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        self.multihead_proj_global = nn.Sequential(
                                    nn.Linear(self.input_size , self.num_modes * self.hidden_size),
                                    nn.LayerNorm(self.num_modes * self.hidden_size),
                                    nn.ReLU(inplace=True))   
        self.apply(init_weights)

    def forward(self, global_embed: torch.Tensor, hidden_state, cn) -> Tuple[torch.Tensor, torch.Tensor]:
        global_embed = self.multihead_proj_global(global_embed).view(-1, self.num_modes, self.hidden_size)  # [N, F, D]
        global_embed = global_embed.transpose(0, 1)  # [F, N, D]
        local_embed = hidden_state.repeat(self.num_modes, 1, 1)  # [F, N, D]
        cn = cn.repeat(self.num_modes, 1, 1)  # [F, N, D]
        pi = self.pi(torch.cat((local_embed, global_embed), dim=-1)).squeeze(-1).t()  # [N, F]
        global_embed = global_embed.reshape(-1, self.hidden_size)  # [F x N, D]
        global_embed = global_embed.expand(self.future_steps, *global_embed.shape)  # [H, F x N, D]
        local_embed = local_embed.reshape(-1, self.input_size).unsqueeze(0)  # [1, F x N, D]
        cn = cn.reshape(-1, self.input_size).unsqueeze(0)  # [1, F x N, D]
        out, _ = self.lstm(global_embed, (local_embed, cn))
        out = out.transpose(0, 1)  # [F x N, H, D]
        loc = self.loc(out)  # [F x N, H, 2]
        scale = F.elu_(self.scale(out), alpha=1.0) + 1.0 + self.min_scale  # [F x N, H, 2]
        loc = loc.view(self.num_modes, -1, self.future_steps, 2) # [F, N, H, 2]
        scale = scale.view(self.num_modes, -1, self.future_steps, 2) # [F, N, H, 2]
        return (loc, scale, pi) # [F, N, H, 2], [F, N, H, 2], [N, F]


class MLPDecoder(nn.Module):

    def __init__(self, args) -> None:
        super(MLPDecoder, self).__init__()
        min_scale: float = 1e-3
        self.args = args
        # self.input_size = self.args.hidden_size + self.args.z_dim
        self.input_size = self.args.hidden_size
        self.hidden_size = self.args.hidden_size
        self.future_steps = args.pred_length
        self.num_modes = args.final_mode
        self.min_scale = min_scale
        self.args = args
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.future_steps * 2))
        self.scale = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.future_steps * 2))
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True))
        self.multihead_proj_global = nn.Sequential(
                                    nn.Linear(self.input_size , self.num_modes * self.hidden_size),
                                    nn.LayerNorm(self.num_modes * self.hidden_size),
                                    nn.ReLU(inplace=True))   
        self.apply(init_weights)

    def forward(self, x_encode: torch.Tensor, hidden_state, cn) -> Tuple[torch.Tensor, torch.Tensor]:
        x_encode = self.multihead_proj_global(x_encode).view(-1, self.num_modes, self.hidden_size)  # [N, F, D]
        x_encode = x_encode.transpose(0, 1)  # [F, N, D]
        local_embed = hidden_state.repeat(self.num_modes, 1, 1)  # [F, N, D]
        pi = self.pi(torch.cat((local_embed, x_encode), dim=-1)).squeeze(-1).t()  # [N, F]
        out = self.aggr_embed(torch.cat((x_encode, local_embed), dim=-1))
        loc = self.loc(out).view(self.num_modes, -1, self.future_steps, 2)  # [F, N, H, 2]
        scale = F.elu_(self.scale(out), alpha=1.0).view(self.num_modes, -1, self.future_steps, 2) + 1.0
        scale = scale + self.min_scale  # [F, N, H, 2]
        return (loc, scale, pi) # [F, N, H, 2], [F, N, H, 2], [N, F]
   
    def plot_pred(self, loc, lock, N=10, groundtruth=True):
        """
        This is the plot function to plot the first scene
        lock:   [N, K, H, 2]
        loc: [N, F, H, 2]
        """
        
        fig,ax = plt.subplots()
        pred_seq = loc.shape[2]
        lock = lock.cpu().detach().numpy()
        loc = loc.cpu().detach().numpy()
        for m in range(loc.shape[0]):
            for i in range(loc.shape[1]):
                y_p_sum = np.cumsum(loc[m,i,:,:], axis=0)
                ax.plot(y_p_sum[:, 0], y_p_sum[:, 1], color='k', linewidth=1)
            for j in range(lock.shape[1]):
                y_sum = np.cumsum(lock[m,j,:,:], axis=0)
                ax.plot(y_sum[:, 0], y_sum[:, 1], color='r', linewidth=3)
            ax.set_aspect("equal")
            path = "plot/kmeans++"
            if not os.path.exists(path):
                os.mkdir(path) 
            plt.savefig(path+"/"+str(len(os.listdir(path)))+".png")
            print(path + "/" +str(len(os.listdir(path)))+".png")
            plt.gcf().clear()
            plt.close() 
            
def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)