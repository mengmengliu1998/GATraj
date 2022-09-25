'''
model script
Author: Mengmeng Liu
Date: 2022/09/24
'''
from utils import *
from basemodel import *
from laplace_decoder import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class SoftTargetCrossEntropyLoss(nn.Module):

    def __init__(self, reduction: str = 'mean') -> None:
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        cross_entropy = torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1)
        if self.reduction == 'mean':
            return cross_entropy.mean()
        elif self.reduction == 'sum':
            return cross_entropy.sum()
        elif self.reduction == 'none':
            return cross_entropy
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

class LaplaceNLLLoss(nn.Module):

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        # print("scale",scale.shape,"loc",loc.shape)
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        # print("nll", nll.shape)
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

class GaussianNLLLoss(nn.Module):
    """https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
    """
    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(GaussianNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        # print("scale",scale.shape,"loc",loc.shape)
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = 0.5*(torch.log(scale**2) + torch.abs(target - loc)**2 / scale**2)
        # print("nll", nll.shape)
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))


class GATraj(nn.Module):
    def __init__(self, args):
        super(GATraj, self).__init__()
        self.args = args
        self.Temperal_Encoder=Temperal_Encoder(self.args)
        self.Laplacian_Decoder=Laplacian_Decoder(self.args)
        if self.args.SR:
            message_passing = []
            for i in range(self.args.pass_time):
                message_passing.append(Global_interaction(args))
            self.Global_interaction = nn.ModuleList(message_passing)
        if self.args.ifGaussian:
            self.reg_loss = GaussianNLLLoss(reduction='mean')
        else:
            self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')

    def forward(self, inputs, epoch, iftest=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_abs_gt, batch_norm_gt, nei_list_batch, nei_num_batch, batch_split = inputs # #[H, N, 2], [H, N, 2], [B, H, N, N], [N, H], [B, 2]
        self.batch_norm_gt = batch_norm_gt
        if self.args.input_offset:
            train_x = batch_norm_gt[1:self.args.obs_length, :, :] - batch_norm_gt[:self.args.obs_length-1, :, :] #[H, N, 2]
        elif self.args.input_mix:
            offset = batch_norm_gt[1:self.args.obs_length, :, :] - batch_norm_gt[:self.args.obs_length-1, :, :] #[H, N, 2]
            position = batch_norm_gt[:self.args.obs_length, :, :] #[H, N, 2]
            pad_offset = torch.zeros_like(position).to(device)
            pad_offset[1:, :, :] = offset
            train_x = torch.cat((position, pad_offset), dim=2)
        elif self.args.input_position:
            train_x = batch_norm_gt[:self.args.obs_length, :, :] #[H, N, 2]
        train_x = train_x.permute(1, 2, 0) #[N, 2, H]
        train_y = batch_norm_gt[self.args.obs_length:, :, :].permute(1, 2, 0) #[N, 2, H]
        self.pre_obs=batch_norm_gt[1:self.args.obs_length]
        self.x_encoded_dense, self.hidden_state_unsplited, cn=self.Temperal_Encoder.forward(train_x)  #[N, D], [N, D]
        self.hidden_state_global = torch.ones_like(self.hidden_state_unsplited, device=device)
        cn_global = torch.ones_like(cn, device=device)
        if self.args.SR:
            for b in range(len(nei_list_batch)):
                left, right = batch_split[b][0], batch_split[b][1]
                element_states = self.hidden_state_unsplited[left: right] #[N, D]
                cn_state = cn[left: right] #[N, D]
                if element_states.shape[0] != 1:
                    corr = batch_abs_gt[self.args.obs_length-1, left: right, :2].repeat(element_states.shape[0], 1, 1) #[N, N, D]
                    corr_index = corr.transpose(0,1)-corr  #[N, N, D]
                    nei_num = nei_num_batch[left:right, self.args.obs_length-1] #[N]
                    nei_index = torch.tensor(nei_list_batch[b][self.args.obs_length-1], device=device) #[N, N]
                    for i in range(self.args.pass_time):
                        element_states, cn_state = self.Global_interaction[i](corr_index, nei_index, nei_num, element_states, cn_state)
                    self.hidden_state_global[left: right] = element_states
                    cn_global[left: right] = cn_state
                else:
                    self.hidden_state_global[left: right] = element_states
                    cn_global[left: right] = cn_state
        else:
            self.hidden_state_global = self.hidden_state_unsplited
            cn_global = cn
        mdn_out = self.Laplacian_Decoder.forward(self.x_encoded_dense, self.hidden_state_global, cn_global, epoch)
        GATraj_loss, full_pre_tra = self.mdn_loss(train_y.permute(2, 0, 1), mdn_out, 1, iftest)  #[K, H, N, 2]
        return GATraj_loss, full_pre_tra

    def mdn_loss(self, y, y_prime, goal_gt, iftest):
        batch_size=y.shape[1]
        y = y.permute(1, 0, 2)  #[N, H, 2]
        # [F, N, H, 2], [F, N, H, 2], [N, F]
        out_mu, out_sigma, out_pi = y_prime 
        y_hat = torch.cat((out_mu, out_sigma), dim=-1)
        reg_loss, cls_loss = 0, 0
        full_pre_tra = []
        l2_norm = (torch.norm(out_mu - y, p=2, dim=-1) ).sum(dim=-1)   # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(batch_size)]
        reg_loss += self.reg_loss(y_hat_best, y)
        soft_target = F.softmax(-l2_norm / self.args.pred_length, dim=0).t().detach() # [N, F]
        cls_loss += self.cls_loss(out_pi, soft_target)
        loss = reg_loss + cls_loss
        #best ADE
        sample_k = out_mu[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
        full_pre_tra.append(torch.cat((self.pre_obs,sample_k), axis=0))
        # best FDE
        l2_norm_FDE = (torch.norm(out_mu[:,:,-1,:] - y[:,-1,:], p=2, dim=-1) )  # [F, N]
        best_mode = l2_norm_FDE.argmin(dim=0)
        sample_k = out_mu[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
        full_pre_tra.append(torch.cat((self.pre_obs,sample_k), axis=0))
        return loss, full_pre_tra


    
