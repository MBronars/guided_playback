"""PGND network, vendored unchanged from the PGND repository (MIT) for standalone inference.

Source: github.com/kywind/pgnd -- pgnd/material/network/{pointnet,nerf,utils}.py and pgnd/material/pgnd.py,
concatenated with their relative imports removed. Keep byte-compatible with training checkpoints:
the module and parameter names here are what a checkpoint state_dict refers to.
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x, x_mask):
        batchsize = x.size()[0]
        x_mask = x_mask.unsqueeze(1)  # B, N -> B, 1, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x * x_mask - (~x_mask) * 1e10
        x = torch.max(x, 2, keepdim=True)[0]
        assert x.min() > -1e10
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x, x_mask):
        batchsize = x.size()[0]
        x_mask = x_mask.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x * x_mask - (~x_mask) * 1e10
        x = torch.max(x, 2, keepdim=True)[0]
        assert x.min() > -1e10
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, feature_dim=1024, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, feature_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(feature_dim)
        self.feature_dim = feature_dim
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x, x_mask=None):
        B, D, N = x.size()
        if x_mask is None:
            x_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        if x_mask.dtype != torch.bool:
            x_mask = x_mask > 0
        trans = self.stn(x, x_mask)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x, x_mask)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if not self.global_feat:
            x = x.view(-1, self.feature_dim, N)
            return x, trans, trans_feat

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.feature_dim)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.feature_dim, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

import torch
import torch.nn as nn


class CondNeRFModel(torch.nn.Module):
    def __init__(
        self,
        xyz_dim=3,
        condition_dim=64,
        out_channel=3,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=4,
    ):
        super(CondNeRFModel, self).__init__()

        self.dim_xyz = xyz_dim
        self.dim_cond = condition_dim
        self.skip_connect_every = skip_connect_every

        self.layer1 = torch.nn.Linear(self.dim_xyz + self.dim_cond, hidden_size)
        
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if self.skip_connect_every is not None and i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers.append(torch.nn.Linear(self.dim_xyz + self.dim_cond + hidden_size, hidden_size))
            else:
                self.layers.append(torch.nn.Linear(hidden_size, hidden_size))

        self.layers_out = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.Linear(hidden_size // 2, out_channel),
        ])
        self.relu = nn.ReLU()

    def forward(self, xyz, cond):
        assert xyz.shape[1] == self.dim_xyz
        xyz = xyz[..., :self.dim_xyz]
        xyz = torch.cat([xyz, cond], 1)
        x = self.layer1(xyz)
        for i in range(len(self.layers)):
            if (
                self.skip_connect_every is not None
                and i % self.skip_connect_every == 0
                and i > 0
                and i != len(self.layers) - 1
            ):
                x = torch.cat((x, xyz), dim=-1)
            x = self.relu(self.layers[i](x))
        for i in range(len(self.layers_out) - 1):
            x = self.relu(self.layers_out[i](x))
        x = self.layers_out[-1](x)
        return x

import torch


def get_grid_locations(x, num_grids_list, dx):
    bsz = x.shape[0]
    x_grid = torch.stack(torch.meshgrid(
        torch.linspace(0, (num_grids_list[0] - 1) * dx, num_grids_list[0], device=x.device),
        torch.linspace(0, (num_grids_list[1] - 1) * dx, num_grids_list[1], device=x.device),
        torch.linspace(0, (num_grids_list[2] - 1) * dx, num_grids_list[2], device=x.device),
    ), dim=-1).reshape(-1, 3)
    grid_idxs = torch.stack(torch.meshgrid(
        torch.linspace(0, (num_grids_list[0] - 1), num_grids_list[0], device=x.device),
        torch.linspace(0, (num_grids_list[1] - 1), num_grids_list[1], device=x.device),
        torch.linspace(0, (num_grids_list[2] - 1), num_grids_list[2], device=x.device),
    ), dim=-1).reshape(-1, 3)

    grid_hits = torch.zeros(bsz, num_grids_list[0], num_grids_list[1], num_grids_list[2], device=x.device)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                grid_hits[:, \
                    ((x[:, :, 0] / dx - 0.5).int() + i).clamp(0, num_grids_list[0] - 1), \
                    ((x[:, :, 1] / dx - 0.5).int() + j).clamp(0, num_grids_list[1] - 1), \
                    ((x[:, :, 2] / dx - 0.5).int() + k).clamp(0, num_grids_list[2] - 1)] = 1
    grid_hits = grid_hits.sum(0) > 0
    
    x_grid = x_grid[grid_hits.reshape(-1)].reshape(1, -1, 3).repeat(bsz, 1, 1)
    grid_idxs = grid_idxs[grid_hits.reshape(-1)]
    return x_grid, grid_idxs


def fill_grid_locations(feat, grid_idxs, num_grids_list):  
    # feat: (bsz, num_active_grids, feature_dim)
    # grid_idxs: (num_active_grids, 3)
    bsz = feat.shape[0]
    feat_filled = torch.zeros(bsz, num_grids_list[0], num_grids_list[1], num_grids_list[2], 3, device=feat.device)
    grid_idxs_1d = grid_idxs[:, 0] * num_grids_list[1] * num_grids_list[2] + grid_idxs[:, 1] * num_grids_list[2] + grid_idxs[:, 2]
    feat_filled = feat_filled.reshape(bsz, -1, 3)
    feat_filled[:, grid_idxs_1d.long()] = feat.clone()
    feat_filled = feat_filled.reshape(bsz, num_grids_list[0], num_grids_list[1], num_grids_list[2], 3)
    return feat_filled
from typing import Optional

from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor






class PGNDModel(nn.Module):

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.feature_dim = 64
        self.radius = cfg.model.material.radius
        self.n_history = cfg.sim.n_history
        self.num_grids_list = cfg.sim.num_grids[:3]
        self.dx = cfg.sim.num_grids[3]
        self.inv_dx = 1 / self.dx
        self.requires_grad = cfg.model.material.requires_grad
        self.pe_num_func = int(np.log2(self.inv_dx)) + cfg.model.material.pe_num_func_res
        self.pe_dim = 3 + self.pe_num_func * 6
        self.output_scale = cfg.model.material.output_scale
        self.input_scale = cfg.model.material.input_scale
        self.absolute_y = cfg.model.material.absolute_y

        self.encoder = PointNetEncoder(
            global_feat=(cfg.model.material.radius <= 0), 
            feature_transform=False, 
            feature_dim=self.feature_dim, 
            channel=6 * (1 + self.n_history),
        )
        self.decoder = CondNeRFModel(
            xyz_dim=self.pe_dim,
            condition_dim=self.feature_dim,
            out_channel=3,
            num_layers=2,
            hidden_size=64,
            skip_connect_every=4,
        )

    def positional_encoding(self, tensor):
        num_encoding_functions = self.pe_num_func
        if num_encoding_functions == 0:
            assert include_input
            return tensor

        encoding = [tensor]  # include the input itself
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

        for freq in frequency_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)

    def forward(self, x: Tensor, v: Tensor, x_his: Tensor, v_his: Tensor, enabled: Tensor) -> Tensor:
        # x: (bsz, num_particles, 3)
        # v: (bsz, num_particles, 3)
        bsz = x.shape[0]
        num_particles = x.shape[1]
        v = v * self.input_scale
        v_his = v_his * self.input_scale

        x_his = x_his.reshape(bsz, num_particles, self.n_history, 3)
        v_his = v_his.reshape(bsz, num_particles, self.n_history, 3)
        x_his = x_his.detach()
        v_his = v_his.detach()
        
        x_grid, grid_idxs = get_grid_locations(x, self.num_grids_list, self.dx)
        x_grid = x_grid.detach()
        grid_idxs = grid_idxs.detach()

        # centering
        x_center = x.mean(1, keepdim=True)
        if self.absolute_y:
            x_center[:, :, 1] = 0  # only centering x and z
        x = x - x_center
        x_his = x_his - x_center[:, :, None]
        if self.training:
            # random azimuth
            theta = torch.rand(bsz, 1, device=x.device) * 2 * np.pi
            rot = torch.stack([
                torch.cos(theta), torch.zeros_like(theta), torch.sin(theta),
                torch.zeros_like(theta), torch.ones_like(theta), torch.zeros_like(theta),
                -torch.sin(theta), torch.zeros_like(theta), torch.cos(theta),
            ], dim=-1).reshape(bsz, 3, 3)
            inv_rot = rot.transpose(1, 2)
        else:
            rot = torch.eye(3, dtype=x.dtype, device=x.device).unsqueeze(0).repeat(bsz, 1, 1)
            inv_rot = rot.transpose(1, 2)
        x = torch.einsum('bij,bjk->bik', x, rot)
        x_his = torch.einsum('bcij,bjk->bcik', x_his, rot)  # (bsz, num_particles, n_history, 3)
        v = torch.einsum('bij,bjk->bik', v, rot)
        v_his = torch.einsum('bcij,bjk->bcik', v_his, rot)  # (bsz, num_particles, n_history, 3)

        x_grid = x_grid - x_center
        x_grid = x_grid @ rot
        x_his = x_his.reshape(bsz, num_particles, self.n_history * 3)
        v_his = v_his.reshape(bsz, num_particles, self.n_history * 3)

        feat = torch.cat([x, v, x_his, v_his], dim=-1)  # (bsz, num_particles, 6 * (1 + n_history))
        feat = feat.permute(0, 2, 1)  # (bsz, 6 * (1 + n_history), num_particles)
        feat, trans, trans_feat = self.encoder(feat, enabled)  # feat: (bsz, feature_dim, num_particles)

        if self.radius > 0:
            # aggregate neighborhood
            # x.shape: (bsz, num_particles, 3)
            # x_grid.shape: (bsz, num_grids_total, 3)
            dist_pt_grid = torch.cdist(x_grid, x, p=2)
            mask = dist_pt_grid < self.radius  # (bsz, num_grids_total, num_particles)
            mask_normed = mask / (mask.sum(dim=-1, keepdim=True) + 1e-5)  # for each grid, normalize the weights
            mask_normed = mask_normed.detach()
            feat = mask_normed @ feat.permute(0, 2, 1)  # (bsz, num_grids_total, feature_dim)
        else:
            # global max pooling
            feat = feat[:, None, :].repeat(1, x_grid.shape[1], 1)  # (bsz, num_grids_total, feature_dim)

        # positional encoding and decoder
        feat = feat.reshape(-1, self.feature_dim)
        x_grid = x_grid.reshape(-1, 3)
        x_grid = self.positional_encoding(x_grid)
        feat = self.decoder(x_grid, feat)
        feat = feat * self.output_scale
        feat = feat.reshape(bsz, -1, feat.shape[-1])
        feat = torch.bmm(feat, inv_rot)

        # expand to full grid
        feat = fill_grid_locations(feat, grid_idxs, self.num_grids_list)
        return feat
