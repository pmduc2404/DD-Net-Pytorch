#! /usr/bin/env python
#! coding:utf-8

from utils import poses_motion
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from torchdiffeq import odeint
from .ulsam import ULSAM1D, ULSAM1Dv2
from linformer import LinformerSelfAttention


# class simam_module(nn.Module):
#     def __init__(self, e_lambda=1e-4):
#         super(simam_module, self).__init__()
#         self.activation = nn.Sigmoid()
#         self.e_lambda = e_lambda

#     def forward(self, x):
#         b, c, h, w = x.size()
#         n = w * h - 1
#         x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
#         y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
#         return x * self.activation(y)

class c1D(nn.Module):
    # input (B,C,D) //batch,channels,dims
    # output = (B,C,filters)
    def __init__(self, input_channels, input_dims, filters, kernel):
        super(c1D, self).__init__()
        self.cut_last_element = (kernel % 2 == 0)
        self.padding = math.ceil((kernel - 1)/2)
        self.conv1 = nn.Conv1d(input_dims, filters,
                               kernel, bias=False, padding=self.padding)
        self.bn = nn.BatchNorm1d(num_features=input_channels)

    def forward(self, x):
        # x (B,D,C)
        x = x.permute(0, 2, 1)
        # output (B,filters,C)
        if(self.cut_last_element):
            output = self.conv1(x)[:, :, :-1]
        else:
            output = self.conv1(x)
        # output = (B,C,filters)
        output = output.permute(0, 2, 1)
        output = self.bn(output)
        output = F.leaky_relu(output, 0.2, True)
        return output
    
# class c1D(nn.Module):
#     def __init__(self, input_channels, input_dims, filters, kernel):
#         super(c1D, self).__init__()
#         self.cut_last_element = (kernel % 2 == 0)
#         self.padding = math.ceil((kernel - 1) / 2)
#         self.depthwise = nn.Conv1d(input_dims, input_dims, kernel_size=kernel, 
#                                    groups=input_dims, padding=self.padding, bias=False)
#         self.pointwise = nn.Conv1d(input_dims, filters, kernel_size=1, bias=False)
#         self.bn = nn.BatchNorm1d(num_features=input_channels)

#     def forward(self, x):
#         x = x.permute(0, 2, 1)  # (B, D, C)
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         if self.cut_last_element:
#             x = x[:, :, :-1]
#         x = x.permute(0, 2, 1)  # (B, C, filters)
#         x = self.bn(x)
#         return F.leaky_relu(x, 0.2, True)


class block(nn.Module):
    def __init__(self, input_channels, input_dims, filters, kernel):
        super(block, self).__init__()
        self.c1D1 = c1D(input_channels, input_dims, filters, kernel)
        self.c1D2 = c1D(input_channels, filters, filters, kernel)

    def forward(self, x):
        output = self.c1D1(x)
        output = self.c1D2(output)
        return output


class d1D(nn.Module):
    def __init__(self, input_dims, filters):
        super(d1D, self).__init__()
        self.linear = nn.Linear(input_dims, filters)
        self.bn = nn.BatchNorm1d(num_features=filters)

    def forward(self, x):
        output = self.linear(x)
        output = self.bn(output)
        output = F.leaky_relu(output, 0.2)
        return output


class spatialDropout1D(nn.Module):
    def __init__(self, p):
        super(spatialDropout1D, self).__init__()
        self.dropout = nn.Dropout1d(p)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x

# class ODEFunc(nn.Module):
#     def __init__(self, filters):
#         super(ODEFunc, self).__init__()
#         self.filters = filters
#         self.net = nn.Sequential(
#             nn.Linear(filters, filters),
#             nn.SiLU(),
#             nn.Linear(filters, filters)
#         )
#         self.norm = nn.LayerNorm(filters)
#         self.time_weight = nn.Linear(1, filters)

#     def forward(self, t, x):
#         x = self.net(x)
#         x = self.norm(x)
#         t_tensor = torch.ones(1, device=x.device) * t
#         time_factor = torch.sigmoid(self.time_weight(t_tensor))
#         x = x * time_factor.view(1, 1, -1)
#         return x

class ODEFunc(nn.Module):
    def __init__(self, filters, dropout=0.1):
        super(ODEFunc, self).__init__()
        self.filters = filters

        self.net = nn.Sequential(
            nn.Linear(filters, filters),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(filters, filters)
        )

        self.norm = nn.LayerNorm(filters)
        self.time_weight = nn.Sequential(
            nn.Linear(1, filters),
            nn.Sigmoid()
        )

    def forward(self, t, x):
        residual = x
        x = self.net(x)
        x = self.norm(x)

        t_tensor = torch.full((x.size(0), 1), t, device=x.device)  # [B, 1]
        time_factor = self.time_weight(t_tensor)  # [B, filters]

        x = x * time_factor.unsqueeze(1)  # broadcast: [B, seq_len, filters]
        return x + residual  # residual connection

    
class SlowODE(nn.Module):
    def __init__(self, frame_l, joint_n, joint_d, filters):
        super(SlowODE, self).__init__()
        self.frame_l = frame_l
        self.joint_n = joint_n
        self.joint_d = joint_d
        self.filters = filters

        self.init_conv = nn.Sequential(
            c1D(frame_l, joint_n * joint_d, filters, 1),
            spatialDropout1D(0.2)  
        )
        self.ode_func = ODEFunc(filters)
        self.ode_dropout = spatialDropout1D(0.1)
        self.pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1)
        )

    def forward(self, x):
        B, frame_l, feat_dim = x.shape
        if frame_l != self.frame_l or feat_dim != self.joint_n * self.joint_d:
            raise ValueError(f"Input shape mismatch: expected ({B}, {self.frame_l}, {self.joint_n * self.joint_d}), got {x.shape}")
        x = self.init_conv(x)
        t_span = torch.linspace(0, 1, 10, device=x.device)  
        x = odeint(self.ode_func, x, t_span, method='euler')[-1]  
        x = self.ode_dropout(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        return x

class FastODE(nn.Module):
    def __init__(self, frame_l, joint_n, joint_d, filters):
        super(FastODE, self).__init__()
        self.frame_l = frame_l // 2  # fast branch uses frame_l//2
        self.joint_n = joint_n
        self.joint_d = joint_d
        self.filters = filters

        self.init_conv = nn.Sequential(
            c1D(self.frame_l, joint_n * joint_d, filters, 1),
            spatialDropout1D(0.1)
        )
        self.ode_func = ODEFunc(filters)
        self.ode_dropout = spatialDropout1D(0.1)
        self.pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=1),  # Keep temporal dimension
            spatialDropout1D(0.1)
        )

    def forward(self, x):
        # x: (B, frame_l//2, joint_n * joint_d)
        B, frame_l, feat_dim = x.shape
        if frame_l != self.frame_l or feat_dim != self.joint_n * self.joint_d:
            raise ValueError(f"Input shape mismatch: expected ({B}, {self.frame_l}, {self.joint_n * self.joint_d}), got {x.shape}")
        x = self.init_conv(x)  # (B, frame_l//2, filters)
        t_span = torch.linspace(0, 1, 10, device=x.device)
        x = odeint(self.ode_func, x, t_span, method='euler')[-1]  # (B, frame_l//2, filters)
        x = self.ode_dropout(x)
        x = x.permute(0, 2, 1)  # (B, filters, frame_l//2)
        x = self.pool(x)  # (B, filters, frame_l//2)
        x = x.permute(0, 2, 1)  # (B, frame_l//2, filters)
        return x

class JCDODE(nn.Module):
    def __init__(self, frame_l, feat_d, filters):
        super(JCDODE, self).__init__()
        self.frame_l = frame_l
        self.feat_d = feat_d
        self.filters = filters

        self.init_conv = nn.Sequential(
            c1D(frame_l, feat_d, filters, 1),
            spatialDropout1D(0.1)
        )
        self.ode_func = ODEFunc(filters)
        self.ode_dropout = spatialDropout1D(0.1)
        self.pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            spatialDropout1D(0.1)
        )

    def forward(self, x):
        B, frame_l, feat_dim = x.shape
        if frame_l != self.frame_l or feat_dim != self.feat_d:
            raise ValueError(f"Input shape mismatch: expected ({B}, {self.frame_l}, {self.feat_d}), got {x.shape}")
        x = self.init_conv(x)
        t_span = torch.linspace(0, 1, 10, device=x.device)
        x = odeint(self.ode_func, x, t_span, method='euler')[-1]
        x = self.ode_dropout(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        return x
    
# class simam_module(nn.Module):
#     def __init__(self, e_lambda=1e-4):
#         super(simam_module, self).__init__()
#         self.activation = nn.Sigmoid()
#         self.e_lambda = e_lambda

#     def forward(self, x):
#         b, c, h, w = x.size()
#         n = w * h - 1
#         x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
#         y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
#         return x * self.activation(y)

# class EleAttG_GRU_SimAM(nn.Module):
#     def __init__(self, embedding_dim, frames, n_hidden=128, output_dim=None):
#         super(EleAttG_GRU_SimAM, self).__init__()

#         assert output_dim is not None

#         self.embedding_dim = embedding_dim
#         self.frames = frames
#         self.n_hidden = n_hidden
#         self.output_dim = output_dim

#         self.attention = simam_module(e_lambda=1e-4)
#         self.gru = nn.GRU(self.embedding_dim, self.n_hidden, batch_first=True)
#         self.fc = nn.Sequential(
#             nn.Linear(self.n_hidden, self.n_hidden),
#             nn.ReLU(),
#             nn.Linear(self.n_hidden, self.output_dim)
#         )

#     def forward(self, X):
#         """
#         X: [batch_size, frames, embedding_dim]
#         """
#         # Reshape for simam_module: [B, embedding_dim, frames, 1]
#         X = X.permute(0, 2, 1).unsqueeze(-1)
#         X = self.attention(X)
#         X = X.squeeze(-1).permute(0, 2, 1)  # [B, frames, embedding_dim]

#         # GRU processing
#         h0 = torch.zeros(1, X.size(0), self.n_hidden).to(X.device)
#         out, _ = self.gru(X, h0)  # [B, frames, n_hidden]

#         # Take the output for all frames and project to embedding_dim
#         out = self.fc(out)  # [B, frames, output_dim]
#         return out

class DDNet_Original(nn.Module):
    def __init__(self, frame_l, joint_n, joint_d, feat_d, filters, class_num):
        super(DDNet_Original, self).__init__()
        # JCD part
        self.jcd_ode = JCDODE(frame_l, feat_d, filters)

        # diff_slow part
        self.slow_ode = SlowODE(frame_l, joint_n, joint_d, filters)

        # fast_part
        self.fast_ode = FastODE(frame_l, joint_n, joint_d, filters)

        # after cat
        self.block1 = block(frame_l//2, 3 * filters, 2 * filters, 3)
        self.block_pool1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2), spatialDropout1D(0.1))

        self.block2 = block(frame_l//4, 2 * filters, 4 * filters, 3)
        self.block_pool2 = nn.Sequential(nn.MaxPool1d(
            kernel_size=2), spatialDropout1D(0.1))

        self.block3 = nn.Sequential(
            block(frame_l//8, 4 * filters, 8 * filters, 3), spatialDropout1D(0.1))

        self.linear1 = nn.Sequential(
            d1D(8 * filters, 128),
            nn.Dropout(0.5)
        )
        self.linear2 = nn.Sequential(
            d1D(128, 128),
            nn.Dropout(0.5)
        )

        self.linear3 = nn.Linear(128, class_num)

        self.attn = ULSAM1Dv2(3 * filters) 
        # self.attn = LinformerSelfAttention(
        #     dim = 3 * filters,
        #     seq_len = frame_l // 2,
        #     heads = 4,
        #     k = 256,
        #     one_kv_head = True,
        #     share_kv = True
        # )

        # self.gru_attn = EleAttG_GRU_SimAM(
        #     embedding_dim=128,
        #     frames=frame_l // 2,
        #     n_hidden=128,
        #     output_dim=class_num
        # )

    def forward(self, M, P=None):

        x = self.jcd_ode(M)

        diff_slow, diff_fast = poses_motion(P)
 
        x_d_slow = self.slow_ode(diff_slow)
        x_d_fast = self.fast_ode(diff_fast)

        # x,x_d_fast,x_d_slow shape: (B,framel//2,filters)

        x = torch.cat((x, x_d_slow, x_d_fast), dim=2)
        x = self.attn(x)

        x = self.block1(x)
        x = x.permute(0, 2, 1)
        x = self.block_pool1(x)
        x = x.permute(0, 2, 1)

        x = self.block2(x)
        x = x.permute(0, 2, 1)
        x = self.block_pool2(x)
        x = x.permute(0, 2, 1)

        x = self.block3(x)
        # max pool over (B,C,D) C channels
        x = torch.max(x, dim=1).values

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)

        # x = x.unsqueeze(1)
        # x= self.gru_attn(x)
        
        return x
    