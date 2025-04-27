#! /usr/bin/env python
#! coding:utf-8

from utils import poses_motion
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from torchdiffeq import odeint

class AdaptiveTimeODE(nn.Module):
    def __init__(self, input_dim, filters):
        super(AdaptiveTimeODE, self).__init__()
        self.input_dim = input_dim
        self.filters = filters
        
        # Time step prediction network
        self.time_net = nn.Sequential(
            nn.Linear(input_dim, filters),
            nn.ReLU(),
            nn.Linear(filters, 1),
            nn.Sigmoid()
        )
        
        # ODE function
        self.ode_func = nn.Sequential(
            nn.Linear(input_dim, filters),
            nn.SiLU(),
            nn.Linear(filters, input_dim)
        )
        self.norm = nn.LayerNorm(input_dim)
        self.time_weight = nn.Linear(1, input_dim)

    def forward(self, t, x):
        x = self.ode_func(x)
        x = self.norm(x)
        t_tensor = torch.ones(1, device=x.device) * t
        time_factor = torch.sigmoid(self.time_weight(t_tensor))
        x = x * time_factor.view(1, 1, -1)
        return x

    def solve_ode(self, x):
        # Get input dimensions
        B, T, C = x.shape
        
        # Process each time step independently
        time_steps = []
        for i in range(T):
            # Process each sample in batch
            x_t = x[:, i, :]  # shape: (B, C)
            time_step = self.time_net(x_t)  # shape: (B, 1)
            time_steps.append(time_step)
        
        # Stack time steps
        time_steps = torch.stack(time_steps, dim=1)  # shape: (B, T, 1)
        time_steps = time_steps.squeeze(-1)  # shape: (B, T)
        
        # Create one-dimensional time span
        t_span = torch.linspace(0, 1, T, device=x.device)  # shape: (T,)
        
        # Solve ODE for each sample in batch
        x_ode = []
        for i in range(B):
            x_i = x[i:i+1]  # shape: (1, T, C)
            x_i_ode = odeint(self, x_i, t_span, method='rk4')
            x_ode.append(x_i_ode[-1])
        
        # Stack results
        x = torch.stack(x_ode, dim=0)  # shape: (B, T, C)
        return x

class MultiScaleFusion(nn.Module):
    def __init__(self, filters):
        super(MultiScaleFusion, self).__init__()
        self.filters = filters
        
        # Multi-scale convolutions
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(filters, filters, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(filters),
                nn.SiLU()
            ) for k in [3, 5, 7]
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(filters * 3, filters),
            nn.BatchNorm1d(filters),
            nn.SiLU()
        )
        
    def forward(self, x):
        # Print tensor shape for debugging
        print("Input shape:", x.shape)
        
        # Reshape input to (B, T, C)
        B, T, C, D = x.shape
        x = x.reshape(B, T, C * D)
        
        # Extract multi-scale features
        features = []
        for conv in self.convs:
            # Reshape for conv1d: (B, T, C*D) -> (B, C*D, T)
            x_conv = x.permute(0, 2, 1)
            feat = conv(x_conv)
            # Reshape back: (B, C*D, T) -> (B, T, C*D)
            feat = feat.permute(0, 2, 1)
            features.append(feat)
        
        # Fuse features
        x = torch.cat(features, dim=-1)  # shape: (B, T, C*D*3)
        x = x.reshape(B * T, -1)  # shape: (B*T, C*D*3)
        x = self.fusion(x)  # shape: (B*T, filters)
        x = x.reshape(B, T, -1)  # shape: (B, T, filters)
        return x

class SpatioTemporalAttention(nn.Module):
    def __init__(self, filters):
        super(SpatioTemporalAttention, self).__init__()
        self.filters = filters
        
        # Spatial attention
        self.spatial_attn = nn.Sequential(
            nn.Conv1d(filters, filters//8, 1),
            nn.ReLU(),
            nn.Conv1d(filters//8, filters, 1),
            nn.Sigmoid()
        )
        
        # Temporal attention
        self.temporal_attn = nn.Sequential(
            nn.Linear(filters, filters//8),
            nn.ReLU(),
            nn.Linear(filters//8, filters),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Spatial attention
        spatial_weights = self.spatial_attn(x.permute(0, 2, 1))
        x = x * spatial_weights.permute(0, 2, 1)
        
        # Temporal attention
        temporal_weights = self.temporal_attn(x)
        x = x * temporal_weights
        
        return x

class EnhancedDDNet_ODE(nn.Module):
    def __init__(self, frame_l, joint_n, joint_d, feat_d, filters, class_num):
        super(EnhancedDDNet_ODE, self).__init__()
        
        # JCD part
        self.jcd_ode = AdaptiveTimeODE(feat_d, filters)
        self.jcd_fusion = MultiScaleFusion(filters)
        self.jcd_attention = SpatioTemporalAttention(filters)
        
        # Motion parts
        self.slow_ode = AdaptiveTimeODE(joint_n * joint_d, filters)
        self.slow_fusion = MultiScaleFusion(filters)
        self.slow_attention = SpatioTemporalAttention(filters)
        
        self.fast_ode = AdaptiveTimeODE(joint_n * joint_d, filters)
        self.fast_fusion = MultiScaleFusion(filters)
        self.fast_attention = SpatioTemporalAttention(filters)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(3 * filters, filters),
            nn.BatchNorm1d(filters),
            nn.SiLU()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(filters, filters//2),
            nn.BatchNorm1d(filters//2),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(filters//2, class_num)
        )
        
    def forward(self, M, P=None):
        # JCD processing
        x_jcd = self.jcd_ode.solve_ode(M)
        x_jcd = self.jcd_fusion(x_jcd)
        x_jcd = self.jcd_attention(x_jcd)
        
        # Motion processing
        diff_slow, diff_fast = poses_motion(P)
        
        x_slow = self.slow_ode.solve_ode(diff_slow)
        x_slow = self.slow_fusion(x_slow)
        x_slow = self.slow_attention(x_slow)
        
        x_fast = self.fast_ode.solve_ode(diff_fast)
        x_fast = self.fast_fusion(x_fast)
        x_fast = self.fast_attention(x_fast)
        
        # Feature fusion
        x = torch.cat([x_jcd, x_slow, x_fast], dim=-1)
        x = self.fusion(x)
        
        # Classification
        x = torch.max(x, dim=1).values
        x = self.classifier(x)
        
        return x 