import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_tensor_type(torch.cuda.FloatTensor)


class SubSpace(nn.Module):
    """
    Subspace class.

    ...

    Attributes
    ----------
    nin : int
        number of input feature volume.

    Methods
    -------
    __init__(nin)
        initialize method.
    forward(x)
        forward pass.

    """

    def __init__(self, nin: int) -> None:
        super(SubSpace, self).__init__()
        self.conv_dws = nn.Conv2d(
            nin, nin, kernel_size=1, stride=1, padding=0, groups=nin
        )
        self.bn_dws = nn.BatchNorm2d(nin, momentum=0.9)
        self.relu_dws = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv_point = nn.Conv2d(
            nin, 1, kernel_size=1, stride=1, padding=0, groups=1
        )
        self.bn_point = nn.BatchNorm2d(1, momentum=0.9)
        self.relu_point = nn.ReLU(inplace=False)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_dws(x)
        out = self.bn_dws(out)
        out = self.relu_dws(out)

        out = self.maxpool(out)

        out = self.conv_point(out)
        out = self.bn_point(out)
        out = self.relu_point(out)

        m, n, p, q = out.shape
        out = self.softmax(out.view(m, n, -1))
        out = out.view(m, n, p, q)

        out = out.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        out = torch.mul(out, x)

        out = out + x

        return out


class ULSAM(nn.Module):
    """
    Grouped Attention Block having multiple (num_splits) Subspaces.

    ...

    Attributes
    ----------
    nin : int
        number of input feature volume.

    nout : int
        number of output feature maps

    h : int
        height of a input feature map

    w : int
        width of a input feature map

    num_splits : int
        number of subspaces

    Methods
    -------
    __init__(nin)
        initialize method.
    forward(x)
        forward pass.

    """

    def __init__(self, nin: int, nout: int, h: int, w: int, num_splits: int) -> None:
        super(ULSAM, self).__init__()

        assert nin % num_splits == 0

        self.nin = nin
        self.nout = nout
        self.h = h
        self.w = w
        self.num_splits = num_splits

        self.subspaces = nn.ModuleList(
            [SubSpace(int(self.nin / self.num_splits)) for i in range(self.num_splits)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group_size = int(self.nin / self.num_splits)

        # split at batch dimension
        sub_feat = torch.chunk(x, self.num_splits, dim=1)

        out = []
        for idx, l in enumerate(self.subspaces):
            out.append(self.subspaces[idx](sub_feat[idx]))

        out = torch.cat(out, dim=1)

        return out

class ULSAM1D(nn.Module):
    def __init__(self, nin, num_splits=4):
        super(ULSAM1D, self).__init__()
        assert nin % num_splits == 0
        self.nin = nin
        self.num_splits = num_splits
        self.chunk_size = nin // num_splits

        self.fc1 = nn.ModuleList([nn.Linear(self.chunk_size, self.chunk_size // 4) for _ in range(num_splits)])
        self.fc2 = nn.ModuleList([nn.Linear(self.chunk_size // 4, self.chunk_size) for _ in range(num_splits)])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, T, D)
        x_chunks = torch.chunk(x, self.num_splits, dim=2)  # chia theo D
        out_chunks = []
        for i in range(self.num_splits):
            xi = x_chunks[i]  # (B, T, D')
            attn = self.fc1[i](xi)
            attn = self.fc2[i](attn)
            attn = self.sigmoid(attn)
            out_chunks.append(xi * attn)
        out = torch.cat(out_chunks, dim=2)  # (B, T, D)
        return out
    

class ULSAM1Dv2(nn.Module):
    def __init__(self, nin, num_splits=4, reduction=4, use_pool=True):
        super(ULSAM1Dv2, self).__init__()
        assert nin % num_splits == 0, "nin must be divisible by num_splits"
        
        self.nin = nin
        self.num_splits = num_splits
        self.chunk_size = nin // num_splits
        self.use_pool = use_pool

        self.fc1 = nn.ModuleList([
            nn.Linear(self.chunk_size, self.chunk_size // reduction)
            for _ in range(num_splits)
        ])
        self.fc2 = nn.ModuleList([
            nn.Linear(self.chunk_size // reduction, self.chunk_size)
            for _ in range(num_splits)
        ])

        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(nin)

        if self.use_pool:
            self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x: (B, T, D)
        residual = x
        if self.use_pool:
            x = x.permute(0, 2, 1)                  # (B, D, T)
            x = self.pool(x)                        # (B, D, T)
            x = x.permute(0, 2, 1)                  # (B, T, D)

        x_chunks = torch.chunk(x, self.num_splits, dim=2)
        out_chunks = []

        for i in range(self.num_splits):
            xi = x_chunks[i]  # (B, T, D')
            attn = self.fc1[i](xi)
            attn = F.relu(attn, inplace=True)
            attn = self.fc2[i](attn)
            attn = self.sigmoid(attn)
            out_chunks.append(xi * attn)

        out = torch.cat(out_chunks, dim=2)          # (B, T, D)
        out = self.norm(out + residual)             # Residual + LayerNorm
        return out
