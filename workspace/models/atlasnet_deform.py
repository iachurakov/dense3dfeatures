import torch
import torch.nn as nn
import torch.nn.functional as F

"""
AtlasNet v2 with patch deformation and mlp adjustment
source:
https://arxiv.org/pdf/1908.04725.pdf
https://github.com/TheoDEPRELLE/AtlasNetV2/blob/522c147984c4659a15b8c1119709363af7e027e5/auxiliary/model.py
"""


class PatchDeformationMLP(nn.Module):
    """deformation of a 2D patch into a 3D surface"""

    def __init__(self, patch_dim_in, patch_dim_out):
        super().__init__()
        layer_size = 128
        self.conv1 = torch.nn.Conv1d(patch_dim_in, layer_size, 1)
        self.conv2 = torch.nn.Conv1d(layer_size, layer_size, 1)
        self.conv3 = torch.nn.Conv1d(layer_size, patch_dim_out, 1)
        self.bn1 = torch.nn.BatchNorm1d(layer_size)
        self.bn2 = torch.nn.BatchNorm1d(layer_size)
        self.th = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.th(self.conv3(x))
        return x


class MLPAdj(torch.nn.Module):
    """
    AtlasNet decoder
    """
    def __init__(self, latent_dim):

        super().__init__()
        self.conv1 = torch.nn.Conv1d(latent_dim, latent_dim, 1)
        self.conv2 = torch.nn.Conv1d(latent_dim, latent_dim // 2, 1)
        self.conv3 = torch.nn.Conv1d(latent_dim // 2, latent_dim // 4, 1)
        self.conv4 = torch.nn.Conv1d(latent_dim // 4, 3, 1)

        self.tanh = torch.nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(latent_dim)
        self.bn2 = torch.nn.BatchNorm1d(latent_dim // 2)
        self.bn3 = torch.nn.BatchNorm1d(latent_dim // 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.tanh(self.conv4(x))
        return x


class AtlasNet(torch.nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 latent_dim: int,
                 n_points: int,
                 n_patches: int,
                 patch_dim_out,
                 patch_dim_in: int = 2,
                 ):
        """
        :param latent_dim: dimension of latent representation
        :param n_points: the number of points in point cloud
        :param n_patches: the number of elementary structures
        :param patch_dim_in: dimension of a elementary structure2
        """
        super().__init__()
        self.encoder = encoder
        self.patch_dim_in = patch_dim_in
        self.n_points = n_points
        self.n_patches = n_patches
        self.grid = []
        self.head = nn.Sequential(
            nn.Linear(1024, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.patch_deformation = nn.ModuleList([PatchDeformationMLP(patch_dim_in, patch_dim_out) for _ in range(n_patches)])
        self.decoder = nn.ModuleList([MLPAdj(patch_dim_out + latent_dim) for _ in range(n_patches)])

    def forward(self, x: torch.FloatTensor):
        x = self.head(self.encoder.forward_features(x))
        patches = []
        parts = []

        for i in range(self.n_patches):
            # random planar patch
            # ==========================================================================
            rand_grid = torch.FloatTensor(x.size(0), self.patch_dim_in, self.n_points // self.n_patches).to(x.device)
            rand_grid.data.uniform_(0, 1)
            #rand_grid[:, 2:, :] = 0
            rand_grid = self.patch_deformation[i](rand_grid.contiguous())
            patches.append(rand_grid[0].transpose(1, 0))
            # ==========================================================================

            # cat with latent vector and decode
            # ==========================================================================
            part = x.unsqueeze(2).expand(-1, -1, rand_grid.size(2)).contiguous()
            part = torch.cat((rand_grid, part), 1).contiguous()
            parts.append(self.decoder[i](part))
            # ==========================================================================

        return torch.cat(parts, 2).transpose(2, 1).contiguous(), patches, parts
