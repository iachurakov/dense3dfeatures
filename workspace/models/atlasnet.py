import torch
import torch.nn.functional as F


class AtlasNet(torch.nn.Module):
    """
    AtlasNet v2 with point translation and mlp adjustment
    source:
    https://arxiv.org/pdf/1908.04725.pdf
    https://github.com/TheoDEPRELLE/AtlasNetV2/blob/522c147984c4659a15b8c1119709363af7e027e5/auxiliary/model.py
    """

    def __init__(self,
                 encoder,
                 latent_dim: int,
                 n_points: int,
                 n_patches: int,
                 patch_dim: int = 2,
                 ):
        """
        :param latent_dim: dimension of latent representation
        :param n_points: the number of points in point cloud
        :param n_patches: the number of elementary structures
        :param patch_dim: dimension of a patch
        """
        super().__init__()
        self.patch_dim = patch_dim
        self.encoder = encoder
        self.grid = []
        self.decoder = torch.nn.ModuleList()
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(1024, 1024),
            # torch.nn.Dropout(p=0.4),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, latent_dim)
        )

        for patch_idx in range(n_patches):
            patch = torch.nn.Parameter(torch.FloatTensor(1, self.patch_dim, n_points // n_patches))
            patch.data.uniform_(0, 1)
            patch.data[:, 2:, :] = 0
            self.register_parameter("patch%d" % patch_idx, patch)
            self.grid.append(patch)
            self.decoder.append(MLPAdj(patch_dim + latent_dim))

    def forward(self, x: torch.FloatTensor):
        features = self.encoder.forward_features(x)
        z_max = features.max(dim=-1)[0]
        z_mean = features.mean(dim=-1)

        z = self.head(torch.cat([z_max, z_mean], dim=1).squeeze())
        patches = []
        decoded_parts = []

        for patch, decoder in zip(self.grid, self.decoder):
            rand_grid = patch.expand(z.size(0), -1, -1)  # (batch_size, patch_dim, pts_in_patch)
            patches.append(rand_grid[0].T)
            # concat latent code with patch, y has shape (batch_size, latent_dim, pts_in_patch)
            y = z.unsqueeze(2).expand(z.size(0), z.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            decoded_parts.append(decoder(y))

        return torch.cat(decoded_parts, 2).transpose(2, 1).contiguous(), patches, decoded_parts


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
