import torch
import torch.nn as nn
from .layers import SpatialDescriptor, StructuralDescriptor, MeshConvolution
from .basemodel import BaseModel


class MeshNet(BaseModel):
    N_OUTPUT = 512
    def __init__(self,
                 num_kernel=64,
                 sigma=0.2,
                 aggregation_mode='Concat',
                 **basemodel_kwargs):
        super().__init__(n_output=self.N_OUTPUT, **basemodel_kwargs)
        self.spatial_descriptor = SpatialDescriptor()
        self.structural_descriptor = StructuralDescriptor(num_kernel, sigma)
        self.mesh_conv1 = MeshConvolution(aggregation_mode, 64, 131, 256, 256)
        self.mesh_conv2 = MeshConvolution(aggregation_mode, 256, 256, 512, 512)
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.concat_mlp = nn.Sequential(
            nn.Conv1d(1792, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.embedding_proj = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1)
        )

    def forward_features(self, data):
        centers, corners, normals, neighbor_index = data
        spatial_fea0 = self.spatial_descriptor(centers)
        structural_fea0 = self.structural_descriptor(corners, normals, neighbor_index)

        spatial_fea1, structural_fea1 = self.mesh_conv1(spatial_fea0, structural_fea0, neighbor_index)
        spatial_fea2, structural_fea2 = self.mesh_conv2(spatial_fea1, structural_fea1, neighbor_index)
        spatial_fea3 = self.fusion_mlp(torch.cat([spatial_fea2, structural_fea2], 1))

        features = self.concat_mlp(torch.cat([spatial_fea1, spatial_fea2, spatial_fea3], 1))
        # global_features = torch.max(features, dim=2)[0].unsqueeze(2).expand(*features.size())
        #global_features = self.global_embedding_proj(global_features)
        features = self.embedding_proj(features)

        return features
