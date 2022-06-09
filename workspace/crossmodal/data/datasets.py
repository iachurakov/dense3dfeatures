from torch.utils.data import Dataset
from ..models.meshcnn.utils.mesh_prepare import from_scratch
from ..models.meshcnn.utils.mesh import Mesh
from enum import Enum
import numpy as np
import torch
import h5py

class Modality(Enum):
    MESHNET = 'meshnet'
    MESHCNN = 'meshcnn'
    POINT_CLOUD = 'point_cloud'

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

MESHCNN_DEFAULT_OPTS = AttrDict({
    'normalize': True,
    'num_aug': 1,
    'scale_verts': True,
    'slide_verts': 0.2,
    'flip_edges': 0.2,
    'is_train': True,
    'ninput_edges': 1000
})


def pad(input_arr, target_length, val=0, dim=1):
    shp = input_arr.shape
    npad = [(0, 0) for _ in range(len(shp))]
    npad[dim] = (0, target_length - shp[dim])
    return np.pad(input_arr, pad_width=npad, mode='constant', constant_values=val)


class CrossmodalDataset(Dataset):
    def __init__(self, data_path, modality, transform=None, meshcnn_opt=None, return_face_indexes=False):
        super().__init__()
        self.modality = modality
        self.transform = transform
        self.meshcnn_opt = meshcnn_opt
        self.return_face_indexes = return_face_indexes
        self.file = h5py.File(data_path, 'r')

        if self.modality is Modality.MESHCNN:
            self.get_mean_std()

    def get_mean_std(self):
        self.mean = 0
        self.std = 0
        for i in range(self.__len__()):
            m = from_scratch(
                (self.file['vertices'][i].reshape(-1, 3), self.file['faces'][i].reshape(-1, 3)),
                self.meshcnn_opt, False
            )
            features = m['features']
            self.mean = self.mean + features.mean(axis=1)
            self.std = self.std + features.std(axis=1)
    
        self.mean = self.mean / self.__len__()
        self.std = self.std / self.__len__()

    def __getitem__(self, index):
        item = None

        if self.modality is Modality.MESHNET:
            features = self.file['features'][index][:].reshape(-1, 15)
            neighbors = self.file['neighbors'][index][:].reshape(-1, 3)
            
            if self.transform is not None:
                features = self.transform(features)
                
                
            features = torch.from_numpy(features).float()
            neighbors = torch.from_numpy(neighbors).long()
        
            features = torch.permute(features, (1, 0))
            centers, corners, normals = features[:3], features[3:12], features[12:]
            corners = corners - np.concatenate([centers, centers, centers], 0)
            
            item = centers, corners, normals, neighbors
        
        elif self.modality is Modality.POINT_CLOUD:
            points = self.file['points'][index][:]
            
            
            if self.transform is not None:
                points = self.transform(points)
                
            points = torch.from_numpy(points).float()
            points = torch.permute(points, (1, 0))
            item = points

        elif self.modality is Modality.MESHCNN:
            mesh = Mesh(from_scratch(
                (self.file['vertices'][index].reshape(-1, 3), self.file['faces'][index].reshape(-1, 3)),
                self.meshcnn_opt, self.meshcnn_opt.is_train
            ), hold_history=True)
            meta = {'mesh': mesh}
            # get edge features
            edge_features = mesh.extract_features()
            edge_features = pad(edge_features, self.meshcnn_opt.ninput_edges)
            meta['edge_features'] = torch.from_numpy((edge_features - self.mean[..., None]) / self.std[..., None]).float()
            item = meta

        if self.return_face_indexes:
            return item, torch.from_numpy(self.file['face_index'][index][:]).long()
        return item

        
    def __len__(self):
        return self.file['points'].shape[0]
    
    
class DoubleDataset(CrossmodalDataset):
    def __init__(self, **multimodal_dataset_kwargs):
        super().__init__(**multimodal_dataset_kwargs)

    def __getitem__(self, idx):
        return super().__getitem__(idx), super().__getitem__(idx)

    def __len__(self):
        return super().__len__()

    
class DoubleModalityDataset(Dataset):
    def __init__(self, dset1, dset2):
        super().__init__()
        self.dset1 = dset1
        self.dset2 = dset2
        
    def __getitem__(self, idx):
        return *self.dset1.__getitem__(idx), *self.dset2.__getitem__(idx)
    
    def __len__(self):
        return self.dset1.__len__()