import torch
import numpy as np
from torch.utils.data import default_collate


def sample(x, num_points=1024):
    device = x.device
    B, C, N = x.shape
    centroids = torch.zeros(B, num_points, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(num_points):
        centroids[:, i] = farthest
        centroid = x[batch_indices, :, farthest].view(B, C, 1)
        dist = torch.sum((x - centroid) ** 2, 1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def move_to_device(data, device='cpu'):
    if isinstance(data, list):
        return [item.to(device) for item in data]
    else:
        return data.to(device)


def multicollate(data, *collators):
    batches = [[] for i in range(len(collators))]
    for item in data:
        for i in range(len(collators)):
            batches[i].append(item[i])

    result = []

    for i, collator in enumerate(collators):
        result.append(collator(batches[i]))
    
    return result
        
        
def collate_clouds(data, num_points=1024, device='cpu', face_indexes=False):
    if face_indexes:
        batch, face_indexes = move_to_device(default_collate(data), device)
    else:
        batch = move_to_device(default_collate(data), device)
    
    centroids_idx = sample(batch, num_points)
    
    batch = torch.gather(batch, 2, centroids_idx.unsqueeze(1).expand(-1, batch.size(1), -1))
    if face_indexes:
        face_indexes = torch.gather(face_indexes, 1, centroids_idx)
        return batch, face_indexes
    return batch

        
def collate_meshnet(data, device='cpu'):
    max_faces = 0
    centers = []
    corners = []
    normals = []
    neighbors = []
    for centers_, corners_, normals_, neighbors_ in data:
        max_faces = max(max_faces, neighbors_.shape[0])
    
    for centers_, corners_, normals_, neighbors_ in data:
        num_faces = neighbors_.shape[0]
        if num_faces < max_faces:
            fill_idx = np.random.choice(num_faces, max_faces - num_faces)
            centers.append(torch.concat([centers_, centers_[:, fill_idx]], dim=1))
            corners.append(torch.concat([corners_, corners_[:, fill_idx]], dim=1))
            normals.append(torch.concat([normals_, normals_[:, fill_idx]], dim=1))
            neighbors.append(torch.concat([neighbors_, neighbors_[fill_idx]]))
        else:
            centers.append(centers_)
            corners.append(corners_)
            normals.append(normals_)
            neighbors.append(neighbors_)
        
    centers = torch.stack(centers).to(device)
    corners = torch.stack(corners).to(device)
    normals = torch.stack(normals).to(device)
    neighbors = torch.stack(neighbors).to(device)
    
    return centers, corners, normals, neighbors


def collate_meshcnn(batch, device='cpu'):
    meshes = []
    edge_feat = []
    for model in batch:
        meshes.append(model['mesh'])
        edge_feat.append(model['edge_features'])

    meta = {'mesh': np.array(meshes),
            'edge_features': torch.stack(edge_feat).float().to(device)}
    return meta