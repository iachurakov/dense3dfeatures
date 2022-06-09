import torch.nn as nn
import torch.nn.functional as F
import torch
import faiss
import faiss.contrib.torch_utils
from faiss.contrib.exhaustive_search import knn
import numpy as np
from faiss import Kmeans


class OrtLayer(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.register_buffer('operator', torch.zeros(out_dim, out_dim))

    def step(self, x):
        identity = torch.eye(x.shape[1], device=x.device)
        gram = x.T @ x + 1e-7 * identity
        l = torch.linalg.cholesky(gram)
        self.operator = torch.linalg.inv(l).T * x.shape[0]**0.5

    def forward(self, x):
        return x @ self.operator


class SpectralNet(nn.Module):
    def __init__(self,
                 n_clusters,
                 n_neighbours,
                 in_dim,
                 hid_dim,
                 out_dim,
                 normalize=True):

        super().__init__()
        self.n_clusters = n_clusters
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_neighbours = n_neighbours
        self.normalize = normalize
        self.clt = None
        self.mapper = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Linear(hid_dim // 2, out_dim),
        )

        self.ort = OrtLayer(self.out_dim)
        self.gpu_res = faiss.StandardGpuResources()
        self.gpu_res.setTempMemory(0)

    def find_knn(self, x):
        bs = x.shape[0] // 2048
        dists = []
        index = []
        for batch_idx in range(bs):
            start = batch_idx * 2048
            end = (batch_idx + 1) * 2048
            idx = torch.arange(start)
            idx = torch.cat([idx, torch.arange(end, len(x))], dim=0)
            q = torch.narrow(x, 0, start, 2048)
            d, i = faiss.knn_gpu(self.gpu_res, q, x[idx], self.n_neighbours)
            i = idx[i.view(-1)].view(-1,  self.n_neighbours).to(x.device)
            dists.append(d)
            index.append(i)

        return torch.cat(dists, dim=0), torch.cat(index, dim=0)

    def compute_affinity_matrix(self, x):
        # dists: n_pts x k
        n_pts = x.shape[0]
        dists, nn_idx = self.find_knn(x)
        sigma = dists.mean(dim=1).unsqueeze(1)
        # sigma = torch.median(dists, dim=1)[0].unsqueeze(1)
        w = torch.exp(-dists**2 / (2 * sigma**2)).view(n_pts * self.n_neighbours).repeat(2)
        nn_idx = nn_idx.view(n_pts * self.n_neighbours)
        rows_idx = torch.arange(n_pts).unsqueeze(1).expand(-1, self.n_neighbours)
        rows_idx = rows_idx.contiguous().view(n_pts * self.n_neighbours).to(x.device)
        idx = torch.vstack([torch.cat([nn_idx, rows_idx], dim=0),
                            torch.cat([rows_idx, nn_idx], dim=0)])

        return torch.sparse_coo_tensor(idx, w, (n_pts, n_pts)) / 2, idx, nn_idx

    def forward(self, x1, x2, ort=True):

        if ort:
            self.eval()
            y = self.mapper(x1)

            if self.normalize:
                affinities, idx, nn_idx = self.compute_affinity_matrix(x1)
                d = torch.sparse.sum(affinities, dim=1).to_dense()
                y = y / d.unsqueeze(1)

            self.ort.step(y)

        if ort:
            self.train()

        affinities, idx, nn_idx = self.compute_affinity_matrix(x2)
        y = self.ort(self.mapper(x2))

        if self.normalize:
            d = torch.sparse.sum(affinities, dim=1).to_dense()
            y = y / d.unsqueeze(1)

        n_pts = x1.shape[0]
        dists = y.unsqueeze(1) - y[nn_idx].view(-1, self.n_neighbours, self.out_dim)
        dists = torch.norm(dists, dim=-1)**2
        dists = dists.view(n_pts * self.n_neighbours).repeat(2)
        dists = torch.sparse_coo_tensor(idx, dists, (n_pts, n_pts))

        return torch.sparse.sum(torch.sparse.mm(affinities, dists)) / n_pts

    def get_embeddings(self, x):
        return self.ort(self.mapper(x))
    
    @torch.no_grad()
    def train_kmeans(self, loader, device):
        self.eval()
        embeddings = []

        for data in loader:
            embeddings.append(self.get_embeddings(data[0].to(device)).cpu())

        embeddings = torch.cat(embeddings, dim=0).numpy()

        '''q = embeddings.dot(embeddings.T)
        centroids_idxs = list(np.unravel_index(q.argmin(), q.shape))

        for _ in range(2, self.n_clusters):
            new_idx = np.argmin(np.max(q[centroids_idxs, :], axis=0))
            centroids_idxs.append(new_idx)'''

        self.clt = Kmeans(self.out_dim, self.n_clusters, spherical=False)
        self.clt.train(embeddings)
        # self.clt = KMeans(n_clusters=self.n_clusters).fit(embeddings)

    @torch.no_grad()
    def assign(self, x, device):
        self.eval()
        embedding = self.get_embeddings(x.to(device)).cpu().numpy()
        return self.clt.index.search(embedding, 1)[1].flatten()
        # return self.clt.predict(embedding)
