import faiss
import torch
import numpy as np
from tqdm import tqdm, trange


class BatchKmeans:
    def __init__(self, n_clusters, device, n_init_batches):
        self.n_clusters = n_clusters
        self.index = None
        self.centroids = None
        self.n_init_batches = n_init_batches
        self.counts = np.zeros(n_clusters)
        self.device = device

    def init_centroids(self, features):
        clt = faiss.Kmeans(features.shape[1], self.n_clusters, niter=20, nredo=3)
        clt.train(features)
        self.index = clt.index
        self.centroids = clt.centroids

    def update(self, features):
        dists, assignment = self.index.search(features, 1)
        assignment = assignment.squeeze(1)
        clt_idx, counts = np.unique(assignment, return_counts=True)
        self.counts[clt_idx] += counts
        mask = (assignment[:, np.newaxis] == clt_idx)[:, :, np.newaxis]
        sum_update = np.where(mask, features[:, np.newaxis], 0).sum(axis=0)
        self.centroids[clt_idx] = self.centroids[clt_idx] + (sum_update - counts[:, np.newaxis] * self.centroids[clt_idx]) / self.counts[clt_idx, np.newaxis]
        self.index.reset()
        self.index.add(self.centroids)

        return dists.mean()

    @torch.no_grad()
    def train(self, model, loader):
        loader_iter = iter(loader)
        features_for_init = []

        for _ in trange(self.n_init_batches, leave=True, position=0):
            data = next(loader_iter)[0].to(self.device)
            features_for_init.append(model.forward_features(data).cpu())

        features_for_init = torch.cat(features_for_init, dim=0).numpy()
        print(features_for_init)
        self.init_centroids(features_for_init)

        cur_loss = 0
        cur_iter = 0

        progress_bar = tqdm(loader_iter, total=len(loader), leave=True, position=0)
        for features, _, _ in progress_bar:
            features = model.forward_features(features.to(self.device)).cpu().numpy()
            loss = self.update(features)
            cur_loss += loss
            cur_iter += 1
            progress_bar.set_postfix({
                'Loss': cur_loss / cur_iter
            })
