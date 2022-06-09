import faiss
import torch
from tqdm import tqdm


class BatchKmeans:
    def __init__(self, n_clusters, n_init_batches, device):
        self.n_clusters = n_clusters
        self.index = None
        self.centroids = None
        self.device = device
        self.n_init_batches = n_init_batches
        self.counts = torch.zeros(n_clusters).to(device)
        self._zero = torch.FloatTensor([0]).to(device)

    def init_centroids(self, features):
        clt = faiss.Kmeans(features.size(1), self.n_clusters, gpu=self.device, niter=20, nredo=3)
        clt.fit(features)
        self.index = clt.index
        self.centroids = clt.centroids

    def update(self, features):
        dists, assignment = self.index.search(features, 1)
        clt_idx, counts = torch.unique(assignment, return_counts=True)
        self.counts[clt_idx] += counts
        mask = (assignment.unsqueeze(1) == clt_idx).to(self.device).unsqueeze(2)
        sum_update = torch.where(mask, features.unsqueeze(1), self._zero).sum(dim=0)
        self.centroids[clt_idx] += (sum_update - self.centroids[clt_idx] * counts) / self.counts[clt_idx]
        self.index.reset()
        self.index.add(self.centroids)

        return dists.mean()

    @torch.no_grad()
    def train(self, model, loader):
        loader_iter = iter(loader)
        features_for_init = []

        for _ in range(self.n_init_batches):
            data = next(loader_iter)[0].to(self.device)
            features_for_init.append(model.forward_features(data))

        features_for_init = torch.cat(features_for_init, dim=0)
        self.init_centroids(features_for_init)

        cur_loss = 0
        cur_iter = 0

        progress_bar = tqdm(loader_iter, total=len(loader), leave=True, position=0)
        for features, _, _ in progress_bar:
            features = model.forward_features(features.to(self.device))
            loss = self.update(features)
            cur_iter += loss
            cur_iter += 1
            progress_bar.set_postfix({
                'Loss': cur_loss / cur_iter
            })
