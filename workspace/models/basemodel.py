import torch
from abc import ABC, abstractmethod


class BaseModel(torch.nn.Module, ABC):
    def __init__(self,
                 n_patches: int,
                 flatten_embeddings: bool = False,
                 normalize: bool = False,
                 mlp: bool = False,
                 n_output: int = -1):
        """
        :param n_patches: max number of patches
        :param flatten_embeddings: if True then model will return tensor with only present embedding,
        else also fake embeddings will be returned with patch statistics
        :param normalize: whether normalize patches' embeddings or not
        :param mlp: use mlp to project patches' embeddings
        :param n_output: dim of pointwise features, required only if mlp is True
        """
        super().__init__()
        self.n_patches = n_patches
        self.flatten_embeddings = flatten_embeddings
        self.n_output = n_output
        self.mlp = None
        self.normalize = normalize

        if mlp and n_output > 0:
            self.mlp = torch.nn.Sequential(
                torch.nn.Conv1d(n_output, 1024, kernel_size=1),
                torch.nn.BatchNorm1d(1024),
                torch.nn.ReLU(),
                torch.nn.Conv1d(1024, 128, kernel_size=1, bias=False)
            )

    def group_by(self, features, labels):
        """
        :param features: features for all face/point/pixel/voxel, batch_size x feature_dim x n_features
        :param labels: patches/segmentation labels, batch_size x n_features
        """
        # batch_size x n_patches x n_features
        mask = labels.unsqueeze(1) == torch.arange(self.n_patches).to(labels.device).unsqueeze(1)
        # batch_size x n_patches x feature_dim x n_features
        zero = torch.tensor([0], dtype=features.dtype).to(labels.device)
        grouped = torch.where(mask.unsqueeze(2), features.unsqueeze(1),
                              zero)
        return grouped, mask

    def get_patch_embeddings(self, features, labels):
        """
        :param features: features for all face/point/pixel/voxel, batch_size x feature_dim x n_features
        :param labels: patches/segmentation labels, batch_size x n_features
        """
        grouped, mask = self.group_by(features, labels)
        counts = mask.sum(axis=-1)
        counts_nonzero = torch.where(counts != 0, counts, 1)

        pooled = grouped.sum(dim=-1) / counts_nonzero.unsqueeze(2)

        if self.mlp is not None:
            pooled = self.mlp(pooled.transpose(2, 1)).transpose(2, 1)

        if self.normalize:
            pooled = torch.nn.functional.normalize(pooled, dim=-1)

        if self.flatten_embeddings:
            return pooled[torch.nonzero(counts, as_tuple=True)]

        return pooled, counts

    @abstractmethod
    def forward_features(self, x):
        pass

    def forward(self, data):
        features, patch_labels = data
        return self.get_patch_embeddings(self.forward_features(features), patch_labels)


class ElemPatchContrastModel(torch.nn.Module):
    def __init__(self, model, sample_frac, max_samples):
        super().__init__()
        self.model = model
        self.flatten_embeddings = False
        self.sample_frac = sample_frac
        self.max_samples = max_samples

    def forward(self, data):
        features, patch_labels = data
        elementwise_features = self.model.forward_features(features)
        grouped_features, mask = self.model.group_by(elementwise_features, patch_labels)
        counts = mask.sum(axis=-1)
        counts_nonzero = torch.where(counts != 0, counts, 1)
        patch_embeddings = grouped_features.sum(dim=-1) / counts_nonzero.unsqueeze(2)

        if self.model.mlp:
            patch_embeddings = self.model.mlp(patch_embeddings.transpose(2, 1)).transpose(2, 1)

        if self.model.mlp:
            projected_elementwise_features = self.model.mlp(elementwise_features)
            grouped_proj_features, mask = self.model.group_by(projected_elementwise_features, patch_labels)
            samples, samples_sizes = self.sample_features_from_patches(grouped_proj_features, mask)
        else:
            samples, samples_sizes = self.sample_features_from_patches(grouped_features, mask)

        return patch_embeddings, counts, samples, samples_sizes

    def sample_features_from_patches(self, grouped, mask):
        # batch_size x n_patches
        batched_patch_sizes = mask.sum(-1)
        sampled_features = []

        actual_sample_sizes = torch.min((batched_patch_sizes * self.sample_frac).int(),
                                        torch.tensor([self.max_samples]).to(batched_patch_sizes.device))
        max_samples = actual_sample_sizes.max().item()
        fake_samples = torch.zeros(grouped.size(2), max_samples).to(grouped.device)

        for patches_features, sample_sizes, sample_weights in zip(grouped, actual_sample_sizes, mask.float()):
            # patches_features has shape n_patches x feature_dim x n_features
            # sample_sizes has shape n_patches
            # sample_weights has shape n_patches x n_features
            batch_sampled_features = []

            for patch_features, sample_size, sample_weight in zip(patches_features, sample_sizes, sample_weights):
                # patch_features has shape feature_dim x n_features
                # sample_size has shape 1
                # sample_weight has shape n_features
                if sample_size == 0:
                    batch_sampled_features.append(fake_samples)
                    continue

                selected_idx = torch.multinomial(sample_weight, sample_size.item())
                samples = patch_features.index_select(1, selected_idx)
                if sample_size < max_samples:
                    samples = torch.nn.functional.pad(samples, (0, max_samples - sample_size))

                batch_sampled_features.append(samples)

            sampled_features.append(torch.stack(batch_sampled_features, dim=0))

        return torch.stack(sampled_features, dim=0), actual_sample_sizes
