import torch
import torch.nn.functional as F


def contrastive_loss(v1_embeddings, v2_embeddings, params):
    v1_embeddings = F.normalize(v1_embeddings, dim=1)
    v2_embeddings = F.normalize(v2_embeddings, dim=1)

    batch_size = v1_embeddings.size(0)
    embs = torch.cat((v1_embeddings, v2_embeddings), dim=0)
    logits = embs @ embs.transpose(1, 0) / params['tau']

    # discard self similarities
    mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=v1_embeddings.device)
    logits = (logits
              .masked_select(mask)
              .view(2 * batch_size, 2 * batch_size - 1)
              .contiguous()
              )

    labels = torch.cat((torch.arange(batch_size) + batch_size - 1,
                        torch.arange(batch_size)
                        ), dim=0).to(v1_embeddings.device)

    return F.cross_entropy(logits, labels)


def groupby(features, face_indexes, max_faces):
    '''
    :param features: features for all face/point/pixel/voxel, batch_size x feature_dim x n_features
    :param labels: patches/segmentation labels, batch_size x n_features
    '''    
    mask = face_indexes.unsqueeze(1) == torch.arange(max_faces).unsqueeze(1).to(face_indexes.device)
    zero = torch.FloatTensor([0]).to(face_indexes.device)
    grouped = torch.where(mask.unsqueeze(2), features.unsqueeze(1), zero)
    
    return grouped, mask


def get_patch_embeddings(features, labels, max_faces=None):
    """
    :param features: features for all face/point/pixel/voxel, batch_size x feature_dim x n_features
    :param labels: patches/segmentation labels, batch_size x n_features
    :return: pooled (BxFxN), counts (BxN)
    """
    grouped, mask = groupby(features, labels, max_faces)
    counts = mask.sum(axis=-1)
    counts_nonzero = torch.where(counts != 0, counts, 1)

    pooled = grouped.sum(dim=-1) / counts_nonzero.unsqueeze(2)

    return pooled.transpose(1, 2), counts


def face_indexes_to_patch_counts(face_indexes, max_faces):
    face_nums = face_indexes.max(1).values
    idx = torch.arange(max_faces).unsqueeze(0)\
          .expand(face_indexes.size(0), -1).to(face_indexes.device)
    return (idx < face_nums.unsqueeze(1)).long()


def patch_contrastive_loss(x1, x2, params):
    """
    x1 and x2 are tuples with not flattened embeddings and patch_counts
    """
    v1_embeddings, v1_patch_counts = x1
    v2_embeddings, v2_patch_counts = x2

    n_patches = v1_embeddings.size(2)

    v1_embeddings = F.normalize(v1_embeddings, dim=1)
    v2_embeddings = F.normalize(v2_embeddings, dim=1)

    embs = torch.cat((v1_embeddings, v2_embeddings), dim=2)
    # b x (2 * n_patches) x (2 * n_patches)
    logits = torch.bmm(embs.transpose(2, 1), embs) / params['tau']

    # discard self similarities
    mask = ~torch.eye(n_patches * 2, dtype=torch.bool, device=v1_embeddings.device)
    logits = (logits
              .masked_select(mask)
              .view(-1, 2 * n_patches, 2 * n_patches - 1)
              .transpose(2, 1)
              .contiguous()
              )

    # ignore error for empty patches
    empty_patches_mask = torch.cat(((v1_patch_counts == 0), (v2_patch_counts == 0)), dim=1)
    labels = torch.cat((torch.arange(n_patches) + n_patches - 1,
                        torch.arange(n_patches)
                        ), dim=0).to(v1_embeddings.device)
    ignore_label = torch.Tensor([-100]).to(v1_embeddings.device).long()
    labels = torch.where(empty_patches_mask, ignore_label, labels)

    return F.cross_entropy(logits, labels)