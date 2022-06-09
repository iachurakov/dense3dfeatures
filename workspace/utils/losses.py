import torch
import torch.nn.functional as F



def byol_loss_fn(x, y):
    '''
    x and y are flattened along last dim embeddings
    '''
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)

    return 2 - 2 * (x * y).sum(dim=-1)
    

def byol_loss(target_embeddings_one, target_embeddings_two, online_predictions_one, online_predictions_two):
    '''
    inputs output of byol model
    '''
    loss_one = byol_loss_fn(target_embeddings_one, online_predictions_two)
    loss_two = byol_loss_fn(target_embeddings_two, online_predictions_one)
    loss = loss_one + loss_two

    return {'byol_loss': loss.mean()}
    

def patch_contrastive_loss(x1, x2, params):
    '''
    x1 and x2 are tuples with not flattened embeddings and patch_counts
    '''
    v1_embeddings, v1_patch_counts = x1
    v2_embeddings, v2_patch_counts = x2

    n_patches = v1_patch_counts.size(1)
    mask = (v1_patch_counts == 0) | (v2_patch_counts == 0)
    ignore_index = torch.LongTensor([-100]).to(v1_embeddings.device)
    rng = torch.arange(n_patches).to(v1_embeddings.device)

    labels = torch.where(mask, ignore_index, rng)

    v1_embeddings = F.normalize(v1_embeddings, dim=-1)
    v2_embeddings = F.normalize(v2_embeddings, dim=-1)

    v1_logits = torch.bmm(v1_embeddings, v2_embeddings.transpose(2, 1)) / params['tau']
    v2_logits = torch.bmm(v2_embeddings, v1_embeddings.transpose(2, 1)) / params['tau']

    '''if params['negative_sampler'] is not None:
        pos_index = torch.arange(v1_logits.shape[1]).unsqueeze(1).unsqueeze(0).expand(v1_logits, -1, -1)
        v1_logits = torch.gather(a, 2, index)'''

    loss1 = F.cross_entropy(v1_logits, labels)
    loss2 = F.cross_entropy(v2_logits, labels)

    return {'ce': loss1 + loss2}


def elem_patch_contrastive_loss(x1, x2, params):
    '''
    x1 and x2 are tuples of embeddings, patch_sizes, samples, samples_sizes
    '''
    def contrast_point_patch(patch_embeddings, samples, patch_sizes, sample_sizes, tau):
        # patch_embeddings bs x n_patches x dim
        # samples bs x n_patches x dim x n_samples
        device = patch_embeddings.device

        batch_size, n_patches, embedding_dim = patch_embeddings.shape
        # bs x dim x (n_patches x n_samples)
        samples = samples.transpose(2, 1).reshape(batch_size, embedding_dim, -1)
        # bs x n_patches (from patch_embeddings) x n_patches (from samples) x n_samples
        logits = torch.bmm(patch_embeddings, samples).reshape(batch_size, n_patches, n_patches, -1) / tau
        # decrease logits of fake patches
        tiny_logit = torch.FloatTensor([-1000.]).to(patch_embeddings.device)
        logits = torch.where(patch_sizes.unsqueeze(2).unsqueeze(3) == 0,
                             tiny_logit, logits)
        # Point-patch contrast – points classification problem, where classes are patches' number
        # bs x n_patches x n_samples
        idx = torch.arange(logits.size(-1)).to(device).unsqueeze(0).unsqueeze(1).expand(batch_size, n_patches, -1)

        ignore_label = torch.LongTensor([-100]).to(patch_embeddings.device)
        labels = torch.where(idx < sample_sizes.unsqueeze(-1),
                             torch.arange(n_patches).to(device).unsqueeze(-1), ignore_label)

        return F.cross_entropy(logits, labels)

    tau = params['train']['tau']
    v1_embeddings, v1_patch_sizes, v1_samples, v1_samples_sizes = x1
    v2_embeddings, v2_patch_sizes, v2_samples, v2_samples_sizes = x2

    v1_embeddings = F.normalize(v1_embeddings, dim=-1)
    v2_embeddings = F.normalize(v2_embeddings, dim=-1)
    v1_samples = F.normalize(v1_samples, dim=2)
    v2_samples = F.normalize(v2_samples, dim=2)

    n_patches = v1_patch_sizes.size(1)
    mask = (v1_patch_sizes == 0) | (v2_patch_sizes == 0)
    ignore_index = torch.LongTensor([-100]).to(v1_embeddings.device)
    rng = torch.arange(n_patches).to(v1_embeddings.device)

    labels = torch.where(mask, ignore_index, rng)

    v1_logits = torch.bmm(v1_embeddings, v2_embeddings.transpose(2, 1)) / tau
    v2_logits = torch.bmm(v2_embeddings, v1_embeddings.transpose(2, 1)) / tau

    patch_loss1 = F.cross_entropy(v1_logits, labels)
    patch_loss2 = F.cross_entropy(v2_logits, labels)

    point_patch_v1_loss = contrast_point_patch(v1_embeddings, v1_samples, v1_patch_sizes, v1_samples_sizes, tau)
    point_patch_v2_loss = contrast_point_patch(v2_embeddings, v2_samples, v2_patch_sizes, v2_samples_sizes, tau)
    point_patch_v12_loss = contrast_point_patch(v1_embeddings, v2_samples, v1_patch_sizes, v2_samples_sizes, tau)
    point_patch_v21_loss = contrast_point_patch(v2_embeddings, v1_samples, v2_patch_sizes, v1_samples_sizes, tau)

    losses = {
            'patch': patch_loss1 + patch_loss2,
            'point_patch_v1': point_patch_v1_loss,
            'point_patch_v2': point_patch_v2_loss,
            'point_patch_v12': point_patch_v12_loss,
            'point_patch_v21': point_patch_v21_loss
        }
    return losses


losses_dict = {
    'byol': byol_loss,
    'patch_contrastive': patch_contrastive_loss,
    'elem_patch_contrastive': elem_patch_contrastive_loss,
}
