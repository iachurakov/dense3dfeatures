import os
import neptune.new as neptune
import tqdm
import yaml
from time import time
import torch.distributed as dist
import torch.multiprocessing as mp
from ..models.dgcnn import DGCNN
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel
from ..models.basemodel import ElemPatchContrastModel
from ..datasets.multimodal_dataset import *
from ..datasets.datasets_meta import datasets_meta_dict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .average_meter import *
from .losses import losses_dict
from ..unsupervised.byol import BYOL
import argparse


def get_dataset(rank, world_size, params):
    modalities = params['data']['modalities']
    dataset_meta = datasets_meta_dict[params['data']['name']]

    train_dataset = DoubleDataset(data_path=params['data']['path'],
                                  split='train',
                                  modalities=modalities,
                                  dataset_meta=dataset_meta,
                                  point_cloud_aug=params['data']['aug'])

    # train_dataset.keys = train_dataset.keys[:200]

    sampler = DistributedSampler(train_dataset, rank=rank, shuffle=True, num_replicas=world_size)

    train_loader = DataLoader(train_dataset,
                              batch_size=params['data']['batch_size'],
                              shuffle=False,
                              sampler=sampler)

    val_dataset = DoubleDataset(data_path=params['data']['path'],
                                split='val',
                                dataset_meta=dataset_meta,
                                modalities=modalities,
                                point_cloud_aug=params['data']['aug'])

    # val_dataset.keys = val_dataset.keys[:200]

    sampler = DistributedSampler(val_dataset, rank=rank, shuffle=True, num_replicas=world_size)

    val_loader = DataLoader(val_dataset,
                            batch_size=params['data']['batch_size'],
                            shuffle=False,
                            sampler=sampler)

    return train_loader, val_loader


def get_model(params, n_iters=0):
    model = DGCNN(n_patches=datasets_meta_dict[params['data']['name']].n_patches,
                  flatten_embeddings=params['model']['flatten_embeddings'],
                  mlp=params['model']['mlp'],
                  reduction=params['model']['reduction'],
                  n_output=params['model']['n_output'])

    if params['train']['framework'] == 'patch_contrastive':
        return model, losses_dict[params['train']['framework']]

    if params['train']['framework'] == 'elem_patch_contrastive':
        model = ElemPatchContrastModel(model, params['train']['sample_frac'], params['train']['max_samples'])

    if params['train']['framework'] == 'byol':
        model = BYOL(model,
                     net_output_dim=params['model']['n_output'],
                     hidden_size=params['model']['hidden_size'],
                     projection_size=params['model']['proj_dim'],
                     moving_average_decay=params['train']['moving_average_decay'],
                     n_iters=n_iters)

    if params['train']['framework'] == 'moco':
        pass

    return model, losses_dict[params['train']['framework']]


def calc_val_loss(model, loader, loss_fn, params, rank, world_size):
    model.eval()
    loss_logs = MultipleAverageMeters(rank, world_size)

    with torch.no_grad():
        for x1, x2 in loader:
            data1, patch_labels1, seg_labels1 = x1
            data2, patch_labels2, seg_labels2 = x2

            # batch_size = data1.shape[0]

            x1 = data1.to(rank), patch_labels1.to(rank)
            x2 = data2.to(rank), patch_labels2.to(rank)

            loss = loss_fn(model, x1, x2, params)
            total_loss = 0
            for loss_v in loss.values():
                total_loss += loss_v

            loss['total_loss'] = total_loss
            loss_logs.update(loss)

    return loss_logs


def cleanup():
    dist.destroy_process_group()


def init_process(rank, size, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)


def init_logger(params):
    logger = neptune.init(project=params['logger']['project_name'],
                          name=params['logger']['exp_name'],
                          tags=params['logger']['tags'],
                          api_token=params['logger']['api_token'])

    logger['parameters'] = params

    return logger


def worker(rank, world_size, params):
    init_process(rank, world_size)

    if rank == 0:
        logger = init_logger(params)
        exp_id = logger['sys/id'].fetch()
        save_dir = Path(params['train']['weights_root']) / exp_id
        save_dir.mkdir(parents=True, exist_ok=True)
        best_val_loss = float('inf')

    dist.barrier(device_ids=[rank])
    train_loader, val_loader = get_dataset(rank, world_size, params)
    n_iters = len(train_loader) * params['train']['total_epochs']
    model, loss_fn = get_model(params, n_iters)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(rank)
    dpp_model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(dpp_model.parameters(), lr=params['train']['lr'] * world_size,
                                  weight_decay=params['train']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           len(train_loader) * params['train']['total_epochs'])

    loss_logs = MultipleAverageMeters(rank, world_size)
    time_meter = AverageMeter('time', rank, world_size)

    if rank == 0:
        print('Start training')

    for epoch in range(1, params['train']['total_epochs'] + 1):
        start_time = time()
        dpp_model.train()
        for x1, x2 in train_loader:
            optimizer.zero_grad()

            data1, patch_labels1, seg_labels1 = x1
            data2, patch_labels2, seg_labels2 = x2

            x1 = data1.to(rank), patch_labels1.to(rank)
            x2 = data2.to(rank), patch_labels2.to(rank)

            loss = loss_fn(dpp_model, x1, x2, params)
            total_loss = 0

            for loss_v in loss.values():
                total_loss += loss_v

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            loss['total_loss'] = total_loss
            loss_logs.update(loss)

        loss_logs.gather()
        time_meter.update(time() - start_time)
        time_meter.gather()

        val_loss = calc_val_loss(model, val_loader, loss_fn, params, rank, world_size)
        val_loss.gather()

        if rank == 0:
            elapsed_min = time_meter.avg / 60
            print(f'Epoch: {epoch}, train_loss: {str(loss_logs)} elapsed: {elapsed_min:.3f} min')
            print(f'Val loss: {str(val_loss)}')

            for loss_type, loss_meter in loss_logs.items:
                logger[f'train/loss/{loss_type}'].log(loss_meter.avg)

            for loss_type, loss_meter in val_loss.items:
                logger[f'val/loss/{loss_type}'].log(loss_meter.avg)

            if val_loss['total_loss'] < best_val_loss:
                best_val_loss = val_loss['total_loss']
                torch.save(model.state_dict(), save_dir / 'val_best.pt')

            if epoch % params['train']['save_every'] == 0:
                torch.save(model.state_dict(), save_dir / f'{epoch}epoch.pt')

            if params['train']['total_epochs'] % params['train']['save_every'] != 0:
                torch.save(model.state_dict(), save_dir / f'{epoch}epoch.pt')

            print()

        dist.barrier(device_ids=[rank])

    if rank == 0:
        logger.stop()

    cleanup()


def main():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config',
                        action='store',
                        type=str,
                        help='path to training config file')

    parser.add_argument('--gpus',
                        action='store',
                        type=int,
                        help='num of gpus')

    opts = parser.parse_args()

    with open(opts.config, 'r') as config:
        params = yaml.load(config, Loader=yaml.FullLoader)

    mp.spawn(worker,
             args=(opts.gpus, params),
             nprocs=opts.gpus,
             join=True)


if __name__ == "__main__":
    main()
