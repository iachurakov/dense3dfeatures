import torch
from typing import *
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


def log_dict(dict_loss, logger, mode):
    for k, v in dict_loss.items():
        logger[mode + '/' + k].log(v)


@torch.no_grad()
def calc_val_loss(model, loader, logger, forward):
    model.eval()
    progress_bar = tqdm(loader, leave=True, position=0, desc='Validation')
    dict_loss = defaultdict(int)
    cur_iter = 0
    
    
    for batch in progress_bar:        
        loss = forward(model, batch, logger, 'val')

        for k, v in loss.items():
            dict_loss[k] += v.item()
        cur_iter += 1

        progress_bar.set_postfix({
            'Loss': dict_loss['loss'] / cur_iter
        })

    return {k: v / cur_iter for k, v in dict_loss.items()}


def train_model(
    model: torch.nn.Module,
    params: Dict[str, Any],
    logger: Any,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    forward: Callable[[torch.nn.Module, Any, Any, str], Dict[str, torch.Tensor]],
):
    '''
    :param model: torch.nn.Module model
    :param params: experiment parameters
    :param logger: logger, neptune instance in our case
    :param train_loader: train loader
    :param val_loader: val loader
    :param optimizer: optimizer
    :param scheduler: scheduler
    :param forward: forward(model, batch, logger, 'val'/'train') -> loss
    loss is dict with Tensors, .backward() is called on 'loss' key, other keys are only logged
    '''
    exp_id = logger['sys/id'].fetch()
    save_dir = Path('{}/{}'.format(params['weights_root'], exp_id))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')

    for epoch in range(1, params['total_epochs'] + 1):
        progress_bar = tqdm(train_loader, leave=True, position=0)

        dict_loss = defaultdict(int)
        cur_iter = 0


        model.train()
        for batch in progress_bar:
            optimizer.zero_grad()

            loss = forward(model, batch, logger, 'train')
            
            for k, v in loss.items():
                dict_loss[k] += v.item()
        
            cur_iter += 1

            loss['loss'].backward()
            optimizer.step()
            scheduler.step()
            progress_bar.set_postfix({
                'Epoch': epoch,
                'Loss': dict_loss['loss'] / cur_iter
            }) 

        dict_loss = {k: v / cur_iter for k, v in dict_loss.items()}

        log_dict(dict_loss, logger, 'train')
        
        val_dict_loss = calc_val_loss(model, val_loader, logger, forward)
        log_dict(val_dict_loss, logger, 'val')

        if val_dict_loss['loss'] < best_val_loss:
            best_val_loss = val_dict_loss['loss']
            torch.save(model.state_dict(), save_dir / 'val_best.pt')
            
                    
        if epoch % params['save_every'] == 0:
            torch.save(model.state_dict(), save_dir / f'{epoch}epoch.pt')
    
    if params['total_epochs'] % params['save_every'] != 0:
        torch.save(model.state_dict(), save_dir / f'{epoch}epoch.pt') 