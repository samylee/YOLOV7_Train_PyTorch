import torch
import torch.nn as nn
import math
import numpy as np
from copy import deepcopy
from torch.cuda import amp
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from tqdm import tqdm

from data.dataset import VOCDataset, InfiniteDataLoader
from models.YOLOV7 import YOLOV7
from utils.loss import YOLOV7Loss
from utils.utils import load_pretrain_weights, init_seeds, torch_distributed_zero_first, ModelEMA


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


if __name__ == '__main__':
    batch_size = 16
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    local_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    sync_bn = True

    # DDP mode
    total_batch_size = batch_size
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        assert batch_size % world_size == 0, '--batch-size must be multiple of CUDA device count'
        batch_size = total_batch_size // world_size

    cuda = device.type != 'cpu'
    init_seeds(2 + local_rank)

    scaler = amp.GradScaler(enabled=cuda)

    base_lr = 0.01
    lrf = 0.1
    weight_decay = 0.0005
    momentum = 0.937
    warmup_epochs = 3.0
    num_workers = 8
    nbs = 64  # nominal batch size
    warmup_momentum = 0.8  # warmup initial momentum
    warmup_bias_lr = 0.1  # warmup initial bias lr
    save_freq = 5

    num_epochs = 300
    B, C = 3, 20
    net_size = 640

    anchors = [[12, 16], [19, 36], [40, 28],
               [36, 75], [76, 55], [72, 146],
               [142, 110], [192, 243], [459, 401]]
    strides = [8, 16, 32]

    pretrain = 'pretrain/yolov7_samylee.pth'
    train_label_list = 'data/voc0712/train.txt'

    accumulate = max(round(nbs / total_batch_size), 1)
    weight_decay *= total_batch_size * accumulate / nbs

    # def model
    yolov7 = YOLOV7(B=B, C=C, strides=strides)
    load_pretrain_weights(yolov7, pretrain)
    yolov7 = yolov7.to(device)

    # EMA
    ema = ModelEMA(yolov7) if local_rank in [-1, 0] else None

    # SyncBatchNorm
    if sync_bn and cuda and local_rank != -1:
        yolov7 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(yolov7).to(device)

    yolov7.half().float()

    # DDP mode
    if cuda and local_rank != -1:
        yolov7 = DDP(yolov7, device_ids=[local_rank], output_device=local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in yolov7.modules()))

    # def loss
    criterion = YOLOV7Loss(B, C, net_size=net_size, strides=strides, anchors=anchors, device=device)

    # def optimizer
    optimizer = torch.optim.SGD(yolov7.parameters(), lr=base_lr, momentum=momentum, nesterov=True)
    lf = one_cycle(1, lrf, num_epochs)  # cosine 1->hyp['lrf']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # def dataset
    with torch_distributed_zero_first(local_rank):
        train_dataset = VOCDataset(train_label_list, net_size=net_size)
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, num_workers])
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if local_rank != -1 else None
    train_loader = InfiniteDataLoader(train_dataset,
                                      batch_size=batch_size,
                                      num_workers=nw,
                                      sampler=sampler,
                                      pin_memory=True,
                                      collate_fn=VOCDataset.collate_fn)

    print('Number of training images: ', len(train_dataset))

    nb = len(train_loader)
    nw = max(round(warmup_epochs * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)

    # train
    for epoch in range(num_epochs):
        yolov7.train()
        total_loss = 0.0

        if local_rank != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        if local_rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)

        optimizer.zero_grad()
        for i, (inputs, targets) in pbar:
            # Warmup
            ni = i + nb * epoch
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [warmup_bias_lr if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [warmup_momentum, momentum])

            current_lr = get_lr(optimizer)

            inputs = inputs.to(device)
            targets = targets.to(device)

            with amp.autocast(enabled=cuda):
                preds = yolov7(inputs)

                loss = criterion(preds, targets)

                if local_rank != -1:
                    loss *= world_size  # gradient averaged between devices in DDP mode
                if local_rank in [-1, 0]:
                    total_loss += loss.item()

            scaler.scale(loss).backward()

            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(yolov7)

            # Print current loss.
            if local_rank in [-1, 0]:
                s = ('Epoch [%d/%d], LR: %.6f, Loss: %.4f, Average Loss: %.4f'
                      % (epoch, num_epochs, current_lr, loss.item(), total_loss / (i + 1)))
                pbar.set_description(s)

        scheduler.step()
        if epoch % save_freq == 0 and local_rank in [-1, 0]:
            torch.save(deepcopy(ema.ema).state_dict(), 'weights/yolov7_' + str(epoch) + '.pth')

    if local_rank in [-1, 0]:
        torch.save(deepcopy(ema.ema).state_dict(), 'weights/yolov7_final.pth')

    dist.destroy_process_group()
    torch.cuda.empty_cache()