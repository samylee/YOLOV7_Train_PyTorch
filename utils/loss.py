import torch
import torch.nn as nn
import numpy as np
from utils.utils import bbox_iou


class YOLOV7Loss(nn.Module):
    def __init__(self, B=3, C=20, net_size=None, strides=None, anchors=None, masks=None, device=None):
        super(YOLOV7Loss, self).__init__()
        self.device = device
        self.B, self.C = B, C
        self.net_size = net_size
        self.strides = strides
        self.anchors = torch.from_numpy(np.asarray(anchors, dtype=np.float32)).to(self.device)
        self.masks = torch.from_numpy(np.asarray(masks, dtype=np.int)).to(self.device)
        self.scale_x_y = [2.0, 2.0, 2.0]
        self.balance = [4.0, 1.0, 0.4]
        self.iou_normalizer = 0.05
        self.obj_normalizer = 0.7 * (self.net_size / 640) ** 2
        self.cls_normalizer = 0.3 * self.C / 80.
        self.anchor_t = 4.0
        self.append_g = 0.5

        self.cls_criterion = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
        self.obj_criterion = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)

    def make_grid(self, nx, ny):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def forward(self, preds, targets, idx):
        batch_size, _, grid_size, _ = preds.shape

        # num_samples, 3(anchors), 13(grid), 13(grid), 25 (tx, ty, tw, th, conf, classes)
        preds_permute = (
            preds.view(batch_size, self.B, self.C+5, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
        )

        # anchors, grids
        grid = self.make_grid(grid_size, grid_size).to(self.device)
        anchor_grid = self.anchors[self.masks[idx]].view(1, -1, 1, 1, 2)

        # tx, ty, tw, th
        preds_txty = (torch.sigmoid(preds_permute[..., 0:2]) * self.scale_x_y[idx] - 0.5 * (self.scale_x_y[idx] - 1) + grid) / grid_size
        preds_twth = torch.exp(preds_permute[..., 2:4]) * anchor_grid / self.net_size
        preds_box = torch.cat((preds_txty, preds_twth), dim=-1)

        # conf, class
        preds_obj = preds_permute[..., 4]
        preds_cls = preds_permute[..., 5:]

        # get targets
        targets_obj, targets_cls, targets_box = self.build_targets(preds_box, preds_obj, preds_cls, targets, idx)

        if targets_obj.sum() > 0:
            obj_mask = targets_obj.type(torch.bool)

            # 1. cls loss
            cls_mask = obj_mask.unsqueeze(-1).expand_as(preds_cls)
            preds_cls_mask = preds_cls[cls_mask].view(-1, self.C)
            targets_cls_mask = targets_cls[cls_mask].view(-1, self.C)
            loss_cls = self.cls_criterion(preds_cls_mask, targets_cls_mask)

            # 2. box loss
            box_mask = obj_mask.unsqueeze(-1).expand_as(preds_box)
            preds_box_mask = preds_box[box_mask].view(-1, 4)
            targets_box_mask = targets_box[box_mask].view(-1, 4)
            iou = bbox_iou(preds_box_mask, targets_box_mask, CIoU=True)
            loss_box = (1. - iou).mean()

            # obj to iou
            targets_obj[obj_mask] = iou.detach().clamp(0).type(targets_obj.dtype)
        else:
            loss_box, loss_cls = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)

        # 3. obj loss
        loss_obj = self.obj_criterion(preds_obj, targets_obj)

        loss = self.obj_normalizer * self.balance[idx] * loss_obj + self.cls_normalizer * loss_cls + self.iou_normalizer * loss_box

        return loss * batch_size

    def build_targets(self, preds_box, preds_obj, preds_cls, targets, idx):
        batch_size, _, grid_size, _, _ = preds_box.shape

        targets_obj = torch.zeros_like(preds_obj, requires_grad=False)
        targets_cls = torch.zeros_like(preds_cls, requires_grad=False)
        targets_box = torch.zeros_like(preds_box, requires_grad=False)
        ij_pos = [False for i in range(5)]
        for b in range(batch_size):
            targets_batch = targets[targets[:, 0] == b][:, 1:]
            for target_batch in targets_batch:
                target_cls_batch = int(target_batch[0])
                assert target_cls_batch < self.C, 'oh shit'
                target_box_batch = target_batch[1:]

                # match targets
                target_box_batch_shift = target_box_batch[2:] * grid_size
                anchors_match_batch_shift = self.anchors[self.masks[idx]] / self.strides[idx]
                ratio = target_box_batch_shift / anchors_match_batch_shift
                ratio_matches = torch.max(ratio, 1. / ratio).max(1)[0] < self.anchor_t

                if ratio_matches.sum() == 0:
                    continue

                # base ij
                target_grid_x = target_box_batch[0] * grid_size
                target_grid_y = target_box_batch[1] * grid_size
                i_base = int(target_grid_x)
                j_base = int(target_grid_y)
                ij_pos[0] = True

                # append positive
                off_i, off_j = target_grid_x % 1., target_grid_y % 1.
                ij_pos[1] = True if off_i < self.append_g and target_grid_x > 1. else False                # left
                ij_pos[2] = True if off_j < self.append_g and target_grid_y > 1. else False                # top
                ij_pos[3] = True if off_i > self.append_g and grid_size - target_grid_x > 1. else False    # right
                ij_pos[4] = True if off_j > self.append_g and grid_size - target_grid_y > 1. else False    # bot

                for pos_idx, pos in enumerate(ij_pos):
                    if pos:
                        if pos_idx == 0:
                            i, j = i_base, j_base
                        elif pos_idx == 1:
                            i, j = i_base - 1, j_base
                        elif pos_idx == 2:
                            i, j = i_base, j_base - 1
                        elif pos_idx == 3:
                            i, j = i_base + 1, j_base
                        else:
                            i, j = i_base, j_base + 1

                        for match_idx, ratio_match in enumerate(ratio_matches):
                            if ratio_match:
                                targets_obj[b, match_idx, j, i] = 1.
                                targets_cls[b, match_idx, j, i, target_cls_batch] = 1.
                                targets_box[b, match_idx, j, i] = target_box_batch

        return targets_obj, targets_cls, targets_box