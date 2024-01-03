import torch
import numpy as np
import cv2
import os
import random
from torch.utils.data import Dataset

from utils.utils import xywh2xyxy, xyxy2xywh, random_perspective, letterbox, augment_hsv


class VOCDataset(Dataset):
    def __init__(self, label_list, net_size=640):
        super(VOCDataset, self).__init__()
        self.mosaic = True
        self.net_size = net_size
        self.mosaic_border = [-self.net_size // 2, -self.net_size // 2]

        with open(label_list, 'r') as f:
            image_path_lines = f.readlines()

        self.images_path = []
        self.labels = []
        for image_path_line in image_path_lines:
            image_path = image_path_line.strip().split()[0]
            label_path = image_path.replace('JPEGImages', 'labels').replace('jpg', 'txt')
            if not os.path.exists(label_path):
                continue

            self.images_path.append(image_path)
            with open(label_path, 'r') as f:
                label_lines = f.readlines()

            labels_tmp = np.empty((len(label_lines), 5), dtype=np.float32)
            for i, label_line in enumerate(label_lines):
                labels_tmp[i] = [float(x) for x in label_line.strip().split()]
            self.labels.append(labels_tmp)

        assert len(self.images_path) == len(self.labels), 'images_path\'s length dont match labels\'s length'
        self.indices = range(len(self.images_path))

    def __getitem__(self, idx):
        # mosaic data augment
        if self.mosaic:
            if random.random() < 0.8:
                image, labels = self.load_mosaic4(idx)
            else:
                image, labels = self.load_mosaic9(idx)
        else:
            image, labels = self.load_origin(idx)

        # Augment hsv
        augment_hsv(image, hgain=0.015, sgain=0.7, vgain=0.4)

        nL = len(labels)
        if nL:
            labels = xyxy2xywh(labels, image.shape[1], image.shape[0])

        # Augment flip
        if random.random() < 0.5:
            image = np.fliplr(image)
            if nL:
                labels[:, 1] = 1 - labels[:, 1]

        # labels to torch
        targets = torch.zeros((nL, 6))
        if nL:
            targets[:, 1:] = torch.from_numpy(labels)

        # images to torch
        image = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
        inputs = torch.from_numpy(image).float().div(255)

        return inputs, targets

    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def collate_fn(batch):
        image, label = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(image, 0), torch.cat(label, 0)

    def load_image(self, index):
        path = self.images_path[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.net_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        return img, img.shape[:2]

    def load_origin(self, index):
        img, (h, w) = self.load_image(index)

        # Letterbox
        img, ratio, pad = letterbox(img, self.net_size, auto=False, scaleup=True)

        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels = xywh2xyxy(labels, ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        img, labels = random_perspective(img, labels,
                                         degrees=0,
                                         translate=0.2,
                                         scale=0.5,
                                         shear=0,
                                         perspective=0)

        return img, labels

    def load_mosaic4(self, index):
        # loads images in a 4-mosaic
        labels4 = []
        s = self.net_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)
        # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels = self.labels[index].copy()
            if labels.size:
                labels = xywh2xyxy(labels, w, h, padw, padh)
            labels4.append(labels)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in labels4[:, 1:]:
            np.clip(x, 0, 2 * s, out=x)

        # Augment
        img4, labels4 = random_perspective(img4, labels4,
                                           degrees=0,
                                           translate=0.2,
                                           scale=0.5,
                                           shear=0,
                                           perspective=0,
                                           border=self.mosaic_border)  # border to remove

        return img4, labels4

    def load_mosaic9(self, index):
        # loads images in a 9-mosaic
        labels9 = []
        s = self.net_size
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, (h, w) = self.load_image(index)

            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

            # Labels
            labels = self.labels[index].copy()
            if labels.size:
                labels = xywh2xyxy(labels, w, h, padx, pady)  # normalized xywh to pixel xyxy format
            labels9.append(labels)

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc

        for x in (labels9[:, 1:]):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()

        # Augment
        img9, labels9 = random_perspective(img9, labels9,
                                           degrees=0,
                                           translate=0.2,
                                           scale=0.5,
                                           shear=0,
                                           perspective=0,
                                           border=self.mosaic_border)  # border to remove

        return img9, labels9