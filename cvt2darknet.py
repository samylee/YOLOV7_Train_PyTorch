import numpy as np
import torch.nn as nn
from detect import model_init

def load_conv_model(module, f):
    if len(list(module.children())) > 0:
        if hasattr(module, 'conv'):
            conv_layer = module.conv
            bn_layer = module.bn
        else:
            conv_layer = module[0]
            bn_layer = module[1]
        # bn bias
        num_b = bn_layer.bias.numel()
        bn_b = bn_layer.bias.data.view(num_b).numpy()
        bn_b.tofile(f)
        # bn weights
        num_w = bn_layer.weight.numel()
        bn_w = bn_layer.weight.data.view(num_w).numpy()
        bn_w.tofile(f)
        # bn running mean
        num_rm = bn_layer.running_mean.numel()
        bn_rm = bn_layer.running_mean.data.view(num_rm).numpy()
        bn_rm.tofile(f)
        # bn running var
        num_rv = bn_layer.running_var.numel()
        bn_rv = bn_layer.running_var.data.view(num_rv).numpy()
        bn_rv.tofile(f)
    else:
        conv_layer = module
        # conv bias
        num_b = conv_layer.bias.numel()
        conv_b = conv_layer.bias.data.view(num_b).numpy()
        conv_b.tofile(f)
    # conv weights
    num_w = conv_layer.weight.numel()
    conv_w = conv_layer.weight.data.view(num_w).numpy()
    conv_w.tofile(f)

print('load pytorch model ... ')
checkpoint_path = 'weights/yolov7_170.pth'
B, C = 3, 20
model = model_init(checkpoint_path, B, C)

print('convert to darknet ... ')
with open('weights/yolov7-170.weights', 'wb') as f:
    np.asarray([0, 2, 0, 32013312, 0], dtype=np.int32).tofile(f)
    for name, module in model.named_children():
        if 'elan' in name or 'mp' in name:
            for subname, submodule in module.named_children():
                if 'mp' in subname:
                    continue
                load_conv_model(submodule, f)
        elif 'sppcspc' in name:
            load_conv_model(module.conv2, f)
            load_conv_model(module.conv1, f)
            load_conv_model(module.conv3, f)
            load_conv_model(module.conv4, f)
            load_conv_model(module.conv5, f)
            load_conv_model(module.conv6, f)
            load_conv_model(module.conv7, f)
        elif 'rep' in name:
            for subname, submodule in module.named_children():
                if 'act' in subname or 'rbr_identity' in subname:
                    continue
                load_conv_model(submodule, f)
        elif 'conv' in name or 'yolo' in name:
            load_conv_model(module, f)
        else:
            print(name, ' -> ignore')

print('done!')