# YOLOV7_Train_PyTorch
1000行代码完美复现YOLOV7的训练和测试，精度、速度以及配置完全相同，两者模型可以无障碍相互转换  

## 指标展示
|Model| train | test | net_size | mAP@0.5 | FPS | tips |
|-----|------|------|-----|-----|-----|-----|
|yolov7(train from yolov7) | 0712 |	2007_test | 640x640 |	89.11 |	161 |	yolov7-ota-loss, `IDetect Head` |
|**yolov7(ours)** | 0712 |	2007_test | 640x640 |	**88.59** |	**161** | yolov7-ota-loss, `IDetect Head` |

### 训练和测试
```shell script
python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py
```
已训练好的模型：  
[DetectMode(提取码:8888)](https://pan.baidu.com/s/1w4nOx0VSx5trbK7sL0PkBA)  

## 参考
https://blog.csdn.net/samylee  
https://github.com/AlexeyAB/darknet  
https://github.com/WongKinYiu/yolov7
