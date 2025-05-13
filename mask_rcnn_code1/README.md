# Mask R-CNN及模型改进

## 该项目参考自pytorch官方torchvision模块中的源码
* https://github.com/pytorch/vision/tree/master/references/detection

## 环境配置：
* Python3.7/3.8/3.9
* Pytorch1.10或以上
* Windows
* 最好使用GPU训练
* 详细环境配置见`requirements.txt`

## 文件结构：
```
  ├── backbone: 特征提取网络
  ├── network_files: Mask R-CNN网络
  ├── train_utils: 训练验证相关模块（包括coco验证相关）
  ├── my_dataset_coco.py: 自定义dataset用于读取COCO2017数据集
  ├── my_dataset_voc.py: 自定义dataset用于读取Pascal VOC数据集
  ├── train.py: Mask-RCNN
  ├── train_mobile.py: Mask-RCNNV3
  ├── train_efficient.py: Mask-RCNNB0
  ├── train_v3best.py: Mask-RCNNV3_best
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测
  ├── validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record_mAP.txt文件
```

## 预训练权重下载地址（下载后放入当前文件夹中）：
* Resnet50预训练权重 https://download.pytorch.org/models/resnet50-0676ba61.pth (注意，下载预训练权重后要重命名，
比如在train.py中读取的是`resnet50.pth`文件，不是`resnet50-0676ba61.pth`)
* Mask R-CNN(Resnet50+FPN)预训练权重 https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth (注意，
载预训练权重后要重命名，比如在train.py中读取的是`maskrcnn_resnet50_fpn_coco.pth`文件，不是`maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth`)



## 训练方法
* 确保提前准备好数据集
* 确保提前下载好对应预训练模型权重
* 确保设置好`--num-classes`和`--data-path`
* 若要使用单GPU训练直接使用train.py训练脚本
* 若要使用多GPU训练，使用`torchrun --nproc_per_node=8 train_multi_GPU.py`指令,`nproc_per_node`参数为使用GPU数量
* 如果想指定使用哪些GPU设备可在指令前加上`CUDA_VISIBLE_DEVICES=0,3`(例如我只要使用设备中的第1块和第4块GPU设备)
* `CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 train_multi_GPU.py`

## 注意事项
1. 在使用训练脚本时，注意要将`--data-path`设置为自己存放数据集的**根目录**：
```
# 假设要使用COCO数据集，启用自定义数据集读取CocoDetection并将数据集解压到成/data/coco2017目录下
python train.py --data-path /data/coco2017

# 假设要使用Pascal VOC数据集，启用自定义数据集读取VOCInstances并数据集解压到成/data/VOCdevkit目录下
python train.py --data-path /data/VOCdevkit
```

