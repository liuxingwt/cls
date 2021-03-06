# Classification
This repository is an implementation of basic image classification framework.

### Environment
+ OS version: Ubuntu 18.04
+ NVIDIA diver version: 465.27
+ Cuda version: 11.3
+ Python version: 3.6.9
+ Python packages installation:  
  `pip3 install -i https://mirrors.aliyun.com/pypi/simple  -r requirements.txt`
 
### Train
`sh run/train.sh`

### Test
`sh run/test.sh`
 
### Reference:
#### Code:
+ pytroch-imageNet: https://github.com/pytorch/examples/blob/master/imagenet/main.py
+ pytorch-image-models: https://github.com/rwightman/pytorch-image-models
+ Vit-Pytorch: https://github.com/lucidrains/vit-pytorch  
+ DeiT: https://github.com/facebookresearch/deit
+ Swin-Transformer: https://github.com/microsoft/Swin-Transformer  
  
#### Paper:
+ Vit: https://arxiv.org/abs/2010.11929
+ DeiT: https://arxiv.org/abs/2012.12877
+ Swin: http://arxiv.org/abs/2103.14030
+ CaiT: https://arxiv.org/abs/2103.17239

### Changing log:
**20210524**: change the order of input image channel from BGR to RGB, to better adapt to pretrained model in ImageNet.
+ dataloader.py：修改了图片加载顺序BGR->RGB
+ test.py：修改了图片加载顺序BGR->RGB 
+ train.py: 在一个epoch内保存模型时，只保存权重  

**20211203**:  
+ optim/: 增加了ranger优化器，并修改了build_optimizer.py和train.py文件  
+ tools/metrics.py：重构了二分类结果evaluation的流程代码  
