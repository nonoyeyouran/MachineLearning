- [Tensorflow数据处理篇](#Tensorflow数据处理篇) <br/>
  - [TF常用操作](#TF常用操作) <br/>

# Tensorflow2.0 guidebook

## Tensorflow数据处理篇
数据处理这里分为了**前置值处理**和**后置处理**，前置值处理主要是特征工程相关的（另行介绍），后置处理主要是标准化、离散化、one-hot化。
### 1.1 数据处理工具
#### 1.1.1 Pandas和Numpy
## 2. Tensorflow模型构建篇
## 3. Tensorflow模型在线服务篇
## TF常用操作
### Tensor降维
  tf.squeeze(input, axis=None, name=None), axis是个list，指定大小为1的那些维度进行缩减
### softmax近似计算
（1）tf.nn.sampled_softmax_loss（https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss）
  ```
  tf.nn.sampled_softmax_loss(
    weights, # 一般是token的embedding矩阵
    biases,
    labels, # 一般是目标token
    inputs,
    num_sampled,
    num_classes,
    num_true=1,
    sampled_values=None,
    remove_accidental_hits=True,
    seed=None,
    name='sampled_softmax_loss'
)
```
    
