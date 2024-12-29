- [Tensorflow数据处理篇](#Tensorflow数据处理篇) <br/>
  - [TF常用操作](#TF常用操作) <br/>
    - [softmax近似计算](#softmax近似计算) <br/>
    - [tensor指定索引处update](#tensor指定索引处update) <br/>
    - [BeamSearch](#BeamSearch) <br/>

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

### tensor指定索引处update
(1) tf.tensor_scatter_nd_update(https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update)
```
tensor = [0, 0, 0, 0, 0, 0, 0, 0]    # tf.rank(tensor) == 1
indices = [[1], [3], [4], [7]]       # num_updates == 4, index_depth == 1
updates = [9, 10, 11, 12]            # num_updates == 4
print(tf.tensor_scatter_nd_update(tensor, indices, updates))
```

### BeamSearch
tf.nn.ctc_beam_search_decoder(https://www.tensorflow.org/api_docs/python/tf/nn/ctc_beam_search_decoder)
```
tf.nn.ctc_beam_search_decoder(
    inputs, sequence_length, beam_width=100, top_paths=1
)
inputs:	3-D float Tensor, size [max_time, batch_size, num_classes]. The logits. # max_time一般等于1
sequence_length:	1-D int32 vector containing sequence lengths, having size [batch_size]. # 表示生成序列的最大长度限制
beam_width:	An int scalar >= 0 (beam search beam width).
top_paths:	An int scalar >= 0, <= beam_width (controls output size).
```
关于其源码的文档：https://zhuanlan.zhihu.com/p/39018903



















    
