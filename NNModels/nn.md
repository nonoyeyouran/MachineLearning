神经网络文档

# attention网络
从attention可以看出，该网络的核心在于模仿人类的学习行为，学习时着重关注相关的信息。因此设计一个网络用于计算（学习）目标A对目标B的关注程度（注意力程度），在0-1之间，然后用该注意力值乘上目标B可以获得目标A从目标B处获得的相关信息；
注意力值越小，表示两者相关度越低，所获取的信息越少。通常被attention的对象B有多个，需要对注意力值进行softmax化。（是否可以不softmax化？全部注意力值之和一定要等于1吗？(#ff8899)）<br/>
目前attention网络包括：基础attention, multi-head attention, self-attention.  
  ## selfAttention
  ## MultiHeadAttention
  ## FlashAttention
  ## PagedAttention
  ## transformer
# Graph Neural Network(GNN)
## GCN
## GraphSage
## GAT

