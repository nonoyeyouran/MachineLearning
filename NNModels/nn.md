神经网络文档

- [序列建模](#序列建模)
  - [attention网络](#attention网络)
  - [transformer](#transformer)  
- [GraphNeuralNetwork](#GraphNeuralNetwork)
- [神经网络优化](#神经网络优化)
  - [参数调优](#参数调优)
  - [计算加速](#计算加速)
  - [内存优化](#内存优化)

# 序列建模
## attention网络
从attention可以看出，该网络的核心在于模仿人类的学习行为，学习时着重关注相关的信息。因此设计一个网络用于计算（学习）目标A对目标B的关注程度（注意力程度），在0-1之间，然后用该注意力值乘上目标B可以获得目标A从目标B处获得的相关信息；
注意力值越小，表示两者相关度越低，所获取的信息越少。通常被attention的对象B有多个，需要对注意力值进行softmax化。（是否可以不softmax化？全部注意力值之和一定要等于1吗？(#ff8899)）<br/>
目前attention网络包括：基础attention, multi-head attention, self-attention.  
### selfAttention
### MultiHeadAttention
### FlashAttention
### PagedAttention
## transformer
transformer是一种基于multi-head attention的序列学习模型。  
![transformer]()
学习资料：https://zhuanlan.zhihu.com/p/338817680  
# GraphNeuralNetwork
图上的学习有两种基础方式：Inductive learning和Transductive learning，前者训练、测试和验证集分开，互不影响（不同数据集中的节点之间的边被去掉），后者则在全部数据上学习（包括训练集、测试集和验证集）；前者着重用于会有大量新节点产生的情况，后者着重于固定图结构的情况。  
## GCN
基础学习方式：Transductive learning  
## GraphSage
基础学习方式：Inductive learning  
## GAT

# 神经网络优化
## 参数调优
1. 正则化
  L1、L2正则化是常用的调优方法，它可以约束模型的复杂度，避免模型过大导致过拟合。  
2. 优化优化算法  
  机器学习训练中一个重要部分是优化算法，优化算法的改进也可以提升模型效果  
3. 优化目标函数
  目标函数直接关系模型学习什么内容，根据任务优化目标函数也可以提升效果  
4. 超参调整  
  超参可以控制模型、目标函数，例如初始化方式、网络大小和层数，调整超参也可以提升模型效果  
5. 改进网络结构或设计新的网络结构  

