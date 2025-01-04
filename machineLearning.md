机器学习基础知识文档
- [基础概念](#基础概念)
  - [监督学习](#监督学习)  

# 基础概念
## 评价指标
T: True, F: False, P: positive, N: Negative  
TP: 正类预测为正类  
FP: 负类预测为正类  
TN: 负类预测为负类  
FN: 正类预测为负类  
（1）精确率（precision rate）  
precision = TP / (TP + FP)，针对正类定义的（通常我们关注正类的指标）  
（2）召回率（recall rate）  
recall = TP / (TP + FN)，也是针对正类定义的   
（3）准确率（accuracy rate）  
accuracy = (TP + TN) / all，反应数据整体预测的效果  
（4）F1-Score  
F1-Score = 2 * precision * recall / (precision + recall)  

## 监督学习
### 生成模型与判别模型
监督学习模型的一般形式是决策函数Y=f(X)或条件概率分布P(Y|X)，对于条件概率的学习有两种方式，一种是直接学习条件概率或者决策函数，称为“判别模型”；一种是学习联合概率分布P(X,Y)和边缘概率分布P(X)，然后通过贝叶斯公式P(Y|X) = P(X,Y) / P(X)得到条件概率分布，称为“生成模型”。  
（1）判别模型  
（2）生成模型  
  可以还原联合概率分布；学习收敛速度快（当样本容量增加时，可以更快收敛于真是模型）；可以用于存在隐变量的任务。
## 损失函数

## InfoNCE
学习文档：https://zhuanlan.zhihu.com/p/334772391  
要点：  
（1）基于NCE思想  
（2）用于1个正例和多个负例的对比学习  
个人理解：InfoNCE其实就是就是softmax。在U2I模型中（推荐召回算法），我们通常需要使用user的embedding取检索item集合的embeddings来进行相似召回。如果对U2I的建模中，我们最后的loss计算采用softmax的方式，这要求我们直接使用item的embedding参与最后的softmax计算，通常我们会初始化所有item的embedding，即此时items的表示矩阵已经存在，例如YoutubeDNN、MIND等模型；但在DSSM模型中，item及相关特征会被送入了一个神经网络，神经网络的输出是item的最终表示，记作embedding_nn(表示通过神经网络获得的表示)，此时我们无法使用user的embedding和所有item的embedding_nn来进行softmax计算，因为我无法获取所有item的embedding_nn；但是我们可以直接计算user的embedding和item的embedding_nn的相似度，只是这种方式相较于softmax要弱，因此选择了一种近似计算方法，在一个batch内的item可以获取embedding_nn，在batch大小上做softmax计算来近似。  


