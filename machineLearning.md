机器学习基础知识文档

# 监督学习
## 生成模型与判别模型
监督学习模型的一般形式是决策函数Y=f(X)或条件概率分布P(Y|X)，对于条件概率的学习有两种方式，一种是直接学习条件概率或者决策函数，称为“判别模型”；一种是学习联合概率分布P(X,Y)和边缘概率分布P(X)，然后通过贝叶斯公式P(Y|X) = P(X,Y) / P(X)得到条件概率分布，称为“生成模型”。  
（1）判别模型  
（2）生成模型  
  可以还原联合概率分布；学习收敛速度快（当样本容量增加时，可以更快收敛于真是模型）；可以用于存在隐变量的任务。
# 损失函数
## InfoNCE
学习文档：https://zhuanlan.zhihu.com/p/334772391  
要点：  
（1）基于NCE思想  
（2）用于1个正例和多个负例的对比学习

