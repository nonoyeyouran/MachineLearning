# 推荐排序相关模型文档
---
- [LLMBased推荐](#LLMBased推荐)<br/>
- [用户行为序列建模](#用户行为序列建模)<br/>
  - [显示地利用用户的正负反馈](#显示地利用用户的正负反馈)<br/>
- [多目标排序](#多目标排序)<br/>

---

## 1. LLMBased推荐
 
  ### 1.1 generative recommender
  论文：《Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations》，2024.2.<br/>
  论文模型代码：https://github.com/facebookresearch/generative-recommenders

---
## 2. 用户行为序列建模
### 2.1 显示地利用用户的正负反馈
在任务的目标函数设计中，显示地增加用户对于正负反馈的不同偏好，例如：将用户向量分别于正负序列做相似度计算，在目标函数中最大化用户向量与正序列的相似度，最小化与负序列的相似度，从而显示学习正负反馈。item只需与user的向量计算相似度即可（与user越相似，则与正反馈序列越相似）
### 2.2 DIN
**论文：** 《Deep Interest Network for Click-Through Rate Prediction》  
**文档：** https://zhuanlan.zhihu.com/p/429433768  
**要点：**  
- 使用target-attention的方式来建模用户行为序列  
- attention时，对于key的权重α不进行归一化，保留兴趣强度  
- attention的输入是query，key和query和key的外积，外积用于获取query和key的交叉特征  
- attention经过DNN网络计算得到权重α，直接乘以key（key=value）得到attention后的表示  
- MBA：mini-batch Aware Regularization，只对在batch样本中出现过的ID类特征进行L2正则
- 设计了一种新的激活函数Data Adaptive Activation Function（Dice），网络送入激活函数前的值的分布不一定是以0为中心的分布，所以需要根据batch内数据进行估计和矫正  
## 3. 多目标排序
多目标学习即同时对多个目标建模，但是底部会不同程度地分享信息，从而达到共同进步的目的，也就是相互辅助学习。当前多目标学习架构主要又share-bottom、MOE(mixtureof expert)、MMOE（multi-gate mixture of expert）、CGC（Customized Gate Control Model）、ESMM（针对具有顺序性的任务）。
### 3.1 ESMM
论文：《Entire Space Multi-Task Model: An Eﬀective Approach for Estimating Post-Click Conversion Rate》，2018，阿里。  
参考文档：https://zhuanlan.zhihu.com/p/57481330   
模型特点：该模型针对具有前后因果逻辑的多个目标任务，例如论文中所诉的ctr和cvr任务，通常在点击之后才会进入是否转化的过程。  
esmm主要针对cvr目前存在的两个问题进行优化:   
(1) cvr任务在点击空间学习，假设了用户会点击，但事实我们在推荐给用户数据时用户的点击行为还没有发生，是未知的  
(2) cvr任务的样本相对ctr任务要小很多，ctr的任务可以帮助cvr的学习（借助多任务学习）  
为解决上诉问题（1），把转化问题放到跟ctr同一个样本空间即曝光空间；在曝光空间，ctr任务数据服从p(click|X)分布，cvr任务数据服从p(click,conversation|X)这个联合概率分布；现在我们要获取P(conversation|click;X)这个条件概率分布，有两种方法可以学习，一种是直接对P(conversation|click;X)建模（类似判别模型），认为曝光样本的点击概率相同，此时条件概率分布和联合概率分布等价，另一种是根据贝叶斯法则P(conversation|click;X) = p(click,conversation|X) / p(click|X)（类似生成模型）间接地学习条件概率分布，认为曝光样本点击与否服从一个概率分布；对于第二种基于贝叶斯方法来学习也对应两种建模方式，一种是分开建模联合概率分布p(click,conversation|X)和边缘概率分布p(click|X)，论文指出分开建模相除会有数值不稳定问题，第二种方式就是结组多目标模型的结构一起学习联合概率分布p(click,conversation|X)和边缘概率分布p(click|X)，即论文采用的方式ESMM。基于多任务学习框架，共享底层特征，来使得两个任务相互辅助学习，缓解了问题（2）。  
  






















  
