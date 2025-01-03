推荐排序相关模型文档

- [LLMBased推荐](#LLMBased推荐)<br/>
- [用户行为序列建模](#用户行为序列建模)<br/>
  - [显示地利用用户的正负反馈](#显示地利用用户的正负反馈)<br/>
- [多目标排序](#多目标排序)<br/>


# LLMBased推荐
  ## 综述
  
  ## generative recommender
  论文：《Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations》，2024.2.<br/>
  论文模型代码：https://github.com/facebookresearch/generative-recommenders

# 用户行为序列建模
  ## 显示地利用用户的正负反馈
      在任务的目标函数设计中，显示地增加用户对于正负反馈的不同偏好，例如：将用户向量分别于正负序列做相似度计算，在目标函数中最大化用户向量与正序列的相似度，最小化与负序列的相似度，从而显示学习正负反馈。
      item只需与user的向量计算相似度即可（与user越相似，则与正反馈序列越相似）

# 多目标排序
  ## ESMM
  论文：《Entire Space Multi-Task Model: An Eﬀective Approach for Estimating Post-Click Conversion Rate》，2018，阿里。<br/>
  参考文档：https://zhuanlan.zhihu.com/p/57481330  
  模型特点：该模型针对具有前后因果逻辑的多个目标任务，例如论文中所诉的ctr和cvr任务，通常在点击之后才会进入是否转化的过程。<br/>
  esmm主要针对cvr目前存在的两个问题进行优化: <br/>
  (1) cvr任务在点击空间学习，假设了用户会点击，但事实我们在推荐给用户数据时用户的点击行为还没有发生，是未知的<br/>
  (2) cvr任务的样本相对ctr任务要小很多，ctr的任务可以帮助cvr的学习（借助多任务学习）<br/>
  为解决上诉问题（1），把转化问题放到跟ctr同一个样本空间即曝光空间；在曝光空间，ctr任务数据服从p(click|X)分布，cvr任务数据服从p(click,conversation|X)这个联合概率分布；现在我们要获取P(conversation|click;X)这个条件概率分布，有两种方法可以学习，一种是直接对P(conversation|click;X)建模（类似判别模型），认为曝光样本的点击概率相同，此时条件概率分布和联合概率分布等价，另一种是根据贝叶斯法则P(conversation|click;X) = p(click,conversation|X) / p(click|X)（类似生成模型）间接地学习条件概率分布，认为曝光样本点击与否服从一个概率分布；论文采用了第二种方式学习。<br/>
  论文采用了生成模型，同时基于多任务学习框架，共享底层特征，来使得两个任务相互辅助学习。<br/>
  






















  
