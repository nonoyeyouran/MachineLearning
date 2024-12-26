推荐排序相关模型文档

- [LLMBased推荐](#LLMBased推荐)<br/>
- [用户行为序列建模](#用户行为序列建模)<br/>
  - [显示地利用用户的正负反馈](#显示地利用用户的正负反馈)<br/>


# LLMBased推荐
  ## 综述
  
  ## generative recommender
  论文：《Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations》，2024.2.<br/>
  论文模型代码：https://github.com/facebookresearch/generative-recommenders

# 用户行为序列建模
  ## 显示地利用用户的正负反馈
      在任务的目标函数设计中，显示地增加用户对于正负反馈的不同偏好，例如：将用户向量分别于正负序列做相似度计算，在目标函数中最大化用户向量与正序列的相似度，最小化与负序列的相似度，从而显示学习正负反馈。
      item只需与user的向量计算相似度即可（与user越相似，则与正反馈序列越相似）
