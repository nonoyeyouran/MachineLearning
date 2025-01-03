
# 启发式重排
## MMR
学习文档：https://blog.csdn.net/qq_39388410/article/details/109706683  
![MMR](https://github.com/nonoyeyouran/MachineLearning/blob/main/applications/recommendation/pictures/MMR.png "MMR")  
Q代表query，在推荐中可以理解为用户，sim1即query和document的相关度（推荐中指精排分数），sim2计算documents之间的相似度。
## DPP
学习文档：https://blog.csdn.net/qq_39388410/article/details/109706683

# 重排模型
## DLCM
学习文档：https://zhuanlan.zhihu.com/p/390857478  
要点：  
（1）需要获取精排最后一层的关于user和item的联合表示，这样重排模型的复杂度才会不低于精排【这一点比较重要】  
（2）使用listwise的优化目标达到整个序列上的学习（对比point-wise和pair-wise）的学习。  
（3）使用序列模型学习序列信息。  
## PRM
