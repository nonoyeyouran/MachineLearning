这边文档是关于近似推断的方法。

- [基于数值采样的近似推断](#基于数值采样的近似推断)
  - [rejectionSampling](#rejectionSampling) <br/>
  - [importanceSampling](#importanceSampling) <br/>

# 基于数值采样的近似推断
## rejectionSampling
rejectionSampling即拒绝采样，是针对特别复杂的分布f, 无法通过计算反函数的方式来进行随机采样；通常用于单变量分布随机采样。该方法需要了解目标分布，否则很难找到一个提议分布。  
文档：https://blog.csdn.net/qq_35939846/article/details/132831871  
1. 一般拒绝采样  
   步骤：
   （1）选择一个合适的提议分布q(x)。  
   （2）从提起分布q(x)采样x1得到y_q，同时把x1带入目标的概率密度函数p(x)得到y_p  
   （3）从均匀分布[0, 1]上采样得到u。  
   （4）计算y_p/y_q，如果大于u，则接受x1和y_p，否则拒绝。  
3. 自适应拒绝采样  

## importanceSampling
重要性采样，直接针对某个函数g在目标分布f上的期望进行近似计算。通常用于单变量随机分布。


