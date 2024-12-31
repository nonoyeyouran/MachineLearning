这边文档是关于近似推断的方法。



- [基于数值采样的近似推断](#基于数值采样的近似推断)
  - [rejectionSampling](#rejectionSampling) <br/>
  - [importanceSampling](#importanceSampling) <br/>

# 基于数值采样的近似推断
## rejectionSampling
rejectionSampling即拒绝采样，是针对特别复杂的分布f, 无法用公式表示，通常用于单变量分布随机采样。该方法需要了解目标分布，否则很难找到一个提议分布。<br/>
文档：https://blog.csdn.net/qq_35939846/article/details/132831871 <br/>
（1）一般拒绝采样 <br/>
（2）自适应拒绝采样 <br/>

## importanceSampling
重要性采样，直接针对某个函数g在目标分布f上的期望进行近似计算。通常用于单变量随机分布。


