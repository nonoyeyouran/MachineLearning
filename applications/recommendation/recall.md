- [召回](#召回)<br/>
    - [召回算法](#召回算法)<br/>
        - [基于属性的召回](#基于属性的召回)<br/>
        - [基于统计的召回](#基于统计的召回)<br/>
        - [基于机器学习模型的召回](#基于机器学习模型的召回)<br/>
            - [I2I类](#I2I类)<br/>
    - [召回评估指标](#召回评估指标)<br/>
    - [召回常见问题](#召回常见问题)<br/>
    
# 召回
## 召回算法
召回算法可以分成三大类，分别是基于属性的召回、基于统计的召回和基于机器学习模型的召回
### 基于属性的召回
包括热度召回、用户画像属性召回、用户人口学属性召回。这一层召回的粒度很粗，召回量很大。
### 基于统计的召回
包括协同过滤（item-base、user-base、与模型结合）、矩阵分解
### 基于机器学习模型的召回
基本是利用用户属性特征和行为特征来召回，包括I2I类（主要是学习item的embedding，一种是根据item本身的信息学习【适合项目冷启动】，另一种是基于用户行为学习）、U2I类（直接利用用户特征和item特征建模）、U2U2I类等
#### I2I类
这一类模型关键在于item embedding的学习，一种是基于item本身的内容学习表示；一种是基于用户行为学习表示；第三种是混合学习（一般在图模型中）。目前主要的学习方式是word2vec和graphEmbedding的方法。  
__（1）word2vec__  
#### U2I类  
（1）YoutubeDNN  
（2）DSSM  
（3）SDM  
（4）多兴趣召回：MIND、ComiRec  
MIND论文：《Multi-Interest Network with Dynamic Routing for Recommendation at Tmall》, 2019  
学习文档：https://zhuanlan.zhihu.com/p/100779249  
要点：  
（1）通过动态路由算法实现单条数据中对用户行为序列进行兴趣聚类【单靠这个算法无法真正学习聚类，基础依赖的仍然是整个数据集所有用户的行为序列中item的共现】  
（2）target-item和兴趣向量的attention帮助着重于item所属兴趣上学习  
可能的改进点： 
（1）因为聚类是隐式学习的，依赖用户行为序列中item的共现和用户真实兴趣分布，模型无法保证所学的多个兴趣互不相关[假设真实的不同兴趣是不相关的]，因此可以在目标函数中对兴趣向量施加约束，让各个兴趣向量之间尽可能正交。  

ComiRec论文：《Controllable Multi-Interest Framework for Recommendation》，2019  
学习文档：https://zhuanlan.zhihu.com/p/568781562  
要点：  
（1）提出了另外一个基于multi-head attention的多兴趣提取模块【还有一个就是MIND里面的动态路由，不同之处在于兴趣数K的设置】  
（2）在线服务时不仅考虑准确性，还考虑多样性，提出了一个综合的线上指标，并使用一个贪心算法来具体实现。  

（5）__长短期兴趣召回：SDM__
长期兴趣代表用户过去存在但是近期不太表现的兴趣，短期兴趣代表用户近期的兴趣。两个结合召回可以增加召回的多样性，避免过于集中于近期的兴趣导致越来越同质化。  
SDM论文：《SDM: Sequential Deep Mmatching Model for Online Large-scale Recommender Systemm》  
学习文档：https://zhuanlan.zhihu.com/p/137775247  
要点：  
（1）长期兴趣使用用户近期之前的行为数据；在item的不同属性上进行更高的抽象。
（2）使用用户embedding对不同属性（兴趣）向量进行attention，找到用户侧重的属性  
（3）使用gate机制融合长短期兴趣  
    
## 召回评估指标
## 召回常见问题
