---
title: 检索方法与工具
date: 2025-03-30
---

# 检索方法与工具

## 目录
1. [检索方法分类及原理](#检索方法分类及原理)
   - [精确检索方法](#精确检索方法)
   - [近似最近邻（ANN）检索方法](#近似最近邻ann检索方法)
   - [索引检索方法](#索引检索方法)
   - [文本检索方法](#文本检索方法)
   - [混合检索方法](#混合检索方法)
2. [常用检索工具](#常用检索工具)
   - [向量检索工具](#向量检索工具)
   - [索引检索工具](#索引检索工具)
   - [综合搜索工具](#综合搜索工具)
   - [文本检索工具](#文本检索工具)
   - [其他新兴工具](#其他新兴工具)
3. [在推荐系统中的应用](#在推荐系统中的应用)
4. [选择建议](#选择建议)

## 检索方法分类及原理

### 精确检索方法
用于小规模数据或需要绝对精确结果的场景。

- **蛮力搜索（Brute Force Search）**: 计算查询与所有数据的距离（如欧氏距离、余弦相似度），复杂度O(n)。
- **KD树（K-Dimensional Tree）**: 基于空间分割的二叉树，适用于低维数据，复杂度O(log n)（低维时）。
- **Ball树**: 用超球分割空间，适合高维数据，复杂度O(log n)。

### 近似最近邻（ANN）检索方法
用于大规模高维向量的快速相似性搜索，推荐系统中常见。

- **局部敏感哈希（LSH）**: 将相似向量映射到相同哈希桶，简单且适合稀疏数据。
- **HNSW（Hierarchical Navigable Small World）**: 多层图结构，高效定位近邻，高精度且低延迟。
- **IVF（Inverted File Index）**: 向量聚类+倒排索引，可扩展，常与量化结合。
- **PQ（Product Quantization）**: 分段量化向量，压缩存储，内存效率高。
- **ScaNN（Scalable Nearest Neighbors）**: 结合量化和各向异性优化，性能优异。

### 索引检索方法
通过键（如Item ID）直接获取数据（如Embedding），无相似性计算。

- **哈希表**: 键值映射，查询复杂度O(1)。
- **倒排索引**: 键到数据的映射，常见于文本和稀疏数据。
- **键值存储**: 磁盘上的高效索引，如LSM树。

### 文本检索方法
基于关键词或语义的搜索，推荐系统中用于过滤或混合召回。

- **倒排索引（Inverted Index）**: 词到文档的映射（如Lucene）。
- **BM25**: 基于词频和文档长度的排名算法。
- **语义搜索**: 使用稀疏/密集向量捕捉语义（结合Transformer模型）。

### 混合检索方法
结合多种技术提升效果。

- **向量+关键词混合**: 如Elasticsearch的RRF（Reciprocal Rank Fusion）。
- **层次召回**: 先用粗糙索引（如IVF）筛选，再用精确方法精排。

## 常用检索工具

### 向量检索工具
专注于高维向量的相似性搜索。

- **Faiss（Facebook AI Similarity Search）**: 支持IVF、HNSW、PQ等多种算法，高性能，支持GPU，广泛用于推荐系统。
- **Annoy（Spotify）**: 支持随机投影树，轻量，适合静态数据。
- **HNSWlib**: 支持HNSW，简单高效，内存占用低。
- **ScaNN（Google）**: 支持量化+优化HNSW，高精度低延迟。
- **NMSLIB（Non-Metric Space Library）**: 支持HNSW、LSH等多种方法，学术性强，灵活性高。

### 索引检索工具
用于键值映射或快速数据访问。

- **Redis**: 内存键值存储，可存Embedding，高并发，低延迟。
- **RocksDB**: 磁盘键值存储，基于LSM树，适合大规模数据，持久化。
- **LevelDB**: 轻量级键值存储，简单高效。
- **HDF5**: 文件存储大数组，适合离线批量访问。

### 综合搜索工具
支持向量、文本和混合检索。

- **Elasticsearch**: 支持HNSW（ANN）、倒排索引、混合搜索，分布式，RESTful API，适合企业级。
- **OpenSearch**: 与Elasticsearch类似（分支项目），开源，社区活跃。
- **Vespa**: 支持HNSW、文本检索、实时更新，高性能，适合推荐和搜索。

### 文本检索工具
专注于关键词或语义搜索。

- **Lucene**: 支持倒排索引，BM25，底层引擎，被ES等集成。
- **Solr**: 基于Lucene的全文搜索，配置丰富，企业友好。

### 其他新兴工具
- **Milvus**: 支持IVF、HNSW、PQ等，分布式向量数据库，易扩展。
- **Pinecone**: 云端ANN服务，无需管理基础设施。

## 在推荐系统中的应用
- **I2I召回**: Faiss（IVF/HNSW）、Elasticsearch（HNSW）用于向量相似性检索。
- **Embedding获取**: Redis、RocksDB用于Item ID到Embedding的索引。
- **过滤候选集**: Elasticsearch（混合搜索）、Lucene（关键词）。
- **实时性**: Redis、Faiss（GPU）满足低延迟需求。
- **大规模**: Milvus、Elasticsearch支持分布式部署。

## 选择建议
- **小规模+高精度**: KD树、哈希表、Lucene。
- **大规模向量检索**: Faiss、HNSWlib、Elasticsearch。
- **实时索引**: Redis、RocksDB。
- **混合需求**: Elasticsearch、Vespa。
