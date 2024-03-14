import sys
import os
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#加载训练数据集
#若是文件直接处理为list，其中每个元素都是一个行为序列（或切词后的一个句子）
raw_sentences = ["the quick brown fox jumps over the lazy dogs","yoyoyo you go home now to sleep"]
sentences= [s.encode('utf-8').split() for s in sentences]

#训练模型
#min_count用于过滤频次小于该值的token
#参数说明：https://blog.csdn.net/szlcw1/article/details/52751314
model = word2vec.Word2Vec(sentences, min_count=5)

#模型保存
model.save("embedding.model")
model.save_word2vec_format('embedding.model.bin', binary=True)

#模型加载
model_ = word2vec.Word2Vec.load("embedding.model")
model_bin = word2vec.Word2Vec.load_word2vec_format('embedding.model.bin', binary=True)

# 计算相似度
sim_score = model.similarity("word1", "word2")
sim_word_list = model.most_similar(["word3"])

# 获取embedding
word4_embedding = model["word4"] # 是个numpy的list，可以转存为文件
