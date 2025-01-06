# seq2seq是一类模型，一般用于NLP任务中翻译、文本生成（自动摘要）等任务

---

## 一、Seq2seq
### 1.1 PointerGeneratorNetwork
**论文：** 《Get To The Point: Summarization with Pointer-Generator Networks》, 2017.ACL. <br/>
**参考理解文档:** https://zhuanlan.zhihu.com/p/53821581 <br/>
**github:** https://github.com/abisee/pointer-generator  
**要点：**  
- 指针生成网络是seq2seq的一种变种，在其基础上增加了attention和pointerGenerator network优化模型，既可以从原文copy，也可以从词表新生成  
- 为了消除或减弱词语重复现象，论文引入了一个新的损失函数，用于限制重复词语的出现
