- [预训练语言模型PLM](#预训练语言模型PLM)  
- [大模型训练内存优化](#大模型训练内存优化)
    - [gradientCheckpointing](#gradientCheckpointing)  

语言模型文档
# 预训练语言模型PLM
## Bert
bert是google提出的一个预训练语言模型，以transformer的encoder结构为基础构建。  
学习文档：https://zhuanlan.zhihu.com/p/46652512   
bert主要用于一些判别性任务，不能用于生成。  
# 大语言模型（LLM）
## 大语言模型（开源和闭源） 
## 大语言模型架构
## 大语言模型训练
## 大语言模型微调
大模型微调是指将基础训练好的通用大模型在具体的下游任务领域的数据上进一步训练，以适应领域的知识，从而在特定领域表现更佳。  
大语言模型微调包括指令微调、propmt tuning、prefix tuning等  
微调方法：  
1. 指令微调  
    指令微调通常是要大模型学习具体[指令，输出]的方式来限定模型的行为符合人类指令或特定领域指令。
微调工具：
github:https://github.com/zejunwang1/LLMTuner  
## 大语言模型对齐
大模型对齐是指将模型的输出进行限制，通常要符合人类规范（法律、道德等）或领域规范。 
学习文档：  
https://cloud.tencent.com/developer/article/2416650  
https://cloud.tencent.com/developer/news/1265002  
对齐方法：  
1. SFT  
   SFT即监督微调，通过使用人类已经注释好的数据来微调模型使其符合（对齐）人类特定规范。  
2. RLHF  
   RLHF即基于人类反馈的强化学习。在SFT基础上进一步基于人类反馈通过强化学习不断微调。一般采用策略强化学习（policy gradient RL），如PPO。  
3. RRHF  
4. FLAN
5. DPO

对齐工具：  

## 大语言模型应用

# 大模型训练内存优化
## gradientCheckpointing
github:https://github.com/cybertronai/gradient-checkpointing  
