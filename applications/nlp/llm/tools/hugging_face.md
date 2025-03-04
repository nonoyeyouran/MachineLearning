# hugging face使用手册

## 目录
1. [大语言模型使用guide](#大语言模型使用guide)


---


## 大语言模型使用guide
## hugging face的预训练模型离线下载，使用时本地加载
步骤： 
访问 Hugging Face Model Hub：https://huggingface.co/models。   
搜索目标模型（如 gpt2、mistralai/Mixtral-8x7B-Instruct-v0.1）。  
点击模型页面右侧的“Files and versions”标签。  
下载必要文件：  
config.json：模型配置文件。  
pytorch_model.bin 或 model.safetensors：权重文件。  
tokenizer.json / vocab.txt / tokenizer_config.json：分词器文件。  
将文件放入本地文件夹（如 ./mistral_model）。  
离线加载：  
```
python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./mistral_model", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("./mistral_model", local_files_only=True)
```




