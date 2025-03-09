# hugging face使用手册

## 目录
1. [大语言模型使用guide](#大语言模型使用guide)


---


## 大语言模型使用guide
## hugging face的预训练模型离线下载，使用时本地加载
步骤:  
1. 访问 Hugging Face Model Hub：https://huggingface.co/models。   
2. 搜索目标模型（如 gpt2、mistralai/Mixtral-8x7B-Instruct-v0.1）。  
3. 点击模型页面右侧的“Files and versions”标签。  
4. 下载必要文件：  
- config.json：模型配置文件。  
- pytorch_model.bin 或 model.safetensors：权重文件。  
- tokenizer.json / vocab.txt / tokenizer_config.json：分词器文件。  
5. 将文件放入本地文件夹（如 ./mistral_model）。  
6. 离线加载：  
```
python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./mistral_model", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("./mistral_model", local_files_only=True)
```
## 微调
### 全量微调
全量微调是指在新的数据集上接着训练模型，所有参数都参与训练。一个全量微调的例子如下：
```
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.memory.set_per_process_memory_fraction(1.0)  # Optional: Use full GPU memory

# 1. 加载数据集
print("加载数据集")
data_files = {
    'train': 'D:\\MachineLearning\\datesets\\imdb\\train-00000-of-00001.parquet',
    'test': 'D:\\MachineLearning\\datesets\\imdb\\test-00000-of-00001.parquet'
}
dataset = load_dataset('parquet', data_files=data_files)  # 使用 IMDb 数据集作为示例

# 2. 加载预训练的 BERT 模型和分词器
print("加载预训练的 BERT 模型和分词器")
model_name = "D:\\MachineLearning\\pretrain_models_hugging_face\\bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
#model = AutoModelForCausalLM.from_pretrained("D:\\MachineLearning\\pretrain_models_hugging_face\\bert-base-uncased", local_files_only=True)
#tokenizer = AutoTokenizer.from_pretrained("D:\\MachineLearning\\pretrain_models_hugging_face\\bert-base-uncased", local_files_only=True, device_map="auto")

# 3. 数据预处理
print("数据预处理")
def preprocess_function(examples):
    # 对文本进行分词处理
    tokenized_inputs = tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=256
    )
    # 将标签也添加到结果中
    tokenized_inputs['labels'] = examples['label']  # 假设标签的键为 'label'
    return tokenized_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, batch_size=4)
print(tokenized_dataset["train"].shape)

# 4. 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",          # 输出目录
    evaluation_strategy="epoch",     # 每个 epoch 评估一次
    learning_rate=2e-5,              # 学习率
    per_device_train_batch_size=4,   # 训练批次大小
    per_device_eval_batch_size=4,    # 评估批次大小
    num_train_epochs=3,              # 训练轮数
    weight_decay=0.01,               # 权重衰减
)

# 5. 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

torch.cuda.empty_cache()  # Releases cached memory PyTorch reserved but isn’t using
print(torch.cuda.memory_allocated() / 1024**3, "GB")  # 当前分配
print(torch.cuda.memory_reserved() / 1024**3, "GB")   # 预留内存
# 6. 开始训练
trainer.train()

# 7. 评估模型
trainer.evaluate()
```





