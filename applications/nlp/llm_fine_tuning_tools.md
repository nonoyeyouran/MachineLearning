# 大模型微调工具指南

## 目录
1. [工具概览](#工具概览)
2. [工具详情](#工具详情)
   - [Hugging Face Transformers](#hugging-face-transformers)
   - [OpenAI Fine-tuning API](#openai-fine-tuning-api)
   - [Google Cloud AutoML](#google-cloud-automl)
   - [AWS SageMaker JumpStart](#aws-sagemaker-jumpstart)
   - [Azure Machine Learning](#azure-machine-learning)
   - [DeepSpeed](#deepspeed)
   - [Fairseq](#fairseq)
   - [Megatron-LM](#megatron-lm)
   - [Cohere](#cohere)
   - [Runway ML](#runway-ml)
   - [Lamini](#lamini)
   - [Banana](#banana)
   - [Replicate](#replicate)
   - [Hugging Face PEFT](#hugging-face-peft)
   - [OpenDelta](#opendelta)
   - [Colossal-AI](#colossal-ai)
   - [Weights & Biases (W&B)](#weights--biases-wb)
   - [Ray Tune](#ray-tune)
   - [LLaMA Factory](#llama-factory)
   - [SwanLab](#swanlab)
3. [总结](#总结)

---

## 工具概览
以下是支持大模型微调的工具及其支持的微调方法：

| 工具/平台                | 支持的微调方法                                                                 | 特点                                       | 官网/资源链接                                                                 |
|--------------------------|------------------------------------------------------------------------------|--------------------------------------------|------------------------------------------------------------------------------|
| Hugging Face Transformers | 全参数微调、部分参数微调、LoRA、适配器微调、提示微调、前缀微调                  | 支持多种任务，丰富的预训练模型             | [https://huggingface.co/transformers/](https://huggingface.co/transformers/)  |
| OpenAI Fine-tuning API    | 全参数微调                                                                   | 无需管理基础设施，快速上线                 | [https://platform.openai.com/docs/guides/fine-tuning](https://platform.openai.com/docs/guides/fine-tuning) |
| Google Cloud AutoML       | 全参数微调、部分参数微调                                                     | 无需编写代码，支持多种任务                 | [https://cloud.google.com/automl](https://cloud.google.com/automl)           |
| AWS SageMaker JumpStart   | 全参数微调、部分参数微调                                                     | 提供预训练模型和示例代码                   | [https://aws.amazon.com/sagemaker/jumpstart/](https://aws.amazon.com/sagemaker/jumpstart/) |
| Azure Machine Learning    | 全参数微调、部分参数微调                                                     | 支持多种框架和任务                         | [https://azure.microsoft.com/](https://azure.microsoft.com/)                 |
| DeepSpeed                 | 全参数微调、LoRA、适配器微调、低秩适应                                        | 支持大规模模型的分布式训练                 | [https://www.deepspeed.ai/](https://www.deepspeed.ai/)                       |
| Fairseq                   | 全参数微调、部分参数微调                                                     | 专注于序列生成任务                         | [https://github.com/facebookresearch/fairseq](https://github.com/facebookresearch/fairseq) |
| Megatron-LM               | 全参数微调、部分参数微调                                                     | 支持超大规模模型的训练                     | [https://github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) |
| Cohere                    | 全参数微调、部分参数微调                                                     | 提供简单的API和界面                        | [https://cohere.ai/](https://cohere.ai/)                                     |
| Runway ML                 | 全参数微调、部分参数微调                                                     | 支持创意任务，用户友好                     | [https://runwayml.com/](https://runwayml.com/)                               |
| Lamini                    | 全参数微调、部分参数微调                                                     | 专注于语言模型微调                         | [https://lamini.ai/](https://lamini.ai/)                                     |
| Banana                    | 全参数微调、部分参数微调                                                     | 支持多种任务的模型微调                     | [https://www.banana.dev/](https://www.banana.dev/)                           |
| Replicate                 | 全参数微调、部分参数微调                                                     | 支持多种开源模型的微调                     | [https://replicate.com/](https://replicate.com/)                             |
| Hugging Face PEFT         | LoRA、适配器微调、提示微调、前缀微调                                           | 参数高效微调，支持多种方法                 | [https://github.com/huggingface/peft](https://github.com/huggingface/peft)   |
| OpenDelta                 | LoRA、适配器微调、提示微调、前缀微调                                           | 提供多种参数高效微调方法                   | [https://github.com/thunlp/OpenDelta](https://github.com/thunlp/OpenDelta)   |
| Colossal-AI               | LoRA、适配器微调、提示微调、前缀微调                                           | 支持大规模模型的分布式训练                 | [https://www.colossalai.org/](https://www.colossalai.org/)                   |
| Weights & Biases (W&B)    | 实验跟踪和超参数优化（支持全参数微调、部分参数微调等）                         | 提供实验跟踪和模型管理功能                 | [https://wandb.ai/](https://wandb.ai/)                                       |
| Ray Tune                  | 超参数优化（支持全参数微调、部分参数微调等）                                   | 提供高效的分布式超参数搜索                 | [https://docs.ray.io/en/latest/tune/](https://docs.ray.io/en/latest/tune/)   |
| LLaMA Factory             | 全参数微调、LoRA、适配器微调、提示微调、前缀微调                               | 支持100+大模型，提供用户友好的WebUI        | [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) |
| SwanLab                   | 实验跟踪和可视化（支持全参数微调、部分参数微调等）                             | 提供实验监控和日志记录功能                 | [https://swanlab.cn](https://swanlab.cn)                                     |

---

## 工具详情

### Hugging Face Transformers
- **支持的微调方法：** 全参数微调、部分参数微调、LoRA、适配器微调、提示微调、前缀微调。
- **特点：** 支持多种任务，丰富的预训练模型。
- **官网：** [https://huggingface.co/transformers/](https://huggingface.co/transformers/)

### OpenAI Fine-tuning API
- **支持的微调方法：** 全参数微调。
- **特点：** 无需管理基础设施，快速上线。
- **官网：** [https://platform.openai.com/docs/guides/fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)

### Google Cloud AutoML
- **支持的微调方法：** 全参数微调、部分参数微调。
- **特点：** 无需编写代码，支持多种任务。
- **官网：** [https://cloud.google.com/automl](https://cloud.google.com/automl)

### AWS SageMaker JumpStart
- **支持的微调方法：** 全参数微调、部分参数微调。
- **特点：** 提供预训练模型和示例代码。
- **官网：** [https://aws.amazon.com/sagemaker/jumpstart/](https://aws.amazon.com/sagemaker/jumpstart/)

### Azure Machine Learning
- **支持的微调方法：** 全参数微调、部分参数微调。
- **特点：** 支持多种框架和任务。
- **官网：** [https://azure.microsoft.com/](https://azure.microsoft.com/)

### DeepSpeed
- **支持的微调方法：** 全参数微调、LoRA、适配器微调、低秩适应。
- **特点：** 支持大规模模型的分布式训练。
- **官网：** [https://www.deepspeed.ai/](https://www.deepspeed.ai/)

### Fairseq
- **支持的微调方法：** 全参数微调、部分参数微调。
- **特点：** 专注于序列生成任务。
- **官网：** [https://github.com/facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)

### Megatron-LM
- **支持的微调方法：** 全参数微调、部分参数微调。
- **特点：** 支持超大规模模型的训练。
- **官网：** [https://github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

### Cohere
- **支持的微调方法：** 全参数微调、部分参数微调。
- **特点：** 提供简单的API和界面。
- **官网：** [https://cohere.ai/](https://cohere.ai/)

### Runway ML
- **支持的微调方法：** 全参数微调、部分参数微调。
- **特点：** 支持创意任务，用户友好。
- **官网：** [https://runwayml.com/](https://runwayml.com/)

### Lamini
- **支持的微调方法：** 全参数微调、部分参数微调。
- **特点：** 专注于语言模型微调。
- **官网：** [https://lamini.ai/](https://lamini.ai/)

### Banana
- **支持的微调方法：** 全参数微调、部分参数微调。
- **特点：** 支持多种任务的模型微调。
- **官网：** [https://www.banana.dev/](https://www.banana.dev/)

### Replicate
- **支持的微调方法：** 全参数微调、部分参数微调。
- **特点：** 支持多种开源模型的微调。
- **官网：** [https://replicate.com/](https://replicate.com/)

### Hugging Face PEFT
- **支持的微调方法：** LoRA、适配器微调、提示微调、前缀微调。
- **特点：** 参数高效微调，支持多种方法。
- **官网：** [https://github.com/huggingface/peft](https://github.com/huggingface/peft)

### OpenDelta
- **支持的微调方法：** LoRA、适配器微调、提示微调、前缀微调。
- **特点：** 提供多种参数高效微调方法。
- **官网：** [https://github.com/thunlp/OpenDelta](https://github.com/thunlp/OpenDelta)

### Colossal-AI
- **支持的微调方法：** LoRA、适配器微调、提示微调、前缀微调。
- **特点：** 支持大规模模型的分布式训练。
- **官网：** [https://www.colossalai.org/](https://www.colossalai.org/)

### Weights & Biases (W&B)
- **支持的微调方法：** 实验跟踪和超参数优化（支持全参数微调、部分参数微调等）。
- **特点：** 提供实验跟踪和模型管理功能。
- **官网：** [https://wandb.ai/](https://wandb.ai/)

### Ray Tune
- **支持的微调方法：** 超参数优化（支持全参数微调、部分参数微调等）。
- **特点：** 提供高效的分布式超参数搜索。
- **官网：** [https://docs.ray.io/en/latest/tune/](https://docs.ray.io/en/latest/tune/)

### LLaMA Factory
- **支持的微调方法：** 全参数微调、LoRA、适配器微调、提示微调、前缀微调。
- **特点：** 支持100+大模型，提供用户友好的WebUI。
- **官网：** [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

### SwanLab
- **支持的微调方法：** 实验跟踪和可视化（支持全参数微调、部分参数微调等）。
- **特点：** 提供实验监控和日志记录功能。
- **官网：** [https://swanlab.cn](https://swanlab.cn)

---

## 总结
- **全参数微调**：几乎所有工具都支持。
- **部分参数微调**：Hugging Face、DeepSpeed、Fairseq 等支持。
- **LoRA**：Hugging Face PEFT、DeepSpeed、OpenDelta、Colossal-AI 等支持。
- **适配器微调**：Hugging Face PEFT、OpenDelta 等支持。
- **提示微调**：Hugging Face PEFT、OpenDelta 等支持。
- **前缀微调**：Hugging Face PEFT、OpenDelta 等支持。

---
