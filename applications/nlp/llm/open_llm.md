# 当前主流开源大模型概览

截至2025年4月6日，开源大模型分为**通用大模型**和**专用大模型**两大类。通用模型支持广泛任务，专用模型针对特定领域优化。本文档列出主流代表及其特点。

## 目录
- [开源通用大模型](#开源通用大模型)
  - [LLaMA 系列（Meta AI）](#llama-系列meta-ai)
  - [Mistral 系列（Mistral AI）](#mistral-系列mistral-ai)
  - [Qwen 系列（阿里巴巴）](#qwen-系列阿里巴巴)
  - [Falcon 系列（TII）](#falcon-系列tii)
  - [Gemma 系列（Google）](#gemma-系列google)
  - [BLOOM（BigScience）](#bloombigscience)
- [开源专用大模型](#开源专用大模型)
  - [CodeLLaMA（Meta AI）](#codellamameta-ai)
  - [DeepSeek-Coder（DeepSeek AI）](#deepseek-coderdeepseek-ai)
  - [StarCoder 2（BigCode）](#starcoder-2bigcode)
  - [MathFormer（社区驱动）](#mathformer社区驱动)
  - [BioMedLM（Stanford & MosaicML）](#biomedlmstanford--mosaicml)
  - [Phi-3（Microsoft）](#phi-3microsoft)
  - [Stable Diffusion（Stability AI）](#stable-diffusionstability-ai)
- [总结与对比](#总结与对比)

---

## 开源通用大模型

通用大模型设计灵活，支持对话、生成、翻译等多种任务。

### LLaMA 系列（Meta AI）
- **参数规模**：8B、70B、405B（LLaMA 3.1）。
- **特点**：
  - 128K上下文长度，多语言支持（8+种）。
  - 在对话、推理、内容生成中表现优秀。
- **用途**：通用对话、问答、研究。
- **许可证**：Apache 2.0（部分版本需申请）。

### Mistral 系列（Mistral AI）
- **参数规模**：7B（Mistral 7B）、141B（Mixtral 8x22B，MoE）。
- **特点**：
  - Mixtral 使用稀疏专家模型，高效且性能强劲。
  - 支持80+语言，128K上下文。
- **用途**：对话、文本生成、企业应用。
- **许可证**：Apache 2.0。

### Qwen 系列（阿里巴巴）
- **参数规模**：1.8B至72B（Qwen2.5）。
- **特点**：
  - 支持29+种语言，128K上下文。
  - 在对话、推理、结构化输出（如JSON）中表现优异。
- **用途**：通用问答、教育、客户服务。
- **许可证**：Apache 2.0。

### Falcon 系列（TII）
- **参数规模**：180B（Falcon 180B）。
- **特点**：
  - 训练于3.5T token，多语言能力强。
  - 适用于大规模文本生成和翻译。
- **用途**：对话、翻译、研究。
- **许可证**：Apache 2.0。

### Gemma 系列（Google）
- **参数规模**：9B、27B（Gemma 2）。
- **特点**：
  - 轻量高效，8K上下文。
  - 基于Gemini技术，通用性强。
- **用途**：个人助理、教育、轻量部署。
- **许可证**：Apache 2.0。

### BLOOM（BigScience）
- **参数规模**：176B。
- **特点**：
  - 支持46种语言和13种编程语言。
  - 注重透明性和多语言能力。
- **用途**：跨语言对话、研究。
- **许可证**：BigScience Open RAIL-M。

---

## 开源专用大模型

专用大模型针对特定领域或任务优化，专业性更强。

### CodeLLaMA（Meta AI）
- **参数规模**：7B、13B、34B。
- **特点**：
  - 基于LLaMA优化，专为代码生成和补全设计。
  - 支持主流编程语言（如Python、Java、C++）。
- **用途**：代码编写、调试、自动化开发。
- **许可证**：Apache 2.0。

### DeepSeek-Coder（DeepSeek AI）
- **参数规模**：1.3B至33B。
- **特点**：
  - 专注于代码生成和数学推理。
  - 支持20+种编程语言，128K上下文。
- **用途**：编程助手、技术文档生成。
- **许可证**：MIT。

### StarCoder 2（BigCode）
- **参数规模**：3B至15B。
- **特点**：
  - 专为代码生成设计，支持80+种编程语言。
  - 训练数据包括开源代码库，生成质量高。
- **用途**：软件开发、代码审查。
- **许可证**：BigCode Open RAIL-M。

### MathFormer（社区驱动）
- **参数规模**：6B至66B。
- **特点**：
  - 专为数学问题求解和公式生成优化。
  - 支持LaTeX输出和复杂推理。
- **用途**：教育、科研、数学建模。
- **许可证**：Apache 2.0。

### BioMedLM（Stanford & MosaicML）
- **参数规模**：2.7B。
- **特点**：
  - 针对生物医学领域，训练于PubMed等专业数据。
  - 在医学问答、文献分析中表现优异。
- **用途**：医疗研究、健康咨询。
- **许可证**：Apache 2.0。

### Phi-3（Microsoft）
- **参数规模**：3.8B。
- **特点**：
  - 小型语言模型（SLM），专为高效推理和本地部署。
  - 在对话和简单任务中表现良好。
- **用途**：边缘设备、嵌入式系统。
- **许可证**：MIT。

### Stable Diffusion（Stability AI）
- **参数规模**：约10B（扩散模型）。
- **特点**：
  - 专用图像生成模型，开源权重和代码。
  - 支持文本到图像生成，广泛应用于艺术创作。
- **用途**：图像生成、设计、艺术。
- **许可证**：CreativeML Open RAIL-M。

---

## 总结与对比

- **通用大模型**：
  - 代表：LLaMA、Mistral、Qwen。
  - 优势：灵活性高，支持多任务，上下文长度和多语言能力突出。
- **专用大模型**：
  - 代表：CodeLLaMA、DeepSeek-Coder、BioMedLM。
  - 优势：在特定领域（如代码、数学、医学）深度优化，专业性强，但通用性较弱。
- **趋势**：
  - 2025年，开源模型规模和性能逼近闭源模型（如GPT-4、Claude）。
  - 通用模型追求更大参数和长上下文，专用模型聚焦效率和领域精准性。

选择模型时，需根据任务需求（如领域专精、计算资源、语言支持）和开源许可灵活性进行评估。开源社区的快速发展使其在研究和商业应用中日益普及。
