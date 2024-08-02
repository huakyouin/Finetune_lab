# 项目名称: Finetune_lab

## 项目简介
本项目旨在通过LoRA（Low-Rank Adaptation）技术对不同的语言模型进行微调。项目包含数据集、模型、输出、设置和工具等文件夹。

## 目录结构

- **datasets**: 存放训练和测试数据集。
- **base**: 存放从别处下载的模型底座。
- **models**: 保存自定义的模型网络。
- **outputs**: 存放训练输出结果，包括日志和生成的模型。
- **examples**: 训练以及调用实例。
- **utils**: 工具文件夹，包含数据处理和辅助脚本。
- **README.md**: 项目说明文件。
- **sft.py**: 对大模型进行sft的脚本，便于不同超参训练。
- **settings**: 存放配合train文件使用的JSON配置文件，相当于记录不同配方。

## 使用方法

### 准备环境
1. 克隆项目到本地并让终端进入该文件夹

2. 安装所需依赖:

   对于cuda<11.4：
   ```bash
    conda create -n sft python=3.10 notebook
    activate sft
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3
    pip install transformers accelerate trl bitsandbytes==0.39 scipy deepspeed hf_transfer modelscope 
    pip install transformers_stream_generator tiktoken
    pip install peft --no-dependencies 
   ```
   Note: 需要把bitsandbytes库中的__init__文件中if("usr/local")块注释掉

   对于更高版本：
   ```bash
    conda create -n sft python=3.10 notebook
    activate sft
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    export PIPR='https://mirrors.aliyun.com/pypi/simple/'
    pip install transformers accelerate trl bitsandbytes deepspeed  peft transformers_stream_generator tiktoken
    pip install scikit-learn pandas numpy matplotlib 
   ```

### 数据与模型准备

#### 从hugging face下载

下载数据集示例如下：
```bash
    pip install hf_transfer
    export HF_HUB_ENABLE_HF_TRANSFER=1
    huggingface-cli download --repo-type dataset --resume-download Maciel/FinCUGE-Instruction  --local-dir data/findata --local-dir-use-symlinks False
    huggingface-cli download --repo-type dataset --resume-download silk-road/alpaca-data-gpt4-chinese  --local-dir data/gpt4data --local-dir-use-symlinks False
```
下载模型：
```bash
    export HF_HUB_ENABLE_HF_TRANSFER=1
    huggingface-cli download --repo-type model --resume-download google-bert/bert-base-chinese  --local-dir base/bert --local-dir-use-symlinks False
    huggingface-cli download --repo-type model --resume-download BAAI/bge-small-zh-v1.5  --local-dir base/bge_small --local-dir-use-symlinks False
```


#### 从魔搭社区下载

魔搭社区上查找模型路径，示例如下：
- 'LLM-Research/Meta-Llama-3-8B'
- "FlagAlpha/Llama3-Chinese-8B-Instruct"  
- "ChineseAlpacaGroup/llama-3-chinese-8b-instruct-lora"  
- 'zhuangxialie/Llama3_Chinese_Sft'
- 'qwen/Qwen-7B-Chat'
- 'qwen/Qwen2-0.5B-Instruct'


```bash
    pip install -U modelscope
    modelscope download --model AI-ModelScope/bge-large-zh --local_dir "base/bge"
```


### 指令集生成
参考utils/chat_data_precessor.py文件。

### 大模型训练
首先修改 `settings` 文件夹中的配置文件，然后运行如下代码:
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun sft.py --model_path base/qwen/Qwen2-0_5B-Instruct \
    --data_path data/processed/SFT/FinCUGE \
    --ckpt_dir outputs/ckpts/qwen2/ \
    --final_save_path outputs/final/Qwen2-0_5B-instruct-lora 
```

### Bert类模型训练及各类模型推理
参考examples文件夹。

