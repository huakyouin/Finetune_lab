import json
import argparse
import torch
import transformers
from typing import Dict
from collections import defaultdict
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq,AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel,PeftConfig,AutoPeftModel
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers.trainer_pt_utils import LabelSmoother
from datasets import load_from_disk, concatenate_datasets, Dataset
from tqdm import tqdm


IGNORE_TOKEN_ID = LabelSmoother.ignore_index
def preprocess(
    messages,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
) -> Dict:
    """Preprocesses the data for supervised fine-tuning."""

    texts = []
    for i, msg in tqdm(enumerate(messages)):
        texts.append(
            tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=True,
                padding="max_length",
                max_length=max_len,
                truncation=True,
            )
        )
    input_ids = torch.tensor(texts, dtype=torch.int)
    target_ids = input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": target_ids
    })



def main(config):

    print("load tokenizer and model...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        # device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            # load_in_4bit=True,
            # bnb_4bit_use_double_quant=True,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.float16,
        ),
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"],
        use_fast=False,
        trust_remote_code=True
    )

    print("load dataset...")
    data = load_from_disk(config["data_path"])
    data = data.shuffle(seed=42).flatten_indices()
    data = data.select(list(range(100)))

    print("split dataset...")
    train_test_split = data.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']
    # 保存数据集到JSONL格式
    import os
    test_dataset.save_to_disk(os.path.join(config["final_save_path"],"eval_data"))

    print("input format process...")
    train_dataset_input = preprocess(train_dataset["message"],tokenizer,584)
    test_dataset_input = preprocess(test_dataset["message"],tokenizer,584)

    print("add lora adapter...")
    lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "gate_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    model = get_peft_model(model, lora_config)

    if  config["gradient_checkpointing"]==True:
        model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法

    print("build trainer...")
    trainer_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        weight_decay=0.01,
        output_dir=config["ckpt_dir"],
        # num_train_epochs=1,
        max_steps=100,       # 最大训练步数
        seed=3047,
        save_steps=20,
        eval_steps=50,
        gradient_checkpointing=config["gradient_checkpointing"],
        save_total_limit=30,
    )
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset_input,
        eval_dataset=test_dataset_input
        # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    print("resume from checkpoint:",config["resume"],"\ntraining...")
    trainer.train(resume_from_checkpoint=config["resume"])


    print("save lora model...")
    model.save_pretrained(config["final_save_path"])


if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="SFT settings.")
    parser.add_argument('--model_path', type=str, help="Path to the model checkpoint or model identifier.")
    parser.add_argument('--data_path', type=str, help="Path to the dataset.")
    parser.add_argument('--resume', help="resume from checkpoint", action="store_true")
    parser.add_argument('--gradient_checkpointing', help="use gradient checkpointing to save GPU usage", action="store_true")
    parser.add_argument('--ckpt_dir', type=str, help="Path to save the model during training.")
    parser.add_argument('--final_save_path', type=str, help="Path to save the trained adapter.")
    args = parser.parse_args()
    main(vars(args))