#!/usr/bin/env python3
"""
Qwen2.5-Coder-7B-Instruct 双A100 40G GPU微调训练脚本
专门优化用于函数名推测任务
"""
import os
import json
import torch
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# 设置环境变量支持双GPU训练
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 使用两块GPU

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    HfArgumentParser
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import deepspeed

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="/root/project/finetune/qwen_new/Qwen2.5-Coder-7B-Instruct",
        metadata={"help": "Qwen模型路径"}
    )
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(
        default="/root/project/finetune/qwen_new/funcname_ft_instruction1.jsonl",
        metadata={"help": "原始数据集路径"}
    )
    max_length: int = field(default=4096)
    val_ratio: float = field(default=0.1, metadata={"help": "验证集比例"})
    test_size: int = field(default=1000, metadata={"help": "验证集大小（优先val_ratio）"})

@dataclass
class LoraArguments:
    use_lora: bool = field(default=True)
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA scaling"})
    lora_dropout: float = field(default=0.05)
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

class QwenFunctionNameDataset:
    """Qwen函数名数据集处理类，专门用于双GPU训练"""
    
    def __init__(self, tokenizer, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_data(self, file_path):
        """加载JSONL格式数据"""
        logger.info(f"Loading data from {file_path}")
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    continue
        logger.info(f"Loaded {len(data)} examples from {file_path}")
        return data
    
    def format_conversation(self, messages):
        """格式化对话为Qwen格式"""
        formatted_text = ""
        for message in messages:
            role = message["role"]
            
            content = message["content"]
            
            if role == "system":
                formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                
        return formatted_text
    
    def tokenize_function(self, examples):
        """批量tokenization"""
        processed_examples = []
        
        for example in examples:
            # 适配instruction/input/output格式
            if 'messages' in example:
                # 处理messages格式
                messages = example['messages']
                input_messages = messages[:-1]  # 除最后一个assistant消息
                target_message = messages[-1]   # 最后一个assistant消息
                
                input_text = self.format_conversation(input_messages)
                input_text += "<|im_start|>assistant\n"
                full_text = input_text + target_message["content"] + "<|im_end|>"
            else:
                # 处理instruction/input/output格式
                instruction = example.get('instruction', '')
                user_input = example.get('input', '')
                output = example.get('output', '')
                
                # 构建系统提示
                system_prompt = f"<|im_start|>system\n{instruction}<|im_end|>\n"
                user_prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n"
                input_text = system_prompt + user_prompt + "<|im_start|>assistant\n"
                full_text = input_text + output + "<|im_end|>"
            
            # 分别tokenize输入和完整文本
            input_ids = self.tokenizer(
                input_text,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=False
            )['input_ids']
            
            full_ids = self.tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=False
            )['input_ids']
            
            # 创建labels：输入部分为-100，输出部分为token_ids
            # 确保labels长度与full_ids一致
            input_len = len(input_ids)
            labels = []
            for i in range(len(full_ids)):
                if i < input_len:
                    labels.append(-100)  # 输入部分mask掉
                else:
                    labels.append(full_ids[i])  # 输出部分保留
            
            # 验证长度一致性
            assert len(full_ids) == len(labels), f"Length mismatch: input_ids={len(full_ids)}, labels={len(labels)}"
            
            # 验证labels中的值都在词汇表范围内
            vocab_size = len(self.tokenizer)
            for i, label in enumerate(labels):
                if label != -100 and (label < 0 or label >= vocab_size):
                    logger.error(f"Invalid label at position {i}: {label} (vocab_size={vocab_size})")
                    logger.error(f"Example: {example}")
                    raise ValueError(f"Label {label} is out of vocabulary range [0, {vocab_size})")
            
            processed_examples.append({
                'input_ids': full_ids,
                'labels': labels
            })
        
        # 转换为批量格式
        batch_input_ids = [ex['input_ids'] for ex in processed_examples]
        batch_labels = [ex['labels'] for ex in processed_examples]
        
        # 创建attention_mask
        batch_attention_mask = [[1] * len(ids) for ids in batch_input_ids]
        
        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'labels': batch_labels
        }
    
    def prepare_dataset(self, data):
        """准备数据集"""
        # 先tokenize一小部分数据来验证
        logger.info("Processing dataset...")
        
        # 分批处理以避免内存问题
        batch_size = 1000
        processed_data = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            processed_batch = self.tokenize_function(batch)
            
            # 转换为list of dict格式
            for j in range(len(processed_batch['input_ids'])):
                processed_data.append({
                    'input_ids': processed_batch['input_ids'][j],
                    'attention_mask': processed_batch['attention_mask'][j],
                    'labels': processed_batch['labels'][j]
                })
                
            logger.info(f"Processed {min(i+batch_size, len(data))}/{len(data)} examples")
        
        return Dataset.from_list(processed_data)

def load_model_and_tokenizer(model_args, lora_args):
    """加载模型和tokenizer"""
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        trust_remote_code=True
    )
    
    # 添加pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info("Loading model...")
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    logger.info(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    logger.info(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")
    
    # 4bit量化配置，针对A100优化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",  # 自动分配到两个GPU
        trust_remote_code=True,
        # attn_implementation="flash_attention_2" if torch.cuda.is_available() else None  # 暂时禁用flash attention
    )
    
    # 准备模型进行量化训练
    model = prepare_model_for_kbit_training(model)
    
    if lora_args.use_lora:
        logger.info("Setting up LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=lora_args.lora_target_modules,
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer

def main():
    # 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # 保证evaluation_strategy和save_strategy一致，避免transformers报错
    if hasattr(training_args, 'evaluation_strategy'):
        if training_args.evaluation_strategy != training_args.save_strategy:
            training_args.save_strategy = training_args.evaluation_strategy
    elif hasattr(training_args, 'eval_strategy'):
        if training_args.eval_strategy != training_args.save_strategy:
            training_args.save_strategy = training_args.eval_strategy

    # 设置输出目录
    if not training_args.output_dir:
        training_args.output_dir = f"./outputs/qwen2.5-coder-7b-funcname-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(training_args.output_dir, exist_ok=True)

    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, lora_args)

    # 加载数据，只用一个数据集，自动划分train/val
    dataset_processor = QwenFunctionNameDataset(tokenizer, data_args.max_length)
    all_data = dataset_processor.load_data(data_args.data_path)
    import random
    random.seed(42)
    random.shuffle(all_data)
    val_num = int(len(all_data) * data_args.val_ratio)
    if val_num < 1:
        val_num = min(data_args.test_size, max(1, len(all_data)//10))
    val_data = all_data[:val_num]
    train_data = all_data[val_num:]

    logger.info(f"Train/Val split: {len(train_data)}/{len(val_data)}")

    # 准备数据集
    train_dataset = dataset_processor.prepare_dataset(train_data)
    val_dataset = dataset_processor.prepare_dataset(val_data)

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # 数据收集器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 开始训练
    logger.info("Starting training...")
    trainer.train()
    
    # 保存模型
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # 保存配置信息
    config_info = {
        "model_name": model_args.model_name_or_path,
        "training_time": datetime.now().isoformat(),
        "total_steps": trainer.state.global_step,
        "final_loss": trainer.state.log_history[-1].get("train_loss", "N/A"),
        "lora_config": {
            "r": lora_args.lora_r,
            "alpha": lora_args.lora_alpha,
            "dropout": lora_args.lora_dropout,
            "target_modules": lora_args.lora_target_modules
        }
    }
    
    with open(os.path.join(training_args.output_dir, "training_config.json"), "w") as f:
        json.dump(config_info, f, indent=2)
    
    logger.info(f"Training completed! Model saved to {training_args.output_dir}")
    
    # 训练结束后，在验证集上运行详细评估
    if training_args.do_eval and val_dataset:
        logger.info("=" * 80)
        logger.info("Running detailed evaluation on validation set...")
        logger.info("=" * 80)
        
        try:
            # 保存验证集数据到临时文件
            val_data_path = os.path.join(training_args.output_dir, "val_data.jsonl")
            with open(val_data_path, 'w', encoding='utf-8') as f:
                for item in val_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"Validation data saved to: {val_data_path}")
            logger.info(f"Validation set size: {len(val_data)}")
            logger.info("")
            logger.info("To run evaluation, use:")
            logger.info(f"python evaluate_qwen_standalone.py \\")
            logger.info(f"  --model_path {training_args.output_dir} \\")
            logger.info(f"  --test_data_path {val_data_path} \\")
            logger.info(f"  --sample_size {len(val_data)}")
            logger.info("")
            
        except Exception as e:
            logger.warning(f"Failed to save validation data: {e}")

if __name__ == "__main__":
    main()
