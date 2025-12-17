#!/usr/bin/env python3
"""
Qwen2.5-Coder-7B-Instruct 微调模型测评脚本
专门用于函数名推测任务的准确率评估
集成 Dependency-Aware Semantic Propagation (CASP) 功能
"""
import os
import json
import torch
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import re
import difflib
import csv

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GlobalSymbolTable:
    """全局符号表，用于管理地址到函数名的映射并支持持久化"""
    def __init__(self, output_dir: str):
        self.symbol_map = {}  # Address (hex string without 0x) -> Function Name
        self.output_csv = os.path.join(output_dir, "global_symbol_table.csv")
        self.headers = ["Address", "Resolved Name"]
        
        # 初始化CSV文件
        with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            
    def update(self, address: str, name: str):
        """更新符号表"""
        # 规范化地址格式：移除0x前缀，转大写，确保一致性
        if not address:
            return
            
        norm_addr = address.replace("0x", "").replace("0X", "").upper()
        self.symbol_map[norm_addr] = name
        
        # 追加到CSV
        with open(self.output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([f"0x{norm_addr}", name])
            
    def resolve_context(self, text: str) -> str:
        """
        在文本中进行语义传播：将已知的 sub_XXXX 替换为解析出的函数名
        """
        if not self.symbol_map or not text:
            return text
            
        # 查找所有 sub_XXXX 模式
        # 假设 sub_ 后跟十六进制地址
        def replace_match(match):
            addr_str = match.group(1).upper()
            if addr_str in self.symbol_map:
                return self.symbol_map[addr_str]
            return match.group(0)
            
        # 匹配 sub_ + 16进制字符
        pattern = r'\b(?:sub_|func_|loc_)([0-9A-Fa-f]+)\b'
        resolved_text = re.sub(pattern, replace_match, text)
        
        return resolved_text

class SRSCalculator:
    """Semantic Richness Score (SRS) 计算器"""
    
    @staticmethod
    def calculate_srs(example: Dict) -> float:
        """
        SRS(f) = alpha * N_str + beta * N_api + gamma * Depth
        alpha=2.0, beta=1.0, gamma=0.5
        """
        alpha = 2.0
        beta = 1.0
        gamma = 0.5
        
        # 1. 获取字符串数量 (N_str)
        n_str = 0
        if 'internal_strings' in example:
            n_str = len(example['internal_strings'])
        elif 'string_arguments' in example: 
             # 如果没有internal_strings，尝试用string_arguments作为代理
            n_str = len(example['string_arguments'])
            
        # 2. 获取API调用数量 (N_api)
        n_api = 0
        # 尝试从 function_body 中估算 API 调用
        # 简单统计 sub_ 或者常见 API 模式
        body = example.get('function_body', '')
        if body:
            # 统计函数调用 pattern: name(...)
            calls = re.findall(r'\b\w+\s*\(', body)
            # 排除控制流关键字
            keywords = {'if', 'while', 'for', 'switch', 'return', 'sizeof'}
            n_api = sum(1 for c in calls if c.split('(')[0].strip() not in keywords)
            
        # 3. 获取调用深度 (Depth)
        depth = 0
        if 'call_chain' in example:
            chains = example['call_chain']
            if chains and isinstance(chains, list):
                # 假设 call_chain 是链的列表，取最大长度
                # 链可能是 ["main", "sub_1", "sub_2"]
                lengths = [len(c) for c in chains if isinstance(c, list)]
                if lengths:
                    depth = max(lengths)
        
        srs = alpha * n_str + beta * n_api + gamma * depth
        return srs

class SimpleEvaluationMetrics:
    """简化的评估指标计算类"""
    
    @staticmethod
    def exact_match_accuracy(predictions: List[str], references: List[str]) -> float:
        """计算精确匹配准确率"""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        matches = sum(1 for pred, ref in zip(predictions, references) 
                     if pred.strip().lower() == ref.strip().lower())
        return matches / len(predictions)
    
    @staticmethod
    def similarity_score(pred: str, ref: str) -> float:
        """计算字符串相似度 (基于序列匹配)"""
        return difflib.SequenceMatcher(None, pred.lower(), ref.lower()).ratio()
    
    @staticmethod
    def average_similarity(predictions: List[str], references: List[str]) -> float:
        """计算平均相似度"""
        similarities = [SimpleEvaluationMetrics.similarity_score(pred, ref) 
                       for pred, ref in zip(predictions, references)]
        return np.mean(similarities)
    
    @staticmethod
    def edit_distance(s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return SimpleEvaluationMetrics.edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def normalized_edit_distance(predictions: List[str], references: List[str]) -> float:
        """计算标准化编辑距离"""
        distances = []
        for pred, ref in zip(predictions, references):
            distance = SimpleEvaluationMetrics.edit_distance(pred, ref)
            max_len = max(len(pred), len(ref))
            normalized_distance = distance / max_len if max_len > 0 else 0
            distances.append(1 - normalized_distance)  # 转换为相似度
        return np.mean(distances)

    @staticmethod
    def subtoken_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算子词级 (Sub-token) Precision, Recall, F1"""
        def tokenize_name(name: str) -> set:
            # 处理下划线和驼峰命名
            # 1. 在大写字母前添加空格 (处理驼峰)
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
            s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
            # 2. 将下划线替换为空格
            s3 = s2.replace('_', ' ')
            # 3. 分割、转小写、去重
            return set(token.lower() for token in s3.split() if token)

        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        n = len(predictions)

        if n == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        for pred, ref in zip(predictions, references):
            pred_tokens = tokenize_name(pred)
            ref_tokens = tokenize_name(ref)
            
            common = len(pred_tokens & ref_tokens)
            
            p = common / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
            r = common / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
            
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            
            total_precision += p
            total_recall += r
            total_f1 += f1
            
        return {
            "precision": total_precision / n,
            "recall": total_recall / n,
            "f1": total_f1 / n
        }

class QwenModelEvaluator:
    """Qwen模型评估器"""
    
    def __init__(self, model_path: str, base_model_path: str = None, use_lora: bool = True):
        """
        初始化评估器
        
        Args:
            model_path: 微调后的模型路径（LoRA适配器路径）
            base_model_path: 基础模型路径，如果为None则从model_path推断
            use_lora: 是否使用LoRA模型
        """
        self.model_path = model_path
        self.use_lora = use_lora
        
        if base_model_path is None:
            # 尝试从配置文件推断基础模型路径
            config_path = os.path.join(model_path, "training_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.base_model_path = config.get("model_name", "/root/project/finetune/qwen_new/Qwen2.5-Coder-7B-Instruct")
            else:
                self.base_model_path = "/root/project/finetune/qwen_new/Qwen2.5-Coder-7B-Instruct"
        else:
            self.base_model_path = base_model_path
            
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """加载模型和tokenizer"""
        logger.info(f"Loading base model from: {self.base_model_path}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 如果使用LoRA，加载适配器
        if self.use_lora and os.path.exists(os.path.join(self.model_path, "adapter_config.json")):
            logger.info(f"Loading LoRA adapter from: {self.model_path}")
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
            # 注意：在推理时保持LoRA状态，不合并权重以节省内存
            
        self.model.eval()
        logger.info("Model loaded successfully")
        
    def generate_function_name(self, prompt: str, instruction: str = None, max_new_tokens: int = 64, temperature: float = 0.1) -> str:
        """
        生成函数名
        
        Args:
            prompt: 输入提示（user input）
            instruction: 系统指令（system prompt）
            max_new_tokens: 最大生成token数
            temperature: 生成温度
            
        Returns:
            生成的函数名
        """
        # 格式化为Qwen对话格式，与训练时保持一致
        if instruction:
            formatted_prompt = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 生成
        # 注意：对于量化模型，使用greedy decoding更稳定
        with torch.no_grad():
            if temperature <= 0.01:
                # 贪婪解码，最稳定
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            else:
                # 采样解码，使用更高的temperature避免数值问题
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=max(temperature, 0.3),  # 最低0.3避免数值问题
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,  # 添加top_k限制
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.05  # 降低repetition_penalty
                )
        
        # 解码响应
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # 清理响应，提取函数名
        response = response.strip()
        
        # 如果响应包含结束标记，截取到结束标记
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()
            
        # 清理额外的空格和换行
        response = re.sub(r'\s+', ' ', response).strip()
        
        # 如果响应太长，可能包含解释，尝试提取函数名
        if len(response) > 50:
            # 尝试匹配函数名模式
            function_name_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
            matches = re.findall(function_name_pattern, response)
            if matches:
                response = matches[0]  # 取第一个匹配的标识符
            
        return response

class QwenTestFramework:
    """Qwen测试框架"""
    
    def __init__(self):
        """初始化测试框架"""
        self.metrics = SimpleEvaluationMetrics()
        
    def prepare_test_data(self, test_data_path: str, sample_size: int = None) -> List[Dict]:
        """
        准备测试数据
        
        Args:
            test_data_path: 测试数据路径
            sample_size: 采样大小，如果为None则使用全部数据
            
        Returns:
            测试数据列表
        """
        logger.info(f"Loading test data from: {test_data_path}")
        
        test_data = []
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    test_data.append(item)
                except json.JSONDecodeError:
                    continue
                    
        logger.info(f"Loaded {len(test_data)} test examples")
        
        # 如果指定了采样大小
        if sample_size and sample_size < len(test_data):
            import random
            random.seed(42)  # 设置随机种子以确保可重现性
            test_data = random.sample(test_data, sample_size)
            logger.info(f"Sampled {sample_size} examples for testing")
            
        return test_data
    
    def extract_user_prompt(self, example: Dict) -> tuple:
        """从数据样本中提取用户提示，返回(instruction, input)"""
        if "messages" in example:
            # 处理messages格式
            instruction = ""
            user_input = ""
            for message in example["messages"]:
                if message.get("role") == "system":
                    instruction = message.get("content", "")
                elif message.get("role") == "user":
                    user_input = message.get("content", "")
            return instruction, user_input
        elif "instruction" in example:
            # 处理instruction/input/output格式
            instruction = example.get("instruction", "")
            user_input = example.get("input", "")
            return instruction, user_input
        return "", ""
    
    def extract_ground_truth(self, example: Dict) -> str:
        """从数据样本中提取标准答案"""
        if "messages" in example:
            # 处理messages格式
            for message in reversed(example["messages"]):
                if message.get("role") == "assistant":
                    return message.get("content", "").strip()
            # 如果没有找到assistant回复，可能是inference-only数据
            return ""
        elif "output" in example:
            # 处理instruction/input/output格式
            return example.get("output", "").strip()
        return ""
    
    def run_evaluation(self, 
                      model_evaluator: QwenModelEvaluator,
                      test_data: List[Dict],
                      output_dir: str = None) -> Dict[str, Any]:
        """
        运行完整评估
        
        Args:
            model_evaluator: Qwen模型评估器
            test_data: 测试数据
            output_dir: 输出目录
            
        Returns:
            评估结果
        """
        if output_dir is None:
            output_dir = f"/root/project/finetune/test_results/qwen_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化全局符号表
        symbol_table = GlobalSymbolTable(output_dir)
        
        # 1. SRS 排序 (Dependency-Aware Scheduling)
        logger.info("Calculating SRS for dependency-aware scheduling...")
        data_with_srs = []
        for item in test_data:
            srs = SRSCalculator.calculate_srs(item)
            data_with_srs.append((srs, item))
            
        # 按SRS降序排序
        data_with_srs.sort(key=lambda x: x[0], reverse=True)
        sorted_test_data = [item for _, item in data_with_srs]
        
        logger.info(f"Sorted {len(sorted_test_data)} examples by SRS.")
        
        results = []
        predictions = []
        ground_truths = []
        
        logger.info(f"Starting evaluation on {len(sorted_test_data)} examples...")
        
        for i, example in enumerate(sorted_test_data):
            if i % 50 == 0:
                logger.info(f"Processing {i}/{len(sorted_test_data)} examples...")
                
            instruction, user_input = self.extract_user_prompt(example)
            ground_truth = self.extract_ground_truth(example)
            
            # 如果没有ground truth (纯推理模式)，也不跳过
            # if not user_input or not ground_truth:
            if not user_input:
                continue
            
            # 2. 语义传播: 更新上下文 (Dynamic Enrichment)
            enriched_input = symbol_table.resolve_context(user_input)
                
            # 生成预测
            try:
                prediction = model_evaluator.generate_function_name(enriched_input, instruction=instruction)
                predictions.append(prediction)
                ground_truths.append(ground_truth)
                
                # 3. 更新符号表 (Semantic Propagation)
                # 尝试从example中获取地址
                func_ea = example.get('func_ea')
                if not func_ea:
                    # 如果没有直接的key，尝试从dummy_name或user_input中正则提取
                    # 假设dummy_name是 sub_XXXX
                    dummy = example.get('dummy_name', '')
                    match = re.search(r'sub_([0-9A-Fa-f]+)', dummy)
                    if match:
                        func_ea = match.group(1)
                
                if func_ea and prediction:
                    symbol_table.update(func_ea, prediction)
                
                # 计算相似度 (如果有ground truth)
                similarity_score = 0.0
                exact_match = False
                if ground_truth:
                    similarity_score = self.metrics.similarity_score(prediction, ground_truth)
                    exact_match = prediction.strip().lower() == ground_truth.strip().lower()
                
                result = {
                    "index": i,
                    "srs_score": data_with_srs[i][0], # 记录SRS
                    "instruction": instruction,
                    "user_input": user_input[:200] + "..." if len(user_input) > 200 else user_input,
                    "enriched_input": enriched_input[:200] + "..." if len(enriched_input) > 200 else enriched_input,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "exact_match": exact_match,
                    "similarity_score": similarity_score,
                    "func_ea": func_ea
                }
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error processing example {i}: {e}")
                continue
        
        # 计算总体指标
        total_examples = len(results)
        exact_matches = sum(1 for r in results if r["exact_match"])
        exact_match_rate = exact_matches / total_examples if total_examples > 0 else 0
        
        similarity_scores = [r["similarity_score"] for r in results]
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
        median_similarity = np.median(similarity_scores) if similarity_scores else 0
        
        # 计算额外的评估指标 (只对有GT的数据)
        if ground_truths and any(ground_truths):
            valid_preds = [p for p, g in zip(predictions, ground_truths) if g]
            valid_gts = [g for g in ground_truths if g]
            if valid_gts:
                avg_sim_metric = self.metrics.average_similarity(valid_preds, valid_gts)
                norm_edit_distance = self.metrics.normalized_edit_distance(valid_preds, valid_gts)
                subtoken_res = self.metrics.subtoken_metrics(valid_preds, valid_gts)
            else:
                 subtoken_res = {"precision": 0, "recall": 0, "f1": 0}
                 avg_sim_metric = 0
                 norm_edit_distance = 0
        else:
            subtoken_res = {"precision": 0, "recall": 0, "f1": 0}
            avg_sim_metric = 0
            norm_edit_distance = 0
        
        summary = {
            "model_path": model_evaluator.model_path,
            "total_examples": total_examples,
            "exact_match_count": exact_matches,
            "exact_match_rate": exact_match_rate,
            "subtoken_precision": subtoken_res["precision"],
            "subtoken_recall": subtoken_res["recall"],
            "subtoken_f1": subtoken_res["f1"],
            "average_similarity": avg_similarity,
            "median_similarity": median_similarity,
            "average_similarity_metric": avg_sim_metric,
            "normalized_edit_distance": norm_edit_distance,
            "evaluation_time": datetime.now().isoformat()
        }
        
        # 保存结果
        with open(os.path.join(output_dir, "detailed_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(output_dir, "summary_results.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            
        # 生成报告
        self.generate_evaluation_report(summary, results, output_dir)
        
        logger.info(f"Evaluation completed. Results saved to: {output_dir}")
        logger.info(f"Exact Match Rate: {exact_match_rate:.4f}")
        logger.info(f"Average Similarity: {avg_similarity:.4f}")
        
        return summary
    
    def generate_evaluation_report(self, summary: Dict, results: List[Dict], output_dir: str):
        """生成评估报告"""
        report_path = os.path.join(output_dir, "evaluation_report.md")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Qwen2.5-Coder-7B-Instruct 函数名推测评估报告 (Context-Aware)\n\n")
            f.write(f"**评估时间**: {summary['evaluation_time']}\n")
            f.write(f"**模型路径**: {summary['model_path']}\n\n")
            
            f.write("## 总体性能指标\n\n")
            f.write(f"- **总测试样本数**: {summary['total_examples']}\n")
            f.write(f"- **精确匹配数量**: {summary['exact_match_count']}\n")
            f.write(f"- **精确匹配率 (Exact Match)**: {summary['exact_match_rate']:.4f} ({summary['exact_match_rate']*100:.2f}%)\n")
            f.write(f"- **Sub-token Precision**: {summary['subtoken_precision']:.4f}\n")
            f.write(f"- **Sub-token Recall**: {summary['subtoken_recall']:.4f}\n")
            f.write(f"- **Sub-token F1**: {summary['subtoken_f1']:.4f}\n")
            f.write(f"- **平均相似度**: {summary['average_similarity']:.4f}\n")
            f.write(f"- **中位数相似度**: {summary['median_similarity']:.4f}\n")
            f.write(f"- **标准化编辑距离**: {summary['normalized_edit_distance']:.4f}\n\n")
            
            # 分析不同相似度区间的分布
            similarity_scores = [r["similarity_score"] for r in results]
            if len(similarity_scores) > 0:
                high_sim = sum(1 for s in similarity_scores if s >= 0.8)
                medium_sim = sum(1 for s in similarity_scores if 0.5 <= s < 0.8)
                low_sim = sum(1 for s in similarity_scores if s < 0.5)
                
                f.write("## 相似度分布\n\n")
                f.write(f"- **高相似度 (≥0.8)**: {high_sim} ({high_sim/len(similarity_scores)*100:.1f}%)\n")
                f.write(f"- **中等相似度 (0.5-0.8)**: {medium_sim} ({medium_sim/len(similarity_scores)*100:.1f}%)\n")
                f.write(f"- **低相似度 (<0.5)**: {low_sim} ({low_sim/len(similarity_scores)*100:.1f}%)\n\n")
            else:
                f.write("## 相似度分布\n\n")
                f.write("- **没有有效的评估结果**\n\n")
            
            # 展示一些示例
            f.write("## 预测示例\n\n")
            f.write("### 成功案例 (精确匹配)\n\n")
            
            success_cases = [r for r in results if r["exact_match"]][:5]
            for i, case in enumerate(success_cases, 1):
                f.write(f"**示例 {i}**:\n")
                f.write(f"- SRS: {case['srs_score']}\n")
                f.write(f"- 标准答案: `{case['ground_truth']}`\n")
                f.write(f"- 模型预测: `{case['prediction']}`\n")
                f.write(f"- 相似度: {case['similarity_score']:.4f}\n\n")
            
            f.write("### 高相似度案例 (未精确匹配但相似度高)\n\n")
            high_sim_cases = [r for r in results if not r["exact_match"] and r["similarity_score"] >= 0.8][:5]
            for i, case in enumerate(high_sim_cases, 1):
                f.write(f"**示例 {i}**:\n")
                f.write(f"- SRS: {case['srs_score']}\n")
                f.write(f"- 标准答案: `{case['ground_truth']}`\n")
                f.write(f"- 模型预测: `{case['prediction']}`\n")
                f.write(f"- 相似度: {case['similarity_score']:.4f}\n\n")
            
            f.write("### 需要改进的案例 (低相似度)\n\n")
            failed_cases = [r for r in results if r["similarity_score"] < 0.5][:5]
            for i, case in enumerate(failed_cases, 1):
                f.write(f"**示例 {i}**:\n")
                f.write(f"- SRS: {case['srs_score']}\n")
                f.write(f"- 标准答案: `{case['ground_truth']}`\n")
                f.write(f"- 模型预测: `{case['prediction']}`\n")
                f.write(f"- 相似度: {case['similarity_score']:.4f}\n\n")
        
        logger.info(f"Evaluation report saved to: {report_path}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen2.5-Coder-7B-Instruct 模型评估")
    parser.add_argument("--model_path", required=True, help="微调后的模型路径")
    parser.add_argument("--base_model_path", help="基础模型路径")
    parser.add_argument("--test_data_path", required=True, help="测试数据路径")
    parser.add_argument("--output_dir", help="输出目录")
    parser.add_argument("--sample_size", type=int, help="测试样本数量")
    parser.add_argument("--no_lora", action="store_true", help="不使用LoRA")
    parser.add_argument("--temperature", type=float, default=0.1, help="生成温度")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="最大生成token数")
    
    args = parser.parse_args()
    
    # 初始化评估器
    model_evaluator = QwenModelEvaluator(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        use_lora=not args.no_lora
    )
    
    # 加载模型
    model_evaluator.load_model()
    
    # 初始化测试框架
    test_framework = QwenTestFramework()
    
    # 准备测试数据
    test_data = test_framework.prepare_test_data(
        args.test_data_path,
        args.sample_size
    )
    
    # 运行评估
    results = test_framework.run_evaluation(
        model_evaluator,
        test_data,
        args.output_dir
    )
    
    print("\n" + "="*50)
    print("评估完成！")
    print(f"精确匹配率 (Exact Match): {results['exact_match_rate']:.4f} ({results['exact_match_rate']*100:.2f}%)")
    print(f"Sub-token F1: {results['subtoken_f1']:.4f}")
    print(f"Sub-token Precision: {results['subtoken_precision']:.4f}")
    print(f"Sub-token Recall: {results['subtoken_recall']:.4f}")
    print(f"平均相似度: {results['average_similarity']:.4f}")
    print(f"标准化编辑距离: {results['normalized_edit_distance']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
