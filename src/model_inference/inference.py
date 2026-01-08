"""
Model Inference Module
模型推理和函数名生成
"""

import re
import logging
import torch
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Prompt构建器"""

    DEFAULT_SYSTEM_PROMPT = """You are a reverse engineering expert specializing in binary analysis and function name recovery.
Your task is to analyze stripped binary functions and infer meaningful, descriptive function names based on:
1. Decompiled function body
2. Calling context from caller functions
3. String arguments passed at call sites
4. Hierarchical call chains

Provide a concise, snake_case or camelCase function name that best describes the function's purpose."""

    @staticmethod
    def build_chatml_prompt(
        function_body: str,
        caller_context: Optional[list] = None,
        string_arguments: Optional[list] = None,
        call_chains: Optional[list] = None,
        system_prompt: Optional[str] = None,
        max_context_items: int = 5
    ) -> str:
        """
        构建ChatML格式的prompt

        Args:
            function_body: 反编译的函数体
            caller_context: 调用者上下文列表
            string_arguments: 字符串参数列表
            call_chains: 调用链列表
            system_prompt: 系统提示
            max_context_items: 最大上下文项数

        Returns:
            格式化的prompt
        """
        if system_prompt is None:
            system_prompt = PromptBuilder.DEFAULT_SYSTEM_PROMPT

        # 构建用户消息
        user_message = "# Target Function\n\n"
        user_message += "## Decompiled Body\n```c\n"
        user_message += function_body[:2000]  # 限制长度
        user_message += "\n```\n\n"

        # 添加调用者上下文
        if caller_context and len(caller_context) > 0:
            user_message += "## Caller Context\n"
            for i, ctx in enumerate(caller_context[:max_context_items]):
                caller_func = ctx.get('caller_function', 'unknown')
                user_message += f"\n### Caller {i+1}: {caller_func}\n"

                # 反编译上下文
                if 'context_decompiled' in ctx:
                    decompiled = ctx['context_decompiled']
                    if decompiled:
                        user_message += "```c\n"
                        user_message += "\n".join(decompiled[:10])
                        user_message += "\n```\n"

        # 添加字符串参数
        if string_arguments and len(string_arguments) > 0:
            user_message += "\n## String Arguments\n"
            for i, arg in enumerate(string_arguments[:max_context_items]):
                if isinstance(arg, dict):
                    value = arg.get('value', '')
                elif isinstance(arg, str):
                    value = arg
                else:
                    continue
                user_message += f"- \"{value}\"\n"

        # 添加调用链
        if call_chains and len(call_chains) > 0:
            user_message += "\n## Call Chains\n"
            for chain in call_chains[:max_context_items]:
                if isinstance(chain, list):
                    chain_str = " -> ".join(chain)
                    user_message += f"- {chain_str}\n"

        # 构建ChatML格式
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return prompt


class QwenInferenceEngine:
    """Qwen模型推理引擎"""

    def __init__(
        self,
        base_model_path: str,
        lora_adapter_path: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "bfloat16"
    ):
        """
        初始化推理引擎

        Args:
            base_model_path: 基础模型路径
            lora_adapter_path: LoRA适配器路径
            device: 设备
            dtype: 数据类型
        """
        self.base_model_path = base_model_path
        self.lora_adapter_path = lora_adapter_path
        self.device = device
        self.dtype_str = dtype

        self.model = None
        self.tokenizer = None

        logger.info(f"Initializing QwenInferenceEngine with model: {base_model_path}")

    def load_model(self):
        """加载模型和tokenizer"""
        logger.info(f"Loading tokenizer from {self.base_model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info(f"Loading base model from {self.base_model_path}")

        dtype = getattr(torch, self.dtype_str)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )

        # 加载LoRA适配器
        if self.lora_adapter_path:
            logger.info(f"Loading LoRA adapter from {self.lora_adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.lora_adapter_path)

        self.model.eval()
        logger.info("Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05
    ) -> str:
        """
        生成函数名

        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 温度
            top_p: nucleus sampling参数
            top_k: top-k sampling参数
            repetition_penalty: 重复惩罚

        Returns:
            生成的函数名
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            if temperature <= 0.01:
                # 贪婪解码
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            else:
                # 采样解码
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=max(temperature, 0.3),
                    do_sample=True,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=repetition_penalty
                )

        # Decode
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # 后处理
        response = self._post_process_response(response)

        return response

    def _post_process_response(self, response: str) -> str:
        """后处理生成的响应"""
        # 清理响应
        response = response.strip()

        # 截取到结束标记
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()

        # 清理额外的空格和换行
        response = re.sub(r'\s+', ' ', response).strip()

        # 如果响应太长,可能包含解释,尝试提取函数名
        if len(response) > 50:
            function_name_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
            matches = re.findall(function_name_pattern, response)
            if matches:
                response = matches[0]

        return response

    def batch_generate(
        self,
        prompts: list,
        **kwargs
    ) -> list:
        """批量生成"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results
