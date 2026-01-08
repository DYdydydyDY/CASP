"""
Semantic Richness Score (SRS) Calculator
实现论文中的语义丰富度评分算法
"""

import re
import logging
from typing import Dict, List, Any
from dataclasses import dataclass

from ..utils.helpers import FunctionNameValidator

logger = logging.getLogger(__name__)


@dataclass
class SRSComponents:
    """SRS分数组成部分"""
    string_count: int = 0
    api_count: int = 0
    call_depth: int = 0
    total_score: float = 0.0


class SRSCalculator:
    """
    语义丰富度评分计算器

    根据论文公式: SRS(f) = α * N_str + β * N_api + γ * Depth
    """

    def __init__(self, alpha: float = 2.0, beta: float = 1.0, gamma: float = 0.5):
        """
        初始化SRS计算器

        Args:
            alpha: 字符串数量权重 (默认2.0)
            beta: API调用数量权重 (默认1.0)
            gamma: 调用深度权重 (默认0.5)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        logger.info(f"SRS Calculator initialized with α={alpha}, β={beta}, γ={gamma}")

    def calculate_srs(self, function_data: Dict[str, Any]) -> float:
        """
        计算函数的语义丰富度分数

        Args:
            function_data: 函数数据字典,包含:
                - internal_strings: 内部字符串列表
                - string_arguments: 字符串参数列表
                - function_body: 函数体代码
                - call_chain: 调用链列表

        Returns:
            SRS分数
        """
        components = self.calculate_components(function_data)
        return components.total_score

    def calculate_components(self, function_data: Dict[str, Any]) -> SRSComponents:
        """
        计算SRS各组成部分

        Args:
            function_data: 函数数据字典

        Returns:
            SRSComponents对象
        """
        # 1. 计算字符串数量 (N_str)
        n_str = self._count_strings(function_data)

        # 2. 计算API调用数量 (N_api)
        n_api = self._count_api_calls(function_data)

        # 3. 计算调用深度 (Depth)
        depth = self._calculate_call_depth(function_data)

        # 4. 计算总分
        total_score = self.alpha * n_str + self.beta * n_api + self.gamma * depth

        return SRSComponents(
            string_count=n_str,
            api_count=n_api,
            call_depth=depth,
            total_score=total_score
        )

    def _count_strings(self, function_data: Dict[str, Any]) -> int:
        """统计有意义的字符串数量"""
        count = 0

        # 从internal_strings获取
        if 'internal_strings' in function_data:
            strings = function_data['internal_strings']
            if isinstance(strings, list):
                count += len(strings)

        # 从string_arguments获取(去重)
        if 'string_arguments' in function_data:
            string_args = function_data['string_arguments']
            if isinstance(string_args, list):
                # 提取字符串值
                arg_values = set()
                for arg in string_args:
                    if isinstance(arg, dict) and 'value' in arg:
                        arg_values.add(arg['value'])
                    elif isinstance(arg, str):
                        arg_values.add(arg)
                count += len(arg_values)

        return count

    def _count_api_calls(self, function_data: Dict[str, Any]) -> int:
        """统计API调用数量"""
        body = function_data.get('function_body', '')
        if not body:
            return 0

        # 统计函数调用pattern: name(...)
        calls = re.findall(r'\b\w+\s*\(', body)

        # 排除控制流关键字
        keywords = {'if', 'while', 'for', 'switch', 'return', 'sizeof', 'typeof'}
        api_calls = [c for c in calls if c.split('(')[0].strip() not in keywords]

        return len(api_calls)

    def _calculate_call_depth(self, function_data: Dict[str, Any]) -> int:
        """计算最大调用深度"""
        if 'call_chain' not in function_data:
            return 0

        chains = function_data['call_chain']
        if not chains or not isinstance(chains, list):
            return 0

        # 假设call_chain是链的列表,每个链是函数名列表
        # 取最长链的长度
        max_depth = 0
        for chain in chains:
            if isinstance(chain, list):
                max_depth = max(max_depth, len(chain))

        return max_depth

    def rank_functions(self, functions: List[Dict[str, Any]]) -> List[tuple]:
        """
        对函数列表按SRS排序

        Args:
            functions: 函数数据列表

        Returns:
            排序后的(srs_score, function_data)元组列表,按分数降序
        """
        scored_functions = []

        for func_data in functions:
            srs = self.calculate_srs(func_data)
            scored_functions.append((srs, func_data))

        # 按SRS降序排序
        scored_functions.sort(key=lambda x: x[0], reverse=True)

        return scored_functions


class CallChainQualityScorer:
    """
    调用链质量评分器

    根据论文公式计算调用链的质量分数
    """

    def __init__(self, semantic_weight: int = 5, length_weight: int = 10):
        """
        初始化调用链质量评分器

        Args:
            semantic_weight: 有意义函数名的权重
            length_weight: 理想长度的权重
        """
        self.semantic_weight = semantic_weight
        self.length_weight = length_weight

    def calculate_quality(self, chain: List[str]) -> float:
        """
        计算调用链质量分数

        根据论文公式:
        S_chain = Σ I(f ∉ G) * w_sem + w_len * I(2 ≤ |Chain| ≤ 4)

        Args:
            chain: 调用链,函数名列表

        Returns:
            质量分数
        """
        score = 0.0

        # 语义分数:有意义的函数名加分
        for func_name in chain:
            if FunctionNameValidator.is_meaningful_function(func_name):
                score += self.semantic_weight

        # 长度分数:适中长度(2-4)加分
        chain_length = len(chain)
        if 2 <= chain_length <= 4:
            score += self.length_weight
        elif chain_length == 5:
            score += self.length_weight * 0.5

        # 特殊函数名加分(main, init等)
        meaningful_patterns = [
            r'main', r'init', r'create', r'process', r'handle',
            r'get', r'set', r'read', r'write', r'open', r'close',
            r'send', r'recv', r'parse', r'format', r'check'
        ]

        for func_name in chain:
            for pattern in meaningful_patterns:
                if re.search(pattern, func_name, re.IGNORECASE):
                    score += 3
                    break

        return score

    def rank_chains(self, chains: List[List[str]]) -> List[tuple]:
        """
        对调用链列表按质量排序

        Args:
            chains: 调用链列表

        Returns:
            排序后的(quality_score, chain)元组列表
        """
        scored_chains = []

        for chain in chains:
            quality = self.calculate_quality(chain)
            scored_chains.append((quality, chain))

        # 按质量分数降序排序
        scored_chains.sort(key=lambda x: x[0], reverse=True)

        return scored_chains

    def filter_top_chains(self, chains: List[List[str]], top_k: int = 8) -> List[List[str]]:
        """
        筛选质量最高的K条调用链

        Args:
            chains: 调用链列表
            top_k: 返回的链数量

        Returns:
            质量最高的K条调用链
        """
        if not chains:
            return []

        ranked = self.rank_chains(chains)
        return [chain for _, chain in ranked[:top_k]]
