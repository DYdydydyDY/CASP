"""
Global Symbol Table and Semantic Propagation Module
实现论文中的依赖感知语义传播机制
"""

import re
import csv
import logging
from typing import Dict, Optional, Set
from pathlib import Path

from ..utils.helpers import format_hex_address

logger = logging.getLogger(__name__)


class GlobalSymbolTable:
    """
    全局符号表

    维护地址到函数名的映射,支持持久化和语义传播
    """

    def __init__(self, output_path: str):
        """
        初始化全局符号表

        Args:
            output_path: CSV输出文件路径
        """
        self.symbol_map: Dict[str, str] = {}  # Address -> Function Name
        self.output_csv = Path(output_path)
        self.headers = ["Address", "Resolved_Name", "Confidence"]

        # 初始化CSV文件
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

        logger.info(f"Global Symbol Table initialized: {self.output_csv}")

    def update(self, address: str, function_name: str, confidence: float = 1.0):
        """
        更新符号表中的函数名

        Args:
            address: 函数地址(十六进制字符串)
            function_name: 推断的函数名
            confidence: 置信度(0-1)
        """
        if not address or not function_name:
            return

        # 规范化地址格式
        norm_addr = format_hex_address(address)

        # 更新内存映射
        self.symbol_map[norm_addr] = function_name

        # 追加到CSV
        with open(self.output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([f"0x{norm_addr}", function_name, f"{confidence:.3f}"])

        logger.debug(f"Symbol table updated: 0x{norm_addr} -> {function_name}")

    def get(self, address: str) -> Optional[str]:
        """
        查询符号表

        Args:
            address: 函数地址

        Returns:
            函数名,如果不存在返回None
        """
        norm_addr = format_hex_address(address)
        return self.symbol_map.get(norm_addr)

    def contains(self, address: str) -> bool:
        """检查地址是否已解析"""
        norm_addr = format_hex_address(address)
        return norm_addr in self.symbol_map

    def size(self) -> int:
        """返回已解析符号的数量"""
        return len(self.symbol_map)

    def resolve_context(self, text: str) -> str:
        """
        在文本中进行语义传播:
        将已知的 sub_XXXX / func_XXXX 替换为解析出的函数名

        这是语义传播的核心机制

        Args:
            text: 原始文本(通常是caller context)

        Returns:
            替换后的文本
        """
        if not self.symbol_map or not text:
            return text

        # 匹配 sub_ / func_ / loc_ + 十六进制地址
        pattern = r'\b(?:sub_|func_|loc_)([0-9A-Fa-f]+)\b'

        def replace_match(match):
            addr_str = match.group(1).upper()
            if addr_str in self.symbol_map:
                resolved_name = self.symbol_map[addr_str]
                logger.debug(f"Propagated: {match.group(0)} -> {resolved_name}")
                return resolved_name
            return match.group(0)

        resolved_text = re.sub(pattern, replace_match, text)
        return resolved_text

    def export_summary(self, output_path: str):
        """导出符号表摘要"""
        summary_path = Path(output_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Global Symbol Table Summary\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Total resolved symbols: {len(self.symbol_map)}\n\n")
            f.write(f"{'Address':<20} {'Function Name':<40}\n")
            f.write(f"{'-'*60}\n")

            for addr, name in sorted(self.symbol_map.items()):
                f.write(f"0x{addr:<18} {name:<40}\n")

        logger.info(f"Symbol table summary exported to {summary_path}")


class SemanticPropagator:
    """
    语义传播器

    实现依赖感知的语义传播算法
    """

    def __init__(self, symbol_table: GlobalSymbolTable):
        """
        初始化语义传播器

        Args:
            symbol_table: 全局符号表实例
        """
        self.symbol_table = symbol_table
        self.propagation_log: list = []  # 记录传播历史

    def enrich_function_data(self, function_data: Dict) -> Dict:
        """
        对单个函数数据进行语义丰富化

        更新所有包含函数调用的文本字段

        Args:
            function_data: 函数数据字典

        Returns:
            丰富化后的函数数据
        """
        enriched_data = function_data.copy()

        # 1. 丰富函数体
        if 'function_body' in enriched_data:
            original_body = enriched_data['function_body']
            enriched_body = self.symbol_table.resolve_context(original_body)
            enriched_data['function_body'] = enriched_body

            if original_body != enriched_body:
                self._log_propagation(
                    function_data.get('func_ea', 'unknown'),
                    'function_body',
                    self._count_replacements(original_body, enriched_body)
                )

        # 2. 丰富调用者上下文
        if 'caller_context' in enriched_data:
            contexts = enriched_data['caller_context']
            if isinstance(contexts, list):
                for ctx in contexts:
                    # 丰富汇编上下文
                    if 'context_asm' in ctx and isinstance(ctx['context_asm'], list):
                        ctx['context_asm'] = [
                            self.symbol_table.resolve_context(line)
                            for line in ctx['context_asm']
                        ]

                    # 丰富反编译上下文
                    if 'context_decompiled' in ctx and isinstance(ctx['context_decompiled'], list):
                        ctx['context_decompiled'] = [
                            self.symbol_table.resolve_context(line)
                            for line in ctx['context_decompiled']
                        ]

        # 3. 丰富调用链
        if 'call_chain' in enriched_data:
            chains = enriched_data['call_chain']
            if isinstance(chains, list):
                enriched_chains = []
                for chain in chains:
                    if isinstance(chain, list):
                        enriched_chain = []
                        for func_name in chain:
                            # 尝试解析sub_XXXX格式的名称
                            enriched_name = self.symbol_table.resolve_context(func_name)
                            enriched_chain.append(enriched_name)
                        enriched_chains.append(enriched_chain)
                    else:
                        enriched_chains.append(chain)
                enriched_data['call_chain'] = enriched_chains

        return enriched_data

    def _count_replacements(self, original: str, enriched: str) -> int:
        """统计进行了多少次替换"""
        # 简单估算:查找sub_模式的数量差异
        original_count = len(re.findall(r'\bsub_[0-9A-Fa-f]+\b', original))
        enriched_count = len(re.findall(r'\bsub_[0-9A-Fa-f]+\b', enriched))
        return max(0, original_count - enriched_count)

    def _log_propagation(self, func_addr: str, field: str, replacements: int):
        """记录传播日志"""
        log_entry = {
            'function': func_addr,
            'field': field,
            'replacements': replacements
        }
        self.propagation_log.append(log_entry)

    def get_propagation_stats(self) -> Dict:
        """获取传播统计信息"""
        total_replacements = sum(entry['replacements'] for entry in self.propagation_log)
        affected_functions = len(set(entry['function'] for entry in self.propagation_log))

        return {
            'total_replacements': total_replacements,
            'affected_functions': affected_functions,
            'propagation_events': len(self.propagation_log),
            'symbol_table_size': self.symbol_table.size()
        }

    def export_propagation_log(self, output_path: str):
        """导出传播日志"""
        import json
        log_path = Path(output_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({
                'log': self.propagation_log,
                'stats': self.get_propagation_stats()
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Propagation log exported to {log_path}")
