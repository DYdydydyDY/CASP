"""
CASP Utilities Module
通用工具函数和辅助类
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path


def setup_logger(name: str, level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class CodeCleaner:
    """反编译代码清理器"""

    @staticmethod
    def clean_decompiled_code(code: str) -> str:
        """
        清理反编译代码,去除局部变量声明和无意义的注释

        Args:
            code: 原始反编译代码

        Returns:
            清理后的代码
        """
        if not code:
            return code

        lines = code.split('\n')
        cleaned_lines = []
        in_var_declaration = False
        brace_count = 0

        for line in lines:
            stripped = line.strip()

            # 跳过空行
            if not stripped:
                continue

            # 检测大括号
            if '{' in line:
                brace_count += line.count('{')
                if brace_count == 1 and not in_var_declaration:
                    in_var_declaration = True
                    cleaned_lines.append(line)
                    continue

            if '}' in line:
                brace_count -= line.count('}')
                in_var_declaration = False
                cleaned_lines.append(line)
                continue

            # 在函数体内,跳过局部变量声明
            if in_var_declaration and brace_count == 1:
                var_patterns = [
                    r'^\s*(int|long|short|char|unsigned|signed|void|bool|float|double|size_t|__int\d+)\s+',
                    r'^\s*(struct|union|enum)\s+\w+\s+',
                    r'^\s*\w+\s+\*?\w+\s*[;=]',
                ]

                is_var_decl = False
                for pattern in var_patterns:
                    if re.match(pattern, stripped):
                        # 但如果包含函数调用,保留
                        if '(' in stripped and ')' in stripped:
                            break
                        is_var_decl = True
                        break

                if is_var_decl:
                    continue

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    @staticmethod
    def normalize_variable_names(code: str) -> str:
        """规范化变量名"""
        # 规范化 Hex-Rays 变量名
        code = re.sub(r'\bv\d+\b', 'VAR', code)
        code = re.sub(r'\ba\d+\b', 'ARG', code)
        code = re.sub(r'\barg_\d+\b', 'ARG', code)
        return code


class StringValidator:
    """字符串验证器"""

    @staticmethod
    def is_meaningful_string(s: str, min_length: int = 3, max_length: int = 300) -> bool:
        """
        判断字符串是否有意义

        Args:
            s: 待验证字符串
            min_length: 最小长度
            max_length: 最大长度

        Returns:
            是否有意义
        """
        if not s or len(s) < min_length or len(s) > max_length:
            return False

        # 过滤只包含单一字符的字符串
        if len(set(s)) <= 1:
            return False

        # 过滤常见的系统字符串和格式字符串
        meaningless_patterns = [
            r'^[%@\?\._\-\s]+$',  # 只包含格式字符
            r'^[0-9\s]+$',        # 只包含数字
            r'^[A-Z_]{1,3}$',     # 短的大写缩写
            r'^\s+$',             # 空白字符
            r'^[xX0]+$',          # 只有x或0
            r'^\.+$',             # 只有点
            r'^_+$',              # 只有下划线
        ]

        for pattern in meaningless_patterns:
            if re.match(pattern, s):
                return False

        # 检查是否包含足够的字母字符
        alpha_count = sum(c.isalpha() for c in s)
        if alpha_count < 2:
            return False

        # 过滤常见的无意义系统字符串
        meaningless_strings = {
            'NULL', 'null', 'nullptr',
            'true', 'false', 'TRUE', 'FALSE',
            'yes', 'no', 'YES', 'NO',
            'void',
        }

        if s in meaningless_strings:
            return False

        return True


class FunctionNameValidator:
    """函数名验证器"""

    DUMMY_PATTERNS = [
        r'^sub_[0-9A-F]+$',  # 仅有地址的函数
        r'^nullsub_',        # 空函数
        r'^j_j_',            # 多层跳转
        r'^_?_?thunk_',      # thunk函数
        r'^loc_[0-9A-F]+$',  # 位置标签
        r'^def_[0-9A-F]+$',  # 默认名称
    ]

    @staticmethod
    def is_dummy_function(func_name: str) -> bool:
        """判断是否是无意义的哑函数"""
        if not func_name:
            return True

        for pattern in FunctionNameValidator.DUMMY_PATTERNS:
            if re.match(pattern, func_name, re.IGNORECASE):
                return True

        return False

    @staticmethod
    def is_meaningful_function(func_name: str) -> bool:
        """判断是否是有意义的函数名"""
        return not FunctionNameValidator.is_dummy_function(func_name)


class SubtokenMetrics:
    """子词级别的评估指标"""

    @staticmethod
    def tokenize_name(name: str) -> Set[str]:
        """
        将函数名分解为子词

        处理驼峰命名和下划线命名
        例如: getUserInfo -> {get, user, info}
              get_user_info -> {get, user, info}
        """
        # 在大写字母前添加空格(处理驼峰)
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
        # 将下划线替换为空格
        s3 = s2.replace('_', ' ')
        # 分割、转小写、去重
        return set(token.lower() for token in s3.split() if token)

    @staticmethod
    def calculate_precision(pred_tokens: Set[str], ref_tokens: Set[str]) -> float:
        """计算精确率"""
        if len(pred_tokens) == 0:
            return 0.0
        return len(pred_tokens & ref_tokens) / len(pred_tokens)

    @staticmethod
    def calculate_recall(pred_tokens: Set[str], ref_tokens: Set[str]) -> float:
        """计算召回率"""
        if len(ref_tokens) == 0:
            return 0.0
        return len(pred_tokens & ref_tokens) / len(ref_tokens)

    @staticmethod
    def calculate_f1(precision: float, recall: float) -> float:
        """计算F1分数"""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


def format_hex_address(addr: Any) -> str:
    """格式化十六进制地址"""
    if isinstance(addr, str):
        # 移除0x前缀并转大写
        return addr.replace("0x", "").replace("0X", "").upper()
    elif isinstance(addr, int):
        return f"{addr:X}"
    else:
        return str(addr)


def parse_address(addr_str: str) -> Optional[int]:
    """解析地址字符串为整数"""
    try:
        if addr_str.startswith(("0x", "0X")):
            return int(addr_str, 16)
        else:
            return int(addr_str, 16)  # 假设是十六进制
    except (ValueError, AttributeError):
        return None
