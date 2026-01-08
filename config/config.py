"""
CASP Configuration Module
集中管理所有系统配置参数
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class FeatureExtractionConfig:
    """特征提取配置"""
    # 反编译参数
    min_func_size: int = 10
    decompilation_timeout: int = 300  # seconds

    # 上下文窗口
    context_lines: int = 8
    asm_context_window: int = 15
    decompiled_context_lines: int = 20

    # 调用链参数
    max_call_depth: int = 5
    max_callers: int = 50
    max_chains: int = 8

    # 字符串提取
    min_string_length: int = 3
    max_string_length: int = 300
    arg_search_range: int = 30

    # 质量过滤
    filter_dummy_functions: bool = True
    filter_thunks: bool = True


@dataclass
class SRSConfig:
    """语义丰富度评分配置"""
    # SRS公式权重: SRS = alpha * N_str + beta * N_api + gamma * Depth
    alpha: float = 2.0  # 字符串权重
    beta: float = 1.0   # API调用权重
    gamma: float = 0.5  # 调用深度权重

    # 调用链质量评分权重
    chain_semantic_weight: int = 5
    chain_length_weight: int = 10


@dataclass
class ModelConfig:
    """模型推理配置"""
    # 模型路径
    base_model_path: str = "/root/project/finetune/qwen_new/Qwen2.5-Coder-7B-Instruct"
    lora_adapter_path: Optional[str] = None
    use_lora: bool = True

    # 推理参数
    max_seq_length: int = 4096
    max_new_tokens: int = 64
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05

    # 设备配置
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Prompt模板
    system_prompt: str = """You are a reverse engineering expert specializing in binary analysis and function name recovery.
Your task is to analyze stripped binary functions and infer meaningful, descriptive function names based on:
1. Decompiled function body
2. Calling context from caller functions
3. String arguments passed at call sites
4. Hierarchical call chains

Provide a concise, snake_case or camelCase function name that best describes the function's purpose."""


@dataclass
class PropagationConfig:
    """语义传播配置"""
    # 符号表
    enable_propagation: bool = True
    symbol_table_csv: str = "global_symbol_table.csv"

    # 传播策略
    min_confidence_threshold: float = 0.3  # 最低置信度
    update_strategy: str = "greedy"  # greedy / iterative


@dataclass
class EvaluationConfig:
    """评估配置"""
    # 评估指标
    compute_exact_match: bool = True
    compute_subtoken_metrics: bool = True
    compute_similarity: bool = True
    compute_edit_distance: bool = True

    # 采样
    sample_size: Optional[int] = None
    random_seed: int = 42


@dataclass
class IDAConfig:
    """IDA Pro配置"""
    ida_path: str = r"D:\software\IDA\IDA_new\ida.exe"
    ida_script: str = "ida_feature_extractor.py"
    batch_timeout: int = 300


@dataclass
class CASPConfig:
    """主配置类 - 整合所有子配置"""
    # 路径配置
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    # 子配置
    feature_extraction: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    srs: SRSConfig = field(default_factory=SRSConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    ida: IDAConfig = field(default_factory=IDAConfig)

    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None

    def __post_init__(self):
        """初始化后处理"""
        # 确保路径存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: dict):
        """从字典创建配置"""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """从YAML文件加载配置"""
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> dict:
        """转换为字典"""
        from dataclasses import asdict
        return asdict(self)


def get_default_config() -> CASPConfig:
    """获取默认配置"""
    return CASPConfig()
