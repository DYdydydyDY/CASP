"""
CASP Pipeline - Context-Aware Semantic Propagation for Function Name Recovery

主Pipeline程序,实现完整的CASP框架流程:
1. 加载函数数据
2. 计算SRS并排序
3. 迭代推理和语义传播
4. 评估和结果输出
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from config.config import CASPConfig
from src.utils import setup_logger
from src.semantic_propagation import (
    SRSCalculator,
    GlobalSymbolTable,
    SemanticPropagator
)
from src.model_inference import (
    QwenInferenceEngine,
    PromptBuilder,
    EvaluationMetrics,
    ResultAnalyzer
)

logger = logging.getLogger(__name__)


class CASPPipeline:
    """CASP主Pipeline"""

    def __init__(self, config: CASPConfig):
        """
        初始化Pipeline

        Args:
            config: CASP配置对象
        """
        self.config = config

        # 初始化组件
        self.srs_calculator = SRSCalculator(
            alpha=config.srs.alpha,
            beta=config.srs.beta,
            gamma=config.srs.gamma
        )

        self.symbol_table = None
        self.propagator = None
        self.inference_engine = None

        logger.info("CASP Pipeline initialized")

    def load_data(self, data_path: str) -> List[Dict]:
        """
        加载函数数据

        Args:
            data_path: 数据文件路径(JSONL格式)

        Returns:
            函数数据列表
        """
        logger.info(f"Loading data from {data_path}")

        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

        logger.info(f"Loaded {len(data)} function records")
        return data

    def initialize_components(self, output_dir: Path):
        """
        初始化运行时组件

        Args:
            output_dir: 输出目录
        """
        # 初始化符号表
        symbol_table_path = output_dir / self.config.propagation.symbol_table_csv
        self.symbol_table = GlobalSymbolTable(str(symbol_table_path))

        # 初始化传播器
        self.propagator = SemanticPropagator(self.symbol_table)

        # 初始化推理引擎
        self.inference_engine = QwenInferenceEngine(
            base_model_path=self.config.model.base_model_path,
            lora_adapter_path=self.config.model.lora_adapter_path,
            device=self.config.model.device,
            dtype=self.config.model.dtype
        )
        self.inference_engine.load_model()

        logger.info("All components initialized")

    def run(
        self,
        data_path: str,
        output_dir: Optional[str] = None,
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        运行完整的CASP Pipeline

        Args:
            data_path: 输入数据路径
            output_dir: 输出目录
            sample_size: 采样大小(用于测试)

        Returns:
            评估结果摘要
        """
        # 准备输出目录
        if output_dir is None:
            output_dir = self.config.output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # 1. 加载数据
        functions = self.load_data(data_path)

        if sample_size and sample_size < len(functions):
            import random
            random.seed(self.config.evaluation.random_seed)
            functions = random.sample(functions, sample_size)
            logger.info(f"Sampled {sample_size} functions for evaluation")

        # 2. 初始化组件
        self.initialize_components(output_dir)

        # 3. 计算SRS并排序
        logger.info("Calculating SRS scores and sorting functions...")
        ranked_functions = self.srs_calculator.rank_functions(functions)
        logger.info(f"Functions ranked by SRS (descending order)")

        # 4. 迭代推理和传播
        logger.info("Starting iterative inference and semantic propagation...")
        results = self._iterative_inference(ranked_functions)

        # 5. 评估
        summary = self._evaluate_results(results, output_dir)

        # 6. 保存结果
        self._save_results(results, summary, output_dir)

        # 7. 导出符号表和传播日志
        self.symbol_table.export_summary(str(output_dir / "symbol_table_summary.txt"))
        self.propagator.export_propagation_log(str(output_dir / "propagation_log.json"))

        logger.info("Pipeline completed successfully")
        return summary

    def _iterative_inference(self, ranked_functions: List[tuple]) -> List[Dict]:
        """
        迭代推理和语义传播

        Args:
            ranked_functions: SRS排序后的函数列表

        Returns:
            推理结果列表
        """
        results = []
        predictions = []
        ground_truths = []

        logger.info(f"Processing {len(ranked_functions)} functions...")

        for i, (srs_score, func_data) in enumerate(tqdm(ranked_functions, desc="Inference")):
            try:
                # Step 1: 语义传播 - 丰富化上下文
                enriched_data = self.propagator.enrich_function_data(func_data)

                # Step 2: 构建Prompt
                prompt = PromptBuilder.build_chatml_prompt(
                    function_body=enriched_data.get('function_body', ''),
                    caller_context=enriched_data.get('caller_context'),
                    string_arguments=enriched_data.get('string_arguments'),
                    call_chains=enriched_data.get('call_chain'),
                    system_prompt=self.config.model.system_prompt
                )

                # Step 3: 模型推理
                prediction = self.inference_engine.generate(
                    prompt=prompt,
                    max_new_tokens=self.config.model.max_new_tokens,
                    temperature=self.config.model.temperature,
                    top_p=self.config.model.top_p,
                    top_k=self.config.model.top_k,
                    repetition_penalty=self.config.model.repetition_penalty
                )

                # Step 4: 更新符号表
                func_addr = func_data.get('func_ea', '')
                if func_addr and prediction:
                    self.symbol_table.update(func_addr, prediction)

                # Step 5: 收集结果
                ground_truth = func_data.get('real_name', '') or func_data.get('output', '')

                if ground_truth:
                    predictions.append(prediction)
                    ground_truths.append(ground_truth)

                    # 计算相似度
                    similarity = EvaluationMetrics.similarity_score(prediction, ground_truth)
                    exact_match = prediction.strip().lower() == ground_truth.strip().lower()
                else:
                    similarity = 0.0
                    exact_match = False

                result = {
                    "index": i,
                    "func_ea": func_addr,
                    "srs_score": srs_score,
                    "dummy_name": func_data.get('dummy_name', ''),
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "exact_match": exact_match,
                    "similarity_score": similarity,
                }

                results.append(result)

            except Exception as e:
                logger.error(f"Error processing function {i}: {e}", exc_info=True)
                continue

        logger.info(f"Inference completed: {len(results)} results")
        return results

    def _evaluate_results(self, results: List[Dict], output_dir: Path) -> Dict:
        """
        评估结果

        Args:
            results: 推理结果列表
            output_dir: 输出目录

        Returns:
            评估摘要
        """
        logger.info("Evaluating results...")

        # 提取预测和真值
        valid_results = [r for r in results if r.get('ground_truth')]
        predictions = [r['prediction'] for r in valid_results]
        ground_truths = [r['ground_truth'] for r in valid_results]

        if not predictions:
            logger.warning("No valid predictions for evaluation")
            return {
                "total_functions": len(results),
                "valid_evaluations": 0,
            }

        # 计算所有指标
        metrics = EvaluationMetrics.compute_all_metrics(predictions, ground_truths)

        # 分析相似度分布
        distribution = ResultAnalyzer.analyze_similarity_distribution(valid_results)

        # 传播统计
        propagation_stats = self.propagator.get_propagation_stats()

        summary = {
            "total_functions": len(results),
            "valid_evaluations": len(valid_results),
            "exact_match_count": sum(1 for r in valid_results if r['exact_match']),
            "exact_match_rate": metrics["exact_match"],
            "subtoken_precision": metrics["subtoken_precision"],
            "subtoken_recall": metrics["subtoken_recall"],
            "subtoken_f1": metrics["subtoken_f1"],
            "average_similarity": metrics["avg_similarity"],
            "normalized_edit_distance": metrics["norm_edit_distance"],
            "similarity_distribution": distribution,
            "propagation_stats": propagation_stats,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Evaluation Summary:")
        logger.info(f"  - Exact Match Rate: {summary['exact_match_rate']:.4f}")
        logger.info(f"  - Sub-token F1: {summary['subtoken_f1']:.4f}")
        logger.info(f"  - Average Similarity: {summary['average_similarity']:.4f}")
        logger.info(f"  - Propagation: {propagation_stats['total_replacements']} replacements")

        return summary

    def _save_results(self, results: List[Dict], summary: Dict, output_dir: Path):
        """
        保存结果

        Args:
            results: 详细结果列表
            summary: 评估摘要
            output_dir: 输出目录
        """
        # 保存详细结果
        with open(output_dir / "detailed_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 保存评估摘要
        with open(output_dir / "evaluation_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 生成Markdown报告
        self._generate_report(results, summary, output_dir)

        logger.info(f"Results saved to {output_dir}")

    def _generate_report(self, results: List[Dict], summary: Dict, output_dir: Path):
        """生成Markdown格式的评估报告"""
        report_path = output_dir / "evaluation_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# CASP Evaluation Report\n\n")
            f.write(f"**Timestamp**: {summary['timestamp']}\n\n")

            # 总体指标
            f.write("## Overall Metrics\n\n")
            f.write(f"- **Total Functions**: {summary['total_functions']}\n")
            f.write(f"- **Valid Evaluations**: {summary['valid_evaluations']}\n")
            f.write(f"- **Exact Match Rate**: {summary['exact_match_rate']:.4f} ({summary['exact_match_rate']*100:.2f}%)\n")
            f.write(f"- **Sub-token Precision**: {summary['subtoken_precision']:.4f}\n")
            f.write(f"- **Sub-token Recall**: {summary['subtoken_recall']:.4f}\n")
            f.write(f"- **Sub-token F1**: {summary['subtoken_f1']:.4f}\n")
            f.write(f"- **Average Similarity**: {summary['average_similarity']:.4f}\n")
            f.write(f"- **Normalized Edit Distance**: {summary['normalized_edit_distance']:.4f}\n\n")

            # 相似度分布
            dist = summary.get('similarity_distribution', {})
            f.write("## Similarity Distribution\n\n")
            f.write(f"- **High (≥0.8)**: {dist.get('high_similarity_count', 0)} ({dist.get('high_similarity_ratio', 0)*100:.1f}%)\n")
            f.write(f"- **Medium (0.5-0.8)**: {dist.get('medium_similarity_count', 0)} ({dist.get('medium_similarity_ratio', 0)*100:.1f}%)\n")
            f.write(f"- **Low (<0.5)**: {dist.get('low_similarity_count', 0)} ({dist.get('low_similarity_ratio', 0)*100:.1f}%)\n\n")

            # 传播统计
            prop_stats = summary.get('propagation_stats', {})
            f.write("## Semantic Propagation Statistics\n\n")
            f.write(f"- **Total Replacements**: {prop_stats.get('total_replacements', 0)}\n")
            f.write(f"- **Affected Functions**: {prop_stats.get('affected_functions', 0)}\n")
            f.write(f"- **Symbol Table Size**: {prop_stats.get('symbol_table_size', 0)}\n\n")

            # 示例案例
            f.write("## Example Cases\n\n")

            # 成功案例
            f.write("### Success Cases (Exact Match)\n\n")
            best_cases = ResultAnalyzer.get_best_cases(results, n=5)
            for i, case in enumerate(best_cases, 1):
                f.write(f"**Example {i}**:\n")
                f.write(f"- SRS: {case['srs_score']:.2f}\n")
                f.write(f"- Ground Truth: `{case['ground_truth']}`\n")
                f.write(f"- Prediction: `{case['prediction']}`\n\n")

            # 高相似度案例
            f.write("### High Similarity Cases (Not Exact Match)\n\n")
            high_sim = ResultAnalyzer.get_high_similarity_cases(results, n=5)
            for i, case in enumerate(high_sim, 1):
                f.write(f"**Example {i}**:\n")
                f.write(f"- SRS: {case['srs_score']:.2f}\n")
                f.write(f"- Ground Truth: `{case['ground_truth']}`\n")
                f.write(f"- Prediction: `{case['prediction']}`\n")
                f.write(f"- Similarity: {case['similarity_score']:.4f}\n\n")

            # 需要改进的案例
            f.write("### Cases Needing Improvement (Low Similarity)\n\n")
            worst = ResultAnalyzer.get_worst_cases(results, n=5)
            for i, case in enumerate(worst, 1):
                f.write(f"**Example {i}**:\n")
                f.write(f"- SRS: {case['srs_score']:.2f}\n")
                f.write(f"- Ground Truth: `{case['ground_truth']}`\n")
                f.write(f"- Prediction: `{case['prediction']}`\n")
                f.write(f"- Similarity: {case['similarity_score']:.4f}\n\n")

        logger.info(f"Report generated: {report_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="CASP - Context-Aware Semantic Propagation for Function Name Recovery")
    parser.add_argument("--data", required=True, help="Input data file (JSONL)")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--model", help="Base model path (overrides config)")
    parser.add_argument("--lora", help="LoRA adapter path (overrides config)")
    parser.add_argument("--sample", type=int, help="Sample size for testing")
    parser.add_argument("--config", help="Custom config file (YAML)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # 加载配置
    if args.config:
        config = CASPConfig.from_yaml(args.config)
    else:
        config = CASPConfig()

    # 命令行参数覆盖
    if args.model:
        config.model.base_model_path = args.model
    if args.lora:
        config.model.lora_adapter_path = args.lora

    # 设置日志
    log_file = config.output_dir / "casp_pipeline.log" if config.log_file is None else config.log_file
    setup_logger("casp", level=args.log_level, log_file=str(log_file))

    logger.info("="*60)
    logger.info("CASP Pipeline Starting")
    logger.info("="*60)

    # 运行Pipeline
    pipeline = CASPPipeline(config)
    summary = pipeline.run(
        data_path=args.data,
        output_dir=args.output,
        sample_size=args.sample
    )

    logger.info("="*60)
    logger.info("CASP Pipeline Completed Successfully")
    logger.info(f"Results saved to: {args.output or config.output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
