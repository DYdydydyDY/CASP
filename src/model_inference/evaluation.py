"""
Evaluation Module
评估指标计算
"""

import logging
import numpy as np
import difflib
from typing import List, Dict
from ..utils.helpers import SubtokenMetrics

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """评估指标计算器"""

    @staticmethod
    def exact_match_accuracy(predictions: List[str], references: List[str]) -> float:
        """计算精确匹配准确率"""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")

        if len(predictions) == 0:
            return 0.0

        matches = sum(
            1 for pred, ref in zip(predictions, references)
            if pred.strip().lower() == ref.strip().lower()
        )
        return matches / len(predictions)

    @staticmethod
    def similarity_score(pred: str, ref: str) -> float:
        """计算字符串相似度(基于序列匹配)"""
        return difflib.SequenceMatcher(None, pred.lower(), ref.lower()).ratio()

    @staticmethod
    def average_similarity(predictions: List[str], references: List[str]) -> float:
        """计算平均相似度"""
        if len(predictions) != len(references) or len(predictions) == 0:
            return 0.0

        similarities = [
            EvaluationMetrics.similarity_score(pred, ref)
            for pred, ref in zip(predictions, references)
        ]
        return float(np.mean(similarities))

    @staticmethod
    def edit_distance(s1: str, s2: str) -> int:
        """计算编辑距离(Levenshtein距离)"""
        if len(s1) < len(s2):
            return EvaluationMetrics.edit_distance(s2, s1)

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
        if len(predictions) != len(references) or len(predictions) == 0:
            return 0.0

        distances = []
        for pred, ref in zip(predictions, references):
            distance = EvaluationMetrics.edit_distance(pred, ref)
            max_len = max(len(pred), len(ref))
            normalized_distance = distance / max_len if max_len > 0 else 0
            distances.append(1 - normalized_distance)  # 转换为相似度

        return float(np.mean(distances))

    @staticmethod
    def subtoken_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        计算子词级别的Precision, Recall, F1

        Returns:
            包含precision, recall, f1的字典
        """
        if len(predictions) != len(references) or len(predictions) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        n = len(predictions)

        for pred, ref in zip(predictions, references):
            pred_tokens = SubtokenMetrics.tokenize_name(pred)
            ref_tokens = SubtokenMetrics.tokenize_name(ref)

            p = SubtokenMetrics.calculate_precision(pred_tokens, ref_tokens)
            r = SubtokenMetrics.calculate_recall(pred_tokens, ref_tokens)
            f1 = SubtokenMetrics.calculate_f1(p, r)

            total_precision += p
            total_recall += r
            total_f1 += f1

        return {
            "precision": total_precision / n,
            "recall": total_recall / n,
            "f1": total_f1 / n
        }

    @staticmethod
    def compute_all_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        计算所有评估指标

        Returns:
            包含所有指标的字典
        """
        if not predictions or not references:
            return {
                "exact_match": 0.0,
                "avg_similarity": 0.0,
                "norm_edit_distance": 0.0,
                "subtoken_precision": 0.0,
                "subtoken_recall": 0.0,
                "subtoken_f1": 0.0,
            }

        # 基础指标
        exact_match = EvaluationMetrics.exact_match_accuracy(predictions, references)
        avg_similarity = EvaluationMetrics.average_similarity(predictions, references)
        norm_edit_dist = EvaluationMetrics.normalized_edit_distance(predictions, references)

        # 子词指标
        subtoken_res = EvaluationMetrics.subtoken_metrics(predictions, references)

        return {
            "exact_match": exact_match,
            "avg_similarity": avg_similarity,
            "norm_edit_distance": norm_edit_dist,
            "subtoken_precision": subtoken_res["precision"],
            "subtoken_recall": subtoken_res["recall"],
            "subtoken_f1": subtoken_res["f1"],
        }


class ResultAnalyzer:
    """结果分析器"""

    @staticmethod
    def analyze_similarity_distribution(results: List[Dict]) -> Dict:
        """分析相似度分布"""
        if not results:
            return {
                "high_similarity_count": 0,
                "medium_similarity_count": 0,
                "low_similarity_count": 0,
                "high_similarity_ratio": 0.0,
                "medium_similarity_ratio": 0.0,
                "low_similarity_ratio": 0.0,
            }

        similarity_scores = [r.get("similarity_score", 0) for r in results]
        total = len(similarity_scores)

        high_sim = sum(1 for s in similarity_scores if s >= 0.8)
        medium_sim = sum(1 for s in similarity_scores if 0.5 <= s < 0.8)
        low_sim = sum(1 for s in similarity_scores if s < 0.5)

        return {
            "high_similarity_count": high_sim,
            "medium_similarity_count": medium_sim,
            "low_similarity_count": low_sim,
            "high_similarity_ratio": high_sim / total if total > 0 else 0,
            "medium_similarity_ratio": medium_sim / total if total > 0 else 0,
            "low_similarity_ratio": low_sim / total if total > 0 else 0,
        }

    @staticmethod
    def get_best_cases(results: List[Dict], n: int = 5) -> List[Dict]:
        """获取最佳案例(精确匹配)"""
        best_cases = [r for r in results if r.get("exact_match", False)]
        return best_cases[:n]

    @staticmethod
    def get_high_similarity_cases(results: List[Dict], n: int = 5) -> List[Dict]:
        """获取高相似度但未精确匹配的案例"""
        high_sim = [
            r for r in results
            if not r.get("exact_match", False) and r.get("similarity_score", 0) >= 0.8
        ]
        high_sim.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return high_sim[:n]

    @staticmethod
    def get_worst_cases(results: List[Dict], n: int = 5) -> List[Dict]:
        """获取最差案例(低相似度)"""
        worst = [r for r in results if r.get("similarity_score", 1.0) < 0.5]
        worst.sort(key=lambda x: x.get("similarity_score", 1.0))
        return worst[:n]
