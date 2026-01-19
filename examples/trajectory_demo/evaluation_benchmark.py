"""
Collision Detection Evaluation Benchmark

Purpose:
  Generate standardized evaluation metrics across multiple test videos
  Support ground truth annotation and accuracy calculation

Usage:
  python evaluation_benchmark.py --results-dir <result_dir> --ground-truth <gt_file>
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


class EvaluationBenchmark:
    """
    Standardized evaluation for collision detection system
    """
    
    def __init__(self, results_dir: str, ground_truth_file: str = None):
        """
        Args:
            results_dir: Path to pipeline output directory
            ground_truth_file: JSON file with manual annotations (optional)
        """
        self.results_dir = Path(results_dir)
        self.collision_events_file = self.results_dir / "5_collision_analysis" / "collision_events.json"
        self.ground_truth = self._load_ground_truth(ground_truth_file) if ground_truth_file else None
        
    def _load_ground_truth(self, gt_file: str) -> List[Dict]:
        """Load manual ground truth annotations"""
        with open(gt_file) as f:
            data = json.load(f)
            # Handle both formats: direct list or dict with 'ground_truth_events' key
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'ground_truth_events' in data:
                return data['ground_truth_events']
            else:
                return []
    
    def _load_detected_events(self) -> List[Dict]:
        """Load automatically detected events"""
        if not self.collision_events_file.exists():
            print(f"❌ 未找到检测结果: {self.collision_events_file}")
            return []
        
        with open(self.collision_events_file) as f:
            return json.load(f)
    
    def evaluate_by_level(self) -> Dict:
        """
        Calculate metrics for each risk level separately
        
        Levels:
          1: Collision (TTC < 1.0s or distance < 0.5m)
          2: Near Miss (0.5-1.5m, TTC < 3.0s)
          3: Avoidance (> 1.5m)
        """
        detected = self._load_detected_events()
        
        if not detected:
            return {"status": "no_detections", "message": "未检测到事件"}
        
        results = {}
        
        # 按等级统计
        for level in [1, 2, 3]:
            level_detected = [e for e in detected if e.get('risk_level') == level]
            
            results[f"Level_{level}"] = {
                "detected_count": len(level_detected),
                "events": level_detected,
                "status": "detected" if level_detected else "no_events"
            }
        
        # 总体统计
        results["summary"] = {
            "total_events": len(detected),
            "level_1_count": len([e for e in detected if e.get('risk_level') == 1]),
            "level_2_count": len([e for e in detected if e.get('risk_level') == 2]),
            "level_3_count": len([e for e in detected if e.get('risk_level') == 3]),
        }
        
        return results
    
    def compare_with_ground_truth(self) -> Dict:
        """
        Compare detected events with ground truth
        Requires ground_truth to be provided
        """
        if not self.ground_truth:
            return {"status": "no_ground_truth", "message": "未提供标注数据"}
        
        detected = self._load_detected_events()
        
        # 按帧号匹配
        def frame_match(det, gt):
            return det.get('frame') == gt.get('frame')
        
        tp = 0  # True Positive
        fp = 0  # False Positive
        fn = 0  # False Negative
        
        for det in detected:
            if any(frame_match(det, gt) for gt in self.ground_truth):
                tp += 1
            else:
                fp += 1
        
        for gt in self.ground_truth:
            if not any(frame_match(det, gt) for det in detected):
                fn += 1
        
        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": f"{precision:.2%}",
            "recall": f"{recall:.2%}",
            "f1_score": f"{f1:.2%}",
            "ground_truth_total": len(self.ground_truth),
            "detected_total": len(detected),
        }
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate comprehensive evaluation report"""
        
        report = []
        report.append("=" * 70)
        report.append("COLLISION DETECTION EVALUATION REPORT")
        report.append(f"Test Video: {self.results_dir.name}")
        report.append("=" * 70)
        report.append("")
        
        # 按等级的统计
        report.append("【Detection by Risk Level】\n")
        level_results = self.evaluate_by_level()
        if "summary" in level_results:
            summary = level_results["summary"]
            report.append(f"Level 1 (Critical Risk, TTC < 1s): {summary['level_1_count']} events")
            report.append(f"Level 2 (Near Miss, 0.5-1.5m): {summary['level_2_count']} events")
            report.append(f"Level 3 (Avoidance, > 1.5m): {summary['level_3_count']} events")
            report.append(f"Total: {summary['total_events']} events\n")
        
        # 与标注数据的对比
        if self.ground_truth:
            report.append("【Accuracy Metrics (vs Ground Truth)】\n")
            acc_results = self.compare_with_ground_truth()
            report.append(f"True Positives: {acc_results['true_positives']}")
            report.append(f"False Positives: {acc_results['false_positives']}")
            report.append(f"False Negatives: {acc_results['false_negatives']}")
            report.append(f"Precision: {acc_results['precision']}")
            report.append(f"Recall: {acc_results['recall']}")
            report.append(f"F1 Score: {acc_results['f1_score']}\n")
        else:
            report.append("【Accuracy Metrics】")
            report.append("⚠️  Ground truth not available")
            report.append("   To enable accuracy metrics, provide annotated ground_truth.json\n")
        
        report_text = "\n".join(report)
        
        # 保存到文件
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_text)
            print(f"✓ 报告已保存: {output_file}")
        
        return report_text


def main():
    parser = argparse.ArgumentParser(description="Collision Detection Evaluation Benchmark")
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Pipeline output directory')
    parser.add_argument('--ground-truth', type=str, default=None,
                       help='Ground truth annotation JSON file (optional)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output report file path')
    
    args = parser.parse_args()
    
    benchmark = EvaluationBenchmark(args.results_dir, args.ground_truth)
    report = benchmark.generate_report(args.output)
    print(report)


if __name__ == "__main__":
    main()
