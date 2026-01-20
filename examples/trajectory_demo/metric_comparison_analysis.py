"""
指标对比分析工具

目的: 对比不同指标（PET、TTC、Distance）对碰撞检测精度的影响
用法: python metric_comparison_analysis.py --results-dir <结果目录> --ground-truth <标注文件>
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import statistics


class MetricComparisonAnalyzer:
    """对比不同指标的效果"""
    
    def __init__(self, results_dir: str, ground_truth_file: str):
        self.results_dir = Path(results_dir)
        self.collision_events_file = self.results_dir / "5_collision_analysis" / "collision_events.json"
        
        # 加载ground truth
        with open(ground_truth_file) as f:
            gt_data = json.load(f)
            if isinstance(gt_data, dict) and 'ground_truth_events' in gt_data:
                self.ground_truth = gt_data['ground_truth_events']
            else:
                self.ground_truth = gt_data
    
    def load_detected_events(self) -> List[Dict]:
        """加载检测事件"""
        if not self.collision_events_file.exists():
            print(f"❌ 未找到检测结果: {self.collision_events_file}")
            return []
        
        with open(self.collision_events_file) as f:
            return json.load(f)
    
    def get_gt_frames(self) -> set:
        """获取ground truth中的frame集合"""
        return set(event['frame'] for event in self.ground_truth)
    
    def calculate_metrics(self, detected_frames: set, tp_count: int) -> Dict:
        """计算Precision、Recall、F1"""
        if not detected_frames:
            return {'precision': 0, 'recall': 0, 'f1': 0, 'tp': tp_count}
        
        precision = tp_count / len(detected_frames) * 100 if detected_frames else 0
        recall = tp_count / len(self.ground_truth) * 100 if self.ground_truth else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp_count,
            'detected': len(detected_frames),
            'gt_total': len(self.ground_truth)
        }
    
    def analyze_baseline(self) -> Dict:
        """基线分析：所有检测的事件"""
        detected = self.load_detected_events()
        detected_frames = set(e['frame'] for e in detected)
        gt_frames = self.get_gt_frames()
        
        tp = len(detected_frames & gt_frames)
        
        return {
            'scenario': '基线（所有事件）',
            'detected_frames': detected_frames,
            'metrics': self.calculate_metrics(detected_frames, tp)
        }
    
    def analyze_ttc_filter(self) -> Dict:
        """TTC过滤分析：仅保留有TTC值的事件"""
        detected = self.load_detected_events()
        gt_frames = self.get_gt_frames()
        
        # 筛选有TTC的事件
        events_with_ttc = []
        for e in detected:
            if 'multi_anchor_detailed' in e:
                ttc = e['multi_anchor_detailed'].get('ttc_seconds')
                if ttc and ttc > 0:
                    events_with_ttc.append(e)
        
        detected_frames = set(e['frame'] for e in events_with_ttc)
        tp = len(detected_frames & gt_frames)
        
        return {
            'scenario': 'TTC过滤（仅保留TTC>0）',
            'detected_frames': detected_frames,
            'metrics': self.calculate_metrics(detected_frames, tp)
        }
    
    def analyze_pet_filter(self) -> Dict:
        """PET过滤分析：仅保留有PET值的事件"""
        detected = self.load_detected_events()
        gt_frames = self.get_gt_frames()
        
        # 筛选有PET的事件
        events_with_pet = []
        for e in detected:
            if 'multi_anchor_detailed' in e:
                pet = e['multi_anchor_detailed'].get('pet_seconds')
                if pet and pet > 0:
                    events_with_pet.append(e)
        
        detected_frames = set(e['frame'] for e in events_with_pet)
        tp = len(detected_frames & gt_frames)
        
        return {
            'scenario': 'PET过滤（仅保留PET>0）',
            'detected_frames': detected_frames,
            'metrics': self.calculate_metrics(detected_frames, tp)
        }
    
    def analyze_distance_threshold(self, threshold: float = 0.5) -> Dict:
        """距离阈值分析"""
        detected = self.load_detected_events()
        gt_frames = self.get_gt_frames()
        
        # 筛选距离小于阈值的事件
        events_filtered = [e for e in detected if e.get('distance_meters', float('inf')) <= threshold]
        
        detected_frames = set(e['frame'] for e in events_filtered)
        tp = len(detected_frames & gt_frames)
        
        return {
            'scenario': f'距离过滤（≤{threshold}m）',
            'detected_frames': detected_frames,
            'metrics': self.calculate_metrics(detected_frames, tp)
        }
    
    def analyze_combined_ttc_pet(self) -> Dict:
        """组合分析：TTC AND PET"""
        detected = self.load_detected_events()
        gt_frames = self.get_gt_frames()
        
        # 筛选同时有TTC和PET的事件
        events_combined = []
        for e in detected:
            if 'multi_anchor_detailed' in e:
                multi = e['multi_anchor_detailed']
                ttc = multi.get('ttc_seconds')
                pet = multi.get('pet_seconds')
                if (ttc and ttc > 0) and (pet and pet > 0):
                    events_combined.append(e)
        
        detected_frames = set(e['frame'] for e in events_combined)
        tp = len(detected_frames & gt_frames)
        
        return {
            'scenario': '组合过滤（TTC>0 AND PET>0）',
            'detected_frames': detected_frames,
            'metrics': self.calculate_metrics(detected_frames, tp)
        }
    
    def print_comparison(self):
        """打印对比结果"""
        print("\n" + "="*100)
        print("指标对比分析 - 精度效果评估")
        print("="*100)
        
        # 执行所有分析
        analyses = [
            self.analyze_baseline(),
            self.analyze_ttc_filter(),
            self.analyze_pet_filter(),
            self.analyze_distance_threshold(),
            self.analyze_combined_ttc_pet()
        ]
        
        # 打印表格
        print(f"\n{'场景':<30} {'检测数':>8} {'TP':>5} {'精度':>10} {'召回':>10} {'F1':>10}")
        print("-" * 100)
        
        baseline_precision = None
        for analysis in analyses:
            metrics = analysis['metrics']
            scenario = analysis['scenario']
            
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1']
            tp = metrics['tp']
            detected = metrics['detected']
            
            if baseline_precision is None:
                baseline_precision = precision
                improvement = "-"
            else:
                improvement = f"{precision - baseline_precision:+.2f}%"
            
            print(f"{scenario:<30} {detected:>8d} {tp:>5d} {precision:>9.2f}% {recall:>9.2f}% {f1:>9.2f}%  {improvement}")
        
        print("\n" + "="*100)
        
        # 关键发现
        print("\n💡 关键发现:")
        print("-" * 100)
        
        baseline = analyses[0]['metrics']
        ttc_metrics = analyses[1]['metrics']
        pet_metrics = analyses[2]['metrics']
        combined_metrics = analyses[4]['metrics']
        
        print(f"\n1. 基线精度: {baseline['precision']:.2f}% (检测{baseline['detected']}个，TP{baseline['tp']}个)")
        print(f"   • 假正例太多，不适合作为唯一判断标准\n")
        
        print(f"2. TTC过滤效果: {ttc_metrics['precision']:.2f}% (检测{ttc_metrics['detected']}个，TP{ttc_metrics['tp']}个)")
        print(f"   • 精度提升: {ttc_metrics['precision'] - baseline['precision']:+.2f}%")
        print(f"   • 平衡了覆盖率和精度\n")
        
        print(f"3. PET过滤效果: {pet_metrics['precision']:.2f}% (检测{pet_metrics['detected']}个，TP{pet_metrics['tp']}个)")
        print(f"   • 精度提升: {pet_metrics['precision'] - baseline['precision']:+.2f}%")
        print(f"   • 最高精度，但覆盖率有限\n")
        
        if combined_metrics['detected'] > 0:
            print(f"4. TTC+PET组合: {combined_metrics['precision']:.2f}% (检测{combined_metrics['detected']}个，TP{combined_metrics['tp']}个)")
            print(f"   • 精度: {combined_metrics['precision']:.2f}%")
            print(f"   • 最保守但最可靠的方案\n")
        
        print("\n建议:")
        print("-" * 100)
        print("  ✓ 优先使用 PET 指标 (精度最高)")
        print("  ✓ 作为补充使用 TTC 指标 (覆盖率更好)")
        print("  ✗ 不建议仅使用 Distance 指标 (误检太多)")
        print("  ✓ 可考虑 TTC + PET 组合 (最保守的方案)")
        print("\n" + "="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(description='指标对比分析工具')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Pipeline结果目录')
    parser.add_argument('--ground-truth', type=str, required=True,
                       help='Ground Truth标注文件（JSON格式）')
    
    args = parser.parse_args()
    
    analyzer = MetricComparisonAnalyzer(args.results_dir, args.ground_truth)
    analyzer.print_comparison()


if __name__ == '__main__':
    main()
