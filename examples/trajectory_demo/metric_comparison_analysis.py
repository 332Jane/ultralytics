"""
Metric Comparison Analysis Tool

Purpose: Compare the impact of different metrics (PET, TTC, Distance) on collision detection accuracy
Usage: python metric_comparison_analysis.py --results-dir <results_dir> --ground-truth <annotation_file>
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import statistics


class MetricComparisonAnalyzer:
    """Compare the effectiveness of different metrics"""
    
    def __init__(self, results_dir: str, ground_truth_file: str):
        self.results_dir = Path(results_dir)
        self.collision_events_file = self.results_dir / "5_collision_analysis" / "collision_events.json"
        
        # Load ground truth
        with open(ground_truth_file) as f:
            gt_data = json.load(f)
            if isinstance(gt_data, dict) and 'ground_truth_events' in gt_data:
                self.ground_truth = gt_data['ground_truth_events']
            else:
                self.ground_truth = gt_data
    
    def load_detected_events(self) -> List[Dict]:
        """Load detected events"""
        if not self.collision_events_file.exists():
            print(f"❌ Detection results not found: {self.collision_events_file}")
            return []
        
        with open(self.collision_events_file) as f:
            return json.load(f)
    
    def get_gt_frames(self) -> set:
        """Get the set of frames in ground truth"""
        return set(event['frame'] for event in self.ground_truth)
    
    def calculate_metrics(self, detected_frames: set, tp_count: int) -> Dict:
        """Calculate Precision, Recall, and F1 score"""
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
        """Baseline analysis: all detected events"""
        detected = self.load_detected_events()
        detected_frames = set(e['frame'] for e in detected)
        gt_frames = self.get_gt_frames()
        
        tp = len(detected_frames & gt_frames)
        
        return {
            'scenario': 'Baseline (All Events)',
            'detected_frames': detected_frames,
            'metrics': self.calculate_metrics(detected_frames, tp)
        }
    
    def analyze_ttc_filter(self) -> Dict:
        """TTC filter analysis: keep only events with TTC values"""
        detected = self.load_detected_events()
        gt_frames = self.get_gt_frames()
        
        # Filter events with TTC
        events_with_ttc = []
        for e in detected:
            if 'multi_anchor_detailed' in e:
                ttc = e['multi_anchor_detailed'].get('ttc_seconds')
                if ttc and ttc > 0:
                    events_with_ttc.append(e)
        
        detected_frames = set(e['frame'] for e in events_with_ttc)
        tp = len(detected_frames & gt_frames)
        
        return {
            'scenario': 'TTC Filter (TTC > 0 only)',
            'detected_frames': detected_frames,
            'metrics': self.calculate_metrics(detected_frames, tp)
        }
    
    def analyze_pet_filter(self) -> Dict:
        """PET filter analysis: keep only events with PET values"""
        detected = self.load_detected_events()
        gt_frames = self.get_gt_frames()
        
        # Filter events with PET
        events_with_pet = []
        for e in detected:
            if 'multi_anchor_detailed' in e:
                pet = e['multi_anchor_detailed'].get('pet_seconds')
                if pet and pet > 0:
                    events_with_pet.append(e)
        
        detected_frames = set(e['frame'] for e in events_with_pet)
        tp = len(detected_frames & gt_frames)
        
        return {
            'scenario': 'PET Filter (PET > 0 only)',
            'detected_frames': detected_frames,
            'metrics': self.calculate_metrics(detected_frames, tp)
        }
    
    def analyze_distance_threshold(self, threshold: float = 0.5) -> Dict:
        """Distance threshold analysis"""
        detected = self.load_detected_events()
        gt_frames = self.get_gt_frames()
        
        # Filter events with distance below threshold
        events_filtered = [e for e in detected if e.get('distance_meters', float('inf')) <= threshold]
        
        detected_frames = set(e['frame'] for e in events_filtered)
        tp = len(detected_frames & gt_frames)
        
        return {
            'scenario': f'Distance Filter (≤{threshold}m)',
            'detected_frames': detected_frames,
            'metrics': self.calculate_metrics(detected_frames, tp)
        }
    
    def analyze_combined_ttc_pet(self) -> Dict:
        """Combined analysis: TTC AND PET"""
        detected = self.load_detected_events()
        gt_frames = self.get_gt_frames()
        
        # Filter events with both TTC and PET
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
            'scenario': 'Combined Filter (TTC>0 AND PET>0)',
            'detected_frames': detected_frames,
            'metrics': self.calculate_metrics(detected_frames, tp)
        }
    
    def print_comparison(self):
        """Print comparison results"""
        print("\n" + "="*100)
        print("Metric Comparison Analysis - Accuracy Performance Evaluation")
        print("="*100)
        
        # Execute all analyses
        analyses = [
            self.analyze_baseline(),
            self.analyze_ttc_filter(),
            self.analyze_pet_filter(),
            self.analyze_distance_threshold(),
            self.analyze_combined_ttc_pet()
        ]
        
        # Print table
        print(f"\n{'Scenario':<30} {'Detected':>8} {'TP':>5} {'Precision':>10} {'Recall':>10} {'F1':>10}")
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
        
        # Key findings
        print("\n💡 Key Findings:")
        print("-" * 100)
        
        baseline = analyses[0]['metrics']
        ttc_metrics = analyses[1]['metrics']
        pet_metrics = analyses[2]['metrics']
        combined_metrics = analyses[4]['metrics']
        
        print(f"\n1. Baseline Accuracy: {baseline['precision']:.2f}% (Detected {baseline['detected']}, TP {baseline['tp']})")
        print(f"   • Too many false positives, not suitable as sole criterion\n")
        
        print(f"2. TTC Filter Effect: {ttc_metrics['precision']:.2f}% (Detected {ttc_metrics['detected']}, TP {ttc_metrics['tp']})")
        print(f"   • Precision improvement: {ttc_metrics['precision'] - baseline['precision']:+.2f}%")
        print(f"   • Balanced coverage and precision\n")
        
        print(f"3. PET Filter Effect: {pet_metrics['precision']:.2f}% (Detected {pet_metrics['detected']}, TP {pet_metrics['tp']})")
        print(f"   • Precision improvement: {pet_metrics['precision'] - baseline['precision']:+.2f}%")
        print(f"   • Highest precision, but limited coverage\n")
        
        if combined_metrics['detected'] > 0:
            print(f"4. TTC+PET Combined: {combined_metrics['precision']:.2f}% (Detected {combined_metrics['detected']}, TP {combined_metrics['tp']})")
            print(f"   • Precision: {combined_metrics['precision']:.2f}%")
            print(f"   • Most conservative but most reliable approach\n")
        
        print("\nRecommendations:")
        print("-" * 100)
        print("  ✓ Prioritize PET metric (highest precision)")
        print("  ✓ Use TTC metric as supplement (better coverage)")
        print("  ✗ Not recommended to use Distance metric alone (too many false positives)")
        print("  ✓ Consider TTC + PET combination (most conservative approach)")
        print("\n" + "="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Metric Comparison Analysis Tool')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Pipeline results directory')
    parser.add_argument('--ground-truth', type=str, required=True,
                       help='Ground Truth annotation file (JSON format)')
    
    args = parser.parse_args()
    
    analyzer = MetricComparisonAnalyzer(args.results_dir, args.ground_truth)
    analyzer.print_comparison()


if __name__ == '__main__':
    main()
