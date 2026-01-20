"""
Ground Truth标注助手工具

目的: 帮助用户快速标记真实碰撞事件
用法: python annotation_helper.py --results-dir <结果目录> --output <输出json文件>
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import subprocess
import sys


class AnnotationHelper:
    """标注助手工具 - 帮助用户标记真实碰撞事件"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.collision_events_file = self.results_dir / "5_collision_analysis" / "collision_events.json"
        self.keyframes_dir = self.results_dir / "3_key_frames"
        
    def load_detected_events(self) -> List[Dict]:
        """加载系统检测到的所有事件"""
        if not self.collision_events_file.exists():
            print(f"❌ 未找到检测结果: {self.collision_events_file}")
            return []
        
        with open(self.collision_events_file) as f:
            return json.load(f)
    
    def print_events_summary(self, events: List[Dict]):
        """打印事件摘要 - 便于用户快速了解"""
        print("\n" + "="*80)
        print("检测到的事件摘要（用于标注）")
        print("="*80)
        
        if not events:
            print("未检测到任何事件")
            return
        
        print(f"\n总共检测到 {len(events)} 个事件\n")
        
        for idx, event in enumerate(events, 1):
            frame = event.get('frame', '?')
            track_ids = event.get('object_ids', [event.get('track_id_1', '?'), event.get('track_id_2', '?')])
            
            # 提取关键信息
            distance = event.get('distance_meters', '?')
            level = event.get('level', '?')
            time_s = event.get('time', '?')
            
            # 检查TTC和PET
            ttc = "N/A"
            pet = "N/A"
            if 'multi_anchor_detailed' in event:
                multi = event['multi_anchor_detailed']
                ttc_val = multi.get('ttc_seconds')
                pet_val = multi.get('pet_seconds')
                
                if ttc_val and ttc_val > 0:
                    ttc = f"{ttc_val:.3f}s"
                if pet_val and pet_val > 0:
                    pet = f"{pet_val:.3f}s"
            
            print(f"[{idx:2d}] Frame {frame:4d} ({time_s:.2f}s) | IDs: {track_ids} | Dist: {distance:.2f}m | TTC: {ttc:>10s} | PET: {pet:>10s} | Level {level}")
        
        print("\n" + "="*80)
    
    def start_interactive_annotation(self, output_file: str):
        """交互式标注模式"""
        events = self.load_detected_events()
        self.print_events_summary(events)
        
        if not events:
            print("无事件可标注")
            return
        
        ground_truth = []
        
        print("\n" + "="*80)
        print("开始标注")
        print("="*80)
        print("\n说明:")
        print("  输入 'y' 或 'yes'  = 标记为真实碰撞事件")
        print("  输入 'n' 或 'no'   = 标记为假正例（误检）")
        print("  输入 's' 或 'skip' = 跳过此事件")
        print("  输入 'q' 或 'quit' = 退出标注（保存已标注部分）")
        print("  输入 'show'        = 显示该事件的关键帧图片")
        print("="*80 + "\n")
        
        for idx, event in enumerate(events, 1):
            frame = event.get('frame')
            track_ids = event.get('object_ids', [event.get('track_id_1'), event.get('track_id_2')])
            distance = event.get('distance_meters', 0)
            
            # 构建描述
            desc = f"[{idx}/{len(events)}] Frame {frame} (IDs: {track_ids}, 距离: {distance:.2f}m)"
            
            while True:
                response = input(f"\n{desc} 是真实碰撞吗？(y/n/show/skip/quit): ").lower().strip()
                
                if response in ['q', 'quit']:
                    print("\n标注中断，保存已标注的事件...")
                    break
                
                elif response == 'show':
                    self._show_keyframe(frame, track_ids[0], track_ids[1])
                    continue
                
                elif response in ['y', 'yes']:
                    ground_truth.append({
                        'frame': frame,
                        'track_id_1': track_ids[0],
                        'track_id_2': track_ids[1],
                        'is_collision': True,
                        'distance_meters': distance,
                        'reason': input("  请输入标记为碰撞的理由（可选）: ").strip() or "手动标记为真实碰撞"
                    })
                    print(f"  ✓ 已标记为 TRUE COLLISION")
                    break
                
                elif response in ['n', 'no']:
                    ground_truth.append({
                        'frame': frame,
                        'track_id_1': track_ids[0],
                        'track_id_2': track_ids[1],
                        'is_collision': False,
                        'distance_meters': distance,
                        'reason': input("  请输入标记为误检的理由（可选）: ").strip() or "手动标记为假正例"
                    })
                    print(f"  ✓ 已标记为 FALSE POSITIVE")
                    break
                
                elif response in ['s', 'skip']:
                    print(f"  ⊘ 已跳过")
                    break
                
                else:
                    print("  ❌ 无效输入，请重试")
                    continue
            
            if response in ['q', 'quit']:
                break
        
        # 保存标注结果
        self._save_annotations(ground_truth, output_file)
    
    def _show_keyframe(self, frame: int, track_id_1: int, track_id_2: int):
        """显示对应的关键帧图片"""
        # 尝试找到keyframe文件
        candidates = [
            self.keyframes_dir / f"keyframe_{frame:04d}_ID{track_id_1}_ID{track_id_2}.jpg",
            self.keyframes_dir / f"keyframe_{frame:04d}_ID{track_id_2}_ID{track_id_1}.jpg"
        ]
        
        for candidate in candidates:
            if candidate.exists():
                print(f"\n  正在显示: {candidate.name}")
                try:
                    subprocess.run(['feh', str(candidate)], timeout=5)
                except:
                    try:
                        subprocess.run(['display', str(candidate)], timeout=5)
                    except:
                        print(f"  ⚠️  无法显示图片，请手动查看: {candidate}")
                return
        
        print(f"  ❌ 未找到对应的keyframe图片")
    
    def _save_annotations(self, annotations: List[Dict], output_file: str):
        """保存标注结果到JSON文件"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 统计
        true_count = sum(1 for a in annotations if a['is_collision'])
        false_count = sum(1 for a in annotations if not a['is_collision'])
        
        # 创建输出
        output_data = {
            'ground_truth_events': [a for a in annotations if a['is_collision']],
            'false_positives': [a for a in annotations if not a['is_collision']],
            'statistics': {
                'total_annotated': len(annotations),
                'true_collisions': true_count,
                'false_positives': false_count,
                'annotation_tool': 'annotation_helper.py'
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 标注完成！")
        print(f"  • 真实碰撞: {true_count} 个")
        print(f"  • 假正例: {false_count} 个")
        print(f"  • 总计: {len(annotations)} 个")
        print(f"  • 保存位置: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description='Ground Truth标注助手工具')
    parser.add_argument('--results-dir', type=str, required=True, 
                       help='Pipeline结果目录')
    parser.add_argument('--output', type=str, default='ground_truth_annotations.json',
                       help='输出标注文件')
    
    args = parser.parse_args()
    
    helper = AnnotationHelper(args.results_dir)
    helper.start_interactive_annotation(args.output)


if __name__ == '__main__':
    main()
