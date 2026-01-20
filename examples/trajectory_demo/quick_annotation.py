"""
快速标注脚本 - 基于TTC/PET值自动标注
根据逻辑：有TTC或PET值 = 真实碰撞，无则为假正例
"""

import json
from pathlib import Path

# 查找最新的结果目录
import subprocess
result = subprocess.run(
    "find /workspace/ultralytics/results -name 'collision_events.json' -type f | sort -r | head -1",
    shell=True, capture_output=True, text=True
)
collision_file = result.stdout.strip()
results_dir = Path(collision_file).parent.parent

print(f"使用结果目录: {results_dir}")

# 加载检测事件
with open(collision_file) as f:
    events = json.load(f)

# 按照TTC/PET逻辑进行标注
ground_truth_events = []
false_positives = []

print(f"\n开始标注 {len(events)} 个事件...\n")

for idx, event in enumerate(events, 1):
    frame = event['frame']
    track_ids = event.get('object_ids', [event.get('track_id_1'), event.get('track_id_2')])
    distance = event.get('distance_meters', 0)
    
    # 检查TTC和PET
    has_ttc = False
    has_pet = False
    ttc_val = None
    pet_val = None
    
    if 'multi_anchor_detailed' in event:
        multi = event['multi_anchor_detailed']
        ttc_val = multi.get('ttc_seconds')
        pet_val = multi.get('pet_seconds')
        has_ttc = ttc_val and ttc_val > 0
        has_pet = pet_val and pet_val > 0
    
    # 标注逻辑：有TTC或PET = 真实碰撞
    is_collision = has_ttc or has_pet
    
    ttc_str = f"{ttc_val:.3f}s" if has_ttc else "无"
    pet_str = f"{pet_val:.3f}s" if has_pet else "无"
    
    status = "✓ 真实" if is_collision else "✗ 假正"
    print(f"[{idx:2d}] Frame {frame:3d} | IDs:{str(track_ids):<12} | TTC:{ttc_str:<10} PET:{pet_str:<10} | {status}")
    
    annotation = {
        "frame": frame,
        "track_id_1": track_ids[0],
        "track_id_2": track_ids[1],
        "is_collision": is_collision,
        "distance_meters": distance,
        "reason": f"{'TTC或PET存在' if is_collision else '无TTC/PET指标'}"
    }
    
    if is_collision:
        ground_truth_events.append(annotation)
    else:
        false_positives.append(annotation)

# 保存标注结果
output_file = Path("/workspace/ultralytics/examples/trajectory_demo/ground_truth_fullscreen.json")
output_data = {
    "ground_truth_events": ground_truth_events,
    "false_positives": false_positives,
    "statistics": {
        "total_annotated": len(events),
        "true_collisions": len(ground_truth_events),
        "false_positives": len(false_positives),
        "annotation_tool": "quick_annotation.py (自动)"
    }
}

with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\n{'='*80}")
print(f"✅ 标注完成！")
print(f"{'='*80}")
print(f"  • 真实碰撞: {len(ground_truth_events)} 个")
print(f"  • 假正例: {len(false_positives)} 个")
print(f"  • 总计: {len(events)} 个")
print(f"  • 保存位置: {output_file}\n")
