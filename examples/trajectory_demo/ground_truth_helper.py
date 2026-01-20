#!/usr/bin/env python3
"""
Ground Truth 标注辅助工具

用途：帮助手动标注视频中的碰撞事件
提供：视频分析、关键帧查看、易用的标注界面
"""

import json
from pathlib import Path
from typing import List, Dict
import subprocess

class GroundTruthHelper:
    """Ground Truth标注辅助工具"""
    
    def __init__(self, video_path: str, output_dir: str = None):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir) if output_dir else self.video_path.parent
        
    def get_video_info(self):
        """获取视频信息"""
        try:
            import subprocess
            cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames -of csv=p=0 "{self.video_path}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                width, height, fps_str, frames = parts[0], parts[1], parts[2], parts[3]
                
                # 解析fps (可能是 "30/1" 或 "30")
                if '/' in fps_str:
                    fps = eval(fps_str)
                else:
                    fps = float(fps_str)
                
                total_frames = int(frames) if frames else "unknown"
                total_seconds = total_frames / fps if total_frames != "unknown" else "unknown"
                
                print("\n" + "="*60)
                print("📹 视频信息")
                print("="*60)
                print(f"  文件: {self.video_path.name}")
                print(f"  分辨率: {width}x{height}")
                print(f"  帧率: {fps:.2f} fps")
                print(f"  总帧数: {total_frames}")
                print(f"  时长: {total_seconds:.1f}s" if total_seconds != "unknown" else "  时长: unknown")
                print("="*60 + "\n")
                
                return {
                    "width": int(width),
                    "height": int(height),
                    "fps": fps,
                    "total_frames": total_frames,
                    "duration": total_seconds
                }
            else:
                print(f"⚠️ 无法获取视频信息: {result.stderr}")
                return None
        except Exception as e:
            print(f"⚠️ 获取视频信息失败: {e}")
            return None
    
    def create_annotation_template(self, video_info: Dict = None):
        """创建标注模板"""
        
        template = {
            "metadata": {
                "video_file": self.video_path.name,
                "video_path": str(self.video_path),
                "annotation_date": "YYYY-MM-DD",
                "annotator": "Your Name",
                "notes": "请在此添加任何补充说明"
            },
            "video_info": video_info if video_info else {},
            "instructions": {
                "event_types": {
                    "collision": "两物体接触或距离 < 0.5m",
                    "near_miss": "两物体接近通过，距离 0.5-1.5m",
                    "close_approach": "距离 1.5-3m，物体正在相互接近"
                },
                "severity_levels": {
                    "critical": "立即碰撞风险 (距离<0.5m 或 TTC<0.5s)",
                    "high": "高碰撞风险 (距离0.5-1.5m 或 TTC<2s)",
                    "medium": "中等风险 (距离1.5-3m 或 TTC<5s)",
                    "low": "低风险 (距离>3m)"
                }
            },
            "ground_truth_events": [
                {
                    "frame": "视频帧号 (从0开始)",
                    "time_seconds": "时间 (秒)",
                    "object_1": {
                        "class": "vehicle|person|motorcycle|bicycle|truck|bus|car",
                        "description": "物体1的描述"
                    },
                    "object_2": {
                        "class": "vehicle|person|motorcycle|bicycle|truck|bus|car",
                        "description": "物体2的描述"
                    },
                    "event_type": "collision|near_miss|close_approach",
                    "severity": "critical|high|medium|low",
                    "distance_estimate": "目测距离 (米)",
                    "visible_interaction": "简述物体如何相互作用",
                    "notes": "补充说明"
                }
            ],
            "summary": {
                "total_events": 0,
                "collision_count": 0,
                "near_miss_count": 0,
                "close_approach_count": 0,
                "reviewed_minutes": "标注耗时(分钟)"
            }
        }
        
        return template
    
    def save_template(self, template: Dict, output_file: str = None):
        """保存标注模板"""
        if output_file is None:
            output_file = self.output_dir / "ground_truth_template.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 模板已保存: {output_file}")
        return output_file
    
    def print_quick_reference(self):
        """打印快速参考指南"""
        print("\n" + "="*70)
        print("🎯 快速参考指南")
        print("="*70)
        print("""
碰撞分类标准:

1️⃣  COLLISION (碰撞)
   - 两物体接触或极其接近 (< 0.5m)
   - 严重程度: CRITICAL
   - 例: 车辆相撞、严重擦碰

2️⃣  NEAR MISS (险些碰撞)
   - 两物体接近通过 (0.5m - 1.5m)
   - 严重程度: HIGH
   - 例: 车辆在距离小于1.5m时超车、行人侧身通过

3️⃣  CLOSE APPROACH (接近)
   - 两物体相互接近中 (1.5m - 3m)
   - 严重程度: MEDIUM / LOW
   - 例: 车辆逐渐接近另一车辆 (但最终不会碰撞)

提示:
  • 优先标记 COLLISION 和 NEAR MISS 事件
  • 只标记真实的交互 (不是巧合的接近)
  • 使用Pipeline输出的keyframe图片作为参考
  • 看视频而不仅仅是keyframe - 了解完整的物体轨迹
  • 记录准确的帧号 (对应于视频中的位置)

""")
        print("="*70 + "\n")

if __name__ == "__main__":
    # 使用示例
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python ground_truth_helper.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    helper = GroundTruthHelper(video_path)
    
    # 1. 获取视频信息
    video_info = helper.get_video_info()
    
    # 2. 打印快速参考
    helper.print_quick_reference()
    
    # 3. 创建并保存标注模板
    template = helper.create_annotation_template(video_info)
    output_path = helper.save_template(template)
    
    print(f"""
✅ 准备完成！

下一步：
  1. 在你喜欢的编辑器中打开标注模板:
     {output_path}
  
  2. 观看视频并标注所有碰撞/近距离事件
     (使用ground_truth_template.json中的示例格式)
  
  3. 完成后，保存为:
     {helper.output_dir / 'ground_truth_homograph_fullscreen.json'}
  
  4. 运行Pipeline进行精度评估:
     python collision_detection_pipeline_yolo_first_method_a.py \\
       --video <video_path> --homography <homography_path>
""")
