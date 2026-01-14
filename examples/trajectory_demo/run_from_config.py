#!/usr/bin/env python3
"""
YOLO-First 碰撞检测管道 - YAML配置版本
支持通过YAML配置文件运行，适配不同的视频和标定参数

使用方式:
    python3 run_from_config.py --config configs/my_video.yaml
    python3 run_from_config.py --config configs/homograph_fullscreen_example.yaml
"""

import argparse
import sys
from pathlib import Path
from collision_detection_pipeline_yolo_first_method_a import YOLOFirstPipelineA


def main():
    parser = argparse.ArgumentParser(
        description="从YAML配置文件运行YOLO-First碰撞检测管道",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用预定义的配置文件
  python3 run_from_config.py --config configs/homograph_fullscreen_example.yaml
  
  # 自定义配置文件
  python3 run_from_config.py --config configs/my_custom_video.yaml
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='YAML配置文件路径'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出模式'
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    
    # 验证配置文件存在
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)
    
    try:
        print(f"📋 加载配置文件: {config_path}")
        
        # 从YAML配置创建pipeline
        pipeline = YOLOFirstPipelineA.from_config(str(config_path))
        
        # 运行pipeline
        print(f"\n▶️  开始执行碰撞检测管道...")
        pipeline.run(conf_threshold=pipeline.config.yolo.confidence_threshold)
        
        print(f"\n✅ 管道执行完成！")
        print(f"📁 结果保存在: {pipeline.run_dir}")
        
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 执行出错: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
