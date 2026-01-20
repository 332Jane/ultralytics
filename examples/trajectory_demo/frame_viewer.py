"""
逐帧查看工具 - 用于手动标注

用法:
  python frame_viewer.py --video <视频路径> --frame <帧号>
  
例如:
  python frame_viewer.py --video ../../videos/Homograph_Teset_FullScreen.mp4 --frame 67
"""

import cv2
import argparse
from pathlib import Path

def view_frame(video_path, frame_num, output_size=(1920, 1080)):
    """显示指定帧的画面"""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    # 跳转到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ 无法读取Frame {frame_num}")
        return
    
    # 缩放显示
    frame_resized = cv2.resize(frame, output_size)
    
    # 显示
    cv2.imshow(f'Frame {frame_num}', frame_resized)
    print(f"✓ 显示 Frame {frame_num}")
    print(f"  按任何键关闭窗口")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='逐帧查看工具')
    parser.add_argument('--video', type=str, required=True, help='视频路径')
    parser.add_argument('--frame', type=int, required=True, help='要查看的帧号')
    
    args = parser.parse_args()
    view_frame(args.video, args.frame)

if __name__ == '__main__':
    main()
