# 手动标注表 - 逐帧判断风险等级

## 11个检测事件列表

根据下表，查看每一帧，然后在最右列填写你的判断：

| # | Frame | 时间 | IDs | 系统判断 | 你的判断 | 理由 |
|----|-------|------|-----|---------|---------|------|
| 1 | 67 | 2.23s | [7,9] | Level 3 | ? | |
| 2 | 67 | 2.23s | [9,10] | Level 3 | ? | PET=0.5s |
| 3 | 70 | 2.33s | [7,9] | Level 3 | ? | |
| 4 | 73 | 2.43s | [7,9] | Level 3 | ? | |
| 5 | 76 | 2.53s | [7,9] | Level 3 | ? | TTC=0.43s |
| 6 | 79 | 2.63s | [7,9] | Level 3 | ? | |
| 7 | 91 | 3.03s | [11,10] | Level 3 | ? | TTC=0.19s |
| 8 | 94 | 3.13s | [11,10] | Level 3 | ? | |
| 9 | 97 | 3.23s | [11,10] | Level 2 | ? | TTC=0.044s |
| 10 | 100 | 3.33s | [11,10] | Level 2 | ? | |
| 11 | 103 | 3.43s | [11,10] | Level 3 | ? | |

## 查看每一帧的命令

```bash
cd /workspace/ultralytics/examples/trajectory_demo

# Frame 67
python frame_viewer.py --video ../../videos/Homograph_Teset_FullScreen.mp4 --frame 67

# Frame 70
python frame_viewer.py --video ../../videos/Homograph_Teset_FullScreen.mp4 --frame 70

# Frame 73
python frame_viewer.py --video ../../videos/Homograph_Teset_FullScreen.mp4 --frame 73

# ...以此类推
```

## 标注规则

填写你的判断为以下之一：

- **Level 1 (Collision)** - 两物体明显要碰撞，距离<0.5m，有即时碰撞风险
- **Level 2 (Near Miss)** - 两物体非常接近，距离0.5-1.5m，有较高接近风险  
- **Level 3 (Avoidance)** - 两物体接近但有避免空间，距离>1.5m，风险较低
- **不是碰撞** - 只是并排行驶或过路，无碰撞风险

## 完成后

把填好的表格给我，我重新对比数据。
