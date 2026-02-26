# Ground Truth 标注指南

## 目标
对测试视频进行手动标注，标记**真实发生的碰撞/近距离通过事件**，用于评估Pipeline准确性

## 标注方法

### 第一步：查看视频
```bash
# 使用播放器打开视频
# 视频位置: /workspace/ultralytics/videos/Homograph_Teset_FullScreen.mp4
# 分辨率: 需要检查
# FPS: 需要检查
```

### 第二步：标记碰撞事件
在观看视频时，记录以下信息：

**对于每个碰撞/近距离通过事件，标记：**

```json
{
  "frame": [帧号],
  "time_seconds": [时间],
  "object_1": {
    "class": ["vehicle", "person", "motorcycle", ...],
    "description": "描述物体1"
  },
  "object_2": {
    "class": ["vehicle", "person", "motorcycle", ...],
    "description": "描述物体2"
  },
  "event_type": ["collision", "near_miss", "close_approach"],
  "severity": ["critical", "high", "medium"],
  "notes": "附加说明"
}
```

### 第三步：碰撞定义

- **Collision (碰撞)**: 两个物体接触或距离 < 0.5m
- **Near Miss (险些碰撞)**: 两物体接近通过，距离 0.5-1.5m，有明确相对运动
- **Close Approach (接近)**: 距离 1.5-3m，物体正在相互接近

## 重要提示

✅ **看视频，不要只看Pipeline输出的keyframe**
- Keyframe是关键帧，但可能遗漏了一些事件
- 通过视频可以看到完整的物体轨迹

✅ **优先标记以下类型**
1. **车辆之间的碰撞** (最重要)
2. **行人与车辆的碰撞** (高优先级)
3. **其他接近事件** (中等优先级)

⚠️ **不要标记**
- 完全静止的物体
- 明显是同一物体的多次检测
- 距离 > 3m 的事件

## 标注工具（可选）

如需逐帧检查，使用：
```bash
# 查看Pipeline生成的keyframe图片
ls /workspace/ultralytics/results/*/3_key_frames/keyframe_*.jpg

# 或使用ffmpeg查看特定帧
ffmpeg -i video.mp4 -vf "select=eq(n\,帧号)" -vsync 0 frame.png
```

## 标注结果格式

完成后，保存为 JSON 文件：
```json
{
  "video": "Homograph_Teset_FullScreen.mp4",
  "total_frames": [总帧数],
  "fps": [帧率],
  "ground_truth_events": [
    {
      "frame": 67,
      "time_seconds": 2.23,
      "object_1": {"class": "vehicle", "description": "红色轿车"},
      "object_2": {"class": "vehicle", "description": "白色货车"},
      "event_type": "near_miss",
      "severity": "high"
    },
    ...
  ]
}
```

## 检查清单

- [ ] 观看了完整视频
- [ ] 标记了所有明显的碰撞/近距离事件
- [ ] 记录了准确的帧号和时间
- [ ] 保存为 JSON 格式的 ground_truth_homograph_fullscreen.json
- [ ] 验证了 JSON 格式正确

---

**保存位置**: `/workspace/ultralytics/examples/trajectory_demo/ground_truth_homograph_fullscreen.json`

完成后，运行Pipeline会自动进行精度评估。
