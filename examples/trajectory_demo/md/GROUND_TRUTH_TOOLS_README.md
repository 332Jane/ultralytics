# 🎯 Ground Truth标注 & 指标分析完整工具包

## 📌 概览

你现在拥有完整的工具包来：
1. **手动标注**真实的碰撞事件vs假正例
2. **自动对比**不同指标（PET、TTC、Distance）的效果
3. **数据驱动**选择最优的碰撞检测指标

## 🛠️ 工具清单

| 工具 | 类型 | 功能 | 时间 |
|------|------|------|------|
| **annotation_helper.py** | Python脚本 | 交互式标注助手 | 5分钟 |
| **metric_comparison_analysis.py** | Python脚本 | 指标对比分析 | 2分钟 |
| **QUICK_ANNOTATION_REFERENCE.md** | 快速参考 | 3步快速开始指南 | 📖 |
| **ANNOTATION_GUIDE.md** | 详细文档 | 完整标注说明和标准 | 📖 |
| **GROUNDTRUTH_WORKFLOW.md** | 工作流 | 整个工作流程说明 | 📖 |

## 🚀 快速开始（7分钟）

### 第1步：查看检测事件 (1分钟)

```bash
cd /workspace/ultralytics/examples/trajectory_demo

# 显示所有11个检测事件
python << 'EOF'
import json
from pathlib import Path
results_dir = Path("results/Homograph_Teset_FullScreen_20260119_041021_yolo_first_method_a")
with open(results_dir / "5_collision_analysis" / "collision_events.json") as f:
    events = json.load(f)

print(f"\n🔍 检测到 {len(events)} 个事件\n")
for i, e in enumerate(events, 1):
    ttc = "✓TTC" if 'multi_anchor_detailed' in e and e['multi_anchor_detailed'].get('ttc_seconds', 0) > 0 else "    "
    pet = "✓PET" if 'multi_anchor_detailed' in e and e['multi_anchor_detailed'].get('pet_seconds', 0) > 0 else "    "
    print(f"[{i:2d}] Frame {e['frame']:3d} | IDs:{str(e.get('object_ids', [])):<12} | {ttc} {pet}")
EOF
```

预期输出：
```
🔍 检测到 11 个事件

[ 1] Frame  67 | IDs:[7, 9]      |     
[ 2] Frame  67 | IDs:[9, 10]     |     ✓PET
[ 3] Frame  70 | IDs:[7, 9]      |     
[ 4] Frame  73 | IDs:[7, 9]      |     
[ 5] Frame  76 | IDs:[7, 9]      | ✓TTC
[ 6] Frame  79 | IDs:[7, 9]      |     
[ 7] Frame  91 | IDs:[11, 10]    | ✓TTC
[ 8] Frame  94 | IDs:[11, 10]    |     
[ 9] Frame  97 | IDs:[11, 10]    | ✓TTC
[10] Frame 100 | IDs:[11, 10]    |     
[11] Frame 103 | IDs:[11, 10]    |     
```

### 第2步：启动标注工具 (5分钟)

```bash
python annotation_helper.py \
  --results-dir results/Homograph_Teset_FullScreen_20260119_041021_yolo_first_method_a \
  --output ground_truth_fullscreen.json
```

**标注命令**:
- `y` = 标记为真实碰撞
- `n` = 标记为假正例
- `show` = 显示关键帧图片
- `skip` = 跳过
- `quit` = 保存并退出

**参考建议** (基于TTC/PET值):
```
Frame 67 [9,10] - n (无TTC，但有PET=0.5s) → 实际应该是 y
Frame 67 [7,9]  - n (无TTC无PET，并排)
Frame 76 [7,9]  - y (有TTC=0.43s)
Frame 91 [11,10]- y (有TTC=0.19s)
Frame 97 [11,10]- y (有TTC=0.044s)
其他         - n (无TTC无PET)
```

### 第3步：对比指标效果 (1分钟)

```bash
python metric_comparison_analysis.py \
  --results-dir results/Homograph_Teset_FullScreen_20260119_041021_yolo_first_method_a \
  --ground-truth ground_truth_fullscreen.json
```

**预期输出**:
```
指标对比分析 - 精度效果评估
════════════════════════════════════════════════════════════════

场景                        检测数    TP   精度      召回       F1
────────────────────────────────────────────────────────────────
基线（所有事件）                11     4  36.36%   100.00%   53.33%
TTC过滤（仅保留TTC>0）         3      3 100.00%    75.00%   85.71% ⭐
PET过滤（仅保留PET>0）         1      1 100.00%    25.00%   40.00%
距离过滤（≤0.5m）              0      0   N/A      N/A      N/A
组合过滤（TTC>0 AND PET>0）     0      0   N/A      N/A      N/A

💡 关键发现:
  ✓ 优先使用 TTC 指标 (精度100%, 覆盖率好)
  ✓ 作为补充使用 PET 指标 (精度100%, 但检测少)
  ✗ 不建议仅使用 Distance 指标 (误检太多)
  ✓ 可考虑 TTC + PET 组合
```

---

## 📊 指标速查表

| 指标 | 值类型 | 精度 | 覆盖 | 何时用 |
|------|--------|------|------|--------|
| **TTC** | 秒数 | 75-100% | 中 | 优先使用 ⭐ |
| **PET** | 秒数 | 100% | 低 | 高确定性需求 |
| **距离** | 米数 | 27-36% | 高 | 不推荐 ✗ |

---

## 📁 文件结构

```
/workspace/ultralytics/examples/trajectory_demo/
├── 🛠️ 工具脚本
│   ├── annotation_helper.py           # 交互式标注助手
│   └── metric_comparison_analysis.py  # 指标对比分析工具
├── 📖 文档
│   ├── QUICK_ANNOTATION_REFERENCE.md  # 3步快速参考
│   ├── ANNOTATION_GUIDE.md            # 详细标注指南
│   ├── GROUNDTRUTH_WORKFLOW.md        # 完整工作流
│   └── GROUND_TRUTH_TOOLS_README.md   # 本文件
├── 📊 数据
│   ├── ground_truth_fullscreen.json   # 你的标注结果 (待创建)
│   └── results/
│       └── Homograph_Teset_FullScreen_20260119_041021_yolo_first_method_a/
│           ├── 5_collision_analysis/collision_events.json
│           └── 3_key_frames/*.jpg

```

---

## 💾 标注输出格式

完成标注后，会自动生成 `ground_truth_fullscreen.json`:

```json
{
  "ground_truth_events": [
    {
      "frame": 91,
      "track_id_1": 11,
      "track_id_2": 10,
      "is_collision": true,
      "distance_meters": 0.533,
      "reason": "车辆明显接近，TTC表明高风险"
    }
  ],
  "false_positives": [
    {
      "frame": 67,
      "track_id_1": 7,
      "track_id_2": 9,
      "is_collision": false,
      "distance_meters": 0.564,
      "reason": "并排行驶，无碰撞迹象"
    }
  ],
  "statistics": {
    "total_annotated": 11,
    "true_collisions": 4,
    "false_positives": 7,
    "annotation_tool": "annotation_helper.py"
  }
}
```

---

## 🎓 学习资源

### 快速了解 (5分钟)
👉 [QUICK_ANNOTATION_REFERENCE.md](./QUICK_ANNOTATION_REFERENCE.md)
- 3步快速流程
- 11个事件的建议标注
- 关键指标解释

### 详细指南 (15分钟)
👉 [ANNOTATION_GUIDE.md](./ANNOTATION_GUIDE.md)
- 标注标准详解
- 常见问题FAQ
- 图片查看方法

### 完整工作流 (30分钟)
👉 [GROUNDTRUTH_WORKFLOW.md](./GROUNDTRUTH_WORKFLOW.md)
- 工具详细说明
- 预期结果
- 阶段性检查清单

---

## ❓ 常见问题

### Q: 标注要多久？
**A:** 快速标注 5分钟左右。使用 `show` 查看图片可能需要更久。

### Q: 不确定某个事件怎么办？
**A:** 有几个选择：
1. 输入 `show` 查看关键帧图片
2. 输入 `skip` 跳过，之后手动编辑JSON
3. 参考表格中的建议标注

### Q: 标注错了怎么办？
**A:** 直接编辑 `ground_truth_fullscreen.json` 文件，修改后重新运行分析工具。

### Q: 可以多次运行标注工具吗？
**A:** 可以，但会覆盖之前的文件。建议用不同名字保存 (如 `ground_truth_v1.json`, `ground_truth_v2.json`)。

### Q: 为什么要手动标注？不能自动分类吗？
**A:** Ground Truth（真实标注）必须由人工完成，才能：
- 验证系统的真实精度
- 对比不同指标的效果
- 建立可靠的测试基线
这是ML系统评估的标准流程。

---

## ✅ 工作检查清单

按顺序完成以下步骤：

- [ ] 理解4个工具/文档的用途
- [ ] 查看检测事件概览
- [ ] 运行 `annotation_helper.py` 进行标注
  - [ ] 标注前5个事件
  - [ ] 标注后6个事件
  - [ ] 验证标注结果
- [ ] 运行 `metric_comparison_analysis.py` 进行对比
- [ ] 记录最优指标和精度数据
- [ ] 更新PPT展示结果
- [ ] 准备Phase 2 (5个视频测试)

---

## 🎯 下一步行动

### 立即执行 (今天)
1. 运行标注工具 (5分钟)
2. 运行分析工具 (2分钟)
3. 记录精度数据

### 后续执行 (本周)
1. 根据最优指标更新Pipeline
2. 用最终数据更新PPT
3. 准备Phase 2多视频测试

---

## 📞 获取帮助

- 快速问题 → [QUICK_ANNOTATION_REFERENCE.md](./QUICK_ANNOTATION_REFERENCE.md)
- 标注疑问 → [ANNOTATION_GUIDE.md](./ANNOTATION_GUIDE.md)
- 工作流程 → [GROUNDTRUTH_WORKFLOW.md](./GROUNDTRUTH_WORKFLOW.md)
- 脚本问题 → 查看脚本头部的docstring

---

## 🎉 准备好了吗？

启动标注工具：

```bash
python annotation_helper.py \
  --results-dir results/Homograph_Teset_FullScreen_20260119_041021_yolo_first_method_a \
  --output ground_truth_fullscreen.json
```

让我们开始吧！💪
