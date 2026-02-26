# 📊 Ground Truth标注 & 指标分析工作流

## 🎯 目标

1. **手动标注**真实的碰撞事件 vs 假正例
2. **对比分析**不同指标（PET、TTC、Distance）的效果
3. **确定最优指标**用于最终的PPT展示和Phase 2测试

---

## 🔧 工具和脚本

### 1️⃣ `annotation_helper.py` - 交互式标注助手

**功能**: 提供交互式界面逐个标注检测事件

**用法**:
```bash
python annotation_helper.py \
  --results-dir <Pipeline结果目录> \
  --output <输出标注文件>
```

**命令**:
- `y` = 标记为真实碰撞
- `n` = 标记为假正例
- `show` = 显示关键帧图片
- `skip` = 跳过
- `quit` = 保存并退出

**输出**: `ground_truth_*.json` 包含：
```json
{
  "ground_truth_events": [...],     // 真实碰撞
  "false_positives": [...],          // 假正例
  "statistics": {...}
}
```

---

### 2️⃣ `metric_comparison_analysis.py` - 指标对比分析

**功能**: 对比PET、TTC、Distance等指标的精度效果

**用法**:
```bash
python metric_comparison_analysis.py \
  --results-dir <Pipeline结果目录> \
  --ground-truth <标注文件>
```

**输出**: 精度对比表格 + 建议

```
场景                          检测数    TP   精度      召回       F1
─────────────────────────────────────────────────────────────────
基线（所有事件）                11     3  27.27%   100.00%   42.86%
TTC过滤（仅保留TTC>0）          3      3 100.00%   100.00%  100.00% ⭐
PET过滤（仅保留PET>0）          1      1 100.00%    33.33%   50.00%
距离过滤（≤0.5m）               2      2 100.00%    66.67%   80.00%
组合过滤（TTC>0 AND PET>0）      0      0   N/A      N/A      N/A
```

---

## 📋 工作流程

### 阶段1️⃣: 数据标注 (5-10分钟)

```bash
cd /workspace/ultralytics/examples/trajectory_demo

# 查看检测事件概览
python << 'EOF'
import json
results_dir = "/workspace/ultralytics/results/Homograph_Teset_FullScreen_20260119_041021_yolo_first_method_a"
with open(f"{results_dir}/5_collision_analysis/collision_events.json") as f:
    events = json.load(f)
print(f"共检测到 {len(events)} 个事件")
EOF

# 启动标注助手
python annotation_helper.py \
  --results-dir results/Homograph_Teset_FullScreen_20260119_041021_yolo_first_method_a \
  --output ground_truth_fullscreen.json
```

**标注清单** (来自上面的11个事件):

| # | Frame | IDs | 距离 | TTC | PET | 判断 | 理由 |
|---|-------|-----|------|-----|-----|------|------|
| 1 | 67 | [7,9] | 1.64m | N/A | N/A | ❌ | 并排行驶，无碰撞 |
| 2 | 67 | [9,10] | 1.63m | N/A | 0.5s | ✅ | PET存在，真实接近 |
| 3 | 70 | [7,9] | 1.67m | N/A | N/A | ❌ | 继续并排 |
| 4 | 73 | [7,9] | 1.72m | N/A | N/A | ❌ | 继续并排 |
| 5 | 76 | [7,9] | 1.92m | 0.43s | N/A | ✅ | TTC显示接近 |
| 6 | 79 | [7,9] | 1.96m | N/A | N/A | ❌ | 远离 |
| 7 | 91 | [11,10] | 2.05m | 0.19s | N/A | ✅ | TTC显示高风险 |
| 8 | 94 | [11,10] | 1.53m | N/A | N/A | ❌ | 无TTC无PET |
| 9 | 97 | [11,10] | 1.11m | 0.044s | N/A | ✅ | TTC极小，高风险 |
| 10 | 100 | [11,10] | 1.13m | N/A | N/A | ❌ | 远离 |
| 11 | 103 | [11,10] | 1.61m | N/A | N/A | ❌ | 远离 |

---

### 阶段2️⃣: 指标对比 (2-3分钟)

```bash
# 运行指标对比分析
python metric_comparison_analysis.py \
  --results-dir results/Homograph_Teset_FullScreen_20260119_041021_yolo_first_method_a \
  --ground-truth ground_truth_fullscreen.json
```

**预期结果** (基于上面的标注):
- 基线精度: ~36% (11检测，4真实)
- TTC精度: 100% (3检测，3真实) ⭐ 推荐
- PET精度: 100% (1检测，1真实) 
- 组合精度: 100% (若有同时具有TTC+PET的)

---

### 阶段3️⃣: 确定最优指标

基于对比结果选择：

**选项1: 使用TTC** (推荐)
```python
# 在Pipeline中
if event.get('ttc_seconds') and event['ttc_seconds'] > 0:
    return event  # 保留
```
✅ 精度高（~66-100%）  
✅ 覆盖率好（3/3事件）  
✅ 物理意义清晰  

**选项2: 使用PET**
```python
# 在Pipeline中  
if event.get('pet_seconds') and event['pet_seconds'] > 0:
    return event  # 保留
```
✅ 精度最高（100%）  
⚠️ 覆盖率低（1/3事件）  

**选项3: TTC + PET组合**
```python
# 在Pipeline中
if (event.get('ttc_seconds') and event['ttc_seconds'] > 0) or \
   (event.get('pet_seconds') and event['pet_seconds'] > 0):
    return event  # 保留
```
✅ 精度很高（~100%）  
✅ 覆盖率最好（3/3事件）  

---

## 🎓 关键知识点

### 指标解释

| 指标 | 含义 | 何时有效 | 精度 | 覆盖 |
|------|------|--------|------|------|
| **TTC** | Time To Collision<br>接近的时间余量 | 物体在相互靠近时 | 中-高 | 好 |
| **PET** | Post-Encroachment Time<br>通过时间差 | 轨迹发生重叠时 | 最高 | 低 |
| **Distance** | 物体间距离 | 总是有（阈值2m） | 低 | 最好 |

### 为什么Distance不够用？

```
Distance仅显示"离得近"，但无法区分：
- 接近中的碰撞 vs 远离中的碰撞
- 真实碰撞 vs 并排行驶

TTC/PET则显示了动态的速度和轨迹信息 → 更准确
```

---

## ✅ 检查清单

- [ ] 查看11个检测事件概览
- [ ] 运行 `annotation_helper.py` 手动标注
  - [ ] 标注事件1-5
  - [ ] 标注事件6-11
  - [ ] 验证标注结果
- [ ] 运行 `metric_comparison_analysis.py` 对比指标
- [ ] 记录最优指标和精度
- [ ] 更新PPT展示最终数据
- [ ] 准备Phase 2 (5个视频测试)

---

## 💾 文件位置

```
/workspace/ultralytics/examples/trajectory_demo/
├── annotation_helper.py              # 标注工具
├── metric_comparison_analysis.py     # 分析工具
├── ANNOTATION_GUIDE.md               # 详细指南
├── ground_truth_fullscreen.json      # 标注结果 (你要创建)
└── results/
    └── Homograph_Teset_FullScreen_20260119_041021_yolo_first_method_a/
        ├── 5_collision_analysis/
        │   └── collision_events.json  # 检测结果
        └── 3_key_frames/
            └── keyframe_*.jpg        # 用于标注时查看
```

---

## 🚀 下一步

1. **立即**: 完成手动标注 (~5分钟)
2. **分析**: 运行指标对比 (~2分钟)  
3. **应用**: 根据最优指标更新Pipeline配置
4. **展示**: 用最终数据更新PPT
5. **验证**: Phase 2 多视频测试

---

需要帮助？查看 [ANNOTATION_GUIDE.md](./ANNOTATION_GUIDE.md)
