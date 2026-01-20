# 标定文件 (Calibration Files)

此文件夹包含所有视频的 Homography 标定矩阵文件。

## 文件说明

| 文件名 | 说明 | 用途 |
|--------|------|------|
| `calibration_homography.json` | 默认标定文件 | 示例/备用 |
| `Homograph_Teset_FullScreen_homography.json` | 全屏测试视频标定 | 全屏视频的透视变换 |
| `homographTest_5s_homography.json` | 5秒测试视频标定 | 短视频测试的透视变换 |
| `NewYorkSample_homography.json` | 纽约样本视频标定 | 纽约场景的透视变换 |

## 文件结构

每个 JSON 文件包含以下内容：

```json
{
  "video_name": "视频名称",
  "homography_matrix": [
    [h11, h12, h13],
    [h21, h22, h23],
    [h31, h32, h33]
  ],
  "pixel_points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
  "world_points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
}
```

### 参数解释

- **homography_matrix**: 3x3 透视变换矩阵，用于将像素坐标转换为世界坐标
- **pixel_points**: 图像中的4个参考点像素坐标 (左上、右上、左下、右下)
- **world_points**: 对应的4个参考点的真实世界坐标 (米为单位)

## 使用方法

### 在 YAML 配置中使用指定的标定文件

```yaml
homography:
  enabled: true
  matrix_file: "calibration/Homograph_Teset_FullScreen_homography.json"
```

### 生成新的标定文件

```bash
python calibration.py \
  --pixel-points "x1,y1 x2,y2 x3,y3 x4,y4" \
  --world-points "wx1,wy1 wx2,wy2 wx3,wy3 wx4,wy4" \
  --output calibration/ \
  --video-name "your_video_name"
```

## 注意事项

- 每个视频应有对应的独立标定文件
- 不要混用不同视频的标定矩阵
- 标定点顺序：左上 → 右上 → 左下 → 右下
- 世界坐标单位应保持一致（通常为米）
