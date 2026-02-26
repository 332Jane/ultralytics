"""
[DEPRECATED] detection_adapter.py
Original Purpose:
  - Convert YOLO result objects to standardized format
  - Handle version compatibility (tensor vs numpy, different bbox formats)
  - Provide unified interface: parse_result() returns List[Dict]
  - Support segmentation masks

Reasons for Removal:
  1. No longer needed - Method A pipeline handles results internally
  2. Legacy code only - old yolo_runner.py scripts used it
  3. Unnecessary layer - adds complexity without benefit
  4. Integrated design - current pipeline doesn't use adapters

Alternative:
  - See collision_detection_pipeline_yolo_first_method_a.py for current implementation

Parse Ultralytics `result` objects into unified detection list format:
[ {"id":..., "cls":..., "x":..., "y":..., "t":..., "conf":..., "bbox": [...]}, ... ]

Field naming is consistent with ObjectStateManager.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Any

# 延迟导入 coord_transform，以避免循环依赖

def parse_result(result, timestamp: float) -> List[Dict[str, Any]]:
    """Parse detection items list from a Ultralytics `result` object.

    Parameters:
    - result: Results object for a single frame (iteration item from model.predict/track)
    - timestamp: Frame timestamp or frame number

    Returns:
    - detections: List where each item is a dict: {id, cls, x, y, t, conf, bbox, mask}
    """
    # Boxes object fields may vary slightly across versions, use getattr as fallback
    boxes = getattr(result, 'boxes', None)
    masks = getattr(result, 'masks', None)  # Segmentation masks (if available)
    
    if boxes is None:
        return []

    xyxy = getattr(boxes, 'xyxy', None)
    cls = getattr(boxes, 'cls', None)
    conf = getattr(boxes, 'conf', None)
    ids = getattr(boxes, 'id', None)
    if ids is None:
        ids = getattr(boxes, 'ids', None)

    # 转为 numpy（若是 tensor）
    if xyxy is None:
        return []

    try:
        xyxy_np = xyxy.cpu().numpy()
    except Exception:
        xyxy_np = np.asarray(xyxy)

    try:
        cls_np = cls.cpu().numpy() if cls is not None else None
    except Exception:
        cls_np = np.asarray(cls) if cls is not None else None

    try:
        conf_np = conf.cpu().numpy() if conf is not None else None
    except Exception:
        conf_np = np.asarray(conf) if conf is not None else None

    try:
        ids_np = ids.cpu().numpy() if ids is not None else None
    except Exception:
        ids_np = np.asarray(ids) if ids is not None else None

    # 尝试获取掩码数据
    masks_data = None
    if masks is not None:
        try:
            masks_data = masks.masks.cpu().numpy()  # shape: (N, H, W)
        except Exception:
            try:
                masks_data = masks.cpu().numpy()
            except Exception:
                masks_data = None

    dets = []
    for i, box in enumerate(xyxy_np):
        x1, y1, x2, y2 = box.tolist()
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)
        
        # 如果有分割掩码，计算掩码中心而不是bbox中心
        if masks_data is not None and i < len(masks_data):
            mask = masks_data[i]
            # 找到掩码中的前景像素
            y_coords, x_coords = np.where(mask > 0)
            if len(x_coords) > 0:
                cx = float(np.mean(x_coords))
                cy = float(np.mean(y_coords))
                # 计算掩码的最小外接矩形
                x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        
        det = {
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'cx': cx,
            'cy': cy,
            't': timestamp,
            'cls': int(cls_np[i]) if cls_np is not None else None,
            'conf': float(conf_np[i]) if conf_np is not None else None,
            'id': int(ids_np[i]) if ids_np is not None else None,
            'has_mask': masks_data is not None and i < len(masks_data)  # 标记是否用了掩码
        }
        dets.append(det)

    return dets
