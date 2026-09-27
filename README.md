施工安全检测-【施工安全数据集】1206张，nc=1 标签names: ['helmet', 'no-helmet', 'no-vest', 'person', 'vest']]名称：【“头盔”“无头盔”“无背心”“人”“背心”】共1206张，（train;997张，val：119张 test：90张。标注文件为YOLO适用的txt格式。可以直接用于模型训练。



| 项目 | 内容 |
|------|------|
| **数据集名称** | 施工安全数据集 |
| **总图像数量** | 1206 张 |
| **类别数量（nc）** | 5 |
| **类别名称（names）** | `['helmet', 'no-helmet', 'no-vest', 'person', 'vest']` |
| **中文类别对应** | - helmet：头盔<br>- no-helmet：无头盔<br>- no-vest：无背心<br>- person：人<br>- vest：背心 |
| **标注格式** | YOLO 格式（每张图像对应一个 `.txt` 文件，含归一化边界框和类别索引） |
| **数据划分** | - 训练集（train）：997 张<br>- 验证集（val）：119 张<br>- 测试集（test）：90 张 |
| **可直接用于** | YOLOv5 / YOLOv8 / YOLOv11 等 Ultralytics 系列模型训练 |

> ✅ 说明：该数据集适用于施工现场人员安全穿戴（安全帽、反光背心）合规性自动检测，支持智能视频监控、无人机巡检等场景。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9192cf31b4484257b9ade62dca1b0203.png)


---

### ✅ 一、数据集信息回顾

- **总图像数**：1206 张  
- **划分**：
  - `train`: 997 张  
  - `val`: 119 张  
  - `test`: 90 张  
- **类别（nc=5）**：
  ```python
  names = ['helmet', 'no-helmet', 'no-vest', 'person', 'vest']
  ```
- **标注格式**：YOLO 格式（`.txt`），已归一化，可直接用于训练。

---

### ✅ 二、推荐目录结构

```
construction_safety_dataset/
├── images/
│   ├── train/      # 997 张 .jpg/.png
│   ├── val/        # 119 张
│   └── test/       # 90 张
└── labels/
    ├── train/      # 对应 .txt 标签
    ├── val/
    └── test/
```

> 💡 确保图像与标签文件名一一对应（如 `001.jpg` ↔ `001.txt`）。

---

### ✅ 三、1. 创建数据集配置文件 `construction.yaml`

在项目根目录下创建 `construction.yaml`：

```yaml
# construction.yaml
path: ./construction_safety_dataset
train: images/train
val: images/val
test: images/test

nc: 5
names: ['helmet', 'no-helmet', 'no-vest', 'person', 'vest']
```

---

### ✅ 四、2. 完整训练与验证代码（Python）

```python
# train_construction_safety.py
from ultralytics import YOLO
import os

# ==================== 配置参数 ====================
DATASET_YAML = 'construction.yaml'     # 数据集配置文件
MODEL_NAME = 'yolov8n.pt'              # 可选: yolov8s.pt, yolov8m.pt
EPOCHS = 150
IMG_SIZE = 640                         # 输入尺寸（支持非正方形）
BATCH_SIZE = 16                        # 根据GPU显存调整（如 8/16/32）
PROJECT_NAME = 'construction_safety_yolov8n'
DEVICE = 0                             # GPU ID，无GPU设为 'cpu'

# ==================== 加载模型 ====================
print("🚀 加载预训练模型...")
model = YOLO(MODEL_NAME)

# ==================== 开始训练 ====================
print(f"📈 开始训练（epochs={EPOCHS}, imgsz={IMG_SIZE}, batch={BATCH_SIZE}）...")
results = model.train(
    data=DATASET_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    name=PROJECT_NAME,
    device=DEVICE,
    patience=30,           # 早停：若30轮内val loss未下降则停止
    save=True,             # 保存 best.pt 和 last.pt
    plots=True,            # 生成训练曲线、混淆矩阵等
    exist_ok=False         # 若实验名已存在则报错（避免覆盖）
)

# ==================== 在测试集上验证 ====================
print("\n🔍 在测试集上评估模型性能...")
metrics = model.val(
    data=DATASET_YAML,
    split='test',          # 使用 test 分区
    conf=0.25,             # 置信度阈值
    iou=0.6,               # NMS IoU 阈值
    verbose=True
)

# 打印关键指标
print("\n✅ 测试集评估结果:")
print(f"  mAP@0.5:       {metrics.box.map50:.4f}")
print(f"  mAP@0.5:0.95:  {metrics.box.map:.4f}")
print(f"  Precision:     {metrics.box.p:.4f}")
print(f"  Recall:        {metrics.box.r:.4f}")

# ==================== 示例推理（可选）====================
# test_img = "construction_safety_dataset/images/test/xxx.jpg"
# results = model.predict(source=test_img, save=True, show=False)
# print(f"✅ 推理结果已保存至: {results[0].save_dir}")
```

---

### ✅ 五、3. 命令行训练方式（等效）

```bash
# 安装依赖（如未安装）
pip install ultralytics

# 启动训练
yolo detect train \
    data=construction.yaml \
    model=yolov8n.pt \
    epochs=150 \
    imgsz=640 \
    batch=16 \
    name=construction_safety_yolov8n \
    device=0
```

---

### ✅ 六、训练结果输出位置

训练完成后，结果保存在：
```
runs/detect/construction_safety_yolov8n/
├── weights/best.pt      # 最佳模型（按 mAP50 选择）
├── weights/last.pt      # 最后一轮模型
├── results.csv          # 训练指标日志
├── labels.jpg           # 标签分布图
├── confusion_matrix.png # 混淆矩阵
└── ...
```

---

### ✅ 七、部署与应用建议

| 场景 | 建议 |
|------|------|
| **工地监控摄像头** | 导出 ONNX 模型 + OpenCV 实时推理 |
| **移动端/边缘设备** | 使用 `yolov8n` + TensorRT / CoreML 加速 |
| **合规性报告生成** | 结合检测结果统计“无头盔/无背心”人数比例 |

导出 ONNX 示例：
```python
model = YOLO('runs/detect/construction_safety_yolov8n/weights/best.pt')
model.export(format='onnx', imgsz=640)
```

---


