# 内容过滤推理系统

一个生产级的、多模态异常检测和内容审核系统。集成了图像特征提取、文本识别、异常检测和分类等多个模块，展现了复杂推理 pipeline 的工程设计。

## 项目概览

本项目是一个完整的**生产推理系统**，用于对图像内容进行智能分析和分类。系统整合了多个深度学习模块，通过模块化设计实现了高效的推理 pipeline。

**核心特性**：
- **模块化架构**：检测 → 识别 → 分类，清晰的数据流
- **性能优化**：并行处理、batch 推理、缓存机制
- **可靠性**：完善的错误处理、日志记录、重试机制
- **可扩展性**：易于集成新模型或新的检测类别

## 系统架构

### 总体流程

```
输入图像
  ↓
[图像预处理]
  • 格式标准化
  • 尺寸调整
  • 归一化
  ↓
[异常区域检测]
  ← ResNet18 (轻量级特征提取)
  ← 返回可疑区域及置信度
  ↓
[文本识别]
  ← PaddleOCR (快速 OCR 模型)
  ← 逐区域文本提取
  ↓
[多模态特征融合]
  ← 视觉特征 (ResNet18 输出)
  ← 文本特征 (BERT 编码)
  ← 注意力机制融合
  ↓
[分类决策]
  ← 多层 FC 分类器
  ← 输出类别和置信度
  ↓
输出结果
  • 检测到的异常类别
  • 位置和置信度
  • 关联的文本信息
```

### 模块设计

#### 1. 检测器（Detector）

**职责**：定位图像中的可疑区域

```python
class MedicalImageDetector:
    def __init__(self):
        # 轻量级骨干网络：ResNet18
        self.backbone = ResNet18(pretrained=True)
    
    def detect(self, image):
        # 返回检测到的区域
        return {
            "regions": [...],
            "confidences": [...],
            "bounding_boxes": [...]
        }
```

**输入**：图像（任意分辨率）
**输出**：检测到的区域列表及其置信度

#### 2. 文本识别器（OCR）

**职责**：从检测到的区域中提取文本

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True)

def extract_text(image):
    result = ocr.ocr(image, cls=True)
    # 返回识别的文本和位置
    return format_ocr_result(result)
```

**输入**：图像或检测到的区域
**输出**：文本字符串和定位信息

#### 3. 分类器（Classifier）

**职责**：融合视觉和文本信息进行最终分类

```python
class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vision_encoder = ResNet18(...)  # 512-d
        self.text_encoder = BertModel(...)    # 768-d
        
        # 融合层
        self.attention = MultiheadAttention(embed_dim=768, ...)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, image, text_tokens):
        img_feat = self.vision_encoder(image)
        txt_feat = self.text_encoder(text_tokens)
        
        # 特征融合
        fused = self._fuse_features(img_feat, txt_feat)
        
        # 分类
        logits = self.classifier(fused)
        return logits
```

**输入**：
- 视觉特征（从图像提取）
- 文本特征（从 OCR 结果提取）

**输出**：
- 分类结果
- 置信度分数
- 详细的推理信息

## API 接口设计

### FastAPI 服务

```python
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    分析上传的图像
    
    Args:
        file: 图像文件（JPEG/PNG）
    
    Returns:
        {
            "status": "success" | "error",
            "result": {
                "classification": "normal" | "anomaly" | ...,
                "confidence": 0.95,
                "regions": [
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": 0.87,
                        "text": "检测到的文本",
                        "category": "..."
                    }
                ]
            },
            "processing_time_ms": 234
        }
    """
```

### 批量处理

```python
@app.post("/analyze_batch")
async def analyze_batch(files: List[UploadFile]):
    """批量分析多个图像"""
    results = []
    for file in files:
        result = await analyze_image(file)
        results.append(result)
    return {"results": results}
```

## 性能优化

### 1. 推理加速

**Batch 处理**：
```python
def batch_inference(images, batch_size=32):
    """
    批量推理多个图像，提升吞吐
    """
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        with torch.no_grad():
            outputs = model(batch)
        results.extend(outputs)
    return results
```

**性能指标**：
- 单图推理：~150ms
- Batch 推理（bs=32）：~3.5ms/image

### 2. 缓存机制

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_model_features(image_hash):
    """
    缓存已处理图像的特征
    减少重复推理开销
    """
    # 如果同一图像多次查询，直接返回缓存
    return precomputed_features
```

### 3. 多进程推理

```python
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

def parallel_process(images):
    """并行处理多个图像"""
    futures = [executor.submit(process_image, img) for img in images]
    results = [f.result() for f in futures]
    return results
```

## 错误处理与恢复

### 常见故障及对策

#### 故障 1：模型加载失败

```python
def load_model_with_fallback(primary_path, fallback_path):
    """
    尝试从主路径加载，失败则用备用路径
    """
    try:
        return load_model(primary_path)
    except Exception as e:
        logger.warning(f"Failed to load {primary_path}: {e}")
        logger.info(f"Falling back to {fallback_path}")
        return load_model(fallback_path)
```

#### 故障 2：推理超时

```python
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Inference timeout")

def inference_with_timeout(model, input, timeout_sec=5):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    
    try:
        result = model(input)
        signal.alarm(0)  # 取消超时
        return result
    except TimeoutError:
        logger.error("Inference exceeded timeout limit")
        return default_result()
```

#### 故障 3：OOM（内存溢出）

```python
def infer_with_memory_check(model, input, max_memory_gb=8):
    """
    监控内存占用，避免 OOM
    """
    import psutil
    
    process = psutil.Process()
    before = process.memory_info().rss / 1e9
    
    try:
        result = model(input)
        after = process.memory_info().rss / 1e9
        
        if after > max_memory_gb:
            logger.warning(f"Memory usage: {after}GB (threshold: {max_memory_gb}GB)")
            torch.cuda.empty_cache()
        
        return result
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU OOM")
        return degrade_to_cpu(model, input)
```

## 监控与日志

### 关键指标

```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "avg_latency_ms": 0,
            "p95_latency_ms": 0,
            "p99_latency_ms": 0
        }
    
    def record_request(self, latency_ms, success):
        self.metrics["total_requests"] += 1
        if success:
            self.metrics["successful"] += 1
        else:
            self.metrics["failed"] += 1
        
        # 更新延迟统计
        self._update_latency_stats(latency_ms)
    
    def get_report(self):
        return {
            "success_rate": self.metrics["successful"] / self.metrics["total_requests"],
            "avg_latency": self.metrics["avg_latency_ms"],
            "p95_latency": self.metrics["p95_latency_ms"],
            "p99_latency": self.metrics["p99_latency_ms"]
        }
```

### 日志记录

```python
import logging

logger = logging.getLogger(__name__)

# 标准输出 + 文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system.log'),
        logging.StreamHandler()
    ]
)

# 关键事件
logger.info(f"Request received: {request_id}")
logger.info(f"Model inference completed in {latency}ms")
logger.error(f"Failed to process {file_name}: {error_msg}")
```

## 文件结构

```
content-filtering-pipeline/
├── README.md                    # 本文件
├── docs/
│   ├── architecture.md          # 系统架构详解
│   ├── api-design.md            # API 设计文档
│   ├── deployment.md            # 部署指南
│   └── troubleshooting.md       # 故障排查
├── code/
│   ├── models.py                # 模型定义
│   ├── detector.py              # 检测器实现
│   ├── ocr_processor.py         # OCR 处理
│   ├── classifier.py            # 分类器实现
│   ├── api_server.py            # FastAPI 服务
│   └── utils.py                 # 工具函数
├── examples/
│   ├── basic_usage.py           # 基本使用示例
│   ├── batch_processing.py      # 批处理示例
│   └── docker_deployment.md     # Docker 部署
└── LICENSE
```

## 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 框架 | PyTorch | 深度学习 |
| 检测 | ResNet18 | 轻量级特征提取 |
| OCR | PaddleOCR | 快速文本识别 |
| 文本编码 | BERT | 文本特征提取 |
| 服务 | FastAPI | RESTful API |
| 异步 | asyncio | 高并发处理 |

## 部署考虑

### 单机部署
- GPU：1× A100 或 2× RTX3090
- CPU：16+ 核
- 内存：64GB+
- 吞吐：~100 请求/秒

### 分布式部署
- 多机负载均衡
- 模型服务化（TensorFlow Serving、KServe）
- 缓存层（Redis）
- 消息队列（Kafka、RabbitMQ）

## 快速开始

1. **阅读架构** → `docs/architecture.md`
2. **了解 API** → `docs/api-design.md`
3. **查看示例** → `examples/`
4. **部署系统** → `docs/deployment.md`

## 许可证

MIT License - 详见 LICENSE 文件

## 后续优化方向

- [ ] 模型量化（INT8/FP16）提升推理速度
- [ ] 多模型 ensemble 提升准确度
- [ ] 在线学习机制适应新数据
- [ ] 可解释性增强（特征可视化、注意力图）

---

**详细技术内容请查阅 `docs/` 和 `code/` 目录。**
