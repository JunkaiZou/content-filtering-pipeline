# 系统架构与设计文档

## 1. 系统概览

### 核心功能

本系统是一个**生产级内容分析系统**，用于：
- 检测图像中的异常/可疑区域
- 识别这些区域中的文本内容
- 融合多模态信息进行智能分类
- 提供高可靠性和可扩展性

### 技术栈

**深度学习框架**：
- PyTorch（核心计算）
- Transformers（预训练模型）

**模型选择**：
- ResNet18（轻量级特征提取）
- BERT（文本编码）
- 多头自注意力（特征融合）

**服务架构**：
- FastAPI（RESTful 接口）
- asyncio（异步处理）

---

## 2. 模块设计

### 2.1 检测器（Detector）

**职责**：识别图像中的可疑区域

**架构**：
```python
class MedicalImageDetector:
    def __init__(self):
        self.backbone = ResNet18(pretrained=True)
    
    def detect(self, image: np.ndarray) -> Dict:
        # 返回检测到的区域
        return {
            "regions": [...],        # 推断特征
            "bboxes": [...],        # 边界框
            "confidences": [...]    # 置信度
        }
```

**关键设计**：
- 使用预训练的 ResNet18 作为骨干网络
- 冻结大部分权重，仅使用特征提取能力
- 输出 512-维特征向量

**输入输出**：
- 输入：RGB 图像，任意分辨率
- 输出：K 个检测区域，每个包含特征向量、bbox、置信度

### 2.2 OCR 处理器（OCR Processor）

**职责**：从检测区域中提取文本

**使用的库**：PaddleOCR（快速且准确）

**处理流程**：
```python
def extract_text(regions: List[np.ndarray]) -> List[Dict]:
    ocr = PaddleOCR(use_angle_cls=True)
    
    results = []
    for region in regions:
        # 识别文本
        ocr_result = ocr.ocr(region, cls=True)
        
        # 提取和格式化
        text = format_ocr_result(ocr_result)
        
        results.append({
            "text": text,
            "confidence": ocr_result.confidence,
            "bbox": ocr_result.bbox
        })
    
    return results
```

**关键优化**：
- 启用角度检测（`use_angle_cls=True`）处理倾斜文本
- 批量处理多个区域以提升吞吐
- 容错处理：OCR 失败的区域使用空字符串

### 2.3 特征融合模块

**职责**：融合视觉和文本特征

**架构设计**：

**视觉通道**：
- 输入：RGB 图像 (3, H, W)
- ResNet18 特征：512-维向量
- 输出：512-维视觉特征

**文本通道**：
- 输入：OCR 识别的文本字符串
- BERT Tokenizer：转换为 token IDs
- BERT 编码器：生成 768-维文本特征
- 输出：768-维文本特征

**融合层**：
```python
class MultiheadAttention(nn.Module):
    """多头自注意力融合"""
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
    
    def forward(self, visual_feat, text_feat):
        # 视觉特征作为 query
        # 文本特征作为 key 和 value
        fused, _ = self.attention(
            visual_feat.unsqueeze(0),
            text_feat.unsqueeze(0),
            text_feat.unsqueeze(0)
        )
        return fused.squeeze(0)
```

**为什么用注意力机制？**
- 动态学习视觉和文本特征的关键信息
- 不同样本可能需要不同的融合比例
- 自注意力天然支持跨模态交互

### 2.4 分类器（Classifier）

**职责**：基于融合特征做最终分类

**架构**：
```python
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(512 + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, fused_feat):
        logits = self.fc(fused_feat)
        return logits
```

**关键设计**：
- 输入维度：1280（512 + 768）
- 隐层维度：256（大幅降维）
- Dropout：0.5（正则化，防止过拟合）
- 输出：分类 logits

---

## 3. 数据流设计

### 3.1 单个样本的处理流程

```
输入图像 (RGB, HxW)
    ↓
[图像预处理]
- 转换色彩空间
- 尺寸调整到标准分辨率
- 值域归一化 [0,1] 或 z-score
    ↓
[检测器]
- ResNet18 提取特征
- 输出 K 个检测框 + 特征向量
    ↓
[区域裁剪]
- 从原图中裁剪每个检测框的像素
    ↓
[并行 OCR]
- 对 K 个区域并行识别文本
- 输出 K 个文本字符串
    ↓
[特征融合]
对每个检测框：
- 视觉特征来自 ResNet18
- 文本特征来自 BERT 编码
- 多头注意力融合
    ↓
[分类]
- FC 网络生成 logits
- Softmax 得到概率分布
    ↓
输出结果：
- 分类类别
- 置信度分数
- 关键信息
```

### 3.2 批处理优化

**为什么需要 batch 处理？**
- 显著提升吞吐：150ms → 3.5ms/image（32 batch）
- 充分利用 GPU 并行计算能力
- 降低 API 请求延迟

**Batch 处理的流程**：

```python
def batch_inference(images: List[np.ndarray], batch_size=32):
    """批量处理多个图像"""
    
    results = []
    
    # 按 batch 切片
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        
        # 批处理
        with torch.no_grad():
            # 统一预处理
            processed = torch.stack([preprocess(img) for img in batch])
            
            # 批量推理
            outputs = model(processed)
            
            # 后处理
            batch_results = [postprocess(out) for out in outputs]
        
        results.extend(batch_results)
    
    return results
```

**关键优化点**：
- 异步数据加载（prefetch）
- GPU 流式处理
- 内存固定化（pinned memory）

---

## 4. 错误处理与恢复

### 4.1 常见故障模式与对策

**故障 1：检测器输出空结果**

原因：图像中没有可疑区域

对策：
```python
if len(detections) == 0:
    return {
        "status": "normal",
        "regions": [],
        "message": "No suspicious regions detected"
    }
```

**故障 2：OCR 识别失败**

原因：文字过小、模糊或非文本内容

对策：
```python
try:
    text = ocr.ocr(region)
except Exception as e:
    text = ""  # 用空字符串替代
    logger.warning(f"OCR failed for region: {e}")
```

**故障 3：推理超时**

原因：某个样本的处理耗时过长

对策：
```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Inference exceeded timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(5)  # 5 秒超时

try:
    result = model(input)
    signal.alarm(0)  # 取消超时
except TimeoutError:
    logger.error("Inference timeout")
    return default_result()
```

**故障 4：GPU OOM**

原因：显存不足，通常因为 batch size 过大

对策：
```python
try:
    result = model(input)
except torch.cuda.OutOfMemoryError:
    # 回退方案：CPU 推理
    logger.warning("GPU OOM, falling back to CPU")
    model.cpu()
    result = model(input)
    model.cuda()
```

### 4.2 重试策略

**指数回退重试**：

```python
def retry_with_backoff(func, max_retries=3):
    """带指数回退的重试机制"""
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            logger.warning(f"Attempt {attempt+1} failed, retrying in {wait_time}s: {e}")
            time.sleep(wait_time)
```

---

## 5. 性能优化

### 5.1 推理加速

**量化**：
- FP32 → FP16：2x 速度，内存减半
- INT8：进一步加速，精度轻微下降

**模型蒸馏**：
- ResNet18 本身已是轻量级模型
- 若需要更快，可蒸馏到 MobileNet

**批量处理**：
- Batch size=1：150ms/image
- Batch size=32：3.5ms/image（42x 加速）

### 5.2 内存管理

**显存优化**：
```python
# 模型加载到 GPU
model = model.cuda()

# 推理时禁用梯度计算
with torch.no_grad():
    output = model(input)

# 定期清理缓存
torch.cuda.empty_cache()
```

**CPU 内存管理**：
```python
# 显式删除大对象
del large_tensor
gc.collect()

# 使用上下文管理器自动释放
with torch.no_grad():
    # 临时张量在此作用域自动释放
    result = model(input)
```

### 5.3 缓存策略

**结果缓存**：
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_features(image_hash: str):
    """缓存已处理图像的特征"""
    # 如果同一图像多次查询，直接返回缓存
    return precomputed_features[image_hash]
```

**预热**：
```python
def warmup_model(model, device):
    """模型预热，消除冷启动延迟"""
    
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
```

---

## 6. 监控与可观测性

### 6.1 关键指标

**延迟指标**（Latency）：
- P50、P95、P99 延迟
- 按模块分解：检测、OCR、融合、分类

**吞吐指标**（Throughput）：
- 每秒处理图像数
- 每秒处理样本数

**精度指标**（Accuracy）：
- 分类准确率
- 精确率、召回率、F1 分数

**系统健康**（System Health）：
- GPU 利用率
- 显存占用
- 错误率

### 6.2 日志记录

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 关键事件记录
logger.info(f"Request received: {request_id}")
logger.info(f"Inference completed: {latency}ms, result: {result}")
logger.error(f"Inference failed: {error_msg}")
```

### 6.3 性能监控

```python
class PerformanceMonitor:
    def __init__(self):
        self.request_times = []
        self.error_count = 0
        self.total_requests = 0
    
    def record_request(self, latency_ms: float, success: bool):
        self.total_requests += 1
        
        if success:
            self.request_times.append(latency_ms)
        else:
            self.error_count += 1
    
    def get_stats(self):
        return {
            "p50": np.percentile(self.request_times, 50),
            "p95": np.percentile(self.request_times, 95),
            "p99": np.percentile(self.request_times, 99),
            "error_rate": self.error_count / self.total_requests,
            "throughput": len(self.request_times) / sum(self.request_times) * 1000
        }
```

---

## 7. 可扩展性设计

### 7.1 多机部署

**负载均衡**：
```
客户端 → 负载均衡器 (Nginx/HAProxy)
         ↓
    [Worker 1]  [Worker 2]  [Worker 3]
         ↓
    共享 Redis 缓存
```

### 7.2 异步处理

**队列系统**：
```
API → 消息队列 (Kafka/RabbitMQ) → Worker → 结果存储
                    ↑
                 异步处理
```

### 7.3 模型更新

**无缝更新**：
- 保持旧模型在线
- 新模型在后台加载测试
- 验证通过后才切换
- 支持灰度发布

