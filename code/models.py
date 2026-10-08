"""
核心模型定义
包含检测器、分类器和多模态融合模块
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class TextDetector(nn.Module):
    """
    异常区域检测器
    基于 ResNet18 的轻量级特征提取
    """

    def __init__(self, num_features: int = 512):
        super().__init__()

        # 使用预训练的 ResNet18
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.IMAGENET1K_V1
        self.backbone = resnet18(weights=weights)

        # 移除最后的分类层，只保留特征提取
        self.backbone.fc = nn.Identity()

        self.num_features = num_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) RGB 图像张量

        Returns:
            features: (B, 512) 特征向量
        """
        features = self.backbone(images)
        return features

    def freeze(self):
        """冻结所有参数（仅用于特征提取）"""
        for param in self.parameters():
            param.requires_grad = False


class TextEncoder(nn.Module):
    """
    文本编码器
    使用 BERT 编码 OCR 识别的文本
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()

        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.embedding_dim = self.model.config.hidden_size

    def forward(self, texts: list) -> torch.Tensor:
        """
        Args:
            texts: ["text1", "text2", ...] 文本列表

        Returns:
            embeddings: (B, 768) 文本特征向量
        """

        # Tokenize
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )

        # 获取 BERT 输出
        with torch.no_grad():
            output = self.model(**encoded)

        # 使用 [CLS] token 作为句子表示
        embeddings = output.last_hidden_state[:, 0, :]

        return embeddings


class MultimodalFusion(nn.Module):
    """
    多模态特征融合
    使用多头自注意力融合视觉和文本特征
    """

    def __init__(
        self,
        visual_dim: int = 512,
        text_dim: int = 768,
        fusion_dim: int = 768,
        num_heads: int = 8,
    ):
        super().__init__()

        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)

        # 多头自注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, batch_first=True
        )

        self.fusion_dim = fusion_dim

    def forward(
        self, visual_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            visual_features: (B, 512) ResNet18 特征
            text_features: (B, 768) BERT 特征

        Returns:
            fused_features: (B, 768) 融合后的特征
        """

        # 投影到相同维度
        visual_proj = self.visual_proj(visual_features)  # (B, 768)
        text_proj = self.text_proj(text_features)  # (B, 768)

        # 视觉特征作为 query，文本特征作为 key/value
        # 这样模型可以学习如何用文本来增强视觉理解
        fused, _ = self.attention(
            visual_proj.unsqueeze(1),  # (B, 1, 768)
            text_proj.unsqueeze(1),  # (B, 1, 768)
            text_proj.unsqueeze(1),  # (B, 1, 768)
        )

        fused = fused.squeeze(1)  # (B, 768)

        return fused


class MultimodalClassifier(nn.Module):
    """
    端到端多模态分类器
    整合检测、编码和融合模块
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.detector = TextDetector(num_features=512)
        self.text_encoder = TextEncoder()

        self.fusion = MultimodalFusion(
            visual_dim=512, text_dim=768, fusion_dim=768, num_heads=8
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self.num_classes = num_classes

    def forward(
        self, images: torch.Tensor, texts: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W) RGB 图像
            texts: ["text1", ...] 文本列表

        Returns:
            logits: (B, num_classes) 分类 logits
            probs: (B, num_classes) 分类概率
        """

        # 视觉特征提取
        visual_feat = self.detector(images)  # (B, 512)

        # 文本特征提取
        text_feat = self.text_encoder(texts)  # (B, 768)

        # 特征融合
        fused_feat = self.fusion(visual_feat, text_feat)  # (B, 768)

        # 分类
        logits = self.classifier(fused_feat)  # (B, num_classes)
        probs = torch.softmax(logits, dim=-1)

        return logits, probs


# 使用示例
if __name__ == "__main__":
    import torch

    # 创建模型
    model = MultimodalClassifier(num_classes=2)
    model.eval()

    # 创建样本数据
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    texts = ["sample text 1", "sample text 2", "sample text 3", "sample text 4"]

    # 前向传播
    with torch.no_grad():
        logits, probs = model(images, texts)

    print(f"Logits shape: {logits.shape}")  # (4, 2)
    print(f"Probs shape: {probs.shape}")  # (4, 2)
    print(f"Predictions: {torch.argmax(probs, dim=-1)}")  # [0 or 1] x 4
