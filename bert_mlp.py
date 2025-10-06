# bert_mlp.py
# BERT backbone + MLP head (freezable). Giữ chữ ký forward(x, lin=0, lout=5) tương thích code hiện có.
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BertMLP(nn.Module):
    """
    Args:
        pretrained_name (str): model name, vd: 'bert-base-uncased'
        num_classes (int): số lớp output
        freeze_backbone (bool): đóng băng BERT (mặc định True)
        use_pooler (bool): ưu tiên dùng outputs.pooler_output nếu có, else dùng CLS token
    """
    def __init__(
        self,
        pretrained_name: str,
        num_classes: int,
        freeze_backbone: bool = True,
        use_pooler: bool = True,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.use_pooler = use_pooler

        self.config = AutoConfig.from_pretrained(pretrained_name)
        self.backbone = AutoModel.from_pretrained(pretrained_name, config=self.config)

        in_dim = getattr(self.config, "hidden_size", 768)
        out_dim = num_classes

        # MLP head theo yêu cầu
        self.net = nn.Sequential(
            # Block 1
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            # Block 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            # Output layer
            nn.Linear(256, out_dim),
        )

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # đảm bảo backbone không có dropout khi train
            self.backbone.eval()

    def _feat_from_outputs(self, outputs):
        # Ưu tiên pooler_output nếu có
        if self.use_pooler and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output  # [B, H]
        # Fallback: CLS token
        return outputs.last_hidden_state[:, 0]  # [B, H]

    @torch.no_grad()
    def encode(self, x):
        """
        Trả về đặc trưng (vector H) từ backbone (bỏ MLP).
        Không gradient để dùng chung cho cả freeze/unfreeze.
        """
        outputs = self.backbone(**x)
        return self._feat_from_outputs(outputs)

    def forward(self, x, lin: int = 0, lout: int = 5):
        """
        x: dict {'input_ids','attention_mask', (optional) 'token_type_ids'}
        lin/lout chỉ để tương thích, không sử dụng.
        """
        if not isinstance(x, dict):
            raise TypeError("BertMLP expects a dict batch (tokenized).")

        if self.freeze_backbone:
            # đảm bảo backbone không dropout ngay cả khi model.train()
            self.backbone.eval()
            with torch.no_grad():
                outputs = self.backbone(**x)
        else:
            outputs = self.backbone(**x)

        feats = self._feat_from_outputs(outputs)  # [B, H]
        logits = self.net(feats)                  # [B, C]
        return logits

    def set_freeze(self, freeze: bool = True):
        """Bật/tắt đóng băng backbone trong lúc chạy."""
        self.freeze_backbone = freeze
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        if freeze:
            self.backbone.eval()

    @property
    def feature_dim(self) -> int:
        return getattr(self.config, "hidden_size", 768)
