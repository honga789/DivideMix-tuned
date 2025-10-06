# dataloader_tuned_text.py
# Dataloader cho bài toán TEXT, bắt chước đúng giao diện của dataloader_tuned.py:
# - mode: warmup / train / test / eval_train
# - train: trả về (labeled_loader, unlabeled_loader)
# - warmup: DataLoader(all) với batch_size = 2*batch_size
# - eval_train: DataLoader(all) (không augment), shuffle=False
# - test: DataLoader(test), shuffle=False
#
# YÊU CẦU ARGUMENTS (đọc từ args):
#   train_csv_path        : CSV train (chứa cột text)
#   train_feather_path    : Feather train (chứa cột 'label' = noisy label dùng để train)
#   test_csv_path         : CSV test (chứa cột text + cột label)
#   train_data_column     : tên cột text trong train CSV
#   test_data_column      : tên cột text trong test CSV
#   test_label_column     : tên cột label trong test CSV (mặc định 'label')
#   pretrained_lm         : tên model tokenizer HF (vd: 'bert-base-uncased')
#   max_length            : độ dài token tối đa (vd: 256)
#   batch_size, num_workers
#
# Ghi chú:
# - Ở các mode liên quan train ('all' / 'labeled' / 'unlabeled'), LABEL sẽ lấy từ feather cột 'label'.
# - Hai “view” cho text hiện để giống nhau (không augment). Nếu muốn, có thể thêm noise/Dropout ở model.
# - pred/prob truyền vào run(mode='train', pred, prob) là mảng/Series theo thứ tự hàng của TRAIN CSV.

from typing import Dict, List, Tuple, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def _stack_dict(list_of_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = list_of_dicts[0].keys()
    return {k: torch.stack([d[k] for d in list_of_dicts], dim=0) for k in keys}


def _collate_labeled(batch):
    # batch: list of (enc1, enc2, noisy_label, prob)
    enc1 = _stack_dict([b[0] for b in batch])
    enc2 = _stack_dict([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    prob = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    return enc1, enc2, labels, prob


def _collate_unlabeled(batch):
    # batch: list of (enc1, enc2)
    enc1 = _stack_dict([b[0] for b in batch])
    enc2 = _stack_dict([b[1] for b in batch])
    return enc1, enc2


def _collate_all(batch):
    # batch: list of (enc1, noisy_label, real_idx)
    enc1 = _stack_dict([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    real_idx = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return enc1, labels, real_idx


def _collate_test(batch):
    # batch: list of (enc, label)
    enc = _stack_dict([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return enc, labels


class _TrainBase:
    """
    Dataset nền cho train/eval_train (lấy TEXT từ train CSV, LABEL từ feather cột 'label')
    """
    def __init__(
        self,
        train_csv_path: str,
        train_feather_path: str,
        text_col: str,
        tokenizer: AutoTokenizer,
        max_length: int,
    ):
        self.df_text = pd.read_csv(train_csv_path).reset_index(drop=False).rename(columns={"index": "real_idx"})
        self.df_lab = pd.read_feather(train_feather_path)
        if len(self.df_text) != len(self.df_lab):
            raise ValueError(
                f"Length mismatch: train CSV ({len(self.df_text)}) vs feather ({len(self.df_lab)}). "
                f"Yêu cầu cùng thứ tự và cùng số hàng."
            )
        if "label" not in self.df_lab.columns:
            raise ValueError("Feather train phải có cột 'label' (noisy label dùng để train).")

        self.text_col = text_col
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df_text)

    def _encode(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tok(
            str(text),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def _get_text(self, i: int) -> str:
        return self.df_text.iloc[i][self.text_col]

    def _get_noisy_label(self, i: int) -> int:
        return int(self.df_lab.iloc[i]["label"])

    def _get_real_idx(self, i: int) -> int:
        return int(self.df_text.iloc[i]["real_idx"])


class TextAllDataset(_TrainBase):
    """ mode='all' & 'eval_all': trả (enc1, noisy_label, real_idx) """
    def __getitem__(self, i: int):
        text = self._get_text(i)
        enc1 = self._encode(text)
        y = self._get_noisy_label(i)
        ridx = self._get_real_idx(i)
        return enc1, y, ridx


class TextLabeledDataset(_TrainBase):
    """ mode='labeled': lọc theo pred==True; trả (enc1, enc2, noisy_label, prob) """
    def __init__(self, *args, pred: List[bool], prob: List[float], **kwargs):
        super().__init__(*args, **kwargs)
        if len(pred) != len(self.df_text) or len(prob) != len(self.df_text):
            raise ValueError("Độ dài pred/prob phải khớp với số hàng train CSV.")
        self.df_text = self.df_text.assign(pred=list(pred), prob=list(prob))
        self.keep_idx = self.df_text.index[self.df_text["pred"] == True].tolist()  # noqa: E712

    def __len__(self):
        return len(self.keep_idx)

    def __getitem__(self, j: int):
        i = self.keep_idx[j]
        text = self._get_text(i)
        enc1 = self._encode(text)
        enc2 = self._encode(text)  # view2 = giống nhau
        y = self._get_noisy_label(i)
        p = float(self.df_text.iloc[i]["prob"])
        return enc1, enc2, y, p


class TextUnlabeledDataset(_TrainBase):
    """ mode='unlabeled': lọc theo pred==False; trả (enc1, enc2) """
    def __init__(self, *args, pred: List[bool], **kwargs):
        super().__init__(*args, **kwargs)
        if len(pred) != len(self.df_text):
            raise ValueError("Độ dài pred phải khớp với số hàng train CSV.")
        self.df_text = self.df_text.assign(pred=list(pred))
        self.keep_idx = self.df_text.index[self.df_text["pred"] == False].tolist()  # noqa: E712

    def __len__(self):
        return len(self.keep_idx)

    def __getitem__(self, j: int):
        i = self.keep_idx[j]
        text = self._get_text(i)
        enc1 = self._encode(text)
        enc2 = self._encode(text)  # view2 = giống nhau
        return enc1, enc2


class TextTestDataset(Dataset):
    """
    Dataset cho test: text + label đều lấy từ test CSV
    """
    def __init__(
        self,
        test_csv_path: str,
        text_col: str,
        label_col: str,
        tokenizer: AutoTokenizer,
        max_length: int,
    ):
        self.df = pd.read_csv(test_csv_path).reset_index(drop=True)
        if text_col not in self.df.columns:
            raise ValueError(f"Thiếu cột text '{text_col}' trong test CSV.")
        if label_col not in self.df.columns:
            raise ValueError(f"Thiếu cột label '{label_col}' trong test CSV.")
        self.text_col = text_col
        self.label_col = label_col
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def _encode(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tok(
            str(text),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        enc = self._encode(row[self.text_col])
        y = int(row[self.label_col])
        return enc, y


class Loader:
    """
    Sử dụng:
        loader = Loader(args)
        warm = loader.run('warmup')
        (L, U) = loader.run('train', pred=pred_bool_array, prob=prob_float_array)
        test = loader.run('test')
        eval = loader.run('eval_train')
    """
    def __init__(self, args):
        self.A = args
        # Cho phép fallback tên cột để hạn chế lỗi chính tả
        self.train_text_col = getattr(args, "train_data_column", "text")
        self.test_text_col = getattr(args, "test_data_column", "text")
        self.test_label_col = getattr(args, "test_label_column", "label")

        self.tok = AutoTokenizer.from_pretrained(self.A.pretrained_lm, use_fast=True)

    def run(self, mode: str, pred: Optional[List[bool]] = None, prob: Optional[List[float]] = None):
        A = self.A
        if mode == "warmup":
            ds = TextAllDataset(
                A.train_csv_path, A.train_feather_path,
                self.train_text_col, self.tok, A.max_length
            )
            return DataLoader(
                ds, batch_size=A.batch_size * 2, shuffle=True,
                num_workers=A.num_workers, pin_memory=True, collate_fn=_collate_all
            )

        if mode == "train":
            if pred is None or prob is None:
                raise ValueError("Cần cung cấp pred và prob cho mode='train'.")
            ds_l = TextLabeledDataset(
                A.train_csv_path, A.train_feather_path,
                self.train_text_col, self.tok, A.max_length,
                pred=list(map(bool, pred)), prob=list(map(float, prob))
            )
            ds_u = TextUnlabeledDataset(
                A.train_csv_path, A.train_feather_path,
                self.train_text_col, self.tok, A.max_length,
                pred=list(map(bool, pred))
            )
            L = DataLoader(
                ds_l, batch_size=A.batch_size, shuffle=True,
                num_workers=A.num_workers, pin_memory=True, collate_fn=_collate_labeled
            )
            U = DataLoader(
                ds_u, batch_size=A.batch_size, shuffle=True,
                num_workers=A.num_workers, pin_memory=True, collate_fn=_collate_unlabeled
            )
            return L, U

        if mode == "test":
            ds = TextTestDataset(
                A.test_csv_path, self.test_text_col, self.test_label_col,
                self.tok, A.max_length
            )
            return DataLoader(
                ds, batch_size=A.batch_size, shuffle=False,
                num_workers=A.num_workers, pin_memory=True, collate_fn=_collate_test
            )

        if mode == "eval_train":
            ds = TextAllDataset(
                A.train_csv_path, A.train_feather_path,
                self.train_text_col, self.tok, A.max_length
            )
            return DataLoader(
                ds, batch_size=A.batch_size, shuffle=False,
                num_workers=A.num_workers, pin_memory=True, collate_fn=_collate_all
            )

        raise ValueError(f"Unknown mode: {mode}")
