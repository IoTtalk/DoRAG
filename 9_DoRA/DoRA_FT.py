import argparse
from pathlib import Path
import shutil
from typing import List, Tuple, Dict, Any, Optional
import random
import json
import hashlib
import csv
from contextlib import nullcontext
from collections import deque, defaultdict
import os
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm  
import matplotlib.pyplot as plt


# =========================
# DoRA / DoRA+LoRA modules
# =========================
class DoRALinearWithLoRA(nn.Module):
    """W ≈ (V + A@B*scaling) * diag(m)；訓練 m, A, B（dropout 只套在 LoRA 分支）"""
    def __init__(self, in_features, out_features, bias=True, r=8, alpha=16.0, dropout_p=0.0, eps=1e-12):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(r, 1)

        # 將 dropout 留給 LoRA 分支（eval 模式自動關閉）
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        # DoRA
        self.m = nn.Parameter(torch.empty(in_features))
        self.register_buffer("V", torch.empty(out_features, in_features), persistent=True)

        # LoRA
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(out_features, r))
            self.lora_B = nn.Parameter(torch.zeros(r, in_features))
            nn.init.normal_(self.lora_A, std=1e-4)
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def init_from_linear(self, linear: nn.Linear):
        W = linear.weight.data.clone()
        m = torch.linalg.vector_norm(W, dim=0)
        safe = m.clone(); safe[safe < self.eps] = 1.0
        V = W / safe.unsqueeze(0)
        self.m.copy_(m)
        self.V.copy_(V)
        if self.bias is not None and linear.bias is not None:
            self.bias.data.copy_(linear.bias.data)

    @classmethod
    def from_linear(cls, linear: nn.Linear, r=8, alpha=16.0, dropout_p=0.0):
        mod = cls(linear.in_features, linear.out_features, bias=(linear.bias is not None),
                  r=r, alpha=alpha, dropout_p=dropout_p)
        mod = mod.to(device=linear.weight.device, dtype=linear.weight.dtype)
        mod.init_from_linear(linear)
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 僅在 LoRA 分支使用 dropout；V 主幹保持穩定
        if self.lora_A is not None:
            A = self.dropout(self.lora_A)        # 訓練時隨機丟棄 LoRA A；eval 時自動關閉
            delta_V = (A @ self.lora_B) * self.scaling
            direction = self.V + delta_V
        else:
            # r==0：沒有 LoRA，direction 就是 V
            direction = self.V

        # === DoRA: column-wise normalization (paper 的版本) ===
        # 對每個 column 做向量 norm，避免除以 0
        col_norm = torch.linalg.vector_norm(direction, dim=0, keepdim=True)
        col_norm = col_norm.clamp_min(self.eps)
        direction_hat = direction / col_norm      # 純方向 (unit-norm per column)

        # 最終權重：W' = (direction_hat) * diag(m)
        weight = direction_hat * self.m.unsqueeze(0)
        return F.linear(x, weight, self.bias)


# =========================
# Replace helpers
# =========================

def replace_linears_with_dora_lora(module: nn.Module, name_filters=None, r=8, alpha=16.0, dropout_p=0.0):
    replaced, names = 0, []
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if (not name_filters) or any(f in child_name for f in name_filters):
                setattr(module, child_name, DoRALinearWithLoRA.from_linear(child, r=r, alpha=alpha, dropout_p=dropout_p))
                replaced += 1
                names.append(child_name)
        else:
            sub_n, sub_names = replace_linears_with_dora_lora(child, name_filters, r=r, alpha=alpha, dropout_p=dropout_p)
            replaced += sub_n
            names.extend([f"{child_name}.{n}" for n in sub_names])
    return replaced, names


# =========================
# Data utils
# =========================
Entry = Dict[str, Any]

def read_json_or_jsonl(path: Path) -> List[Entry]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            raise ValueError("JSON 應為 list[dict]。")
    else:
        raise ValueError(f"不支援的副檔名：{path.suffix}")

def basic_validate(rows: List[Entry], group_field: str):
    req = {"query", "positive_doc", "hard_negative_doc", group_field}
    miss_any = req - set(rows[0].keys())
    if miss_any:
        raise ValueError(f"資料缺欄位：{list(miss_any)}")

def group_indices(rows: List[Entry], group_field: str) -> Dict[Any, List[int]]:
    g: Dict[Any, List[int]] = {}
    for i, r in enumerate(rows):
        g.setdefault(r[group_field], []).append(i)
    return g

def _pick_groups_for_target(groups: Dict[Any, List[int]], target_items: int, seed: int):
    rng = random.Random(seed)
    keys = list(groups.keys())
    rng.shuffle(keys)
    #keys.sort(key=lambda k: len(groups[k]), reverse=True)
    chosen, count = set(), 0
    for k in keys:
        if count >= target_items: break
        chosen.add(k); count += len(groups[k])
    if not chosen and keys:
        chosen.add(keys[0])
    if len(chosen) == len(keys) and len(chosen) > 1:
        smallest = min(keys, key=lambda k: len(groups[k]))
        chosen.remove(smallest)
    return chosen, count

def split_grouped(rows: List[Entry], group_field="chunk_id", test_ratio=0.05, dev_ratio=0.05, seed=42):
    total = len(rows)
    groups = group_indices(rows, group_field)
    target_test = max(1, int(round(total * test_ratio)))
    test_groups, _ = _pick_groups_for_target(groups, target_items=target_test, seed=seed)
    remaining = {k: v for k, v in groups.items() if k not in test_groups}
    target_dev = max(1, int(round(total * dev_ratio)))
    dev_groups, _ = _pick_groups_for_target(remaining, target_items=target_dev, seed=seed+1)

    # 為了可重現性，先固定展開順序
    test_rows = [rows[i] for g in sorted(test_groups) for i in groups[g]]
    dev_rows  = [rows[i] for g in sorted(dev_groups)  for i in groups[g]]
    train_rows = [rows[i] for g in sorted(groups.keys())
                  if g not in test_groups and g not in dev_groups
                  for i in groups[g]]

    # ★ split 內再做一次 shuffle（避免同 chunk_id 資料在檔案原始順序聚集）
    rng = random.Random(seed)
    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)
    rng.shuffle(test_rows)

    # 驗證互斥
    def uniq(rs): return {r[group_field] for r in rs}
    assert uniq(train_rows).isdisjoint(uniq(test_rows))
    assert uniq(train_rows).isdisjoint(uniq(dev_rows))
    assert uniq(dev_rows).isdisjoint(uniq(test_rows))
    return train_rows, dev_rows, test_rows


# =========================
# Embedding helpers & Loss
# =========================
def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    return last_hidden_states[torch.arange(last_hidden_states.size(0), device=last_hidden_states.device), seq_lens]

def get_doc_ids_from_batch(rows_batch: List[Entry], prefer_field="chunk_id") -> torch.Tensor:
    ids = []
    for r in rows_batch:
        if prefer_field in r:
            ids.append(int(r[prefer_field]))
        else:
            h = int.from_bytes(hashlib.sha1(r["positive_doc"].encode("utf-8")).digest()[:8], "big", signed=False)
            ids.append(h)
    return torch.tensor(ids, dtype=torch.long)

def masked_infonce(q, p, doc_ids, temperature=0.05):
    logits = (q @ p.T) / temperature
    B = logits.size(0)
    labels = torch.arange(B, device=logits.device)

    same_doc = doc_ids[:, None] == doc_ids[None, :]
    eye = torch.eye(B, dtype=torch.bool, device=logits.device)
    mask = same_doc & (~eye)

    logits = logits.masked_fill(mask, float("-inf"))
    loss = F.cross_entropy(logits, labels)
    return loss, logits

def hard_negative_margin(q, p, n, margin=0.2):
    s_pos = (q * p).sum(dim=1)
    s_neg = (q * n).sum(dim=1)
    return F.relu(margin - s_pos + s_neg).mean()

def retrieval_loss_masked(q, p, n, doc_ids, temperature=0.05, margin=0.2, lambda_hn=0.5):
    loss_infonce, _ = masked_infonce(q, p, doc_ids, temperature=temperature)
    loss_hn = hard_negative_margin(q, p, n, margin=margin)
    return loss_infonce + lambda_hn * loss_hn, {"infonce": loss_infonce.item(), "hard_neg": loss_hn.item()}

def pairwise_accuracy_batch(q, p, n):
    pos = (q * p).sum(dim=1)
    neg = (q * n).sum(dim=1)
    return (pos > neg).float().mean().item()

# ===== 新增：兩個補充指標 =====
def inbatch_recall_at1(q: torch.Tensor, p: torch.Tensor, doc_ids: torch.Tensor) -> float:
    """
    In-batch R@1（排除同 doc 的重複正例）
    對每個 query，在整個 batch 的所有正樣本中找 argmax，
    但會把同一份 doc（除了本身配對）遮掉。
    """
    logits = q @ p.T
    B = logits.size(0)
    same_doc = doc_ids[:, None] == doc_ids[None, :]
    eye = torch.eye(B, dtype=torch.bool, device=logits.device)
    dup_mask = same_doc & (~eye)        # 其他樣本與我同 doc 的正例 -> 遮
    logits = logits.masked_fill(dup_mask, float("-inf"))
    top = torch.argmax(logits, dim=1)
    return (top == torch.arange(B, device=logits.device)).float().mean().item()

def pairwise_margin_acc(q: torch.Tensor, p: torch.Tensor, n: torch.Tensor, margin: float) -> float:
    """
    與 hard-negative margin 對齊的成對準確率：
    1 if s(q,p) > s(q,n) + margin else 0
    """
    sp = (q * p).sum(dim=1)
    sn = (q * n).sum(dim=1)
    return (sp > sn + margin).float().mean().item()


# =========================
# IO helpers (save/plot/merge)
# =========================
def save_model_bundle(model, tokenizer, base_local_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    for py in base_local_dir.glob("*.py"):
        shutil.copy2(py, out_dir / py.name)
    readme = base_local_dir / "README.md"
    if readme.exists():
        shutil.copy2(readme, out_dir / "README.md")

def _ema(xs, alpha=0.2):
    if alpha is None or alpha <= 0 or alpha >= 1:
        return xs
    out = []
    m = None
    for v in xs:
        m = v if m is None else (alpha * v + (1 - alpha) * m)
        out.append(m)
    return out

def plot_curves_from_csv(csv_path: Path, out_png: Path, ema: float = None, show_epoch_boundaries: bool = True):
    """
    同時輸出 per_micro 和 per_step 兩種圖。
    ema: 指數滑動平均的係數（0~1，越大越平滑；None=不平滑）
    """
    # 讀 CSV
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "epoch": int(r.get("epoch", 1) or 1),
                "step":  int(r.get("step", 0) or 0),
                "loss":  float(r.get("loss_total", 0.0) or 0.0),
                "acc":   float(r.get("batch_acc", 0.0) or 0.0),
                "r1":    float(r.get("batch_r1", 0.0) or 0.0),
                "macc":  float(r.get("batch_acc_margin", 0.0) or 0.0),
                "v_loss": (r.get("val_loss_total", "") or ""),
            })

    if not rows:
        print("[Plot] CSV is empty.")
        return

    def _plot(mode: str, suffix: str):
        if mode == "per_micro":
            # micro-batch
            x = list(range(1, len(rows) + 1))
            losses = [r["loss"] for r in rows]
            accs   = [r["acc"]  for r in rows]
            r1s    = [r["r1"]   for r in rows]
            maccs  = [r["macc"] for r in rows]
            epochs = [r["epoch"] for r in rows]
            boundaries = [i+1 for i in range(1, len(epochs)) if epochs[i] != epochs[i-1]]
            x_label = "micro step"
            v_steps, v_losses = [], []
            for i, r in enumerate(rows, start=1):
                if r["v_loss"] != "":
                    v_steps.append(i)
                    v_losses.append(float(r["v_loss"]))
        else:
            # per-step 平均
            from collections import defaultdict
            from statistics import mode as stats_mode
            buckets = defaultdict(lambda: {"loss": [], "acc": [], "r1": [], "macc": [], "epoch": []})
            for r in rows:
                s = r["step"]
                buckets[s]["loss"].append(r["loss"])
                buckets[s]["acc"].append(r["acc"])
                buckets[s]["r1"].append(r["r1"])
                buckets[s]["macc"].append(r["macc"])
                buckets[s]["epoch"].append(r["epoch"])
            steps_sorted = sorted(buckets.keys())
            x = steps_sorted
            losses = [sum(buckets[s]["loss"]) / len(buckets[s]["loss"]) for s in steps_sorted]
            accs   = [sum(buckets[s]["acc"])  / len(buckets[s]["acc"])  for s in steps_sorted]
            r1s    = [sum(buckets[s]["r1"])   / len(buckets[s]["r1"])   for s in steps_sorted]
            maccs  = [sum(buckets[s]["macc"]) / len(buckets[s]["macc"]) for s in steps_sorted]
            epochs = [stats_mode(buckets[s]["epoch"]) for s in steps_sorted]
            boundaries = [i for i in range(1, len(epochs)) if epochs[i] != epochs[i-1]]
            x_label = "optimizer step"
            v_steps, v_losses = [], []
            for r in rows:
                if r["v_loss"] != "" and r["step"] in steps_sorted:
                    v_steps.append(r["step"])
                    v_losses.append(float(r["v_loss"]))

        # 平滑
        losses_s = _ema(losses, ema)
        accs_s   = _ema(accs, ema)
        r1s_s    = _ema(r1s, ema)
        maccs_s  = _ema(maccs, ema)
        v_losses_s = _ema(v_losses, ema) if v_losses else None

        # 畫圖
        plt.figure(figsize=(10, 8))
        ax1 = plt.subplot(2, 2, 1); ax1.plot(x, losses_s); ax1.set_title("Training Loss"); ax1.set_xlabel(x_label); ax1.set_ylabel("loss")
        if v_steps:
            ax1.plot(v_steps, v_losses_s if v_losses_s is not None else v_losses, linestyle="--")
            ax1.legend(["Train", "Validation"], loc="best")
        ax2 = plt.subplot(2, 2, 2); ax2.plot(x, accs_s);   ax2.set_title("Pairwise Acc");  ax2.set_xlabel(x_label); ax2.set_ylabel("acc")
        ax3 = plt.subplot(2, 2, 3); ax3.plot(x, r1s_s);    ax3.set_title("In-batch R@1");  ax3.set_xlabel(x_label); ax3.set_ylabel("R@1")
        ax4 = plt.subplot(2, 2, 4); ax4.plot(x, maccs_s);  ax4.set_title("Margin Acc");    ax4.set_xlabel(x_label); ax4.set_ylabel("acc")
        if show_epoch_boundaries and boundaries:
            for ax in [ax1, ax2, ax3, ax4]:
                for b in boundaries:
                    ax.axvline(x[b-1], linestyle="--", linewidth=0.8, alpha=0.5)
        plt.tight_layout()
        out_file = out_png.with_name(f"{out_png.stem}_{suffix}{out_png.suffix}")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, dpi=180)
        plt.close()
        print(f"[Plot] saved -> {out_file} (mode={mode}, ema={ema})")

    # 同時輸出兩張圖
    _plot("per_micro", "per_micro")
    _plot("per_step", "per_step")


def _get_module_by_name(root: nn.Module, name: str) -> nn.Module:
    mod = root
    for attr in name.split("."):
        mod = getattr(mod, attr)
    return mod

@torch.no_grad()
def _effective_weight_and_bias(dora_like: nn.Module) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    '''if isinstance(dora_like, DoRALinear):
        W = dora_like.V * dora_like.m.unsqueeze(0)
        b = dora_like.bias if getattr(dora_like, "bias", None) is not None else None
        return W, b'''
    if isinstance(dora_like, DoRALinearWithLoRA):
        if dora_like.lora_A is not None:
            delta = (dora_like.lora_A @ dora_like.lora_B) * dora_like.scaling
        else:
            delta = 0.0

        # 與 forward 一致：先做方向 + LoRA，再做 column-wise normalization
        direction = dora_like.V + delta
        col_norm = torch.linalg.vector_norm(direction, dim=0, keepdim=True)
        col_norm = col_norm.clamp_min(dora_like.eps)
        direction_hat = direction / col_norm

        W = direction_hat * dora_like.m.unsqueeze(0)
        b = dora_like.bias if getattr(dora_like, "bias", None) is not None else None
        return W, b
    else:
        raise TypeError("Not a DoRA-like module.")

@torch.no_grad()
def merge_and_save_current(model_dora: nn.Module,
                           tokenizer,
                           base_local_dir: Path,
                           export_dir: Path):
    """
    將當前（DoRA/DoRA+LoRA）模型合成實際 Linear 權重，輸出乾淨 HF 模型到 export_dir。
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # 1) 載入無 DoRA 結構的 base 模型
    base = AutoModel.from_pretrained(base_local_dir, trust_remote_code=True)
    base_sd = base.state_dict()
    dora_sd = model_dora.state_dict()

    # 2) 建新 state_dict：DoRA 層改寫為合併權重，其餘沿用 dora_sd
    new_sd = {}
    for k, v in base_sd.items():
        if k.endswith(".weight") or k.endswith(".bias"):
            mod_name, param_name = k.rsplit(".", 1)
            try:
                dora_mod = _get_module_by_name(model_dora, mod_name)
            except AttributeError:
                dora_mod = None
            if isinstance(dora_mod, DoRALinearWithLoRA):
                W, b = _effective_weight_and_bias(dora_mod)
                if param_name == "weight":
                    new_sd[k] = W.detach().to(dtype=v.dtype, device="cpu")
                    continue
                elif param_name == "bias":
                    if b is not None:
                        new_sd[k] = b.detach().to(dtype=v.dtype, device="cpu")
                        continue
        if k in dora_sd:
            new_sd[k] = dora_sd[k].detach().to(dtype=v.dtype, device="cpu")
        else:
            new_sd[k] = v
    base.load_state_dict(new_sd, strict=True)

    tokenizer.save_pretrained(export_dir)
    base.save_pretrained(export_dir)
    for py in base_local_dir.glob("*.py"):
        shutil.copy2(py, export_dir / py.name)
    readme = base_local_dir / "README.md"
    if readme.exists():
        shutil.copy2(readme, export_dir / "README.md")
    print(f"[merge] exported merged weights -> {export_dir.resolve()}")


# =========================
# Encode & Eval
# =========================
@torch.no_grad()
def encode_texts(model, tokenizer, texts, is_query, task_prompt, max_len, device):
    if is_query:
        #inputs = [t for t in texts]
        inputs = [f"Instruct: {task_prompt}\nQuery: {t}" for t in texts]
    else:
        inputs = texts
    batch = tokenizer(inputs, max_length=max_len, padding=True, truncation=True, return_tensors="pt").to(device)
    outputs = model(**batch).last_hidden_state
    emb = F.normalize(last_token_pool(outputs, batch["attention_mask"]), p=2, dim=1)
    return emb

@torch.no_grad()
def evaluate_metrics(model, tokenizer, rows: List[Entry], task_prompt, max_len, device,
                     batch_size=64, show_progress=False, desc="Eval", margin=0.2):
    """
    回傳三個評估指標平均：pairwise acc, in-batch R@1, margin acc
    """
    model.eval()
    acc_sum = r1_sum = macc_sum = 0.0
    cnt = 0
    it = range(0, len(rows), batch_size)
    if show_progress:
        total = (len(rows) + batch_size - 1) // batch_size
        it = tqdm(it, total=total, desc=desc, dynamic_ncols=True)

    for i in it:
        chunk = rows[i:i+batch_size]
        qs  = [r["query"] for r in chunk]
        pos = [r["positive_doc"] for r in chunk]
        neg = [r["hard_negative_doc"] for r in chunk]
        doc_ids = get_doc_ids_from_batch(chunk, prefer_field="chunk_id").to(device)

        qe = encode_texts(model, tokenizer, qs,  True, task_prompt, max_len, device)
        pe = encode_texts(model, tokenizer, pos, False, task_prompt, max_len, device)
        ne = encode_texts(model, tokenizer, neg, False, task_prompt, max_len, device)

        bsz = len(chunk)
        acc_sum  += pairwise_accuracy_batch(qe, pe, ne) * bsz
        r1_sum   += inbatch_recall_at1(qe, pe, doc_ids) * bsz
        macc_sum += pairwise_margin_acc(qe, pe, ne, margin=margin) * bsz
        cnt += bsz

    cnt = max(1, cnt)
    return {
        "pair_acc": acc_sum / cnt,
        "r1": r1_sum / cnt,
        "margin_acc": macc_sum / cnt
    }

# ==== 新增：同時計算 validation loss + metrics ====
@torch.no_grad()
def evaluate_loss_and_metrics(model, tokenizer, rows: List[Entry], task_prompt, max_len, device,
                              batch_size=64, show_progress=False, desc="Eval",
                              temperature=0.05, margin=0.2, lambda_hn=0.5):
    """
    回傳：平均 loss（total/infonce/hardneg）以及三個評估指標（pair_acc, r1, margin_acc）
    """
    model.eval()
    acc_sum = r1_sum = macc_sum = 0.0
    loss_total_sum = loss_infonce_sum = loss_hn_sum = 0.0
    cnt = 0
    it = range(0, len(rows), batch_size)
    if show_progress:
        total = (len(rows) + batch_size - 1) // batch_size
        it = tqdm(it, total=total, desc=desc, dynamic_ncols=True)

    for i in it:
        chunk = rows[i:i+batch_size]
        qs  = [r["query"] for r in chunk]
        pos = [r["positive_doc"] for r in chunk]
        neg = [r["hard_negative_doc"] for r in chunk]
        doc_ids = get_doc_ids_from_batch(chunk, prefer_field="chunk_id").to(device)

        qe = encode_texts(model, tokenizer, qs,  True,  task_prompt, max_len, device)
        pe = encode_texts(model, tokenizer, pos, False, task_prompt, max_len, device)
        ne = encode_texts(model, tokenizer, neg, False, task_prompt, max_len, device)

        # loss
        loss_infonce, _logits = masked_infonce(qe, pe, doc_ids, temperature=temperature)
        loss_hn = hard_negative_margin(qe, pe, ne, margin=margin)
        loss_total = loss_infonce + lambda_hn * loss_hn

        bsz = len(chunk)
        acc_sum  += pairwise_accuracy_batch(qe, pe, ne) * bsz
        r1_sum   += inbatch_recall_at1(qe, pe, doc_ids) * bsz
        macc_sum += pairwise_margin_acc(qe, pe, ne, margin=margin) * bsz

        loss_total_sum   += float(loss_total.item())   * bsz
        loss_infonce_sum += float(loss_infonce.item()) * bsz
        loss_hn_sum      += float(loss_hn.item())      * bsz
        cnt += bsz

    cnt = max(1, cnt)
    return {
        "pair_acc":    acc_sum / cnt,
        "r1":          r1_sum / cnt,
        "margin_acc":  macc_sum / cnt,
        "val_loss_total":   loss_total_sum   / cnt,
        "val_loss_infonce": loss_infonce_sum / cnt,
        "val_loss_hardneg": loss_hn_sum      / cnt,
    }


# =========================
# Training
# =========================
def build_batches_index(N: int, batch_size: int, seed: int):
    idxs = list(range(N))
    random.Random(seed).shuffle(idxs)
    for i in range(0, N, batch_size):
        yield idxs[i:i+batch_size]

def train_loop(model, tokenizer, train_rows, dev_rows, test_rows, args, device, base_local_dir: Path):
    """
    1) 使用 Validation Loss (val_loss_total, 越小越好) 選 best（支援 min_delta / patience）
    2) **每次 optimizer step（accumulate 完成）都跑一次 Dev 評估**，並將 val_* 寫入 CSV，
       因此圖表會顯示逐 step 的 Validation Loss（以 step 做 x 軸）。
    """
    # 只開 DoRA/LoRA 參數
    for p in model.parameters():
        p.requires_grad_(False)
    train_tensors = []
    for m in model.modules():
        '''if isinstance(m, DoRALinear):
            m.m.requires_grad_(True); train_tensors.append(m.m)'''
        if isinstance(m, DoRALinearWithLoRA):
            m.m.requires_grad_(True)
            train_tensors.append(m.m)
            if m.lora_A is not None:
                m.lora_A.requires_grad_(True)
                m.lora_B.requires_grad_(True)
                train_tensors += [m.lora_A, m.lora_B]

    '''optim = torch.optim.AdamW([
        {"params": [p for p in train_tensors if p.ndim == 1], "lr": args.lr_m},    # m
        {"params": [p for p in train_tensors if p.ndim == 2], "lr": args.lr_lora}, # LoRA A/B
    ], weight_decay=args.weight_decay)'''
    optim = torch.optim.AdamW([
        # DoRA 的 m：1D 參數，無 weight decay
        {"params": [p for p in train_tensors if p.ndim == 1], "lr": args.lr_m, "weight_decay": 0.0},

        # LoRA 的 A/B：2D 參數，保留你原本的 decay（例如 0.01）
        {"params": [p for p in train_tensors if p.ndim == 2], "lr": args.lr_lora, "weight_decay": args.weight_decay},
    ])

    # AMP：fp16 用 GradScaler；bf16 不用
    dtype_flag = getattr(args, "dtype", "")
    use_fp16_amp = (device.type == "cuda") and (dtype_flag.lower() in ["fp16", "float16", "half"])
    autocast_dtype = (torch.float16 if use_fp16_amp
                      else (torch.bfloat16 if device.type == "cuda" else None))
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_amp)

    def amp_ctx():
        return (torch.amp.autocast("cuda", dtype=autocast_dtype)
                if device.type == "cuda" else nullcontext())

    # 建立訓練日誌 CSV
    logs_dir = Path(args.out_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    train_csv = logs_dir / "train_log.csv"
    with train_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step", "epoch",
                "loss_total", "loss_infonce", "loss_hardneg",
                "batch_acc", "batch_r1", "batch_acc_margin",
                "val_loss_total", "val_loss_infonce", "val_loss_hardneg"
            ]
        )
        writer.writeheader()

    global_step = 0          # optimizer step（每次 accumulate 完成 +1）
    micro_step = 0           # micro-batch 次數
    accum_steps = max(1, int(args.grad_accum))

    # 以 val loss 選 best
    best_val_loss = float("inf")
    no_improve = 0
    min_delta = float(getattr(args, "min_delta", 1e-4))
    patience  = int(getattr(args, "patience", 0))  # 0=不 early stop；>0 時以「連續 validation 次數」計

    for epoch in range(1, args.epochs + 1):
        model.train()
        n_batches = (len(train_rows) + args.batch_size - 1) // args.batch_size
        pbar = tqdm(build_batches_index(len(train_rows), args.batch_size, seed=args.seed + epoch),
                    total=n_batches, desc=f"Epoch {epoch}/{args.epochs} (ga={accum_steps})", dynamic_ncols=True)

        optim.zero_grad(set_to_none=True)

        for batch_ids in pbar:
            micro_step += 1
            rows_batch = [train_rows[i] for i in batch_ids]
            qs  = [r["query"] for r in rows_batch]
            pos = [r["positive_doc"] for r in rows_batch]
            neg = [r["hard_negative_doc"] for r in rows_batch]

            max_len_q = args.max_len_q or args.max_len
            max_len_d = args.max_len_d or args.max_len

            q_inputs = tokenizer([q for q in qs],
                                 max_length=max_len_q, padding=True, truncation=True, return_tensors="pt").to(device)            
            p_inputs = tokenizer(pos, max_length=max_len_d, padding=True, truncation=True, return_tensors="pt").to(device)
            n_inputs = tokenizer(neg, max_length=max_len_d, padding=True, truncation=True, return_tensors="pt").to(device)

            with amp_ctx():
                q_out = model(**q_inputs).last_hidden_state
                p_out = model(**p_inputs).last_hidden_state
                n_out = model(**n_inputs).last_hidden_state

                q = F.normalize(last_token_pool(q_out, q_inputs["attention_mask"]), p=2, dim=1)
                p = F.normalize(last_token_pool(p_out, p_inputs["attention_mask"]), p=2, dim=1)
                n = F.normalize(last_token_pool(n_out, n_inputs["attention_mask"]), p=2, dim=1)

                doc_ids = get_doc_ids_from_batch(rows_batch, prefer_field=args.group_field).to(device)
                loss, parts = retrieval_loss_masked(
                    q, p, n, doc_ids,
                    temperature=args.temperature, margin=args.margin, lambda_hn=args.lambda_hn
                )
                loss_to_backprop = loss / accum_steps  # 梯度累積

                # 即時指標（per micro-batch）
                batch_acc = pairwise_accuracy_batch(q.detach(), p.detach(), n.detach())
                batch_r1 = inbatch_recall_at1(q.detach(), p.detach(), doc_ids)
                batch_acc_margin = pairwise_margin_acc(q.detach(), p.detach(), n.detach(), margin=args.margin)

            # 反向與 step
            if use_fp16_amp:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            did_step = False
            if (micro_step % accum_steps) == 0:
                if use_fp16_amp:
                    if args.max_grad_norm and args.max_grad_norm > 0:
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(train_tensors, args.max_grad_norm)
                    scaler.step(optim); scaler.update()
                    optim.zero_grad(set_to_none=True)
                else:
                    if args.max_grad_norm and args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(train_tensors, args.max_grad_norm)
                    optim.step(); optim.zero_grad(set_to_none=True)
                global_step += 1
                did_step = True

            # tqdm 顯示
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                infonce=f"{parts['infonce']:.4f}",
                hn=f"{parts['hard_neg']:.4f}",
                acc=f"{batch_acc:.3f}",
                r1=f"{batch_r1:.3f}",
                macc=f"{batch_acc_margin:.3f}",
            )

            # 記錄 micro-batch 訓練資料到 CSV（val 欄位空著）
            

            # 每次 accumulate 完成 -> 做一次 Dev 評估 + 寫 val_* 到 CSV + 更新 BEST
            if did_step:
                dev_metrics = evaluate_loss_and_metrics(
                    model, tokenizer, dev_rows, args.task_prompt,
                    args.max_len_d or args.max_len, device,
                    show_progress=False, desc=f"Dev @ step {global_step}",
                    temperature=args.temperature, margin=args.margin, lambda_hn=args.lambda_hn
                )

                # 把本次 validation 寫入 CSV（此筆只有 val_*）
                with train_csv.open("a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "step", "epoch",
                            "loss_total", "loss_infonce", "loss_hardneg",
                            "batch_acc", "batch_r1", "batch_acc_margin",
                            "val_loss_total", "val_loss_infonce", "val_loss_hardneg"
                        ]
                    )
                    writer.writerow({
                        "step": global_step,  # 注意：optimizer step
                        "epoch": epoch,
                        "loss_total": float(loss.item()),
                        "loss_infonce": float(parts["infonce"]),
                        "loss_hardneg": float(parts["hard_neg"]),
                        "batch_acc": float(batch_acc),
                        "batch_r1": float(batch_r1),
                        "batch_acc_margin": float(batch_acc_margin),
                        "val_loss_total": float(dev_metrics["val_loss_total"]),
                        "val_loss_infonce": float(dev_metrics["val_loss_infonce"]),
                        "val_loss_hardneg": float(dev_metrics["val_loss_hardneg"]),
                    })

                # 更新 BEST（用 val_loss_total）
                curr_loss = float(dev_metrics["val_loss_total"])
                improved = (best_val_loss - curr_loss) > min_delta or (best_val_loss == float("inf"))
                if improved:
                    best_val_loss = curr_loss
                    no_improve = 0
                    best_dir = Path(args.out_dir) / "best"
                    save_model_bundle(model, tokenizer, base_local_dir, best_dir)
                    print(f"[best@step {global_step}] updated BEST (val_loss_total={best_val_loss:.6f}) -> {best_dir}")
                else:
                    no_improve += 1
                    if patience > 0 and no_improve >= patience:
                        print(f"[EarlyStop] no improvement for {patience} validations. "
                              f"Best val_loss_total={best_val_loss:.6f}.")
                        # 仍在離開前畫圖
                        fig_path = Path(args.out_dir) / "logs" / "training_curves.png"
                        
                        plot_curves_from_csv(train_csv, fig_path)
                        
                        return best_val_loss, train_csv, fig_path
                    
            else:
                with train_csv.open("a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "step", "epoch",
                            "loss_total", "loss_infonce", "loss_hardneg",
                            "batch_acc", "batch_r1", "batch_acc_margin",
                            "val_loss_total", "val_loss_infonce", "val_loss_hardneg"
                        ]
                    )
                    writer.writerow({
                        "step": global_step,  # 注意：optimizer step
                        "epoch": epoch,
                        "loss_total": float(loss.item()),
                        "loss_infonce": float(parts["infonce"]),
                        "loss_hardneg": float(parts["hard_neg"]),
                        "batch_acc": float(batch_acc),
                        "batch_r1": float(batch_r1),
                        "batch_acc_margin": float(batch_acc_margin),
                        "val_loss_total": "",
                        "val_loss_infonce": "",
                        "val_loss_hardneg": "",
                    })
            
        # 每個 epoch 結束仍保留原先 checkpoint/merge（可保留歷史）
        epoch_dir = Path(args.out_dir) / f"epoch_{epoch:02d}"
        save_model_bundle(model, tokenizer, base_local_dir, epoch_dir)
        print(f"[checkpoint] saved epoch model -> {epoch_dir}")

        #if args.merge_each_epoch:
        merged_epoch_dir = Path(args.out_dir) / f"epoch_{epoch:02d}_merged"
        merge_and_save_current(model, tokenizer, base_local_dir, merged_epoch_dir)
        
        test_metrics = evaluate_metrics(model, tokenizer, test_rows, args.task_prompt,
                                        args.max_len_d or args.max_len, device,
                                        show_progress=True, desc="Test", margin=args.margin)
        print(f"[TEST] metrics: "
              f"pair_acc={test_metrics['pair_acc']:.4f} "
              f"r1={test_metrics['r1']:.4f} "
              f"margin_acc={test_metrics['margin_acc']:.4f}")
        
        test_result_file = Path(args.out_dir) / "test_metrics.txt"
        with open(test_result_file, "a", encoding="utf-8") as f:
            f.write(f"===== TEST METRICS (epoch_{epoch:02d}) =====\n")
            f.write(f"pairwise_accuracy : {test_metrics['pair_acc']:.6f}\n")
            f.write(f"inbatch_r1        : {test_metrics['r1']:.6f}\n")
            f.write(f"margin_accuracy   : {test_metrics['margin_acc']:.6f}\n")
            f.write(f"best_val_loss     : {best_val_loss:.6f}\n\n")

        print(f"[TEST] metrics saved -> {test_result_file.resolve()}")

    # 畫圖（含逐 step 的 Validation Loss 折線）
    fig_path = Path(args.out_dir) / "logs" / "training_curves.png"    
    plot_curves_from_csv(train_csv, fig_path)
    
    return best_val_loss, train_csv, fig_path



# =========================
# Main
# =========================
def _str2dtype(s: str):
    if not s: return None
    s = s.lower()
    if s in ("bf16", "bfloat16"): return torch.bfloat16
    if s in ("fp16", "float16", "half"): return torch.float16
    if s in ("fp32", "float32"): return torch.float32
    raise ValueError(f"Unknown dtype: {s}")



def parse_config_file(config_path: str):
    args_dict = {}
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key, val = key.strip(), val.strip()

            # 嘗試轉型
            if val.lower() in ["true", "false"]:
                val = val.lower() == "true"
            elif val in ["None", "null"]:
                val = None
            else:
                try:
                    val = ast.literal_eval(val)
                except Exception:
                    # 如果不能 eval，就保留原始字串
                    pass
            args_dict[key] = val
    return argparse.Namespace(**args_dict)


def main():  
    
    config_file = "config.txt"   # 你要讀的檔案
    args = parse_config_file(config_file)   
     
    lines = [f"{k}: {v}" for k, v in vars(args).items()]
    msg = "\n".join(lines)    
   
    #args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # 將 config 存成文字檔
    
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.txt"), "w", encoding="utf-8") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    local_dir = Path(args.local_dir)
    out_dir = Path(args.out_dir)
    assert local_dir.exists(), f"{local_dir} not found"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 1) 先在 CPU 載入
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(local_dir, trust_remote_code=True)

    # 2) 印原始結構
    print("\n========== Original Model Structure (Before DoRA) ==========\n")
    print(model)

    # 3) DoRA / DoRA+LoRA 替換（仍在 CPU）
    TARGETS = ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"] #["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"]
    
    cnt, names = replace_linears_with_dora_lora(model, name_filters=TARGETS,
                                                r=args.rank, alpha=args.alpha, dropout_p=args.lora_dropout)
    #mode = "DoRA+LoRA"
    
    '''else:
        cnt, names = replace_linears_with_dora(model, name_filters=TARGETS)
        mode = "DoRA"'''
        
    print(f"\n[DoRA Init] 已替換 Linear 層數量：{cnt}")
    print("被替換的層：", names, " ...")

    # 4) 替換後再搬到 GPU（指定 dtype）
    target_dtype = _str2dtype(args.dtype)
    move_kwargs = {}
    if target_dtype is not None:
        move_kwargs["dtype"] = target_dtype
    if device.type == "cuda":
        move_kwargs["non_blocking"] = True
    model = model.to(device, **move_kwargs)

    # 5) 訓練建議：關閉 use_cache（避免內部 KV cache 佔記憶體）
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass

    # 6) 印替換後結構 + smoke test
    print(f"\n========== DoRA Model Structure (After Replace) ==========\n")
    print(model)
    with torch.no_grad():
        dummy = tokenizer(["hello world"], return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        _ = model(**dummy)
    print("\n[Smoke Test] Forward OK with replaced modules.")

    # 7) 訓練（可選）
    if args.train:
        rows = read_json_or_jsonl(Path(args.data_path))
        basic_validate(rows, group_field=args.group_field)
        train_rows, dev_rows, test_rows = split_grouped(
            rows, group_field=args.group_field,
            test_ratio=args.test_ratio, dev_ratio=args.dev_ratio, seed=args.seed
        )
        print(f"[Split] train={len(train_rows)} | dev={len(dev_rows)} | test={len(test_rows)} (group_field={args.group_field})")

        best_dev, train_csv, fig_path = train_loop(model, tokenizer, train_rows, dev_rows, test_rows, args, device, local_dir)
        print(f"[Training] best val_loss_total = {best_dev:.4f}")
        
        print(f"[Logs] per-batch metrics -> {train_csv}")        
        print(f"[Figure] training curves -> {fig_path}")
    

    print("\n[Done] complete.")  

if __name__ == "__main__":
    main()
