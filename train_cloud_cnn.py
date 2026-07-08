# -*- coding: utf-8 -*-
"""
train_cloud_cnn.py
==================
CCSN(Cirrus Cumulus Stratus Nimbus) 공개 구름 데이터셋으로
ImageNet 사전학습 MobileNetV2 를 **실제로** 파인튜닝하여 구름 11종 분류 CNN을 만든다.

두 모델을 학습/저장한다:
  - baseline  : ImageNet 가중치 그대로(백본 freeze) + 선형 분류 헤드만 학습  (= "도메인 적응 전")
  - finetuned : 백본 상위층 unfreeze + 헤드 함께 파인튜닝              (= "도메인 적응 후")
두 모델의 실제 train/val loss·acc 곡선과 최종 검증 정확도/Top-3 를
models/training_history.json 에 저장 → 노트북 4절(파인튜닝 결과)에 실제 수치로 표시.

사용:
  python train_cloud_cnn.py            # 데이터 없으면 data/ccsn.zip 자동 해제
  python train_cloud_cnn.py --epochs 8 --batch 32

데이터: https://github.com/Callmewuxin/CCSN_dataset  (CCSN 11클래스, 256x256, 2,543장)
백본 사전학습 가중치: torchvision MobileNetV2 IMAGENET1K_V2
"""
import argparse, json, random, shutil, zipfile
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from torchvision import transforms

# ---------------- 경로 / 상수 ----------------
ROOT = Path(__file__).resolve().parent
CCSN_DIR = ROOT / "data" / "ccsn"
CCSN_ZIP = ROOT / "data" / "ccsn.zip"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# CCSN 폴더약자 -> 표준 이름 (WMO genera). Ct=Contrail(비행운)
CCSN_FOLDERS = {
    "Ci": "Cirrus", "Cc": "Cirrocumulus", "Cs": "Cirrostratus",
    "Ac": "Altocumulus", "As": "Altostratus", "Ns": "Nimbostratus",
    "Sc": "Stratocumulus", "St": "Stratus", "Cu": "Cumulus",
    "Cb": "Cumulonimbus", "Ct": "Contrail",
}
# 노트북과 공유할 최종 11종 라벨 순서 (실제 CCSN 기반; 가짜 Clear 대신 Contrail)
LABELS = ["Cirrus", "Cirrocumulus", "Cirrostratus", "Altocumulus", "Altostratus",
          "Nimbostratus", "Stratocumulus", "Stratus", "Cumulus", "Cumulonimbus",
          "Contrail"]
LABEL_KO = {
    "Cirrus": "권운", "Cirrocumulus": "권적운", "Cirrostratus": "권층운",
    "Altocumulus": "고적운", "Altostratus": "고층운", "Nimbostratus": "난층운",
    "Stratocumulus": "층적운", "Stratus": "층운", "Cumulus": "적운",
    "Cumulonimbus": "적란운", "Contrail": "비행운",
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


CCSN_URL = "https://github.com/Callmewuxin/CCSN_dataset/archive/refs/heads/master.zip"


def _download(url, dest):
    import urllib.request
    print(f"[데이터] 다운로드 {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"[데이터] 다운로드 완료 ({dest.stat().st_size/1e6:.1f} MB)")


def prepare_data():
    """data/ccsn/<Class>/*.jpg 구조 확보. zip 없으면 자동 다운로드(Colab 지원)."""
    existing = [d for d in CCSN_DIR.glob("*") if d.is_dir()]
    if existing:
        print(f"[데이터] {CCSN_DIR} 에 {len(existing)}개 폴더 존재 → 재사용")
        return
    if not CCSN_ZIP.exists():
        (ROOT / "data").mkdir(parents=True, exist_ok=True)
        _download(CCSN_URL, CCSN_ZIP)
    print(f"[데이터] {CCSN_ZIP} 해제 중...")
    with zipfile.ZipFile(CCSN_ZIP) as z:
        z.extractall(ROOT / "data")
    cand = list((ROOT / "data").glob("CCSN_dataset-*"))
    if cand and not existing:
        src = cand[0]
        CCSN_DIR.mkdir(parents=True, exist_ok=True)
        for sub in src.iterdir():
            dest = CCSN_DIR / sub.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(sub), str(dest))
        shutil.rmtree(src, ignore_errors=True)
    print(f"[데이터] 해제 완료: {CCSN_DIR}")


def collect_samples():
    """(path, class_name) 리스트. CCSN 하위에 약자 폴더(Ci, Cu..) 또는 풀네임 폴더 모두 허용."""
    samples = []
    name_to_canon = {v: k for k, v in CCSN_FOLDERS.items()}  # 풀네임->canon
    for d in sorted(CCSN_DIR.iterdir()):
        if not d.is_dir():
            continue
        key = d.name
        if key in CCSN_FOLDERS:
            canon = CCSN_FOLDERS[key]
        elif key in name_to_canon:
            canon = name_to_canon[key]
        else:
            continue
        for p in d.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                samples.append((str(p), canon))
    return samples


class CloudDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
        self.label_to_idx = {c: i for i, c in enumerate(LABELS)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, cls = self.samples[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # 손상 이미지 회피: 회색 placeholder
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        x = self.transform(img)
        y = self.label_to_idx[cls]
        return x, y


def stratified_split(samples, val_ratio=0.2, seed=42):
    rng = random.Random(seed)
    by_cls = {}
    for s in samples:
        by_cls.setdefault(s[1], []).append(s)
    train, val = [], []
    for cls, items in by_cls.items():
        items = list(items)
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio)))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def make_transforms():
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.25, 0.25, 0.20, 0.04),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def build_mobilenetv2(num_classes, freeze_backbone, unfreeze_from=None):
    """ImageNet 사전학습 MobileNetV2. 분류 헤드는 11종으로 교체."""
    try:
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        weights = MobileNet_V2_Weights.IMAGENET1K_V2
        model = mobilenet_v2(weights=weights)
    except Exception:
        # 구버전 torchvision 호환
        model = torchvision.models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
        if unfreeze_from is not None:
            # features[unfreeze_from:] 만 학습 가능
            for m in model.features[unfreeze_from:]:
                for p in m.parameters():
                    p.requires_grad = True
    return model


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total, n = 0.0, 0
    correct = top3_correct = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            out = model(xb)
            loss = criterion(out, yb)
            if train:
                loss.backward()
                optimizer.step()
        bs = yb.size(0)
        total += loss.item() * bs
        n += bs
        pred = out.argmax(1)
        correct += (pred == yb).sum().item()
        top3 = out.topk(3, dim=1).indices
        top3_correct += top3.eq(yb.view(-1, 1)).any(dim=1).sum().item()
    return total / n, correct / n, top3_correct / n


def train_model(name, freeze_backbone, unfreeze_from, train_samples, val_samples,
                epochs, batch, device, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    train_tf, val_tf = make_transforms()
    train_ds = CloudDataset(train_samples, train_tf)
    val_ds = CloudDataset(val_samples, val_tf)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)

    model = build_mobilenetv2(len(LABELS), freeze_backbone, unfreeze_from).to(device)
    criterion = nn.CrossEntropyLoss()
    # 학습 가능 파라미터에만 다른 lr 적용: 백본은 낮게, 헤드는 높게
    head_params = list(model.classifier.parameters())
    body_params = [p for p in model.features.parameters() if p.requires_grad]
    param_groups = [{"params": head_params, "lr": 3e-4 if body_params else 1e-3}]
    if body_params:
        param_groups.append({"params": body_params, "lr": 3e-5})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
            "val_top3": []}
    best_acc = 0.0
    best_state = None
    for ep in range(epochs):
        tl, ta, t3 = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl, va, v3 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)
        hist["train_acc"].append(ta)
        hist["val_acc"].append(va)
        hist["val_top3"].append(v3)
        print(f"[{name}] epoch {ep+1}/{epochs}  "
              f"train_loss={tl:.4f} train_acc={ta:.3f} | "
              f"val_loss={vl:.4f} val_acc={va:.3f} top3={v3:.3f}")
        if va > best_acc:
            best_acc = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    # 최고 val_acc 시점 가중치로 저장
    save_path = MODELS / f"cloud_mobilenetv2_{name}.pt"
    torch.save({"state_dict": best_state, "labels": LABELS,
                "val_acc": best_acc, "label_ko": LABEL_KO}, save_path)
    print(f"[{name}] 저장: {save_path}  (best val_acc={best_acc:.3f})")
    hist["best_val_acc"] = best_acc
    hist["final_val_top3"] = hist["val_top3"][-1]
    hist["final_val_loss"] = hist["val_loss"][-1]
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", choices=["baseline", "finetuned", "both"], default="both")
    args = ap.parse_args()

    prepare_data()
    samples = collect_samples()
    print(f"[데이터] 전체 샘플: {len(samples)}")
    from collections import Counter
    dist = Counter(c for _, c in samples)
    for c in LABELS:
        print(f"   {c:14s} {dist.get(c,0)}")
    if len(samples) == 0:
        raise RuntimeError("CCSN 샘플이 0장. data/ccsn 폴더 구조를 확인하세요.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[환경] device = {device}  (torch {torch.__version__})")

    train_samples, val_samples = stratified_split(samples, 0.2, args.seed)
    print(f"[분할] train={len(train_samples)} val={len(val_samples)}")

    history = {"classes": LABELS, "label_ko": LABEL_KO, "n_train": len(train_samples),
               "n_val": len(val_samples), "epochs": args.epochs}

    if args.only in ("baseline", "both"):
        # baseline: 백본 완전 freeze + 선형 헤드만 (ImageNet 특징 그대로 = 도메인 적응 전)
        history["baseline"] = train_model(
            "baseline", freeze_backbone=True, unfreeze_from=None,
            train_samples=train_samples, val_samples=val_samples,
            epochs=args.epochs, batch=args.batch, device=device, seed=args.seed)
    if args.only in ("finetuned", "both"):
        # finetuned: 백본 전체 unfreeze(= full fine-tune) + 헤드 함께 학습 (도메인 적응 후)
        history["finetuned"] = train_model(
            "finetuned", freeze_backbone=True, unfreeze_from=0,
            train_samples=train_samples, val_samples=val_samples,
            epochs=args.epochs, batch=args.batch, device=device, seed=args.seed)

    out = MODELS / "training_history.json"
    out.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[완료] 히스토리 저장: {out}")
    print(f"  baseline val_acc = {history.get('baseline',{}).get('best_val_acc')}")
    print(f"  finetuned val_acc = {history.get('finetuned',{}).get('best_val_acc')}")


if __name__ == "__main__":
    main()