# -*- coding: utf-8 -*-
"""
classify_captures.py
===================
학습된 구름 분류 CNN(models/cloud_mobilenetv2_finetuned.pt) 로 data/captures/*.jpg
전체를 **실제 추론**하여 data/capture_log.csv 를 (재)생성한다.

노트북의 "데이터셋 한눈에 보기(2절)" 그리드와 "시계열 분석(7절)" 전이행렬이
이 실제 예측 결과를 사용하도록 한다.

timestamp 규칙(파일명에서 복원, 실패 시 파일 mtime):
  - YYYYMMDD_HHMM.jpg          -> 그대로
  - 1783505798416.jpg (epoch ms)-> datetime.fromtimestamp(ms/1000)
  - KakaoTalk_20260708_18...    -> 안의 8자리 날짜+시간 파싱
사용:
  python classify_captures.py
"""
import json, re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
CAP_DIR = ROOT / "data" / "captures"
LOG_CSV = ROOT / "data" / "capture_log.csv"
MODEL_PATH = ROOT / "models" / "cloud_mobilenetv2_finetuned.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

WEATHER_MAP = {
    "Cirrus": "맑음 유지 (선풍 전선 의심)",
    "Cirrocumulus": "맑음/약흐림",
    "Cirrostratus": "맑음, 하늘활/황사 유의",
    "Altocumulus": "약흐림, 오후 소나기 유의",
    "Altostratus": "흐려짐",
    "Nimbostratus": "지속적 비(1~3h 내 강수 확률 높음)",
    "Stratocumulus": "흐림",
    "Stratus": "안개/흐림, 약한 이슬비 가능",
    "Cumulus": "맑음/소나기 유의(대류성)",
    "Cumulonimbus": "뇌우/소나기(단시간 강수)",
    "Contrail": "맑음(비행운, 상층 습윤 의심)",
}


def parse_timestamp(p: Path) -> str:
    name = p.stem
    # YYYYMMDD_HHMM
    m = re.search(r"(\d{8})_(\d{4})", name)
    if m:
        try:
            dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")
            return dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass
    # 13자리 epoch ms
    m = re.match(r"^(\d{13})$", name)
    if m:
        try:
            dt = datetime.fromtimestamp(int(m.group(1)) / 1000.0)
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
    # 안에 8자리 날짜 + 6자리 시간
    m = re.search(r"(\d{8})(\d{6})", name)
    if m:
        try:
            dt = datetime.strptime(m.group(1) + m.group(2)[:4], "%Y%m%d%H%M")
            return dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass
    # fallback: 파일 수정시각
    dt = datetime.fromtimestamp(p.stat().st_mtime)
    return dt.strftime("%Y-%m-%d %H:%M")


def extract_simple_features(img):
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)
    r, b = arr[..., 0].mean(), arr[..., 2].mean()
    small = gray[::4, ::4]
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    return {
        "brightness": round(float(gray.mean()), 4),
        "rb": round(float(r / (b + 1e-6)), 4),
        "edge": round(float((gx + gy) / 2), 5),
    }


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델이 없습니다: {MODEL_PATH}\n먼저 python train_cloud_cnn.py 로 학습하세요.")
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    labels = ckpt["labels"]
    label_ko = ckpt["label_ko"]
    num_classes = len(labels)

    model = torchvision.models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[classify] device={device}  모델 로드 완료  클래스={num_classes}")

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    files = sorted(CAP_DIR.glob("*.jpg"))
    if not files:
        raise RuntimeError(f"캡처 이미지가 없습니다: {CAP_DIR}")
    print(f"[classify] 분석 대상: {len(files)} 장")

    rows = []
    with torch.no_grad():
        for p in files:
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"  skip {p.name}: {e}")
                continue
            x = tf(img).unsqueeze(0).to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
            idx = int(prob.argmax())
            kind = labels[idx]
            feat = extract_simple_features(img)
            rows.append({
                "timestamp": parse_timestamp(p),
                "file": p.name,
                "brightness": feat["brightness"],
                "rb": feat["rb"],
                "edge": feat["edge"],
                "kind": kind,
                "kind_ko": label_ko[kind],
                "confidence": round(float(prob[idx]), 4),
                "weather": WEATHER_MAP[kind],
            })
            print(f"  {p.name:40s} -> {kind:14s} ({label_ko[kind]}) "
                  f"{prob[idx]*100:5.1f}%")

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df.to_csv(LOG_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[classify] 완료: {LOG_CSV}  ({len(df)} 행)")
    print(df[["timestamp", "file", "kind_ko", "confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()