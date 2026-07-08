# 제안 A — 전천 카메라 구름 자동 분류 + 초단기 날씨 변화 예측

ImageNet 사전학습 **MobileNetV2** 를 공개 구름 데이터셋 **CCSN(Cirrus Cumulus Stratus Nimbus)** 으로
**실제로 파인튜닝**한 CNN 으로 하늘 사진의 구름 유형을 11종 분류하고, 그 결과로 초단기 날씨를 예측하는
Jupyter 노트북. 가짜 휴리스틱 분류기가 아니라 **진짜 CNN 추론**이 돈다.

## 구조
```
cloud-classifier-demo/
├── README.md
├── requirements.txt
├── cloud_classifier_demo.ipynb   # 메인 노트북 (실제 CNN 추론 시연)
├── train_cloud_cnn.py             # 실제 학습: CCSN + MobileNetV2 파인튜닝 → models/ 저장
├── classify_captures.py            # 실제 추론: data/captures/*.jpg → capture_log.csv 재생성
├── capture_sim.py                  # [컨셉] 라즈베리파이 캡처 데몬 목업
├── data/
│   ├── ccsn/                       # CCSN 데이터셋 — train_cloud_cnn.py 가 자동 다운로드/해제 (커밋 안 됨)
│   ├── captures/                  # 분석 대상 하늘 사진 23장 (커밋됨)
│   └── capture_log.csv            # classify_captures.py 가 실제 추론으로 (재)생성 (커밋 안 됨)
├── assets/                        # 개념도 자산(도메인 분포 등)
└── models/                        # 학습된 가중치 + training_history.json (커밋 안 됨 — Colab이 학습해 생성)
    ├── cloud_mobilenetv2_baseline.pt   # 도메인 적응 전 (백본 freeze + 선형 헤드)
    └── cloud_mobilenetv2_finetuned.pt  # 도메인 적응 후 (full fine-tune)
```
> 모델 가중치(`models/*.pt`, `training_history.json`)는 **커밋하지 않는다** — Colab에서
> `train_cloud_cnn.py` 가 직접 학습해 생성. CCSN 원본(`data/ccsn/`)도 자동 다운로드라 미포함.

## 실행 (Colab 권장 — GPU)
노트북을 Colab에서 **Run All** 한 번이면 끝:
1. **첫 셀(부트스트랩)**: 레포 자동 클론 + cwd 이동 (Colab의 GitHub 로더로 열었을 때)
2. **env → 자동학습 셀**: 모델이 없으므로 `train_cloud_cnn.py` 를 **Colab GPU로 자동 실행**
   (CCSN 자동 다운로드 + MobileNetV2 파인튜닝, 수 분~10분) → `models/` 생성
3. 이후 셀들이 커밋된 23장 사진을 **실제 CNN으로 분류** (그리드 · 라이브 분석 · 학습곡선 · 전이행렬)

다시 학습하려면 `models/` 를 지우고 Run All (또는 터미널에서
`python train_cloud_cnn.py --epochs 15 --batch 64`).

### 로컬에서 실행
```bash
pip install -r requirements.txt
# GPU(torch): python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python classify_captures.py          # 커밋된 모델로 data/captures/*.jpg 실제 추론 → capture_log.csv
jupyter notebook cloud_classifier_demo.ipynb   # Restart & Run All
```

## 발표 시나리오 (핵심)
1. 노트북을 위에서 아래로 실행하며 제안 A 서사 설명 (동기 → 도메인 적응 → 파인튜닝)
2. **"데이터셋 한눈에 보기"** 셀 → 캡처 사진 전체를 실제 CNN 이 분류한 그리드
3. **"파인튜닝 결과"** 셀 → 실제 학습 곡선/검증 정확도(baseline vs fine-tuned)
4. **"★ 라이브 분석"** 셀에서 **직접 찍은 하늘 사진 한 장**으로 교체 후 실행
   - 사진을 `data/captures/` 에 저장
   - `LIVE_IMAGE = "data/captures/파일명.jpg"` 경로만 수정
   - 전처리 → 실제 CNN 순전파 → 중간층 활성화맵(forward hook) → 11종 softmax → 결과 카드 → 날씨 예측
5. 밝은 사진 ↔ 어두운 사진을 바꿔 가며 실제 추론 결과가 달라지는 것을 확인

## 진짜 / 가짜 구분 (현재)
- **진짜로 동작**: `train_cloud_cnn.py`(학습), `classify_captures.py`+노트북 라이브 셀(추론).
  CNN = ImageNet 사전학습 MobileNetV2 + CCSN 파인튜닝. 특징맵·확률·학습곡선·전이행렬 모두 실제 연산 결과.
- **개념도 자산(보조)**: `assets/ccsn_vs_school_dist.png` 등은 도메인 격차 개념도.
  노트북 3절은 CCSN/캡처 실측 분포로 대체해 그린다.

## 사진 추가 / 교체
발표용 하늘 사진을 `data/captures/` 에 아무 파일명으로 넣으면,
- "데이터셋 한눈에 보기" 그리드에 자동 포함(실시간 분류)
- "라이브 분석" 셀에서 `LIVE_IMAGE` 경로로 지정해 분석
파일명은 `YYYYMMDD_HHMM.jpg`, epoch ms(`1783...jpg`), 카카오톡 파일명 모두 타임스탬프로 복원된다.

## 데이터 / 모델 출처
- CCSN: Zhang et al. (2018), *CloudNet*, GRL. https://doi.org/10.1029/2018GL077787
  데이터: https://doi.org/10.7910/DVN/CADDPD (공식) / https://github.com/Callmewuxin/CCSN_dataset (미러)
- 백본 사전학습: torchvision MobileNetV2 IMAGENET1K_V2 가중치
- 11클래스: Cirrus, Cirrocumulus, Cirrostratus, Altocumulus, Altostratus, Nimbostratus,
  Stratocumulus, Stratus, Cumulus, Cumulonimbus, Contrail(비행운) — 실제 CCSN taxonomy