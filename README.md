# 제안 A — 전천 카메라 구름 자동 분류 + 초단기 날씨 변화 예측

ImageNet 사전학습 **MobileNetV2** 를 공개 구름 데이터셋 **CCSN(Cirrus Cumulus Stratus Nimbus)** 으로
**실제로 파인튜닝**한 CNN 으로 하늘 사진의 구름 유형을 11종 분류하고, 그 결과로 초단기 날씨를 예측하는
Jupyter 노트북. 가짜 휴리스틱 분류기가 아니라 **진짜 CNN 추론**이 돈다.

## 구조
```
yyh/
├── README.md
├── requirements.txt
├── cloud_classifier_demo.ipynb   # 메인 노트북 (실제 CNN 추론 시연)
├── train_cloud_cnn.py            # 실제 학습: CCSN + MobileNetV2 파인튜닝 → models/ 저장
├── classify_captures.py           # 실제 추론: data/captures/*.jpg → capture_log.csv 재생성
├── capture_sim.py                # [컨셉] 라즈베리파이 캡처 데몬 목업
├── generate_assets.py             # [구] 개념도 자산 생성(현재 노트북은 실데이터 사용)
├── build_notebook.py              # [구] 과거 노트북 생성기(현재 내용과 다름 — 실행 금지)
├── data/
│   ├── ccsn/                      # CCSN 데이터셋(11클래스) — train_cloud_cnn.py 가 자동 해제
│   ├── ccsn.zip                   # CCSN 다운로드본 (GitHub 미러)
│   ├── captures/                  # 분석 대상 하늘 사진 (여기에 사진을 넣으세요)
│   └── capture_log.csv            # classify_captures.py 가 실제 추론으로 (재)생성
├── assets/                        # 개념도 자산(도메인 분포 등; 일부는 노트북이 실측으로 대체)
└── models/                        # 학습된 가중치 + training_history.json (실제 학습 결과)
```

## 실행
```bash
# 1) 의존성 (GPU 권장)
pip install -r requirements.txt
#   torch/torchvision GPU 설치:
#   python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) CCSN 데이터셋 다운로드 (최초 1회)
curl -L -o data/ccsn.zip https://github.com/Callmewuxin/CCSN_dataset/archive/refs/heads/master.zip

# 3) 실제 학습 → models/cloud_mobilenetv2_finetuned.pt + training_history.json 생성
python train_cloud_cnn.py --epochs 8 --batch 64
#    (GPU 없으면 자동 CPU. CPU는 수십 분 소요)

# 4) 캡처 사진 전체 실제 분류 → data/capture_log.csv 갱신
python classify_captures.py

# 5) 노트북 실행
jupyter notebook cloud_classifier_demo.ipynb     # Restart & Run All
```
노트북 안에서도 2절 직전 셀이 `classify_captures.py` 를 다시 불러 캡처 전체를 실시간 분류한다.

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
  노트북 3절은 가능하면 CCSN/캡처 실측 분포로 대체해 그린다.

## 사진 넣는 법
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