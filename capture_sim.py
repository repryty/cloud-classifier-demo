# -*- coding: utf-8 -*-
"""
capture_sim.py
==============
[컨셉] 라즈베리파이(옥상)에서 5분 간격으로 전천(하늘) 이미지를 촬영하던
실장 스크립트의 축약본. 실제 장비 없이 로그만 남기는 시뮬레이션 모드.

실제 운영 시:
  - 라즈베리파이 카메라 모듈(piwbo libcamera-still) 호출
  - 캡처 이미지를 data/captures/YYYYMMDD_HHMM.jpg 로 저장
  - 메타데이터를 data/capture_log.csv 에 append

여기서는 장비 없이 "이런 스크립트가 매 5분 cron 으로 돌고 있었다"는 흔적을
보여주기 위한 목업이다.
"""
import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
CAP_DIR = ROOT / "data" / "captures"
LOG_CSV = ROOT / "data" / "capture_log.csv"
CAP_DIR.mkdir(parents=True, exist_ok=True)

# 5분 간격 자동 촬영
INTERVAL_MIN = 5
# 낮 시간만 촬영 (야간은 의미 없음)
DAY_START, DAY_END = 6, 20


def capture_once(simulate=True):
    """한 장 캡처. simulate=True 면 실제 카메라 호출 없이 로그만."""
    now = datetime.now()
    if not (DAY_START <= now.hour <= DAY_END):
        return None, "skip: 야간(촬영 안 함)"

    fname = now.strftime("%Y%m%d_%H%M") + ".jpg"
    path = CAP_DIR / fname

    if not simulate:
        # === 실제 라즈베리파이 환경에서만 실행되는 부분 ===
        # import subprocess
        # subprocess.run(["libcamera-still", "-o", str(path),
        #                  "--width", "1640", "--height", "922",
        #                  "--timeout", "2000"], check=True)
        # 밝기/라벨은 분류 서버에 추론 요청 후 기록
        pass
    else:
        # 목업: 가짜 메타데이터
        path.write_bytes(b"")  # 자리표시

    return fname, "ok"


def append_log(fname, status):
    write_header = not LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "file", "status"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), fname, status])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="한 번만 캡처")
    ap.add_argument("--interval", type=int, default=INTERVAL_MIN,
                    help="촬영 간격(분)")
    args = ap.parse_args()

    print(f"[capture_sim] 전천 카메라 자동 촬영 데몬 (간격 {args.interval}분)")
    while True:
        fname, status = capture_once(simulate=True)
        if fname:
            append_log(fname, status)
            print(f"  {datetime.now():%Y-%m-%d %H:%M} -> {fname} [{status}]")
        if args.once:
            break
        # 데몬 모드: interval 분 대기. (목업이라 실제론 1회만)
        print(f"  ...다음 촬영까지 {args.interval}분 대기 (시뮬레이션 종료)")
        break


if __name__ == "__main__":
    main()