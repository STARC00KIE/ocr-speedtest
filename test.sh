#!/bin/bash
# Conda 쉘 함수를 현재 스크립트 환경에 로드
source /home/vtw/workspace/yes/etc/profile.d/conda.sh

# olmocr-test 경로 이동
# cd olm

# olmocr-test 테스트 수행
# conda activate olmocr-test
# python --version
# echo "olmocr 테스트 시작"
# python olmocr_test.py
# echo "olmocr 테스트 종료"

# deepseekocr 경로 이동
# cd ..
# cd deepseekocr

# 딥시크 ocr 테스트 수행
# conda activate deepseek-ocr
# python --version
# echo "deepseek-ocr 테스트 시작"
# python deepseekocr_test.py
# echo "deepseek-ocr 테스트 종료"

# chandraocr 경로 이동
# cd ..
cd chandraocr

# 찬드라 ocr 테스트 수행
conda activate chandraocr
python --version
echo "chandraocr 테스트 시작"
python chandraocr_test.py
echo "chandraocr 테스트 종료"