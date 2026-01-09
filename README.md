# OCR Speedtest & Performance Analysis

이 프로젝트는 OlmOCR, DeepSeekOCR, ChandraOCR 세 가지 OCR 모델의 성능을 비교하고 추론 시간을 분석하기 위한 도구입니다. PPT 파일을 PDF로 변환하고, 각 모델을 통해 마크다운으로 변환된 결과와 상세한 추론 시간 로그를 생성합니다.

## 디렉터리 구조

- **root**
  - `ppt_to_pdf.py`: PPT/PPTX 파일을 PDF로 변환하는 스크립트
  - `test.sh`: 전체 테스트 실행을 위한 쉘 스크립트 (각 모델별 Conda 환경 활성화 및 테스트 실행)
  - `data/`
    - `raw/`: 변환할 원본 PPT 파일 위치
    - `inputs/`: OCR 모델의 입력으로 사용될 PDF 파일 위치
    - `outputs/`: 각 모델별 결과물(마크다운, 로그) 저장 위치
  - `olm/`, `deepseekocr/`, `chandraocr/`: 각 모델별 테스트 스크립트 폴더

## 사전 요구 사항 (Prerequisites)

각 모델은 별도의 Conda 환경에서 실행하는 것을 권장합니다 (참조: `test.sh`).
- `olmocr-test`
- `deepseek-ocr`
- `chandraocr`

공통적으로 필요한 라이브러리: `transformers`, `torch`, `pymupdf` (fitz), `aspose.slides` (PPT 변환용) 등.

## 사용 방법 (Usage)

### 1. PPT -> PDF 변환

`data/raw` 폴더에 있는 PPT/PPTX 파일을 `data/outputs`로 변환합니다.
(※ 주의: OCR 스크립트들은 `data/inputs` 폴더의 PDF를 읽으므로, 변환된 파일을 `data/inputs`로 이동시켜야 할 수 있습니다.)

```bash
python ppt_to_pdf.py
```

### 2. OCR 성능 테스트 실행

`test.sh` 스크립트를 사용하여 각 모델의 추론 테스트를 수행할 수 있습니다. 파일을 열어 원하는 모델의 주석을 해제하거나 수정하여 사용하세요.

```bash
./test.sh
```

또는 각 폴더로 이동하여 개별적으로 실행할 수 있습니다.

**Example (Chandra OCR):**
```bash
cd chandraocr
conda activate chandraocr
python chandraocr_test.py
```

## 지원 모델 및 특징

| 모델 | 스크립트 위치 | 특징 |
|------|--------------|------|
| **olmOCR** | `olm/olmocr_test.py` | `allenai/olmOCR-2-7B-1025` 사용. 페이지별 분석 및 결과 병합. |
| **DeepSeek-OCR** | `deepseekocr/deepseekocr_test.py` | `deepseek-ai/DeepSeek-OCR` 사용. 임시 이미지 생성 및 crop 모드 지원. |
| **Chandra OCR** | `chandraocr/chandraocr_test.py` | `datalab-to/chandra` (Qwen3-VL 기반) 사용. |

## 출력 결과 (Outputs)

테스트가 완료되면 `data/outputs/{model_name}/` 폴더에 다음과 같은 파일이 생성됩니다.

1.  **Markdown 결과**: `{filename}_{model}_full_output.md`  
    - PDF의 각 페이지가 마크다운으로 변환되어 저장됩니다.
2.  **개별 로그**: `{filename}_{model}_inference_log.log`  
    - 페이지별 추론 시간 및 전체 소요 시간이 기록됩니다.
3.  **전체 통합 로그**: `{model}_sum_total_inference_summary.log`  
    - 처리된 모든 파일에 대한 총 추론 시간 합계가 저장됩니다.
