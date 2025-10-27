import time

from transformers import AutoModel, AutoProcessor
from chandra.model.hf import generate_hf
from chandra.model.schema import BatchInputItem
from chandra.output import parse_markdown

# 수정된 코드
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

import torch

import fitz  # PyMuPDF
from PIL import Image
import io
import os

model = Qwen3VLForConditionalGeneration.from_pretrained("datalab-to/chandra",
                                  torch_dtype=torch.bfloat16,
                                  trust_remote_code=True).cuda()
model.processor = AutoProcessor.from_pretrained("datalab-to/chandra")

# PDF to jpg
pdf_path = "../data/inputs/rfp1.pdf"
page_num = 0

print(f"PDF 파일 '{pdf_path}' 열기...")
try:
    doc = fitz.open(pdf_path)
except fitz.FileNotFoundError:
    print(f"오류: PDF 파일 '{pdf_path}'을(를) 찾을 수 없습니다.")
    exit() # 파일이 없으면 프로그램 종료

total_pages = len(doc)
print(f"총 {total_pages} 페이지 감지.")

all_page_markdown_results = [] # 각 페이지의 마크다운 결과를 저장할 리스트
page_times = [] # 추론시간 저장할 리스트
total_inference_time = 0
script_start_time = time.time() # 모델 로드 이후 전체 스크립트 실행 시작 시간

for page_num in range(total_pages):
    print(f"페이지 {page_num + 1}/{total_pages} 처리 중...")
    page = doc[page_num]
    pix = page.get_pixmap() # 페이지를 픽셀맵으로 렌더링
    img = Image.open(io.BytesIO(pix.tobytes("png"))) # PIL_IMAGE로 변환

    # 모델 입력 준비
    batch = [
        BatchInputItem(
            image=img,
            prompt_type="ocr_layout"
        )
    ]

    torch.cuda.synchronize()
    start_time = time.time()  # 전체 처리 시작 시각

    # 추론 실행
    result = generate_hf(batch, model)[0]

    torch.cuda.synchronize()
    end_time = time.time()  # 전체 처리 종료 시각
    elapsed = end_time - start_time
    total_inference_time += elapsed
    page_times.append(f"페이지 {page_num + 1}: {elapsed:.2f} 초") # 페이지별 시간 기록
    print(f"페이지 {page_num + 1} 추론 시간: {elapsed:.2f} 초")

    # 결과 파싱
    markdown = parse_markdown(result.raw)
    # 각 페이지의 결과를 구분하기 위해 페이지 번호와 함께 저장
    all_page_markdown_results.append(f"\n\n--- Page {page_num + 1} ---\n\n" + markdown)

print(f"\n총 {total_pages} 페이지 추론 시간: {total_inference_time:.2f} 초")

# 5. PDF 문서 닫기 (Close PDF document)
doc.close()

# 6. 모든 페이지의 마크다운 결과를 하나의 파일로 저장 (Save all page markdown results to a single file)
final_markdown_output = "\n".join(all_page_markdown_results)

# PDF 파일 이름을 기반으로 출력 파일 경로 설정
pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
output_dir = "../data/outputs/chandraocr/"
output_path = f"../data/outputs/chandraocr/{pdf_base_name}_full_output.md"


with open(output_path, "w", encoding="utf-8") as f:
    f.write(final_markdown_output)

print(f"\n모델의 전체 출력 내용이 '{output_path}'에 저장되었습니다.")

## 추론시간 log 관련
script_end_time = time.time() # 전체 스크립트 실행 종료 시간
overall_duration = script_end_time - script_start_time # 모델 로드 이후 총 실행 시간

output_log_path = os.path.join(output_dir, f"{pdf_base_name}_inference_log.log")

## 추론시간 log 추가
with open(output_log_path, "w", encoding="utf-8") as log_f:
    log_f.write(f"--- 추론 로그: '{pdf_path}' ---\n")
    log_f.write(f"로그 생성 시각: {time.ctime(time.time())}\n")
    log_f.write(f"총 PDF 페이지 수: {total_pages}\n\n")
    log_f.write("--- 페이지별 추론 시간 ---\n")
    log_f.write("\n".join(page_times) + "\n\n")
    log_f.write(f"--- 요약 ---\n")
    log_f.write(f"전체 모델 추론 시간 (generate만 합산): {total_inference_time:.2f} 초\n")
    log_f.write(f"전체 PDF 처리 시간 (모델 로드 이후): {overall_duration:.2f} 초\n")

print(f"추론 시간 로그가 '{output_log_path}'에 저장되었습니다.")