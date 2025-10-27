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
import glob

"""
모델 및 프로세서 초기화
"""
model = Qwen3VLForConditionalGeneration.from_pretrained("datalab-to/chandra",
                                  torch_dtype=torch.bfloat16,
                                  trust_remote_code=True).cuda()
model.processor = AutoProcessor.from_pretrained("datalab-to/chandra")

"""
단일 PDF 처리 로직 함수
"""
def process_single_pdf(pdf_path, output_dir, model):
    """
    단일 PDF 파일을 처리하고 결과를 Markdown 파일과 로그 파일로 저장

    Args:
        pdf_path (str): 처리할 PDF 파일의 전체 경로.
        output_dir (str): 결과 Markdown 및 로그 파일을 저장할 디렉토리.
        model: Qwen3_VLForConditionalGeneration 모델 인스턴스.
        processor: AutoProcessor 인스턴스.
        device: 모델이 로드된 PyTorch 장치 (예: 'cuda' 또는 'cpu').
    """
    all_page_outputs = [] # 각 페이지의 마크다운 결과를 저장할 리스트
    page_times = [] # 추론시간 저장할 리스트
    total_inference_time = 0
    script_start_time = time.time() # 모델 로드 이후 전체 스크립트 실행 시작 시간

    print(f"\n--- PDF 파일 처리 시작: {os.path.basename(pdf_path)} ---")

    page_num = 0
    try:
        pdf_document = fitz.open(pdf_path)
        num_pages = pdf_document.page_count
        print(f"PDF 파일 '{pdf_path}'의 총 페이지 수: {num_pages}")
    except Exception as e:
        print(f"PDF 파일을 열거나 페이지 수를 읽는 중 오류 발생: {e}")
        exit()


    for page_idx in range(num_pages):
        page_num = page_idx + 1
        print(f"\n--- PDF: {os.path.basename(pdf_path)}, 페이지 {page_num}/{num_pages} 처리 중 ---")
        
        try:
            page = pdf_document[page_idx]
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

            # 추론시간 측정 관련 코드
            torch.cuda.synchronize()
            end_time = time.time()  # 전체 처리 종료 시각
            elapsed = end_time - start_time
            total_inference_time += elapsed
            page_times.append(f"페이지 {page_num}: {elapsed:.2f} 초") # 페이지별 시간 기록
            print(f"페이지 {page_num} 추론 시간: {elapsed:.2f} 초")

            # 결과 파싱
            markdown = parse_markdown(result.raw)
            # 각 페이지의 결과를 구분하기 위해 페이지 번호와 함께 저장
            all_page_outputs.append(f"\n\n--- Page {page_num} ---\n\n" + markdown)

        finally:
            torch.cuda.empty_cache()

    # PDF 문서 닫기 (Close PDF document)
    pdf_document.close()

    print(f"\nPDF '{os.path.basename(pdf_path)}' 총 {num_pages} 페이지 추론 시간: {total_inference_time:.2f} 초")

    # 모든 페이지의 마크다운 결과를 하나의 파일로 저장 (Save all page markdown results to a single file)
    final_markdown_content = "\n".join(all_page_outputs)

    # PDF 파일 이름을 기반으로 출력 파일 경로 설정
    pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_markdown_path = os.path.join(output_dir, f"{pdf_base_name}_chandraocr_full_output.md")

    with open(output_markdown_path, "w", encoding="utf-8") as f:
        f.write(final_markdown_content)
    print(f"\n모델의 전체 출력 내용이 '{output_markdown_path}'에 저장되었습니다.")

    ## 추론시간 log 관련
    script_end_time = time.time() # 전체 스크립트 실행 종료 시간
    overall_duration = script_end_time - script_start_time # 모델 로드 이후 총 실행 시간

    output_log_path = os.path.join(output_dir, f"{pdf_base_name}_chandraocr_inference_log.log")

    ## 추론시간 log 추가
    with open(output_log_path, "w", encoding="utf-8") as log_f:
        log_f.write(f"--- 추론 로그: '{os.path.basename(pdf_path)}' ---\n")
        log_f.write(f"로그 생성 시각: {time.ctime(time.time())}\n")
        log_f.write(f"총 PDF 페이지 수: {num_pages}\n\n")
        log_f.write("--- 페이지별 추론 시간 ---\n")
        log_f.write("\n".join(page_times) + "\n\n")
        log_f.write(f"--- 요약 ---\n")
        log_f.write(f"전체 모델 추론 시간 (generate만 합산): {total_inference_time:.2f} 초\n")
        log_f.write(f"전체 PDF 처리 시간 (모델 로드 이후): {overall_duration:.2f} 초\n")
    print(f"추론 시간 로그가 '{output_log_path}'에 저장되었습니다.")

    return total_inference_time

"""
코드 실행
"""
if __name__ == "__main__":
    input_dir = "../data/inputs/"
    output_dir = "../data/outputs/chandraocr/"

    os.makedirs(output_dir, exist_ok=True)

    pdf_files_to_process = glob.glob(os.path.join(input_dir, "*.pdf"))

    if not pdf_files_to_process:
        print(f"경고: '{input_dir}'에서 처리할 PDF 파일을 찾을 수 없습니다.")
        print("스크립트를 종료합니다.")
        exit()

    print(f"'{input_dir}'에서 총 {len(pdf_files_to_process)}개의 PDF 파일을 찾았습니다.")

    # 총 추론 시간 누적 변수
    sum_total_inference_time = 0
    for current_pdf_path in pdf_files_to_process:
        # 정의한 함수를 호출하여 각 PDF 파일을 처리합니다.
        pdf_inference_time = process_single_pdf(current_pdf_path, output_dir, model)
        sum_total_inference_time += pdf_inference_time

    # 모든 PDF 파일에 대한 총 추론 시간을 별도의 파일에 저장
    sum_total_log_path = os.path.join(output_dir, "olmocr_sum_total_inference_summary.log")
    with open(sum_total_log_path, "w", encoding="utf-8") as f:
        f.write(f"--- 모든 PDF 파일들에 대한 모델 총 추론 시간 (generate_hf만 합산): {sum_total_inference_time:.2f} 초 ---\n")
        f.write(f"요약 생성 시각: {time.ctime(time.time())}\n")
        f.write(f"처리된 PDF 파일 수: {len(pdf_files_to_process)}개\n")
    
    print("\n--- 모든 PDF 파일 처리 완료 ---")

