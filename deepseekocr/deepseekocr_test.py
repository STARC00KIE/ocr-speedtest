from transformers import AutoModel, AutoTokenizer

import torch
import os
import time
import glob

import fitz
from PIL import Image

"""
모델 및 프로세서 초기화
"""
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
     model_name, 
     _attn_implementation='flash_attention_2', 
     torch_dtype=torch.bfloat16, 
     trust_remote_code=True, 
     use_safetensors=True)
model = model.eval().to('cuda')

"""
단일 PDF 처리 로직 함수
"""
def process_single_pdf(pdf_path, output_dir, model, tokenizer):
    """
    단일 PDF 파일을 처리하고 결과를 Markdown 파일과 로그 파일로 저장

    Args:
        pdf_path (str): 처리할 PDF 파일의 전체 경로.
        output_dir (str): 결과 Markdown 및 로그 파일을 저장할 디렉토리.
        model: AutoModel 모델 인스턴스.
        tokenizer
        processor: AutoProcessor 인스턴스.
    """
    all_page_outputs = []
    page_times = [] # 추론시간 저장할 리스트
    total_inference_time = 0
    script_start_time = time.time() # 모델 로드 이후 전체 스크립트 실행 시작 시간

    print(f"\n--- PDF 파일 처리 시작: {os.path.basename(pdf_path)} ---")

    # PDF 파일 열기 및 총 페이지 수 확인
    try:
        pdf_document = fitz.open(pdf_path)
        num_pages = pdf_document.page_count
        print(f"PDF 파일 '{pdf_path}'의 총 페이지 수: {num_pages}")
    except Exception as e:
        print(f"PDF 파일을 열거나 페이지 수를 읽는 중 오류 발생: {e}")
        exit()

    # DeepSeek-OCR의 개별 페이지 Markdown 결과가 저장될 임시 디렉토리
    temp_output_dir = '../data/outputs/deepseekocr/deepseek_ocr_temp_outputs'
    os.makedirs(temp_output_dir, exist_ok=True) # 디렉토리 생성

    # 각 페이지를 순회하며 처리
    for page_idx in range(num_pages): # fitz는 0부터 페이지 인덱싱
        page_num = page_idx + 1
        print(f"\n--- PDF: {os.path.basename(pdf_path)}, 페이지 {page_num}/{num_pages} 처리 중 ---")

        # olmocr과 구조 비슷하게 변경
        try:
            # 페이지를 이미지로 변환하여 임시 파일로 저장
            temp_image_file = os.path.join(temp_output_dir, f"temp_page_{page_idx}.jpg")
            page = pdf_document[page_idx]
            zoom = 2.0 # 해상도 조절 (높을수록 좋지만 메모리 사용량 증가)
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)
            pix.save(temp_image_file)
            print(f"페이지 {page_num}를 '{temp_image_file}'로 저장했습니다.")

            # DeepSeek-OCR 모델 프롬프트
            prompt = "<image>\n<|grounding|>Convert the document to markdown. "

            # 추론시간 체크
            torch.cuda.synchronize()
            start_time = time.time()

            # model.infer 함수는 save_results=True일 때 output_path에 결과를 저장
            res = model.infer(tokenizer,
                        prompt=prompt,
                        image_file=temp_image_file,
                        output_path=temp_output_dir, # 각 페이지의 개별 Markdown 결과를 이 디렉토리에 저장
                        base_size=1024,
                        image_size=640,
                        crop_mode=True,
                        save_results=True, # 결과를 파일로 저장하도록 설정
                        test_compress=True)

            # 추론시간 측정 관련 코드
            torch.cuda.synchronize()
            end_time = time.time()
            elapsed = end_time - start_time
            total_inference_time += elapsed
            page_times.append(f"페이지 {page_num}: {elapsed:.2f} 초") # 페이지별 시간 기록
            print(f"페이지 {page_num} 추론 시간: {elapsed:.2f} 초")

            # 페이지 출력 결과 읽기
            result_mmd_file = os.path.join(temp_output_dir, "result.mmd")
            if os.path.exists(result_mmd_file):
                        with open(result_mmd_file, 'r', encoding='utf-8') as f:
                            page_markdown = f.read()
                        os.remove(result_mmd_file) # 읽은 후 result.mmd 파일 삭제
            else:
                raise FileNotFoundError(f"'result.mmd' 파일을 '{result_mmd_file}'에서 찾을 수 없습니다.")
            
            if page_markdown is not None and isinstance(page_markdown, str):
                all_page_outputs.append(page_markdown)

            # 임시 이미지 파일 삭제
            try:
                os.remove(temp_image_file)
            except OSError as e:
                print(f"임시 이미지 파일 삭제 중 오류 발생: {e}")

        except Exception as e:
            print(f"오류: PDF '{os.path.basename(pdf_path)}' 페이지 {page_num} 처리 중 오류 발생: {e}")
            all_page_outputs.append(f"--- 오류 발생 (페이지 {page_num}): {e} ---") 
            continue # 다음 페이지로 계속 진행
        
        finally:
            torch.cuda.empty_cache()

    # PDF 문서 닫기
    pdf_document.close()

    print(f"\nPDF '{os.path.basename(pdf_path)}' 총 {num_pages} 페이지 추론 시간: {total_inference_time:.2f} 초")

    # 모든 페이지의 Markdown 출력을 하나의 파일로 결합
    # 각 페이지 사이에 Markdown 구분선 (---)을 추가하여 페이지 구분을 명확히 합니다.
    final_markdown_content = "\n\n---\n\n".join(all_page_outputs)

    # PDF 파일 이름을 기반으로 출력 파일 경로 설정
    pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_markdown_path = os.path.join(output_dir, f"{pdf_base_name}_deepseekocr_full_output.md")

    # 임시 디렉토리 정리 (비어있다면 삭제)
    try:
        if os.path.exists(temp_output_dir) and not os.listdir(temp_output_dir):
            os.rmdir(temp_output_dir)
            print(f"임시 디렉토리 '{temp_output_dir}'를 삭제했습니다.")
        elif os.path.exists(temp_output_dir):
            print(f"경고: 임시 출력 디렉토리 '{temp_output_dir}'에 남아있는 파일이 있습니다. 수동으로 정리해주세요.")
    except OSError as e:
        print(f"임시 디렉토리 삭제 중 오류 발생: {e}")

    with open(output_markdown_path, "w", encoding="utf-8") as f:
        f.write(final_markdown_content)
    print(f"\n모델의 전체 출력 내용이 '{output_markdown_path}'에 저장되었습니다.")

    ## 추론시간 log 관련
    script_end_time = time.time() # 전체 스크립트 실행 종료 시간
    overall_duration = script_end_time - script_start_time # 모델 로드 이후 총 실행 시간
    output_log_path = os.path.join(output_dir, f"{pdf_base_name}_deepseekocr_inference_log.log")

    ## 추론시간 log 추가
    with open(output_log_path, "w", encoding="utf-8") as log_f:
        log_f.write(f"--- 추론 로그: '{pdf_path}' ---\n")
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
주요 구조 정보
"""
# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

# 실행 함수
if __name__ == "__main__":
    input_dir = "../data/inputs/"
    output_dir = "../data/outputs/deepseekocr/"

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
        pdf_inference_time = process_single_pdf(current_pdf_path, output_dir, model, tokenizer)
        sum_total_inference_time += pdf_inference_time

    # 모든 PDF 파일에 대한 총 추론 시간을 별도의 파일에 저장
    sum_total_log_path = os.path.join(output_dir, "deepseekocr_sum_total_inference_summary.log")
    with open(sum_total_log_path, "w", encoding="utf-8") as f:
        f.write(f"--- 모든 PDF 파일들에 대한 모델 총 추론 시간 (generate_hf만 합산): {sum_total_inference_time:.2f} 초 ---\n")
        f.write(f"요약 생성 시각: {time.ctime(time.time())}\n")
        f.write(f"처리된 PDF 파일 수: {len(pdf_files_to_process)}개\n")

        print("\n--- 모든 PDF 파일 처리 완료 ---")
