import torch
import base64
import urllib.request
import time

from io import BytesIO
from PIL import Image
import PyPDF2
import os
import glob

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt

"""
모델 및 프로세서 초기화
"""
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("allenai/olmOCR-2-7B-1025", torch_dtype=torch.bfloat16).eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

"""
단일 PDF 처리 로직 함수
"""
def process_single_pdf(pdf_path, output_dir, model, processor, device):
    """
    단일 PDF 파일을 처리하고 결과를 Markdown 파일과 로그 파일로 저장

    Args:
        pdf_path (str): 처리할 PDF 파일의 전체 경로.
        output_dir (str): 결과 Markdown 및 로그 파일을 저장할 디렉토리.
        model: Qwen2_5_VLForConditionalGeneration 모델 인스턴스.
        processor: AutoProcessor 인스턴스.
        device: 모델이 로드된 PyTorch 장치 (예: 'cuda' 또는 'cpu').
    """
    all_page_outputs = []
    page_times = []
    total_inference_time = 0
    script_start_time = time.time() # 현재 PDF 파일 처리 시작 시간

    print(f"\n--- PDF 파일 처리 시작: {os.path.basename(pdf_path)} ---")

    # 2.1. 현재 PDF 파일의 총 페이지 수 가져오기
    num_pages = 0
    try:
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            num_pages = len(pdf_reader.pages)
        print(f"PDF 파일 '{os.path.basename(pdf_path)}'의 총 페이지 수: {num_pages}")
    except Exception as e:
        print(f"오류: PDF 파일 '{os.path.basename(pdf_path)}'의 페이지 수를 읽는 중 오류 발생: {e}")
        return # 오류 발생 시 함수 종료

    if num_pages == 0:
        print(f"경고: PDF 파일 '{os.path.basename(pdf_path)}'에 페이지가 없습니다. 건너뜁니다.")
        return # 페이지가 없으면 함수 종료

    # 2.2. 각 페이지를 순회하며 처리하는 내부 반복문
    for page_num in range(1, num_pages + 1):
        print(f"\n--- PDF: {os.path.basename(pdf_path)}, 페이지 {page_num}/{num_pages} 처리 중 ---")
        
        try:
            # PDF 페이지를 이미지로 렌더링
            image_base64 = render_pdf_to_base64png(pdf_path, page_num, target_longest_image_dim=1288)
            
            # 프롬프트 구성
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_no_anchoring_v4_yaml_prompt()},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ]
            
            # 채팅 템플릿 적용 및 프로세서 준비
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            main_image = Image.open(BytesIO(base64.b64decode(image_base64)))
            inputs = processor(
                text=[text],
                images=[main_image],
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for (key, value) in inputs.items()}
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            # 출력 생성
            output = model.generate(
                **inputs,
                temperature=0.1,
                max_new_tokens=1024,
                num_return_sequences=1,
                do_sample=True,
            )
            
            # 추론시간 측정 관련 코드
            torch.cuda.synchronize()
            end_time = time.time()
            elapsed = end_time - start_time
            total_inference_time += elapsed
            page_times.append(f"페이지 {page_num}: {elapsed:.2f} 초")
            print(f"페이지 {page_num} 추론 시간: {elapsed:.2f} 초")
            
            # 출력 디코딩
            prompt_length = inputs["input_ids"].shape[1]
            new_tokens = output[:, prompt_length:]
            text_output = processor.tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True
            )
            
            all_page_outputs.append(text_output[0])
            print(f"페이지 {page_num}의 첫 50자 출력: {text_output[0][:50]}...")

        except Exception as e:
            print(f"오류: PDF '{os.path.basename(pdf_path)}' 페이지 {page_num} 처리 중 오류 발생: {e}")
            all_page_outputs.append(f"--- 오류 발생 (페이지 {page_num}): {e} ---") 
            continue # 다음 페이지로 계속 진행

            """
            추론 끝날 때마다 GPU 캐시 제거
            """
        finally:
            torch.cuda.empty_cache()

    print(f"\nPDF '{os.path.basename(pdf_path)}' 총 {num_pages} 페이지 추론 시간: {total_inference_time:.2f} 초")

    # 2.3. 결과 저장 (각 PDF 파일마다 별도로 저장)
    final_markdown_content = "\n\n---\n\n".join(all_page_outputs)

    pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_markdown_path = os.path.join(output_dir, f"{pdf_base_name}_olmocr_full_output.md")

    with open(output_markdown_path, "w", encoding="utf-8") as f:
        f.write(final_markdown_content)
    print(f"\n모델의 전체 출력 내용이 '{output_markdown_path}'에 저장되었습니다.")

    ## 추론 시간 로그 관련
    script_end_time = time.time()
    overall_duration = script_end_time - script_start_time
    output_log_path = os.path.join(output_dir, f"{pdf_base_name}_olmocr_inference_log.log")

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

if __name__ == "__main__":
    input_dir = "../data/inputs/"
    output_dir = "../data/outputs/olmocr/"

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
        pdf_inference_time = process_single_pdf(current_pdf_path, output_dir, model, processor, device)
        sum_total_inference_time += pdf_inference_time

    # 모든 PDF 파일에 대한 총 추론 시간을 별도의 파일에 저장
    sum_total_log_path = os.path.join(output_dir, "olmocr_sum_total_inference_summary.log")
    with open(sum_total_log_path, "w", encoding="utf-8") as f:
        f.write(f"--- 모든 PDF 파일들에 대한 모델 총 추론 시간 (generate_hf만 합산): {sum_total_inference_time:.2f} 초 ---\n")
        f.write(f"요약 생성 시각: {time.ctime(time.time())}\n")
        f.write(f"처리된 PDF 파일 수: {len(pdf_files_to_process)}개\n")
    
    print("\n--- 모든 PDF 파일 처리 완료 ---")
