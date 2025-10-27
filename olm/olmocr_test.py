import torch
import base64
import urllib.request
import time

from io import BytesIO
from PIL import Image
import PyPDF2
import os

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt

# Initialize the model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("allenai/olmOCR-2-7B-1025", torch_dtype=torch.bfloat16).eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 샘플 pdf 가져오기
# urllib.request.urlretrieve("https://olmocr.allenai.org/papers/olmocr.pdf", "./paper.pdf")

# pdf 경로
pdf_path = "../data/inputs/rfp1.pdf"

all_page_outputs = []
page_times = [] # 추론시간 저장할 리스트
total_inference_time = 0
script_start_time = time.time() # 모델 로드 이후 전체 스크립트 실행 시작 시간

# pdf 총 페이지 수
try:
    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        num_pages = len(pdf_reader.pages)
    print(f"PDF 파일 '{pdf_path}'의 총 페이지 수: {num_pages}")
except Exception as e:
    print(f"PDF 페이지 수를 읽는 중 오류 발생: {e}")
    exit()

# 각 페이지를 순회하며 처리
for page_num in range(1, num_pages + 1): # 페이지 번호는 1부터 시작
    print(f"\n--- 페이지 {page_num}/{num_pages} 처리 중 ---")

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

    torch.cuda.synchronize()  # GPU 작업 싱크 맞추기
    start_time = time.time()  # 시작 시각

    # 출력 생성
    output = model.generate(
        **inputs,
        temperature=0.1,
        max_new_tokens=1024,
        num_return_sequences=1,
        do_sample=True,
    )

    torch.cuda.synchronize()
    end_time = time.time()  # 종료 시각
    elapsed = end_time - start_time
    total_inference_time += elapsed
    page_times.append(f"페이지 {page_num}: {elapsed:.2f} 초") # 페이지별 시간 기록
    print(f"페이지 {page_num} 추론 시간: {elapsed:.2f} 초")

    # 출력 디코딩
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(
        new_tokens, skip_special_tokens=True
    )

    # 각 페이지의 출력을 리스트에 추가
    all_page_outputs.append(text_output[0])
    print(f"페이지 {page_num}의 첫 50자 출력: {text_output[0][:50]}...")

print(f"\n총 {num_pages} 페이지 추론 시간: {total_inference_time:.2f} 초")

# 모든 페이지의 출력을 하나의 Markdown 문자열로 결합
# 각 페이지 사이에 Markdown 구분선 (---)을 추가하여 페이지 구분을 명확
final_markdown_content = "\n\n---\n\n".join(all_page_outputs)

# PDF 파일 이름을 기반으로 출력 파일 경로 설정
pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
output_dir = "../data/outputs/olmocr/"
output_path = f"../data/outputs/olmocr/{pdf_base_name}_full_output.md"

# 최종 Markdown 내용을 파일에 저장
with open(output_path, "w", encoding="utf-8") as f:
    f.write(final_markdown_content)
print(f"\n모델의 전체 출력 내용이 '{output_path}'에 저장되었습니다.")

## 추론시간 log 관련
script_end_time = time.time() # 전체 스크립트 실행 종료 시간
overall_duration = script_end_time - script_start_time # 모델 로드 이후 총 실행 시간

output_log_path = os.path.join(output_dir, f"{pdf_base_name}_inference_log.log")

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