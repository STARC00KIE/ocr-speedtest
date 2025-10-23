"""
1. 폴더 내부 파일명 읽어오기
2. aspose.slides 사용하여 .pptx -> .pdf로 변환
"""
import os
import aspose.slides as slides

def main():
    folder_dir = "./data/raw"
    file_names = os.listdir(folder_dir)
    output_dir = "./data/outputs"

    for fn in file_names:
        if fn.lower().endswith((".pptx", ".ppt")):
            input_path = os.path.join(folder_dir, fn)
            output_path = os.path.join(output_dir, f"{os.path.splitext(fn)[0]}.pdf")

            with slides.Presentation(input_path) as pres:
                pres.save(output_path, slides.export.SaveFormat.PDF)
                print(f"Converted: {fn} → {output_path}")

if __name__ == "__main__":
    main()