!pip install -U olmocr bitsandbytes
!apt-get update
!apt-get install -y poppler-utils

import os
import torch
import gc
import base64
import re
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt

# --- CẤU HÌNH ---
PDF_PATH = "/content/HTL ebook 9 ĐHY.Dược Học Cổ Truyền (NXB Y Học 2002) - Phạm Xuân Sinh, 469 Trang.pdf"
OUTPUT_FILE = "/content/Ket_Qua_OCR_Vinh_Vien.txt"

# --- KHỞI TẠO MODEL 7B ---
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(model_id)

def run_ocr_no_skip(start, end):
    for pg in range(start, end + 1):
        try:
            gc.collect()
            torch.cuda.empty_cache()

            # Giữ 1536px để bảo đảm "Tuệ Tĩnh" đúng chính tả
            img_b64 = render_pdf_to_base64png(PDF_PATH, pg, target_longest_image_dim=2048)
            
            prompt = build_no_anchoring_v4_yaml_prompt()
            
            msgs = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]}]

            input_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            raw_img = Image.open(BytesIO(base64.b64decode(img_b64)))
            inputs = processor(text=[input_text], images=[raw_img], padding=True, return_tensors="pt").to("cuda")

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=4096, temperature=0, do_sample=False)

            res = processor.tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

            # 5. Ghi file
            if len(res) > 10:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(f"\n\n--- TRANG {pg} ---\n\n{res}")
                    f.flush()
                    os.fsync(f.fileno())
                print(f"✔ Đã quét xong Trang {pg} ({len(res)} ký tự)")
            else:
                print(f"⚠ Trang {pg} không có nội dung đáng kể.")

        except Exception as e:
            print(f"❌ Trang {pg} lỗi: {str(e)}")

# Chạy cụm trang (3, 7)
run_ocr_no_skip(20, 45)