!apt-get install tesseract-ocr tesseract-ocr-vie poppler-utils
!pip install pytesseract opencv-python Pillow

import os
import pytesseract
from PIL import Image
import cv2
import numpy as np
import gc

# --- CẤU HÌNH ---
PDF_PATH = "/content/HTL ebook 9 ĐHY.Dược Học Cổ Truyền (NXB Y Học 2002) - Phạm Xuân Sinh, 469 Trang.pdf"
TEMP_DIR = "/content/temp_images"
OUTPUT_FILE = "/content/Ban_Nhap_Soat_Loi.txt"

# Tạo thư mục tạm
os.makedirs(TEMP_DIR, exist_ok=True)

def preprocess_image_v2(img_path):
    """
    Xử lý ảnh chuyên sâu cho sách cũ: khử nhiễu, tăng tương phản cục bộ
    """
    # Đọc ảnh bằng OpenCV (Grayscale)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # 1. Khử nhiễu nhẹ (giữ nét chữ nhưng xóa hạt bụi trên giấy cũ)
    denoised = cv2.fastNlMeansDenoising(img, h=10)
    
    # 2. Tăng độ tương phản bằng CLAHE (Cân bằng độ sáng thông minh)
    # Giúp chữ mờ trở nên đậm hơn mà không làm cháy ảnh
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(denoised)
    
    # 3. Chuyển về PIL Image để Pytesseract đọc
    return Image.fromarray(contrast_enhanced)

def run_stable_ocr_system(start, end):
    # Kiểm tra file PDF có tồn tại không
    if not os.path.exists(PDF_PATH):
        print(f"❌ Không tìm thấy file PDF tại: {PDF_PATH}")
        return

    print(f"--- Bước 1: Tách PDF từ trang {start} đến {end} (300 DPI) ---")
    
    # Xóa ảnh cũ
    os.system(f"rm -rf {TEMP_DIR}/*")
    
    # Sử dụng pdftoppm (từ gói poppler-utils) cực nhanh và ít tốn RAM
    # %03d giúp sắp xếp thứ tự file chính xác 001, 002...
    os.system(f"pdftoppm -f {start} -l {end} -r 300 '{PDF_PATH}' {TEMP_DIR}/page")

    print(f"--- Bước 2: OCR tiếng Việt với Tesseract LSTM ---")
    
    # Lấy danh sách ảnh và sắp xếp đúng thứ tự
    image_files = sorted([f for f in os.listdir(TEMP_DIR) if f.endswith(('.ppm', '.jpg', '.png'))])
    
    # Reset file output nếu chạy lại từ đầu (tùy chọn)
    # with open(OUTPUT_FILE, "w", encoding="utf-8") as f: f.write("")

    for i, file_name in enumerate(image_files):
        current_pg = start + i
        img_path = os.path.join(TEMP_DIR, file_name)
        
        try:
            # Tiền xử lý ảnh
            processed_img = preprocess_image_v2(img_path)
            
            # Cấu hình OCR: 
            # --psm 3: Tự động phân tích bố cục trang sách
            # --oem 3: Sử dụng engine LSTM tốt nhất hiện nay
            custom_config = r'--oem 3 --psm 3'
            text = pytesseract.image_to_string(processed_img, lang='vie', config=custom_config)

            # Ghi file (Chế độ Append 'a' để tránh mất dữ liệu nếu bị ngắt quãng)
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n\n{'='*30}\n TRANG {current_pg}\n{'='*30}\n\n{text}")
            
            print(f"✅ Hoàn thành trang {current_pg}")
            
            # Giải phóng bộ nhớ ngay lập tức
            os.remove(img_path)
            del processed_img
            gc.collect()

        except Exception as e:
            print(f"❌ Lỗi tại trang {current_pg}: {e}")

    print(f"\n--- XONG! Kết quả lưu tại: {OUTPUT_FILE} ---")

# --- THỰC THI ---
# Bạn có thể chia nhỏ để chạy, ví dụ 20 đến 50
run_stable_ocr_system(20, 25)