"""
species_detector.py
--------------------
Dùng Gemini (vision) để nhận diện GIỐNG CÂY TRỒNG từ ảnh, giới hạn trong
đúng danh sách cây hệ thống đang hỗ trợ (đọc động từ crops_config.yaml).

Vì sao dùng Gemini thay vì tự train:
- Nhận diện giống cây là bài toán RỘNG, tổng quát — Gemini đã có sẵn kiến
  thức về hàng nghìn loại cây từ lúc huấn luyện, không cần bạn tự thu
  thập ảnh riêng cho từng giống mới thêm vào.
- Ngược lại, GIAI ĐOẠN SINH TRƯỞNG của từng cây là đặc trưng hẹp, riêng
  biệt theo dữ liệu thực tế của bạn -> vẫn phải tự train YOLO như đã làm
  với lúa (xem train.py). File này KHÔNG thay thế việc đó.

Khi thêm cây mới vào crops_config.yaml (vd xoài), KHÔNG cần sửa file
này — danh sách lựa chọn được đọc động từ config mỗi lần gọi.

Yêu cầu: đặt biến môi trường GEMINI_API_KEY trước khi gọi
(vd trong Colab: os.environ["GEMINI_API_KEY"] = getpass(...)).

Cài đặt: pip install google-genai pillow pyyaml
"""
from __future__ import annotations

import io
import os
import re
import time
import unicodedata
from pathlib import Path

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "crops_config.yaml"

# Trả về giá trị này khi Gemini không xác định được, hoặc cây không nằm
# trong danh sách hệ thống đang hỗ trợ.
UNKNOWN = "khong_xac_dinh"

# Công tắc tắt Gemini để demo không phụ thuộc quota API
USE_GEMINI = os.environ.get("USE_GEMINI", "0") == "1"

# Các dấu hiệu lỗi TẠM THỜI từ phía Google (server quá tải, hết hạn mức
# tức thời...) — đáng để thử lại. Lỗi khác (sai API key, ảnh hỏng...) thì
# không thử lại vì thử lại cũng sẽ lỗi y hệt, chỉ tổ chờ lâu vô ích.
_RETRYABLE_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "timeout", "Timeout")


def load_crops_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _supported_crops(config: dict) -> dict[str, str]:
    """key trong crops_config.yaml -> tên hiển thị."""
    return {
        key: cfg.get("display_name", key)
        for key, cfg in config["crops"].items()
    }


def _normalize(text: str) -> str:
    """Bỏ dấu tiếng Việt, chữ thường, bỏ ký tự thừa -> dễ so khớp hơn."""
    text = text.strip().lower()
    text = text.strip("\"'`.*_ \n\t")
    nfkd = unicodedata.normalize("NFD", text)
    without_diacritics = "".join(c for c in nfkd if not unicodedata.combining(c))
    without_diacritics = without_diacritics.replace("đ", "d")
    return re.sub(r"[^a-z0-9_]+", "_", without_diacritics).strip("_")


def _build_prompt(supported: dict[str, str]) -> str:
    options_text = "\n".join(f'- "{key}" ({name})' for key, name in supported.items())
    return f"""Bạn là trợ lý nông nghiệp chuyên nhận diện cây trồng.
Nhìn kỹ bức ảnh và xác định đây là loại cây gì.

Ưu tiên trả lời đúng key trong danh sách hệ thống đang hỗ trợ (nếu khớp):
{options_text}

Nếu không phải cây trong danh sách trên, hãy trả lời tên cây phổ biến bằng tiếng Việt, viết thường, không dấu, dùng gạch dưới thay khoảng trắng (ví dụ: ca_phe, dua_hau, khoai_lang, hoa_hong...).

Nếu không nhận diện được thì trả lời: {UNKNOWN}

Chỉ trả lời đúng 1 từ khóa duy nhất, không giải thích, không thêm dấu câu hay markdown.
"""


def _call_gemini_with_retry(client, model: str, contents: list, max_retries: int = 3,
                             base_delay: float = 2.0):
    """
    Gọi Gemini, tự động thử lại nếu gặp lỗi TẠM THỜI (503 quá tải, 429 vượt
    hạn mức tức thời...). Lỗi khác (sai key, ảnh hỏng...) raise ngay, không
    thử lại vô ích. Thời gian chờ tăng dần: 2s, 4s, 8s.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except Exception as e:
            last_error = e
            is_retryable = any(marker in str(e) for marker in _RETRYABLE_MARKERS)
            if not is_retryable or attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
    raise last_error  # không bao giờ tới đây, chỉ để rõ ràng cho type checker


def detect_crop_species(
    image: Image.Image,
    api_key: str | None = None,
    model: str = "gemini-3.5-flash",
) -> tuple[str | None, str]:
    """
    Nhận diện giống cây trồng trong ảnh.

    Trả về (crop_key, raw_answer):
    - crop_key: key khớp trong crops_config.yaml (vd "lua"), hoặc None
      nếu Gemini không xác định được / trả lời cây không được hỗ trợ.
    - raw_answer: câu trả lời gốc từ Gemini (để log/debug khi cần).
    """
    if not USE_GEMINI:
        return None, "Gemini đang tắt (USE_GEMINI=0) — vui lòng chọn thủ công."

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ImportError("Cần cài: pip install google-genai") from e

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Chưa có GEMINI_API_KEY. Đặt biến môi trường trước khi gọi hàm này, vd:\n"
            '  os.environ["GEMINI_API_KEY"] = "..."'
        )

    config = load_crops_config()
    supported = _supported_crops(config)
    prompt = _build_prompt(supported)

    # Encode ảnh tường minh bằng Part.from_bytes thay vì truyền thẳng đối
    # tượng PIL.Image vào contents — cách truyền thẳng có thể không được
    # một số phiên bản SDK google-genai xử lý đúng, khiến Gemini không
    # thực sự "nhìn thấy" ảnh và luôn trả lời mặc định.
    buffer = io.BytesIO()
    rgb_image = image.convert("RGB")  # phòng trường hợp ảnh có kênh alpha (PNG)
    rgb_image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    client = genai.Client(api_key=api_key)
    response = _call_gemini_with_retry(
        client,
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
    )
    raw_answer = (response.text or "").strip()
    normalized_answer = _normalize(raw_answer)

    if normalized_answer in supported:
        return normalized_answer, raw_answer

    normalized_display_map = {_normalize(name): key for key, name in supported.items()}
    if normalized_answer in normalized_display_map:
        return normalized_display_map[normalized_answer], raw_answer

    if normalized_answer and normalized_answer != _normalize(UNKNOWN):
        return normalized_answer, raw_answer

    return None, raw_answer


if __name__ == "__main__":
    # Test nhanh: python species_detector.py duong_dan_anh.jpg
    import sys

    if len(sys.argv) < 2:
        print("Dùng: python species_detector.py <đường_dẫn_ảnh>")
        raise SystemExit(1)

    img = Image.open(sys.argv[1])
    crop_key, raw = detect_crop_species(img)
    if crop_key:
        print(f"Nhận diện: {crop_key}")
    else:
        print(f"Không xác định được cây được hỗ trợ (Gemini trả lời: '{raw}')")


def generate_stage_advice(
    crop_display_name: str,
    stage_name: str,
    api_key: str | None = None,
    model: str = "gemini-3.6-flash",
) -> dict:
    """
    Dùng Gemini sinh tư vấn kỹ thuật cho 1 giai đoạn, trả về dict đúng
    cấu trúc {"tom_tat": str, "hanh_dong_de_xuat": [...], "canh_bao_rui_ro": [...]}.
    Raise Exception nếu lỗi, để app.py tự fallback sang advice tĩnh.
    """
    from google import genai
    import json as _json

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Chưa có GEMINI_API_KEY.")

    prompt = f"""Bạn là kỹ sư nông nghiệp. Cây {crop_display_name} đang ở giai
đoạn "{stage_name}". Hãy trả lời DUY NHẤT 1 JSON object hợp lệ (không có
markdown, không có ```json, không giải thích gì thêm), đúng format sau:

{{"tom_tat": "1-2 câu tóm tắt đặc điểm giai đoạn này",
"hanh_dong_de_xuat": ["hành động cụ thể 1", "hành động cụ thể 2", "hành động cụ thể 3"],
"canh_bao_rui_ro": ["rủi ro cần chú ý 1", "rủi ro cần chú ý 2"]}}

Nội dung bằng tiếng Việt, thực tế, cụ thể, dễ áp dụng ngay cho nông dân."""

    client = genai.Client(api_key=api_key)
    response = _call_gemini_with_retry(
        client, model=model, contents=[prompt], max_retries=2, base_delay=2.0,
    )
    text = (response.text or "").strip()
    text = text.strip("`")
    if text.startswith("json"):
        text = text[4:].strip()

    data = _json.loads(text)
    if not isinstance(data, dict) or "tom_tat" not in data:
        raise ValueError("Gemini trả JSON sai cấu trúc.")
    return data


def generate_combined_advice(
    crop_display_name: str,
    stage_name: str | None,
    disease_names: list[str] | None,
    api_key: str | None = None,
    model: str = "gemini-3.6-flash",
) -> dict:
    """
    Dùng Gemini sinh tư vấn TỔNG HỢP dựa trên CẢ giai đoạn sinh trưởng
    LẪN bệnh phát hiện được (nếu có).
    """
    from google import genai
    import json as _json

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Chưa có GEMINI_API_KEY.")

    context_lines = [f"Cây trồng: {crop_display_name}"]
    if stage_name:
        context_lines.append(f"Giai đoạn sinh trưởng hiện tại: {stage_name}")
    if disease_names:
        context_lines.append(f"Dấu hiệu bệnh phát hiện được: {', '.join(disease_names)}")
    else:
        context_lines.append("Không phát hiện dấu hiệu bệnh nào trong ảnh.")
    context = "\n".join(context_lines)

    prompt = f"""Bạn là kỹ sư nông nghiệp. Dựa trên thông tin sau:

{context}

Hãy trả lời DUY NHẤT 1 JSON object hợp lệ (không markdown, không giải
thích thêm), đúng format:

{{"tom_tat": "1-2 câu tóm tắt tình trạng cây (kết hợp cả giai đoạn và bệnh nếu có)",
"hanh_dong_de_xuat": ["hành động cụ thể 1", "hành động cụ thể 2", "hành động cụ thể 3"],
"canh_bao_rui_ro": ["rủi ro cần chú ý 1", "rủi ro cần chú ý 2"]}}

Nếu có bệnh, hành động đề xuất cần ưu tiên xử lý bệnh trước, đồng thời
lưu ý phù hợp với giai đoạn sinh trưởng hiện tại. Nội dung tiếng Việt,
thực tế, cụ thể, dễ áp dụng ngay."""

    client = genai.Client(api_key=api_key)
    response = _call_gemini_with_retry(
        client, model=model, contents=[prompt], max_retries=2, base_delay=2.0,
    )
    text = (response.text or "").strip().strip("`")
    if text.startswith("json"):
        text = text[4:].strip()

    data = _json.loads(text)
    if not isinstance(data, dict) or "tom_tat" not in data:
        raise ValueError("Gemini trả JSON sai cấu trúc.")
    return data
