"""
demo/app.py — chạy thẳng trên crops_config.yaml.

Mới thêm: nhận diện GIỐNG CÂY TRỒNG tự động bằng Gemini (species_detector.py)
trước khi chạy YOLO giai đoạn sinh trưởng — chọn "🤖 Tự động (AI)" trong
dropdown. Nếu Gemini không xác định được hoặc chưa có GEMINI_API_KEY,
demo báo lỗi rõ ràng và gợi ý chọn tay thay vì im lặng dùng sai model.

Model giai đoạn dùng ở đây là 1 YOLO DUY NHẤT detect thẳng 3 class mỗi
cây (vd lúa: sinh_truong, tro_bong, chin) theo đúng crops_config.yaml.

Thêm cây mới: thêm 1 block trong crops_config.yaml -> "crops". Cả phần
nhận diện giống cây (Gemini) lẫn dropdown chọn tay đều tự cập nhật theo,
không cần sửa file này.

Chạy trong Colab:
    !pip install -q gradio ultralytics pyyaml google-genai
    import os
    os.environ["GEMINI_API_KEY"] = "..."
    !python demo/app.py
"""
from __future__ import annotations

import os

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

import species_detector

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "crops_config.yaml"

AUTO_DETECT = "__auto__"  # giá trị đặc biệt trong dropdown -> nhờ Gemini đoán cây

STAGE_LABELS_VI = {
    "sinh_truong": "Sinh trưởng",
    "tro_bong": "Trổ bông",
    "chin": "Chín",
}
# màu RGB (không phải BGR) vì vẽ bằng PIL
STAGE_COLORS_RGB = {
    "sinh_truong": (76, 175, 80),
    "tro_bong": (255, 193, 7),
    "chin": (244, 67, 54),
}

_VIETNAMESE_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

_model_cache: dict[str, object] = {}


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _VIETNAMESE_FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_crops_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model(model_path: str):
    """Cache theo đường dẫn weight để không load lại model mỗi lần bấm phân tích."""
    if model_path not in _model_cache:
        from ultralytics import YOLO
        full_path = PROJECT_ROOT / model_path
        if not full_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy weight tại {full_path}. "
                f"Kiểm tra lại model_path trong crops_config.yaml."
            )
        _model_cache[model_path] = YOLO(str(full_path))
    return _model_cache[model_path]


def load_advice(advice_path: str) -> dict:
    full_path = PROJECT_ROOT / advice_path
    if not full_path.exists():
        return {}
    return json.loads(full_path.read_text(encoding="utf-8"))


def _draw_boxes(image_bgr: np.ndarray, boxes: list[tuple]) -> np.ndarray:
    """boxes: list of (x1, y1, x2, y2, stage_name, confidence)"""
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _load_font(16)

    for x1, y1, x2, y2, stage, conf in boxes:
        color = STAGE_COLORS_RGB.get(stage, (200, 200, 200))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        label = f"{STAGE_LABELS_VI.get(stage, stage)} {conf:.0%}"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y1 = max(0, y1 - th - 8)
        draw.rectangle([x1, label_y1, x1 + tw + 8, label_y1 + th + 6], fill=color)
        draw.text((x1 + 4, label_y1 + 2), label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _make_distribution_chart(distribution: dict, empty_message: str | None = None):
    """
    Luôn trả về 1 matplotlib Figure hợp lệ, KHÔNG bao giờ trả None — một số
    phiên bản Gradio xử lý None cho gr.Plot() không nhất quán, có thể gây
    lỗi ở tầng Gradio (không phải lỗi logic của mình) mà không hiện rõ
    nguyên nhân trên giao diện.
    """
    fig, ax = plt.subplots(figsize=(4.5, 3))

    if not distribution:
        ax.text(0.5, 0.5, empty_message or "Chưa có dữ liệu",
                 ha="center", va="center", fontsize=11, color="#999999")
        ax.axis("off")
        fig.tight_layout()
        return fig

    labels = [STAGE_LABELS_VI.get(k, k) for k in distribution]
    values = list(distribution.values())
    color_map = {"Sinh trưởng": "#4CAF50", "Trổ bông": "#FFC107", "Chín": "#F44336"}
    colors = [color_map.get(l, "#999999") for l in labels]

    bars = ax.bar(labels, [v * 100 for v in values], color=colors)
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{v:.0%}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def _format_advice_markdown(advice: dict, dominant_stage: str | None) -> str:
    if not advice:
        if dominant_stage is None:
            return "_Không phát hiện được cây nào trong ảnh, thử ảnh khác rõ hơn._"
        return f"_Chưa có tài liệu tư vấn cho giai đoạn '{dominant_stage}' trong advice/lua.json._"

    lines = [f"### 📋 Tóm tắt\n{advice.get('tom_tat', '')}\n"]

    actions = advice.get("hanh_dong_de_xuat") or []
    if actions:
        lines.append("### ✅ Hành động đề xuất")
        lines += [f"- {a}" for a in actions]
        lines.append("")

    risks = advice.get("canh_bao_rui_ro") or []
    if risks:
        lines.append("### ⚠️ Cảnh báo rủi ro")
        lines += [f"- {r}" for r in risks]

    return "\n".join(lines)


def process(image: np.ndarray, crop_name: str):
    empty_chart = lambda msg: _make_distribution_chart({}, msg)  # noqa: E731

    if image is None:
        return None, empty_chart("Chưa có ảnh"), "Vui lòng tải lên một ảnh.", ""

    species_note = ""  # ghi chú hiển thị nếu có dùng nhận diện tự động

    # --- Chế độ tự động: nhờ Gemini đoán đây là cây gì trước ---
    if crop_name == AUTO_DETECT:
        try:
            pil_image = Image.fromarray(image)
            detected_key, raw_answer = species_detector.detect_crop_species(pil_image)
        except Exception as e:  # thiếu API key, lỗi mạng, v.v.
            return (
                None, empty_chart("Lỗi Gemini"),
                f"⚠️ Không gọi được Gemini để nhận diện cây: {e}\n\n"
                f"Chọn thủ công loại cây ở ô bên dưới rồi thử lại.",
                "",
            )

        if detected_key is None:
            return (
                None, empty_chart("Không xác định được cây"),
                f"⚠️ Gemini không xác định được đây là cây nào "
                f"(trả lời: '{raw_answer}'). Thử ảnh rõ hơn hoặc chọn thủ công.",
                "",
            )

        crop_name = detected_key
        config_preview = load_crops_config()

        if crop_name in config_preview["crops"]:
            display = config_preview["crops"][crop_name].get("display_name", crop_name)
        else:
            display = raw_answer

        species_note = f"🤖 **Gemini nhận diện: {display}**\n\n"

    config = load_crops_config()

    if crop_name not in config["crops"]:
        display = crop_name.replace("_", " ").title()
        summary = species_note + f"**Gemini nhận diện: {display}**\n\n"
        summary += "ℹ️ Cây này chưa có model nhận diện giai đoạn sinh trưởng.\n"
        summary += "Hiện chỉ hỗ trợ nhận diện loài."
        empty_chart_local = _make_distribution_chart({}, "Chưa có model giai đoạn")
        return None, empty_chart_local, summary, "_Chưa có tư vấn cho cây này._"

    crop_cfg = config["crops"][crop_name]

    model_path = crop_cfg.get("model_path") or ""
    if not model_path or not (PROJECT_ROOT / model_path).exists():
        display = crop_cfg.get("display_name", crop_name)
        summary = species_note + f"**Gemini nhận diện: {display}**\n\n"
        summary += "⚠️ Cây này chưa có model nhận diện giai đoạn sinh trưởng.\n"
        summary += "Hiện chỉ hỗ trợ nhận diện loài. Phần giai đoạn sẽ bổ sung sau khi train YOLO."
        empty_chart_local = _make_distribution_chart({}, "Chưa có model giai đoạn")
        return None, empty_chart_local, summary, "_Chưa có tư vấn giai đoạn cho cây này._"

    try:
        class_names = crop_cfg["classes"]  # {0: "sinh_truong", ...}
        model = get_model(crop_cfg["model_path"])
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        results = model.predict(image_bgr, verbose=False)[0]

        boxes_for_draw = []
        stage_counts: dict[str, int] = {}
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            stage = class_names.get(cls_id, str(cls_id))
            boxes_for_draw.append((x1, y1, x2, y2, stage, conf))
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

        annotated_bgr = _draw_boxes(image_bgr, boxes_for_draw)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        total = sum(stage_counts.values())
        distribution = {k: v / total for k, v in stage_counts.items()} if total else {}
        dominant_stage = max(distribution, key=distribution.get) if distribution else None
    except Exception as e:
        import traceback
        traceback.print_exc()  # in đầy đủ lỗi ra log Colab để debug
        return (
            None, empty_chart("Lỗi khi chạy model"),
            f"⚠️ Lỗi khi chạy model nhận diện giai đoạn: {e}",
            "",
        )

    chart = _make_distribution_chart(distribution, "Không phát hiện được cây nào")

    summary_md = species_note + f"**Số vùng phát hiện:** {total}\n"
    if dominant_stage:
        summary_md += f"**Giai đoạn chính:** {STAGE_LABELS_VI.get(dominant_stage, dominant_stage)}\n"
    else:
        summary_md += "⚠️ Không phát hiện được cây nào — thử ảnh rõ hơn hoặc gần hơn.\n"

    advice = {}
    advice_source_note = ""
    if dominant_stage:
        use_gemini_advice = os.environ.get("USE_GEMINI", "0") == "1"
        if use_gemini_advice:
            try:
                from species_detector import generate_stage_advice
                display_name = crop_cfg.get("display_name", crop_name)
                stage_label_vi = STAGE_LABELS_VI.get(dominant_stage, dominant_stage)
                advice = generate_stage_advice(display_name, stage_label_vi)
                advice_source_note = "_🤖 Tư vấn được Gemini sinh theo thời gian thực._\n\n"
            except Exception as e:
                print(f"[WARN] Gemini advice lỗi, dùng advice tĩnh: {e}")
                advice = {}

        if not advice:
            advice_data = load_advice(crop_cfg["advice_path"])
            advice = advice_data.get(dominant_stage, {})
            if advice:
                advice_source_note = "_📄 Tư vấn tham khảo từ tài liệu có sẵn._\n\n"

    advice_md = advice_source_note + _format_advice_markdown(advice, dominant_stage)

    return annotated_rgb, chart, summary_md, advice_md


def build_ui():
    import gradio as gr

    config = load_crops_config()
    crop_names = list(config["crops"].keys())
    display_names = {c: config["crops"][c].get("display_name", c) for c in crop_names}

    with gr.Blocks(title="Nhận diện & Tư vấn sinh trưởng cây trồng", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🌾 Nhận diện & Tư vấn sinh trưởng cây trồng\n"
            "Ứng dụng YOLO + Gemini — tự động nhận diện cây trồng và giai đoạn sinh trưởng."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Ảnh ruộng/cây", type="numpy")
                crop_input = gr.Dropdown(
                    choices=[("🤖 Tự động (AI nhận diện)", AUTO_DETECT)]
                    + [(display_names[c], c) for c in crop_names],
                    value=AUTO_DETECT,
                    label="Cây trồng",
                    info="Để 'Tự động' nếu muốn Gemini tự đoán, hoặc chọn tay nếu đã biết chắc.",
                )
                run_btn = gr.Button("🔍 Phân tích", variant="primary")
            with gr.Column(scale=1):
                image_output = gr.Image(label="Kết quả nhận diện")
                chart_output = gr.Plot(label="Phân bố giai đoạn")

        summary_output = gr.Markdown()
        advice_output = gr.Markdown()

        gr.Markdown(
            "---\n"
            "<sub>🔧 Phiên bản demo: v4 (không còn trả None cho biểu đồ, bọc lỗi khi chạy model, "
            "in traceback đầy đủ ra log Colab) "
            "— nếu bạn KHÔNG thấy dòng này, bạn đang xem bản demo cũ, cần dừng tiến trình cũ và mở link mới.</sub>"
        )

        run_btn.click(
            fn=process,
            inputs=[image_input, crop_input],
            outputs=[image_output, chart_output, summary_output, advice_output],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(share=True)
