import os
import glob
import re
import gradio as gr
from ultralytics import YOLO
import google.generativeai as genai
from PIL import Image
 
# ==========================================
# 1. TỰ ĐỘNG NẠP FILE WEIGHTS YOLO
# ==========================================
def find_weight_path(pattern):
    matches = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    return matches[0] if matches else None
 
MODEL_PATHS = {
    "lua_stage": find_weight_path("*lua-lua/**/best*.pt"),
    "xoai_stage": find_weight_path("*best*(xoai)*.pt"),
    "lua_disease": find_weight_path("*disease_v1*/**/best.pt")
}
 
LOADED_MODELS = {}
print("⏳ Đang nạp hệ thống mô hình YOLO...")
for k, path in MODEL_PATHS.items():
    if path and os.path.exists(path):
        try:
            LOADED_MODELS[k] = YOLO(path)
            print(f"  ✅ Nạp thành công [{k}]: {path}")
        except Exception as e:
            print(f"  ❌ Lỗi nạp {k}: {e}")
 
# ==========================================
# 2. HÀM TÌM MODEL GEMINI ĐANG HOẠT ĐỘNG
# ==========================================
def get_working_gemini_model(api_key):
    genai.configure(api_key=api_key.strip())
 
    # Model cũ (1.5/2.0/2.5) đã hoặc sắp bị Google ngừng hỗ trợ tính đến
    # thời điểm hiện tại -> ưu tiên bản 3.x còn đang hoạt động trước,
    # giữ vài bản cũ ở cuối chỉ để dự phòng nếu tài khoản có quyền truy
    # cập đặc biệt.
    preferred_models = [
        'gemini-3.5-flash-lite',   # 500 lượt/ngày — dùng để code/test
        'gemini-3.5-flash',
        'gemini-3.6-flash',
        'gemini-2.0-flash',
        'gemini-1.5-flash',
    ]
    try:
        available = [
            m.name.replace('models/', '')
            for m in genai.list_models()
            if 'generateContent' in m.supported_generation_methods
        ]
        for pref in preferred_models:
            if pref in available:
                return genai.GenerativeModel(pref), pref
        if available:
            return genai.GenerativeModel(available[0]), available[0]
    except Exception as e:
        print(f"⚠️ Không liệt kê được danh sách model Gemini: {e}")
 
    # Mặc định dùng bản đang hoạt động thay vì model cũ có thể đã bị gỡ
    return genai.GenerativeModel('gemini-3.5-flash'), 'gemini-3.5-flash'
 
# ==========================================
# 3. HÀM XÁC ĐỊNH TÊN CÂY TRỒNG
# ==========================================
def get_clean_plant_name(model, pil_img):
    prompt = """
Đây là cây gì trong hình? 
Hãy trả lời duy nhất tên loại cây bằng TIẾNG VIỆT (Ví dụ: Cây Ớt, Cây Mít, Cây Chôm Chôm, Cây Lúa, Cây Xoài).
BẮT BUỘC: Không viết câu dài, không giải thích, không viết tiếng Anh, không ghi suy luận.
"""
    try:
        res = model.generate_content([pil_img, prompt])
        if res and res.text:
            text = res.text.strip()
            # Lọc sạch ký tự markdown (*, #, _, -)
            clean_text = re.sub(r'[*#_`~]', '', text)
            lines = [l.strip() for l in clean_text.split('\n') if l.strip() and not l.lower().startswith('thinking')]
            if lines:
                name = lines[0]
                # Lọc bỏ các tiền tố thừa
                name = re.sub(r'^(tên cây|đây là cây|đây là|cây trồng|loại cây):\s*', '', name, flags=re.IGNORECASE).strip()
                if not name.lower().startswith('cây'):
                    name = f"Cây {name}"
                return name
        # Gemini trả lời rỗng (không exception, nhưng cũng không có text)
        print("⚠️ Gemini trả lời rỗng khi nhận diện tên cây.")
        return "Cây chưa xác định (Gemini trả lời rỗng)"
    except Exception as e:
        # In lỗi RA CẢ giao diện thay vì chỉ console, để không còn phải
        # đoán mò khi gặp lỗi như lần trước.
        print(f"Lỗi đọc tên cây: {e}")
        return f"Cây chưa xác định (lỗi Gemini: {e})"
 
# ==========================================
# 4. HÀM XỬ LÝ CHÍNH
# ==========================================
def process_analysis(pil_img, api_key):
    if pil_img is None:
        return None, "❌ Vui lòng tải ảnh lên!"
 
    if not api_key or len(api_key.strip()) < 10:
        return pil_img, "⚠️ **CHƯA BẬT GEMINI:** Vui lòng nhập Gemini API Key hợp lệ vào ô bên trái!"
 
    try:
        # BƯỚC 1: Kết nối Gemini
        g_model, model_name = get_working_gemini_model(api_key)
 
        # BƯỚC 2: Nhận diện chính xác tên cây
        plant_name = get_clean_plant_name(g_model, pil_img)
        p_lower = plant_name.lower()
 
        # Nếu chính bước nhận diện tên đã lỗi/rỗng -> dừng sớm, hiện rõ
        # nguyên nhân thay vì đi tiếp và báo nhầm "chưa nằm trong dataset".
        if plant_name.startswith("Cây chưa xác định ("):
            return pil_img, f"❌ **Không nhận diện được cây:** {plant_name}\n\nKiểm tra lại API key hoặc thử ảnh khác."
 
        # Kiểm tra cây có thuộc Lúa hoặc Xoài hay không
        is_supported = any(k in p_lower for k in ["lúa", "xoài", "lua", "xoai"])
 
        # BƯỚC 3: NẾU LÀ CÂY KHÁC (Ớt, Mít, Chôm Chôm, Bưởi...) -> TẮT YOLO, KHÔNG VẼ KHUNG
        if not is_supported:
            msg = f"🔍 **Kết quả nhận diện:** Đây là **{plant_name}**.\n\n"
            msg += "⚠️ **THÔNG BÁO:** Hệ thống hiện tại chỉ hỗ trợ khoanh vùng và tư vấn chuyên sâu cho **Cây Lúa** và **Cây Xoài**.\n"
            msg += f"Do **{plant_name}** chưa nằm trong dữ liệu huấn luyện (Dataset YOLO) của hệ thống nên sẽ **không vẽ khung** và **không đưa ra bài tư vấn**."
 
            # Trả về ảnh gốc hoàn toàn KHÔNG VẼ KHUNG
            return pil_img, msg
 
        # BƯỚC 4: NẾU ĐÚNG CÂY LÚA HOẶC CÂY XOÀI -> BẬT YOLO KHOANH VÙNG & TƯ VẤN
        annotated_img = pil_img
        yolo_text = ""
        best_match = None
        highest_conf = -1.0
 
        for m_key, model in LOADED_MODELS.items():
            results = model.predict(pil_img, conf=0.35, imgsz=640)
            boxes = results[0].boxes
            if len(boxes) > 0:
                top_box = sorted(boxes, key=lambda x: float(x.conf[0]), reverse=True)[0]
                conf = float(top_box.conf[0])
                if conf > highest_conf:
                    highest_conf = conf
                    best_match = {
                        "m_key": m_key,
                        "results": results,
                        "box": top_box,
                        "conf": conf
                    }
 
        if best_match:
            m_key = best_match["m_key"]
            raw_label = LOADED_MODELS[m_key].names[int(best_match["box"].cls[0])]
            conf = best_match["conf"]
            annotated_img = best_match["results"][0].plot()[:, :, ::-1]
 
            LUA_STAGE_MAP = {'sinh_truong': '🌱 Giai đoạn Mạ / Đẻ nhánh', 'tro_bong': '🌾 Giai đoạn Trỗ bông', 'chin': '🌾 Giai đoạn Lúa chín'}
            XOAI_STAGE_MAP = {'sinh_truong': '🌿 Phát triển thân lá', 'ra_hoa': '🌸 Giai đoạn Ra hoa', 'phat_trien_qua': '🥭 Giai đoạn Quả lớn'}
            LUA_DISEASE_MAP = {'than_vang': '⚠️ Bệnh Thán vàng', 'dao_on': '⚠️ Bệnh Đạo ôn', 'chay_la': '⚠️ Bệnh Cháy lá', 'dom_van': '⚠️ Bệnh Đốm vằn', 'dom_nau': '⚠️ Bệnh Đốm nâu'}
 
            if m_key == "lua_stage": pretty = LUA_STAGE_MAP.get(raw_label, raw_label)
            elif m_key == "xoai_stage": pretty = XOAI_STAGE_MAP.get(raw_label, raw_label)
            else: pretty = LUA_DISEASE_MAP.get(raw_label, raw_label)
 
            yolo_text = f"🎯 **Khoanh vùng bởi YOLO:** {pretty} (Độ tin cậy: {conf:.1%})\n"
 
        # Sinh bài tư vấn chuyên sâu
        advice_prompt = f"""
Bạn là chuyên gia kỹ sư nông nghiệp. Hình ảnh này là {plant_name}.
{f'Kết quả YOLO nhận diện: {yolo_text}' if yolo_text else ''}
 
Hãy tư vấn chi tiết cho nông dân BẰNG TIẾNG VIỆT gồm:
1. 📌 **Đánh giá hiện trạng**: Tình trạng sức khỏe / giai đoạn sinh trưởng / bệnh hại.
2. 🧪 **Biện pháp kỹ thuật**: Loại phân bón, tưới nước hoặc thuốc BVTV.
3. ⚠️ **Lưu ý phòng ngừa**: Rủi ro tiếp theo.
"""
        adv_res = g_model.generate_content([pil_img, advice_prompt])
        advice_text = adv_res.text if adv_res and adv_res.text else "Không sinh được bản tư vấn."
 
        header = f"🔍 **Kết quả nhận diện:** Đây là **{plant_name}**.\n\n"
        if yolo_text:
            header += yolo_text + "==================================================\n\n"
 
        return annotated_img, header + f"🤖 **TƯ VẤN CHUYÊN SÂU TỪ GEMINI LLM:**\n\n" + advice_text
 
    except Exception as e:
        return pil_img, f"❌ **LỖI:** {str(e)}"
 
# ==========================================
# 5. GIAO DIỆN GRADIO
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(), title="YOLO + Gemini LLM") as demo:
    gr.Markdown("<h2 style='text-align: center;'>🌾 ỨNG DỤNG YOLO & GEMINI LLM NHẬN DIỆN - TƯ VẤN CÂY TRỒNG</h2>")
 
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Tải ảnh cây trồng lên")
            api_key_input = gr.Textbox(
                label="🔑 Gemini API Key (Bắt buộc)",
                placeholder="Dán API Key Gemini vào đây...",
                type="password"
            )
            btn_run = gr.Button("🚀 Phân Tích & Tư Vấn Chi Tiết", variant="primary")
 
        with gr.Column():
            img_output = gr.Image(label="Hình ảnh phân tích")
            txt_output = gr.Markdown(label="Kết quả từ AI")
 
    btn_run.click(
        fn=process_analysis,
        inputs=[img_input, api_key_input],
        outputs=[img_output, txt_output]
    )
 
demo.launch(share=True, debug=True)
