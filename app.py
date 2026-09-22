import os
import glob
import re
import streamlit as st
from ultralytics import YOLO
import google.generativeai as genai
from PIL import Image

# ==========================================
# CẤU HÌNH TRANG STREAMLIT
# ==========================================
st.set_page_config(
    page_title="YOLO & Gemini - Nhận diện & Tư vấn Cây trồng",
    page_icon="🌾",
    layout="wide"
)

# ==========================================
# 1. TỰ ĐỘNG NẠP FILE WEIGHTS YOLO (TÌM TRONG ĐƯỜNG DẪN DỰ ÁN)
# ==========================================
@st.cache_resource
def load_all_yolo_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    def find_weight_path(pattern):
        # Tìm file weights trong thư mục dự án thay vì /kaggle/input
        matches = glob.glob(os.path.join(base_dir, "**", pattern), recursive=True)
        return matches[0] if matches else None

    model_paths = {
        "lua_stage": find_weight_path("*lua*.pt"),
        "xoai_stage": find_weight_path("*xoai*.pt"),
        "lua_disease": find_weight_path("*disease*.pt"),
        "mit_stage": find_weight_path("*mit*.pt") # Dự phòng khi bạn thêm weights Mít
    }

    loaded_models = {}
    for k, path in model_paths.items():
        if path and os.path.exists(path):
            try:
                loaded_models[k] = YOLO(path)
                print(f"✅ Nạp thành công [{k}]: {path}")
            except Exception as e:
                print(f"❌ Lỗi nạp {k}: {e}")
    return loaded_models

LOADED_MODELS = load_all_yolo_models()

# ==========================================
# 2. HÀM TÌM MODEL GEMINI ĐANG HOẠT ĐỘNG
# ==========================================
def get_working_gemini_model(api_key):
    genai.configure(api_key=api_key.strip())
    preferred_models = [
        'gemini-2.0-flash',
        'gemini-1.5-flash',
        'gemini-1.5-pro'
    ]
    try:
        available = [
            m.name.replace('models/', '')
            for m in genai.list_models()
            if 'generateContent' in m.supported_generation_methods
        ]
        for pref in preferred_models:
            if pref in available:
                return genai.GenerativeModel(pref)
        if available:
            return genai.GenerativeModel(available[0])
    except Exception as e:
        print(f"⚠️ Không liệt kê được danh sách model Gemini: {e}")

    return genai.GenerativeModel('gemini-1.5-flash')

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
            clean_text = re.sub(r'[*#_`~]', '', text)
            lines = [l.strip() for l in clean_text.split('\n') if l.strip()]
            if lines:
                name = lines[0]
                name = re.sub(r'^(tên cây|đây là cây|đây là|cây trồng|loại cây):\s*', '', name, flags=re.IGNORECASE).strip()
                if not name.lower().startswith('cây'):
                    name = f"Cây {name}"
                return name
        return "Cây chưa xác định (Gemini trả lời rỗng)"
    except Exception as e:
        return f"Cây chưa xác định (Lỗi Gemini: {e})"

# ==========================================
# 4. HÀM XỬ LÝ CHÍNH
# ==========================================
def process_analysis(pil_img, api_key):
    g_model = get_working_gemini_model(api_key)
    plant_name = get_clean_plant_name(g_model, pil_img)
    p_lower = plant_name.lower()

    if plant_name.startswith("Cây chưa xác định"):
        return pil_img, f"❌ **Không nhận diện được cây:** {plant_name}\n\nVui lòng kiểm tra lại API key hoặc chọn ảnh rõ hơn."

    # Đã bổ sung "mít" và "mit" vào danh sách hỗ trợ
    supported_crops = ["lúa", "xoài", "mít", "lua", "xoai", "mit"]
    is_supported = any(k in p_lower for k in supported_crops)

    if not is_supported:
        msg = f"🔍 **Kết quả nhận diện:** Đây là **{plant_name}**.\n\n"
        msg += "⚠️ **THÔNG BÁO:** Hệ thống hiện tại hỗ trợ khoanh vùng và tư vấn chuyên sâu cho **Cây Lúa**, **Cây Xoài** và **Cây Mít**.\n"
        msg += f"Do **{plant_name}** chưa nằm trong danh mục hỗ trợ nên hệ thống sẽ không thực hiện phân tích."
        return pil_img, msg

    # Nếu thuộc danh mục cây được hỗ trợ
    annotated_img = pil_img
    yolo_text = ""
    best_match = None
    highest_conf = -1.0

    # Chạy qua các mô hình YOLO đã nạp
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
        MIT_STAGE_MAP = {'cua_ga': '🌸 Nhú cựa gà', 'trai_non': '🍈 Trái non', 'bao_trai': '🛍️ Đã bao túi', 'trai_chin': '🥭 Trái chín / Thu hoạch'}

        if m_key == "lua_stage": pretty = LUA_STAGE_MAP.get(raw_label, raw_label)
        elif m_key == "xoai_stage": pretty = XOAI_STAGE_MAP.get(raw_label, raw_label)
        elif m_key == "mit_stage": pretty = MIT_STAGE_MAP.get(raw_label, raw_label)
        else: pretty = raw_label

        yolo_text = f"🎯 **Khoanh vùng bởi YOLO:** {pretty} (Độ tin cậy: {conf:.1%})\n"

    # Gửi prompt cho Gemini sinh bản tư vấn
    advice_prompt = f"""
Bạn là một chuyên gia kỹ sư nông nghiệp giàu kinh nghiệm. 
Hình ảnh phân tích là: {plant_name}.
{f'Kết quả YOLO nhận diện được: {yolo_text}' if yolo_text else 'Chưa phát hiện vùng bất thường bằng YOLO.'}

Hãy viết một bản tư vấn chi tiết BẰNG TIẾNG VIỆT giúp người dân chăm sóc cây trồng:
1. 📌 **Đánh giá hiện trạng**: Nhận xét về tình trạng sức khỏe, giai đoạn phát triển hoặc mầm bệnh hiện tại.
2. 🧪 **Biện pháp kỹ thuật**: Hướng dẫn cụ thể về loại phân bón, chế độ tưới nước hoặc thuốc BVTV phù hợp.
3. ⚠️ **Lưu ý phòng ngừa**: Các rủi ro sâu bệnh hại tiếp theo (ví dụ với Mít lưu ý Xơ đen, Xì mủ) và cách xử lý sớm.
"""
    adv_res = g_model.generate_content([pil_img, advice_prompt])
    advice_text = adv_res.text if adv_res and adv_res.text else "Không sinh được nội dung tư vấn."

    header = f"🔍 **Kết quả nhận diện:** Đây là **{plant_name}**.\n\n"
    if yolo_text:
        header += yolo_text + "\n---\n\n"

    return annotated_img, header + "🤖 **TƯ VẤN CHUYÊN SÂU TỪ GEMINI LLM:**\n\n" + advice_text

# ==========================================
# 5. GIAO DIỆN STREAMLIT CHUẨN
# ==========================================
st.title("🌾 ỨNG DỤNG YOLO & GEMINI LLM NHẬN DIỆN - TƯ VẤN CÂY TRỒNG")
st.write("Hệ thống nhận diện nông nghiệp thông minh hỗ trợ Cây Lúa, Cây Xoài và Cây Mít.")

col1, col2 = st.columns(2)

with col1:
    api_key_input = st.text_input("🔑 Gemini API Key (Bắt buộc)", type="password", placeholder="Dán Gemini API Key vào đây...")
    uploaded_file = st.file_uploader("📸 Tải ảnh cây trồng lên", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh bạn đã tải lên", use_container_width=True)

with col2:
    if uploaded_file and api_key_input:
        if st.button("🚀 Phân Tích & Tư Vấn Chi Tiết", type="primary", use_container_width=True):
            with st.spinner("⏳ Đang phân tích hình ảnh và khởi tạo bài tư vấn..."):
                try:
                    image = Image.open(uploaded_file)
                    res_img, res_text = process_analysis(image, api_key_input)
                    st.image(res_img, caption="Kết quả khoanh vùng / Phân tích", use_container_width=True)
                    st.markdown(res_text)
                except Exception as e:
                    st.error(f"❌ Có lỗi xảy ra trong quá trình xử lý: {e}")
    elif not api_key_input and uploaded_file:
        st.warning("⚠️ Vui lòng nhập Gemini API Key ở cột bên trái để bắt đầu phân tích.")
