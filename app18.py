import streamlit as st
import cv2
import numpy as np
from rembg import remove, new_session
from PIL import Image
import io
import zipfile

st.set_page_config(layout="wide")

# Função para estimar o tamanho do arquivo em MB com base nas dimensões
def estimate_file_size(img):
    h, w = img.shape[:2]
    return (w * h * 3) / (1024 * 1024)

# Função para redimensionar imagem para corresponder a um tamanho alvo em MB
def resize_for_preview(img, target_mb):
    h, w = img.shape[:2]
    current_mb = estimate_file_size(img)
    if current_mb <= target_mb:
        return img
    scale = np.sqrt(target_mb / current_mb)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Função para remover fundo e retornar máscara
@st.cache_data(show_spinner=False)
def get_person_mask(image: np.ndarray, model_name: str, mask_size: int, dilation: int):
    session = new_session(model_name=model_name)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgba = remove(rgb, session=session)
    alpha = rgba[:, :, 3]
    _, bin_mask = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(bin_mask, np.ones((max(1, dilation), max(1, dilation)), np.uint8), iterations=1)
    if mask_size <= 0:
        mask_float = dilated.astype(np.float32) / 255.0
    else:
        ksize = mask_size if mask_size % 2 == 1 else mask_size + 1
        mask_float = cv2.GaussianBlur(dilated, (ksize, ksize), 0).astype(np.float32) / 255.0
    return mask_float, bin_mask

# Função aprimorada para detectar defeitos, com reforço para riscos horizontais
def detect_defects(img, sensitivity, mask_float=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, max(5, sensitivity // 4), sensitivity / 1.5)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    _, sobel_bin = cv2.threshold(sobel, sensitivity, 255, cv2.THRESH_BINARY)
    _, sobely_bin = cv2.threshold(np.uint8(np.abs(sobely)), sensitivity / 1.2, 255, cv2.THRESH_BINARY)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horiz_lines = cv2.morphologyEx(sobely_bin, cv2.MORPH_OPEN, horiz_kernel)
    edges_combined = cv2.bitwise_or(edges, sobel_bin)
    edges_combined = cv2.bitwise_or(edges_combined, horiz_lines)
    edges_combined = cv2.dilate(edges_combined, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    if mask_float is not None:
        mask = (mask_float > 0.5).astype(np.uint8)
        edges_combined[mask == 1] = 0
    return edges_combined

# Função para reinserir a pessoa com suavização de borda
def reinsert_foreground(repaired_img, orig_img, mask_float):
    k = 21 if 21 % 2 == 1 else 22
    mask_smooth = cv2.GaussianBlur(mask_float, (k, k), 0)
    m3 = cv2.merge([mask_smooth] * 3)
    out = repaired_img.astype(np.float32) * (1 - m3) + orig_img.astype(np.float32) * m3
    return np.clip(out, 0, 255).astype(np.uint8), mask_smooth

# Centralização da pessoa
def center_person(image):
    mask_float, bin_mask = get_person_mask(image, "isnet-general-use", 55, 25)
    h, w = image.shape[:2]
    moments = cv2.moments(bin_mask)
    if moments['m00'] == 0:
        return image
    cx = int(moments['m10'] / moments['m00'])
    offset = (w // 2) - cx
    M = np.float32([[1, 0, offset], [0, 1, 0]])
    shifted = cv2.warpAffine(image, M, (w, h), borderValue=(0, 0, 0))
    if offset != 0:
        region = image[:, :50] if offset > 0 else image[:, -50:]
        avg = np.mean(region, axis=1).astype(np.uint8)
        fill = np.repeat(avg[:, None, :], abs(offset), axis=1)
        if offset > 0:
            shifted[:, :offset] = fill
        else:
            shifted[:, w + offset:] = fill
    return shifted

# Função para processar uma imagem com as configurações fornecidas
def process_image(img, correction_strength, blur_type, k, d, sigma_color, sigma_space,
                  sat_option, sat_level, brightness_option, brightness_value,
                  person_brightness_option, person_brightness_value, tint_color, tint_strength,
                  centralizar, border_transparency, border_brightness, border_color,
                  border_thickness, apply_border_blur, blur_d, blur_sigma_color, blur_sigma_space):
    mask_float, bin_mask = get_person_mask(img, "isnet-general-use", 55, 25)
    defects = detect_defects(img, 20, mask_float)
    repaired = cv2.inpaint(img, defects, correction_strength, cv2.INPAINT_TELEA)

    # Ajuste de brilho da pessoa/objeto
    if person_brightness_option != "Nenhum":
        factor = (1 - person_brightness_value/100) if person_brightness_option == "Escurecer" else (1 + person_brightness_value/100)
        subj = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    else:
        subj = img.copy()
    subj, mask_smooth = reinsert_foreground(repaired, subj, mask_float)

    # Prepara background
    if blur_type == "Normal":
        bg = cv2.blur(subj, (k, k))
    elif blur_type == "Bilateral":
        bg = cv2.bilateralFilter(subj, d, sigma_color, sigma_space)
    else:
        bg = subj.copy()

    # Aplica saturação
    if sat_option != "Nenhum":
        hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV).astype(np.float32)
        if sat_option == "Remover Totalmente":
            hsv[:, :, 1] = 0
        else:
            hsv[:, :, 1] *= (1 - sat_level / 100)
        bg = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Aplica brilho do fundo
    if brightness_option != "Nenhum":
        factor = (1 - brightness_value/100) if brightness_option == "Escurecer" else (1 + brightness_value/100)
        bg = np.clip(bg.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # Aplica tint
    if tint_strength > 0:
        r = int(tint_color[1:3], 16)
        g = int(tint_color[3:5], 16)
        b = int(tint_color[5:7], 16)
        overlay = np.full(bg.shape, (b, g, r), dtype=np.uint8)
        alpha = tint_strength / 100.0
        bg = cv2.addWeighted(bg, 1 - alpha, overlay, alpha, 0)

    # Recombina sujeito + fundo
    m3 = cv2.merge([mask_smooth] * 3)
    final = (subj.astype(np.float32) * m3 + bg.astype(np.float32) * (1 - m3)).astype(np.uint8)

    # Borda ajustável (quando tint ou brilho)
    if tint_strength > 0 or brightness_option != "Nenhum":
        strict_mask_float, _ = get_person_mask(img, "isnet-general-use", 0, 0)
        strict_mask = strict_mask_float > 0.5
        transition = cv2.dilate(strict_mask.astype(np.uint8),
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_thickness*2+1, border_thickness*2+1)),
                                iterations=1).astype(bool) & ~strict_mask
        # cor da borda
        r = int(border_color[1:3], 16); g = int(border_color[3:5], 16); b = int(border_color[5:7], 16)
        bc = np.array([b, g, r], np.float32) * (1 + border_brightness/100)
        alpha = border_transparency
        for c in range(3):
            final[:, :, c][transition] = (alpha*bc[c] + (1-alpha)*final[:, :, c][transition]).astype(np.uint8)
        if apply_border_blur:
            blurred = cv2.bilateralFilter(final, blur_d, blur_sigma_color, blur_sigma_space)
            final[transition] = blurred[transition]

    # Centraliza sujeito
    if centralizar == "Sim":
        final = center_person(final)
    return final

# Interface Streamlit
st.title("Removedor de Riscos e Sujeiras - 4 Etapas + Borda Ajustável e Blur")

# Qualidade da pré-visualização
preview_quality = st.selectbox("Qualidade da Pré-visualização:", [
    "Resolução Baixa (preview rápido, pobreza de detalhe)",
    "Resolução Média (preview mais demorado, detalhes razoáveis)",
    "Resolução Original (preview pode ser muito lento, detalhes reais)"
], index=1)

# Upload de múltiplas imagens
files = st.file_uploader("Envie imagens", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if files:
    images, preview_images = [], []
    for file in files:
        if not file.type.startswith("image/"):
            st.error(f"⚠️ Arquivo {file.name} inválido. Use JPG, JPEG ou PNG.")
            continue
        data = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        images.append((file.name, img))
        if preview_quality == "Resolução Original (preview pode ser muito lento, detalhes reais)" or estimate_file_size(img) <= 12:
            preview_images.append((file.name, img))
        else:
            target = 4 if preview_quality.startswith("Resolução Baixa") else 12
            preview_images.append((file.name, resize_for_preview(img, target)))
    if not images:
        st.stop()

    # Seleção de imagem
    selected = st.session_state.get("selected_image", images[0][0])
    idx = next(i for i,(n,_) in enumerate(images) if n==selected)
    cols = st.columns(min(len(images),5))
    for i,(n,pi) in enumerate(preview_images):
        with cols[i%5]:
            if st.button(f"{i+1}", key=f"thumb{i}"):
                st.session_state["selected_image"] = n
                selected, idx = n, i
            st.image(cv2.cvtColor(pi,cv2.COLOR_BGR2RGB), width=100)

    img = preview_images[idx][1]
    etapa = st.radio("Selecione a etapa:", [
        "1. Ajustar máscara da pessoa",
        "2. Detecção de ruídos",
        "3. Resultado final"
    ])

    if etapa == "1. Ajustar máscara da pessoa":
        col1, col2 = st.columns(2)
        with st.sidebar:
            model = st.selectbox("Tipo de Detecção:", [
                "Detectar pessoa + objeto (isnet-general-use)",
                "Detectar apenas pessoa (u2net_human_seg)",
                "Alta qualidade (u2net)",
                "Modelo leve (u2netp)",
                "Pessoa contorno mínimo (u2net_human_seg)"
            ])
            map_m = {"Detectar pessoa + objeto (isnet-general-use)":"isnet-general-use",
                     "Detectar apenas pessoa (u2net_human_seg)":"u2net_human_seg",
                     "Alta qualidade (u2net)":"u2net",
                     "Modelo leve (u2netp)":"u2netp",
                     "Pessoa contorno mínimo (u2net_human_seg)":"u2net_human_seg"}
            dilation = st.slider("Largura da borda da máscara",0,50,25)
            blur_sz = st.slider("Tolerância da máscara",0,101,55,2)
            if model.endswith("contorno mínimo"): dilation, blur_sz = 0, 0
        mf,_ = get_person_mask(img, map_m[model], blur_sz, dilation)
        col1.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        col2.image((mf*255).astype(np.uint8), caption="Máscara", use_container_width=True)

    elif etapa == "2. Detecção de ruídos":
        with st.sidebar:
            sens = st.slider("Sensibilidade de Detecção",5,100,20)
        mf,_ = get_person_mask(img, "isnet-general-use",55,25)
        defects = detect_defects(img, sens, mf)
        st.columns(2)[0].image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), caption="Original")
        st.columns(2)[1].image(defects, caption="Ruídos")

    else:
        with st.sidebar:
            corr = st.slider("Intensidade de Inpaint",1,30,10)
            bt = st.selectbox("Blur no Fundo",["Nenhum","Normal","Bilateral"])
            if bt=="Normal": k = st.slider("Kernel (ímpar)",1,51,15,2)
            elif bt=="Bilateral":
                d = st.slider("Diâmetro",1,25,9)
                sc = st.slider("Sigma Color",1,200,75)
                ss = st.slider("Sigma Space",1,200,75)
            sat = st.selectbox("Saturação do Fundo",["Nenhum","Remover Totalmente","Gradual"])
            lv = st.slider("Nível (%)",0,100,50) if sat=="Gradual" else 50
            brt = st.selectbox("Brilho do Fundo",["Nenhum","Escurecer","Clarear"])
            bv = st.slider("Intensidade (%)",0,100,30) if brt!="Nenhum" else 30
            pbrt = st.selectbox("Brilho da Pessoa",["Nenhum","Escurecer","Clarear"])
            pbv = st.slider("Intensidade (%)",0,100,30) if pbrt!="Nenhum" else 30
            tint = st.color_picker("Cor de Tint","#FFFFFF")
            ts = st.slider("Força do Tint (%)",0,100,0)
            # Novas opções de borda
            bt_th = st.slider("Espessura da Borda",1,50,5) if ts>0 or brt!="Nenhum" else 5
            apply_blur = st.checkbox("Blur bilateral na borda",True) if ts>0 or brt!="Nenhum" else False
            bd = st.slider("Diâmetro Blur Borda",1,25,9) if apply_blur else 9
            bc = st.slider("SigmaColor Blur Borda",1,200,75) if apply_blur else 75
            bs = st.slider("SigmaSpace Blur Borda",1,200,75) if apply_blur else 75

        final = process_image(
            img, corr, bt,
            k if bt=="Normal" else 15,
            d if bt=="Bilateral" else 9,
            sc if bt=="Bilateral" else 75,
            ss if bt=="Bilateral" else 75,
            sat, lv, brt, bv,
            pbrt, pbv, tint, ts,
            "Sim" if st.radio("Centralizar?",["Não","Sim"])=="Sim" else "Não",
            st.slider("Transparência da Borda",0.0,1.0,0.3) if ts>0 or brt!="Nenhum" else 0.3,
            st.slider("Brilho da Borda (%)",-50,50,0) if ts>0 or brt!="Nenhum" else 0,
            st.color_picker("Cor da Borda","#FFFFFF") if ts>0 or brt!="Nenhum" else "#FFFFFF",
            bt_th, apply_blur, bd, bc, bs
        )

        cols = st.columns(2)
        cols[0].image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), caption="Original")
        cols[1].image(cv2.cvtColor(final,cv2.COLOR_BGR2RGB), caption="Final")

        if st.button("Processar e Baixar Todas as Imagens"):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as zf:
                for i,(n,full) in enumerate(images):
                    out = process_image(
                        full, corr, bt,
                        k if bt=="Normal" else 15,
                        d if bt=="Bilateral" else 9,
                        sc if bt=="Bilateral" else 75,
                        ss if bt=="Bilateral" else 75,
                        sat, lv, brt, bv,
                        pbrt, pbv, tint, ts,
                        "Sim",
                        0.3, 0, "#FFFFFF",
                        bt_th, apply_blur, bd, bc, bs
                    )
                    _, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY,100])
                    zf.writestr(f"resultado_{i+1}_{n}", buf.tobytes())
            buffer.seek(0)
            st.download_button("📥 Baixar ZIP", data=buffer, file_name="resultados.zip", mime="application/zip")
