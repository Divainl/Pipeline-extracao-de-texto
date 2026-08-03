import cv2
import numpy as np
import pytesseract
import easyocr
import os
import glob

# ==============================================================================
# CONFIGURAÇÕES INICIAIS
# ==============================================================================
# Se estiver no Windows, mude o caminho abaixo para o seu executável do Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Inicializa o leitor do EasyOCR para a língua portuguesa (padrão Mercosul/Brasil)
reader_easyocr = easyocr.Reader(['pt'])

# Diretórios do pipeline
INPUT_DIR = "./placas_exercicio"       # Pasta onde estão as imagens fornecidas (1.JPG, 2.JPG, etc.)
OUTPUT_CROP_DIR = "./placas_extraidas"
OUTPUT_TXT_TESSERACT = "resultado_tesseract.txt"
OUTPUT_TXT_EASYOCR = "resultado_easyocr.txt"

# Cria as pastas de saída se não existirem
os.makedirs(OUTPUT_CROP_DIR, exist_ok=True)

# ==============================================================================
# ETAPA 1: SEGMENTAÇÃO E EXTRAÇÃO DA PLACA (OPENCV)
# ==============================================================================
def extrair_regiao_placa(img_path):
    """
    Carrega a imagem, aplica filtros morfológicos e busca o contorno retangular 
    correspondente à placa veicular.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Pré-processamento para destacar bordas da placa
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 200)
    
    # Encontra contornos na imagem modificada
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    screen_cnt = None
    for c in contours:
        # Aproxima o contorno
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        
        # Se o contorno tem 4 pontos, encontramos teoricamente o retângulo da placa
        if len(approx) == 4:
            screen_cnt = approx
            break
            
    # Se encontrou o contorno por aproximação, recorta. Caso contrário, faz fallback 
    # para a área central inferior da imagem (onde a maioria das placas está localizada)
    if screen_cnt is not None:
        x, y, w, h = cv2.boundingRect(screen_cnt)
        roi = img[y:y+h, x:x+w]
    else:
        # Fallback de segurança baseado na proporção padrão do dataset enviado
        h_img, w_img, _ = img.shape
        roi = img[int(h_img*0.4):int(h_img*0.85), int(w_img*0.2):int(w_img*0.8)]
        
    return roi

# ==============================================================================
# ETAPA 2: PRÉ-PROCESSAMENTO ESPECÍFICO PARA OCR
# ==============================================================================
def pre_processar_para_ocr(img_placa):
    """
    Prepara a imagem recortada da placa melhorando o contraste e binarizando 
    para facilitar a leitura dos caracteres pelos motores de OCR.
    """
    # Redimensiona para aumentar a resolução dos caracteres
    img_large = cv2.resize(img_placa, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)
    
    # Aplica limiarização adaptativa (Otsu) para binarizar em preto e branco
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

# ==============================================================================
# PIPELINE PRINCIPAL DE EXECUÇÃO
# ==============================================================================
def executar_pipeline():
    # Extensões comuns de imagem
    img_extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.png"]
    image_paths = []
    for ext in img_extensions:
        image_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        
    if not image_paths:
        print(f"Nenhuma imagem encontrada na pasta '{INPUT_DIR}'. Crie a pasta e coloque as imagens dentro.")
        return

    # Abre os arquivos de texto onde serão salvos os resultados
    with open(OUTPUT_TXT_TESSERACT, "w", encoding="utf-8") as f_tess, \
         open(OUTPUT_TXT_EASYOCR, "w", encoding="utf-8") as f_easy:
         
        f_tess.write("--- RESULTADOS TESSERACT OCR ---\n\n")
        f_easy.write("--- RESULTADOS EASYOCR ---\n\n")

        for img_path in sorted(image_paths):
            filename = os.path.basename(img_path)
            print(f"Processando: {filename}...")
            
            # Segmentação
            placa_cortada = extrair_regiao_placa(img_path)
            if placa_cortada is None or placa_cortada.size == 0:
                print(f"Erro ao segmentar imagem: {filename}")
                continue
                
            # Salva a imagem da placa extraída para verificação visual
            crop_path = os.path.join(OUTPUT_CROP_DIR, f"crop_{filename}")
            cv2.imwrite(crop_path, placa_cortada)
            
            # Pré-processamento focado em OCR
            placa_processada = pre_processar_para_ocr(placa_cortada)
            
            # --- Execução Tesseract OCR ---
            # Configuração psm 7: Tratar a imagem como uma única linha de texto.
            # whitelist: Restringe a leitura apenas para letras maiúsculas e números.
            custom_config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            txt_tesseract = pytesseract.image_to_string(placa_processada, config=custom_config)
            txt_tesseract = txt_tesseract.strip().replace("\n", "")
            
            # --- Execução EasyOCR ---
            # EasyOCR performa melhor mandando a imagem colorida recortada diretamente
            result_easy = reader_easyocr.readtext(placa_cortada, detail=0, paragraph=True)
            txt_easyocr = "".join(result_easy).strip().upper()
            # Limpa caracteres especiais indesejados que possam vir do EasyOCR
            txt_easyocr = "".join(c for c in txt_easyocr if c.isalnum())

            # Grava os resultados nos respectivos arquivos txt
            f_tess.write(f"Imagem: {filename} -> Placa: {txt_tesseract}\n")
            f_easy.write(f"Imagem: {filename} -> Placa: {txt_easyocr}\n")
            
            print(f"  > Tesseract: {txt_tesseract}")
            print(f"  > EasyOCR:   {txt_easyocr}\n")

    print(f"Processamento concluído com sucesso!")
    print(f"Resultados salvos em: {OUTPUT_TXT_TESSERACT} e {OUTPUT_TXT_EASYOCR}")

if __name__ == "__main__":
    executar_pipeline()