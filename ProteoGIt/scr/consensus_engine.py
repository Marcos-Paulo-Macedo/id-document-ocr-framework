import cv2
import numpy as np
import easyocr
import tensorflow as tf

class ConsensusEngine:
    def __init__(self, model_path: str, class_indices: dict):
        self.reader = easyocr.Reader(['pt'])
        self.model = tf.keras.models.load_model(model_path)
        self.labels = {v: k for k, v in class_indices.items()}

    def evaluate_document(self, processed_image: np.ndarray):
        """
        Aplica validação multicamada comparando o resultado OCR
        com a predição da rede neural de visão computacional.
        """
        h, w = processed_image.shape[:2]
        header_crop = processed_image[0:int(h * 0.35), :]
        
        # 1. Camada OCR por Palavras-chave
        text_lines = self.reader.readtext(header_crop, detail=0)
        extracted_text = " ".join(text_lines).upper()
        
        ocr_vote = "UNKNOWN"
        if any(keyword in extracted_text for keyword in ["HABILITACAO", "TRANSITO", "MOTORISTA", "CNH"]):
            ocr_vote = "CNH"
        elif any(keyword in extracted_text for keyword in ["REGISTRO", "IDENTIDADE", "CIVIL", "RG"]):
            ocr_vote = "RG"

        # 2. Camada de Classificação CNN
        img_resized = cv2.resize(processed_image, (240, 240))
        if len(img_resized.shape) == 2:
            img_3ch = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        else:
            img_3ch = img_resized
            
        img_normalized = img_3ch / 255.0
        input_tensor = np.expand_dims(img_normalized, axis=0)

        preds = self.model.predict(input_tensor, verbose=0)
        class_id = np.argmax(preds)
        cnn_vote = self.labels[class_id].upper()
        confidence = float(np.max(preds))

        # 3. Regra de Decisão/Consenso
        final_verdict = ocr_vote if ocr_vote != "UNKNOWN" else cnn_vote

        return {
            "final_verdict": final_verdict,
            "ocr_vote": ocr_vote,
            "cnn_vote": cnn_vote,
            "confidence": f"{confidence * 100:.1f}%",
            "extracted_text": extracted_text
        }