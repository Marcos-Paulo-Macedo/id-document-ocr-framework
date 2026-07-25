import os
import json
import cv2
import shutil
from flask import Flask, render_template, jsonify, request, send_from_directory
from image_pipeline import detect_and_crop_document, apply_adaptive_filters
from consensus_engine import ConsensusEngine

app = Flask(__name__)

# Caminhos Relativos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE_DIR, 'samples')
RETRAIN_DIR = os.path.join(BASE_DIR, 'retrain_queue')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'document_classifier_v11.h5')
JSON_PATH = os.path.join(BASE_DIR, 'config', 'class_indices.json')

# Inicialização
engine = None
if os.path.exists(MODEL_PATH) and os.path.exists(JSON_PATH):
    with open(JSON_PATH, 'r') as f:
        class_indices = json.load(f)
    engine = ConsensusEngine(MODEL_PATH, class_indices)

@app.route('/')
def index():
    return jsonify({"status": "running", "service": "ID Document OCR Framework API"})

@app.route('/files', methods=['GET'])
def list_files():
    if not os.path.exists(TEST_DIR):
        return jsonify([])
    files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    return jsonify(files)

@app.route('/validate', methods=['GET'])
def validate_file():
    filename = request.args.get('filename')
    block_size = int(request.args.get('block', 11))
    c_val = int(request.args.get('c', 2))

    filepath = os.path.join(TEST_DIR, filename)
    if not os.path.exists(filepath) or engine is None:
        return jsonify({"error": "Arquivo não encontrado ou modelo não carregado."}), 400

    img_orig = cv2.imread(filepath)
    img_crop = detect_and_crop_document(img_orig)
    img_bw = apply_adaptive_filters(img_crop, block_size, c_val)

    result = engine.evaluate_document(img_bw)
    return jsonify(result)

@app.route('/feedback', methods=['POST'])
def save_feedback():
    data = request.json
    source = os.path.join(TEST_DIR, data['filename'])
    target_dir = os.path.join(RETRAIN_DIR, data['corrected_class'])
    
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(source, os.path.join(target_dir, data['filename']))
    
    return jsonify({"status": "success", "message": f"Enviado para {data['corrected_class']}"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)