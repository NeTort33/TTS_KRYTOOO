import os
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, flash
from werkzeug.utils import secure_filename
import torch
import logging

from tts_engine import TTSEngine
from model_loader import ModelLoader
from utils import validate_model_files, get_supported_audio_formats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')

# Configuration
UPLOAD_FOLDER = Path('uploads')
OUTPUT_FOLDER = Path('output')
ALLOWED_EXTENSIONS = {'pth', 'index'}

# Create directories if they don't exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Global variables
current_model = None
tts_engine = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    """Favicon handler"""
    return '', 204

@app.route('/upload_model', methods=['POST'])
def upload_model():
    """Upload and load voice model files"""
    global current_model, tts_engine
    
    try:
        logger.info("Received model upload request")
        
        # Check if files are present
        if 'pth_file' not in request.files or 'index_file' not in request.files:
            logger.error("Missing files in request")
            return jsonify({'error': 'Оба файла (.pth и .index) обязательны'}), 400
        
        pth_file = request.files['pth_file']
        index_file = request.files['index_file']
        
        # Validate files
        if pth_file.filename == '' or index_file.filename == '':
            return jsonify({'error': 'Файлы не выбраны'}), 400
        
        # Check filenames exist
        if not pth_file.filename or not index_file.filename:
            return jsonify({'error': 'Имена файлов не определены'}), 400
            
        if not (allowed_file(pth_file.filename) and allowed_file(index_file.filename)):
            return jsonify({'error': 'Неподдерживаемый формат файла'}), 400
        
        # Save files
        pth_filename = secure_filename(pth_file.filename)
        index_filename = secure_filename(index_file.filename)
        
        pth_path = UPLOAD_FOLDER / pth_filename
        index_path = UPLOAD_FOLDER / index_filename
        
        pth_file.save(pth_path)
        index_file.save(index_path)
        
        # Validate model files
        validation_result = validate_model_files(pth_path, index_path)
        if not validation_result['valid']:
            return jsonify({'error': f'Ошибка валидации моделей: {validation_result["error"]}'}), 400
        
        # Load model
        model_loader = ModelLoader()
        current_model = model_loader.load_model(pth_path, index_path)
        
        if current_model is None:
            return jsonify({'error': 'Не удалось загрузить модель'}), 500
        
        # Initialize TTS engine
        try:
            tts_engine = TTSEngine(current_model)
            logger.info(f"TTS Engine успешно инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации TTS Engine: {str(e)}")
            current_model = None
            return jsonify({'error': f'Ошибка инициализации TTS: {str(e)}'}), 500
        
        logger.info(f"Модель успешно загружена: {pth_filename}")
        return jsonify({
            'success': True, 
            'message': 'Модель успешно загружена и готова к использованию',
            'model_info': {
                'pth_file': pth_filename,
                'index_file': index_filename,
                'model_size': validation_result.get('model_size', 'Unknown')
            }
        })
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {str(e)}")
        return jsonify({'error': f'Ошибка загрузки модели: {str(e)}'}), 500

@app.route('/generate_speech', methods=['POST'])
def generate_speech():
    """Generate speech from text"""
    global tts_engine
    
    try:
        if tts_engine is None:
            return jsonify({'error': 'Модель не загружена. Загрузите модель сначала.'}), 400
        
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Текст не может быть пустым'}), 400
        
        if len(text) > 1000:
            return jsonify({'error': 'Текст слишком длинный (максимум 1000 символов)'}), 400
        
        # Generation parameters
        params = {
            'pitch': float(data.get('pitch', 0.0)),
            'speed': float(data.get('speed', 1.0)),
            'volume': float(data.get('volume', 1.0)),
            'sample_rate': int(data.get('sample_rate', 22050))
        }
        
        # Validate parameters
        if not (-12.0 <= params['pitch'] <= 12.0):
            return jsonify({'error': 'Высота тона должна быть между -12 и 12'}), 400
        
        if not (0.1 <= params['speed'] <= 3.0):
            return jsonify({'error': 'Скорость должна быть между 0.1 и 3.0'}), 400
        
        if not (0.1 <= params['volume'] <= 2.0):
            return jsonify({'error': 'Громкость должна быть между 0.1 и 2.0'}), 400
        
        # Generate unique filename
        output_filename = f"tts_output_{uuid.uuid4().hex[:8]}.wav"
        output_path = OUTPUT_FOLDER / output_filename
        
        # Generate speech
        success = tts_engine.generate_speech(text, output_path, **params)
        
        if not success:
            return jsonify({'error': 'Ошибка при генерации речи'}), 500
        
        logger.info(f"Речь успешно сгенерирована: {output_filename}")
        return jsonify({
            'success': True,
            'filename': output_filename,
            'message': 'Речь успешно сгенерирована'
        })
        
    except Exception as e:
        logger.error(f"Ошибка при генерации речи: {str(e)}")
        return jsonify({'error': f'Ошибка генерации: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download generated audio file"""
    try:
        file_path = OUTPUT_FOLDER / secure_filename(filename)
        
        if not file_path.exists():
            return jsonify({'error': 'Файл не найден'}), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        logger.error(f"Ошибка при скачивании файла: {str(e)}")
        return jsonify({'error': 'Ошибка при скачивании файла'}), 500

@app.route('/play/<filename>')
def play_file(filename):
    """Serve audio file for playing"""
    try:
        file_path = OUTPUT_FOLDER / secure_filename(filename)
        
        if not file_path.exists():
            return jsonify({'error': 'Файл не найден'}), 404
        
        return send_file(file_path, mimetype='audio/wav')
        
    except Exception as e:
        logger.error(f"Ошибка при воспроизведении файла: {str(e)}")
        return jsonify({'error': 'Ошибка при воспроизведении файла'}), 500

@app.route('/model_status')
def model_status():
    """Get current model status"""
    global current_model, tts_engine
    
    return jsonify({
        'loaded': current_model is not None,
        'ready': tts_engine is not None
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'Файл слишком большой (максимум 500MB)'}), 413

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle general exceptions"""
    logger.error(f"Необработанная ошибка: {str(e)}")
    return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

if __name__ == '__main__':
    print("Запуск TTS приложения...")
    print(f"PyTorch версия: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
