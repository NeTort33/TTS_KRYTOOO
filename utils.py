
import os
import torch
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def validate_model_files(pth_path, index_path):
    """
    Validate uploaded model files
    
    Args:
        pth_path: Path to .pth file
        index_path: Path to .index file
        
    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': False,
        'error': None,
        'model_size': None
    }
    
    try:
        # Check if files exist
        if not Path(pth_path).exists():
            result['error'] = 'Файл .pth не найден'
            return result
            
        if not Path(index_path).exists():
            result['error'] = 'Файл .index не найден'
            return result
        
        # Check file sizes
        pth_size = Path(pth_path).stat().st_size
        index_size = Path(index_path).stat().st_size
        
        if pth_size == 0:
            result['error'] = 'Файл .pth пустой'
            return result
            
        if pth_size > 500 * 1024 * 1024:  # 500MB limit
            result['error'] = 'Файл .pth слишком большой (максимум 500MB)'
            return result
        
        # Try to load .pth file to validate it's a valid PyTorch model
        try:
            device = torch.device('cpu')  # Use CPU for validation
            checkpoint = torch.load(pth_path, map_location=device)
            
            # Basic validation - check if it contains expected structures
            if not isinstance(checkpoint, (dict, torch.nn.Module)):
                result['error'] = 'Неверный формат файла .pth'
                return result
                
            result['model_size'] = format_file_size(pth_size)
            
        except Exception as e:
            result['error'] = f'Не удается загрузить файл .pth: {str(e)}'
            return result
        
        # Validate index file (basic check)
        try:
            with open(index_path, 'rb') as f:
                # Try to read first few bytes
                header = f.read(100)
                if len(header) == 0:
                    result['error'] = 'Файл .index пустой'
                    return result
        except Exception as e:
            result['error'] = f'Не удается прочитать файл .index: {str(e)}'
            return result
        
        result['valid'] = True
        return result
        
    except Exception as e:
        result['error'] = f'Ошибка валидации: {str(e)}'
        return result

def get_supported_audio_formats():
    """
    Get list of supported audio formats
    
    Returns:
        List of supported audio formats
    """
    return ['wav', 'mp3', 'flac', 'ogg']

def format_file_size(size_bytes):
    """
    Format file size in human readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def check_system_requirements():
    """
    Check system requirements for TTS generation
    
    Returns:
        Dictionary with system status
    """
    requirements = {
        'torch_available': False,
        'cuda_available': False,
        'memory_available': True,
        'disk_space_available': True
    }
    
    try:
        import torch
        requirements['torch_available'] = True
        requirements['cuda_available'] = torch.cuda.is_available()
        
        if requirements['cuda_available']:
            cuda_memory = torch.cuda.get_device_properties(0).total_memory
            requirements['cuda_memory_gb'] = cuda_memory / (1024**3)
        
    except ImportError:
        pass
    
    try:
        import psutil
        
        # Check RAM
        memory = psutil.virtual_memory()
        requirements['ram_total_gb'] = memory.total / (1024**3)
        requirements['ram_available_gb'] = memory.available / (1024**3)
        requirements['memory_available'] = memory.available > 1024**3  # At least 1GB
        
        # Check disk space
        disk = psutil.disk_usage('.')
        requirements['disk_total_gb'] = disk.total / (1024**3)
        requirements['disk_free_gb'] = disk.free / (1024**3)
        requirements['disk_space_available'] = disk.free > 1024**3  # At least 1GB
        
    except ImportError:
        pass
    
    return requirements

def cleanup_old_files(directory, max_age_hours=24):
    """
    Clean up old files from directory
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age of files to keep in hours
    """
    try:
        import time
        
        directory = Path(directory)
        if not directory.exists():
            return
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in directory.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Could not delete old file {file_path}: {e}")
                        
    except Exception as e:
        logger.error(f"Error cleaning up old files: {e}")

def ensure_directories_exist():
    """
    Ensure required directories exist
    """
    directories = ['uploads', 'output', 'static', 'templates']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Directory ensured: {directory}")

def validate_text_input(text, max_length=1000):
    """
    Validate text input for TTS generation
    
    Args:
        text: Input text string
        max_length: Maximum allowed length
        
    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': False,
        'error': None,
        'cleaned_text': None
    }
    
    try:
        if not text or not isinstance(text, str):
            result['error'] = 'Текст не может быть пустым'
            return result
        
        # Clean text
        cleaned_text = text.strip()
        
        if not cleaned_text:
            result['error'] = 'Текст не может быть пустым'
            return result
        
        if len(cleaned_text) > max_length:
            result['error'] = f'Текст слишком длинный (максимум {max_length} символов)'
            return result
        
        # Basic character validation
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-()[]{}"\'\n\t')
        allowed_chars.update('абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')
        
        invalid_chars = set(cleaned_text) - allowed_chars
        if invalid_chars:
            logger.warning(f"Text contains invalid characters: {invalid_chars}")
            # Remove invalid characters instead of rejecting
            cleaned_text = ''.join(c for c in cleaned_text if c in allowed_chars)
        
        result['valid'] = True
        result['cleaned_text'] = cleaned_text
        return result
        
    except Exception as e:
        result['error'] = f'Ошибка валидации текста: {str(e)}'
        return result

def get_audio_info(file_path):
    """
    Get information about audio file
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with audio information
    """
    try:
        import soundfile as sf
        
        info = sf.info(file_path)
        
        return {
            'duration': info.duration,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'format': info.format,
            'subtype': info.subtype,
            'frames': info.frames
        }
        
    except Exception as e:
        logger.error(f"Error getting audio info: {e}")
        return {}

def log_system_info():
    """Log system information for debugging"""
    try:
        requirements = check_system_requirements()
        
        logger.info("=== System Information ===")
        logger.info(f"PyTorch available: {requirements.get('torch_available', False)}")
        logger.info(f"CUDA available: {requirements.get('cuda_available', False)}")
        logger.info(f"RAM available: {requirements.get('ram_available_gb', 'Unknown')} GB")
        logger.info(f"Disk space available: {requirements.get('disk_free_gb', 'Unknown')} GB")
        
        if requirements.get('cuda_available'):
            logger.info(f"CUDA memory: {requirements.get('cuda_memory_gb', 'Unknown')} GB")
        
        # Log Python packages
        try:
            import torch
            logger.info(f"PyTorch version: {torch.__version__}")
        except ImportError:
            logger.warning("PyTorch not available")
        
        try:
            import numpy
            logger.info(f"NumPy version: {numpy.__version__}")
        except ImportError:
            logger.warning("NumPy not available")
        
        try:
            import soundfile
            logger.info(f"SoundFile version: {soundfile.__version__}")
        except ImportError:
            logger.warning("SoundFile not available")
        
        try:
            import librosa
            logger.info(f"Librosa version: {librosa.__version__}")
        except ImportError:
            logger.warning("Librosa not available")
        
        logger.info("=== End System Information ===")
        
    except Exception as e:
        logger.error(f"Error logging system info: {e}")
