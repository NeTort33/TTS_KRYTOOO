import torch
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def validate_model_files(pth_path, index_path):
    """
    Validate model files (.pth and .index)
    
    Args:
        pth_path: Path to .pth file
        index_path: Path to .index file
        
    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': False,
        'error': None,
        'pth_valid': False,
        'index_valid': False,
        'model_size': None
    }
    
    try:
        # Check if files exist
        if not Path(pth_path).exists():
            result['error'] = f"PTH файл не найден: {pth_path}"
            return result
        
        if not Path(index_path).exists():
            result['error'] = f"Index файл не найден: {index_path}"
            return result
        
        # Check file sizes
        pth_size = Path(pth_path).stat().st_size
        index_size = Path(index_path).stat().st_size
        
        if pth_size == 0:
            result['error'] = "PTH файл пустой"
            return result
        
        if pth_size > 2 * 1024 * 1024 * 1024:  # 2GB limit
            result['error'] = "PTH файл слишком большой (>2GB)"
            return result
        
        # Validate PTH file
        try:
            checkpoint = torch.load(pth_path, map_location='cpu')
            result['pth_valid'] = True
            result['model_size'] = f"{pth_size / (1024*1024):.1f} MB"
            logger.info(f"PTH файл валиден, размер: {result['model_size']}")
        except Exception as e:
            result['error'] = f"Некорректный PTH файл: {str(e)}"
            return result
        
        # Validate index file (less strict)
        try:
            # Try to read index file in different formats
            index_valid = False
            
            # Try pickle
            try:
                import pickle
                with open(index_path, 'rb') as f:
                    pickle.load(f)
                index_valid = True
            except:
                pass
            
            # Try numpy
            if not index_valid:
                try:
                    np.load(index_path, allow_pickle=True)
                    index_valid = True
                except:
                    pass
            
            # Try torch
            if not index_valid:
                try:
                    torch.load(index_path, map_location='cpu')
                    index_valid = True
                except:
                    pass
            
            # Try text format
            if not index_valid:
                try:
                    with open(index_path, 'r', encoding='utf-8') as f:
                        f.read(100)  # Try to read first 100 chars
                    index_valid = True
                except:
                    pass
            
            result['index_valid'] = index_valid
            if not index_valid:
                logger.warning("Index файл может быть в неподдерживаемом формате")
            
        except Exception as e:
            logger.warning(f"Проблема с index файлом: {str(e)}")
            result['index_valid'] = False
        
        # Overall validation
        result['valid'] = result['pth_valid']  # Index is optional
        
        logger.info(f"Валидация завершена: PTH={result['pth_valid']}, Index={result['index_valid']}")
        return result
        
    except Exception as e:
        result['error'] = f"Ошибка валидации: {str(e)}"
        logger.error(result['error'])
        return result

def get_supported_audio_formats():
    """
    Get list of supported audio formats for export
    
    Returns:
        List of supported formats
    """
    return [
        {'format': 'wav', 'description': 'WAV (высокое качество)', 'extension': '.wav'},
        {'format': 'mp3', 'description': 'MP3 (сжатый)', 'extension': '.mp3'},
        {'format': 'flac', 'description': 'FLAC (без потерь)', 'extension': '.flac'},
        {'format': 'ogg', 'description': 'OGG Vorbis', 'extension': '.ogg'}
    ]

def sanitize_filename(filename):
    """
    Sanitize filename for safe file operations
    
    Args:
        filename: Input filename
        
    Returns:
        Sanitized filename
    """
    import re
    
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'\s+', '_', filename)
    filename = filename.strip('._')
    
    # Limit length
    if len(filename) > 100:
        name, ext = Path(filename).stem, Path(filename).suffix
        filename = name[:100-len(ext)] + ext
    
    return filename

def format_file_size(size_bytes):
    """
    Format file size in human readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
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
        requirements['disk_free_gb'] = disk.free / (1024**3)
        requirements['disk_space_available'] = disk.free > 5 * 1024**3  # At least 5GB
        
    except ImportError:
        pass
    
    return requirements

def log_model_info(model_data):
    """
    Log detailed information about loaded model
    
    Args:
        model_data: Model data dictionary
    """
    if not model_data:
        return
    
    logger.info("=== Model Information ===")
    logger.info(f"PTH Path: {model_data.get('pth_path', 'Unknown')}")
    logger.info(f"Index Path: {model_data.get('index_path', 'Unknown')}")
    
    model_info = model_data.get('model_info', {})
    logger.info(f"Model Size: {model_info.get('size_mb', 'Unknown')} MB")
    logger.info(f"Parameters: {model_info.get('parameters', 'Unknown')}")
    
    if 'epoch' in model_info:
        logger.info(f"Training Epoch: {model_info['epoch']}")
    
    if 'version' in model_info:
        logger.info(f"Model Version: {model_info['version']}")
    
    index_data = model_data.get('index_data', {})
    if index_data:
        logger.info(f"Index Data Type: {type(index_data).__name__}")
        if isinstance(index_data, dict):
            logger.info(f"Index Keys: {len(index_data)} entries")
        elif hasattr(index_data, 'shape'):
            logger.info(f"Index Shape: {index_data.shape}")
    
    logger.info("========================")
import os
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def validate_model_files(pth_path, index_path):
    """
    Validate uploaded model files
    
    Args:
        pth_path: Path to .pth file
        index_path: Path to .index file
        
    Returns:
        dict: Validation result with 'valid' boolean and optional 'error' message
    """
    try:
        # Check if files exist
        if not Path(pth_path).exists():
            return {'valid': False, 'error': 'PTH файл не найден'}
        
        if not Path(index_path).exists():
            return {'valid': False, 'error': 'Index файл не найден'}
        
        # Check file sizes
        pth_size = Path(pth_path).stat().st_size
        index_size = Path(index_path).stat().st_size
        
        if pth_size == 0:
            return {'valid': False, 'error': 'PTH файл пустой'}
        
        if pth_size > 2 * 1024 * 1024 * 1024:  # 2GB limit
            return {'valid': False, 'error': 'PTH файл слишком большой (максимум 2GB)'}
        
        # Try to load PTH file to validate it's a valid PyTorch model
        try:
            checkpoint = torch.load(pth_path, map_location='cpu')
            logger.info("PTH file validation successful")
        except Exception as e:
            return {'valid': False, 'error': f'Некорректный PTH файл: {str(e)}'}
        
        # Format file sizes for display
        pth_size_mb = pth_size / (1024 * 1024)
        index_size_mb = index_size / (1024 * 1024)
        
        if pth_size_mb > 1024:
            model_size = f"{pth_size_mb/1024:.1f} GB"
        else:
            model_size = f"{pth_size_mb:.1f} MB"
        
        return {
            'valid': True,
            'model_size': model_size,
            'pth_size_mb': pth_size_mb,
            'index_size_mb': index_size_mb
        }
        
    except Exception as e:
        logger.error(f"Error validating model files: {str(e)}")
        return {'valid': False, 'error': f'Ошибка валидации: {str(e)}'}

def get_supported_audio_formats():
    """
    Get list of supported audio formats
    
    Returns:
        list: Supported audio format extensions
    """
    return ['wav', 'mp3', 'flac', 'ogg']

def cleanup_old_files(directory, max_age_hours=24):
    """
    Clean up old files from a directory
    
    Args:
        directory: Directory path to clean
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
        logger.error(f"Error during cleanup: {str(e)}")
