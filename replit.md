# Overview

This is a Text-to-Speech (TTS) Flask web application that allows users to upload custom voice models and generate speech from text input. The system uses PyTorch-based models with `.pth` and `.index` files to create personalized voice synthesis.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: HTML/CSS/JavaScript with Bootstrap 5 for UI components
- **Interface Design**: Responsive web interface with model upload, text input, and audio generation controls
- **User Experience**: Real-time progress tracking, audio playback controls, and parameter adjustment sliders
- **Static Assets**: Organized in `/static/` directory with separate CSS and JavaScript files

### Backend Architecture
- **Framework**: Flask web framework with Python 3.11
- **Model Architecture**: Modular design with separate components for model loading, TTS engine, and utilities
- **Audio Processing**: Integration with PyTorch, librosa, and soundfile for audio generation and manipulation
- **File Handling**: Secure file upload with validation for `.pth` and `.index` model files

## Key Components

### Core Modules
1. **app.py**: Main Flask application with route handlers for model upload and TTS generation
2. **model_loader.py**: PyTorch model loading and validation for custom voice models
3. **tts_engine.py**: Text-to-speech generation engine with preprocessing and audio synthesis
4. **utils.py**: Utility functions for model validation and audio format support

### Frontend Components
1. **templates/index.html**: Main web interface with model upload, text input, and generation controls
2. **static/script.js**: JavaScript application class handling user interactions and API communication
3. **static/style.css**: Custom styling with CSS variables and responsive design

### File Structure
- `/uploads/`: Directory for uploaded model files (`.pth` and `.index`)
- `/output/`: Directory for generated audio files
- `/templates/`: HTML templates for the web interface
- `/static/`: Static assets (CSS, JavaScript)

## Data Flow

1. **Model Upload**: Users upload `.pth` and `.index` files through the web interface
2. **Model Validation**: Backend validates file formats, sizes, and PyTorch model structure
3. **Model Loading**: ModelLoader class loads and prepares the voice model for inference
4. **Text Processing**: Input text is preprocessed and converted to model-compatible format
5. **Audio Generation**: TTS engine generates audio using the loaded model and user parameters
6. **Audio Output**: Generated audio is saved to `/output/` and served to the user

## External Dependencies

### Python Packages
- **flask**: Web framework for HTTP handling and routing
- **torch**: PyTorch for deep learning model inference
- **librosa**: Audio processing and feature extraction
- **soundfile**: Audio file I/O operations
- **numpy**: Numerical computing for audio data manipulation
- **werkzeug**: WSGI utilities for secure file handling

### System Dependencies
- **ffmpeg-full**: Audio/video processing capabilities
- **libsndfile**: Audio file format support

### Frontend Dependencies
- **Bootstrap 5**: CSS framework for responsive UI components
- **Feather Icons**: Icon set for user interface elements

## Deployment Strategy

### Development Environment
- **Platform**: Replit with Nix package management
- **Python Version**: 3.11 with uv package manager
- **Ports**: Flask development server on port 5000
- **Auto-installation**: Dependencies installed automatically on startup

### Configuration
- **File Limits**: 500MB maximum upload size for model files
- **Model Limits**: 2GB maximum size for `.pth` files
- **Security**: Secret key configuration for Flask sessions
- **Device Support**: Automatic CPU/GPU detection for PyTorch inference

### Production Considerations
- Environment variables for sensitive configuration
- File cleanup strategies for uploaded and generated files
- Error handling and logging for model loading failures
- Audio format optimization for web delivery

The application follows a modular architecture that separates concerns between model handling, audio processing, and web interface, making it maintainable and extensible for different TTS model types.