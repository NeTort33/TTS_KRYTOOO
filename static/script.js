class TTSApp {
    constructor() {
        this.currentAudioFile = null;
        this.modelLoaded = false;
        this.isGenerating = false;
        this.isUploading = false;

        this.initializeElements();
        this.bindEvents();
        this.initializeSliders();
        this.checkModelStatus();
    }

    initializeElements() {
        // Form elements
        this.modelUploadForm = document.getElementById('modelUploadForm');
        this.pthFileInput = document.getElementById('pthFile');
        this.indexFileInput = document.getElementById('indexFile');
        this.textInput = document.getElementById('textInput');

        // Buttons
        this.uploadBtn = document.getElementById('uploadBtn');
        this.generateBtn = document.getElementById('generateBtn');
        this.playBtn = document.getElementById('playBtn');
        this.downloadBtn = document.getElementById('downloadBtn');
        this.generateNewBtn = document.getElementById('generateNewBtn');

        // Progress bars
        this.uploadProgress = document.getElementById('uploadProgress');
        this.generateProgress = document.getElementById('generateProgress');

        // Status and result elements
        this.modelStatus = document.getElementById('modelStatus');
        this.audioResult = document.getElementById('audioResult');
        this.noAudio = document.getElementById('noAudio');
        this.audioPlayer = document.getElementById('audioPlayer');

        // Settings sliders
        this.pitchSlider = document.getElementById('pitchSlider');
        this.speedSlider = document.getElementById('speedSlider');
        this.volumeSlider = document.getElementById('volumeSlider');
        this.sampleRateSelect = document.getElementById('sampleRateSelect');

        // Value displays
        this.pitchValue = document.getElementById('pitchValue');
        this.speedValue = document.getElementById('speedValue');
        this.volumeValue = document.getElementById('volumeValue');
        this.charCount = document.getElementById('charCount');

        // Toast
        this.toast = document.getElementById('toast');
        this.toastBody = document.getElementById('toastBody');
        this.bsToast = new bootstrap.Toast(this.toast);
    }

    bindEvents() {
        // Upload form submission
        if (this.modelUploadForm) {
            this.modelUploadForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.uploadModel();
            });
        }

        // Upload button click
        if (this.uploadBtn) {
            this.uploadBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.uploadModel();
            });
        }

        // Generate button
        if (this.generateBtn) {
            this.generateBtn.addEventListener('click', () => this.generateSpeech());
        }

        // Play button
        if (this.playBtn) {
            this.playBtn.addEventListener('click', () => this.playAudio());
        }

        // Download button
        if (this.downloadBtn) {
            this.downloadBtn.addEventListener('click', () => this.downloadAudio());
        }

        // Generate new button
        if (this.generateNewBtn) {
            this.generateNewBtn.addEventListener('click', () => this.resetForNewGeneration());
        }

        // File input changes
        if (this.pthFileInput) {
            this.pthFileInput.addEventListener('change', () => this.validateInputs());
        }
        if (this.indexFileInput) {
            this.indexFileInput.addEventListener('change', () => this.validateInputs());
        }

        // Text input
        if (this.textInput) {
            this.textInput.addEventListener('input', () => this.updateCharCount());
            this.textInput.addEventListener('input', () => this.validateInputs());
        }
    }

    initializeSliders() {
        // Pitch slider
        if (this.pitchSlider && this.pitchValue) {
            this.pitchSlider.addEventListener('input', () => {
                this.pitchValue.textContent = parseFloat(this.pitchSlider.value).toFixed(1);
            });
        }

        // Speed slider
        if (this.speedSlider && this.speedValue) {
            this.speedSlider.addEventListener('input', () => {
                this.speedValue.textContent = parseFloat(this.speedSlider.value).toFixed(1);
            });
        }

        // Volume slider
        if (this.volumeSlider && this.volumeValue) {
            this.volumeSlider.addEventListener('input', () => {
                this.volumeValue.textContent = parseFloat(this.volumeSlider.value).toFixed(1);
            });
        }

        // Initialize values
        this.pitchValue.textContent = this.pitchSlider.value;
        this.speedValue.textContent = this.speedSlider.value;
        this.volumeValue.textContent = this.volumeSlider.value;
    }

    updateCharCount() {
        const length = this.textInput.value.length;
        this.charCount.textContent = length;

        // Update color based on length
        if (length > 800) {
            this.charCount.style.color = 'var(--bs-danger)';
        } else if (length > 600) {
            this.charCount.style.color = 'var(--bs-warning)';
        } else {
            this.charCount.style.color = 'var(--bs-primary)';
        }
    }

    validateInputs() {
        const hasText = this.textInput.value.trim().length > 0;
        const hasValidLength = this.textInput.value.length <= 1000;

        // Check model status before enabling generate button
        this.checkModelStatus().then(() => {
            this.generateBtn.disabled = !this.modelLoaded || !hasText || !hasValidLength || this.isGenerating;
        });
    }

    async checkModelStatus() {
        try {
            const response = await fetch('/model_status');
            if (response.ok) {
                const data = await response.json();
                this.modelLoaded = data.loaded && data.ready;
                
                if (this.modelLoaded) {
                    this.updateModelStatus('Модель загружена и готова к использованию', 'success');
                } else {
                    this.updateModelStatus('Загрузите модель для начала работы', 'warning');
                }
                
                this.validateInputs();
            } else {
                console.error('Model status check failed:', response.status);
                this.modelLoaded = false;
                this.updateModelStatus('Ошибка проверки статуса модели', 'error');
                this.validateInputs();
            }
        } catch (error) {
            console.error('Error checking model status:', error);
            this.modelLoaded = false;
            this.updateModelStatus('Нет соединения с сервером', 'error');
            this.validateInputs();
        }
    }

    updateModelStatus(message = null, type = null) {
        if (this.modelLoaded) {
            this.modelStatus.innerHTML = `
                <div class="alert alert-success model-loaded">
                    <i data-feather="check-circle" class="me-2"></i>
                    ${message || 'Модель загружена и готова к использованию'}
                </div>
            `;
        } else if (type === 'error') {
            this.modelStatus.innerHTML = `
                <div class="alert alert-danger model-error">
                    <i data-feather="x-circle" class="me-2"></i>
                    ${message || 'Ошибка загрузки модели'}
                </div>
            `;
        } else if (type === 'loading') {
            this.modelStatus.innerHTML = `
                <div class="alert alert-info">
                    <i data-feather="loader" class="me-2 loading"></i>
                    ${message || 'Загрузка модели...'}
                </div>
            `;
        } else {
            this.modelStatus.innerHTML = `
                <div class="alert alert-warning">
                    <i data-feather="alert-circle" class="me-2"></i>
                    ${message || 'Модель не загружена'}
                </div>
            `;
        }

        // Reinitialize feather icons
        feather.replace();
    }

    async uploadModel() {
        if (this.isUploading) return;

        const pthFile = this.pthFileInput?.files[0];
        const indexFile = this.indexFileInput?.files[0];

        if (!pthFile || !indexFile) {
            this.showToast('Выберите оба файла модели', 'error');
            return;
        }

        // Validate file extensions
        if (!pthFile.name.toLowerCase().endsWith('.pth')) {
            this.showToast('PTH файл должен иметь расширение .pth', 'error');
            return;
        }

        if (!indexFile.name.toLowerCase().endsWith('.index')) {
            this.showToast('Index файл должен иметь расширение .index', 'error');
            return;
        }

        // Check file sizes
        const maxSize = 500 * 1024 * 1024; // 500MB
        if (pthFile.size > maxSize) {
            this.showToast('PTH файл слишком большой (максимум 500MB)', 'error');
            return;
        }

        this.isUploading = true;
        this.uploadBtn.disabled = true;
        this.uploadProgress.style.display = 'block';
        this.updateModelStatus('Загрузка файлов модели...', 'loading');

        try {
            const formData = new FormData();
            formData.append('pth_file', pthFile);
            formData.append('index_file', indexFile);

            const response = await fetch('/upload_model', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok && data.success) {
                this.modelLoaded = true;
                this.updateModelStatus(data.message, 'success');
                this.showToast('Модель успешно загружена!', 'success');

                // Show model info if available
                if (data.model_info) {
                    const info = data.model_info;
                    this.showToast(`Размер модели: ${info.model_size}`, 'info');
                }

                // Clear file inputs to allow re-upload
                this.pthFileInput.value = '';
                this.indexFileInput.value = '';

                // Recheck model status to ensure UI is updated
                setTimeout(() => this.checkModelStatus(), 1000);
            } else {
                this.modelLoaded = false;
                this.updateModelStatus(data.error, 'error');
                this.showToast(data.error || 'Ошибка загрузки модели', 'error');
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.modelLoaded = false;
            this.updateModelStatus('Ошибка сети при загрузке модели', 'error');
            this.showToast('Ошибка сети при загрузке модели', 'error');
        } finally {
            this.isUploading = false;
            this.uploadBtn.disabled = false;
            this.uploadProgress.style.display = 'none';
            this.validateInputs();
        }
    }

    async generateSpeech() {
        if (this.isGenerating || !this.modelLoaded) return;

        const text = this.textInput.value.trim();
        if (!text) {
            this.showToast('Введите текст для генерации', 'error');
            return;
        }

        this.isGenerating = true;
        this.generateBtn.disabled = true;
        this.generateProgress.style.display = 'block';

        // Update button text
        const originalHtml = this.generateBtn.innerHTML;
        this.generateBtn.innerHTML = '<i data-feather="loader" class="me-2 loading"></i>Генерация...';
        feather.replace();

        try {
            const requestData = {
                text: text,
                pitch: parseFloat(this.pitchSlider.value),
                speed: parseFloat(this.speedSlider.value),
                volume: parseFloat(this.volumeSlider.value),
                sample_rate: parseInt(this.sampleRateSelect.value)
            };

            const response = await fetch('/generate_speech', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });

            const data = await response.json();

            if (response.ok && data.success) {
                this.currentAudioFile = data.filename;
                this.showAudioResult();
                this.showToast('Речь успешно сгенерирована!', 'success');
            } else {
                this.showToast(data.error || 'Ошибка генерации речи', 'error');
            }
        } catch (error) {
            console.error('Generation error:', error);
            this.showToast('Ошибка сети при генерации речи', 'error');
        } finally {
            this.isGenerating = false;
            this.generateBtn.disabled = false;
            this.generateBtn.innerHTML = originalHtml;
            this.generateProgress.style.display = 'none';
            this.validateInputs();
            feather.replace();
        }
    }

    showAudioResult() {
        if (!this.currentAudioFile) return;

        // Set audio source
        this.audioPlayer.src = `/play/${this.currentAudioFile}`;

        // Show result section
        this.audioResult.style.display = 'block';
        this.noAudio.style.display = 'none';

        // Scroll to result
        this.audioResult.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    playAudio() {
        if (this.audioPlayer.src) {
            this.audioPlayer.play().catch(error => {
                console.error('Play error:', error);
                this.showToast('Ошибка воспроизведения аудио', 'error');
            });
        }
    }

    downloadAudio() {
        if (this.currentAudioFile) {
            const link = document.createElement('a');
            link.href = `/download/${this.currentAudioFile}`;
            link.download = `tts_output_${new Date().getTime()}.wav`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            this.showToast('Загрузка начата', 'info');
        }
    }

    resetForNewGeneration() {
        this.currentAudioFile = null;
        this.audioResult.style.display = 'none';
        this.noAudio.style.display = 'block';
        this.audioPlayer.src = '';

        // Focus on text input
        this.textInput.focus();
    }

    showToast(message, type = 'info') {
        // Update toast content
        this.toastBody.textContent = message;

        // Update toast header icon and color based on type
        const header = this.toast.querySelector('.toast-header');
        const icon = header.querySelector('i[data-feather]');

        // Remove existing classes
        this.toast.classList.remove('text-bg-success', 'text-bg-danger', 'text-bg-warning', 'text-bg-info');

        switch (type) {
            case 'success':
                icon.setAttribute('data-feather', 'check-circle');
                this.toast.classList.add('text-bg-success');
                break;
            case 'error':
                icon.setAttribute('data-feather', 'x-circle');
                this.toast.classList.add('text-bg-danger');
                break;
            case 'warning':
                icon.setAttribute('data-feather', 'alert-triangle');
                this.toast.classList.add('text-bg-warning');
                break;
            default:
                icon.setAttribute('data-feather', 'info');
                this.toast.classList.add('text-bg-info');
        }

        feather.replace();
        this.bsToast.show();
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TTSApp();
});

// Handle file drag and drop
document.addEventListener('DOMContentLoaded', () => {
    const fileInputs = document.querySelectorAll('input[type="file"]');

    fileInputs.forEach(input => {
        const parent = input.closest('.mb-3');

        parent.addEventListener('dragover', (e) => {
            e.preventDefault();
            parent.classList.add('dragover');
        });

        parent.addEventListener('dragleave', (e) => {
            e.preventDefault();
            parent.classList.remove('dragover');
        });

        parent.addEventListener('drop', (e) => {
            e.preventDefault();
            parent.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                input.files = files;
                input.dispatchEvent(new Event('change'));
            }
        });
    });
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl+Enter to generate speech
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        const generateBtn = document.getElementById('generateBtn');
        if (!generateBtn.disabled) {
            generateBtn.click();
        }
    }

    // Escape to reset
    if (e.key === 'Escape') {
        const generateNewBtn = document.getElementById('generateNewBtn');
        if (generateNewBtn && !generateNewBtn.style.display === 'none') {
            generateNewBtn.click();
        }
    }
});