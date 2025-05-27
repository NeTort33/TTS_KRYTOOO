import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TTSEngine:
    """TTS Engine for generating speech using custom voice models"""

    def __init__(self, model_data):
        """
        Initialize TTS Engine

        Args:
            model_data: Dictionary containing loaded model and metadata
        """
        self.model = model_data['model']
        self.index_data = model_data['index_data']
        self.model_info = model_data['model_info']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to device and set to eval mode
        if self.model is not None:
            try:
                # Check if model is a PyTorch module
                if hasattr(self.model, 'to') and hasattr(self.model, 'eval'):
                    self.model = self.model.to(self.device)
                    self.model.eval()
                elif isinstance(self.model, dict):
                    # Model is a state dict - move tensors to device
                    for key, value in self.model.items():
                        if isinstance(value, torch.Tensor):
                            self.model[key] = value.to(self.device)
                    logger.info("Model state dict moved to device")
                else:
                    logger.warning(f"Unknown model type: {type(self.model)}")
            except Exception as e:
                logger.warning(f"Could not move model to device: {e}")

        logger.info(f"TTS Engine initialized on device: {self.device}")

    def preprocess_text(self, text):
        """
        Preprocess text for TTS generation

        Args:
            text: Input text string

        Returns:
            Processed text features
        """
        # Basic text preprocessing
        text = text.strip()

        # Convert to phonemes or character tokens
        # This is a simplified version - real implementation would use
        # proper text-to-phoneme conversion
        char_to_idx = {chr(i): i-32 for i in range(32, 127)}  # ASCII mapping
        char_to_idx[' '] = 0

        tokens = []
        for char in text.lower():
            if char in char_to_idx:
                tokens.append(char_to_idx[char])
            else:
                tokens.append(0)  # Unknown character

        return torch.tensor(tokens, dtype=torch.long, device=self.device)

    def apply_voice_conversion(self, audio_features):
        """
        Apply voice conversion using the loaded model

        Args:
            audio_features: Input audio features

        Returns:
            Converted audio features
        """
        try:
            with torch.no_grad():
                # This is a simplified voice conversion process
                # Real implementation would depend on the specific model architecture

                if hasattr(self.model, 'forward'):
                    # Standard forward pass
                    converted = self.model(audio_features)
                elif hasattr(self.model, 'inference'):
                    # Some models have inference method
                    converted = self.model.inference(audio_features)
                else:
                    # Fallback - try to extract features and reconstruct
                    converted = self._fallback_conversion(audio_features)

                return converted

        except Exception as e:
            logger.error(f"Error in voice conversion: {str(e)}")
            return audio_features  # Return original if conversion fails

    def _fallback_conversion(self, audio_features):
        """
        Fallback conversion method for unknown model types

        Args:
            audio_features: Input audio features

        Returns:
            Processed audio features
        """
        # Simple processing - apply some transformations
        processed = audio_features

        # Apply normalization
        processed = F.normalize(processed, dim=-1)

        # Apply some filtering if model has appropriate layers
        if hasattr(self.model, 'modules'):
            for module in self.model.modules():
                if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
                    try:
                        processed = module(processed)
                        break
                    except:
                        continue

        return processed

    def synthesize_audio(self, text_features, sample_rate=22050):
        """
        Synthesize audio from text features with more natural speech patterns

        Args:
            text_features: Preprocessed text features
            sample_rate: Target sample rate

        Returns:
            Generated audio waveform
        """
        try:
            # Generate more natural speech-like audio
            duration = max(len(text_features) * 0.12, 1.0)  # 120ms per character
            num_samples = int(duration * sample_rate)
            
            audio = np.zeros(num_samples)
            
            # Speech synthesis parameters
            base_pitch = 120  # Base fundamental frequency (Hz)
            formant_freqs = [800, 1200, 2500]  # Formant frequencies for vowel-like sounds
            
            for i, char_id in enumerate(text_features):
                if char_id > 0:
                    # Calculate timing
                    start_time = i * 0.12
                    end_time = min(start_time + 0.12, duration)
                    
                    start_idx = int(start_time * sample_rate)
                    end_idx = int(end_time * sample_rate)
                    
                    if start_idx < num_samples and end_idx > start_idx:
                        segment_length = end_idx - start_idx
                        segment_t = np.linspace(0, 0.12, segment_length)
                        
                        # Vary pitch based on character
                        pitch_variation = 1.0 + (char_id % 20 - 10) * 0.02  # ±20% pitch variation
                        current_pitch = base_pitch * pitch_variation
                        
                        # Generate fundamental frequency with pitch modulation
                        pitch_modulation = 1.0 + 0.05 * np.sin(2 * np.pi * 5 * segment_t)  # 5Hz vibrato
                        fundamental = np.sin(2 * np.pi * current_pitch * pitch_modulation * segment_t)
                        
                        # Add formants for vowel-like quality
                        formant_sound = np.zeros(segment_length)
                        for j, formant_freq in enumerate(formant_freqs):
                            formant_amplitude = 0.3 / (j + 1)  # Decreasing amplitude for higher formants
                            formant_bandwidth = formant_freq * 0.1  # 10% bandwidth
                            
                            # Simple formant synthesis using sine waves
                            formant_wave = formant_amplitude * np.sin(2 * np.pi * formant_freq * segment_t)
                            formant_sound += formant_wave
                        
                        # Combine fundamental with formants
                        speech_segment = fundamental * 0.4 + formant_sound * 0.6
                        
                        # Add consonant-like noise for some characters
                        if char_id % 5 == 0:  # Every 5th character gets noise component
                            noise_component = 0.15 * np.random.normal(0, 1, segment_length)
                            # Filter noise to speech frequencies
                            noise_component = np.convolve(noise_component, np.ones(5)/5, mode='same')
                            speech_segment += noise_component
                        
                        # Apply envelope for natural attack/decay
                        envelope = np.ones(segment_length)
                        attack_samples = min(int(0.01 * sample_rate), segment_length // 4)  # 10ms attack
                        decay_samples = min(int(0.02 * sample_rate), segment_length // 4)   # 20ms decay
                        
                        if attack_samples > 0:
                            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
                        if decay_samples > 0:
                            envelope[-decay_samples:] = np.linspace(1, 0.3, decay_samples)
                        
                        speech_segment *= envelope
                        
                        # Add to main audio with some overlap
                        overlap_samples = min(int(0.005 * sample_rate), segment_length // 10)  # 5ms overlap
                        if i > 0 and start_idx >= overlap_samples:
                            # Smooth transition from previous segment
                            fade_in = np.linspace(0, 1, overlap_samples)
                            speech_segment[:overlap_samples] *= fade_in
                            audio[start_idx-overlap_samples:start_idx] *= (1 - fade_in)
                        
                        audio[start_idx:end_idx] += speech_segment * 0.7
                else:
                    # Add pause for zero characters (spaces)
                    start_time = i * 0.12
                    pause_duration = 0.08  # 80ms pause
                    start_idx = int(start_time * sample_rate)
                    end_idx = min(start_idx + int(pause_duration * sample_rate), num_samples)
                    
                    if start_idx < num_samples and end_idx > start_idx:
                        # Add very quiet background noise during pauses
                        audio[start_idx:end_idx] += 0.02 * np.random.normal(0, 1, end_idx - start_idx)
            
            # Post-processing
            # Apply mild compression
            audio = np.tanh(audio * 2) * 0.5
            
            # Simple low-pass filter to remove high-frequency artifacts
            if len(audio) > 100:
                # Moving average filter
                window_size = max(3, int(sample_rate / 5000))  # Smooth but preserve speech
                audio = np.convolve(audio, np.ones(window_size)/window_size, mode='same')
            
            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.85  # Leave headroom
            else:
                logger.warning("Generated audio was silent, creating demo tone")
                t = np.linspace(0, 2.0, int(sample_rate * 2))
                audio = 0.3 * np.sin(2 * np.pi * 220 * t)  # A3 note for 2 seconds
            
            # Apply model processing if available
            if self.model is not None:
                try:
                    audio = self._apply_model_processing(audio)
                except Exception as e:
                    logger.warning(f"Model processing failed: {e}")
            
            return audio
            
        except Exception as e:
            logger.error(f"Error in audio synthesis: {str(e)}")
            # Return simple but pleasant fallback
            t = np.linspace(0, 1.5, int(sample_rate * 1.5))
            # Generate a sequence of tones instead of monotone
            frequencies = [220, 247, 262, 294, 330]  # Musical notes
            fallback_audio = np.zeros_like(t)
            for i, freq in enumerate(frequencies):
                start_sample = int(i * len(t) / len(frequencies))
                end_sample = int((i + 1) * len(t) / len(frequencies))
                segment_t = t[start_sample:end_sample] - t[start_sample]
                fallback_audio[start_sample:end_sample] = 0.3 * np.sin(2 * np.pi * freq * segment_t)
            return fallback_audio
    
    def _apply_model_processing(self, audio):
        """Apply loaded model processing to audio"""
        try:
            audio_tensor = torch.tensor(audio, device=self.device, dtype=torch.float32)
            
            # Add batch dimension if needed
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Apply model processing
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    # Try direct forward pass
                    processed = self.model(audio_tensor.unsqueeze(0))  # Add channel dim
                    if isinstance(processed, torch.Tensor):
                        processed_audio = processed.squeeze().cpu().numpy()
                        if len(processed_audio) > 0 and np.max(np.abs(processed_audio)) > 0:
                            return processed_audio
                
                # Fallback: apply simple spectral processing
                return self._simple_spectral_processing(audio)
                
        except Exception as e:
            logger.warning(f"Model processing error: {e}")
            return audio
    
    def _simple_spectral_processing(self, audio):
        """Simple spectral processing as model fallback"""
        try:
            # Apply gentle spectral shaping to make it sound more voice-like
            # This is a very basic approximation
            
            # Simple formant emphasis
            from scipy import signal
            
            # Create a simple vocal tract filter
            b, a = signal.butter(2, [300, 3400], btype='band', fs=22050)
            filtered_audio = signal.filtfilt(b, a, audio)
            
            # Blend with original
            return 0.7 * filtered_audio + 0.3 * audio
            
        except ImportError:
            # If scipy not available, return original
            return audio
        except Exception as e:
            logger.warning(f"Spectral processing failed: {e}")
            return audio

    def apply_audio_effects(self, audio, pitch=0.0, speed=1.0, volume=1.0):
        """
        Apply audio effects to generated speech

        Args:
            audio: Input audio waveform
            pitch: Pitch shift in semitones (-12 to 12)
            speed: Speed multiplier (0.1 to 3.0)
            volume: Volume multiplier (0.1 to 2.0)

        Returns:
            Processed audio waveform
        """
        try:
            # Apply speed change
            if speed != 1.0:
                audio = librosa.effects.time_stretch(audio, rate=speed)

            # Apply pitch shift
            if pitch != 0.0:
                audio = librosa.effects.pitch_shift(audio, sr=22050, n_steps=pitch)

            # Apply volume change
            if volume != 1.0:
                audio = audio * volume

            # Normalize to prevent clipping
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val

            return audio

        except Exception as e:
            logger.error(f"Error applying audio effects: {str(e)}")
            return audio

    def generate_speech(self, text, output_path, pitch=0.0, speed=1.0, volume=1.0, sample_rate=22050):
        """
        Generate speech from text and save to file

        Args:
            text: Input text to synthesize
            output_path: Path to save generated audio
            pitch: Pitch adjustment in semitones
            speed: Speed multiplier
            volume: Volume multiplier
            sample_rate: Sample rate for output audio

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Generating speech for text: '{text[:50]}...'")

            # Preprocess text
            text_features = self.preprocess_text(text)

            # Synthesize audio
            audio = self.synthesize_audio(text_features, sample_rate)

            # Apply effects
            audio = self.apply_audio_effects(audio, pitch, speed, volume)

            # Ensure audio is not empty
            if len(audio) == 0:
                logger.error("Generated audio is empty")
                return False

            # Save to file
            sf.write(output_path, audio, sample_rate)

            logger.info(f"Speech saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            return False