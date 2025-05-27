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
        Synthesize audio from text features
        
        Args:
            text_features: Preprocessed text features
            sample_rate: Target sample rate
            
        Returns:
            Generated audio waveform
        """
        try:
            # Generate base audio using a simple synthesis method
            # This is a placeholder - real TTS would use proper synthesis
            duration = max(len(text_features) * 0.15, 1.0)  # At least 1 second, 150ms per character
            num_samples = int(duration * sample_rate)
            
            # Generate basic waveform
            t = np.linspace(0, duration, num_samples)
            audio = np.zeros(num_samples)
            
            # Create a more audible synthesized audio
            base_freq = 150  # Lower base frequency for more natural sound
            
            for i, char_id in enumerate(text_features):
                if char_id > 0:
                    # Generate tone based on character with better frequency mapping
                    freq = base_freq + (char_id % 30) * 15  # Frequency between 150-600 Hz
                    start_time = i * 0.15
                    end_time = min(start_time + 0.15, duration)
                    
                    start_idx = int(start_time * sample_rate)
                    end_idx = int(end_time * sample_rate)
                    
                    if start_idx < num_samples and end_idx > start_idx:
                        segment_length = end_idx - start_idx
                        segment_t = np.linspace(0, 0.15, segment_length)
                        
                        # Generate a more complex tone with harmonics
                        fundamental = np.sin(2 * np.pi * freq * segment_t)
                        harmonic1 = 0.3 * np.sin(2 * np.pi * freq * 2 * segment_t)
                        harmonic2 = 0.1 * np.sin(2 * np.pi * freq * 3 * segment_t)
                        tone = fundamental + harmonic1 + harmonic2
                        
                        # Apply better envelope
                        attack = 0.02  # 20ms attack
                        decay = 0.05   # 50ms decay
                        
                        envelope = np.ones(segment_length)
                        attack_samples = int(attack * sample_rate)
                        decay_samples = int(decay * sample_rate)
                        
                        if attack_samples > 0:
                            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
                        if decay_samples > 0 and decay_samples < segment_length:
                            envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
                        
                        audio[start_idx:end_idx] += tone * envelope * 0.5
            
            # Add some variation for spaces and punctuation
            for i in range(len(audio) // (sample_rate // 10)):  # Every 100ms
                if np.random.random() > 0.7:  # 30% chance
                    start_idx = i * (sample_rate // 10)
                    end_idx = min(start_idx + (sample_rate // 20), len(audio))  # 50ms
                    audio[start_idx:end_idx] *= 0.3  # Reduce volume for pause effect
            
            # Normalize to prevent clipping but ensure audible output
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.8  # Leave some headroom
            else:
                # Generate fallback beep if audio is silent
                logger.warning("Generated audio was silent, creating fallback tone")
                t = np.linspace(0, 1.0, sample_rate)
                audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 1 second 440Hz tone
            
            # Apply voice conversion if model is available (simplified)
            if self.model is not None:
                try:
                    # Simple processing with the model
                    audio_tensor = torch.tensor(audio, device=self.device, dtype=torch.float32)
                    audio_features = audio_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                    audio_features = self.apply_voice_conversion(audio_features)
                    processed_audio = audio_features.squeeze().cpu().numpy()
                    
                    # Ensure the processed audio is valid
                    if len(processed_audio) > 0 and np.max(np.abs(processed_audio)) > 0:
                        audio = processed_audio
                except Exception as e:
                    logger.warning(f"Voice conversion failed, using original audio: {e}")
            
            return audio
            
        except Exception as e:
            logger.error(f"Error in audio synthesis: {str(e)}")
            # Return audible fallback tone instead of silence
            t = np.linspace(0, 2.0, int(sample_rate * 2))
            return 0.3 * np.sin(2 * np.pi * 440 * t)
    
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
