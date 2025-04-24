import os
import logging
import whisper
import torch
import numpy as np
from pydub import AudioSegment
import tempfile
from typing import Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SpeechToText:
    """Class for handling speech-to-text conversion using Whisper"""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the speech-to-text converter.
        
        Args:
            model_size (str): Size of the Whisper model to use (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Whisper model (size: {model_size}) on {self.device}")
        self.model = whisper.load_model(model_size, device=self.device)
    
    def preprocess_audio(self, audio_data: Union[bytes, str]) -> str:
        """
        Preprocess audio data for transcription.
        
        Args:
            audio_data: Audio data in bytes or path to audio file
            
        Returns:
            str: Path to the preprocessed audio file
        """
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                
                # If audio_data is bytes, write it to the temp file
                if isinstance(audio_data, bytes):
                    temp_file.write(audio_data)
                # If audio_data is a string (file path), copy the file
                else:
                    audio = AudioSegment.from_file(audio_data)
                    audio.export(temp_path, format="wav")
                
                return temp_path
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
            raise
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            # Load and preprocess the audio
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            
            # Detect the spoken language
            _, probs = self.model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            logger.info(f"Detected language: {detected_lang}")
            
            # Decode the audio
            options = whisper.DecodingOptions(fp16=False)
            result = whisper.decode(self.model, mel, options)
            
            # Clean up the temporary file
            try:
                os.unlink(audio_path)
            except Exception as e:
                logger.warning(f"Error deleting temporary file: {str(e)}")
            
            return result.text
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise
    
    def process_audio_file(self, audio_file) -> str:
        """
        Process an audio file and return its transcription.
        
        Args:
            audio_file: File object or path to audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            # Read the audio file
            if hasattr(audio_file, 'read'):
                audio_data = audio_file.read()
            else:
                audio_data = audio_file
            
            # Preprocess the audio
            temp_path = self.preprocess_audio(audio_data)
            
            # Transcribe the audio
            transcription = self.transcribe(temp_path)
            
            return transcription
        except Exception as e:
            logger.error(f"Error processing audio file: {str(e)}")
            raise 