# src/audio_analysis.py

import torch
import librosa
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from config import DEVICE, MODEL_CACHE_DIR

class AudioToText:
    def __init__(self, model_name="openai/whisper-tiny"):
        print(f"[AudioToText] Loading Whisper model '{model_name}' on device: {DEVICE}")
        self.processor = WhisperProcessor.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
        self.model.to(DEVICE)
    
    def transcribe(self, audio_file_path):
        audio, sr = librosa.load(audio_file_path, sr=16000)
        
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)

        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
