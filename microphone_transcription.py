import io
import numpy as np
import soundfile as sf
import speech_recognition as sr
import whisper


class MicrophoneTranscription:
    def __init__(
        self,
        whisper_model="turbo",
        language="es",
    ):
        self.language = language
        self.whisper_model = whisper.load_model(whisper_model)
        self.recogniser = sr.Recognizer()

    def listen(self):
        with sr.Microphone() as source:
            print("Say something!")
            audio = self.recogniser.listen(source)

            wav_bytes = audio.get_wav_data(convert_rate=16000)
            wav_stream = io.BytesIO(wav_bytes)
            audio_array, _ = sf.read(wav_stream)
            audio_array = audio_array.astype(np.float32)

            options = {
                "language": self.language,
                "task": "transcribe",
                "fp16": False,
            }
            result = whisper.transcribe(self.whisper_model, audio_array, **options)
            text = result["text"]
            print(f"You said: {text}")
            return text
