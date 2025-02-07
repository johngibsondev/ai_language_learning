import os
import time
import wave
from piper.voice import PiperVoice
import simpleaudio as sa
from playsound import playsound


class SpeechGeneration:
    def __init__(
        self,
        models_path: str = "models",
        model_name: str = "es_MX-claude-14947-epoch-high.onnx",
    ):
        self.voice = PiperVoice.load(os.path.join(models_path, model_name))

    def generate_speech(self, text: str):
        try:
            temp_file = os.path.abspath(f"{time.time()}.wav")
            wav_file = wave.open(temp_file, "w")
            self.voice.synthesize(text, wav_file)
            wav_file.close()
            playsound(temp_file)
            # wave_obj = sa.WaveObject.from_wave_file(temp_file)
            # play_obj = wave_obj.play()
            # play_obj.wait_done()
            os.remove(temp_file)
        except Exception as e:
            print(f"Error: {e}")
