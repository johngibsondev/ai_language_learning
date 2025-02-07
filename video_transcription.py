import os
from pathlib import Path
from typing import Tuple, Iterable
import torch
from pyannote.core import Annotation
from pyannote.audio import Pipeline
from moviepy import VideoFileClip
from pydub import AudioSegment
import mlx_whisper


class VideoTranscription:
    def __init__(
        self,
        pyannote_token: str,
        output_audio_path="output_audio",
        output_text_path="output_text",
        temp_path="temp",
        whisper_model="mlx-community/whisper-large-v3-turbo",
        device: str = None,
    ):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=pyannote_token,
        )
        if device:
            self.pipeline.to(torch.device(device))

        self.output_audio_path = output_audio_path
        self.output_text_path = output_text_path
        self.temp_path = temp_path
        self.whisper_model = whisper_model

        if not os.path.exists(output_audio_path):
            os.makedirs(output_audio_path)
        if not os.path.exists(output_text_path):
            os.makedirs(output_text_path)
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

    def extract_audio(self, video_path: str):
        path = Path(video_path)
        mp3_file = os.path.join(self.temp_path, f"{path.stem}.mp3")
        wav_file = os.path.join(self.output_audio_path, f"{path.stem}.wav")
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(mp3_file)
        audio_clip.close()
        video_clip.close()
        sound: AudioSegment = AudioSegment.from_mp3(mp3_file)
        sound.export(wav_file, format="wav")
        os.remove(mp3_file)

    def transcribe_sentence(
        self, temp_sentence_audio: str, clip_path: str = None, clip_name: str = None
    ) -> Tuple[str, str]:
        result = mlx_whisper.transcribe(
            temp_sentence_audio,
            path_or_hf_repo=self.whisper_model,
        )
        clip_text = result["text"] if "text" in result else ""
        output_text_path = os.path.join(
            self.output_text_path, clip_path, f"{clip_name}.txt"
        )
        if clip_path and clip_name:
            with open(
                output_text_path,
                "w",
                encoding="utf-8",
            ) as f:
                f.write(clip_text)
        return clip_text, output_text_path

    def split_sentences(self, audio_path: str) -> Iterable[Tuple[str, str, str]]:
        diarization: Annotation = self.pipeline(audio_path)
        wav_path = Path(audio_path)
        clip_path = wav_path.stem
        text_directory = os.path.join(self.output_text_path, clip_path)
        if not os.path.exists(text_directory):
            os.makedirs(text_directory)
        clip_number = 0
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            clip_start = int(turn.start)
            clip_end = int(turn.end)
            clip_name = f"[{clip_number:05d}].{speaker}.[{clip_start:05d}.{clip_end-clip_start:03d}]"
            clip_number += 1
            if not os.path.exists(os.path.join(self.temp_path, clip_path)):
                os.makedirs(os.path.join(self.temp_path, clip_path))
            temp_sentence_audio = os.path.join(
                self.temp_path, clip_path, f"{clip_name}.wav"
            )
            sound: AudioSegment = AudioSegment.from_wav(audio_path)
            start_seconds = turn.start * 1000
            end_seconds = turn.end * 1000
            sound[start_seconds:end_seconds].export(temp_sentence_audio, format="wav")
            yield temp_sentence_audio, clip_path, clip_name

    def transcribe_video(self, video_path: str):
        self.extract_audio(video_path)
        for temp_sentence_audio, clip_name in self.split_sentences(video_path):
            self.transcribe_sentences(temp_sentence_audio, clip_name)
