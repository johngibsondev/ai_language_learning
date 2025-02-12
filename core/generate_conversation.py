from typing import List
from openai import OpenAI
from core.microphone_transcription import MicrophoneTranscription
from core.speech_generation import SpeechGeneration


class Speaker:
    language = ""
    name = ""
    model = ""

    def __init__(self, name, language, model):
        self.name = name
        self.language = language
        self.model = model


class GenerateConversation:
    def __init__(
        self,
        openai_url: str,
        openai_key: str,
        model: str = "llama-3.3-70b-instruct",
        language: str = "es",
        prompt: str = "",
        speakers: List[Speaker] = [
            {
                "name": "Claude",
                "language": "es",
                "model": "es_MX-claude-14947-epoch-high.onnx",
            }
        ],
    ):
        self.prompt = prompt
        self.language = language
        self.client = OpenAI(base_url=openai_url, api_key=openai_key)
        self.model = model
        self.microphone_transcription = MicrophoneTranscription(language=language)
        self.speakers = {
            speaker["name"]: SpeechGeneration(
                "models",
                speaker["model"],
            )
            for speaker in speakers
        }

    def generate(self):
        messages = [
            {
                "role": "system",
                "content": self.prompt,
            }
        ]

        self.speakers["Claude"].generate_speech("Hola")
        while True:
            client_dialogue = self.microphone_transcription.listen()
            messages.append(
                {
                    "role": "user",
                    "content": client_dialogue,
                }
            )
            converation_completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                timeout=800,
            )

            agent_response = converation_completion.choices[0].message.content
            print(f"Agent: {agent_response}")
            self.speakers["Claude"].generate_speech(agent_response)
            messages.append(
                {
                    "role": "assistant",
                    "content": agent_response,
                }
            )
