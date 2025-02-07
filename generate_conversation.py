from openai import OpenAI

from microphone_transcription import MicrophoneTranscription
from speech_generation import SpeechGeneration


class GenerateConversation:
    def __init__(
        self,
        openai_url: str,
        openai_key: str,
        model: str = "llama-3.3-70b-instruct",
        language: str = "es",
        prompt: str = "",
    ):
        self.prompt = prompt
        self.language = language
        self.client = OpenAI(base_url=openai_url, api_key=openai_key)
        self.model = model
        self.speech_generation = SpeechGeneration(
            "models",
            "es_MX-claude-14947-epoch-high.onnx",
        )
        self.microphone_transcription = MicrophoneTranscription(language=language)

    def generate(self):
        messages = [
            {
                "role": "system",
                "content": self.prompt,
            }
        ]

        while True:
            self.speech_generation.generate_speech("Hola")
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
            self.speech_generation.generate_speech(agent_response)
            messages.append(
                {
                    "role": "assistant",
                    "content": agent_response,
                }
            )


# pip3 install PyObjC
