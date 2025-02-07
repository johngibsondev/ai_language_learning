import os
import click
import glob
from dynaconf import Dynaconf
from anki_deck_generation import AnkiDeckGeneration
from generate_conversation import GenerateConversation
from video_transcription import VideoTranscription
from vocabulary_extraction import VocabularyExtraction
from vocabulary_translation import VocabularyTranslation

settings = Dynaconf(settings_files=["config.toml", ".secrets.toml"])


def get_vocabulary_from_video(name):
    video_transcription = VideoTranscription(
        settings.pyannote.auth_key,
        settings.config.output_audio_path,
        settings.config.output_text_path,
        settings.config.temp_path,
        settings.whisper.model,
        "mps",
    )
    vocabulary_extraction = VocabularyExtraction(
        settings.openai_server.url,
        settings.openai_server.api_key,
        settings.config.output_text_path,
        settings.openai_server.model,
    )
    vocabulary_translation = VocabularyTranslation(
        settings.openai_server.url,
        settings.openai_server.api_key,
        settings.config.output_text_path,
        settings.openai_server.model,
    )
    for video_file in glob.glob("input_videos/*.*"):
        video_transcription.extract_audio(video_file)
    for audio_file in glob.glob(f"{settings.config.output_audio_path}/*.wav"):
        for sentence_audio, clip_path, clip_name in video_transcription.split_sentences(
            audio_file
        ):
            sentence_text, sentence_output_path = (
                video_transcription.transcribe_sentence(
                    sentence_audio, clip_path, clip_name
                )
            )

            sentence_output_path = sentence_output_path or os.path.join(
                settings.config.output_text_path, clip_path, f"{clip_name}.txt"
            )

            for vocab_pos in vocabulary_extraction.tag_part_of_speech(sentence_text):
                if vocab_pos:
                    verbs = vocabulary_extraction.get_verbs(
                        sentence_text,
                        vocab_pos,
                        sentence_output_path.replace(".txt", "-verbs.txt"),
                    )
                    nouns = vocabulary_extraction.get_nouns(
                        sentence_text,
                        vocab_pos,
                        sentence_output_path.replace(".txt", "-nouns.txt"),
                    )

                    translated_sentence = vocabulary_translation.translate_sentence(
                        sentence_text,
                        sentence_output_path.replace(".txt", "-translated.txt"),
                    )
                    translated_verbs = vocabulary_translation.translate_verbs(
                        sentence_text,
                        verbs,
                        sentence_output_path.replace(".txt", "-translated_verbs.txt"),
                    )
                    print(f"Translated sentence: {translated_sentence}")
                    print(f"Translated verbs: {translated_verbs}")

    for directory in os.listdir(settings.config.output_text_path):
        anki_deck_generation = AnkiDeckGeneration(
            directory, settings.config.output_text_path
        )
        content = anki_deck_generation.get_deck_content()
        anki_deck_generation.generate_deck(content)


def generate_conversation():
    conversation = GenerateConversation(
        settings.openai_server.url,
        settings.openai_server.api_key,
        prompt="""You are a spanish shop assistent and you are helping a customer to find a product in the store. 
        The customer is asking you for a product that you don't have in the store. How do you respond to the customer? Keep your responses short and simple.
        (speak only in Argentine Spanish)""",
    )
    conversation.generate()


@click.command()
@click.argument("command", type=click.Choice(["conversation", "anki_deck"]))
def app(command: str):
    if command == "conversation":
        generate_conversation()
    elif command == "anki_deck":
        get_vocabulary_from_video()


if __name__ == "__main__":
    app()
