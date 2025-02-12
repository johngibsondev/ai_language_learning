import os
import glob
from typing import List
from openai import OpenAI


class VocabularyTranslation:

    def __get_message_translation_prompt(self, text: str):
        return [
            {
                "role": "system",
                "content": """"Translate the following sentence to English. If there are no words just return -. Don't comment on or annotate the answer in any other way.""",
            },
            {
                "role": "user",
                "content": f"{text}",
            },
        ]

    def __get_verb_translation_prompt(self, text: str, vocab: str):
        messages = (
            [
                {
                    "role": "system",
                    "content": """"Using the following sentence as context, translate the Spanish verb and it's infinitive like this: 
                Tienes:tener
                Es:ser
                Should become:
                Es:It is
                Ser:To be
                ---
                Tienes:You have
                tener:To have
                ---
                If the Spanish verb indicates a pronoun include it in the non-infinitive translation. If there are no verbs just return -. Don't comment on or annotate the answer in any other way.""",
                },
                {
                    "role": "user",
                    "content": f"Sample sentence: {text}\n\nVerbs:\n{vocab}",
                },
            ],
        )
        return messages[0]

    def __init__(
        self,
        openai_url: str,
        openai_api_key: str,
        text_path: str,
        model: str = "llama-3.3-70b-instruct",
        nouns_prefix: str = "nouns",
        verbs_prefix: str = "verbs",
    ):
        self.openai_url = openai_url
        self.openai_api_key = openai_api_key
        self.text_path = text_path
        self.model = model
        self.client = OpenAI(base_url=openai_url, api_key=openai_api_key)
        self.nouns_prefix = nouns_prefix
        self.verbs_prefix = verbs_prefix

    def get_sentences(self, glob_pattern: str = "./**/*].txt"):
        texts = []
        files = glob.glob(os.path.join(self.text_path, glob_pattern))
        for file_path in files:
            nouns_file_path = file_path.replace(".txt", f"-{self.nouns_prefix}.txt")
            verbs_file_path = file_path.replace(".txt", f"-{self.verbs_prefix}.txt")
            if not os.path.exists(verbs_file_path):
                continue
            with open(file_path, "r", encoding="utf-8") as text_file:
                with open(verbs_file_path, "r", encoding="utf-8") as verbs_file:
                    with open(nouns_file_path, "r", encoding="utf-8") as nouns_file:
                        text = text_file.read()
                        verbs = verbs_file.read()
                        nouns = nouns_file.read()

                        texts.append((file_path, text, nouns, verbs))
        return texts

    def translate_sentence(self, text: str, translated_sentence_path: str) -> str:
        messages = self.__get_message_translation_prompt(text)
        verbs_completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            timeout=800,
        )
        translation = verbs_completion.choices[0].message.content
        with open(translated_sentence_path, "w") as f:
            f.write(translation)
        return translation

    def translate_verbs(
        self, text: str, vocab: str, translated_verbs_path: str
    ) -> List[str]:
        messages = self.__get_verb_translation_prompt(text, vocab)
        verbs_completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            timeout=800,
        )
        translation = verbs_completion.choices[0].message.content
        with open(translated_verbs_path, "w") as f:
            f.write(translation)
        return translation.split("\n")
