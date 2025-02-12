import glob
import os
from typing import Counter, Iterable, List

from openai import OpenAI


class VocabularyExtraction:

    def get_message_translation_prompt(self, text: str):
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

    def get_verb_translation_prompt(self, text: str, vocab: str):
        return (
            [
                {
                    "role": "system",
                    "content": """"Using the following sentence as context, translate the Spanish verb and it's infinitive like this: `
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

    def get_verbs_prompt(self, text: str, vocab_pos: str):
        return [
            {
                "role": "system",
                "content": "Using the following sentence as context, for list of words and their POS labels below list only the verbs one per line with the verb and its infinitive like this verb:inifintive. If there are no verbs just put a -. Don't comment on or annotate the answer in any other way.",
            },
            {
                "role": "user",
                "content": f"Sample sentence: {text}\n\nWords:\n{vocab_pos}",
            },
        ]

    def get_nouns_prompt(self, text: str, vocab_pos: str):
        return [
            {
                "role": "system",
                "content": "Using the following sentence as context, for list of words and their POS labels below list only the non-proper nouns one per line. If there are no nouns just put a -. Don't comment on or annotate the answer in any other way.",
            },
            {
                "role": "user",
                "content": f"Sample sentence: {text}\n\nWords:\n{vocab_pos}",
            },
        ]

    def get_part_of_speech_prompt(self, text: str):
        return [
            {
                "role": "system",
                "content": "In the text below, for each word state the part of speach (POS) it belongs to. Return each word on a line followed by a colon and the part of speach. For example, 'The: determiner'. Don't comment on or annotate the answer in any other way.",
            },
            {
                "role": "user",
                "content": text,
            },
        ]

    def __init__(
        self,
        openai_url: str,
        openai_api_key: str,
        text_path: str,
        model: str = "llama-3.3-70b-instruct",
    ):
        self.openai_url = openai_url
        self.openai_api_key = openai_api_key
        self.text_path = text_path
        self.model = model
        self.client = OpenAI(base_url=openai_url, api_key=openai_api_key)

    def get_sentences(self, glob_pattern: str = "./**/*].txt") -> List[str]:
        sentences = []
        files = glob.glob(os.path.join(self.text_path, glob_pattern))
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                sentence = f.read()
                sentences.append((file, sentence))
        return sentences

    def tag_part_of_speech(self, text: str) -> Iterable[list[str]]:

        words = text.split()
        if len(set(words)) < Counter(words).most_common(1)[0][1]:
            # Skip if there are too many duplicate words (breaks the llm)
            yield None

        vocab_pos_completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.get_part_of_speech_prompt(text),
            temperature=0.2,
            timeout=800,
        )
        vocab_pos = vocab_pos_completion.choices[0].message.content

        yield vocab_pos.split("\n")

    def get_verbs(
        self, text: str, vocab_pos: list[str], output_vocabulary_file=None
    ) -> List[str]:
        verbs_completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.get_verbs_prompt(text, "\n".join(vocab_pos)),
            temperature=0.2,
            timeout=800,
        )
        verbs = verbs_completion.choices[0].message.content
        if output_vocabulary_file:
            with open(output_vocabulary_file, "w") as f:
                f.write(verbs)
        return verbs.split("\n")

    def get_nouns(
        self, text: str, vocab_pos: list[str], output_vocabulary_file=None
    ) -> List[str]:
        nouns_completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.get_nouns_prompt(text, "\n".join(vocab_pos)),
            temperature=0.2,
            timeout=800,
        )
        nouns = nouns_completion.choices[0].message.content
        if output_vocabulary_file:
            with open(output_vocabulary_file, "w") as f:
                f.write(nouns)
        return nouns.split("\n")

    def translate_nouns(self):
        texts = []
        files = glob.glob(os.path.join(self.text_path, "./**/*)].txt"))
        for file_path in files:
            vocab_file_path = file_path.replace(".txt", "-vocab.txt")
            if not os.path.exists(vocab_file_path):
                continue
            with open(file_path, "r", encoding="utf-8") as text_file:
                with open(vocab_file_path, "r", encoding="utf-8") as vocab_file:
                    text = text_file.read()
                    vocab = vocab_file.read()

                texts.append((file_path, text, vocab))
