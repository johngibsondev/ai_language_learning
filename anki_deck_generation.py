import glob
import os
from pathlib import Path
from typing import List, Tuple
import genanki
import hashlib
import uuid


class AnkiDeckGeneration:
    def __init__(self, deck_name: str, text_content_path: str):
        self.anki_model = genanki.Model(
            1091735104,
            "Simple Model with Media",
            fields=[
                {
                    "name": "Front",
                    "font": "Arial",
                },
                {
                    "name": "Front-Context",
                    "font": "Arial",
                },
                {
                    "name": "Back-Definition",
                    "font": "Arial",
                },
                {
                    "name": "Back-Context",
                    "font": "Arial",
                },
            ],
            templates=[
                {
                    "name": "Card 1",
                    "qfmt": '{{Front}}<br><br><div class="example">{{Front-Context}}</div>',
                    "afmt": '{{FrontSide}}\n\n<hr id=answer><b>{{Back-Definition}}</b><br><br><div class="example">{{Back-Context}}</div>',
                },
            ],
            css="""
            .card { 
                font-family: arial; 
                font-size: 20px; 
                text-align: center; 
                color: black; 
                background-color: white;
            }
            .example { 
                font-family: Liberation Sans; 
                font-size: 8;
            }
            """,
        )

        self.deck_name = deck_name
        self.text_content_path = text_content_path

    def get_uuid_from_string(self, string: str):
        m = hashlib.md5()
        m.update(string.encode("utf-8"))
        new_uuid = uuid.UUID(m.hexdigest())
        temp = int(m.hexdigest(), 16)
        return str(new_uuid)

    def get_hash_from_string(self, string: str):
        m = hashlib.md5()
        m.update(string.encode("utf-8"))
        return int(m.hexdigest(), 16) % 2**32

    def get_deck_content(
        self,
        original_text_glob_pattern: str = "./**/*].txt",
        translation_suffix: str = "-translated",
        nouns_suffix: str = "-nouns",
        verbs_suffix: str = "-verbs",
    ):
        translated_words = []
        duplicate_words = set()

        for original_sentences_path in glob.glob(
            os.path.join(self.text_content_path, original_text_glob_pattern)
        ):
            path = Path(original_sentences_path)

            translated_sentence_path = path.with_stem(
                path.stem + translation_suffix
            ).as_posix()

            example_en = open(
                translated_sentence_path,
                "r",
                encoding="utf-8",
            ).read()

            example_es = open(
                original_sentences_path,
                "r",
                encoding="utf-8",
            ).read()

            translated_verbs_path = path.with_stem(path.stem + verbs_suffix).as_posix()
            with open(translated_verbs_path, "r", encoding="utf-8") as f:
                while True:
                    line = f.readline()
                    if line == "-":
                        break
                    try:
                        verb_es, verb_en = line.split(":")
                        line = f.readline()
                        infinitive_es, infinitive_en = line.split(":")
                        line = f.readline()
                        if infinitive_es not in duplicate_words:
                            duplicate_words.add(infinitive_es)
                            translated_words.append(
                                {
                                    "verb_es": verb_es.strip(),
                                    "verb_en": verb_en.strip(),
                                    "infinitive_es": infinitive_es.strip(),
                                    "infinitive_en": infinitive_en.strip(),
                                    "example_es": example_es.strip(),
                                    "example_en": example_en.strip(),
                                }
                            )
                        if line != "---":
                            break
                    except Exception as e:
                        print(f"{translated_verbs_path}: Error in line: {line}")
                        print(e)
                        break
        return translated_words

    def generate_deck(self, deck_content: List[Tuple[str, str, str, str, str, str]]):
        deck_id = self.get_hash_from_string(self.deck_name)
        my_deck = genanki.Deck(deck_id, self.deck_name)
        for card_content in deck_content:
            my_note = genanki.Note(
                model=self.anki_model,
                fields=[
                    f"{card_content['verb_en']} ({card_content['infinitive_en']})",
                    card_content["example_en"],
                    f"{card_content['verb_es']} ({card_content['infinitive_es']})",
                    card_content["example_es"],
                    # f"my_sound_file.mp3",
                ],
                guid=self.get_uuid_from_string(card_content["infinitive_es"]),
            )
            my_deck.add_note(my_note)

        package = genanki.Package(my_deck)

        # my_package = genanki.Package(my_deck)
        # my_package.media_files = ["my_sound_file.mp3", "images/my_image_file.jpg"]

        package.write_to_file(f"{self.deck_name}.apkg")
