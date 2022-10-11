import argparse
import configparser
import csv
import json
import os
from typing import Sequence, Set

import spacy
from tqdm import tqdm

nlp = spacy.load('de_core_news_sm')

LABELING_PROMPT = ('\n == Please label sentiment (p = positive, n = negative, '
                   'h = hostile, enter: neutral): ')

config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
config = configparser.ConfigParser()
config.read(config_path)
SEARCH_KEYWORDS = config.get("ArticleSelection", "search_words").lower().split(", ")


def has_keyword(text: str) -> bool:
    return any(keyword in text.lower() for keyword in SEARCH_KEYWORDS)


def sentencize(texts: Sequence[str]) -> Sequence[str]:
    all_sentences = []
    for doc in tqdm(nlp.pipe(texts, batch_size=40), total=len(texts), desc='Parsing sentences'):
        sents = [s.text.strip() for s in doc.sents]
        all_sentences.extend(sents)
    return all_sentences


def extract_relevant_sections(
        texts: Sequence[str],
        include_next_sentence: bool,
        already_annotated: Set[str]
) -> Sequence[str]:
    skip = already_annotated.copy()
    sentences = sentencize(texts)
    relevant_sections = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if sentence in skip:
            continue
        if has_keyword(sentence):
            section = sentence
            skip.add(sentence)
            if include_next_sentence:
                for j in range(i + 1, len(sentences)):
                    next_sentence = sentences[j]
                    section = f'{section}\n{next_sentence}'
                    skip.add(next_sentence)
                    if not has_keyword(next_sentence):
                        break
            relevant_sections.append(section)

    return relevant_sections


def sentiment_annotation(texts: Sequence[str], output_path: str) -> None:
    translation = {'p': 0, 'n': 1, '': 2, 'h': 3}

    for section in tqdm(texts, desc='Annotating'):
        sentiment = input(LABELING_PROMPT)
        while sentiment not in translation.keys():
            print(LABELING_PROMPT)
            print(f'\n\n{section}')
            sentiment = input()

        entry = (section, translation[sentiment])
        write_entry_to_csv(entry, csv_path=output_path)


def write_entry_to_csv(row: Sequence[str], csv_path: str) -> None:
    with open(csv_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(row)
        f.flush()
        os.fsync(f)


def extract_texts(input_path: str) -> Sequence[str]:
    with open(os.path.expanduser(input_path), 'r') as f:
        articles = json.load(f)
    texts = [article['text'] for article in articles.values()]
    return texts


def get_already_annotated(csv_path: str) -> Set[str]:
    if not os.path.exists(csv_path):
        return set()
    already_annoated = set()
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for text, label in reader:
            sents = nlp(text).sents
            for sent in sents:
                already_annoated.add(sent.text)
    return already_annoated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(epilog=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        'source_path',
        help='Path to JSON file.'
    )
    parser.add_argument(
        'output_path',
        help='Path to CSV file.',
    )
    args = parser.parse_args()

    source_path = os.path.expanduser(args.source_path)
    output_path = os.path.expanduser(args.output_path)

    texts = extract_texts(source_path)
    already_annoated = get_already_annotated(output_path)
    if already_annoated:
    relevant_sections = extract_relevant_sections(
        texts, include_next_sentence=True, already_annotated=already_annoated
    )
    sentiment_annotation(relevant_sections, output_path)
