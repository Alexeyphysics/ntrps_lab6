"""Модуль предобработки, очистки и лемматизации текстов для ML.

Лабораторная работа №6 по курсу «Новые технологии в РПС».
Выполнил: студент Кондратенко А.С. (гр. M_PIN_RIS_2607)
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import List, Set

import nltk
from nltk.corpus import stopwords
import pandas as pd
import pymorphy3

from eda_service import load_dataset_dataframe

try:
    _BASE_STOPS = set(stopwords.words("russian"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    _BASE_STOPS = set(stopwords.words("russian"))

_DOMAIN_STOPS: Set[str] = _BASE_STOPS.union({
    "это", "свой", "который", "весь", "мочь", "год",
    "достоинство", "недостаток", "сбербанк", "банк",
    "сбер", "отзыв", "очень", "просто", "также", "день",
})
_MORPH = pymorphy3.MorphAnalyzer()


def preprocess_text(raw_text: str) -> str:
    """Очищает, нормализует и лемматизирует входной текст.

    Этапы: нижний регистр, удаление знаков препинания и цифр,
    фильтрация стоп-слов, приведение к начальной форме (лемме).

    :param raw_text: Исходный необработанный текст отзыва.
    :return: Строка нормализованных лемм, разделенных пробелом.
    """
    cleaned = re.sub(r"[^а-яА-ЯёЁ\s]", " ", str(raw_text).lower())
    words = cleaned.split()
    lemmas: List[str] = []

    for word in words:
        if len(word) > 2 and word not in _DOMAIN_STOPS:
            normal_form = _MORPH.parse(word)[0].normal_form
            if normal_form not in _DOMAIN_STOPS and len(normal_form) > 2:
                lemmas.append(normal_form)

    return " ".join(lemmas)


def prepare_ml_dataframe(dataset_dir: Path | str) -> pd.DataFrame:
    """Загружает датасет и формирует столбец clean_text для TF-IDF.

    Объединяет заголовок и тело отзыва для большей информативности.

    :param dataset_dir: Путь к папке датасета.
    :return: Подготовленный DataFrame без пустых текстов.
    """
    df = load_dataset_dataframe(dataset_dir)
    if df.empty:
        return df

    full_text = df["title"].astype(str) + " " + df["text"].astype(str)
    df["clean_text"] = full_text.apply(preprocess_text)
    df = df[df["clean_text"].str.strip().str.len() > 0].copy()
    return df.reset_index(drop=True)


if __name__ == "__main__":
    print("Самотест модуля text_preprocessing.py...")
    sample = "Отвратительный сервис, карты заблокировали, деньги не вернули!"
    print("Исходный текст:", sample)
    print("После лемматизации:", preprocess_text(sample))
    if Path("dataset").exists():
        ml_df = prepare_ml_dataframe("dataset")
        print(f"Подготовлено строк для ML: {len(ml_df)}")
