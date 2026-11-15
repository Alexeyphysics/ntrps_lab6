"""Сервис машинного обучения: разбиение, TF-IDF, обучение и инференс.

Лабораторная работа №6 по курсу «Новые технологии в РПС».
Выполнил: студент Кондратенко А.С. (гр. M_PIN_RIS_2607)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from text_preprocessing import preprocess_text


def split_dataset_three_way(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разделяет датасет на тренировочную, валидационную и демо выборки.

    Применяет стратификацию по оценкам (rating) для сохранения баланса.

    :param df: Подготовленный DataFrame с колонками rating и clean_text.
    :param train_ratio: Доля тренировочной выборки (остаток пополам на val/demo).
    :param random_state: Зерно генератора случайных чисел.
    :return: Кортеж (train_df, val_df, demo_df).
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=1.0 - train_ratio,
        stratify=df["rating"],
        random_state=random_state,
    )
    val_df, demo_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["rating"],
        random_state=random_state,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        demo_df.reset_index(drop=True),
    )


def create_ml_pipeline(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    max_features: int = 1000,
    random_state: int = 42,
) -> Pipeline:
    """Создает Pipeline из векторизатора TF-IDF и классификатора.

    :param n_estimators: Количество деревьев в лесу.
    :param max_depth: Максимальная глубина каждого дерева.
    :param min_samples_split: Минимум объектов для расщепления узла.
    :param max_features: Максимальный размер словаря TF-IDF.
    :param random_state: Фиксатор воспроизводимости.
    :return: Необученный объект sklearn Pipeline.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
    )
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def train_model(pipeline: Pipeline, train_df: pd.DataFrame) -> Pipeline:
    """Обучает Pipeline на тренировочной выборке.

    :param pipeline: Конвейер TF-IDF + модель.
    :param train_df: Тренировочный DataFrame.
    :return: Обученный Pipeline.
    """
    pipeline.fit(train_df["clean_text"], train_df["rating"])
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    eval_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Оценивает качество обученной модели на заданной выборке.

    :param pipeline: Обученный Pipeline.
    :param eval_df: Валидационная или демонстрационная выборка.
    :return: Словарь метрик (accuracy, f1_weighted, report).
    """
    y_true = eval_df["rating"]
    y_pred = pipeline.predict(eval_df["clean_text"])

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    report = str(classification_report(y_true, y_pred, zero_division=0))

    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "report": report,
    }


def save_model(pipeline: Pipeline, model_path: Path | str) -> None:
    """Сохраняет обученный Pipeline на диск с помощью joblib.

    :param pipeline: Обученный Pipeline.
    :param model_path: Путь к файлу модели.
    """
    path = Path(model_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(model_path: Path | str) -> Pipeline:
    """Загружает сохраненный Pipeline с диска.

    :param model_path: Путь к файлу .joblib.
    :return: Загруженный Pipeline.
    """
    path = Path(model_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {path}")
    return joblib.load(path)


def predict_single_review(
    pipeline: Pipeline,
    raw_text: str,
) -> Tuple[int, Dict[int, float], str]:
    """Выполняет предсказание оценки для нового сырого текста отзыва.

    :param pipeline: Обученный Pipeline.
    :param raw_text: Сырой текст отзыва пользователя.
    :return: (предсказанная_оценка, словарь_вероятностей, леммы_текста).
    """
    cleaned = preprocess_text(raw_text)
    if not cleaned:
        return (0, {}, "")

    pred_class = int(pipeline.predict([cleaned])[0])
    probs_array = pipeline.predict_proba([cleaned])[0]
    classes = pipeline.classes_
    probs_dict = {int(c): float(p) for c, p in zip(classes, probs_array)}

    return (pred_class, probs_dict, cleaned)


if __name__ == "__main__":
    from text_preprocessing import prepare_ml_dataframe

    if Path("dataset").exists():
        data = prepare_ml_dataframe("dataset")
        tr, va, de = split_dataset_three_way(data)
        print(f"Разбиение: train={len(tr)}, val={len(va)}, demo={len(de)}")
        pipe = create_ml_pipeline()
        train_model(pipe, tr)
        res = evaluate_model(pipe, va)
        print(f"Валидация -> Accuracy: {res['accuracy']:.4f}, "
              f"F1: {res['f1_weighted']:.4f}")
