"""Модуль тестирования приложения (Unit Tests) для ЛР6.

Лабораторная работа №6 по курсу «Новые технологии в РПС».
Выполнил: студент Кондратенко А.С. (гр. M_PIN_RIS_2607)
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from dataset_service import create_annotation_file
from eda_service import load_dataset_dataframe
from ml_service import (
    create_ml_pipeline,
    evaluate_model,
    load_model,
    predict_single_review,
    save_model,
    split_dataset_three_way,
    train_model,
)
from review_iterator import ReviewIterator
from text_preprocessing import prepare_ml_dataframe, preprocess_text


def test_positive_load_dataset_dataframe() -> None:
    """Позитивный тест: загрузка существующего датасета."""
    df = load_dataset_dataframe("dataset")
    assert not df.empty
    assert "word_count" in df.columns


def test_positive_annotation_creation(tmp_path: Path) -> None:
    """Позитивный тест: генерация CSV аннотации."""
    out_csv = tmp_path / "test_annotation.csv"
    df = create_annotation_file("dataset", out_csv)
    assert out_csv.exists()
    assert len(df) > 0


def test_positive_text_preprocessing() -> None:
    """Позитивный тест: очистка и лемматизация текста."""
    raw = "Ужасный обман, карты заблокировали, деньги пропали!"
    cleaned = preprocess_text(raw)
    assert "обман" in cleaned
    assert "карта" in cleaned
    assert "деньга" in cleaned


def test_positive_ml_pipeline_and_save_load(tmp_path: Path) -> None:
    """Позитивный тест: разбиение, обучение, сохранение и прогноз."""
    df = prepare_ml_dataframe("dataset")
    tr, va, de = split_dataset_three_way(df)
    assert len(tr) > len(va)
    assert len(va) > 0 and len(de) > 0

    pipe = create_ml_pipeline(n_estimators=10, max_depth=5)
    train_model(pipe, tr)
    metrics = evaluate_model(pipe, va)
    assert 0.0 <= metrics["accuracy"] <= 1.0

    model_file = tmp_path / "test_model.joblib"
    save_model(pipe, model_file)
    assert model_file.exists()

    loaded_pipe = load_model(model_file)
    pred, probs, _ = predict_single_review(
        loaded_pipe, "Ужасный обман, заблокировали карту"
    )
    assert pred in (1, 2, 3, 4, 5)
    assert pytest.approx(sum(probs.values()), 0.01) == 1.0


def test_negative_load_nonexistent_dataset() -> None:
    """Негативный тест: чтение из несуществующей директории."""
    df = load_dataset_dataframe("non_existent_folder_12345")
    assert df.empty


def test_negative_load_nonexistent_model() -> None:
    """Негативный тест: попытка загрузки несуществующей модели."""
    with pytest.raises(FileNotFoundError):
        load_model("models/missing_model_999.joblib")


def test_negative_predict_only_stopwords() -> None:
    """Негативный тест: предсказание для текста без значимых слов."""
    pipe = create_ml_pipeline(n_estimators=10)
    pred, probs, lemmas = predict_single_review(pipe, "и в на по 123 !!")
    assert pred == 0
    assert probs == {}
    assert lemmas == ""


def test_edge_iterator_stop_iteration() -> None:
    """Исключительная ситуация: выброс StopIteration в итераторе."""
    iterator = ReviewIterator("dataset", "99")
    with pytest.raises(StopIteration):
        next(iterator)
