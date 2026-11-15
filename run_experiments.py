"""Скрипт проведения экспериментов по подбору гиперпараметров.

Лабораторная работа №6 по курсу «Новые технологии в РПС».
Выполнил: студент Кондратенко А.С. (гр. M_PIN_RIS_2607)
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from ml_service import (
    create_ml_pipeline,
    evaluate_model,
    save_model,
    split_dataset_three_way,
    train_model,
)
from text_preprocessing import prepare_ml_dataframe


def setup_logger(
    log_file: Path | str = "logs/experiments.log",
) -> logging.Logger:
    """Настраивает логгер для записи в файл и консоль."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ml_experiments")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def get_experiment_configs() -> List[Dict[str, Any]]:
    """Возвращает список конфигураций (3 гиперпараметра x 2 изменения)."""
    return [
        {
            "name": "Exp 1: Baseline",
            "param": "baseline",
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
        },
        {
            "name": "Exp 2: n_estimators=10",
            "param": "n_estimators",
            "n_estimators": 10,
            "max_depth": None,
            "min_samples_split": 2,
        },
        {
            "name": "Exp 3: n_estimators=300",
            "param": "n_estimators",
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
        },
        {
            "name": "Exp 4: max_depth=5",
            "param": "max_depth",
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 2,
        },
        {
            "name": "Exp 5: max_depth=20",
            "param": "max_depth",
            "n_estimators": 100,
            "max_depth": 20,
            "min_samples_split": 2,
        },
        {
            "name": "Exp 6: min_samples_split=5",
            "param": "min_samples_split",
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 5,
        },
        {
            "name": "Exp 7: min_samples_split=15",
            "param": "min_samples_split",
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 15,
        },
    ]


def write_text_report(
    results: List[Dict[str, Any]],
    best_exp: Dict[str, Any],
    best_demo_report: str,
    report_path: Path | str = "reports/hyperparameters_report.txt",
) -> None:
    """Формирует текстовый отчет по итогам экспериментов и влиянию."""
    out_file = Path(report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sep = "=" * 68

    lines = [
        sep,
        "ОТЧЁТ ПО ЭКСПЕРИМЕНТАМ С ГИПЕРПАРАМЕТРАМИ (ЛР6, Вариант 9)",
        "Студент: Кондратенко А.С. (гр. M_PIN_RIS_2607)",
        "Модель: Pipeline(TfidfVectorizer + RandomForestClassifier)",
        f"{sep}\n",
        "1. СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТОВ:\n",
        f"{'Эксперимент':<30} | {'Val Acc':<8} | {'Val F1':<8} | "
        f"{'Demo Acc':<8} | {'Demo F1':<8} | {'Время(с)'}",
        "-" * 85,
    ]

    for r in results:
        lines.append(
            f"{r['name']:<30} | {r['val_acc']:<8.4f} | "
            f"{r['val_f1']:<8.4f} | {r['demo_acc']:<8.4f} | "
            f"{r['demo_f1']:<8.4f} | {r['elapsed']:.3f}"
        )

    lines.extend([
        f"\n{sep}",
        f"2. ЛУЧШАЯ МОДЕЛЬ: {best_exp['name']}",
        f"   Параметры: n_estimators={best_exp['n_estimators']}, "
        f"max_depth={best_exp['max_depth']}, "
        f"min_samples_split={best_exp['min_samples_split']}",
        f"   Метрики на Val : Accuracy={best_exp['val_acc']:.4f}, "
        f"F1={best_exp['val_f1']:.4f}",
        f"   Метрики на Demo: Accuracy={best_exp['demo_acc']:.4f}, "
        f"F1={best_exp['demo_f1']:.4f}",
        "\n--- Classification Report на демонстрационной выборке ---",
        best_demo_report,
        sep,
        "3. АНАЛИЗ ВЛИЯНИЯ ГИПЕРПАРАМЕТРОВ НА РЕЗУЛЬТАТ:",
        "",
        "а) n_estimators (количество деревьев в ансамбле):",
        "   - При малом значении (10) ансамбль имеет высокую дисперсию "
        "(недообучение), точность падает.",
        "   - Увеличение до 300 снижает разброс предсказаний за счет "
        "усреднения, повышает качество, но увеличивает время обучения.",
        "",
        "б) max_depth (максимальная глубина деревьев):",
        "   - Сильное ограничение (max_depth=5) не дает деревьям уловить "
        "комбинации TF-IDF признаков (недообучение).",
        "   - Увеличение до 20 или None позволяет строить глубокие правила "
        "для разреженных текстовых векторов.",
        "",
        "в) min_samples_split (минимум примеров для расщепления узла):",
        "   - Порог 2 позволяет строить детальные листья.",
        "   - Увеличение порога до 15 выступает в роли сильного "
        "регуляризатора, снижая точность на небольшой выборке (234 отзыва).",
    ])

    out_file.write_text("\n".join(lines), encoding="utf-8")


def run_all_experiments() -> None:
    """Основной пайплайн проведения экспериментов и сохранения модели."""
    logger = setup_logger("logs/experiments.log")
    logger.info("=== Старт серии экспериментов (ЛР6, Вариант 9) ===")

    df = prepare_ml_dataframe("dataset")
    train_df, val_df, demo_df = split_dataset_three_way(df)

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(data_dir / "train_split.csv", index=False)
    val_df.to_csv(data_dir / "val_split.csv", index=False)
    demo_df.to_csv(data_dir / "demo_split.csv", index=False)

    logger.info(
        f"Разбиение датасета: train={len(train_df)}, "
        f"val={len(val_df)}, demo={len(demo_df)}"
    )

    configs = get_experiment_configs()
    results: List[Dict[str, Any]] = []
    best_score = -1.0
    best_exp: Dict[str, Any] = {}
    best_pipeline: Optional[Any] = None
    best_demo_report = ""

    for cfg in configs:
        start_t = time.perf_counter()
        pipe = create_ml_pipeline(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_split"],
        )
        train_model(pipe, train_df)
        elapsed = time.perf_counter() - start_t

        val_res = evaluate_model(pipe, val_df)
        demo_res = evaluate_model(pipe, demo_df)

        record = {
            **cfg,
            "val_acc": val_res["accuracy"],
            "val_f1": val_res["f1_weighted"],
            "demo_acc": demo_res["accuracy"],
            "demo_f1": demo_res["f1_weighted"],
            "elapsed": elapsed,
        }
        results.append(record)

        logger.info(
            f"{cfg['name']} | Val Acc={val_res['accuracy']:.4f} "
            f"F1={val_res['f1_weighted']:.4f} | "
            f"Demo Acc={demo_res['accuracy']:.4f} "
            f"F1={demo_res['f1_weighted']:.4f} | "
            f"Time={elapsed:.3f}s"
        )

        if val_res["f1_weighted"] > best_score:
            best_score = val_res["f1_weighted"]
            best_exp = record
            best_pipeline = pipe
            best_demo_report = demo_res["report"]

    if best_pipeline is not None:
        save_model(best_pipeline, "models/best_model.joblib")
        logger.info(
            f"Лучшая модель ({best_exp['name']}) сохранена в "
            "models/best_model.joblib"
        )

    write_text_report(results, best_exp, best_demo_report)
    logger.info("Отчет сохранен в reports/hyperparameters_report.txt")
    logger.info("=== Эксперименты успешно завершены ===")


if __name__ == "__main__":
    run_all_experiments()
