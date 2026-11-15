"""Скрипт генерации проверенного jupyter-ноутбука для ЛР6."""

import ast
import json


def create_lab6_notebook() -> None:
    """Формирует и валидирует lab6_notebook.ipynb."""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Лабораторная работа №6. Классификация текстов и подбор гиперпараметров\n",
                "\n",
                "**Дисциплина:** Новые технологии в РПС  \n",
                "**Вариант:** №9 (Текстовые отзывы с сервиса *Otzovik*, классы 1–5 звёзд)  \n",
                "**Выполнил:** студент Кондратенко А.С. (гр. M_PIN_RIS_2607)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import subprocess\n",
                "from pathlib import Path\n",
                "\n",
                "for pkg in ['pymorphy3', 'scikit-learn', 'joblib']:\n",
                "    try:\n",
                "        __import__(pkg.replace('-', '_'))\n",
                "    except ImportError:\n",
                "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])\n",
                "\n",
                "if not Path('ml_service.py').exists():\n",
                "    os.system('rm -rf _tmp_repo')\n",
                "    os.system('git clone https://github.com/Alexeyphysics/ntrps_lab6.git _tmp_repo')\n",
                "    os.system('cp -r _tmp_repo/* ./')\n",
                "    os.system('rm -rf _tmp_repo')\n",
                "\n",
                "print('Окружение и проект ntrps_lab6 готовы!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Подготовка текста, лемматизация и разбиение на 3 выборки"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from text_preprocessing import prepare_ml_dataframe\n",
                "from ml_service import split_dataset_three_way\n",
                "\n",
                "df = prepare_ml_dataframe('dataset')\n",
                "train_df, val_df, demo_df = split_dataset_three_way(df)\n",
                "print(f'Всего записей: {len(df)}')\n",
                "print(f'Train: {len(train_df)} | Validation: {len(val_df)} | Demo: {len(demo_df)}')\n",
                "df[['rating', 'title', 'clean_text']].head(5)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Серия экспериментов: изменение 3 гиперпараметров (TF-IDF + RandomForest)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "from run_experiments import get_experiment_configs\n",
                "from ml_service import create_ml_pipeline, train_model, evaluate_model\n",
                "\n",
                "rows = []\n",
                "for cfg in get_experiment_configs():\n",
                "    pipe = create_ml_pipeline(\n",
                "        n_estimators=cfg['n_estimators'],\n",
                "        max_depth=cfg['max_depth'],\n",
                "        min_samples_split=cfg['min_samples_split']\n",
                "    )\n",
                "    train_model(pipe, train_df)\n",
                "    v_res = evaluate_model(pipe, val_df)\n",
                "    d_res = evaluate_model(pipe, demo_df)\n",
                "    rows.append({\n",
                "        'Эксперимент': cfg['name'],\n",
                "        'Val Acc': round(v_res['accuracy'], 4),\n",
                "        'Val F1': round(v_res['f1_weighted'], 4),\n",
                "        'Demo Acc': round(d_res['accuracy'], 4),\n",
                "        'Demo F1': round(d_res['f1_weighted'], 4)\n",
                "    })\n",
                "\n",
                "exp_df = pd.DataFrame(rows)\n",
                "display(exp_df)\n",
                "\n",
                "# График сравнения результатов экспериментов\n",
                "exp_df.plot(x='Эксперимент', y=['Val F1', 'Demo F1'], kind='bar', figsize=(10, 4.5))\n",
                "plt.title('Сравнение F1-score при различных гиперпараметрах')\n",
                "plt.ylabel('F1-Score (Weighted)')\n",
                "plt.xticks(rotation=25, ha='right')\n",
                "plt.grid(axis='y', linestyle='--', alpha=0.7)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Загрузка лучшей сохраненной модели и тестирование на новых отзывах"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from ml_service import load_model, predict_single_review\n",
                "\n",
                "best_model = load_model('models/best_model.joblib')\n",
                "demo_eval = evaluate_model(best_model, demo_df)\n",
                "print('--- Classification Report на демо-выборке ---')\n",
                "print(demo_eval['report'])\n",
                "\n",
                "# Прогноз на новых отзывах\n",
                "for sample in [\n",
                "    'Ужасный обман, карту заблокировали, поддержка молчит!',\n",
                "    'Отличное отделение, вежливый сотрудник быстро помог с кредитом.'\n",
                "]:\n",
                "    pred, probs, _ = predict_single_review(best_model, sample)\n",
                "    print(f'Отзыв: \"{sample}\" -> Прогноз: ★ {pred} (уверенность: {probs[pred]*100:.1f}%)')"
            ]
        }
    ]

    for cell in cells:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))

    notebook_data = {
        "cells": cells,
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open("lab6_notebook.ipynb", "w", encoding="utf-8") as f_out:
        json.dump(notebook_data, f_out, ensure_ascii=False, indent=1)

    print("Ноутбук lab6_notebook.ipynb успешно создан и проверен!")


if __name__ == "__main__":
    create_lab6_notebook()
