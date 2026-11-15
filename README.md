# ML-система классификации отзывов (ЛР6, Вариант 9)

**Дисциплина:** Новые технологии в РПС  
**Вариант:** №9 (Классификация текстовых отзывов Otzovik по 1–5 звёздам)  
**Выполнил:** студент Кондратенко А.С. (гр. M_PIN_RIS_2607)

---

## 📌 Описание работы
Завершающая лабораторная работа, внедряющая модель машинного обучения (**TF-IDF + RandomForestClassifier**) в созданное ранее десктопное приложение PySide6.

### Ключевые этапы:
1. **Предобработка текста (`text_preprocessing.py`):** очистка, удаление стоп-слов (`nltk`) и лемматизация (`pymorphy3`).
2. **ML-конвейер (`ml_service.py`):** стратифицированное разбиение на `train` (163), `val` (35) и `demo` (36) выборки; векторизация `TfidfVectorizer(ngram_range=(1, 2))` + `RandomForestClassifier`.
3. **Серия экспериментов (`run_experiments.py`):** изменение 3 гиперпараметров (`n_estimators`, `max_depth`, `min_samples_split`) с логированием в `logs/experiments.log` и текстовым анализом в `reports/hyperparameters_report.txt`.
4. **Сохранение модели:** лучшая конфигурация (`n_estimators=300`) сохранена в `models/best_model.joblib`.
5. **Интеграция в GUI (`main_window.py`):** добавлена 4-я вкладка для загрузки `.joblib` модели, предсказания оценок на новых отзывах и расчета метрик качества.

---

## 🚀 Запуск и тестирование

```bash
source venv/bin/activate
flake8 text_preprocessing.py ml_service.py run_experiments.py main_window.py main.py tests/test_app.py
python3 -m pytest -v
python3 run_experiments.py
python3 main.py
