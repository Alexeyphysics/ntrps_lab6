"""Главное графическое окно приложения (PySide6 / Qt6).

Лабораторная работа №6 по курсу «Новые технологии в РПС».
Выполнил: студент Кондратенко А.С. (гр. M_PIN_RIS_2607)
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Optional

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from dataset_service import (
    copy_dataset_by_class,
    copy_dataset_random_names,
    create_annotation_file,
)
from eda_service import (
    generate_boxplot_image,
    generate_word_freq_barchart,
    get_dataset_statistics,
    get_top_lemmas_by_class,
    load_dataset_dataframe,
)
from ml_service import (
    evaluate_model,
    load_model,
    predict_single_review,
)
from review_iterator import ReviewIterator
from scraper_service import scrape_reviews_batch


class MainWindow(QMainWindow):
    """Главное окно приложения с 4 вкладками функционала ЛР1–ЛР6."""

    def __init__(self) -> None:
        """Инициализация компонентов и автозагрузка лучшей модели."""
        super().__init__()
        self.setWindowTitle(
            "ML-классификатор отзывов - ЛР6 (Кондратенко А.С.)"
        )
        self.setMinimumSize(980, 780)

        default_dir = Path("dataset").resolve()
        self.current_dataset_dir: Path = (
            default_dir if default_dir.exists() else Path.cwd()
        )

        self.iterators: Dict[str, ReviewIterator] = {}
        self._init_iterators()

        self.ml_model: Optional[Any] = None
        self.model_path_str: str = "Модель не загружена"
        default_model = Path("models/best_model.joblib")
        if default_model.exists():
            try:
                self.ml_model = load_model(default_model)
                self.model_path_str = str(default_model.resolve())
            except Exception:
                pass

        self._init_ui()

    def _init_iterators(self) -> None:
        """Сброс и инициализация итераторов по звездам."""
        self.iterators = {
            str(s): ReviewIterator(self.current_dataset_dir, str(s))
            for s in range(1, 6)
        }

    def _init_ui(self) -> None:
        """Настройка интерфейса с 4 вкладками QTabWidget."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        top_group = QGroupBox("Рабочий каталог датасета")
        top_layout = QHBoxLayout(top_group)
        self.path_display = QLineEdit(str(self.current_dataset_dir))
        self.path_display.setReadOnly(True)
        top_layout.addWidget(self.path_display)

        btn_select = QPushButton("Выбрать папку...")
        btn_select.clicked.connect(self._select_dataset_folder)
        top_layout.addWidget(btn_select)
        main_layout.addWidget(top_group)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.tab_dataset = QWidget()
        self._setup_dataset_tab(self.tab_dataset)
        self.tabs.addTab(self.tab_dataset, "1. Датасет (ЛР1-2)")

        self.tab_viewer = QWidget()
        self._setup_viewer_tab(self.tab_viewer)
        self.tabs.addTab(self.tab_viewer, "2. Итератор (ЛР3)")

        self.tab_eda = QWidget()
        self._setup_eda_tab(self.tab_eda)
        self.tabs.addTab(self.tab_eda, "3. Анализ EDA (ЛР4-5)")

        self.tab_ml = QWidget()
        self._setup_ml_tab(self.tab_ml)
        self.tabs.addTab(self.tab_ml, "4. Классификация ML (ЛР6)")

    def _setup_dataset_tab(self, parent: QWidget) -> None:
        """Наполнение вкладки управления датасетом."""
        layout = QVBoxLayout(parent)

        btn_scrape = QPushButton("Запустить Web-Scraping (докачать отзывы)")
        btn_scrape.clicked.connect(self._run_scraper)
        layout.addWidget(btn_scrape)

        btn_annot = QPushButton("Создать CSV файл-аннотацию датасета")
        btn_annot.clicked.connect(self._create_annotation)
        layout.addWidget(btn_annot)

        btn_class = QPushButton("Создать копию: <класс>_<номер>.txt")
        btn_class.clicked.connect(self._create_class_copy)
        layout.addWidget(btn_class)

        btn_rand = QPushButton("Создать копию: случайные номера (0..10000)")
        btn_rand.clicked.connect(self._create_random_copy)
        layout.addWidget(btn_rand)

        layout.addStretch()

    def _setup_viewer_tab(self, parent: QWidget) -> None:
        """Наполнение вкладки просмотра отзывов через итератор."""
        layout = QVBoxLayout(parent)

        btn_panel = QHBoxLayout()
        for star in range(1, 6):
            btn = QPushButton(f"Следующий: ★ {star}")
            btn.clicked.connect(
                lambda ch=False, s=str(star): self._show_next_review(s)
            )
            btn_panel.addWidget(btn)
        layout.addLayout(btn_panel)

        self.info_label = QLabel("Нажмите кнопку класса для просмотра")
        self.info_label.setStyleSheet("font-weight: bold; color: #2980b9;")
        layout.addWidget(self.info_label)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        layout.addWidget(self.text_area)

    def _setup_eda_tab(self, parent: QWidget) -> None:
        """Наполнение вкладки исследовательского анализа данных."""
        layout = QVBoxLayout(parent)

        ctrl_panel = QHBoxLayout()
        btn_stats = QPushButton("Рассчитать статистику (describe)")
        btn_stats.clicked.connect(self._show_eda_stats)
        ctrl_panel.addWidget(btn_stats)

        btn_box = QPushButton("Построить Boxplot длин отзывов")
        btn_box.clicked.connect(self._show_boxplot)
        ctrl_panel.addWidget(btn_box)

        btn_neg = QPushButton("Частотный анализ ★ 1 (NLP)")
        btn_neg.clicked.connect(lambda: self._show_word_freq(1))
        ctrl_panel.addWidget(btn_neg)

        btn_pos = QPushButton("Частотный анализ ★ 5 (NLP)")
        btn_pos.clicked.connect(lambda: self._show_word_freq(5))
        ctrl_panel.addWidget(btn_pos)
        layout.addLayout(ctrl_panel)

        self.eda_text_output = QTextEdit()
        self.eda_text_output.setReadOnly(True)
        self.eda_text_output.setMaximumHeight(180)
        layout.addWidget(self.eda_text_output)

        self.chart_label = QLabel("Здесь будет отображен график")
        self.chart_label.setAlignment(Qt.AlignCenter)
        self.chart_label.setStyleSheet(
            "border: 1px solid #bdc3c7; background: #ffffff;"
        )
        layout.addWidget(self.chart_label)

    def _setup_ml_tab(self, parent: QWidget) -> None:
        """Наполнение вкладки классификации и оценки качества модели."""
        layout = QVBoxLayout(parent)

        model_grp = QGroupBox("Выбор обученной модели")
        m_layout = QHBoxLayout(model_grp)
        self.model_path_edit = QLineEdit(self.model_path_str)
        self.model_path_edit.setReadOnly(True)
        m_layout.addWidget(self.model_path_edit)

        btn_load_model = QPushButton("Загрузить модель (.joblib)...")
        btn_load_model.clicked.connect(self._select_model_file)
        m_layout.addWidget(btn_load_model)
        layout.addWidget(model_grp)

        pred_grp = QGroupBox("Проверка классификации на новом отзыве")
        p_layout = QVBoxLayout(pred_grp)

        self.new_review_input = QTextEdit()
        self.new_review_input.setMaximumHeight(100)
        self.new_review_input.setPlaceholderText(
            "Введите или вставьте текст нового отзыва для классификации..."
        )
        p_layout.addWidget(self.new_review_input)

        p_btns = QHBoxLayout()
        btn_sample_neg = QPushButton("Пример негатива")
        btn_sample_neg.clicked.connect(self._fill_negative_sample)
        p_btns.addWidget(btn_sample_neg)

        btn_sample_pos = QPushButton("Пример позитива")
        btn_sample_pos.clicked.connect(self._fill_positive_sample)
        p_btns.addWidget(btn_sample_pos)

        btn_predict = QPushButton("Предсказать оценку (Классифицировать)")
        btn_predict.setStyleSheet("font-weight: bold; color: #27ae60;")
        btn_predict.clicked.connect(self._run_single_prediction)
        p_btns.addWidget(btn_predict)
        p_layout.addLayout(p_btns)

        self.pred_result_label = QLabel("Прогноз: ожидание ввода")
        self.pred_result_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #2c3e50;"
        )
        p_layout.addWidget(self.pred_result_label)
        layout.addWidget(pred_grp)

        eval_grp = QGroupBox(
            "Оценка качества модели на демонстрационных данных"
        )
        e_layout = QVBoxLayout(eval_grp)

        btn_eval_demo = QPushButton(
            "Оценить качество на демо-выборке (demo_split.csv)"
        )
        btn_eval_demo.clicked.connect(self._evaluate_on_demo_data)
        e_layout.addWidget(btn_eval_demo)

        self.eval_report_area = QTextEdit()
        self.eval_report_area.setReadOnly(True)
        self.eval_report_area.setStyleSheet("font-family: monospace;")
        e_layout.addWidget(self.eval_report_area)
        layout.addWidget(eval_grp)

    def _select_dataset_folder(self) -> None:
        """Выбор папки датасета через диалог."""
        folder = QFileDialog.getExistingDirectory(
            self, "Выберите папку датасета", str(self.current_dataset_dir)
        )
        if folder:
            self.current_dataset_dir = Path(folder).resolve()
            self.path_display.setText(str(self.current_dataset_dir))
            self._init_iterators()

    def _run_scraper(self) -> None:
        """Запуск модуля сбора данных."""
        res = scrape_reviews_batch(
            base_dir=str(self.current_dataset_dir), target_count=50
        )
        QMessageBox.information(
            self, "Скрейпинг завершен", f"Текущий состав датасета:\n{res}"
        )
        self._init_iterators()

    def _create_annotation(self) -> None:
        """Сохранение аннотации датасета."""
        csv_p, _ = QFileDialog.getSaveFileName(
            self, "Сохранить аннотацию", "annotation.csv", "CSV (*.csv)"
        )
        if csv_p:
            df = create_annotation_file(self.current_dataset_dir, csv_p)
            QMessageBox.information(
                self, "Успех", f"Аннотация создана!\nЗаписей: {len(df)}"
            )

    def _create_class_copy(self) -> None:
        """Создание копии класс_номер."""
        target = QFileDialog.getExistingDirectory(self, "Выберите папку")
        if target:
            annot = Path(target) / "annotation_class.csv"
            df = copy_dataset_by_class(
                self.current_dataset_dir, target, annot
            )
            QMessageBox.information(
                self, "Успех", f"Скопировано файлов: {len(df)}"
            )

    def _create_random_copy(self) -> None:
        """Создание копии со случайными номерами."""
        target = QFileDialog.getExistingDirectory(self, "Выберите папку")
        if target:
            annot = Path(target) / "annotation_random.csv"
            df = copy_dataset_random_names(
                self.current_dataset_dir, target, annot
            )
            QMessageBox.information(
                self, "Успех", f"Скопировано файлов: {len(df)}"
            )

    def _show_next_review(self, star: str) -> None:
        """Вывод очередного отзыва через итератор."""
        it = self.iterators.get(star)
        if not it:
            return
        try:
            f_path, title, body = next(it)
            msg = (
                f"Класс ★ {star} | Отзыв {it.cursor}/{len(it)} | "
                f"{Path(f_path).name}"
            )
            self.info_label.setText(msg)
            self.text_area.setPlainText(
                f"ЗАГОЛОВОК:\n{title}\n\nТЕКСТ:\n{body}"
            )
        except StopIteration:
            QMessageBox.information(
                self, "Завершено", f"Отзывы класса ★ {star} исчерпаны!"
            )
            it.reset()

    def _show_eda_stats(self) -> None:
        """Вывод описательной статистики."""
        df = load_dataset_dataframe(self.current_dataset_dir)
        text = get_dataset_statistics(df)
        self.eda_text_output.setPlainText(text)

    def _show_boxplot(self) -> None:
        """Генерация и отображение графика Boxplot."""
        df = load_dataset_dataframe(self.current_dataset_dir)
        if df.empty:
            QMessageBox.warning(self, "Ошибка", "Датасет пуст!")
            return
        img_p = generate_boxplot_image(df, "boxplot_gui.png")
        pix = QPixmap(img_p)
        self.chart_label.setPixmap(
            pix.scaled(
                self.chart_label.width(), 380, Qt.KeepAspectRatio
            )
        )

    def _show_word_freq(self, star: int) -> None:
        """Генерация и отображение частотной гистограммы слов."""
        df = load_dataset_dataframe(self.current_dataset_dir)
        if df.empty:
            QMessageBox.warning(self, "Ошибка", "Датасет пуст!")
            return
        freq = get_top_lemmas_by_class(df, star, top_n=12)
        img_p = generate_word_freq_barchart(
            freq, star, f"top_words_{star}_gui.png"
        )
        if img_p:
            pix = QPixmap(img_p)
            self.chart_label.setPixmap(
                pix.scaled(
                    self.chart_label.width(), 380, Qt.KeepAspectRatio
                )
            )

    def _select_model_file(self) -> None:
        """Диалог выбора файла модели .joblib."""
        f_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл модели", "models", "Joblib (*.joblib)"
        )
        if not f_path:
            return
        try:
            self.ml_model = load_model(f_path)
            self.model_path_str = f_path
            self.model_path_edit.setText(f_path)
            QMessageBox.information(
                self, "Успех", f"Модель успешно загружена:\n{f_path}"
            )
        except Exception as err:
            QMessageBox.critical(
                self, "Ошибка загрузки", f"Не удалось загрузить: {err}"
            )

    def _fill_negative_sample(self) -> None:
        """Заполняет поле текстом негативного отзыва."""
        self.new_review_input.setPlainText(
            "Ужасный обман! Заблокировали карту и списали деньги без причины, "
            "поддержка на сайте не отвечает сутками, сплошные проблемы."
        )

    def _fill_positive_sample(self) -> None:
        """Заполняет поле текстом позитивного отзыва."""
        self.new_review_input.setPlainText(
            "Отличное отделение! Вежливый сотрудник помог быстро оформить "
            "кредит и открыть новый счёт. Очень доволен обслуживанием!"
        )

    def _run_single_prediction(self) -> None:
        """Запускает классификацию введенного отзыва."""
        if self.ml_model is None:
            QMessageBox.warning(
                self, "Внимание", "Сначала загрузите обученную модель!"
            )
            return

        raw_text = self.new_review_input.toPlainText().strip()
        if not raw_text:
            QMessageBox.warning(
                self, "Внимание", "Введите текст отзыва для проверки!"
            )
            return

        pred, probs, lemmas = predict_single_review(self.ml_model, raw_text)
        if pred == 0:
            self.pred_result_label.setText(
                "Недостаточно значимых слов после очистки от стоп-слов."
            )
            return

        probs_str = " | ".join(
            f"★{k}: {v * 100:.1f}%" for k, v in sorted(probs.items())
        )
        result_text = (
            f"Прогноз модели: ★ {pred}   [{probs_str}]\n"
            f"Леммы: {lemmas[:70]}..."
        )
        self.pred_result_label.setText(result_text)

    def _evaluate_on_demo_data(self) -> None:
        """Оценивает качество модели на демонстрационной выборке."""
        if self.ml_model is None:
            QMessageBox.warning(
                self, "Внимание", "Сначала загрузите обученную модель!"
            )
            return

        demo_csv = Path("data/demo_split.csv")
        if not demo_csv.exists():
            f_path, _ = QFileDialog.getOpenFileName(
                self, "Выберите CSV демо-выборки", "", "CSV (*.csv)"
            )
            if not f_path:
                return
            demo_csv = Path(f_path)

        demo_df = pd.read_csv(demo_csv)
        demo_df = demo_df.dropna(subset=["clean_text", "rating"])
        res = evaluate_model(self.ml_model, demo_df)

        output = (
            f"Файл выборки: {demo_csv.name} (Записей: {len(demo_df)})\n"
            f"Accuracy (Точность): {res['accuracy']:.4f}\n"
            f"F1-Score (Weighted): {res['f1_weighted']:.4f}\n\n"
            f"--- Полный отчет классификации (Classification Report) ---\n"
            f"{res['report']}"
        )
        self.eval_report_area.setPlainText(output)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
