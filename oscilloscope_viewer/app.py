from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QDoubleValidator, QImage
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .models import OscilloscopeData
from .readers import READERS


def choose_time_unit(time_arrays: list[np.ndarray]) -> tuple[str, float]:
    finite_arrays = [
        values[np.isfinite(values)]
        for values in time_arrays
        if values.size and np.any(np.isfinite(values))
    ]
    if not finite_arrays:
        return "s", 1.0

    minimum = min(float(np.min(values)) for values in finite_arrays)
    maximum = max(float(np.max(values)) for values in finite_arrays)
    span = abs(maximum - minimum)

    if span < 1e-6:
        return "ns", 1e9
    if span < 1e-3:
        return "us", 1e6
    if span < 1.0:
        return "ms", 1e3
    return "s", 1.0


def choose_unit_for_value(value: float) -> tuple[str, float]:
    magnitude = abs(value)
    if magnitude < 1e-6:
        return "ns", 1e9
    if magnitude < 1e-3:
        return "us", 1e6
    if magnitude < 1.0:
        return "ms", 1e3
    return "s", 1.0


def format_volt_div(value: float) -> str:
    if value >= 1.0:
        return f"{int(round(value))} V/div"
    if value >= 1e-3:
        return f"{int(round(value * 1e3))} mV/div"
    if value >= 1e-6:
        return f"{int(round(value * 1e6))} uV/div"
    return f"{value:.0e} V/div"


def compute_fwhm(time: np.ndarray, signal: np.ndarray) -> float | None:
    mask = np.isfinite(time) & np.isfinite(signal)
    if np.count_nonzero(mask) < 3:
        return None

    time_valid = time[mask]
    signal_valid = signal[mask]
    peak_index = int(np.argmax(signal_valid))
    peak = signal_valid[peak_index]
    if peak <= 0:
        return None

    half = peak / 2.0
    left = None
    for index in range(peak_index - 1, -1, -1):
        if signal_valid[index] < half <= signal_valid[index + 1]:
            left = _interpolate_crossing(
                time_valid[index],
                time_valid[index + 1],
                signal_valid[index],
                signal_valid[index + 1],
                half,
            )
            break

    right = None
    for index in range(peak_index, len(signal_valid) - 1):
        if signal_valid[index] >= half > signal_valid[index + 1]:
            right = _interpolate_crossing(
                time_valid[index],
                time_valid[index + 1],
                signal_valid[index],
                signal_valid[index + 1],
                half,
            )
            break

    if left is None or right is None or right <= left:
        return None
    return right - left


def _interpolate_crossing(
    time_1: float,
    time_2: float,
    signal_1: float,
    signal_2: float,
    target: float,
) -> float:
    if signal_2 == signal_1:
        return time_1
    fraction = (target - signal_1) / (signal_2 - signal_1)
    return time_1 + fraction * (time_2 - time_1)


@dataclass
class TraceSettings:
    visible: bool = True
    label: str = ""
    offset: float = 0.0
    volts_per_division: float = 1.0
    automatic_legend: bool = True
    normalize: bool = False


@dataclass
class LoadedDataset:
    data: OscilloscopeData
    display_name: str
    reader_name: str
    traces: dict[str, TraceSettings] = field(default_factory=dict)

    @property
    def identity(self) -> tuple[str, ...]:
        return tuple(
            sorted(str(path.resolve()).casefold() for path in self.data.source_files)
        )


class OscilloscopeViewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Oscilloscope Viewer")

        self.datasets: list[LoadedDataset] = []
        self.current_file_dir = "."
        self.selected_trace: tuple[int, str] | None = None
        self.use_custom_x_range = False
        self.custom_x_min: float | None = None
        self.custom_x_max: float | None = None
        self.use_custom_y_range = False
        self.custom_y_min: float | None = None
        self.custom_y_max: float | None = None
        self.current_time_unit = "s"
        self.current_time_scale = 1.0
        self.ft_main = 14
        self.ft_legend = 14

        self.fig, self.ax = plt.subplots(figsize=(8.0, 8.0 / 1.5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.navigation_toolbar = NavigationToolbar(self.canvas, self)

        self._create_widgets()
        self._create_layout()
        self._connect_signals()
        self._create_menus()
        self._update_figure_size()
        self.statusBar().showMessage("Ready")
        self.update_plot()

    def _create_widgets(self) -> None:
        self.reader_combo = QComboBox()
        self.reader_combo.addItems(READERS.keys())
        self.open_button = QPushButton("Open Files")
        self.add_button = QPushButton("Add Files")
        self.remove_button = QPushButton("Remove")
        self.clear_button = QPushButton("Clear")
        self.save_button = QPushButton("Save Figure")
        self.copy_button = QPushButton("Copy Image")

        self.dataset_tree = QTreeWidget()
        self.dataset_tree.setHeaderLabels(["Files and channels"])
        self.dataset_tree.setSelectionMode(
            QTreeWidget.SelectionMode.ExtendedSelection
        )

        self.trace_group = QGroupBox("Selected Trace")
        self.trace_name_label = QLabel("No trace selected")
        self.legend_edit = QLineEdit()
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setDecimals(6)
        self.offset_spin.setRange(-1e6, 1e6)
        self.offset_spin.setSingleStep(0.01)
        self.vdiv_spin = QDoubleSpinBox()
        self.vdiv_spin.setDecimals(6)
        self.vdiv_spin.setRange(1e-9, 1e6)
        self.vdiv_spin.setSingleStep(0.01)
        self.automatic_legend_checkbox = QCheckBox("Show automatic legend")
        self.automatic_legend_checkbox.setChecked(True)
        self.normalize_checkbox = QCheckBox("Normalize max |amplitude| to 1")
        self.fwhm_button = QPushButton("Compute FWHM")
        self.fwhm_label = QLabel("N/A")
        self._set_trace_controls_enabled(False)

        self.plot_group = QGroupBox("Plot Settings")
        self.unified_scale_checkbox = QCheckBox(
            "Use common voltage scale"
        )
        self.font_spin = QSpinBox()
        self.font_spin.setRange(6, 40)
        self.font_spin.setValue(self.ft_main)
        self.legend_font_spin = QSpinBox()
        self.legend_font_spin.setRange(6, 40)
        self.legend_font_spin.setValue(self.ft_legend)
        self.figure_width_spin = QDoubleSpinBox()
        self.figure_width_spin.setDecimals(1)
        self.figure_width_spin.setRange(2.0, 30.0)
        self.figure_width_spin.setSingleStep(0.5)
        self.figure_width_spin.setSuffix(" in")
        self.figure_width_spin.setValue(8.0)
        self.aspect_spin = QDoubleSpinBox()
        self.aspect_spin.setDecimals(2)
        self.aspect_spin.setRange(0.2, 10.0)
        self.aspect_spin.setSingleStep(0.1)
        self.aspect_spin.setValue(1.5)

        range_validator = QDoubleValidator(-1e300, 1e300, 12, self)
        range_validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
        self.xmin_edit = self._make_range_edit("X min", range_validator)
        self.xmax_edit = self._make_range_edit("X max", range_validator)
        self.ymin_edit = self._make_range_edit("Y min", range_validator)
        self.ymax_edit = self._make_range_edit("Y max", range_validator)

    def _create_layout(self) -> None:
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Oscilloscope:"))
        top_bar.addWidget(self.reader_combo)
        top_bar.addWidget(self.open_button)
        top_bar.addWidget(self.add_button)
        top_bar.addWidget(self.save_button)
        top_bar.addWidget(self.copy_button)
        top_bar.addStretch()

        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.addWidget(QLabel("Loaded Data"))
        sidebar_layout.addWidget(self.dataset_tree, 1)

        dataset_buttons = QHBoxLayout()
        dataset_buttons.addWidget(self.remove_button)
        dataset_buttons.addWidget(self.clear_button)
        sidebar_layout.addLayout(dataset_buttons)

        trace_form = QFormLayout(self.trace_group)
        trace_form.addRow(self.trace_name_label)
        trace_form.addRow("Legend:", self.legend_edit)
        trace_form.addRow("Offset [V]:", self.offset_spin)
        trace_form.addRow("V/div:", self.vdiv_spin)
        trace_form.addRow(self.automatic_legend_checkbox)
        trace_form.addRow(self.normalize_checkbox)
        trace_form.addRow(self.fwhm_button, self.fwhm_label)
        sidebar_layout.addWidget(self.trace_group)

        plot_form = QFormLayout(self.plot_group)
        plot_form.addRow(self.unified_scale_checkbox)
        plot_form.addRow("Font size:", self.font_spin)
        plot_form.addRow("Legend font:", self.legend_font_spin)
        plot_form.addRow("Figure width:", self.figure_width_spin)
        plot_form.addRow("Aspect (W/H):", self.aspect_spin)
        sidebar_layout.addWidget(self.plot_group)

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setWidget(sidebar_content)
        self.sidebar = sidebar_scroll

        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.navigation_toolbar)

        graph_layout = QGridLayout()
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setHorizontalSpacing(6)
        graph_layout.setVerticalSpacing(4)

        y_controls = QVBoxLayout()
        y_controls.setContentsMargins(0, 8, 0, 28)
        y_controls.addWidget(QLabel("Y max"))
        y_controls.addWidget(self.ymax_edit)
        y_controls.addStretch()
        y_controls.addWidget(QLabel("Y min"))
        y_controls.addWidget(self.ymin_edit)
        graph_layout.addLayout(y_controls, 0, 0)
        graph_layout.addWidget(self.canvas, 0, 1)

        x_controls = QHBoxLayout()
        x_controls.setContentsMargins(10, 0, 10, 0)
        x_controls.addWidget(QLabel("X min"))
        x_controls.addWidget(self.xmin_edit)
        x_controls.addStretch()
        x_controls.addWidget(QLabel("X max"))
        x_controls.addWidget(self.xmax_edit)
        graph_layout.addLayout(x_controls, 1, 1)
        graph_layout.setColumnStretch(1, 1)
        graph_layout.setRowStretch(0, 1)
        plot_layout.addLayout(graph_layout, 1)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(plot_widget)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([320, 900])

        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.addLayout(top_bar)
        main_layout.addWidget(self.splitter, 1)
        self.setCentralWidget(central)

    def _connect_signals(self) -> None:
        self.open_button.clicked.connect(lambda: self.open_files(replace=True))
        self.add_button.clicked.connect(lambda: self.open_files(replace=False))
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button.clicked.connect(self.clear_datasets)
        self.save_button.clicked.connect(self.save_figure)
        self.copy_button.clicked.connect(self.copy_image)
        self.dataset_tree.itemChanged.connect(self._tree_item_changed)
        self.dataset_tree.currentItemChanged.connect(self._tree_selection_changed)
        self.legend_edit.textChanged.connect(self._update_selected_trace_settings)
        self.offset_spin.valueChanged.connect(self._update_selected_trace_settings)
        self.vdiv_spin.valueChanged.connect(self._update_selected_trace_settings)
        self.automatic_legend_checkbox.toggled.connect(
            self._automatic_legend_toggled
        )
        self.normalize_checkbox.toggled.connect(
            self._update_selected_trace_settings
        )
        self.fwhm_button.clicked.connect(self.compute_selected_fwhm)
        self.unified_scale_checkbox.toggled.connect(
            self._automatic_legend_context_changed
        )
        self.font_spin.valueChanged.connect(self._update_font_sizes)
        self.legend_font_spin.valueChanged.connect(self._update_font_sizes)
        self.figure_width_spin.valueChanged.connect(self._update_figure_size)
        self.aspect_spin.valueChanged.connect(self._update_figure_size)
        self.xmin_edit.editingFinished.connect(self.apply_x_range)
        self.xmax_edit.editingFinished.connect(self.apply_x_range)
        self.ymin_edit.editingFinished.connect(self.apply_y_range)
        self.ymax_edit.editingFinished.connect(self.apply_y_range)

    def _create_menus(self) -> None:
        view_menu = self.menuBar().addMenu("View")
        self.sidebar_action = QAction("Show Sidebar", self)
        self.sidebar_action.setCheckable(True)
        self.sidebar_action.setChecked(True)
        self.sidebar_action.toggled.connect(self.sidebar.setVisible)
        view_menu.addAction(self.sidebar_action)

    def _make_range_edit(
        self, name: str, validator: QDoubleValidator
    ) -> QLineEdit:
        edit = QLineEdit()
        edit.setValidator(validator)
        edit.setFixedWidth(46)
        edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        edit.setToolTip(
            f"{name}. Press Enter to apply; clear the field for automatic range."
        )
        return edit

    def open_files(self, replace: bool) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Open oscilloscope data",
            self.current_file_dir,
            "Data files (*.csv *.txt *.dat);;All files (*.*)",
        )
        if filenames:
            self.load_files(filenames, replace=replace)

    def load_files(self, filenames: list[str], replace: bool = False) -> None:
        if replace:
            self.datasets.clear()
            self.use_custom_x_range = False
            self.custom_x_min = None
            self.custom_x_max = None
            self.use_custom_y_range = False
            self.custom_y_min = None
            self.custom_y_max = None

        reader_name = self.reader_combo.currentText()
        reader = READERS[reader_name]
        existing = {dataset.identity for dataset in self.datasets}
        loaded_count = 0
        errors: list[str] = []

        for filename in filenames:
            self.current_file_dir = os.path.dirname(filename)
            try:
                data = reader.read(filename)
                dataset = self._make_dataset(data, reader_name)
            except Exception as exc:
                errors.append(f"{Path(filename).name}: {exc}")
                continue

            if dataset.identity in existing:
                continue
            self.datasets.append(dataset)
            existing.add(dataset.identity)
            loaded_count += 1

        self._rebuild_dataset_tree()
        self.update_plot()
        self.statusBar().showMessage(
            f"Loaded {loaded_count} dataset(s); {len(self.datasets)} total",
            5000,
        )

        if errors:
            QMessageBox.warning(
                self,
                "Some files could not be loaded",
                "\n\n".join(errors),
            )

    def _make_dataset(
        self, data: OscilloscopeData, reader_name: str
    ) -> LoadedDataset:
        primary_path = data.source_files[0] if data.source_files else Path("Data")
        display_name = primary_path.name
        traces = {}
        for channel_name in data.channels:
            vdiv = _default_volts_per_division(data.metadata, channel_name)
            traces[channel_name] = TraceSettings(volts_per_division=vdiv)
        dataset = LoadedDataset(data, display_name, reader_name, traces)
        for channel_name, settings in dataset.traces.items():
            settings.label = self._automatic_legend(
                dataset, channel_name, settings
            )
        return dataset

    def _rebuild_dataset_tree(self) -> None:
        self.dataset_tree.blockSignals(True)
        self.dataset_tree.clear()

        for dataset_index, dataset in enumerate(self.datasets):
            parent = QTreeWidgetItem([dataset.display_name])
            parent.setFlags(
                parent.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsAutoTristate
            )
            parent.setCheckState(0, Qt.CheckState.Checked)
            parent.setData(0, Qt.ItemDataRole.UserRole, ("dataset", dataset_index))
            self.dataset_tree.addTopLevelItem(parent)

            for channel_name, settings in dataset.traces.items():
                child = QTreeWidgetItem([channel_name])
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(
                    0,
                    Qt.CheckState.Checked
                    if settings.visible
                    else Qt.CheckState.Unchecked,
                )
                child.setData(
                    0,
                    Qt.ItemDataRole.UserRole,
                    ("trace", dataset_index, channel_name),
                )
                parent.addChild(child)
            parent.setExpanded(True)

        self.dataset_tree.blockSignals(False)
        self.selected_trace = None
        self._show_selected_trace(None)

    def _tree_item_changed(
        self, item: QTreeWidgetItem, column: int
    ) -> None:
        reference = item.data(0, Qt.ItemDataRole.UserRole)
        if reference and reference[0] == "trace":
            _, dataset_index, channel_name = reference
            self.datasets[dataset_index].traces[channel_name].visible = (
                item.checkState(0) == Qt.CheckState.Checked
            )
        self.update_plot()

    def _tree_selection_changed(
        self,
        current: QTreeWidgetItem | None,
        previous: QTreeWidgetItem | None,
    ) -> None:
        del previous
        reference = (
            current.data(0, Qt.ItemDataRole.UserRole) if current else None
        )
        if reference and reference[0] == "trace":
            self.selected_trace = (reference[1], reference[2])
        else:
            self.selected_trace = None
        self._show_selected_trace(self.selected_trace)

    def _show_selected_trace(
        self, reference: tuple[int, str] | None
    ) -> None:
        self.legend_edit.blockSignals(True)
        self.offset_spin.blockSignals(True)
        self.vdiv_spin.blockSignals(True)
        self.automatic_legend_checkbox.blockSignals(True)
        self.normalize_checkbox.blockSignals(True)

        if reference is None:
            self.trace_name_label.setText("No trace selected")
            self.legend_edit.clear()
            self.offset_spin.setValue(0.0)
            self.vdiv_spin.setValue(1.0)
            self.automatic_legend_checkbox.setChecked(True)
            self.normalize_checkbox.setChecked(False)
            self.fwhm_label.setText("N/A")
            self._set_trace_controls_enabled(False)
        else:
            dataset_index, channel_name = reference
            dataset = self.datasets[dataset_index]
            settings = dataset.traces[channel_name]
            self.trace_name_label.setText(
                f"{dataset.display_name} / {channel_name}"
            )
            self.legend_edit.setText(settings.label)
            self.offset_spin.setValue(settings.offset)
            self.vdiv_spin.setValue(settings.volts_per_division)
            self.automatic_legend_checkbox.setChecked(
                settings.automatic_legend
            )
            self.normalize_checkbox.setChecked(settings.normalize)
            self.fwhm_label.setText("N/A")
            self._set_trace_controls_enabled(True)

        self.legend_edit.blockSignals(False)
        self.offset_spin.blockSignals(False)
        self.vdiv_spin.blockSignals(False)
        self.automatic_legend_checkbox.blockSignals(False)
        self.normalize_checkbox.blockSignals(False)

    def _set_trace_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.legend_edit,
            self.offset_spin,
            self.vdiv_spin,
            self.automatic_legend_checkbox,
            self.normalize_checkbox,
            self.fwhm_button,
        ):
            widget.setEnabled(enabled)

    def _update_selected_trace_settings(self) -> None:
        if self.selected_trace is None:
            return
        dataset_index, channel_name = self.selected_trace
        dataset = self.datasets[dataset_index]
        settings = dataset.traces[channel_name]
        if self.sender() is self.legend_edit:
            settings.label = self.legend_edit.text().strip()
            settings.automatic_legend = False
            self.automatic_legend_checkbox.blockSignals(True)
            self.automatic_legend_checkbox.setChecked(False)
            self.automatic_legend_checkbox.blockSignals(False)
        settings.offset = self.offset_spin.value()
        settings.volts_per_division = self.vdiv_spin.value()
        settings.normalize = self.normalize_checkbox.isChecked()
        if settings.automatic_legend:
            settings.label = self._automatic_legend(
                dataset, channel_name, settings
            )
            self._set_legend_edit_text(settings.label)
        self.update_plot()

    def _automatic_legend_toggled(self, checked: bool) -> None:
        if self.selected_trace is None:
            return
        dataset_index, channel_name = self.selected_trace
        dataset = self.datasets[dataset_index]
        settings = dataset.traces[channel_name]
        settings.automatic_legend = checked
        if checked:
            settings.label = self._automatic_legend(
                dataset, channel_name, settings
            )
            self._set_legend_edit_text(settings.label)
        self.update_plot()

    def _automatic_legend_context_changed(self) -> None:
        for dataset in self.datasets:
            for channel_name, settings in dataset.traces.items():
                if settings.automatic_legend:
                    settings.label = self._automatic_legend(
                        dataset, channel_name, settings
                    )
        if self.selected_trace is not None:
            dataset_index, channel_name = self.selected_trace
            settings = self.datasets[dataset_index].traces[channel_name]
            self._set_legend_edit_text(settings.label)
        self.update_plot()

    def _automatic_legend(
        self,
        dataset: LoadedDataset,
        channel_name: str,
        settings: TraceSettings,
    ) -> str:
        parts = [f"{dataset.display_name} / {channel_name}"]
        if not self.unified_scale_checkbox.isChecked():
            parts.append(format_volt_div(settings.volts_per_division))
        return " (".join(parts) + (")" if len(parts) > 1 else "")

    def _set_legend_edit_text(self, text: str) -> None:
        self.legend_edit.blockSignals(True)
        self.legend_edit.setText(text)
        self.legend_edit.blockSignals(False)

    def remove_selected(self) -> None:
        dataset_indexes = set()
        for item in self.dataset_tree.selectedItems():
            reference = item.data(0, Qt.ItemDataRole.UserRole)
            if reference:
                dataset_indexes.add(reference[1])
        if not dataset_indexes:
            return
        self.datasets = [
            dataset
            for index, dataset in enumerate(self.datasets)
            if index not in dataset_indexes
        ]
        self._rebuild_dataset_tree()
        self.update_plot()
        self.statusBar().showMessage("Selected data removed", 3000)

    def clear_datasets(self) -> None:
        self.datasets.clear()
        self._rebuild_dataset_tree()
        self.update_plot()
        self.statusBar().showMessage("All data cleared", 3000)

    def save_figure(self) -> None:
        if not self.datasets:
            self.statusBar().showMessage("No data to save", 3000)
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save figure",
            self.current_file_dir,
            "PNG Image (*.png);;PDF File (*.pdf);;SVG File (*.svg)",
        )
        if not filename:
            return
        try:
            self.fig.savefig(filename, dpi=300)
        except Exception as exc:
            QMessageBox.critical(self, "Could not save figure", str(exc))
            return
        self.statusBar().showMessage(f"Saved {filename}", 5000)

    def copy_image(self) -> None:
        if not self.datasets:
            self.statusBar().showMessage("No data to copy", 3000)
            return
        buffer = io.BytesIO()
        self.fig.savefig(buffer, format="png", dpi=200)
        image = QImage.fromData(buffer.getvalue(), "PNG")
        QApplication.clipboard().setImage(image)
        self.statusBar().showMessage("Plot image copied to clipboard", 5000)

    def apply_x_range(self) -> None:
        parsed = self._parse_range(self.xmin_edit, self.xmax_edit, "X")
        if parsed is None:
            return
        minimum, maximum = parsed
        self.use_custom_x_range = True
        self.custom_x_min = minimum
        self.custom_x_max = maximum
        self.update_plot()

    def reset_x_range(self) -> None:
        self.use_custom_x_range = False
        self.custom_x_min = None
        self.custom_x_max = None
        self.update_plot()

    def apply_y_range(self) -> None:
        parsed = self._parse_range(self.ymin_edit, self.ymax_edit, "Y")
        if parsed is None:
            return
        minimum, maximum = parsed
        self.use_custom_y_range = True
        self.custom_y_min = minimum
        self.custom_y_max = maximum
        self.update_plot()

    def reset_y_range(self) -> None:
        self.use_custom_y_range = False
        self.custom_y_min = None
        self.custom_y_max = None
        self.update_plot()

    def _parse_range(
        self, minimum_edit: QLineEdit, maximum_edit: QLineEdit, axis: str
    ) -> tuple[float, float] | None:
        minimum_text = minimum_edit.text().strip()
        maximum_text = maximum_edit.text().strip()
        if not minimum_text or not maximum_text:
            if axis == "X":
                self.reset_x_range()
            else:
                self.reset_y_range()
            return None

        try:
            minimum = float(minimum_text)
            maximum = float(maximum_text)
        except ValueError:
            self.statusBar().showMessage(f"Invalid {axis} range value", 4000)
            self._restore_range_edits(axis)
            return None

        if minimum >= maximum:
            self.statusBar().showMessage(
                f"Invalid {axis} range: minimum must be less than maximum",
                4000,
            )
            self._restore_range_edits(axis)
            return None
        return minimum, maximum

    def _restore_range_edits(self, axis: str) -> None:
        if axis == "X":
            self._set_x_range_controls(*self.ax.get_xlim())
        else:
            self._set_y_range_controls(*self.ax.get_ylim())

    def _update_font_sizes(self) -> None:
        self.ft_main = self.font_spin.value()
        self.ft_legend = self.legend_font_spin.value()
        self.update_plot()

    def _update_figure_size(self) -> None:
        width_inches = self.figure_width_spin.value()
        aspect = self.aspect_spin.value()
        height_inches = width_inches / aspect
        self.fig.set_size_inches(width_inches, height_inches, forward=True)

        dpi = self.fig.get_dpi()
        self.canvas.setFixedSize(
            max(1, round(width_inches * dpi)),
            max(1, round(height_inches * dpi)),
        )
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def visible_traces(
        self,
    ) -> list[tuple[LoadedDataset, str, TraceSettings]]:
        traces = []
        for dataset in self.datasets:
            for channel_name, settings in dataset.traces.items():
                signal = dataset.data.channels[channel_name]
                if (
                    settings.visible
                    and signal.size
                    and not np.all(np.isnan(signal))
                ):
                    traces.append((dataset, channel_name, settings))
        return traces

    def update_plot(self) -> None:
        self.ax.clear()
        self.ax.tick_params(axis="both", labelsize=self.ft_main)
        visible = self.visible_traces()

        if not visible:
            self.ax.set_title("No visible data", fontsize=self.ft_main)
            self.ax.set_xlabel("Time [s]", fontsize=self.ft_main)
            self.ax.set_ylabel("Voltage [V]", fontsize=self.ft_main)
            self.ax.grid(True)
            self.fig.tight_layout()
            self.canvas.draw_idle()
            return

        time_unit, scale = choose_time_unit(
            [dataset.data.time for dataset, _, _ in visible]
        )
        self.current_time_unit = time_unit
        self.current_time_scale = scale
        common_scale = self.unified_scale_checkbox.isChecked()

        all_scaled_times = []
        for dataset, channel_name, settings in visible:
            time_scaled = dataset.data.time * scale
            signal = dataset.data.channels[channel_name] + settings.offset
            if common_scale:
                plotted_signal = signal
            else:
                plotted_signal = signal / max(
                    settings.volts_per_division, 1e-12
                )
            if settings.normalize:
                finite = np.abs(plotted_signal[np.isfinite(plotted_signal)])
                maximum = float(np.max(finite)) if finite.size else 0.0
                if maximum > 0:
                    plotted_signal = plotted_signal / maximum

            label = settings.label or "_nolegend_"
            self.ax.plot(time_scaled, plotted_signal, label=label)
            all_scaled_times.append(time_scaled)

        self.ax.set_xlabel(f"Time [{time_unit}]", fontsize=self.ft_main)
        normalized_count = sum(
            settings.normalize for _, _, settings in visible
        )
        if normalized_count == len(visible):
            y_label = "Normalized amplitude"
        elif normalized_count:
            y_label = "Amplitude [mixed scale]"
        else:
            y_label = "Voltage [V]" if common_scale else "Amplitude [div]"
        self.ax.set_ylabel(y_label, fontsize=self.ft_main)
        self.ax.grid(True, which="both", linestyle="--", alpha=0.7)
        if any(settings.label for _, _, settings in visible):
            self.ax.legend(fontsize=self.ft_legend)

        if not common_scale:
            minimum, maximum = self.ax.get_ylim()
            if np.isfinite(minimum) and np.isfinite(maximum):
                low = np.floor(minimum)
                high = np.ceil(maximum)
                if high > low:
                    self.ax.set_yticks(np.arange(low, high + 1, 1.0))

        if (
            self.use_custom_y_range
            and self.custom_y_min is not None
            and self.custom_y_max is not None
        ):
            self.ax.set_ylim(self.custom_y_min, self.custom_y_max)
        else:
            minimum, maximum = self.ax.get_ylim()
            self._set_y_range_controls(minimum, maximum)

        if (
            self.use_custom_x_range
            and self.custom_x_min is not None
            and self.custom_x_max is not None
        ):
            self.ax.set_xlim(self.custom_x_min, self.custom_x_max)
        else:
            minimum = min(float(np.nanmin(values)) for values in all_scaled_times)
            maximum = max(float(np.nanmax(values)) for values in all_scaled_times)
            if maximum > minimum:
                self.ax.set_xlim(minimum, maximum)
            self._set_x_range_controls(minimum, maximum)

        self.ax.set_aspect("auto")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _set_x_range_controls(self, minimum: float, maximum: float) -> None:
        self._set_range_edits(self.xmin_edit, self.xmax_edit, minimum, maximum)

    def _set_y_range_controls(self, minimum: float, maximum: float) -> None:
        self._set_range_edits(self.ymin_edit, self.ymax_edit, minimum, maximum)

    @staticmethod
    def _set_range_edits(
        minimum_edit: QLineEdit,
        maximum_edit: QLineEdit,
        minimum: float,
        maximum: float,
    ) -> None:
        minimum_edit.blockSignals(True)
        maximum_edit.blockSignals(True)
        minimum_edit.setText(f"{minimum:.6g}")
        maximum_edit.setText(f"{maximum:.6g}")
        minimum_edit.blockSignals(False)
        maximum_edit.blockSignals(False)

    def compute_selected_fwhm(self) -> None:
        if self.selected_trace is None:
            return
        dataset_index, channel_name = self.selected_trace
        dataset = self.datasets[dataset_index]
        settings = dataset.traces[channel_name]
        signal = dataset.data.channels[channel_name] + settings.offset
        width = compute_fwhm(dataset.data.time, signal)
        if width is None:
            self.fwhm_label.setText("N/A")
            return
        unit, scale = choose_unit_for_value(width)
        self.fwhm_label.setText(f"{width * scale:.3g} {unit}")


def _default_volts_per_division(
    metadata: dict[str, str], channel_name: str
) -> float:
    values = (
        metadata.get(f"{channel_name} Volt/div"),
        metadata.get("VerticalScale"),
    )
    for value in values:
        if value is None:
            continue
        try:
            parsed = float(value)
        except ValueError:
            continue
        if parsed > 0:
            return parsed
    return 1.0


def main() -> None:
    app = QApplication(sys.argv)
    viewer = OscilloscopeViewer()
    viewer.resize(1280, 820)
    viewer.show()
    sys.exit(app.exec())
