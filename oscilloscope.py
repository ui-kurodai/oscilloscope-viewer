import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from oscilloscope_viewer.readers import READERS

from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QFileDialog,
    QLineEdit,
    QLabel,
    QDoubleSpinBox,
    QCheckBox,
    QSpinBox,
    QMessageBox,
)


def choose_time_unit(t):
    """Choose appropriate time unit and scaling factor."""
    if t.size == 0:
        return "s", 1.0

    span = float(np.nanmax(t) - np.nanmin(t))
    span = abs(span)

    if span < 1e-6:
        return "ns", 1e9
    elif span < 1e-3:
        return "us", 1e6
    elif span < 1.0:
        return "ms", 1e3
    else:
        return "s", 1.0


def choose_unit_for_value(val):
    """Choose time unit for a single positive value."""
    v = abs(val)
    if v < 1e-6:
        return "ns", 1e9
    elif v < 1e-3:
        return "us", 1e6
    elif v < 1.0:
        return "ms", 1e3
    else:
        return "s", 1.0


def format_volt_div(v_val):
    """Format Volt/div value as integer-based text."""
    try:
        v = float(v_val)
    except (ValueError, TypeError):
        return ""

    if v >= 1.0:
        return f"{int(round(v))} V/div"
    elif v >= 1e-3:
        return f"{int(round(v * 1e3))} mV/div"
    elif v >= 1e-6:
        return f"{int(round(v * 1e6))} uV/div"
    else:
        return f"{v:.0e} V/div"


def compute_fwhm(t, y):
    """Compute simple FWHM of a single peak. Returns width in seconds or None."""
    mask = np.isfinite(t) & np.isfinite(y)
    if np.count_nonzero(mask) < 3:
        return None

    t_valid = t[mask]
    y_valid = y[mask]

    if t_valid.size < 3:
        return None

    idx_max = np.argmax(y_valid)
    y_max = y_valid[idx_max]
    if y_max <= 0:
        return None

    half = y_max / 2.0

    # Left crossing
    left_idx = None
    for i in range(idx_max - 1, -1, -1):
        if y_valid[i] < half <= y_valid[i + 1]:
            y1, y2 = y_valid[i], y_valid[i + 1]
            t1, t2 = t_valid[i], t_valid[i + 1]
            if y2 != y1:
                frac = (half - y1) / (y2 - y1)
                t_left = t1 + frac * (t2 - t1)
            else:
                t_left = t1
            left_idx = t_left
            break

    # Right crossing
    right_idx = None
    for i in range(idx_max, len(y_valid) - 1):
        if y_valid[i] >= half > y_valid[i + 1]:
            y1, y2 = y_valid[i], y_valid[i + 1]
            t1, t2 = t_valid[i], t_valid[i + 1]
            if y2 != y1:
                frac = (half - y1) / (y2 - y1)
                t_right = t1 + frac * (t2 - t1)
            else:
                t_right = t2
            right_idx = t_right
            break

    if left_idx is None or right_idx is None:
        return None

    width = right_idx - left_idx
    if width <= 0:
        return None

    return width


class OscilloscopeViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Oscilloscope Viewer")

        self.metadata = None
        self.channels = None
        self.time = None
        self.current_file_dir = "."

        # Font sizes (main text and legend)
        self.ft_main = 14
        self.ft_legend = 14

        # X-range control state
        self.use_custom_x_range = False
        self.custom_x_min = None
        self.custom_x_max = None

        # Time unit state for current plot
        self.current_time_unit = "s"
        self.current_time_scale = 1.0

        # Set up matplotlib figure and canvas
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.ax.set_xlabel("Time [s]", fontsize=self.ft_main)
        self.ax.set_ylabel("Voltage [V]", fontsize=self.ft_main)
        self.ax.tick_params(axis="both", labelsize=self.ft_main)

        # Widgets
        self.open_button = QPushButton("Open file")
        self.save_button = QPushButton("Save figure")
        self.reader_combo = QComboBox()
        self.reader_combo.addItems(READERS.keys())
        self.unified_scale_checkbox = QCheckBox("Use common V scale (no div normalization)")

        # Font size controls
        self.font_spin = QSpinBox()
        self.font_spin.setRange(6, 40)
        self.font_spin.setValue(self.ft_main)

        self.legend_font_spin = QSpinBox()
        self.legend_font_spin.setRange(6, 40)
        self.legend_font_spin.setValue(self.ft_legend)

        # X-range controls
        self.xmin_spin = QDoubleSpinBox()
        self.xmax_spin = QDoubleSpinBox()
        self.apply_xrange_button = QPushButton("Apply X range")
        self.reset_xrange_button = QPushButton("Reset X range")

        self.xmin_spin.setDecimals(6)
        self.xmax_spin.setDecimals(6)
        self.xmin_spin.setRange(-1e9, 1e9)
        self.xmax_spin.setRange(-1e9, 1e9)
        self.xmin_spin.setSingleStep(0.1)
        self.xmax_spin.setSingleStep(0.1)

        # Aspect ratio control
        self.aspect_spin = QDoubleSpinBox()
        self.aspect_spin.setDecimals(2)
        self.aspect_spin.setRange(0.0, 10.0)
        self.aspect_spin.setSingleStep(0.1)
        self.aspect_spin.setValue(0.0)  # 0 = auto

        # Legend text edits, offset and V/div spin boxes per channel
        self.legend_edits = {}
        self.offset_spins = {}
        self.vdiv_spins = {}

        # FWHM labels per channel
        self.fwhm_labels = {}
        self.fwhm_button = QPushButton("Compute FWHM")

        # Layouts
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        controls_widget.setLayout(controls_layout)

        # Top button row (file, save, scale)
        button_row = QHBoxLayout()
        button_row.addWidget(QLabel("Oscilloscope:"))
        button_row.addWidget(self.reader_combo)
        button_row.addWidget(self.open_button)
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.unified_scale_checkbox)
        button_row.addStretch()
        controls_layout.addLayout(button_row)

        # Font and aspect row
        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("Font size:"))
        font_row.addWidget(self.font_spin)
        font_row.addWidget(QLabel("Legend font:"))
        font_row.addWidget(self.legend_font_spin)
        font_row.addWidget(QLabel("Aspect (0=auto):"))
        font_row.addWidget(self.aspect_spin)
        font_row.addStretch()
        controls_layout.addLayout(font_row)

        # X-range row
        xrange_row = QHBoxLayout()
        xrange_row.addWidget(QLabel("X range (current time unit):"))
        xrange_row.addWidget(QLabel("min"))
        xrange_row.addWidget(self.xmin_spin)
        xrange_row.addWidget(QLabel("max"))
        xrange_row.addWidget(self.xmax_spin)
        xrange_row.addWidget(self.apply_xrange_button)
        xrange_row.addWidget(self.reset_xrange_button)
        xrange_row.addStretch()
        controls_layout.addLayout(xrange_row)

        # Channel controls grid + per-channel FWHM
        grid = QGridLayout()
        grid.addWidget(QLabel("Channel"), 0, 0)
        grid.addWidget(QLabel("Legend text"), 0, 1)
        grid.addWidget(QLabel("Offset [V]"), 0, 2)
        grid.addWidget(QLabel("V/div"), 0, 3)
        grid.addWidget(QLabel("FWHM"), 0, 4)
        grid.addWidget(self.fwhm_button, 0, 5)  # Compute FWHM button in header row

        for i in range(4):
            ch_name = f"CH{i+1}"
            row = i + 1

            ch_label = QLabel(ch_name)
            legend_edit = QLineEdit()
            offset_spin = QDoubleSpinBox()
            vdiv_spin = QDoubleSpinBox()
            fwhm_label = QLabel("N/A")  # Per-channel FWHM display (normal font size)

            legend_edit.setPlaceholderText(ch_name)

            offset_spin.setDecimals(6)
            offset_spin.setRange(-10.0, 10.0)
            offset_spin.setSingleStep(0.01)
            offset_spin.setValue(0.0)

            vdiv_spin.setDecimals(6)
            vdiv_spin.setRange(1e-6, 10.0)
            vdiv_spin.setSingleStep(0.01)
            vdiv_spin.setValue(1.0)

            self.legend_edits[ch_name] = legend_edit
            self.offset_spins[ch_name] = offset_spin
            self.vdiv_spins[ch_name] = vdiv_spin
            self.fwhm_labels[ch_name] = fwhm_label

            grid.addWidget(ch_label, row, 0)
            grid.addWidget(legend_edit, row, 1)
            grid.addWidget(offset_spin, row, 2)
            grid.addWidget(vdiv_spin, row, 3)
            grid.addWidget(fwhm_label, row, 4)

            # Update plot when any of these change
            legend_edit.textChanged.connect(self.update_plot)
            offset_spin.valueChanged.connect(self.update_plot)
            vdiv_spin.valueChanged.connect(self.update_plot)

        controls_layout.addLayout(grid)



        # Central layout
        central = QWidget()
        main_layout = QVBoxLayout()
        central.setLayout(main_layout)

        main_layout.addWidget(controls_widget)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)

        self.setCentralWidget(central)

        # Connect signals
        self.open_button.clicked.connect(self.open_file)
        self.save_button.clicked.connect(self.save_figure)
        self.unified_scale_checkbox.toggled.connect(self.update_plot)
        self.apply_xrange_button.clicked.connect(self.apply_x_range)
        self.reset_xrange_button.clicked.connect(self.reset_x_range)
        self.fwhm_button.clicked.connect(self.handle_fwhm)

        self.font_spin.valueChanged.connect(self.update_font_sizes)
        self.legend_font_spin.valueChanged.connect(self.update_font_sizes)
        self.aspect_spin.valueChanged.connect(self.update_plot)

        # Initial empty plot
        self.update_plot()

    def update_font_sizes(self):
        """Update stored font sizes and redraw."""
        self.ft_main = self.font_spin.value()
        self.ft_legend = self.legend_font_spin.value()
        self.update_plot()

    def open_file(self):
        """Open oscilloscope file and update plot."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open oscilloscope data",
            self.current_file_dir,
            "Data files (*.csv *.txt *.dat *.*)",
        )
        if not filename:
            return

        self.current_file_dir = os.path.dirname(filename)

        try:
            reader_name = self.reader_combo.currentText()
            reader = READERS[reader_name]
            data = reader.read(filename)
        except Exception as e:
            print(f"Failed to read file: {e}")
            QMessageBox.critical(
                self,
                "Could not read oscilloscope data",
                f"{reader_name}\n\n{e}",
            )
            return

        self.metadata = data.metadata
        self.channels = data.channels
        self.time = data.time

        print(f"Loaded file: {filename}")
        print("Reader:", reader_name)
        print("Source files:", ", ".join(str(path) for path in data.source_files))
        print("Points:", len(data.time))

        # Reset offsets, legend texts, V/div spins
        for i in range(4):
            ch_name = f"CH{i+1}"
            self.offset_spins[ch_name].setValue(0.0)
            self.legend_edits[ch_name].clear()

            volt_div_key = f"CH{i+1} Volt/div"
            v_str = self.metadata.get(volt_div_key, None)
            if v_str is None and ch_name in self.channels:
                v_str = self.metadata.get("VerticalScale")
            if v_str is not None:
                try:
                    v_val = float(v_str)
                    if v_val > 0:
                        self.vdiv_spins[ch_name].setValue(v_val)
                except ValueError:
                    self.vdiv_spins[ch_name].setValue(1.0)
            else:
                self.vdiv_spins[ch_name].setValue(1.0)

        # Reset x-range & titles
        self.use_custom_x_range = False

        # Reset FWHM display
        for ch_name, label in self.fwhm_labels.items():
            label.setText(f"{ch_name}: N/A")

        self.update_plot()

    def save_figure(self):
        """Save current figure to file."""
        if self.channels is None:
            print("No data to save.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save figure",
            self.current_file_dir,
            "PNG Image (*.png);;PDF File (*.pdf);;SVG File (*.svg);;All Files (*)",
        )
        if not filename:
            return

        try:
            self.fig.savefig(filename, dpi=300)
            print(f"Figure saved to: {filename}")
        except Exception as e:
            print(f"Failed to save figure: {e}")

    def apply_x_range(self):
        """Apply custom x-range based on spin boxes."""
        self.use_custom_x_range = True
        self.custom_x_min = self.xmin_spin.value()
        self.custom_x_max = self.xmax_spin.value()
        if self.custom_x_min >= self.custom_x_max:
            print("Invalid X range: min >= max.")
            return
        self.update_plot()

    def reset_x_range(self):
        """Reset x-range to full data range."""
        self.use_custom_x_range = False
        self.custom_x_min = None
        self.custom_x_max = None
        self.update_plot()

    def update_plot(self):
        """Redraw plot according to current data and GUI settings."""
        self.ax.clear()
        self.ax.tick_params(axis="both", labelsize=self.ft_main)

        if self.channels is None or self.metadata is None or self.time is None:
            self.ax.set_title("No data loaded", fontsize=self.ft_main)
            self.ax.grid(True)
            self.canvas.draw()
            return

        t = self.time

        time_unit, scale = choose_time_unit(t)
        self.current_time_unit = time_unit
        self.current_time_scale = scale
        t_scaled = t * scale

        use_common_v_scale = self.unified_scale_checkbox.isChecked()
        lines = []

        for i in range(4):
            ch_name = f"CH{i+1}"
            if ch_name not in self.channels:
                continue

            y = self.channels[ch_name]
            if y.size == 0 or np.all(np.isnan(y)):
                continue

            offset_v = self.offset_spins[ch_name].value()
            y_v = y + offset_v

            vdiv_override = self.vdiv_spins[ch_name].value()
            if vdiv_override <= 0:
                vdiv_override = 1.0

            if use_common_v_scale:
                y_plot = y_v
            else:
                y_plot = y_v / vdiv_override

            custom_label = self.legend_edits[ch_name].text().strip()
            base_label = custom_label if custom_label else ch_name

            if use_common_v_scale:
                label = base_label
            else:
                volt_div_text = format_volt_div(vdiv_override)
                if volt_div_text:
                    label = f"{base_label} ({volt_div_text})"
                else:
                    label = base_label

            line, = self.ax.plot(t_scaled, y_plot, label=label)
            lines.append(line)

        # Title / labels: use GUI text if non-empty, otherwise auto
        # title_text = "Oscilloscope Data"
        # self.ax.set_title(title_text, fontsize=self.ft_main)

        xlabel_text = f"Time [{time_unit}]"
        self.ax.set_xlabel(xlabel_text, fontsize=self.ft_main)

        if use_common_v_scale:
            default_ylabel = "Voltage [V]"
        else:
            default_ylabel = "Amplitude [div]"

        ylabel_text = default_ylabel
        self.ax.set_ylabel(ylabel_text, fontsize=self.ft_main)

        # Grid
        self.ax.grid(True, which="both", linestyle="--", alpha=0.7)
        # In div mode, make grid lines correspond to 1 div (integer ticks)
        if (not use_common_v_scale) and lines:
            # Get current y-limits and snap ticks to integers
            ymin, ymax = self.ax.get_ylim()
            if np.isfinite(ymin) and np.isfinite(ymax):
                lo = np.floor(ymin)
                hi = np.ceil(ymax)
                if hi > lo:  # avoid zero-length range
                    yticks = np.arange(lo, hi + 1, 1.0)
                    self.ax.set_yticks(yticks)

        if lines:
            self.ax.legend(fontsize=self.ft_legend)

        # X-range
        if self.use_custom_x_range and self.custom_x_min is not None and self.custom_x_max is not None:
            self.ax.set_xlim(self.custom_x_min, self.custom_x_max)
        else:
            if t_scaled.size > 0:
                self.ax.set_xlim(np.nanmin(t_scaled), np.nanmax(t_scaled))

        # Update x-range spin boxes when not using custom
        if not self.use_custom_x_range and t_scaled.size > 0:
            self.xmin_spin.blockSignals(True)
            self.xmax_spin.blockSignals(True)
            self.xmin_spin.setValue(float(np.nanmin(t_scaled)))
            self.xmax_spin.setValue(float(np.nanmax(t_scaled)))
            self.xmin_spin.blockSignals(False)
            self.xmax_spin.blockSignals(False)

        # Aspect ratio
        aspect_val = self.aspect_spin.value()
        if aspect_val > 0.0:
            self.ax.set_aspect(aspect_val)
        else:
            self.ax.set_aspect("auto")

        self.fig.tight_layout()
        self.canvas.draw()

    def handle_fwhm(self):
        """Compute FWHM for all channels and display results."""
        if self.channels is None or self.metadata is None or self.time is None:
            for ch_name, label in self.fwhm_labels.items():
                label.setText(f"{ch_name}: N/A (no data)")
            return

        t = self.time

        for i in range(4):
            ch_name = f"CH{i+1}"
            label = self.fwhm_labels[ch_name]
            if ch_name not in self.channels:
                label.setText(f"{ch_name}: N/A")
                continue

            y = self.channels[ch_name]
            if y.size == 0 or np.all(np.isnan(y)):
                label.setText(f"{ch_name}: N/A")
                continue

            offset_v = self.offset_spins[ch_name].value()
            y_v = y + offset_v

            width = compute_fwhm(t, y_v)
            if width is None:
                label.setText(f"{ch_name}: N/A")
            else:
                unit, scale = choose_unit_for_value(width)
                width_scaled = width * scale
                label.setText(f"{ch_name}: {width_scaled:.3g} {unit}")


def main():
    app = QApplication(sys.argv)
    viewer = OscilloscopeViewer()
    viewer.resize(1200, 800)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
