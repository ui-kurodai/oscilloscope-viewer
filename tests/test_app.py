import os
from pathlib import Path

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from oscilloscope_viewer.app import OscilloscopeViewer


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def application() -> QApplication:
    app = QApplication.instance() or QApplication([])
    return app


def _write_lecroy_export(path: Path, multiplier: float) -> None:
    path.write_text(
        "\n".join(
            [
                "ModelName,WaveJet 354A",
                "Delta(second),0.5",
                "Trigger Address,1",
                "CH1 Volt/div,0.2",
                "CH2 Volt/div,0.5",
                "Ch1 V,Ch2 V",
                f"{1 * multiplier},{2 * multiplier}",
                f"{3 * multiplier},{4 * multiplier}",
                f"{1 * multiplier},{2 * multiplier}",
            ]
        ),
        encoding="utf-8",
    )


def test_viewer_loads_and_overlays_multiple_files(
    application: QApplication, tmp_path: Path
) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_lecroy_export(first, 1.0)
    _write_lecroy_export(second, 2.0)

    viewer = OscilloscopeViewer()
    viewer.load_files([str(first), str(second)], replace=True)

    assert len(viewer.datasets) == 2
    assert viewer.dataset_tree.topLevelItemCount() == 2
    assert len(viewer.visible_traces()) == 4
    assert len(viewer.ax.lines) == 4
    viewer.close()


def test_channel_visibility_and_copy_image(
    application: QApplication, tmp_path: Path
) -> None:
    export = tmp_path / "waveform.csv"
    _write_lecroy_export(export, 1.0)

    viewer = OscilloscopeViewer()
    viewer.load_files([str(export)], replace=True)
    first_channel = viewer.dataset_tree.topLevelItem(0).child(0)
    first_channel.setCheckState(0, Qt.CheckState.Unchecked)

    assert len(viewer.visible_traces()) == 1
    viewer.copy_image()
    assert not QApplication.clipboard().image().isNull()
    viewer.close()


def test_trace_normalization_and_editable_automatic_legend(
    application: QApplication, tmp_path: Path
) -> None:
    export = tmp_path / "normalized.csv"
    _write_lecroy_export(export, 1.0)

    viewer = OscilloscopeViewer()
    viewer.load_files([str(export)], replace=True)
    first_channel = viewer.dataset_tree.topLevelItem(0).child(0)
    viewer.dataset_tree.setCurrentItem(first_channel)

    assert "200 mV/div" in viewer.legend_edit.text()
    legend_before_normalization = viewer.legend_edit.text()
    viewer.normalize_checkbox.setChecked(True)

    plotted = viewer.ax.lines[0]
    assert max(abs(plotted.get_ydata())) == pytest.approx(1.0)
    assert plotted.get_label() == legend_before_normalization
    assert plotted.get_label() == viewer.legend_edit.text()

    viewer.legend_edit.setText("My edited legend (custom)")
    assert not viewer.automatic_legend_checkbox.isChecked()
    viewer.vdiv_spin.setValue(0.5)
    assert viewer.ax.lines[0].get_label() == "My edited legend (custom)"

    viewer.automatic_legend_checkbox.setChecked(True)
    assert "500 mV/div" in viewer.legend_edit.text()
    assert viewer.ax.get_ylabel() == "Amplitude [mixed scale]"
    viewer.close()


def test_custom_y_range_survives_plot_updates(
    application: QApplication, tmp_path: Path
) -> None:
    export = tmp_path / "yrange.csv"
    _write_lecroy_export(export, 1.0)

    viewer = OscilloscopeViewer()
    viewer.load_files([str(export)], replace=True)
    first_channel = viewer.dataset_tree.topLevelItem(0).child(0)
    viewer.dataset_tree.setCurrentItem(first_channel)

    viewer.ymin_edit.setText("-2")
    viewer.ymax_edit.setText("5")
    viewer.apply_y_range()
    viewer.legend_edit.setText("Updated legend")

    assert viewer.use_custom_y_range
    assert viewer.ax.get_ylim() == pytest.approx((-2.0, 5.0))

    viewer.reset_y_range()
    assert not viewer.use_custom_y_range
    assert viewer.ax.get_ylim() != pytest.approx((-2.0, 5.0))
    viewer.close()


def test_clearing_plot_edge_range_restores_auto_range(
    application: QApplication, tmp_path: Path
) -> None:
    export = tmp_path / "edge-range.csv"
    _write_lecroy_export(export, 1.0)

    viewer = OscilloscopeViewer()
    viewer.load_files([str(export)], replace=True)
    viewer.xmin_edit.setText("-0.25")
    viewer.xmax_edit.setText("0.25")
    viewer.apply_x_range()
    assert viewer.ax.get_xlim() == pytest.approx((-0.25, 0.25))

    viewer.xmin_edit.clear()
    viewer.apply_x_range()
    assert not viewer.use_custom_x_range
    assert viewer.ax.get_xlim() != pytest.approx((-0.25, 0.25))
    viewer.close()


def test_figure_width_and_aspect_control_canvas_size(
    application: QApplication,
) -> None:
    viewer = OscilloscopeViewer()
    viewer.figure_width_spin.setValue(6.0)
    viewer.aspect_spin.setValue(2.0)

    width, height = viewer.fig.get_size_inches()
    assert width == pytest.approx(6.0)
    assert height == pytest.approx(3.0)
    assert viewer.canvas.width() == 600
    assert viewer.canvas.height() == 300
    assert viewer.xmin_edit.width() == 46
    viewer.close()
