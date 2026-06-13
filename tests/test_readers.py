from pathlib import Path

import numpy as np
import pytest

from oscilloscope_viewer.readers import (
    LeCroyWaveJetReader,
    OscilloscopeReadError,
    RTE1204Reader,
)


def test_lecroy_reader_builds_trigger_relative_time(tmp_path: Path) -> None:
    export = tmp_path / "wavejet.csv"
    export.write_text(
        "\n".join(
            [
                "ModelName,WaveJet 354A",
                "Delta(second),0.5",
                "Trigger Address,1",
                "CH1 Volt/div,0.2",
                "Ch1 V,Ch2 V",
                "1.0,2.0",
                "3.0,",
                "5.0,6.0",
            ]
        ),
        encoding="utf-8",
    )

    data = LeCroyWaveJetReader.read(export)

    np.testing.assert_allclose(data.time, [-0.5, 0.0, 0.5])
    np.testing.assert_allclose(data.channels["CH1"], [1.0, 3.0, 5.0])
    assert np.isnan(data.channels["CH2"][1])
    assert data.metadata["ModelName"] == "WaveJet 354A"


def test_rte_reader_accepts_metadata_file_and_uses_hardware_axis(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "pulse.csv"
    waveform = tmp_path / "pulse.Wfm.csv"
    metadata.write_text(
        "\n".join(
            [
                "Resolution:2e-11:",
                "RecordLength:3:",
                "XStart:-1e-8:",
                "SignalRecordLength:3:",
                "HardwareXStart:-1.12e-8:",
                "SignalHardwareRecordLength:4:",
                "SignalResolution:2e-11:",
                "VerticalScale:0.02:",
                "Source:Ch1:",
            ]
        ),
        encoding="utf-8",
    )
    waveform.write_text("0.1\n0.2\n0.3\n0.4\n", encoding="utf-8")

    data = RTE1204Reader.read(metadata)

    assert data.source_files == (metadata, waveform)
    np.testing.assert_allclose(
        data.time,
        [-1.12e-8, -1.118e-8, -1.116e-8, -1.114e-8],
    )
    np.testing.assert_allclose(data.channels["CH1"], [0.1, 0.2, 0.3, 0.4])


def test_rte_reader_accepts_waveform_file(tmp_path: Path) -> None:
    metadata = tmp_path / "pulse.csv"
    waveform = tmp_path / "pulse.Wfm.csv"
    metadata.write_text(
        "SignalResolution:1e-9:\n"
        "XStart:-1e-9:\n"
        "SignalRecordLength:2:\n"
        "Source:Ch2:\n",
        encoding="utf-8",
    )
    waveform.write_text("1\n2\n", encoding="utf-8")

    data = RTE1204Reader.read(waveform)

    np.testing.assert_allclose(data.time, [-1e-9, 0.0])
    assert list(data.channels) == ["CH2"]


def test_rte_reader_reports_missing_pair(tmp_path: Path) -> None:
    metadata = tmp_path / "pulse.csv"
    metadata.write_text("Source:Ch1:\n", encoding="utf-8")

    with pytest.raises(OscilloscopeReadError, match=r"pulse\.Wfm\.csv"):
        RTE1204Reader.read(metadata)
