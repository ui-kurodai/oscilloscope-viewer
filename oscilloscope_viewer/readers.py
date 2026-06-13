from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .models import OscilloscopeData


class OscilloscopeReadError(ValueError):
    """Raised when an oscilloscope export cannot be parsed."""


def _read_text(path: Path) -> str:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "cp932"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise OscilloscopeReadError(
        f"Could not decode '{path.name}' as UTF-8 or CP932."
    ) from last_error


class LeCroyWaveJetReader:
    display_name = "LeCroy WaveJet 354A"

    @classmethod
    def read(cls, filepath: str | Path) -> OscilloscopeData:
        path = Path(filepath)
        metadata: dict[str, str] = {}
        channel_header: list[str] | None = None
        data_rows: list[list[str]] = []

        for row in csv.reader(_read_text(path).splitlines()):
            parts = [part.strip() for part in row]
            if not parts or not any(parts):
                continue

            if channel_header is None:
                if parts[0].casefold() == "ch1 v":
                    channel_header = parts
                elif len(parts) >= 2:
                    metadata[parts[0]] = parts[1]
            else:
                data_rows.append(parts)

        if channel_header is None:
            raise OscilloscopeReadError(
                "Channel header beginning with 'Ch1 V' was not found."
            )

        column_count = len(channel_header)
        float_rows: list[list[float]] = []
        for row_number, row in enumerate(data_rows, start=1):
            if len(row) != column_count:
                raise OscilloscopeReadError(
                    f"Waveform row {row_number} has {len(row)} columns; "
                    f"expected {column_count}."
                )
            try:
                float_rows.append(
                    [np.nan if value == "" else float(value) for value in row]
                )
            except ValueError as exc:
                raise OscilloscopeReadError(
                    f"Waveform row {row_number} contains a non-numeric value."
                ) from exc

        if not float_rows:
            raise OscilloscopeReadError("The file contains no waveform data.")

        data_array = np.asarray(float_rows, dtype=float)
        channels = {
            f"CH{index + 1}": data_array[:, index]
            for index in range(column_count)
        }

        delta = _metadata_float(metadata, "Delta(second)", default=1.0)
        trigger_index = _metadata_int(metadata, "Trigger Address", default=0)
        trigger_index = max(0, min(trigger_index, len(data_array) - 1))
        time = (np.arange(len(data_array), dtype=float) - trigger_index) * delta

        return OscilloscopeData(
            metadata=metadata,
            time=time,
            channels=channels,
            source_files=(path,),
        )


class RTE1204Reader:
    display_name = "Rohde & Schwarz RTE1204"
    waveform_suffix = ".Wfm.csv"

    @classmethod
    def read(cls, filepath: str | Path) -> OscilloscopeData:
        selected_path = Path(filepath)
        metadata_path, waveform_path = cls._resolve_pair(selected_path)
        metadata = cls._read_metadata(metadata_path)
        waveform = cls._read_waveform(waveform_path)
        time = cls._build_time_axis(metadata, waveform.size)

        source = metadata.get("Source", "Ch1").strip()
        channel_name = source.upper() if source else "CH1"
        if not channel_name.startswith("CH"):
            channel_name = "CH1"

        return OscilloscopeData(
            metadata=metadata,
            time=time,
            channels={channel_name: waveform},
            source_files=(metadata_path, waveform_path),
        )

    @classmethod
    def _resolve_pair(cls, selected_path: Path) -> tuple[Path, Path]:
        name_lower = selected_path.name.lower()
        suffix_lower = cls.waveform_suffix.lower()

        if name_lower.endswith(suffix_lower):
            metadata_path = selected_path.with_name(
                selected_path.name[: -len(cls.waveform_suffix)] + ".csv"
            )
            waveform_path = selected_path
        elif selected_path.suffix.lower() == ".csv":
            metadata_path = selected_path
            waveform_path = selected_path.with_name(
                selected_path.stem + cls.waveform_suffix
            )
        else:
            raise OscilloscopeReadError(
                "Select either the metadata .csv file or the matching .Wfm.csv file."
            )

        missing = [
            path.name for path in (metadata_path, waveform_path) if not path.is_file()
        ]
        if missing:
            raise OscilloscopeReadError(
                "Matching RTE1204 export file not found: " + ", ".join(missing)
            )
        return metadata_path, waveform_path

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, str]:
        metadata: dict[str, str] = {}
        for line in _read_text(path).splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            key, separator, remainder = stripped.partition(":")
            if not separator:
                continue
            metadata[key.strip()] = remainder.rstrip(":").strip()

        required = ("SignalResolution", "Source")
        missing = [key for key in required if key not in metadata]
        if missing:
            raise OscilloscopeReadError(
                "RTE1204 metadata is missing: " + ", ".join(missing)
            )
        return metadata

    @staticmethod
    def _read_waveform(path: Path) -> np.ndarray:
        values: list[float] = []
        for line_number, line in enumerate(_read_text(path).splitlines(), start=1):
            stripped = line.strip().rstrip(",;")
            if not stripped:
                continue
            try:
                values.append(float(stripped))
            except ValueError as exc:
                raise OscilloscopeReadError(
                    f"Waveform line {line_number} is not numeric."
                ) from exc

        if not values:
            raise OscilloscopeReadError("The waveform file contains no samples.")
        return np.asarray(values, dtype=float)

    @staticmethod
    def _build_time_axis(metadata: dict[str, str], point_count: int) -> np.ndarray:
        resolution = _metadata_float(metadata, "SignalResolution")

        hardware_count = _metadata_int(
            metadata, "SignalHardwareRecordLength", default=-1
        )
        signal_count = _metadata_int(metadata, "SignalRecordLength", default=-1)

        if hardware_count == point_count and "HardwareXStart" in metadata:
            start = _metadata_float(metadata, "HardwareXStart")
        elif signal_count == point_count and "XStart" in metadata:
            start = _metadata_float(metadata, "XStart")
        elif "HardwareXStart" in metadata:
            start = _metadata_float(metadata, "HardwareXStart")
        else:
            start = _metadata_float(metadata, "XStart")

        return start + np.arange(point_count, dtype=float) * resolution


def _metadata_float(
    metadata: dict[str, str], key: str, default: float | None = None
) -> float:
    value = metadata.get(key)
    if value is None:
        if default is not None:
            return default
        raise OscilloscopeReadError(f"Metadata field '{key}' is missing.")
    try:
        return float(value)
    except ValueError as exc:
        raise OscilloscopeReadError(
            f"Metadata field '{key}' is not numeric: {value!r}"
        ) from exc


def _metadata_int(
    metadata: dict[str, str], key: str, default: int | None = None
) -> int:
    value = metadata.get(key)
    if value is None:
        if default is not None:
            return default
        raise OscilloscopeReadError(f"Metadata field '{key}' is missing.")
    try:
        return int(float(value))
    except ValueError as exc:
        raise OscilloscopeReadError(
            f"Metadata field '{key}' is not an integer: {value!r}"
        ) from exc


READERS = {
    LeCroyWaveJetReader.display_name: LeCroyWaveJetReader,
    RTE1204Reader.display_name: RTE1204Reader,
}
