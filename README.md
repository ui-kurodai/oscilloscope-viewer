# Oscilloscope Viewer

A PyQt desktop application for loading and plotting oscilloscope waveforms,
saving figures, and calculating the full width at half maximum (FWHM).

## Supported Oscilloscopes

- LeCroy WaveJet 354A
- Rohde & Schwarz RTE1204

Select the oscilloscope model from the `Oscilloscope` menu at the top of the
window before opening an export file.

### LeCroy WaveJet 354A

Select a single CSV export file. The reader treats the lines before the column
header beginning with `Ch1 V` as metadata and the following lines as waveform
data.

### Rohde & Schwarz RTE1204

Keep the two exported files with matching names in the same directory:

```text
measurement.csv
measurement.Wfm.csv
```

You can select either file. The application automatically finds and loads the
matching file.

## Development

Install the dependencies and run the application with:

```powershell
uv sync
uv run oscilloscope.py
```

Run the automated tests with:

```powershell
uv run pytest
```

## Building the Windows Executable

Run:

```powershell
.\build_exe.cmd
```

The executable is created at `dist\OscilloscopeViewer.exe`. It is packaged as
a single file for easy distribution. Startup may be slightly slower than
running the application directly from the Python environment.
