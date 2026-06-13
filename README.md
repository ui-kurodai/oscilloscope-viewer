# Oscilloscope Viewer

A PyQt desktop application for loading and plotting oscilloscope waveforms,
saving or copying figures, comparing multiple exports, and calculating the
full width at half maximum (FWHM).

## Features

- Load one or more oscilloscope exports into the same plot
- Add more files without clearing the current plot
- Show or hide individual channels
- Edit the legend, voltage offset, and V/div setting for each trace
- Normalize individual traces to a maximum absolute amplitude of 1
- Edit the complete legend text, including automatically generated details
- Enable automatic legend generation from the current trace settings
- Set persistent X and Y ranges using the fields around the plot
- Set a fixed figure width and width-to-height aspect ratio
- Save the plot or copy it directly to the clipboard
- Calculate FWHM for a selected trace
- Resize or hide the data sidebar

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
uv run python main.py
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

## Project Structure

```text
main.py                       Application entry point
oscilloscope.py               Backward-compatible entry point
oscilloscope_viewer/app.py    PyQt user interface
oscilloscope_viewer/readers.py
oscilloscope_viewer/models.py
tests/
```
