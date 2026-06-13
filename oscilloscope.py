"""Backward-compatible entry point. Prefer running main.py."""

from oscilloscope_viewer.app import OscilloscopeViewer, main

__all__ = ["OscilloscopeViewer", "main"]


if __name__ == "__main__":
    main()
