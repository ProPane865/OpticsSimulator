"""Allow running the GUI with ``python -m gui`` (avoids the double-import
of ``gui.app`` that ``python -m gui.app`` triggers)."""

from .app import main

if __name__ == "__main__":
    raise SystemExit(main())
