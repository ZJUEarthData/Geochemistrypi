"""Stable console entry point for the installation lifecycle."""

from .lifecycle.setup import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
