"""Stable console entry point for installation diagnostics."""

from .lifecycle.doctor import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
