"""Stable console entry point for release bundle operations."""

from .lifecycle.release import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
