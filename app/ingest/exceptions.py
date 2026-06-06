"""Ingest pipeline exceptions."""


class JobPaused(Exception):
    """Raised when an ingest job is cooperatively paused."""
