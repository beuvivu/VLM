"""Domain-specific exceptions. The API layer maps them to HTTP status codes."""

from __future__ import annotations


class VQEError(Exception):
    """Base class for all Vietlott-Quant-Engine errors."""


class DataValidationError(VQEError):
    """A draw, ticket or request failed domain validation."""


class SourceError(VQEError):
    """A remote data source failed after all retries or returned unparseable data."""


class StorageError(VQEError):
    """The persistence layer failed."""


class InsufficientDataError(VQEError):
    """Not enough history for the requested computation."""


class LookAheadError(VQEError):
    """A strategy tried to access information from the future during a backtest."""
