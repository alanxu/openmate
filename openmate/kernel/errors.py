"""Typed errors for the kernel and adapters."""

from __future__ import annotations


class OpenMateError(Exception):
    """Base class for all OpenMate errors."""


class ConfigError(OpenMateError):
    """Misconfiguration — e.g. a missing API key or unknown adapter."""


class ProviderError(OpenMateError):
    """A model provider call failed."""


class ToolError(OpenMateError):
    """A tool could not be resolved or executed."""
