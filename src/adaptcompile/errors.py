"""Package-specific exceptions."""


class AdaptCompileError(Exception):
    """Base exception for errors raised by adaptcompile."""


class ValidationError(AdaptCompileError, ValueError):
    """Raised when an adaptcompile value is malformed or inconsistent."""


class SerializationError(AdaptCompileError, TypeError):
    """Raised when a record cannot be represented safely as JSON."""
