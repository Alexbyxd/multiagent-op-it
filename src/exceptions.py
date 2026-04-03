"""Excepciones personalizadas del proyecto MASO."""


class MasoError(Exception):
    """Error base del proyecto."""
    pass


class DatabaseError(MasoError):
    """Error de base de datos."""
    pass


class VectorStoreError(MasoError):
    """Error del vector store."""
    pass


class LLMError(MasoError):
    """Error del modelo de lenguaje."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when an LLM call exceeds its configured timeout.

    Attributes:
        timeout_seconds: The timeout value that was exceeded.
        model: Optional model identifier for error messages.
    """

    def __init__(self, timeout_seconds: float, model: str = "") -> None:
        self.timeout_seconds = timeout_seconds
        self.model = model
        message = f"LLM call timed out after {timeout_seconds}s"
        if model:
            message += f" (model: {model})"
        super().__init__(message)


class ToolError(MasoError):
    """Error en ejecución de herramienta."""
    pass