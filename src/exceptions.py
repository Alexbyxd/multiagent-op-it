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


class ToolError(MasoError):
    """Error en ejecución de herramienta."""
    pass