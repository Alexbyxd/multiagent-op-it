"""Retry logic con exponential backoff para APIs externas."""
import time
import logging
from functools import wraps
from typing import Callable, Any, Type, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from src.exceptions import LLMError, VectorStoreError, DatabaseError, ToolError

logger = logging.getLogger(__name__)


# Configuración de retry
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MIN_WAIT = 1  # segundos
DEFAULT_MAX_WAIT = 10  # segundos
DEFAULT_MULTIPLIER = 2


# Excepciones que deben trigger retry
RETRY_EXCEPTIONS = (
    LLMError,
    VectorStoreError,
    DatabaseError,
    ToolError,
)


def create_retry_decorator(
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    min_wait: int = DEFAULT_MIN_WAIT,
    max_wait: int = DEFAULT_MAX_WAIT,
    multiplier: int = DEFAULT_MULTIPLIER,
    exceptions: tuple = RETRY_EXCEPTIONS
) -> Callable:
    """Crea un decorador de retry con exponential backoff.
    
    Args:
        max_attempts: Máximo número de intentos.
        min_wait: Espera mínima entre intentos (segundos).
        max_wait: Espera máxima entre intentos (segundos).
        multiplier: Multiplicador para exponential backoff.
        exceptions: Tupla de excepciones que trigger retry.
    
    Returns:
        Decorador de retry configurado.
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=multiplier,
            min=min_wait,
            max=max_wait
        ),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )


# Decoradores pre-configurados
def retry_on_llm_error(max_attempts: int = DEFAULT_MAX_ATTEMPTS):
    """Decorador específico para errores de LLM."""
    return create_retry_decorator(
        max_attempts=max_attempts,
        exceptions=(LLMError,)
    )


def retry_on_vector_error(max_attempts: int = DEFAULT_MAX_ATTEMPTS):
    """Decorador específico para errores de vector store."""
    return create_retry_decorator(
        max_attempts=max_attempts,
        exceptions=(VectorStoreError,)
    )


def retry_on_db_error(max_attempts: int = DEFAULT_MAX_ATTEMPTS):
    """Decorador específico para errores de base de datos."""
    return create_retry_decorator(
        max_attempts=max_attempts,
        exceptions=(DatabaseError,)
    )


def retry_on_any_error(max_attempts: int = DEFAULT_MAX_ATTEMPTS):
    """Decorador para reintentar en cualquier error."""
    return create_retry_decorator(
        max_attempts=max_attempts,
        exceptions=(Exception,)
    )


def retry_with_backoff(
    func: Callable,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = RETRY_EXCEPTIONS
) -> Any:
    """Ejecuta una función con retry manual.
    
    Útil para casos donde no se puede usar decorador.
    
    Args:
        func: Función a ejecutar.
        max_attempts: Máximo número de intentos.
        initial_delay: Delay inicial en segundos.
        backoff_factor: Factor de backoff exponencial.
        exceptions: Excepciones que trigger retry.
    
    Returns:
        Resultado de la función.
    
    Raises:
        La última excepción si todos los intentos fallan.
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt == max_attempts:
                logger.error(f"Todos los {max_attempts} intentos fallaron")
                raise
            
            logger.warning(f"Intento {attempt}/{max_attempts} falló: {e}. Reintentando en {delay}s...")
            time.sleep(delay)
            delay *= backoff_factor
    
    raise last_exception


class RetryContext:
    """Contexto para retry con tracking de intentos.
    
    Uso:
        with RetryContext(max_attempts=3) as ctx:
            ctx.execute(llm.invoke, messages)
    """
    
    def __init__(self, max_attempts: int = DEFAULT_MAX_ATTEMPTS):
        self.max_attempts = max_attempts
        self.attempt = 0
        self.last_error: Optional[Exception] = None
    
    def __enter__(self):
        self.attempt = 0
        self.last_error = None
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.last_error = exc_val
        return False  # No suprimir excepciones
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Ejecuta función con retry."""
        self.attempt += 1
        
        try:
            return func(*args, **kwargs)
        except RETRY_EXCEPTIONS as e:
            self.last_error = e
            
            if self.attempt >= self.max_attempts:
                logger.error(f"Intentos agotados ({self.max_attempts})")
                raise
            
            delay = min(2 ** self.attempt, 10)
            logger.warning(f"Intento {self.attempt} falló: {e}. Reintentando en {delay}s...")
            time.sleep(delay)
            raise
