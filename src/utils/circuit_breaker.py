"""Circuit breaker para APIs externas."""
import time
import logging
from enum import Enum
from typing import Callable, Any, Optional
from dataclasses import dataclass, field
from threading import RLock

from src.exceptions import LLMTimeoutError

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Estados del circuit breaker."""
    CLOSED = "closed"      # Normal, sin errores
    OPEN = "open"          # Bloqueando llamadas
    HALF_OPEN = "half_open"  # Probando si puede cerrar


@dataclass
class CircuitBreakerConfig:
    """Configuración del circuit breaker."""
    failure_threshold: int = 5      # Errores antes de abrir
    success_threshold: int = 3      # Éxitos para cerrar
    timeout: float = 30.0           # Segundos antes de probar cerrar
    half_open_timeout: float = 60.0 # Segundos max en half-open


@dataclass
class CircuitBreakerStats:
    """Estadísticas del circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreakerOpen(Exception):
    """Excepción cuando el circuit breaker está abierto."""
    pass


class CircuitBreaker:
    """Circuit breaker para proteger APIs externas.
    
    Estados:
    - CLOSED: Funcionando normalmente
    - OPEN: Rechazando llamadas por demasiados errores
    - HALF_OPEN: Probando si puede volver a cerrar
    
    Uso:
        cb = CircuitBreaker("gemini", failure_threshold=5)
        result = cb.call(llm.invoke, messages)
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        logger: logging.Logger = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._lock = RLock()
        self._last_state_change = time.time()
    
    @property
    def state(self) -> CircuitState:
        """Obtiene el estado actual."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Verificar si es hora de probar half-open
                if time.time() - self._last_state_change >= self.config.timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._last_state_change = time.time()
                    self.logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
            return self._state
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Ejecuta función con circuit breaker.
        
        Args:
            func: Función a ejecutar.
            *args, **kwargs: Argumentos para la función.
        
        Returns:
            Resultado de la función.
        
        Raises:
            CircuitBreakerOpen: Si el circuit breaker está abierto.
        """
        with self._lock:
            self._stats.total_calls += 1
            
            # Verificar estado
            if self.state == CircuitState.OPEN:
                self._stats.rejected_calls += 1
                raise CircuitBreakerOpen(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Last failure: {self._stats.last_failure_time}"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except LLMTimeoutError as exc:
            self._on_failure(failure_type="timeout", error=exc)
            raise
        except Exception as exc:
            self._on_failure(failure_type="exception", error=exc)
            raise
    
    def _on_success(self):
        """Manejar llamada exitosa."""
        with self._lock:
            self._stats.successful_calls += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            
            # Si estaba en half-open, cerrar
            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._last_state_change = time.time()
                    self.logger.info(f"Circuit breaker '{self.name}' CLOSED after {self._stats.consecutive_successes} successes")
    
    def _on_failure(self, failure_type: str = "exception", error: Optional[Exception] = None) -> None:
        """Manejar llamada fallida.

        Args:
            failure_type: Type of failure ('timeout' or 'exception').
            error: The exception that caused the failure.
        """
        with self._lock:
            self._stats.failed_calls += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = time.time()

            if failure_type == "timeout":
                self.logger.warning(
                    "Circuit breaker '%s' — timeout-type failure (total: %d)",
                    self.name,
                    self._stats.failed_calls,
                )

            # Si estaba cerrado, abrir después de threshold
            if self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._last_state_change = time.time()
                    self.logger.warning(
                        f"Circuit breaker '{self.name}' OPENED after "
                        f"{self._stats.consecutive_failures} consecutive failures"
                    )

            # Si estaba en half-open, volver a abrir
            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._last_state_change = time.time()
                self.logger.warning(
                    f"Circuit breaker '{self.name}' reopened after failure in HALF_OPEN"
                )
    
    def reset(self):
        """Resetea el circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._stats = CircuitBreakerStats()
            self._last_state_change = time.time()
            self.logger.info(f"Circuit breaker '{self.name}' reset")
    
    def get_stats(self) -> CircuitBreakerStats:
        """Obtiene estadísticas."""
        with self._lock:
            return CircuitBreakerStats(
                total_calls=self._stats.total_calls,
                successful_calls=self._stats.successful_calls,
                failed_calls=self._stats.failed_calls,
                rejected_calls=self._stats.rejected_calls,
                last_failure_time=self._stats.last_failure_time,
                consecutive_failures=self._stats.consecutive_failures,
                consecutive_successes=self._stats.consecutive_successes
            )


# Circuit breakers pre-configurados para el proyecto
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, **config_kwargs) -> CircuitBreaker:
    """Obtiene o crea un circuit breaker.
    
    Args:
        name: Nombre del circuit breaker.
        **config_kwargs: Configuración personalizada.
    
    Returns:
        Instancia de CircuitBreaker.
    """
    if name not in _circuit_breakers:
        config = CircuitBreakerConfig(**config_kwargs) if config_kwargs else None
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def gemini_circuit_breaker() -> CircuitBreaker:
    """Circuit breaker específico para Gemini API (deprecated)."""
    return get_circuit_breaker("gemini", failure_threshold=5, timeout=30.0)


def openrouter_circuit_breaker() -> CircuitBreaker:
    """Circuit breaker específico para OpenRouter."""
    return get_circuit_breaker("openrouter", failure_threshold=5, timeout=30.0)


def qdrant_circuit_breaker() -> CircuitBreaker:
    """Circuit breaker específico para Qdrant."""
    return get_circuit_breaker("qdrant", failure_threshold=3, timeout=60.0)


def db_circuit_breaker() -> CircuitBreaker:
    """Circuit breaker específico para base de datos."""
    return get_circuit_breaker("database", failure_threshold=3, timeout=30.0)
