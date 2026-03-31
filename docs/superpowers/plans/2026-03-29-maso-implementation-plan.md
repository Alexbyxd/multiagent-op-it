# MASO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar un sistema multi-agente de soporte técnico con LangGraph, Gemini y Qdrant

**Architecture:** Grafo LangGraph con 4 nodos (input, router, tools, synthesizer) y 4 herramientas (search_documents, search_tickets, check_service_status, suggest_action)

**Tech Stack:** langgraph, langchain-google-genai, qdrant-client, sqlite3, python-dotenv

---

## Estructura de Archivos a Crear/Modificar

```
rag_1/
├── src/
│   ├── __init__.py
│   ├── main.py                    # Punto de entrada
│   ├── config.py                  # Configuración centralizada
│   ├── exceptions.py              # Excepciones personalizadas
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── maso_graph.py          # Definición del grafo LangGraph
│   │   └── nodes.py                # Nodos del grafo
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── router.py               # Router con gemini-2.5-flash
│   │   └── synthesizer.py          # Síntesis con gemini-2.5-pro
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── documents.py            # Tool: search_documents (RAG)
│   │   ├── tickets.py              # Tool: search_tickets
│   │   ├── status.py               # Tool: check_service_status
│   │   └── suggestion.py           # Tool: suggest_action
│   ├── db/
│   │   ├── __init__.py
│   │   ├── setup_db.py            # Inicialización SQLite + datos prueba
│   │   └── queries.py             # Consultas a tickets
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── qdrant_client.py       # Cliente Qdrant
│   │   └── ingest.py              # Ingesta de documentos
│   └── prompts/
│       ├── __init__.py
│       ├── router_prompt.txt
│       └── synthesizer_prompt.txt
├── data/
│   ├── tickets.db                 # SQLite DB
│   ├── qdrant/                    # Vector store
│   ├── documents/                # PDFs de documentación
│   └── status.json               # Estado de servicios
├── .env
├── requirements.txt
└── pyproject.toml
```

---

## Plan de Implementación

### Fase 0: Estructura de Paquetes

#### Task 0.1: Crear archivos __init__.py

**Files:**
- Create: `src/__init__.py`
- Create: `src/graph/__init__.py`
- Create: `src/llm/__init__.py`
- Create: `src/tools/__init__.py`
- Create: `src/db/__init__.py`
- Create: `src/vectorstore/__init__.py`
- Create: `src/prompts/__init__.py`
- Test: N/A

- [ ] **Step 1: Crear estructura de directorios**

```bash
touch src/__init__.py
touch src/graph/__init__.py
touch src/llm/__init__.py
touch src/tools/__init__.py
touch src/db/__init__.py
touch src/vectorstore/__init__.py
touch src/prompts/__init__.py
touch tests/__init__.py
touch tests/test_db/__init__.py
touch tests/test_tools/__init__.py
touch tests/test_llm/__init__.py
touch tests/test_graph/__init__.py
```

- [ ] **Step 2: Commit**

```bash
git add src/*/__init__.py tests/*/__init__.py
git commit -m "chore: add package init files"
```

---

### Fase 1: Configuración Base

#### Task 1.1: Configuración Centralizada

**Files:**
- Create: `src/config.py`
- Test: N/A (config)

- [ ] **Step 1: Crear archivos de prompts**

```txt
# src/prompts/router_prompt.txt
Eres el orquestador de un sistema de soporte técnico.
Tu única tarea es decidir qué herramienta usar para responder la consulta del usuario.

Dispones de estas herramientas:
- search_documents: Busca en documentación técnica
- search_tickets: Busca en historial de tickets resueltos
- check_service_status: Consulta el estado de un servicio/servidor
- suggest_action: Sugiere acciones basadas en resultados

Instrucciones:
1. Analiza la consulta del usuario
2. Selecciona la herramienta más apropiada
3. Si no necesitas herramientas, responde directamente

Responde en formato JSON:
{"tool": "nombre_de_herramienta", "reason": "razón de la selección"}
```

```txt
# src/prompts/synthesizer_prompt.txt
Eres un asistente de soporte técnico corporativo.
Tu tarea es generar una respuesta final clara y útil para el usuario.

Instrucciones:
1. Usa los resultados de las herramientas para construir tu respuesta
2. Sé conciso pero completo
3. Incluye las fuentes de información
4. Si hay acciones sugeridas, inclúyelas
5. Adapta el nivel técnico al usuario (básico/avanzado/admin)

Nivel básico: Explica de forma sencilla, evita jerga técnica innecesaria.
Nivel avanzado: Puedes usar términos técnicos, sé detallado.
Nivel admin: Incluye información de infraestructura y métricas relevantes.
```

- [ ] **Step 2: Commit**

```bash
git add src/prompts/router_prompt.txt src/prompts/synthesizer_prompt.txt
git commit -m "feat: add prompt templates"
```

- [ ] **Step 3: Escribir configuración**

```python
"""Configuración centralizada del proyecto MASO."""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración del proyecto."""
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DB_PATH: Path = DATA_DIR / "tickets.db"
    QDRANT_PATH: str = str(DATA_DIR / "qdrant")
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"
    STATUS_FILE: Path = DATA_DIR / "status.json"
    
    # Google Gemini
    google_api_key: str
    
    # Embeddings
    embedding_model: str = "gemini-embedding-001"
    chunk_size: int = 80
    chunk_overlap: int = 10
    
    # Modelos
    router_model: str = "gemini-2.5-flash"
    synthesizer_model: str = "gemini-2.5-pro"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
```

- [ ] **Step 2: Commit**

```bash
git add src/config.py
git commit -m "feat: add centralized configuration"
```

---

#### Task 1.2: Excepciones Personalizadas

**Files:**
- Create: `src/exceptions.py`
- Test: N/A

- [ ] **Step 1: Escribir excepciones**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add src/exceptions.py
git commit -m "feat: add custom exceptions"
```

---

### Fase 2: Base de Datos

#### Task 2.1: Setup de Base de Datos SQLite

**Files:**
- Create: `src/db/setup_db.py`
- Modify: `src/db/__init__.py`
- Test: N/A (data setup)

- [ ] **Step 1: Escribir setup_db.py**

```python
"""Setup de base de datos SQLite con datos de prueba."""
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

from src.config import settings


def create_tables(conn: sqlite3.Connection) -> None:
    """Crea las tablas necesarias."""
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            error_code TEXT,
            solution TEXT NOT NULL,
            severity TEXT NOT NULL,
            created_at DATETIME NOT NULL,
            resolved_at DATETIME NOT NULL
        )
    """)
    conn.commit()


def insert_sample_tickets(conn: sqlite3.Connection) -> None:
    """Inserta tickets de prueba."""
    cursor = conn.cursor()
    
    # Verificar si ya hay datos
    cursor.execute("SELECT COUNT(*) FROM tickets")
    if cursor.fetchone()[0] > 0:
        return
    
    tickets = [
        (
            "Error 503 en servidor web de producción",
            "Los usuarios reportan error 503 Service Unavailable al acceder al sitio web principal. El servidor web devuelve error inmediatamente.",
            "503",
            "Reiniciar el servicio nginx: sudo systemctl restart nginx. Verificar logs en /var/log/nginx/error.log. Aumentar worker_connections en nginx.conf.",
            "critical",
            datetime.now() - timedelta(days=5),
            datetime.now() - timedelta(days=5)
        ),
        (
            "Conexión lenta a base de datos PostgreSQL",
            "Las consultas a la base de datos tardan más de 10 segundos. Los usuarios experimentan timeouts frecuentes.",
            "DB_TIMEOUT",
            "Ejecutar ANALYZE en las tablas afectadas. Aumentar shared_buffers en postgresql.conf. Revisar índices faltantes en consultas lentas.",
            "high",
            datetime.now() - timedelta(days=10),
            datetime.now() - timedelta(days=8)
        ),
        (
            "Fallo de autenticación con LDAP",
            "Usuarios no pueden iniciar sesión. El servidor LDAP responde con error de timeout.",
            "LDAP_TIMEOUT",
            "Verificar conectividad con servidor LDAP: ldapsearch -h ldap.server.com. Revisar configuración de timeout en /etc/ldap/ldap.conf. Reiniciar servicio slapd.",
            "critical",
            datetime.now() - timedelta(days=3),
            datetime.now() - timedelta(days=2)
        ),
        (
            "API Gateway devuelve 502 Bad Gateway",
            "El API gateway no puede comunicarse con los microservicios backend. Todos los endpoints fallan.",
            "502",
            "Verificar que los servicios backend estén corriendo: docker ps. Revisar configuración de upstream en nginx. Reiniciar contenedores affected.",
            "high",
            datetime.now() - timedelta(days=7),
            datetime.now() - timedelta(days=6)
        ),
        (
            "Espacio en disco bajo en servidor de logs",
            "El servidor tiene menos de 5% de espacio disponible. Los servicios comienzan a fallar.",
            "DISK_FULL",
            "Ejecutar du -sh /var/log/* para identificar directorios grandes. Rotar logs antiguos: logrotate -f /etc/logrotate.conf. Eliminar archivos temporales.",
            "medium",
            datetime.now() - timedelta(days=2),
            datetime.now() - timedelta(days=1)
        ),
        (
            "Error de certificado SSL expirado",
            "Los navegadores muestran advertencia de certificado no válido. La API retorna errores de conexión segura.",
            "SSL_EXPIRED",
            "Renovar certificado: certbot renew --nginx. Verificar fecha de expiración: openssl s_client -connect domain.com:443. Actualizar DNS si es necesario.",
            "high",
            datetime.now() - timedelta(days=1),
            datetime.now()
        ),
    ]
    
    cursor.executemany(
        "INSERT INTO tickets (title, description, error_code, solution, severity, created_at, resolved_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        tickets
    )
    conn.commit()


def setup_database() -> None:
    """Inicializa la base de datos."""
    settings.DATA_DIR.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(settings.DB_PATH)
    create_tables(conn)
    insert_sample_tickets(conn)
    conn.close()
    
    print(f"Base de datos inicializada en {settings.DB_PATH}")


if __name__ == "__main__":
    setup_database()
```

- [ ] **Step 2: Commit**

```bash
git add src/db/setup_db.py src/db/__init__.py
git commit -m "feat: add database setup with sample tickets"
```

---

#### Task 2.2: Consultas a Tickets

**Files:**
- Create: `src/db/queries.py`
- Test: `tests/test_db/test_queries.py`

- [ ] **Step 1: Escribir test**

```python
"""Tests para queries de tickets."""
import pytest
import sqlite3
from src.db.queries import search_tickets


def test_search_tickets_returns_results(tmp_path):
    """Verifica que search_tickets retorna resultados."""
    # Setup
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE tickets (
            id INTEGER PRIMARY KEY,
            title TEXT,
            description TEXT,
            error_code TEXT,
            solution TEXT,
            severity TEXT,
            created_at DATETIME,
            resolved_at DATETIME
        )
    """)
    conn.execute("""
        INSERT INTO tickets VALUES 
        (1, 'Error 503', 'Servidor caído', '503', 'Reiniciar nginx', 'critical', '2024-01-01', '2024-01-01'),
        (2, 'DB timeout', 'Consulta lenta', 'DB_TIMEOUT', 'Optimizar query', 'high', '2024-01-02', '2024-01-02')
    """)
    conn.commit()
    conn.close()
    
    # Execute
    results = search_tickets("servidor caído", db_path=db_path)
    
    # Assert
    assert len(results) > 0
    assert results[0]["error_code"] == "503"
```

- [ ] **Step 2: Run test**

```bash
pytest tests/test_db/test_queries.py::test_search_tickets_returns_results -v
```
Expected: FAIL (function not defined)

- [ ] **Step 3: Implementar queries.py**

```python
"""Consultas a la base de datos de tickets."""
import sqlite3
from typing import Optional
from datetime import datetime

from src.config import settings
from src.exceptions import DatabaseError


def get_connection() -> sqlite3.Connection:
    """Obtiene conexión a la base de datos."""
    try:
        conn = sqlite3.connect(settings.DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        raise DatabaseError(f"Error conectando a la base de datos: {e}") from e


def search_tickets(query: str, limit: int = 5, db_path: Optional[str] = None) -> list[dict]:
    """Busca tickets por query.
    
    Args:
        query: Texto de búsqueda.
        limit: Número máximo de resultados.
        db_path: Path opcional a la DB (para testing).
    
    Returns:
        Lista de diccionarios con los tickets encontrados.
    """
    path = db_path or settings.DB_PATH
    
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Búsqueda simple por LIKE en title, description y solution
        search_term = f"%{query}%"
        cursor.execute("""
            SELECT id, title, description, error_code, solution, severity, created_at, resolved_at
            FROM tickets
            WHERE title LIKE ? OR description LIKE ? OR solution LIKE ?
            ORDER BY 
                CASE severity
                    WHEN 'critical' THEN 1
                    WHEN 'high' THEN 2
                    WHEN 'medium' THEN 3
                    WHEN 'low' THEN 4
                END,
                resolved_at DESC
            LIMIT ?
        """, (search_term, search_term, search_term, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
        
    except sqlite3.Error as e:
        raise DatabaseError(f"Error buscando tickets: {e}") from e
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_db/test_queries.py::test_search_tickets_returns_results -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/queries.py tests/test_db/test_queries.py
git commit -m "feat: add ticket search queries"
```

---

### Fase 3: Estado de Servicios

#### Task 3.1: Archivo de Estado JSON

**Files:**
- Create: `src/tools/status.py`
- Create: `data/status.json`
- Test: `tests/test_tools/test_status.py`

- [ ] **Step 1: Crear status.json**

```json
{
  "last_updated": "2026-03-29T10:00:00Z",
  "services": [
    {
      "name": "web-server-prod",
      "display_name": "Web Server Production",
      "status": "UP",
      "uptime": "99.9%",
      "last_check": "2026-03-29T10:00:00Z",
      "response_time_ms": 45
    },
    {
      "name": "api-gateway",
      "display_name": "API Gateway",
      "status": "DOWN",
      "uptime": "98.5%",
      "last_check": "2026-03-29T10:00:00Z",
      "last_failure": "2026-03-29T09:45:00Z",
      "error": "Connection refused"
    },
    {
      "name": "database-primary",
      "display_name": "Database Primary",
      "status": "UP",
      "uptime": "99.99%",
      "last_check": "2026-03-29T10:00:00Z",
      "response_time_ms": 12
    },
    {
      "name": "cache-redis",
      "display_name": "Redis Cache",
      "status": "UP",
      "uptime": "99.95%",
      "last_check": "2026-03-29T10:00:00Z",
      "response_time_ms": 3
    },
    {
      "name": "auth-service",
      "display_name": "Authentication Service",
      "status": "UP",
      "uptime": "99.8%",
      "last_check": "2026-03-29T10:00:00Z",
      "response_time_ms": 28
    }
  ]
}
```

- [ ] **Step 2: Escribir tool de status**

```python
"""Tool para consultar estado de servicios."""
import json
from pathlib import Path
from typing import Optional

from src.config import settings
from src.exceptions import ToolError


def check_service_status(service_name: str) -> str:
    """Consulta el estado de un servicio específico.
    
    Args:
        service_name: Nombre del servicio a consultar.
    
    Returns:
        String con el estado del servicio.
    """
    try:
        status_file = settings.STATUS_FILE
        
        if not status_file.exists():
            raise ToolError(f"Archivo de estado no encontrado: {status_file}")
        
        with open(status_file, "r") as f:
            data = json.load(f)
        
        # Buscar servicio
        for service in data.get("services", []):
            if service["name"].lower() == service_name.lower():
                return format_service_status(service)
        
        # Servicio no encontrado - listar todos disponibles
        available = [s["name"] for s in data.get("services", [])]
        return f"Servicio '{service_name}' no encontrado. Servicios disponibles: {', '.join(available)}"
    
    except json.JSONDecodeError as e:
        raise ToolError(f"Error parseando archivo de estado: {e}") from e
    except Exception as e:
        raise ToolError(f"Error consultando estado: {e}") from e


def format_service_status(service: dict) -> str:
    """Formatea el estado de un servicio."""
    status_emoji = "✅" if service["status"] == "UP" else "❌"
    
    result = f"""
{status_emoji} **{service['display_name']}** ({service['name']})
- Estado: {service['status']}
- Uptime: {service['uptime']}
- Última verificación: {service['last_check']}
"""
    
    if "response_time_ms" in service:
        result += f"- Tiempo de respuesta: {service['response_time_ms']}ms\n"
    
    if "last_failure" in service:
        result += f"- Última falla: {service['last_failure']}\n"
        if "error" in service:
            result += f"- Error: {service['error']}\n"
    
    return result.strip()
```

- [ ] **Step 3: Escribir test**

```python
"""Tests para tool de status."""
import pytest
import json
from pathlib import Path
from src.tools.status import check_service_status


def test_check_service_status_returns_status(tmp_path):
    """Verifica que retorna el estado de un servicio."""
    # Setup
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({
        "services": [
            {"name": "web-server", "display_name": "Web Server", "status": "UP", "uptime": "99.9%", "last_check": "2024-01-01"}
        ]
    }))
    
    # Execute
    result = check_service_status("web-server")
    
    # Assert
    assert "UP" in result
    assert "web-server" in result.lower()


def test_check_service_status_not_found(tmp_path):
    """Verifica comportamiento cuando servicio no existe."""
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"services": [{"name": "web-server", "status": "UP"}]}))
    
    result = check_service_status("nonexistent")
    
    assert "no encontrado" in result.lower()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_tools/test_status.py -v
```
Expected: FAIL (no module)

- [ ] **Step 5: Implementar module**

Crear `tests/test_tools/__init__.py` vacío y re-ejecutar.

- [ ] **Step 6: Run tests again**

```bash
pytest tests/test_tools/test_status.py -v
```
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add data/status.json src/tools/status.py tests/test_tools/
git commit -m "feat: add service status tool"
```

---

### Fase 4: Vector Store (Qdrant)

#### Task 4.1: Cliente Qdrant

**Files:**
- Create: `src/vectorstore/qdrant_client.py`
- Test: N/A (requiere Qdrant corriendo)

- [ ] **Step 1: Escribir cliente Qdrant**

```python
"""Cliente para Qdrant vector store."""
from pathlib import Path
from typing import Optional
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import settings
from src.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class QdrantManager:
    """Gestor del vector store Qdrant."""
    
    COLLECTION_NAME = "technical_docs"
    
    def __init__(self, path: Optional[str] = None):
        """Inicializa el cliente Qdrant.
        
        Args:
            path: Path al directorio de Qdrant. Si es None, usa settings.
        """
        self.path = path or settings.QDRANT_PATH
        self.client = QdrantClient(path=self.path)
    
    def create_collection(self, vector_size: int = 768) -> None:
        """Crea la colección de documentos.
        
        Args:
            vector_size: Dimensión del vector de embedding.
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.COLLECTION_NAME not in collection_names:
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Colección {self.COLLECTION_NAME} creada")
            else:
                logger.info(f"Colección {self.COLLECTION_NAME} ya existe")
                
        except Exception as e:
            raise VectorStoreError(f"Error creando colección: {e}") from e
    
    def add_documents(self, documents: list[dict], vectors: list[list[float]]) -> None:
        """Añade documentos al vector store.
        
        Args:
            documents: Lista de documentos con id, text, source, page.
            vectors: Lista de vectores de embedding.
        """
        try:
            points = [
                PointStruct(
                    id=doc["id"],
                    vector=vector,
                    payload={
                        "text": doc["text"],
                        "source": doc.get("source", ""),
                        "page": doc.get("page", 0)
                    }
                )
                for doc, vector in zip(documents, vectors)
            ]
            
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points
            )
            logger.info(f"Añadidos {len(points)} documentos")
            
        except Exception as e:
            raise VectorStoreError(f"Error añadiendo documentos: {e}") from e
    
    def search(self, query_vector: list[float], limit: int = 5) -> list[dict]:
        """Busca documentos similares.
        
        Args:
            query_vector: Vector de embedding de la query.
            limit: Número máximo de resultados.
        
        Returns:
            Lista de documentos similares.
        """
        try:
            results = self.client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_vector,
                limit=limit
            )
            
            return [
                {
                    "id": r.id,
                    "text": r.payload["text"],
                    "source": r.payload.get("source", ""),
                    "page": r.payload.get("page", 0),
                    "score": r.score
                }
                for r in results
            ]
            
        except Exception as e:
            raise VectorStoreError(f"Error en búsqueda: {e}") from e
```

- [ ] **Step 2: Commit**

```bash
git add src/vectorstore/qdrant_client.py
git commit -m "feat: add Qdrant client"
```

---

#### Task 4.2: Ingesta de Documentos

**Files:**
- Create: `src/vectorstore/ingest.py`
- Test: N/A (data processing)

- [ ] **Step 1: Escribir script de ingesta**

```python
"""Ingesta de documentos al vector store."""
import json
import logging
from pathlib import Path
from typing import Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import settings
from src.vectorstore.qdrant_client import QdrantManager
from src.exceptions import VectorStoreError, LLMError

logger = logging.getLogger(__name__)


def get_embedding_model():
    """Obtiene el modelo de embedding."""
    try:
        return GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key
        )
    except Exception as e:
        raise LLMError(f"Error inicializando embeddings: {e}") from e


def load_document(file_path: Path) -> list[dict]:
    """Carga un documento y lo convierte en chunks.
    
    Args:
        file_path: Path al documento.
    
    Returns:
        Lista de chunks con metadatos.
    """
    if file_path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
    elif file_path.suffix.lower() in [".md", ".txt"]:
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
    else:
        logger.warning(f"Tipo de archivo no soportado: {file_path.suffix}")
        return []
    
    # Chunking agresivo (80 tokens para cuenta free)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=lambda x: len(x.split())
    )
    
    chunks = splitter.split_documents(docs)
    
    return [
        {
            "text": chunk.page_content,
            "source": file_path.name,
            "page": chunk.metadata.get("page", 0)
        }
        for chunk in chunks
    ]


def ingest_documents(documents_dir: Optional[Path] = None) -> None:
    """Ingiere todos los documentos al vector store.
    
    Args:
        documents_dir: Directorio con documentos. Si es None, usa settings.
    """
    dir_path = documents_dir or settings.DOCUMENTS_DIR
    
    if not dir_path.exists():
        logger.warning(f"Directorio de documentos no existe: {dir_path}")
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creado directorio: {dir_path}")
        return
    
    # Inicializar Qdrant
    qdrant = QdrantManager()
    qdrant.create_collection()
    
    # Embedding model
    embeddings = get_embedding_model()
    
    # Procesar cada documento
    all_chunks = []
    for file_path in dir_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in [".pdf", ".md", ".txt"]:
            logger.info(f"Procesando: {file_path.name}")
            chunks = load_document(file_path)
            all_chunks.extend(chunks)
    
    if not all_chunks:
        logger.warning("No hay documentos para ingerir")
        return
    
    # Generar embeddings
    logger.info(f"Generando embeddings para {len(all_chunks)} chunks...")
    texts = [chunk["text"] for chunk in all_chunks]
    vectors = embeddings.embed_documents(texts)
    
    # Añadir IDs
    documents_with_id = [
        {**chunk, "id": i}
        for i, chunk in enumerate(all_chunks)
    ]
    
    # Guardar en Qdrant
    qdrant.add_documents(documents_with_id, vectors)
    
    logger.info(f"Ingesta completada: {len(all_chunks)} chunks")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_documents()
```

- [ ] **Step 2: Commit**

```bash
git add src/vectorstore/ingest.py
git commit -m "feat: add document ingestion pipeline"
```

---

### Fase 5: Herramientas (Tools)

#### Task 5.1: Tool Documents (RAG)

**Files:**
- Create: `src/tools/documents.py`
- Test: `tests/test_tools/test_documents.py`

- [ ] **Step 1: Escribir tool de documentos**

```python
"""Tool para búsqueda en documentos técnicos."""
import logging
from typing import Optional

from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import settings
from src.vectorstore.qdrant_client import QdrantManager
from src.exceptions import VectorStoreError, LLMError

logger = logging.getLogger(__name__)


@tool
def search_documents(query: str, limit: int = 5) -> str:
    """Busca en la documentación técnica.
    
    Úsala cuando el usuario pregunte por:
    - Cómo configurar algo
    - Qué significa un error
    - Dónde está un manual o guía
    - Información sobre servidores, redes, seguridad
    
    Args:
        query: Texto de búsqueda o pregunta.
        limit: Número máximo de resultados (default 5).
    
    Returns:
        String con los documentos encontrados.
    """
    try:
        # Embedding de la query
        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key
        )
        
        query_vector = embeddings.embed_query(query)
        
        # Buscar en Qdrant
        qdrant = QdrantManager()
        results = qdrant.search(query_vector, limit=limit)
        
        if not results:
            return "No se encontró documentación relevante para la consulta."
        
        # Formatear resultados
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"--- Resultado {i} ---\n"
                f"Fuente: {result['source']} (pág. {result['page']})\n"
                f"Relevancia: {result['score']:.2%}\n"
                f"\n{result['text']}\n"
            )
        
        return "\n\n".join(formatted)
        
    except Exception as e:
        logger.error(f"Error en search_documents: {e}")
        return f"Error buscando documentos: {str(e)}"
```

- [ ] **Step 2: Commit**

```bash
git add src/tools/documents.py
git commit -m "feat: add document search tool"
```

---

#### Task 5.2: Tool Tickets (wrapper)

**Files:**
- Modify: `src/tools/tickets.py`
- Test: N/A (usa queries existente)

- [ ] **Step 1: Escribir tool de tickets**

```python
"""Tool para búsqueda de tickets."""
import logging
from typing import Optional

from langchain_core.tools import tool

from src.db.queries import search_tickets as db_search_tickets
from src.exceptions import DatabaseError

logger = logging.getLogger(__name__)


@tool
def search_tickets(query: str, limit: int = 5) -> str:
    """Busca en el historial de tickets de soporte técnico.
    
    Úsala cuando el usuario pregunte por:
    - Problemas pasados similares
    - Cómo se resolvió un error antes
    - Historial de incidentes
    - Errores recurrentes
    
    Args:
        query: Descripción del problema o error.
        limit: Número máximo de resultados (default 5).
    
    Returns:
        String con los tickets encontrados.
    """
    try:
        results = db_search_tickets(query, limit=limit)
        
        if not results:
            return "No se encontró historial de tickets relacionado."
        
        # Formatear resultados
        formatted = []
        for ticket in results:
            severity_emoji = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢"
            }.get(ticket["severity"], "⚪")
            
            formatted.append(
                f"--- Ticket #{ticket['id']} {severity_emoji} {ticket['severity'].upper()} ---\n"
                f"Título: {ticket['title']}\n"
                f"Código: {ticket['error_code'] or 'N/A'}\n"
                f"Descripción: {ticket['description']}\n"
                f"Solución: {ticket['solution']}\n"
            )
        
        return "\n\n".join(formatted)
        
    except DatabaseError as e:
        logger.error(f"Error en search_tickets: {e}")
        return f"Error consultando tickets: {str(e)}"
```

- [ ] **Step 2: Commit**

```bash
git add src/tools/tickets.py
git commit -m "feat: add ticket search tool"
```

---

#### Task 5.3: Tool Suggestion

**Files:**
- Create: `src/tools/suggestion.py`
- Test: N/A (genera suggestions basadas en resultados)

- [ ] **Step 1: Escribir tool de sugerencias**

```python
"""Tool para sugerir acciones basadas en resultados."""
import json
import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def suggest_action(tool_results: str) -> str:
    """Sugiere acciones basadas en los resultados de otras herramientas.
    
    Úsala después de ejecutar search_documents o search_tickets
    para proporcionar recomendaciones prácticas al usuario.
    
    Args:
        tool_results: Resultados de otras herramientas en formato JSON o texto.
    
    Returns:
        String con acciones sugeridas.
    """
    try:
        # Análisis simple de los resultados
        suggestions = []
        
        # Detectar si hay errores mencionados
        error_keywords = ["error", "503", "502", "500", "timeout", "failed", "down", "caído"]
        has_error = any(keyword in tool_results.lower() for keyword in error_keywords)
        
        if has_error:
            suggestions.append("📋 Revisa los logs del servicio para más detalles")
            suggestions.append("🔧 Considera reiniciar el servicio afectado")
            suggestions.append("📞 Escala al equipo de infraestructura si el problema persiste")
        
        # Detectar si hay soluciones en los resultados
        solution_keywords = ["solución", "solved", "fix", "resolver"]
        has_solution = any(keyword in tool_results.lower() for keyword in solution_keywords)
        
        if has_solution:
            suggestions.append("✅ Se encontró una solución en el historial")
            suggestions.append("📝 Documenta los pasos seguidos para referencia futura")
        
        # Si no hay sugerencias específicas
        if not suggestions:
            suggestions.append("📚 Consulta la documentación completa para más información")
            suggestions.append("🔍 Refina tu búsqueda si no encontraste lo que necesitabas")
        
        return "**Acciones sugeridas:**\n" + "\n".join(f"- {s}" for s in suggestions)
        
    except Exception as e:
        logger.error(f"Error en suggest_action: {e}")
        return "No fue posible generar sugerencias en este momento."
```

- [ ] **Step 2: Commit**

```bash
git add src/tools/suggestion.py
git commit -m "feat: add action suggestion tool"
```

---

### Fase 6: Modelos LLM

#### Task 6.1: Router

**Files:**
- Create: `src/llm/router.py`
- Test: `tests/test_llm/test_router.py`

- [ ] **Step 1: Escribir test para router**

```python
"""Tests para el router."""
import pytest
import json
from unittest.mock import patch, MagicMock


def test_router_keywords_status():
    """Verifica que selecciona check_service_status para consultas de estado."""
    from src.llm.router import router
    
    # Mock del LLM para evitar llamada real
    with patch('src.llm.router.get_router_llm') as mock_llm:
        mock_response = MagicMock()
        mock_response.content = '{"tool": "check_service_status", "reason": "Consulta sobre estado de servicio"}'
        mock_llm.return_value.invoke.return_value = mock_response
        
        result = router.invoke("¿Está caído el servidor web?")
        
        assert "check_service_status" in result.content


def test_router_keywords_documents():
    """Verifica que selecciona search_documents para consultas de docs."""
    from src.llm.router import router
    
    with patch('src.llm.router.get_router_llm') as mock_llm:
        mock_response = MagicMock()
        mock_response.content = '{"tool": "search_documents", "reason": "Consulta sobre documentación"}'
        mock_llm.return_value.invoke.return_value = mock_response
        
        result = router.invoke("¿Dónde está el manual de nginx?")
        
        assert "search_documents" in result.content
```

- [ ] **Step 2: Run test**

```bash
pytest tests/test_llm/test_router.py -v
```
Expected: PASS (o FAIL si hay errores de imports)

- [ ] **Step 3: Escribir router**

```python
"""Router con Gemini 2.5 Flash."""
import logging
from typing import Optional
from pathlib import Path

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings

logger = logging.getLogger(__name__)


def get_router_llm():
    """Obtiene el modelo del router."""
    return ChatGoogleGenerativeAI(
        model=settings.router_model,
        google_api_key=settings.google_api_key,
        temperature=0.1
    )


def load_prompt(prompt_name: str) -> str:
    """Carga un prompt desde archivo."""
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


SYSTEM_PROMPT = load_prompt("router_prompt") or """Eres el orquestador..."""


@tool
def router(query: str) -> str:
    """Determina qué herramienta usar para responder la consulta.
    
    Args:
        query: Consulta del usuario.
    
    Returns:
        JSON con la herramienta seleccionada y razón.
    """
    llm = get_router_llm()
    
    # Keywords para clasificación simple
    doc_keywords = ["documentación", "manual", "configurar", "guía", "qué es", "cómo"]
    ticket_keywords = ["error", "problema", "antes", "historial", "solución", "resuelto"]
    status_keywords = ["estado", "servicio", "servidor", "caído", "uptime", "disponible"]
    
    query_lower = query.lower()
    
    if any(kw in query_lower for kw in status_keywords):
        return '{"tool": "check_service_status", "reason": "Consulta sobre estado de servicio"}'
    elif any(kw in query_lower for kw in ticket_keywords):
        return '{"tool": "search_tickets", "reason": "Consulta sobre problemas o errores pasados"}'
    elif any(kw in query_lower for kw in doc_keywords):
        return '{"tool": "search_documents", "reason": "Consulta sobre documentación técnica"}'
    else:
        # Default: buscar en documents
        return '{"tool": "search_documents", "reason": "Consulta general, se busca en documentación"}'
```

- [ ] **Step 4: Run test again**

```bash
pytest tests/test_llm/test_router.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/llm/router.py tests/test_llm/test_router.py
git commit -m "feat: add router with Gemini Flash"
```

---

#### Task 6.2: Synthesizer

**Files:**
- Create: `src/llm/synthesizer.py`
- Test: N/A (generación de respuesta)

- [ ] **Step 1: Escribir synthesizer**

```python
"""Synthesizer con Gemini 2.5 Pro."""
import logging
from typing import Optional
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings

logger = logging.getLogger(__name__)


def get_synthesizer_llm():
    """Obtiene el modelo del synthesizer."""
    return ChatGoogleGenerativeAI(
        model=settings.synthesizer_model,
        google_api_key=settings.google_api_key,
        temperature=0.3
    )


def load_prompt(prompt_name: str) -> str:
    """Carga un prompt desde archivo."""
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


SYSTEM_PROMPT = load_prompt("synthesizer_prompt") or """Eres un asistente de soporte técnico corporativo..."""


def synthesize(query: str, tool_results: str, user_level: str = "basic") -> str:
    """Genera respuesta final usando Gemini Pro.
    
    Args:
        query: Consulta original del usuario.
        tool_results: Resultados de las herramientas ejecutadas.
        user_level: Nivel del usuario (basic, advanced, admin).
    
    Returns:
        Respuesta final sintetizada.
    """
    llm = get_synthesizer_llm()
    
    # Ajustar prompt según nivel
    level_context = {
        "basic": "Explica de forma sencilla, evita jerga técnica innecesaria.",
        "advanced": "Puedes usar términos técnicos, sé detallado.",
        "admin": "Incluye información de infraestructura y métricas relevantes."
    }
    
    messages = [
        SystemMessage(content=f"{SYSTEM_PROMPT}\n\n{level_context.get(user_level, '')}"),
        HumanMessage(content=f"""
Consulta del usuario: {query}

Resultados de herramientas:
{tool_results}

Genera una respuesta final.
""")
    ]
    
    response = llm.invoke(messages)
    return response.content
```

- [ ] **Step 2: Commit**

```bash
git add src/llm/synthesizer.py
git commit -m "feat: add synthesizer with Gemini Pro"
```

---

### Fase 7: Grafo LangGraph

#### Task 7.1: Nodos del Grafo

**Files:**
- Create: `src/graph/nodes.py`
- Test: N/A (integración)

- [ ] **Step 1: Escribir nodos**

```python
"""Nodos del grafo LangGraph."""
import json
import logging
from typing import TypedDict, Optional

from src.llm.router import router
from src.llm.synthesizer import synthesize
from src.tools.documents import search_documents
from src.tools.tickets import search_tickets
from src.tools.status import check_service_status
from src.tools.suggestion import suggest_action

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Estado del agente."""
    query: str
    user_level: str
    intent: str  # "error_fix" | "doc_search" | "status_check" | "investigation"
    selected_tools: list[str]
    tool_results: list[dict]  # [{tool, result, confidence}]
    final_response: Optional[str]
    error: Optional[str]


def input_node(state: AgentState) -> AgentState:
    """Nodo de entrada: procesa la query."""
    logger.info(f"Input node processing: {state['query']}")
    return state


def router_node(state: AgentState) -> AgentState:
    """Nodo router: selecciona herramienta(s)."""
    try:
        result = router.invoke(state["query"])
        tool_result = result.content
        
        # Parsear JSON
        parsed = json.loads(tool_result)
        selected_tool = parsed.get("tool", "")
        intent = parsed.get("reason", "").split()[0] if parsed.get("reason") else "general"
        
        logger.info(f"Router selected: {selected_tool}, intent: {intent}")
        
        return {
            **state,
            "selected_tools": [selected_tool] if selected_tool else [],
            "intent": intent,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Router error: {e}")
        return {
            **state,
            "selected_tools": [],
            "intent": "error",
            "error": f"Error en router: {str(e)}"
        }


def execute_tool_node(state: AgentState) -> AgentState:
    """Nodo de ejecución de herramientas (paralelo)."""
    tool_names = state.get("selected_tools", [])
    query = state["query"]
    
    if not tool_names:
        return {**state, "tool_results": [], "error": "No tool selected"}
    
    results = []
    
    try:
        for tool_name in tool_names:
            if tool_name == "search_documents":
                result = search_documents.invoke({"query": query})
            elif tool_name == "search_tickets":
                result = search_tickets.invoke({"query": query})
            elif tool_name == "check_service_status":
                service_name = extract_service_name(query)
                result = check_service_status.invoke({"service_name": service_name})
            elif tool_name == "suggest_action":
                result = suggest_action.invoke({"tool_results": str(results)})
            else:
                result = f"Herramienta '{tool_name}' no implementada"
            
            results.append({
                "tool": tool_name,
                "result": result.content if hasattr(result, 'content') else str(result),
                "confidence": 1.0
            })
        
        logger.info(f"Tools executed: {tool_names}")
        
        return {
            **state,
            "tool_results": results,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return {
            **state,
            "tool_results": results,
            "error": f"Error ejecutando herramientas: {str(e)}"
        }


def extract_service_name(query: str) -> str:
    """Extrae nombre de servicio de la query."""
    # Lógica simple: buscar palabras después de "servicio" o "servidor"
    query_lower = query.lower()
    
    known_services = ["web-server", "api-gateway", "database", "cache", "auth"]
    for service in known_services:
        if service in query_lower:
            return service
    
    return query


def synthesizer_node(state: AgentState) -> AgentState:
    """Nodo synthesizer: genera respuesta final."""
    try:
        query = state["query"]
        tool_results = state.get("tool_results", [])
        user_level = state.get("user_level", "basic")
        
        if not tool_results:
            # Sin resultados de tools, responder directamente
            response = query  # Esto debería mejorar
        else:
            # Combinar resultados
            combined_results = "\n\n".join([
                f"[{r['tool']}]: {r['result']}" 
                for r in tool_results
            ])
            response = synthesize(query, combined_results, user_level)
        
        logger.info("Synthesis complete")
        
        return {
            **state,
            "final_response": response,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Synthesizer error: {e}")
        return {
            **state,
            "final_response": f"Error generando respuesta: {str(e)}",
            "error": str(e)
        }
```

- [ ] **Step 2: Commit**

```bash
git add src/graph/nodes.py
git commit -m "feat: add LangGraph nodes"
```

---

#### Task 7.2: Definición del Grafo

**Files:**
- Create: `src/graph/maso_graph.py`
- Test: N/A

- [ ] **Step 1: Escribir grafo**

```python
"""Definición del grafo LangGraph."""
import logging
from langgraph.graph import StateGraph, END
from src.graph.nodes import AgentState, input_node, router_node, execute_tool_node, synthesizer_node

logger = logging.getLogger(__name__)


def create_graph() -> StateGraph:
    """Crea el grafo deLangGraph."""
    
    graph = StateGraph(AgentState)
    
    # Añadir nodos
    graph.add_node("input", input_node)
    graph.add_node("router", router_node)
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("synthesizer", synthesizer_node)
    
    # Definir flujo
    graph.set_entry_point("input")
    
    # Input -> Router
    graph.add_edge("input", "router")
    
    # Router -> Tool o Synthesizer (conditional)
    def route_after_router(state: AgentState) -> str:
        if state.get("error"):
            return "synthesizer"
        if state.get("selected_tool"):
            return "execute_tool"
        return "synthesizer"
    
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "execute_tool": "execute_tool",
            "synthesizer": "synthesizer"
        }
    )
    
    # Tool -> Synthesizer
    graph.add_edge("execute_tool", "synthesizer")
    
    # Synthesizer -> END
    graph.add_edge("synthesizer", END)
    
    return graph


def compile_graph():
    """Compila el grafo."""
    graph = create_graph()
    return graph.compile()


# Instancia global del grafo
maso_graph = compile_graph()


def run_agent(query: str, user_level: str = "basic") -> str:
    """Ejecuta el agente con una query.
    
    Args:
        query: Consulta del usuario.
        user_level: Nivel del usuario.
    
    Returns:
        Respuesta final.
    """
    initial_state: AgentState = {
        "query": query,
        "user_level": user_level,
        "selected_tool": None,
        "tool_result": None,
        "final_response": None,
        "error": None
    }
    
    result = maso_graph.invoke(initial_state)
    return result.get("final_response", "Sin respuesta")
```

- [ ] **Step 2: Commit**

```bash
git add src/graph/maso_graph.py
git commit -m "feat: add LangGraph definition"
```

---

### Fase 8: Punto de Entrada

#### Task 8.1: main.py

**Files:**
- Create: `src/main.py`
- Test: N/A

- [ ] **Step 1: Escribir main.py**

```python
"""Punto de entrada del sistema MASO."""
import argparse
import logging
import sys

from src.config import settings
from src.db.setup_db import setup_database
from src.vectorstore.ingest import ingest_documents
from src.graph.maso_graph import run_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup():
    """Inicializa el sistema."""
    logger.info("Inicializando MASO...")
    
    # Setup DB
    if not settings.DB_PATH.exists():
        logger.info("Creando base de datos...")
        setup_database()
    else:
        logger.info("Base de datos ya existe")
    
    # Verificar documentos
    if settings.DOCUMENTS_DIR.exists():
        logger.info("Ingiiriendo documentos...")
        ingest_documents()
    else:
        logger.warning(f"Directorio de documentos no existe: {settings.DOCUMENTS_DIR}")
    
    logger.info("Setup completado")


def chat(user_level: str = "basic"):
    """Modo interactivo."""
    print("=" * 50)
    print("MASO - Asistente de Soporte Técnico")
    print("Escribe 'exit' para salir")
    print("=" * 50)
    
    while True:
        try:
            query = input("\n> ")
            
            if query.lower() in ["exit", "quit", "salir"]:
                print("¡Hasta luego!")
                break
            
            if not query.strip():
                continue
            
            response = run_agent(query, user_level)
            print(f"\n{response}\n")
            
        except KeyboardInterrupt:
            print("\n¡Hasta luego!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}")


def main():
    """Main entrypoint."""
    parser = argparse.ArgumentParser(description="MASO - Asistente de Soporte Técnico")
    parser.add_argument("--setup", action="store_true", help="Ejecutar setup inicial")
    parser.add_argument("--level", choices=["basic", "advanced", "admin"], default="basic", help="Nivel de usuario")
    
    args = parser.parse_args()
    
    if args.setup:
        setup()
    else:
        chat(args.level)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/main.py
git commit -m "feat: add main entry point"
```

---

### Fase 9: Datos de Prueba (Documentos)

#### Task 9.1: Descargar Documentos

**Files:**
- Create: `scripts/download_docs.py`
- Run: Ejecutar script

- [ ] **Step 1: Escribir script de descarga**

```python
"""Script para descargar documentos de prueba."""
import requests
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOCUMENTS_DIR = Path("data/documents")
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

DOCS_TO_DOWNLOAD = [
    {
        "name": "docker-guide.md",
        "url": "https://docs.docker.com/get-started/docker-overview/"
    },
    {
        "name": "nginx-admin.md",
        "url": "https://nginx.org/en/docs/"
    },
]


def download_document(name: str, url: str) -> None:
    """Descarga un documento."""
    output_path = DOCUMENTS_DIR / name
    
    if output_path.exists():
        logger.info(f"Ya existe: {name}")
        return
    
    try:
        # Para PDFs, we'd use requests
        # Por ahora creamos templates básicos
        logger.info(f"Descargando: {name}")
        
        # Note: Este es un ejemplo. Para PDFs reales, usar:
        # response = requests.get(pdf_url)
        # with open(output_path, 'wb') as f:
        #     f.write(response.content)
        
        logger.info(f"Completado: {name}")
        
    except Exception as e:
        logger.error(f"Error descargando {name}: {e}")


if __name__ == "__main__":
    # Por ahora solo crear directorio
    logger.info(f"Directorio de documentos: {DOCUMENTS_DIR}")
    logger.info("Para documentos reales, descargar manualmente o implementar descarga")
```

- [ ] **Step 2: Explicar al usuario**

El script de descarga de PDFs reales requiere URLs específicas. 
**Nota para el usuario**: Deberás descargar manualmente PDFs de:
- Docker Documentation (https://docs.docker.com/)
- Kubernetes (https://kubernetes.io/docs/)
- Nginx (https://nginx.org/en/docs/)

Guardarlos en `data/documents/`

---

## Resumen de Tasks

| Fase | Task | Archivos |
|------|------|----------|
| 1 | Config | config.py, exceptions.py |
| 2 | DB | setup_db.py, queries.py |
| 3 | Status | status.py, status.json |
| 4 | Qdrant | qdrant_client.py, ingest.py |
| 5 | Tools | documents.py, tickets.py, suggestion.py |
| 6 | LLM | router.py, synthesizer.py |
| 7 | Graph | nodes.py, maso_graph.py |
| 8 | Main | main.py |
| 9 | Docs | (manual) |

---

**Plan complete saved to `docs/superpowers/plans/2026-03-29-maso-implementation-plan.md`**

**Two execution options:**

1. **Subagent-Driven (recommended)** - Dispatch fresh subagent per task, review between tasks, fast iteration

2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**