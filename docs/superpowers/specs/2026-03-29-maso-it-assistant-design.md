# Spec: Sistema Multi-Agente de Soporte Técnico (MASO)

## 1. Visión General

**Proyecto**: Asistente Corporativo de Operaciones IT  
**Tipo**: Sistema Multi-Agente RAG con LangGraph  
**Objetivo**: Ayudar al equipo de soporte técnico con respuestas precisas usando documentación, historial de tickets y estado de servicios

---

## 2. Caso de Uso

- **Usuario final**: Técnicos nivel 1, nivel 2/3, Administradores de IT
- **Consultas frecuentes**:
  - Cómo resolver errores específicos
  - Dónde encontrar documentación técnica
  - Estado actual de servicios/servidores
  - Investigación de problemas complejos

---

## 3. Arquitectura

### Flujo Principal
```
Usuario → Router (gemini-2.5-flash) → [Tools] → Síntesis (gemini-2.5-pro) → Usuario
```

### Nodos LangGraph
| Nodo | Función |
|------|---------|
| input_node | Recibe y normaliza consulta |
| router_node | Clasifica intención y selecciona herramientas |
| tool_nodes | Ejecuta herramientas en paralelo |
| synthesizer_node | Compila resultados y genera respuesta |

### Estado (AgentState)
```python
class AgentState(TypedDict):
    query: str
    user_level: str  # "basic" | "advanced" | "admin"
    intent: str  # "error_fix" | "doc_search" | "status_check" | "investigation"
    selected_tools: list[str]
    tool_results: list[dict]  # [{tool, result, confidence}]
    final_response: str
    error: Optional[str]
```

---

## 4. Herramientas

### 4.1 search_documents
- **Propósito**: Búsqueda en documentación técnica
- **Entrada**: Query de usuario
- **Proceso**: RAG con Qdrant sobre PDFs/Markdown
- **Salida**: Top 3-5 documentos relevantes con citas
- **Chunking**: 80 tokens por chunk (optimizado para cuenta free)

### 4.2 search_tickets
- **Propósito**: Historial de problemas resueltos
- **Entrada**: Descripción del error/problema
- **Proceso**: Búsqueda en SQLite por similitud o keywords
- **Salida**: Tickets similares con solución aplicada

### 4.3 check_service_status
- **Propósito**: Estado en tiempo real de servicios
- **Entrada**: Nombre del servicio/servidor
- **Proceso**: Consulta JSON de estado
- **Salida**: Estado (UP/DOWN), uptime, última verificación

### 4.4 suggest_action
- **Propósito**: Basado en resultados, sugerir siguiente paso
- **Entrada**: Resultados de otras tools
- **Salida**: Lista de acciones sugeridas

---

## 5. Datos de Prueba

### 5.1 Documentación Técnica
- Docker Documentation (PDF)
- Kubernetes Basics (PDF)
- Nginx Admin Guide (PDF)
- Linux System Administration (PDF)

### 5.2 Tickets Históricos (SQLite)
**Esquema**:
```sql
CREATE TABLE tickets (
    id INTEGER PRIMARY KEY,
    title TEXT,
    description TEXT,
    error_code TEXT,
    solution TEXT,
    severity TEXT,
    created_at DATETIME,
    resolved_at DATETIME
);
```

**Escenarios de prueba**:
- Error 503 en servidor web
- Problemas de conexión a base de datos
- Fallos de autenticación
- Caída de servicios

### 5.3 Estado de Servicios (JSON)
```json
{
  "services": [
    {"name": "web-server-prod", "status": "UP", "uptime": "99.9%"},
    {"name": "api-gateway", "status": "DOWN", "last_failure": "..."},
    {"name": "database-primary", "status": "UP", "uptime": "99.99%"}
  ]
}
```

---

## 6. Configuración

### Variables de Entorno (.env)
```env
GOOGLE_API_KEY=tu_api_key_aqui
DB_PATH=data/tickets.db
QDRANT_PATH=data/qdrant
EMBEDDING_MODEL=gemini-embedding-001
CHUNK_SIZE=80
CHUNK_OVERLAP=10
STATUS_API_URL=http://localhost:8080/status
```

### Modelos
| Componente | Modelo |
|------------|--------|
| Router | gemini-2.5-flash |
| Síntesis | gemini-2.5-pro |
| Embeddings | gemini-embedding-001 |

---

## 7. Presentación de Resultados

- **Texto**: Respuesta clara y concisa
- **Fuentes**: Indicación de dónde vino cada información
- **Acciones**: Sugerencias de pasos a seguir
- **Profundizar**: Opción de explorar cada fuente en detalle

---

## 8. Autonomía

- **Nivel B**: Responde + sugiere acciones pero usuario las ejecuta
- El sistema no ejecuta acciones automáticamente
- Recomendaciones basadas en contexto recuperado

---

## 9. Limitaciones Consideradas

- **Cuenta free de Gemini**: Chunks pequeños (80 tokens) para evitar límite de tokens
- **Datos locales**: No guardar en la nube
- **Rate limiting**: Respetar límites de API

---

## 10. Métricas de Éxito

- [ ] Router selecciona herramienta correcta ≥ 90%
- [ ] RAG retorna documentos relevantes ≥ 85%
- [ ] Síntesis genera respuestas coherentes
- [ ] Tiempo de respuesta < 10 segundos
