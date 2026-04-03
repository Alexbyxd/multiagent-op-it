"""Nodos del grafo LangGraph."""
import json
import logging
import time
from typing import TypedDict, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from src.config import settings
from src.llm.router import router
from src.llm.synthesizer import synthesize
from src.tools.documents import search_documents
from src.tools.tickets import search_tickets
from src.tools.status import check_service_status
from src.tools.suggestion import suggest_action

logger = logging.getLogger(__name__)


# Mapeo de herramientas a funciones
TOOL_MAP = {
    "search_documents": search_documents,
    "search_tickets": search_tickets,
    "check_service_status": check_service_status,
    "suggest_action": suggest_action
}


class AgentState(TypedDict):
    """Estado del agente."""
    query: str
    user_level: str
    intent: str
    selected_tools: list[str]
    tool_results: list[dict]
    final_response: Optional[str]
    error: Optional[str]


def input_node(state: AgentState) -> AgentState:
    """Nodo de entrada: procesa la query."""
    logger.info(f"Input node processing: {state['query']}")
    return state


def router_node(state: AgentState) -> AgentState:
    """Nodo router: selecciona herramienta(s) usando tool calling real."""
    try:
        # El router ahora devuelve un dict con tools, args, reason
        result = router(state["query"])
        
        # Result puede ser dict (nuevo router) o string (fallback)
        if isinstance(result, dict):
            selected_tools = result.get("tools", [])
            intent = result.get("reason", "general")
            direct = result.get("direct", False)
        else:
            # Fallback para formato antiguo
            try:
                parsed = json.loads(result)
                selected_tools = parsed.get("tools", parsed.get("tool", ""))
                if isinstance(selected_tools, str):
                    selected_tools = [selected_tools] if selected_tools else []
                intent = parsed.get("reason", "general")
            except (json.JSONDecodeError, AttributeError):
                selected_tools = []
                intent = "error"
        
        logger.info(f"Router selected: {selected_tools}, intent: {intent}")
        
        return {
            **state,
            "selected_tools": selected_tools,
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
    """Nodo de ejecución de herramientas (paralelo para múltiples tools)."""
    tool_names = state.get("selected_tools", [])
    query = state["query"]
    
    if not tool_names:
        return {**state, "tool_results": [], "error": "No tool selected"}
    
    results = []
    
    # Filtrar tools que existen
    valid_tools = [t for t in tool_names if t in TOOL_MAP]
    
    if not valid_tools:
        return {**state, "tool_results": [], "error": f"Ninguna herramienta válida: {tool_names}"}
    
    def execute_single_tool(tool_name: str) -> dict:
        """Ejecuta una sola herramienta."""
        try:
            tool_func = TOOL_MAP[tool_name]
            
            # Preparar argumentos según la tool
            if tool_name == "check_service_status":
                service_name = extract_service_name(query)
                result = tool_func.invoke({"service_name": service_name})
            elif tool_name == "suggest_action":
                # suggest_action debe ejecutarse AL FINAL con TODOS los resultados previos
                # No ejecutar en paralelo - se ejecuta después de otros tools
                return {
                    "tool": tool_name,
                    "result": "pending",  # Se ejecuta después
                    "confidence": 0.0,
                    "success": False,
                    "pending": True
                }
            else:
                # search_documents, search_tickets
                result = tool_func.invoke({"query": query})
            
            return {
                "tool": tool_name,
                "result": result if isinstance(result, str) else (result.content if hasattr(result, 'content') else str(result)),
                "confidence": 1.0,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error ejecutando {tool_name}: {e}")
            return {
                "tool": tool_name,
                "result": f"Error: {str(e)}",
                "confidence": 0.0,
                "success": False,
                "error": str(e)
            }
    
    try:
        # Separar suggest_action de la ejecución paralela
        # suggest_action debe ejecutarse al final con todos los resultados
        main_tools = [t for t in valid_tools if t != "suggest_action"]
        
        if len(main_tools) > 1:
            logger.info(f"Ejecutando {len(main_tools)} herramientas en paralelo")
            executor = ThreadPoolExecutor(max_workers=min(len(main_tools), 4))
            try:
                future_to_tool = {executor.submit(execute_single_tool, t): t for t in main_tools}
                for future in list(future_to_tool.keys()):
                    tool_name = future_to_tool[future]
                    try:
                        results.append(future.result(timeout=settings.tool_execution_timeout))
                    except FuturesTimeout:
                        logger.warning(
                            "Tool '%s' timed out after %ds — skipping",
                            tool_name,
                            settings.tool_execution_timeout,
                        )
                        results.append({
                            "tool": tool_name,
                            "result": f"Error: tool execution timed out after {settings.tool_execution_timeout}s",
                            "confidence": 0.0,
                            "success": False,
                            "error": "timeout",
                        })
                        # Cancel the slow future so shutdown doesn't block
                        future.cancel()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        elif main_tools:
            # Ejecución secuencial para una sola tool con timeout
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(execute_single_tool, main_tools[0])
                try:
                    results.append(future.result(timeout=settings.tool_execution_timeout))
                except FuturesTimeout:
                    tool_name = main_tools[0]
                    logger.warning(
                        "Tool '%s' timed out after %ds — skipping",
                        tool_name,
                        settings.tool_execution_timeout,
                    )
                    results.append({
                        "tool": tool_name,
                        "result": f"Error: tool execution timed out after {settings.tool_execution_timeout}s",
                        "confidence": 0.0,
                        "success": False,
                        "error": "timeout",
                    })
                    future.cancel()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        
        # Si suggest_action está en la lista, ejecutarla al final con todos los resultados
        if "suggest_action" in valid_tools:
            logger.info("Ejecutando suggest_action con resultados acumulados")
            combined_results = "\n".join([
                f"[{r['tool']}]: {r['result']}" 
                for r in results if r.get('success', False)
            ])
            suggest_func = TOOL_MAP["suggest_action"]
            try:
                suggest_result = suggest_func.invoke({"tool_results": combined_results})
                results.append({
                    "tool": "suggest_action",
                    "result": suggest_result if isinstance(suggest_result, str) else str(suggest_result),
                    "confidence": 1.0,
                    "success": True
                })
            except Exception as e:
                logger.error(f"Error en suggest_action: {e}")
                results.append({
                    "tool": "suggest_action",
                    "result": f"Error: {str(e)}",
                    "confidence": 0.0,
                    "success": False,
                    "error": str(e)
                })
        
        logger.info(f"Tools executed: {valid_tools}")
        
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
            response = query
        else:
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
