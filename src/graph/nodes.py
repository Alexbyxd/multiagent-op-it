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
    """Nodo router: selecciona herramienta(s)."""
    try:
        result = router.invoke(state["query"])
        tool_result = result.content
        
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
