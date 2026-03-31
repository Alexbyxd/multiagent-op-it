"""Definición del grafo LangGraph."""
import logging
from langgraph.graph import StateGraph, END
from src.graph.nodes import AgentState, input_node, router_node, execute_tool_node, synthesizer_node

logger = logging.getLogger(__name__)


def create_graph() -> StateGraph:
    """Crea el grafo deLangGraph."""
    
    graph = StateGraph(AgentState)
    
    graph.add_node("input", input_node)
    graph.add_node("router", router_node)
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("synthesizer", synthesizer_node)
    
    graph.set_entry_point("input")
    
    graph.add_edge("input", "router")
    
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
    
    graph.add_edge("execute_tool", "synthesizer")
    
    graph.add_edge("synthesizer", END)
    
    return graph


def compile_graph():
    """Compila el grafo."""
    graph = create_graph()
    return graph.compile()


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