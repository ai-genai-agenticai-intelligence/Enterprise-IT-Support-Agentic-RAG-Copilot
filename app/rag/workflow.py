import logging
import time
from typing import Literal
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import get_retriever

logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None
_web_search = None


def extract_text(response) -> str:
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
                elif isinstance(item, str):
                    texts.append(item)
            if texts:
                return "\n".join(texts)
            return str(content)
        return str(content)
    return str(response)


def llm():
    global _llm
    if _llm is None:
        if settings.openai_api_key:
            from langchain_openai import ChatOpenAI
            _llm = ChatOpenAI(
                model=settings.openai_model,
                temperature=0,
                api_key=settings.openai_api_key,
                max_retries=3,
            )
        elif settings.gemini_api_key:
            from langchain_google_genai import ChatGoogleGenerativeAI
            _llm = ChatGoogleGenerativeAI(
                model=settings.gemini_model or "gemini-1.5-flash",
                temperature=0,
                google_api_key=settings.gemini_api_key,
                max_retries=4,
            )
        else:
            raise RuntimeError("Neither OPENAI_API_KEY nor GEMINI_API_KEY is configured in .env")
    return _llm


def web_search_tool():
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing in .env")
        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search


def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]


def safe_invoke_llm(prompt: str, structured_schema=None):
    if settings.gemini_api_key and not settings.openai_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        models_to_try = [
            settings.gemini_model,
            "gemini-2.5-flash-lite",
            "gemini-flash-lite-latest",
            "gemini-2.5-flash",
            "gemini-3.8-flash"
        ]
        # deduplicate while keeping order
        seen = set()
        unique_models = []
        for m in models_to_try:
            if m and m not in seen:
                seen.add(m)
                unique_models.append(m)
        
        last_error = None
        for model_name in unique_models:
            try:
                curr_client = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0,
                    google_api_key=settings.gemini_api_key,
                    max_retries=0,
                )
                if structured_schema:
                    return curr_client.with_structured_output(structured_schema).invoke(prompt)
                return curr_client.invoke(prompt)
            except Exception as e:
                last_error = e
                continue
        raise last_error
    else:
        client = llm()
        if structured_schema:
            return client.with_structured_output(structured_schema).invoke(prompt)
        return client.invoke(prompt)


def route_question(state: AgentState):
    decision = safe_invoke_llm(f"""
You route messages for an enterprise IT support assistant.
Use kb for questions about company IT policies, VPN, password reset, MFA, laptop setup,
software access, security, email, devices, troubleshooting, or technology support.
Use direct only for greetings, thanks, or casual chat that needs no company knowledge.
Question: {state['question']}
""", structured_schema=RouteDecision)
    route = getattr(decision, "route", "kb") if decision else "kb"
    return {"source_used": route, "trace": add_trace(state, f"Router -> {str(route).upper()}")}


def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"


def retrieve_kb(state: AgentState):
    docs = get_retriever().invoke(state["current_query"])
    return {"kb_docs": docs, "trace": add_trace(state, f"Private KB retrieval -> {len(docs)} chunks")}


def grade_kb(state: AgentState):
    context = "\n\n".join(f"Source: {d.metadata.get('source','unknown')}\n{d.page_content}" for d in state["kb_docs"])
    grade_obj = safe_invoke_llm(f"""
You grade evidence for an enterprise IT support assistant.
Question: {state['question']}
Private company KB evidence:
{context}
Return good only if the evidence is sufficient to answer confidently and specifically.
Otherwise return weak.
""", structured_schema=EvidenceGrade)
    grade = getattr(grade_obj, "grade", "weak") if grade_obj else "weak"
    return {"kb_grade": grade, "trace": add_trace(state, f"KB evidence grade -> {str(grade).upper()}")}


def after_kb(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    return "generate_from_kb" if state["kb_grade"] == "good" else "search_web"


def search_web(state: AgentState):
    result = web_search_tool().invoke({"query": state["current_query"]})
    lines, citations = [], []
    if isinstance(result, dict):
        if result.get("answer"):
            lines.append("Search answer: " + result["answer"])
        for item in result.get("results", []):
            title, url, content = item.get("title", ""), item.get("url", ""), item.get("content", "")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            citations.append({"title": title or url, "url": url, "type": "web"})
    else:
        lines.append(str(result))
    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(state, "Web fallback -> Tavily search"),
    }


def grade_web(state: AgentState):
    grade_obj = safe_invoke_llm(f"""
Question: {state['question']}
Web evidence:
{state['web_results']}
Return good if the evidence is sufficient and directly relevant; otherwise weak.
""", structured_schema=EvidenceGrade)
    grade = getattr(grade_obj, "grade", "weak") if grade_obj else "weak"
    return {"web_grade": grade, "trace": add_trace(state, f"Web evidence grade -> {str(grade).upper()}")}


def after_web(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"
    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"
    return "insufficient"


def rewrite_query(state: AgentState):
    response = safe_invoke_llm(f"""
Rewrite this IT support question for better private knowledge retrieval and vendor web search.
Preserve intent, add useful technical keywords, do not answer, return only the query.
Question: {state['question']}
""")
    rewritten = extract_text(response).strip()
    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(state, f"Query rewrite -> {rewritten}"),
    }


def generate_from_kb(state: AgentState):
    context = "\n\n".join(f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}" for d in state["kb_docs"])
    response = safe_invoke_llm(f"""
You are an enterprise IT support copilot. Answer ONLY from the private company KB below.
Be concise, actionable, and safe. If steps are present, present them clearly.
Do not invent policy details. Mention that the answer is based on the company's private knowledge base.
Question: {state['question']}

Private KB:
{context}
""")
    answer = extract_text(response)
    citations = []
    seen = set()
    for d in state["kb_docs"]:
        src = d.metadata.get("source", "Private KB")
        if src not in seen:
            seen.add(src)
            citations.append({"title": src.split("/")[-1], "url": "", "type": "private_kb"})
    return {"answer": answer, "source_used": "private_kb", "citations": citations, "trace": add_trace(state, "Answer generation -> PRIVATE KB")}


def generate_from_web(state: AgentState):
    response = safe_invoke_llm(f"""
You are an enterprise IT support copilot. The private company KB was insufficient.
Answer ONLY from the web evidence below. Clearly say this is external web information and may need IT validation before changing company-managed systems.
Question: {state['question']}

Web evidence:
{state['web_results']}
""")
    answer = extract_text(response)
    return {"answer": answer, "source_used": "web_search", "trace": add_trace(state, "Answer generation -> WEB SEARCH")}


def direct_answer(state: AgentState):
    response = safe_invoke_llm(f"Respond briefly and naturally to: {state['question']}")
    answer = extract_text(response)
    return {"answer": answer, "source_used": "direct", "trace": add_trace(state, "Direct response -> no retrieval")}


def insufficient(state: AgentState):
    return {
        "answer": "I couldn't find enough reliable evidence in the company knowledge base or external search to answer confidently. Please contact the IT help desk or provide more details.",
        "source_used": "insufficient_evidence",
        "trace": add_trace(state, "Stopped -> insufficient reliable evidence"),
    }


def build_graph():
    graph = StateGraph(AgentState)
    for name, fn in {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges("route_question", route_after_router, {
        "retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"
    })
    graph.add_edge("retrieve_kb", "grade_kb")
    graph.add_conditional_edges("grade_kb", after_kb, {
        "generate_from_kb": "generate_from_kb", "search_web": "search_web"
    })
    graph.add_edge("search_web", "grade_web")
    graph.add_conditional_edges("grade_web", after_web, {
        "generate_from_web": "generate_from_web", "rewrite_query": "rewrite_query", "insufficient": "insufficient"
    })
    graph.add_edge("rewrite_query", "retrieve_kb")
    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)
    return graph.compile()


agent_graph = build_graph()


def ask(question: str):
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    return agent_graph.invoke(initial)