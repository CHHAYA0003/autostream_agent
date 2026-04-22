import os
import time
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from rag_pipeline import get_retriever

load_dotenv()

# Define the state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    name: str
    email: str
    platform: str
    lead_captured: bool

# Define Intent schema for structured output
class IntentClassification(BaseModel):
    intent: str = Field(description="The intent of the user. Must be one of: 'casual_greeting', 'product_inquiry', 'high_intent_lead (ready to sign up)'")

# Initialize LLM - gemini-2.5-flash-lite
llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0, max_retries=6)

# Initialize Retriever
retriever = get_retriever()

_last_call_time = 0

def wait_for_quota():
    """Ensures we don't exceed 5 requests per minute (1 every 12 seconds)."""
    global _last_call_time
    now = time.time()
    elapsed = now - _last_call_time
    if elapsed < 13: # 13s buffer
        time.sleep(13 - elapsed)
    _last_call_time = time.time()

def mock_lead_capture(name, email, platform):
    print(f"Lead captured successfully: {name}, {email}, {platform}")
    return True

# --- Nodes ---

def classify_intent(state: AgentState):
    """Classifies the intent based on the last user message."""
    messages = state["messages"]
    last_message = messages[-1].content
    
    
    is_collecting = (state.get("name") or state.get("email") or state.get("platform"))
    last_intent = state.get("intent")
    
    if (is_collecting or last_intent == 'high_intent_lead') and not state.get("lead_captured"):
        return {"intent": 'high_intent_lead'}

    structured_llm = llm.with_structured_output(IntentClassification)
    prompt = f"""You are an intent classifier for AutoStream, a SaaS product that provides automated video editing tools.
Based on the user's message, classify their intent into exactly one of these categories:
1. 'casual_greeting': The user is just saying hello, asking how you are, etc.
2. 'product_inquiry': The user is asking about pricing, features, company policies, or how the product works.
3. 'high_intent_lead': The user expresses a clear desire to sign up, buy, subscribe, or try a specific plan (e.g. 'I want to sign up' or 'Ready to buy').

User message: {last_message}
"""
    wait_for_quota()
    result = structured_llm.invoke(prompt)
    return {"intent": result.intent}

def handle_casual(state: AgentState):
    messages = state["messages"]
    system_msg = SystemMessage(content="You are a helpful assistant for AutoStream. Respond casually and politely to the user's greeting.")
    wait_for_quota()
    response = llm.invoke([system_msg] + messages[-5:])
    return {"messages": [response]}

def handle_rag(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1].content
    
    docs = retriever.invoke(last_message)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    system_msg = SystemMessage(content=f"""You are a helpful sales assistant for AutoStream.
Use the following knowledge base context to answer the user's question accurately.
If the answer is not in the context, say you don't know and offer to connect them with support.

Context:
{context}
""")
    wait_for_quota()
    response = llm.invoke([system_msg] + messages[-5:])
    return {"messages": [response]}

# LLM for extracting lead details
class LeadExtraction(BaseModel):
    name: str = Field(description="The user's name if mentioned.", default="")
    email: str = Field(description="The user's email if mentioned.", default="")
    platform: str = Field(description="The user's Creator Platform (YouTube, Instagram, etc.) if mentioned.", default="")

def handle_lead(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1].content
    
    name = state.get("name", "")
    email = state.get("email", "")
    platform = state.get("platform", "")
    
    if state.get("lead_captured"):
        return {"messages": [AIMessage(content="We already have your details!")]}

    # Extract info
    extractor = llm.with_structured_output(LeadExtraction)
    wait_for_quota()
    extracted = extractor.invoke(f"Extract lead details from: '{last_message}'")
    
    # Sanitize and assign
    def sanitize(v):
        if not v or str(v).lower().strip() in ["null", "none", "unknown", ""]:
            return ""
        return str(v).strip()

    ext_name = sanitize(extracted.name)
    ext_email = sanitize(extracted.email)
    ext_plat = sanitize(extracted.platform)

    if ext_name and not name: name = ext_name
    if ext_email and not email: email = ext_email
    if ext_plat and not platform: platform = ext_plat

    missing = []
    if not name: missing.append("Name")
    if not email: missing.append("Email")
    if not platform: missing.append("Creator Platform")
    
    if missing:
        # Fetch RAG context specifically for the plan the user showed interest in
        wait_for_quota()
        # Find which plan was mentioned in history
        plan_query = "AutoStream features and policies"
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                content = m.content.lower()
                if "pro" in content or "premium" in content:
                    plan_query = "AutoStream Pro Premium plan features price and policies"
                    break
                if "basic" in content or "standard" in content:
                    plan_query = "AutoStream Basic standard plan features price and policies"
                    break
        
        docs = retriever.invoke(plan_query)
        context = "\n".join([d.page_content for d in docs[:3]])
        
        system_msg = f"""You are a helpful sales assistant. The user wants to sign up for a plan. 
Confirm the plan features (Basic: $29, 720p | Pro: $79, 4K, 24/7 support) and policies (No refunds after 7 days) briefly.
Then ask for the MISSING information: {missing[0]}.
If asking for platform, say: 'Which Creator Platform do you use (e.g., YouTube, Instagram, etc.)?'
Context: {context}"""
        
        wait_for_quota()
        response = llm.invoke([SystemMessage(content=system_msg)] + messages[-3:])
        
        return {
            "messages": [response],
            "name": name, "email": email, "platform": platform
        }
    else:
        mock_lead_capture(name, email, platform)
        success_msg = f"Thanks {name}! We've captured your details ({email}, {platform}). Your account is being set up. Remember, as a Pro user you get 24/7 support and 4K resolution! Our team will contact you soon."
        return {
            "messages": [AIMessage(content=success_msg)],
            "name": name, "email": email, "platform": platform, "lead_captured": True
        }

# --- Routing ---
def route_intent(state: AgentState):
    intent = state["intent"]
    if intent == "casual_greeting": return "casual_chat"
    if intent == "product_inquiry": return "rag_qa"
    if intent == "high_intent_lead": return "lead_capture"
    return "rag_qa"

# --- Build Graph ---
builder = StateGraph(AgentState)
builder.add_node("classifier", classify_intent)
builder.add_node("casual_chat", handle_casual)
builder.add_node("rag_qa", handle_rag)
builder.add_node("lead_capture", handle_lead)

builder.set_entry_point("classifier")
builder.add_conditional_edges("classifier", route_intent, 
    {"casual_chat": "casual_chat", "rag_qa": "rag_qa", "lead_capture": "lead_capture"})
builder.add_edge("casual_chat", END)
builder.add_edge("rag_qa", END)
builder.add_edge("lead_capture", END)

graph = builder.compile()

if __name__ == "__main__":
    pass
