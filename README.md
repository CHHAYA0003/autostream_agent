# AutoStream AI Agent

![AutoStream Agent](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![LangChain](https://img.shields.io/badge/LangChain-Integration-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Workflow-orange)
![Gradio](https://img.shields.io/badge/Gradio-UI-yellow)

AutoStream AI Agent is a RAG-powered, conversational assistant built for the **AutoStream SaaS Platform**. It is designed to classify user intents, provide accurate answers from a knowledge base using Retrieval-Augmented Generation (RAG), and seamlessly capture high-intent leads.

This project was built using **LangChain**, **LangGraph**, **Google Gemini**, and **Gradio** for an interactive user interface.

## Key Features

- **Intent Classification**: Uses an LLM with structured output to classify user queries into three categories: `casual_greeting`, `product_inquiry`, or `high_intent_lead`.
- **Retrieval-Augmented Generation (RAG)**: Answers product-related inquiries based on the provided `knowledge_base.md`. Uses a **Persistent Chroma VectorStore** for fast retrieval without re-embedding.
- **Stateful Lead Capture Flow**: Employs LangGraph to manage conversational state, automatically prompting users for missing details (Name, Email, Platform) when they show high intent.
- **Rate-Limit Resilience**: Specifically optimized for the Google Gemini Free Tier with built-in **exponential backoff** and strategic delays to prevent `RESOURCE_EXHAUSTED` errors.
- **Interactive UI**: A sleek, professional dark-mode Gradio web interface with quick-prompt buttons.

## Architecture Explanation

**Why LangGraph?**
LangGraph was chosen over standard LangChain chains or AutoGen because it provides deterministic, stateful routing for conversational agents. In a lead-capture scenario, it's critical to control the flow and prevent the LLM from hallucinating premature tool calls or losing track of the required fields. LangGraph’s directed cyclic graph architecture allows us to define clear nodes (Casual, RAG, Lead Capture) and strictly govern transitions via edges, ensuring the `mock_lead_capture` tool is *only* executed when the agent state contains all required fields.

**State Management**
State is managed using a defined `TypedDict` (`AgentState`) containing the conversation history (a list of LangChain `BaseMessage` objects), the classified intent, and the lead fields (`name`, `email`, `platform`, `lead_captured`). As the graph executes, each node receives the current state, performs its operation (e.g., intent classification or entity extraction), and returns a dictionary of updates. LangGraph automatically merges these updates into the global state, ensuring memory persists cleanly across 5–6 conversation turns and preventing redundant prompts.

## WhatsApp Deployment Integration

**Integrating with WhatsApp using Webhooks:**
To deploy this agent to WhatsApp, we would use the Meta/WhatsApp Cloud API. 
1. **Webhook Setup:** We would configure a webhook endpoint (e.g., using FastAPI or Flask) to receive incoming HTTP POST requests from WhatsApp whenever a user sends a message.
2. **Message Processing:** The endpoint would extract the user's phone number (as a unique session ID) and the message text from the payload.
3. **State Persistence:** Since the HTTP webhook is stateless, we would use a database (like Redis or PostgreSQL) to retrieve the serialized LangGraph state associated with that phone number.
4. **Agent Invocation:** We pass the user's message and the retrieved state into the LangGraph workflow. The graph updates the state and returns a response.
5. **Sending the Reply:** We save the updated state back to the database and trigger a POST request to the WhatsApp API to send the generated response back to the user's phone.

## Prerequisites & Setup

1. **Clone or Open the Repository**
2. **Set up the Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Environment Variables**
   Create a `.env` file in the root directory and add your Google Gemini API Key. Since LangChain looks for `GOOGLE_API_KEY` by default, set it up like this:
   ```env
   GOOGLE_API_KEY="AIzaSyAXUptMErzADMWpA42fViYytfveH6GpE_8"
   ```

## Running the Application

To launch the interactive web interface, run:
```bash
python app.py
```
This will start a local Gradio server, usually accessible at `http://127.0.0.1:7860`. The first run will automatically build and persist the vector store in `.chroma_db/`.

## Project Structure

- `app.py`: The Gradio frontend application.
- `agent.py`: Contains the LangGraph setup, state definition, and node logic.
- `rag_pipeline.py`: Defines the text-splitting, embedding, and vector store retrieval logic.
- `knowledge_base.md`: The markdown file acting as the source of truth for the RAG pipeline.
- `requirements.txt`: Project dependencies.

## Demo Scenarios to Try

- **Casual**: *"Hey there, how are you?"*
- **Product Inquiry**: *"What features are included in the Pro plan?"*
- **High Intent / Lead Capture**: *"I want to buy the Pro plan. My name is Alex, and my email is alex@example.com."* (The agent should then ask for your creator platform).
