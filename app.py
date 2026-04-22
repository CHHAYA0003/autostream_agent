import gradio as gr
import time
import traceback
import warnings
from langchain_core.messages import HumanMessage
from agent import graph

# Suppress harmless multiprocessing leaked semaphore warning on shutdown
warnings.filterwarnings("ignore", module="multiprocessing.resource_tracker")

# ─── CSS Definitions ─────────────────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Dashboard styling */
.panel-card {
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid var(--border-color-primary);
    background-color: var(--background-fill-secondary);
}

.terminal {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85em !important;
    padding: 15px !important;
    border-radius: 8px !important;
    height: 250px;
    overflow-y: auto;
    background-color: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
}

.workflow-step {
    height: 6px;
    flex: 1;
    background: var(--border-color-primary);
    border-radius: 3px;
    transition: all 0.3s ease;
}
.workflow-step.active {
    background: #06b6d4;
    box-shadow: 0 0 10px #06b6d4;
}

.dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    border: 2px solid var(--border-color-primary);
}
.dot.filled {
    background: #06b6d4;
    border-color: #06b6d4;
}

/* Fix Chat Bubbles to adapt to Light/Dark Mode */
/* Removed hardcoded dark colors so it looks native and visible */
footer { display: none !important; }

/* Sleek Chat Wrapper */
.chat-wrapper {
    background-color: var(--background-fill-secondary) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    border: 1px solid var(--border-color-primary) !important;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1) !important;
}

/* Custom Input Row */
.input-wrapper {
    background-color: var(--background-fill-primary) !important;
    border-radius: 12px !important;
    padding: 5px 10px !important;
    border: 1px solid var(--border-color-primary) !important;
    margin-top: 15px !important;
    align-items: center !important;
}

.input-wrapper textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--body-text-color) !important;
    font-size: 1.1em !important;
}

.input-wrapper textarea:focus {
    border: none !important;
    box-shadow: none !important;
}

/* Chat Header */
.chat-header {
    font-size: 1.25em;
    font-weight: 600;
    color: var(--body-text-color);
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Top Branding */
.top-brand {
    font-size: 1.5em;
    font-weight: 700;
    color: var(--body-text-color);
    display: flex;
    align-items: center;
    gap: 10px;
}
.top-brand span {
    background: #334155;
    color: #94a3b8;
    font-size: 0.45em;
    padding: 4px 10px;
    border-radius: 12px;
    font-weight: 500;
}


.message-wrap .message p {
    font-size: 1.15em !important;
    line-height: 1.5 !important;
}

/* Make sure the text color adapts well */
.message-wrap .message {
    color: var(--body-text-color) !important;
}

/* Custom send button matching image 2's vibe */
.send-btn {
    background-color: #06b6d4 !important;
    border-radius: 50% !important;
    min-width: 45px !important;
    width: 45px !important;
    height: 45px !important;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0 !important;
    color: white !important;
    font-size: 1.2em !important;
}

"""

# ─── UI Helper Components ────────────────────────────────────────────────────
def get_workflow_html(stage):
    steps = ["GREETING", "INQUIRY", "QUALIFICATION", "CAPTURE", "SUCCESS"]
    html = '<div style="display:flex; gap:8px; margin-bottom:8px;">'
    for i in range(1, 6):
        cls = "active" if i <= stage else ""
        html += f'<div class="workflow-step {cls}"></div>'
    html += '</div><div style="display:flex; justify-content:space-between; font-size:0.65em; color:var(--body-text-color-subdued); text-transform:uppercase; font-weight:600;">'
    for s in steps: html += f'<span>{s}</span>'
    html += '</div>'
    return html

def get_checklist_html(st):
    items = [("Name [REQUIRED]", st.get("name")), 
             ("Email [REQUIRED]", st.get("email")), 
             ("Platform [REQUIRED]", st.get("platform"))]
    html = '<div style="display:flex; flex-direction:column; gap:10px;">'
    for label, val in items:
        # Sanitize "null" strings
        val_display = "" if not val or str(val).lower().strip() in ["null", "none", ""] else str(val).strip()
        border_col = "#06b6d4" if val_display else "var(--border-color-primary)"
        html += f'''
        <div style="display:flex; align-items:center; background:var(--background-fill-primary); 
                    border:1px solid {border_col}; padding:10px 15px; border-radius:8px; font-size:0.9em; color:var(--body-text-color);">
            {val_display if val_display else label}
        </div>
        '''
    html += '</div>'
    if st.get("lead_captured"):
         html += '<div style="margin-top:10px; color:#10b981; font-size:0.8em; font-weight: bold;">✅ TOOL EXECUTION SUCCESSFUL</div>'
    elif st.get("name") or st.get("email") or st.get("platform"):
         html += '<div style="margin-top:10px; color:#f97316; font-size:0.8em; font-weight: bold;">⚠️ Awaiting full lead details...</div>'
    return html

def format_logs(logs):
    html = '<div class="terminal">'
    for l in reversed(logs):
        color = "#06b6d4" if l.get("tag") == "INFO" else "#10b981" if l.get("tag") == "WAIT" else "#f43f5e" if "ERROR" in l.get("msg","") else "var(--body-text-color)"
        html += f'<div style="margin-bottom:4px;"><span style="color:{color};">[{l["ts"]}]</span> {l["msg"]}</div>'
    html += '</div>'
    return html

# Change theme dynamically based on user selection
def toggle_theme(is_dark):
    # This toggles via JS mostly, but we can return the theme string
    pass

# ─── MAIN APP ───────────────────────────────────────────────────────────────
with gr.Blocks(title="AutoStream | Agent Portal") as demo:
    
    st_val = gr.State({"messages": [], "name": "", "email": "", "platform": "", "lead_captured": False, "intent": "N/A"})
    lg_val = gr.State([{"ts": time.strftime("%H:%M:%S"), "tag": "INFO", "msg": "Initializing AutoStream Agent..."}])
    
    
    # ─── TOP BRANDING ───
    with gr.Row():
        gr.HTML("""
        <div style="display:flex; justify-content: space-between; align-items:center; width: 100%; padding-bottom:20px;">
            <div class="top-brand">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right:-2px; margin-bottom:-2px;">
                  <defs>
                    <linearGradient id="aGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stop-color="#22d3ee" />
                      <stop offset="100%" stop-color="#4f46e5" />
                    </linearGradient>
                  </defs>
                  <path d="M12 3L4 20" stroke="url(#aGrad)" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>
                  <path d="M12 3L20 20" stroke="url(#aGrad)" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>
                  <path d="M7 14C9 14 12 11 17.5 14" stroke="url(#aGrad)" stroke-width="3.5" stroke-linecap="round"/>
                </svg>
                AutoStream 
                <span>ServiceHive Intern Portal</span>
            </div>
            <button onclick="document.body.classList.toggle('dark');" style="height: 35px; padding: 0 15px; border-radius: 8px; border: 1px solid var(--border-color-primary); background: var(--background-fill-primary); cursor:pointer; color:var(--body-text-color);">🌗 Toggle Theme</button>
        </div>
        """)

    with gr.Row():
        # ─── LEFT PANEL: CHAT ───
        with gr.Column(scale=3, elem_classes="chat-wrapper"):
            gr.HTML('<div class="chat-header">AutoStream Agent Chat</div>')
            
            chatbot = gr.Chatbot(
                show_label=False, 
                height=550, 
                layout="bubble"
            )
            
            with gr.Row(elem_classes="input-wrapper"):
                msg_in = gr.Textbox(
                    placeholder="Type your message...", 
                    scale=9, 
                    container=False, 
                    show_label=False
                )
                snd_bt = gr.Button("➤", scale=1, variant="primary", elem_classes="send-btn")

            gr.Examples(
                examples=[
                    "Hi! What are your pricing plans?",
                    "Do you offer 4K resolution exports?",
                    "I want to sign up for the Pro plan. I'm Alex."
                ],
                inputs=msg_in,
                label=""
            )

        # ─── RIGHT PANEL: DASHBOARD ───
        with gr.Sidebar(open=False, position="right"):
            gr.HTML("""
            <div style="padding-bottom:15px; margin-bottom:20px;">
                <h3 style="margin:0; font-weight:600; color:var(--body-text-color);">Agent Intelligence & Workflow</h3>
            </div>
            """)
            
            with gr.Column(elem_classes="panel-card"):
                gr.Markdown("#### Workflow Status")
                wf_ui = gr.HTML(get_workflow_html(1))
                
            with gr.Column(elem_classes="panel-card"):
                gr.Markdown("#### Agent State")
                with gr.Row():
                    intent_ui = gr.Markdown("**Intent:** N/A")
                    turn_ui = gr.Markdown("**Turn:** 0")
                    
            with gr.Column(elem_classes="panel-card"):
                gr.Markdown("#### Lead Qualification")
                cl_ui = gr.HTML(get_checklist_html({}))
                
            with gr.Column(elem_classes="panel-card"):
                gr.Markdown("#### System Logs")
                lo_ui = gr.HTML(format_logs([{"ts": time.strftime("%H:%M:%S"), "tag": "INFO", "msg": "Initializing..."}]))

    # ─── EVENTS ───
    def process_chat(msg, history, state, logs):
        if not msg.strip():
            yield history, state, logs, get_workflow_html(1), "**Intent:** N/A", "**Turn:** 0", get_checklist_html(state), format_logs(logs), ""
            return
            
        ts = time.strftime("%H:%M:%S")
        logs.append({"ts": ts, "tag": "INPUT", "msg": f"Received: {msg}"})
        
        # Immediate UI update with user message
        history.append({"role": "user", "content": msg})
        yield history, state, logs, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), ""
        
        try:
            state["messages"].append(HumanMessage(content=msg))
            result = graph.invoke(state)
            state.update(result)
            
            res_msg = result["messages"][-1].content
            bot_msg = str(res_msg) if not isinstance(res_msg, list) else res_msg[0].get("text", str(res_msg))
            
            intent = state.get("intent", "N/A")
            stage = 1
            if intent == "casual_greeting": stage = 1
            elif intent == "product_inquiry": stage = 2
            elif intent == "high_intent_lead": stage = 3
            if state.get("lead_captured"): stage = 5
            
            logs.append({"ts": time.strftime("%H:%M:%S"), "tag": "INFO", "msg": f"Intent classified: {intent.upper()}"})
            history.append({"role": "assistant", "content": bot_msg})
            
            turns = len([m for m in history if m["role"] == "user"])
            
            yield (
                history, 
                state, 
                logs, 
                get_workflow_html(stage), 
                f"**Intent:**\n{intent}", 
                f"**Turn:**\n{turns}", 
                get_checklist_html(state), 
                format_logs(logs),
                ""
            )
            
        except Exception as e:
            err = traceback.format_exc()
            logs.append({"ts": time.strftime("%H:%M:%S"), "tag": "ERROR", "msg": f"ERROR: {str(e)}"})
            history.append({"role": "assistant", "content": "I encountered an error. Please check the backend logs."})
            
            turns = len([m for m in history if m["role"] == "user"])
            yield (
                history, state, logs, gr.update(), gr.update(), gr.update(), gr.update(), format_logs(logs), ""
            )

    ev_args = dict(fn=process_chat, inputs=[msg_in, chatbot, st_val, lg_val], 
                   outputs=[chatbot, st_val, lg_val, wf_ui, intent_ui, turn_ui, cl_ui, lo_ui, msg_in])
    
    snd_bt.click(**ev_args)
    msg_in.submit(**ev_args)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", css=custom_css, inbrowser=True)
