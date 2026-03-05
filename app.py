import gradio as gr
import base64
import os
import time
import re
from dotenv import load_dotenv

# Load environment variables first
load_dotenv() 

from lib.psychology_codex import map_interests_to_superpowers
from lib.coaching_agent import generate_socratic_stream
from lib.outcome_engine import synthesize_blueprint
from lib.storybook_generator import generate_heros_journey_text, generate_pixel_art_illustration

# --- CSS FOR ARCADE/NEON VIBE ---
css = """
body { background: linear-gradient(135deg, #09090b 0%, #170f2e 100%); color: #f8fafc; font-family: 'Inter', sans-serif; }
.gradio-container { max-width: 1400px !important; border: none !important; }
.hud-panel { background: rgba(30, 30, 40, 0.6); border: 1px solid rgba(34, 211, 238, 0.4); border-radius: 12px; padding: 25px; box-shadow: 0px 8px 32px rgba(0,0,0,0.5); }
.hud-title { color: #22d3ee; font-size: 1.8rem; font-weight: bold; margin-bottom: 5px; text-shadow: 0 0 10px rgba(34, 211, 238, 0.3);}
.hud-subtitle { color: #8b5cf6; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px; margin-top:0; margin-bottom: 20px;}
.hud-section { color: #38bdf8; font-size: 1.1rem; font-weight: bold; margin-top: 20px; margin-bottom: 10px;}
.trait-pill { display: inline-block; background: #1e1b4b; color: #c084fc; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: bold; margin: 3px; border: 1px solid #c084fc;}
.skill-pill { display: inline-block; background: rgba(56, 189, 248, 0.1); color: #38bdf8; padding: 5px 12px; border-radius: 6px; font-size: 0.85rem; margin: 3px; border: 1px solid rgba(56, 189, 248, 0.3);}
.quest-box { background: rgba(34, 211, 238, 0.1); border-left: 4px solid #22d3ee; padding: 15px; border-radius: 4px; color: #e2e8f0; margin-top: 15px;}
"""

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
        return " ".join(texts)
    return str(content)

DEFAULT_HUD = """
<div class='hud-panel'>
    <h2 class='hud-title'>Future Blueprint</h2>
    <p class='hud-subtitle'>Awaiting Data...</p>
    <p style='color:#64748b; margin-top: 20px;'>Start the simulation in the chat panel to unlock your psychological profile.</p>
</div>
"""

def user_submit(user_message, history):
    if history is None:
        history =[]
    history.append({"role": "user", "content": user_message})
    return "", history

def process_simulation(history, state_data):
    # 1. Initialization (Run Psychology Codex on first message)
    if not state_data.get("superpowers"):
        user_input = extract_text(history[-1]["content"])
        interests =[i.strip() for i in user_input.split(",")]
        state_data["superpowers"] = map_interests_to_superpowers(interests)
    
    # 2. Add an empty AI message to the history for streaming
    history.append({"role": "assistant", "content": ""})
    
    # 3. CRITICAL FIX: Clean the history! 
    # We must strip the HTML <img> tags out of the history so we don't send 
    # massive Base64 strings back to Gemini and crash the API.
    backend_history = []
    for msg in history[:-1]:
        raw_content = extract_text(msg["content"])
        clean_content = re.sub(r"<img[^>]*>", "", raw_content).strip()
        if not clean_content:
            continue
        
        role = "user" if msg["role"] == "user" else "model"
        backend_history.append({"role": role, "text": clean_content})
    
    # 4. Stream the Socratic Agent
    stream = generate_socratic_stream(state_data["superpowers"], backend_history)
    
    for chunk in stream:
        if chunk["type"] == "text":
            history[-1]["content"] += chunk["data"]
            yield history, gr.update() 
            
        elif chunk["type"] == "image":
            img_md = f"\n\n<img src='data:image/png;base64,{chunk['data']}' width='300' style='border-radius:10px; margin:10px 0; border: 1px solid #8b5cf6;'>\n\n"
            history[-1]["content"] += img_md
            yield history, gr.update() 
            time.sleep(0.1) 

    # 5. Background Update: Run the Outcome Engine (Pro Model)
    last_msg_text = extract_text(history[-2]["content"])
    latest_user_msg =[{"role": "user", "text": last_msg_text}]
    blueprint = synthesize_blueprint(backend_history + latest_user_msg, state_data["superpowers"])
    state_data["blueprint"] = blueprint
    
    # 6. Format the HUD HTML
    hud_html = format_hud(blueprint, state_data["superpowers"])
    yield history, hud_html

def format_hud(bp, sp):
    if not bp: return DEFAULT_HUD
    
    traits = "".join([f"<span class='trait-pill'>{t}</span>" for t in bp.get("psychology_traits", [])])
    skills = "".join([f"<span class='skill-pill'>{s}</span>" for s in bp.get("skills", [])])
    careers = "".join([f"<li><span style='color:#38bdf8; font-weight:bold;'>{c}</span></li>" for c in bp.get("careers", [])])
    analysis = bp.get("path_analysis", "...").replace("\n", "<br><br>")
    
    return f"""
    <div class='hud-panel'>
        <h2 class='hud-title'>{bp.get('hero_title', 'Analyzing...')}</h2>
        <p class='hud-subtitle'>Role: {sp.get('primary', 'Explorer')} | Style: {sp.get('secondary', 'Unknown')}</p>
        
        <div>{traits}</div>
        
        <h4 class='hud-section'>🧠 Profile Analysis</h4>
        <p style='color:#cbd5e1; line-height: 1.5; font-size: 0.95rem;'>{analysis}</p>
        
        <h4 class='hud-section'>🛠️ Verified Skills</h4>
        <div>{skills}</div>
        
        <h4 class='hud-section'>🚀 Target Careers</h4>
        <ul style='color:#cbd5e1; font-size: 0.95rem;'>{careers}</ul>
        
        <div class='quest-box'>
            <strong>🎯 Today's Quest:</strong><br>{bp.get('next_step', 'Keep exploring.')}
        </div>
    </div>
    """

def generate_poster(state_data):
    bp = state_data.get("blueprint")
    sp = state_data.get("superpowers")
    if not bp or not sp:
        return "<p style='color: #ff4a4a; text-align: center;'>⚠️ Please complete at least one chat turn before generating a poster!</p>"
        
    story_text = generate_heros_journey_text({"superpowers": sp})
    poster_b64 = generate_pixel_art_illustration(f"A futuristic {bp.get('hero_title', 'hero')} working on a cool project.")
    
    if poster_b64:
        return f"""
        <div style='text-align: center; background: rgba(0,0,0,0.4); padding: 20px; border-radius: 15px; border: 1px solid #8b5cf6;'>
            <img src='data:image/png;base64,{poster_b64}' style='max-width:100%; border-radius:10px; border:2px solid #38bdf8; box-shadow: 0px 0px 20px rgba(56, 189, 248, 0.3);'>
            <h3 style='color:#22d3ee; margin-top:20px; font-weight: normal; font-style: italic; line-height: 1.5;'>"{story_text}"</h3>
        </div>
        """
    return f"<h3 style='color:#22d3ee; text-align:center;'>{story_text}</h3>"

# --- GRADIO UI LAYOUT ---
with gr.Blocks() as demo:
    gr.HTML(f"<style>{css}</style>")
    state = gr.State({"superpowers": {}, "blueprint": None})
    
    gr.HTML("<h1 style='text-align:center; color:#f8fafc; margin-bottom: 0;'>🕹️ ArcMotivate</h1>")
    gr.HTML("<p style='text-align:center; color:#a78bfa; margin-top:0; letter-spacing: 2px;'>INTERACTIVE CAREER SIMULATOR</p>")
    
    with gr.Row():
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": "👋 **System Online.** I am Arc. Let's map your future. To start: What are 3 hobbies or topics you love exploring?"}],
                height=600,
                show_label=False
            )
            with gr.Row():
                msg_input = gr.Textbox(placeholder="Enter your next move here...", show_label=False, scale=8)
                submit_btn = gr.Button("Execute 🚀", variant="primary", scale=2)
                
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.TabItem("🧬 Live Matrix"):
                    hud_output = gr.HTML(DEFAULT_HUD)
                
                with gr.TabItem("🖨️ Export Poster"):
                    gr.HTML("<p style='text-align:center; color:#94a3b8; margin-top:10px;'>When you are happy with your profile, print your final Career Legend card.</p>")
                    poster_btn = gr.Button("Generate High-Res Arcade Poster", variant="secondary")
                    poster_output = gr.HTML("")

    submit_event = msg_input.submit(user_submit,[msg_input, chatbot], [msg_input, chatbot], queue=False)
    submit_btn.click(user_submit,[msg_input, chatbot], [msg_input, chatbot], queue=False)
    
    submit_event.then(process_simulation, [chatbot, state], [chatbot, hud_output])
    submit_btn.click(process_simulation, [chatbot, state], [chatbot, hud_output])
    
    poster_btn.click(generate_poster, [state], [poster_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)