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
from lib.outcome_engine import synthesize_single_tile
from lib.storybook_generator import generate_heros_journey_text, generate_comic_book, generate_pixel_art_illustration

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
.visualizing-spinner { color: #a78bfa; font-style: italic; font-size: 0.9em; margin-top: 10px; animation: pulse 2s infinite; }
.canvas-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; padding: 10px; }
.canvas-tile { background: rgba(30, 30, 40, 0.8); border: 1px solid rgba(139, 92, 246, 0.6); border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.5); transition: transform 0.2s; }
.canvas-tile:hover { transform: translateY(-5px); border-color: #38bdf8; box-shadow: 0 8px 25px rgba(56, 189, 248, 0.3); }
.tile-img { width: 100%; height: 160px; object-fit: cover; border-bottom: 2px solid #22d3ee; }
.tile-body { padding: 15px; }
.tile-category { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: #c084fc; font-weight: bold; margin-bottom: 5px; }
.tile-title { font-size: 1.2rem; color: #f8fafc; margin: 0 0 10px 0; }
.tile-desc { font-size: 0.9rem; color: #cbd5e1; line-height: 1.4; margin-bottom: 15px; }
.tile-meta { font-size: 0.8rem; background: rgba(56, 189, 248, 0.1); padding: 5px 8px; border-radius: 4px; border: 1px solid rgba(56, 189, 248, 0.2); display: inline-block; margin-right: 5px; margin-bottom: 5px; color: #38bdf8; }
.comic-panel { background: rgba(0,0,0,0.4); padding: 15px; border-radius: 12px; border: 1px solid #8b5cf6; margin-bottom: 20px; text-align: center; }
.comic-img { max-width: 100%; border-radius: 8px; border: 2px solid #38bdf8; box-shadow: 0px 0px 15px rgba(56, 189, 248, 0.3); }
.comic-cap { color: #22d3ee; margin-top: 15px; font-weight: normal; font-style: italic; line-height: 1.4; }
@keyframes pulse { 0% { opacity: 0.5; } 50% { opacity: 1; } 100% { opacity: 0.5; } }
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

    # Yield intermediate UI state for the HUD while Outcome Engine runs
    yield history, "<div class='visualizing-spinner' style='text-align:center;'>🖼️ Synthesizing new canvas tile...</div>"

    # 5. Background Update: Run the Tile Generator
    last_msg_text = extract_text(history[-2]["content"])
    latest_user_msg =[{"role": "user", "text": last_msg_text}]
    
    new_tile_data = synthesize_single_tile(backend_history + latest_user_msg, state_data["superpowers"])
    
    if new_tile_data:
        # Generate the pixel art for the tile
        img_b64 = generate_pixel_art_illustration(new_tile_data.get("image_prompt", "Abstract career concept art, neon pixel style"))
        new_tile_data["image_b64"] = img_b64
        state_data["tiles"].append(new_tile_data)
    
    # 6. Format the Canvas HTML
    canvas_html = format_canvas(state_data["tiles"])
    yield history, canvas_html

def format_canvas(tiles):
    if not tiles:
        return "<p style='text-align:center; color:#64748b; margin-top: 20px;'>Start exploring to generate your unique Canvas Tiles.</p>"
    
    html = "<div class='canvas-grid'>"
    # Render newest tiles first
    for tile in reversed(tiles):
        img_src = f"data:image/png;base64,{tile.get('image_b64')}" if tile.get('image_b64') else ""
        img_html = f"<img src='{img_src}' class='tile-img'>" if img_src else "<div class='tile-img' style='background:#1e1b4b; display:flex; align-items:center; justify-content:center; color:#8b5cf6;'>[Image Failed]</div>"
        
        meta_html = ""
        meta_list = tile.get("metadata", [])
        if isinstance(meta_list, list):
            for item in meta_list:
                meta_html += f"<span class='tile-meta'>{item}</span>"
        
        html += f"""
        <div class='canvas-tile'>
            {img_html}
            <div class='tile-body'>
                <div class='tile-category'>{tile.get('category', 'Insight')}</div>
                <h3 class='tile-title'>{tile.get('title', 'Unknown')}</h3>
                <p class='tile-desc'>{tile.get('content', '')}</p>
                <div>{meta_html}</div>
            </div>
        </div>
        """
    html += "</div>"
    return html

def generate_comic(history):
    if not history or len(history) < 2:
        yield "<p style='color: #ff4a4a; text-align: center;'>⚠️ Please chat with Arc first to give it material for your comic!</p>"
        return
        
    yield "<div class='visualizing-spinner' style='text-align:center;'>🖌️ Synthesizing your story into a comic book...</div>"
    
    # Run the background generator logic
    # Clean history for gen
    clean_history = []
    for msg in history:
        if msg["role"] == "assistant":
            clean_history.append({"role": "model", "text": extract_text(msg["content"])})
        else:
            clean_history.append({"role": "user", "text": extract_text(msg["content"])})
            
    panels = generate_comic_book(clean_history)
    
    if not panels:
        yield "<p style='color: #ff4a4a; text-align: center;'>⚠️ Failed to generate comic. Please try again.</p>"
        return
        
    final_html = ""
    for panel in panels:
        img_src = f"data:image/png;base64,{panel['image_b64']}" if panel['image_b64'] else ""
        img_tag = f"<img src='{img_src}' class='comic-img'>" if img_src else ""
        
        final_html += f"""
        <div class='comic-panel'>
            {img_tag}
            <h3 class='comic-cap'>"{panel['caption']}"</h3>
        </div>
        """
    yield final_html

# --- GRADIO UI LAYOUT ---
with gr.Blocks() as demo:
    gr.HTML(f"<style>{css}</style>")
    state = gr.State({"superpowers": {}, "tiles": []})
    
    gr.HTML("<h1 style='text-align:center; color:#f8fafc; margin-bottom: 0;'>🕹️ ArcMotivate</h1>")
    gr.HTML("<p style='text-align:center; color:#a78bfa; margin-top:0; letter-spacing: 2px;'>INTERACTIVE CAREER SIMULATOR</p>")
    
    with gr.Row():
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": "👋 **System Online.** ArcMotivate helps young minds reach their full potential. Use this tool to support you on exploring some career paths that you might want to explore. To start: What are 3 hobbies or topics you love exploring?"}],
                height=600,
                show_label=False
            )
            with gr.Row():
                msg_input = gr.Textbox(placeholder="Enter your next move here...", show_label=False, scale=8)
                submit_btn = gr.Button("Execute 🚀", variant="primary", scale=2)
                
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.TabItem("🖼️ Live Canvas"):
                    canvas_output = gr.HTML("<p style='text-align:center; color:#64748b; margin-top: 20px;'>Start exploring to generate your unique Canvas Tiles.</p>")
                
                with gr.TabItem("📖 My Career Comic"):
                    gr.HTML("<p style='text-align:center; color:#94a3b8; margin-top:10px;'>Turn your interactions into a custom comic book summarizing your journey.</p>")
                    comic_btn = gr.Button("🎨 Generate Comic Book", variant="secondary")
                    comic_output = gr.HTML("")

    submit_event = msg_input.submit(user_submit,[msg_input, chatbot], [msg_input, chatbot], queue=False)
    submit_btn.click(user_submit,[msg_input, chatbot], [msg_input, chatbot], queue=False)
    
    submit_event.then(process_simulation, [chatbot, state], [chatbot, canvas_output])
    submit_btn.click(process_simulation, [chatbot, state], [chatbot, canvas_output])
    
    comic_btn.click(generate_comic, [chatbot], [comic_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)