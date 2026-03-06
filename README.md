# ArcMotivate 🕹️

**Interactive Career Simulator for Young Minds**

> **Note:** This submission was generated for the showcase hackathon requiring entrants to develop a NEW next-generation AI Agent that utilizes multimodal inputs and outputs, moving beyond simple text-in/text-out interactions, and leveraging Google’s Live API with the creative power of image generation to solve complex problems and create entirely new user experiences (specifically within the "Creative Storyteller" or "Live Agents" category).

ArcMotivate is a sleek, modern, and realistic career exploration simulator designed for users aged 8–18. It goes beyond patronizing or unrealistic career ideas and provides grounded, actionable, contemporary industry insights. 

## Features

- **Socratic Career Coaching:** A responsive chat agent powered by Gemini that takes it slow, asking deeply engaging questions one at a time to build a realistic career trajectory.
- **Multimodal Live Canvas:** During the conversation, a background AI engine analyzes your choices and streams uniquely generated pixel-art Canvas Tiles into an interactive masonry grid. Each tile represents a learned skill, discovered trait, or active quest.
- **Career Comic Book:** Turn your entire career exploration session into a synthesized 3-panel comic book with custom pixel-art illustrations.
- **Session State Management:** Secure, dynamic session tracking using Gradio's state objects without requiring an external database.

## Technologies Used

- **Google GenAI Python SDK** (Powered by `gemini-3-flash-preview` and `gemini-3.1-flash-image-preview`)
- **Gradio** (For real-time streaming and multimodal UI)
- **Vanilla CSS** (For a retro-arcade, neon cyberpunk aesthetic)

## Setup & Running

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up your `.env` file with your Gemini API Key:
   ```
   GEMINI_API_KEY="your_api_key_here"
   ```
3. Run the application:
   ```bash
   python3 app.py
   ```
