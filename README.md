# ArcMotivate - Your Interactive Career Exploration Agent 🕹️

ArcMotivate is a high-performance career simulator that bridges the gap between personality traits and professional roles for users aged 8–18. It uses a Socratic coaching loop and visual artifacts to help users discover realistic career paths.

It utilises the multimodal interleaved capabilities of Gemini 3.1 Flash Lite to generate responsive answers to the users question.

Each user is free to generate their own storybook of their future by exploring new careers 'Arcs'. Unlike a traditional career coach, the tool assumes that anything is possible, hence it is possible to explore freely in a world where education often limits the choices. 

## Core Logic

- **Professional Archetyping**: Analyzes chat and images to map user interests to "Professional Archetypes" (e.g., *The Technical Maker*, *The Data Storyteller*).
- **Interleaved Visuals**: Streams neon pixel-art metaphors directly into the chat flow using Gemini-powered markers.
- **Career Tiles**: Dynamically generates interactive cards featuring job titles, industry skill tags, actionable nudges, and direct Google search links.
- **Identity Snapshot**: A periodic background process that asynchronously builds a persistent identity:
    - **Custom Avatar**: Personalized profile icon.
    - **Identity Comic**: 3-panel pixel-art story of your exploration.
    - **Future Postcard**: A forward-looking visual and message from your "future self".
- **Silent UX**: Focused, distraction-free interface optimized for performance.

## Tech Stack

- **Large Language Models**: Powered by the Gemini Flash family for low-latency reasoning and image generation.
- **Frontend**: Gradio-based SPA with custom retro/cyberpunk styling.
- **State Engine**: Thread-safe session management for background identity generation.

### ☁️ Secure Cloud Run Deployment

1.  **Connect Repo**: Point your Cloud Run service to this repository.
2.  **Secret Manager Setup**:
    *   Create a secret named `gemini-api-key` in [Google Secret Manager](https://console.cloud.google.com/security/secret-manager).
    *   Add your Gemini API Key as the secret value.
    *   Grant the **Secret Manager Secret Accessor** role to your Cloud Run service account (`cloud-run-sa@...`).
    *   **Crucial**: Grant the **Service Account User** role to the Cloud Build service account (`[PROJECT_NUMBER]@cloudbuild.gserviceaccount.com`) on the Cloud Run service account to allow the deploy to finish.
    *   **Manual Mapping**: If the build succeeds but the app fails to start, go to **Cloud Run > Edit & Deploy New Revision > Variables & Secrets** and manually map the `gemini-api-key` secret to the `GOOGLE_API_KEY` environment variable.
3.  **Deploy**: The included `cloudbuild.yaml` automatically maps the secret to `GOOGLE_API_KEY`.
    *   *Note*: If the build fails initially, ensure the `gemini-api-key` secret actually exists in Secret Manager.

---
## Spin-Up Instructions

### 1. Prerequisites
- Python 3.10+
- A Google Gemini API Key

### 2. Setup
```bash
# Clone the repository
cd arcmotivate

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY="your_api_key_here"
```

### 4. Run
```bash
python app.py
```
The interface will be available at `http://localhost:7860`.
