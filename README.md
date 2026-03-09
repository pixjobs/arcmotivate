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

## 🛠️ Tech Stack & Architecture

- **Large Language Models**: Powered by the Gemini Flash family for low-latency reasoning, dynamic narrative generation, and pixel-art image synthesis.
- **Frontend**: Mobile-optimized, responsive Gradio SPA (Single Page Application) featuring custom retro/cyberpunk CSS styling, touch-friendly inputs, and dynamic viewport scaling.
- **State Engine**: Thread-safe session management for background identity generation and asynchronous artifact rendering.
- **Zero-Latency Caching (Cloud Run Optimized)**: Implements a custom 3-Tier Waterfall Cache (RAM → Local `/tmp` Disk → Google Cloud Storage Blob) to completely eliminate serverless cold-start latency and redundant API costs for global assets.

---

## ☁️ Secure Cloud Run Deployment

To deploy this in a production serverless environment, follow these steps:

### 1. Connect Repository
Point your Google Cloud Run service to this repository for continuous deployment.

### 2. Secret Manager Setup (API Keys)
*   Create a secret named `gemini-api-key` in [Google Secret Manager](https://console.cloud.google.com/security/secret-manager).
*   Add your Gemini API Key as the secret value.
*   Grant the **Secret Manager Secret Accessor** role to your Cloud Run service account.
*   **Crucial**: Grant the **Service Account User** role to the Cloud Build service account (`[PROJECT_NUMBER]@cloudbuild.gserviceaccount.com`) on the Cloud Run service account to allow the deploy to finish.
*   *Note:* The included `cloudbuild.yaml` automatically maps this secret to the `GOOGLE_API_KEY` environment variable. If it fails, you can manually map it in the Cloud Run UI under **Variables & Secrets**.

### 3. Cloud Storage Setup (Cross-Container Caching)
To enable the zero-latency 3-tier cache across ephemeral containers:
*   Create a **Private** standard Google Cloud Storage bucket (e.g., `arc-motivate-cache`). Keep "Enforce public access prevention" **checked**.
*   Go to the bucket's **Permissions** tab and grant the **Storage Object Admin** role to your Cloud Run Service Account.
*   In your Cloud Run service settings, add a standard Environment Variable:
    *   `CACHE_BUCKET_NAME` = `your-bucket-name`

---

## 🚀 Local Spin-Up Instructions

### 1. Prerequisites
- Python 3.10+
- A Google Gemini API Key
- *(Optional)* A Google Cloud Storage bucket for testing the blob cache.

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/your-repo/arcmotivate.git
cd arcmotivate

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

. Configuration
Create a .env file in the root directory:

# Required
GOOGLE_API_KEY="your_gemini_api_key_here"

# Optional (Enables the 3-Tier GCS Cache)
CACHE_BUCKET_NAME="your-gcs-bucket-name"

4. Run

python app.py

The interface will be available at http://localhost:7860.
(Note: On the very first load, the app will take a few seconds to generate the intro message and image. Subsequent loads and refreshes will be instant as it reads from the local /tmp cache or GCS bucket).