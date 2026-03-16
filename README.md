# ArcMotivate -- Interactive Career Exploration Agent 🕹️

ArcMotivate is an interactive career exploration tool designed for young
people aged **8--18**. It helps users connect their interests, hobbies,
and personality traits with **real-world career paths** through
conversational exploration and visual storytelling.

The system combines a Socratic-style coaching conversation with
multimodal AI to guide users through potential "career arcs". As users
explore, the system generates visual artifacts that summarize their
journey and help them imagine future possibilities.

ArcMotivate encourages curiosity and reflection, allowing users to
experiment with different ideas and discover careers they may not have
previously encountered.

------------------------------------------------------------------------

# Core Experience

## Conversational Exploration

Users interact with a responsive chat interface that encourages
reflection about interests, skills, and aspirations. The conversation
dynamically adapts based on the user's responses.

The system uses the **Gemini Flash family of models** (primarily
`gemini-3.1-flash-lite-preview` and `gemini-3-flash-preview`) to
generate structured responses, prompts, and visual artifacts.

## Career Discovery

During the conversation, ArcMotivate surfaces **Career
Tiles**---interactive cards containing:

-   Job titles
-   Industry skill tags
-   Exploration prompts
-   Safe Google search links for further learning

These help users connect their interests to **real career paths**.

## Identity Snapshot (Storybook)

As the user explores, ArcMotivate asynchronously builds a visual
"Identity Snapshot" summarizing their journey:

-   **Custom Avatar** -- a personalized profile icon\
-   **Identity Comic** -- a 3‑panel pixel‑art narrative of their
    exploration *(Spark → Experiment → Direction)*\
-   **Future Postcard** -- a message and scene from a possible future
    self

These artifacts form a lightweight **storybook of the user's career
exploration**.

------------------------------------------------------------------------

# 🛠️ Tech Stack & Architecture

## AI Models

ArcMotivate uses the **Google GenAI SDK** with models from the Gemini
Flash family:

-   `gemini-3.1-flash-lite-preview`
-   `gemini-3-flash-preview`

These models power:

-   Conversational reasoning
-   Structured JSON generation
-   Narrative creation
-   Image prompt generation

## Frontend

-   **Gradio Single Page Application (SPA)**
-   Mobile‑optimized responsive layout
-   Custom retro / cyberpunk pixel-art UI styling

## Structured Generation

All model outputs are constrained using **strict JSON schemas**. This
ensures the application receives predictable structured data that
separates:

-   Narrative text
-   Image prompts
-   Career suggestions
-   Resource links

## Parallel Artifact Generation

To maintain responsiveness, ArcMotivate generates visual artifacts
concurrently using:

`concurrent.futures.ThreadPoolExecutor`

This allows the avatar, comic panels, and future postcard to render in
parallel without blocking the conversation.

## Session State Engine

Thread‑safe session state manages:

-   User conversation history
-   Background artifact generation
-   Persistent identity snapshots

------------------------------------------------------------------------

# Performance & Caching

ArcMotivate implements a **3‑tier cache architecture** optimized for
Cloud Run deployments:

1.  **RAM Cache** -- fastest asset retrieval
2.  **Local `/tmp` Disk Cache** -- container‑level persistence
3.  **Google Cloud Storage Blob Cache** -- cross‑container asset sharing

This approach reduces cold‑start latency and avoids repeated API calls
for global assets.

------------------------------------------------------------------------

# ☁️ Secure Cloud Run Deployment

## 1. Connect Repository

Point your Google Cloud Run service to this repository for continuous
deployment.

## 2. Secret Manager Setup (API Keys)

Create a secret named:

`gemini-api-key`

in **Google Secret Manager**.

Steps:

1.  Add your Gemini API key as the secret value.
2.  Grant the **Secret Manager Secret Accessor** role to the Cloud Run
    service account.
3.  Grant the **Service Account User** role to the Cloud Build service
    account:

`[PROJECT_NUMBER]@cloudbuild.gserviceaccount.com`

This allows Cloud Build to deploy the service successfully.

The included `cloudbuild.yaml` automatically maps the secret to:

`GOOGLE_API_KEY`

If needed, you can also map this manually in **Cloud Run → Variables &
Secrets**.

------------------------------------------------------------------------

## 3. Cloud Storage Setup (Cross‑Container Caching)

To enable the shared cache:

1.  Create a **private Google Cloud Storage bucket** (for example
    `arc-motivate-cache`).
2.  Keep **Public Access Prevention enabled**.
3.  Grant the **Storage Object Admin** role to the Cloud Run service
    account.

Add the environment variable:

`CACHE_BUCKET_NAME=your-bucket-name`

------------------------------------------------------------------------

# 🚀 Local Setup

## Prerequisites

-   Python **3.10+**
-   A **Google Gemini API key**
-   *(Optional)* A Google Cloud Storage bucket for testing the shared
    cache

## Installation

``` bash
# Clone the repository
git clone https://github.com/pixjobs/arcmotivate.git
cd arcmotivate

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

    GOOGLE_API_KEY="your_gemini_api_key_here"

    # Optional – enables cross‑container cache
    CACHE_BUCKET_NAME="your-gcs-bucket-name"

## Run the application

    python app.py

The interface will be available at:

`http://localhost:7860`

On the very first load the system generates initial assets and messages.
Subsequent loads will be significantly faster thanks to the local `/tmp`
cache and optional Cloud Storage cache.
