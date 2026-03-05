# ArcMotivate: System Architecture & User Flow

The following architecture diagram illustrates the end-to-end user journey within **ArcMotivate**. It demonstrates how user inputs are processed through psychological frameworks, disrupted by gamified resilience challenges, and ultimately transformed into a personalized, magical career exploration story.

mermaid
flowchart TD
    %% Styling
    classDef frontend fill:#f9f0ff,stroke:#d0bdf4,stroke-width:2px,color:#333
    classDef backend fill:#e0f7fa,stroke:#4dd0e1,stroke-width:2px,color:#333
    classDef psychology fill:#fff3e0,stroke:#ffb74d,stroke-width:2px,color:#333
    classDef gamification fill:#ffebee,stroke:#e57373,stroke-width:2px,color:#333
    classDef generation fill:#e8f5e9,stroke:#81c784,stroke-width:2px,color:#333
    classDef api fill:#eceff1,stroke:#90a4ae,stroke-width:2px,color:#333

    %% Nodes & Subgraphs
    subgraph Frontend [Magical Portal]
        User([Kid/Teen Explorer])
        UI[Streamlit UI]
    end

    subgraph Core Backend [Intelligence Layer]
        CA{Coaching Agent}
        DB[(User Profile & Quest DB)]
    end

    subgraph Psychological Framework [The Psychology Codex]
        PC[(Psychology Codex)]
        Savickas[Savickas Career Construction]
        SDT[Self-Determination Theory]
    end

    subgraph Resilience Engine [Gamification]
        PTE{Plot Twist Engine}
    end

    subgraph Content Generation [Story Forge]
        SG[Storybook Generator]
    end

    subgraph External APIs [Magic Providers]
        LLM[LLM API <br/>OpenAI/Anthropic]
        IMG[Image API <br/>DALL-E/Midjourney]
    end

    %% Connections
    User -->|Explores & Makes Choices| UI
    UI -->|Displays Story & Next Steps| User

    UI -->|Sends Interests & Actions| CA
    
    CA <-->|Reads/Writes State| DB
    CA -->|Queries Frameworks| PC
    PC --- Savickas
    PC --- SDT
    
    CA -->|Evaluates Progress & Triggers Event| PTE
    
    PTE -->|Interrupts Flow: Injects Challenge| SG
    CA -->|Sends User Profile & Coaching Goals| SG
    
    SG <-->|Generates Narrative Text| LLM
    SG <-->|Generates Chapter Illustrations| IMG
    
    SG -->|Returns Compiled Story Chapter| UI

    %% Apply Styles
    class User,UI frontend
    class CA,DB backend
    class PC,Savickas,SDT psychology
    class PTE gamification
    class SG generation
    class LLM,IMG api


### Architecture Explanation

The ArcMotivate platform is designed as a dynamic, state-driven narrative loop. Here is how the components interact to create a magical and educational experience:

1. **Streamlit UI (The Magical Portal):** 
   The entry point for our 8-18 year-old explorers. It provides a highly visual, interactive interface where users make choices about their interests (e.g., "Do you want to build a robot or heal a magical creature?").

2. **Coaching Agent & The Psychology Codex:** 
   When the user makes a choice, the data flows to the **Coaching Agent**. This agent acts as the "Dungeon Master" of the career journey. It continuously queries the **Psychology Codex**, which houses two primary frameworks:
   * **Savickas Career Construction Theory:** Helps the agent guide the user in making meaning of their choices, turning isolated interests into a cohesive life theme.
   * **Self-Determination Theory (SDT):** Ensures the agent's prompts foster *Autonomy* (giving the user control), *Competence* (making challenges achievable), and *Relatedness* (connecting their skills to helping others in the story).

3. **Plot Twist Engine (Resilience Builder):** 
   To prevent the exploration from becoming a linear, predictable quiz, the **Plot Twist Engine** periodically interrupts the flow. Based on the user's current "Quest," it injects a contextual challenge (e.g., *The bridge to the coding kingdom has collapsed!*). This teaches adaptability and resilience, requiring the user to pivot or use a secondary interest to solve the problem.

4. **Storybook Generator & External APIs:** 
   The Coaching Agent (with its psychological insights) and the Plot Twist Engine (with its injected conflict) send their combined payload to the **Storybook Generator**. This service orchestrates calls to external APIs:
   * **LLM API:** Drafts the next chapter of the user's personalized story, ensuring the tone is age-appropriate, encouraging, and magical.
   * **Image API:** Generates a custom illustration of the user's current scenario.
   
   Finally, the compiled chapter—complete with narrative, imagery, and the next set of choices—is routed back to the Streamlit UI, continuing the loop.