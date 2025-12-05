## Overview and highlights

This is the backend repository for restaurant voice agent.

Tech Stack:
Frontend + Routes: NextJS
Backend: Livekit, Langchain, Python
DB: MongoDB Atlas cluster
TTS model: deepgram aura-asteria-en
STT model: deepgram flux-general-en
LLM model: Cerebras llama-3.3-70b
Weather-api and seat preference: Openweathermap, Groq llama-3.3-70b-versatile LLM

Architecture Diagram:

![System Design](/assets/system-design.png)

### Highlights
- Supports natural language processing for date and time input (uses dateparser)
- Uses openweathermap api and langchain llm output to suggest seat preference
- Supports multilingual turn detection and VAD using silero

## Requirements

Build an intelligent voice agent that helps users book restaurant tables through natural
conversation. The agent should be able to:

1. Greet the user and understand their booking intent
2. Collect booking information through voice:
    - Number of guests
    - Preferred date and time
    - Cuisine preference (Italian, Chinese, Indian, etc.)
    - Special requests (birthday, anniversary, dietary restrictions)
3. Fetch real-time weather for the booking date and suggest indoor/outdoor seating
4. Confirm booking details via voice
5. Store booking in a database

## Dev Setup

1. Clone the repository and install dependencies to a virtual environment:

```console
cd restaurant-booking-voice-agent
source setup.sh
```

2. Run Livekit server locally and setup env variables

```console
curl -sSL https://get.livekit.io | bash
livekit-server --dev
cp .env.example .env
```

3. Change api keys in the .env file

## Run the agent

Before your first run, you must download certain models such as [Silero VAD](https://docs.livekit.io/agents/build/turns/vad/) and the [LiveKit turn detector](https://docs.livekit.io/agents/build/turns/turn-detector/):

```console
uv run python src/agent.py download-files
```

Recommended:
To run the agent for use with a frontend or telephony, use the `dev` command:

```console
uv run python src/agent.py dev
```

Alternative:
To run this command to speak to your agent directly in your terminal:

```console
uv run python src/agent.py console
```

## Frontend & Telephony

Checkout the frontend for this project in the [Link](https://github.com/MSVelan/restaurant-booking-voice-agent-frontend)

## Tests and evals

This project includes a complete suite of evals, based on the LiveKit Agents [testing & evaluation framework](https://docs.livekit.io/agents/build/testing/). To run them, use `pytest`. Not implemented for all functions yet.

```console
uv run pytest
```

## Deploying to production

This project is production-ready and includes a working `Dockerfile`. To deploy it to LiveKit Cloud or another environment, see the [deploying to production](https://docs.livekit.io/agents/ops/deployment/) guide.

## Self-hosted LiveKit

You can also self-host LiveKit instead of using LiveKit Cloud. See the [self-hosting](https://docs.livekit.io/home/self-hosting/) guide for more information. If you choose to self-host, you'll need to also use [model plugins](https://docs.livekit.io/agents/models/#plugins) instead of LiveKit Inference and will need to remove the [LiveKit Cloud noise cancellation](https://docs.livekit.io/home/cloud/noise-cancellation/) plugin.
