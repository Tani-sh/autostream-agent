# Integrating AutoStream Agent with WhatsApp via Webhooks

> **Note:** This is a design and integration guide only. No changes are required to the existing agent codebase — the AutoStream LangGraph agent works as-is; only an HTTP bridge layer needs to be built around it.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Step 1 — Set Up Meta Developer App](#step-1--set-up-meta-developer-app)
5. [Step 2 — Build the FastAPI Webhook Server](#step-2--build-the-fastapi-webhook-server)
6. [Step 3 — Multi-Turn Memory with LangGraph](#step-3--multi-turn-memory-with-langgraph)
7. [Step 4 — Sending Replies Back to WhatsApp](#step-4--sending-replies-back-to-whatsapp)
8. [Step 5 — Deploy the Webhook Server](#step-5--deploy-the-webhook-server)
9. [Step 6 — Register the Webhook with Meta](#step-6--register-the-webhook-with-meta)
10. [Security Considerations](#security-considerations)
11. [Environment Variables Reference](#environment-variables-reference)
12. [End-to-End Message Flow](#end-to-end-message-flow)
13. [Limitations and Production Considerations](#limitations-and-production-considerations)

---

## Overview

The AutoStream conversational AI agent currently runs as a CLI application. To deploy it on WhatsApp, a lightweight **webhook server** acts as a bridge between the **Meta WhatsApp Business API** and the **LangGraph agent**. The agent itself does not need any modification — it accepts a `HumanMessage` and returns an AI response, regardless of whether that input comes from the terminal or WhatsApp.

**What changes:**
- A FastAPI server receives incoming WhatsApp messages via HTTP POST.
- The server extracts the user's phone number (used as the LangGraph `thread_id` to maintain per-user memory).
- The agent response is sent back to the user via the Meta WhatsApp Cloud API.

**What stays the same:**
- `agent/graph.py` — unchanged
- `agent/nodes.py` — unchanged
- `rag/` pipeline — unchanged
- Lead capture logic — unchanged

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        WhatsApp User                             │
│                  (sends a message on WhatsApp)                   │
└───────────────────────────┬──────────────────────────────────────┘
                            │  WhatsApp message
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│              Meta WhatsApp Business API (Cloud)                  │
│  Validates message → fires HTTP POST to your registered          │
│  webhook URL with a JSON payload                                  │
└───────────────────────────┬──────────────────────────────────────┘
                            │  HTTP POST /webhook
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                 FastAPI Webhook Server (your app)                 │
│                                                                  │
│  1. Verify X-Hub-Signature-256 (security)                        │
│  2. Extract phone number + message text from payload             │
│  3. Invoke LangGraph agent (thread_id = phone number)            │
│  4. Extract AI response from result                              │
│  5. POST reply to Meta WhatsApp API → delivered to user          │
└───────────────────────────┬──────────────────────────────────────┘
                            │  Uses existing code unchanged
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│              AutoStream LangGraph Agent (existing)               │
│                                                                  │
│  classify_intent → [retrieve_knowledge] →                        │
│  [collect_lead_info] → [capture_lead] → generate_response        │
└──────────────────────────────────────────────────────────────────┘
```

**Key insight on multi-turn memory:** LangGraph's `MemorySaver` uses a `thread_id` to scope conversation state. On WhatsApp, the user's phone number (e.g., `+919876543210`) is used as the `thread_id`. This means each WhatsApp user automatically gets their own isolated, persistent conversation session — exactly the same as the CLI's `THREAD_ID` variable, but scoped per user.

---

## Prerequisites

| Requirement | Details |
|---|---|
| Meta Developer Account | Free at [developers.facebook.com](https://developers.facebook.com) |
| Meta Business Account | Required to activate WhatsApp Business API |
| WhatsApp Business Phone Number | Provided by Meta (test number available for free) |
| `WHATSAPP_TOKEN` | Permanent System User Access Token from Meta |
| `APP_SECRET` | From your Meta App's "App Secret" field |
| `PHONE_NUMBER_ID` | From the WhatsApp dashboard in Meta Developer Console |
| Python 3.10+ with existing requirements installed | Already done for the CLI setup |
| A public HTTPS URL for the webhook | Use Render, Railway, or ngrok for local testing |

---

## Step 1 — Set Up Meta Developer App

1. Go to [developers.facebook.com](https://developers.facebook.com) → **My Apps** → **Create App**.
2. Select **Business** as the app type.
3. Add the **WhatsApp** product to your app.
4. Under **WhatsApp → Getting Started**, you'll see:
   - A **test phone number** (Meta provides one for free for testing)
   - A **Phone Number ID** — copy this, you'll need it as `PHONE_NUMBER_ID`
5. Generate a **Temporary Access Token** for testing (valid 24 hours) or create a **Permanent System User Token** for production.
6. Note down the **App Secret** from **App Settings → Basic**.

---

## Step 2 — Build the FastAPI Webhook Server

Create a new file `webhook_server.py` in the project root. This file is the **only new code needed**.

```python
"""
webhook_server.py — WhatsApp Webhook Bridge for AutoStream Agent

This server:
  1. Handles the GET verification handshake from Meta.
  2. Handles POST requests with incoming WhatsApp messages.
  3. Invokes the LangGraph agent and sends the reply back.

Run locally:
    uvicorn webhook_server:app --port 8000 --reload

Required env vars:
    GOOGLE_API_KEY      — Gemini API key (same as CLI)
    WHATSAPP_TOKEN      — Meta permanent system user token
    APP_SECRET          — Meta app secret (for signature verification)
    PHONE_NUMBER_ID     — Your WhatsApp Business phone number ID
    VERIFY_TOKEN        — A secret string you invent for webhook verification
"""

import hashlib
import hmac
import json
import logging
import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

from agent.graph import graph  # The existing LangGraph agent — unchanged

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AutoStream WhatsApp Webhook")

WHATSAPP_TOKEN = os.environ["WHATSAPP_TOKEN"]
APP_SECRET = os.environ["APP_SECRET"]
PHONE_NUMBER_ID = os.environ["PHONE_NUMBER_ID"]
VERIFY_TOKEN = os.environ["VERIFY_TOKEN"]

WA_API_URL = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"


# ── Signature Verification ──────────────────────────────────────────────────

def verify_signature(payload: bytes, signature_header: str) -> bool:
    """
    Validate Meta's HMAC-SHA256 webhook signature.
    Prevents spoofed requests from reaching the agent.
    """
    if not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(
        APP_SECRET.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    received = signature_header[len("sha256="):]
    return hmac.compare_digest(expected, received)


# ── Send Reply to WhatsApp ──────────────────────────────────────────────────

async def send_whatsapp_message(to: str, text: str) -> None:
    """POST a text reply back to the WhatsApp user via the Meta Cloud API."""
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(WA_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        logger.info("Reply sent to %s: HTTP %s", to, response.status_code)


# ── Invoke the Agent ────────────────────────────────────────────────────────

def invoke_agent(phone: str, user_text: str) -> str:
    """
    Run the existing LangGraph agent.
    Uses the caller's phone number as thread_id — this gives every WhatsApp
    user their own independent, persistent conversation memory automatically.
    """
    config = {"configurable": {"thread_id": phone}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_text)]},
        config=config,
    )
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    return ai_messages[-1].content if ai_messages else "Sorry, something went wrong."


# ── Webhook Endpoints ───────────────────────────────────────────────────────

@app.get("/webhook")
async def verify_webhook(request: Request):
    """
    Meta sends a GET request to verify ownership of the webhook URL.
    We must echo back the hub.challenge value if our VERIFY_TOKEN matches.
    """
    params = request.query_params
    if (
        params.get("hub.mode") == "subscribe"
        and params.get("hub.verify_token") == VERIFY_TOKEN
    ):
        logger.info("Webhook verified by Meta.")
        return Response(content=params.get("hub.challenge"), media_type="text/plain")
    logger.warning("Webhook verification failed.")
    return Response(status_code=403)


@app.post("/webhook")
async def handle_message(request: Request):
    """
    Receives incoming WhatsApp messages from Meta.
    Verifies the signature, extracts the message, invokes the agent,
    and sends the reply back via the WhatsApp Cloud API.
    """
    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")

    if not verify_signature(body, signature):
        logger.warning("Invalid signature — request rejected.")
        return Response(status_code=401)

    data = json.loads(body)

    # Navigate the Meta webhook payload structure
    try:
        entry = data["entry"][0]["changes"][0]["value"]
        if "messages" not in entry:
            # Could be a status update (read receipts, delivery) — ignore
            return {"status": "ignored"}

        message = entry["messages"][0]
        if message.get("type") != "text":
            # Only handle text messages for now
            return {"status": "non-text ignored"}

        phone = message["from"]       # e.g., "919876543210"
        user_text = message["text"]["body"]

        logger.info("Message from %s: %s", phone, user_text)

        # Invoke the existing LangGraph agent (no changes to agent code)
        agent_reply = invoke_agent(phone=phone, user_text=user_text)

        # Send reply back to the user on WhatsApp
        await send_whatsapp_message(to=phone, text=agent_reply)

    except (KeyError, IndexError) as e:
        logger.error("Unexpected payload structure: %s", e)

    return {"status": "ok"}
```

### Additional dependency to install

```bash
pip install fastapi uvicorn httpx
```

Add these to `requirements.txt` before deploying:
```
fastapi>=0.110.0
uvicorn>=0.29.0
httpx>=0.27.0
```

---

## Step 3 — Multi-Turn Memory with LangGraph

This is **already handled for free** by the existing `MemorySaver` in `agent/graph.py`.

```
CLI:         thread_id = "autostream-demo-001"  (single shared session)
WhatsApp:    thread_id = phone number           (one session per user)
```

When a user with phone `+919876543210` sends their first message:
- `graph.invoke(...)` is called with `thread_id = "919876543210"`.
- LangGraph creates a new checkpoint for that thread and persists the full `AgentState`.

When the same user sends a follow-up message minutes later:
- The same `thread_id` is used.
- LangGraph loads the existing checkpoint — the agent remembers their name, email, platform, and intent from before.
- The conversation continues exactly where it left off.

This means **each WhatsApp user gets isolated, persistent memory** across the full lead-capture flow — automatically, with zero extra code.

> **Important:** `MemorySaver` stores state **in-memory** (RAM). If the server restarts, conversation state is lost. For production, replace it with `SqliteSaver` or `PostgresSaver` from `langgraph.checkpoint.sqlite` / `langgraph-checkpoint-postgres`. The graph compilation line in `agent/graph.py` only needs the checkpointer argument changed — all other code stays the same.

---

## Step 4 — Sending Replies Back to WhatsApp

The `send_whatsapp_message()` function in Step 2 POSTs to the Meta Graph API:

```
POST https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages
Authorization: Bearer {WHATSAPP_TOKEN}
Content-Type: application/json

{
  "messaging_product": "whatsapp",
  "to": "919876543210",
  "type": "text",
  "text": { "body": "Agent reply here..." }
}
```

The response from the agent (which may contain rich text, bullet points, or markdown) should be **stripped of markdown formatting** before sending to WhatsApp, since WhatsApp renders only bold (`*text*`), italic (`_text_`), and strikethrough (`~text~`). A simple regex cleanup function handles this.

---

## Step 5 — Deploy the Webhook Server

The webhook URL must be:
- **Publicly accessible** (not `localhost`)
- **HTTPS** (Meta rejects plain HTTP webhooks)

### Option A — Render (Recommended, Free Tier)

1. Push the project (including `webhook_server.py`) to GitHub.
2. Go to [render.com](https://render.com) → **New Web Service** → connect your repo.
3. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn webhook_server:app --host 0.0.0.0 --port $PORT`
4. Add all environment variables in the Render dashboard (see [Environment Variables Reference](#environment-variables-reference)).
5. Deploy — Render provides a public HTTPS URL like `https://autostream-agent.onrender.com`.

### Option B — Railway

Same process as Render. Railway auto-detects Python and provides HTTPS.

### Option C — Local Testing with ngrok

For quick local testing without deploying:
```bash
# Terminal 1 — start the server
uvicorn webhook_server:app --port 8000 --reload

# Terminal 2 — expose it publicly
ngrok http 8000
```
ngrok gives you a temporary public HTTPS URL (e.g., `https://abc123.ngrok.io`). Use this as your webhook URL in the Meta console during development.

---

## Step 6 — Register the Webhook with Meta

1. Go to **Meta Developer Console** → your app → **WhatsApp → Configuration**.
2. Under **Webhook**, click **Edit**:
   - **Callback URL:** `https://your-deployed-url.com/webhook`
   - **Verify Token:** the value you set as `VERIFY_TOKEN` in your `.env`
3. Click **Verify and Save** — Meta will call `GET /webhook` and check the token.
4. Under **Webhook Fields**, subscribe to the **`messages`** field.
5. Under **WhatsApp → Phone Numbers**, add a real number or use the test number.

That's it. Send a WhatsApp message to your test number — the agent will respond.

---

## Security Considerations

| Concern | Mitigation |
|---|---|
| **Spoofed webhook requests** | `verify_signature()` validates `X-Hub-Signature-256` using HMAC-SHA256 and the App Secret on every POST |
| **Webhook verification spoofing** | `VERIFY_TOKEN` is a secret you set; Meta must match it to pass the GET verification |
| **API key exposure** | All secrets in `.env` — never hardcoded, never committed to git |
| **Rate limiting** | Meta enforces rate limits on the Cloud API; agent invocations are naturally throttled per-user |
| **Replay attacks** | Meta signs requests with the current payload; the HMAC check makes replayed requests with altered bodies invalid |
| **SSRF / injection** | User message text is passed only as a `HumanMessage` string to the agent — no eval, no shell execution |
| **Production memory** | Replace `MemorySaver` with a persistent checkpointer so conversation state survives server restarts |

---

## Environment Variables Reference

Add these to your `.env` file (or deployment platform's secret manager):

```env
# Existing (already in .env.example)
GOOGLE_API_KEY=your_gemini_api_key_here

# New — required for WhatsApp integration
WHATSAPP_TOKEN=your_permanent_system_user_token
APP_SECRET=your_meta_app_secret
PHONE_NUMBER_ID=your_whatsapp_phone_number_id
VERIFY_TOKEN=any_secret_string_you_invent
```

| Variable | Where to find it |
|---|---|
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `WHATSAPP_TOKEN` | Meta Developer Console → WhatsApp → API Setup → Generate Token |
| `APP_SECRET` | Meta Developer Console → App Settings → Basic → App Secret |
| `PHONE_NUMBER_ID` | Meta Developer Console → WhatsApp → API Setup → Phone Number ID |
| `VERIFY_TOKEN` | You invent this — any random string (e.g., `autostream-whatsapp-2024`) |

---

## End-to-End Message Flow

```
User types "Hi, what plans do you have?" on WhatsApp
    │
    ▼
Meta WhatsApp Cloud API detects new message
    │
    ▼
Meta fires: POST https://your-server.com/webhook
  Headers: X-Hub-Signature-256: sha256=<hmac>
  Body: { "entry": [{ "changes": [{ "value": {
           "messages": [{ "from": "919876543210",
                          "text": { "body": "Hi, what plans do you have?" } }]
         }}]}]}
    │
    ▼
FastAPI: verify_signature(body, signature) → ✅ valid
    │
    ▼
Extract: phone = "919876543210", text = "Hi, what plans do you have?"
    │
    ▼
invoke_agent(phone="919876543210", user_text="Hi, what plans do you have?")
  → graph.invoke({messages: [HumanMessage(...)]}, config={thread_id: "919876543210"})
  → classify_intent → "inquiry"
  → retrieve_knowledge (FAISS search on knowledge_base/autostream_kb.md)
  → generate_response → "AutoStream has two plans! Basic at $29/month..."
    │
    ▼
send_whatsapp_message(to="919876543210", text="AutoStream has two plans!...")
  → POST graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages
    │
    ▼
User receives the reply on WhatsApp ✅
```

---

## Limitations and Production Considerations

| Topic | Detail |
|---|---|
| **In-memory state** | `MemorySaver` loses state on restart. Use `SqliteSaver` or `PostgresSaver` for production. |
| **24-hour messaging window** | WhatsApp restricts outbound messages to users who messaged in the last 24h. Within that window, free-form text replies work. After 24h, only pre-approved template messages are allowed. |
| **Media messages** | This implementation handles text only. Audio, image, and document messages from users are ignored. Extend `handle_message` to transcribe audio or extract image context if needed. |
| **Concurrency** | `uvicorn` with `--workers 4` handles concurrent requests. For high volume, use `gunicorn + uvicorn` workers and a Redis-backed checkpointer for thread-safe state access. |
| **Async agent** | The current `graph.invoke()` is synchronous. To avoid blocking the FastAPI event loop under load, wrap it with `asyncio.to_thread(invoke_agent, ...)`. |
| **Rate limits** | Gemini free tier has per-minute token limits. Under high WhatsApp traffic, consider request queuing or upgrading to a paid tier. |
| **Phone number verification** | For production, Meta requires a verified business and a real phone number to go beyond the test sandbox. |
