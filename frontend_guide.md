# TransformerChat Frontend — Complete Guide

## Stack Choice: Vite + React + Tailwind CSS

**Why not Vanilla HTML?** Vite gives you:
- One-command deploy to **Vercel / Netlify** (zero config)
- **Dev proxy** that eliminates CORS during development (no C++ server changes needed)
- Hot Module Replacement for fast iteration
- Production bundle ≈ 150KB gzipped (React + your app)

---

## File Tree

```
frontend/
├── .env.example              ← copy to .env.local; change VITE_API_BASE_URL
├── .gitignore
├── .dockerignore
├── Dockerfile                ← multi-stage: node builder → nginx alpine
├── nginx.conf.template       ← envsubst template; /api/* → C++ server
├── index.html                ← Inter + JetBrains Mono fonts, SEO meta
├── package.json
├── vite.config.js            ← dev proxy /api → VITE_API_BASE_URL
├── tailwind.config.js        ← custom design tokens + keyframe animations
├── postcss.config.js
└── src/
    ├── main.jsx
    ├── App.jsx               ← root: multi-conversation state, health probe
    ├── index.css             ← Tailwind + custom scrollbar + animations
    ├── config.js             ← ALL env vars in one place
    ├── api/
    │   └── predict.js        ← predictNextToken, checkServerHealth, ApiError
    ├── hooks/
    │   ├── useChat.js        ← autoregressive generation loop
    │   └── useToast.js       ← toast queue with auto-dismiss
    └── components/
        ├── Sidebar.jsx       ← conversation history, model badge
        ├── ChatWindow.jsx    ← scrollable message list, empty state
        ├── MessageBubble.jsx ← user/assistant bubbles, XSS-safe, copy button
        ├── TypingIndicator.jsx ← bouncing dots while waiting
        ├── ChatInput.jsx     ← auto-grow textarea, Send↔Stop button
        └── Toast.jsx         ← Toast + ToastContainer (error/warning/success/info)
```

---

## Quick Start (Local Dev)

```bash
cd frontend

# 1. Configure environment
cp .env.example .env.local
# Edit .env.local: set VITE_API_BASE_URL=http://localhost:8080

# 2. Install & run
npm install
npm run dev
# → http://localhost:5173
# → /api/predict is proxied to http://localhost:8080/predict (no CORS!)
```

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8080` | C++ server URL (used by Vite proxy / nginx) |
| `VITE_MAX_GEN_TOKENS` | `300` | Max characters to auto-regressively generate |
| `VITE_TOKEN_TIMEOUT_MS` | `8000` | Per-token request timeout in ms |
| `VITE_MODEL_NAME` | `C++ Transformer v1.0` | Displayed in sidebar badge |
| `VITE_APP_TITLE` | `TransformerChat` | Browser title + header |

> [!IMPORTANT]
> `VITE_*` variables are **compiled into the JS bundle** at build time. They are visible in the browser. Do NOT put API keys or secrets here.

---

## How the "Streaming" Works

The C++ server returns **one character** per `/predict` request.  
The frontend simulates streaming by looping:

```
context = formatConversation(history) + "User: {msg}\nAssistant:"
generated = ""

for i in range(MAX_GEN_TOKENS):
    char = POST /api/predict { "prompt": context + generated }
    generated += char
    updateBubble(generated)      ← live React update = streaming effect!
    if generated.endswith("\n\n"): break
```

Each character triggers a `setConversations()` state update, which React batches and renders efficiently via Concurrent Mode.

---

## Error Handling

| Error | Code | Toast Shown |
|---|---|---|
| Server offline | `NETWORK` | 🔌 "Cannot reach inference server…" |
| Request timeout | `TIMEOUT` | ⏱ "Inference server timed out — it may be waking up…" |
| HTTP 4xx/5xx | `SERVER` | 🚨 "Server error 500: …" |
| User pressed Stop | `CANCELLED` | *(silent — expected user action)* |

Server health is probed on app load via `GET /api/health`. Status dot in the top-right shows online (green) / offline (red) / checking (amber pulse).

---

## XSS Safety

React renders `{message.content}` as a **text node** (`textContent`), never `innerHTML`.  
A model output like `<script>alert(1)</script>` is displayed as literal text — it will **never execute**.  
No DOMPurify required.

---

## Production Deployment

### Option A — Vercel / Netlify (Recommended, 1 command)

```bash
# Vercel
npm i -g vercel
vercel --prod
# Set VITE_API_BASE_URL in Vercel dashboard → Environment Variables

# Netlify
npm run build
netlify deploy --prod --dir=dist
```

> [!NOTE]
> For Vercel/Netlify, leave `VITE_API_BASE_URL` **empty** — the frontend will call relative `/api/predict`. You then need to add a **Vercel Rewrites** or **Netlify Redirects** rule to proxy `/api/*` to your C++ server.

**netlify.toml:**
```toml
[[redirects]]
  from = "/api/*"
  to = "http://YOUR_CPP_SERVER:8080/:splat"
  status = 200
  force = true
```

**vercel.json:**
```json
{
  "rewrites": [
    { "source": "/api/(.*)", "destination": "http://YOUR_CPP_SERVER:8080/$1" }
  ]
}
```

---

### Option B — Docker (Self-hosted / AWS / GCP)

```bash
# Build image (bake API URL into bundle if desired)
docker build \
  --build-arg VITE_API_BASE_URL="" \
  -t transformer-chat .

# Run — nginx proxies /api/* → cpp-server:8080 internally
docker run -d \
  -p 80:80 \
  -e CPP_SERVER_HOST=your-cpp-server.internal \
  -e CPP_SERVER_PORT=8080 \
  --name transformer-chat \
  transformer-chat
```

**docker-compose.yml (C++ server + frontend together):**
```yaml
version: '3.9'
services:
  cpp-server:
    image: your-cpp-transformer-image
    ports: ["8080:8080"]

  frontend:
    build: ./frontend
    ports: ["80:80"]
    environment:
      CPP_SERVER_HOST: cpp-server
      CPP_SERVER_PORT: 8080
    depends_on: [cpp-server]
```

With this setup, **no CORS headers are needed on the C++ server** — nginx proxies everything internally between containers.

---

## Design Features

| Feature | Implementation |
|---|---|
| Dark theme | Custom Tailwind colors (`bg-[#07071a]`, sidebar `#0c0c1e`) |
| Streaming effect | Character-by-character `setConversations()` loop |
| Typing dots | CSS `bounce-dot` keyframe + stagger via `.dot-1/2/3` delay classes |
| Blinking cursor | `animate-pulse` inline span while `isStreaming && content !== ''` |
| Copy button | Hover-reveal on assistant bubbles, 2s "Copied!" state |
| Mobile sidebar | Slide-in with backdrop overlay, hamburger toggle |
| Custom scrollbar | Violet-tinted 5px scrollbar via `::webkit-scrollbar` |
| Gradient border | CSS `::before` pseudo-element trick on input field |
| Toast system | Top-right fixed, `slide-in-right` animation, 4 severity levels |
