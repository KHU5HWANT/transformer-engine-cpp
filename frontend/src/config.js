/**
 * @file    src/config.js
 * @brief   Single source of truth for all environment-driven configuration.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * How to change environments
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  Local dev:
 *    1. cp .env.example .env.local
 *    2. Set VITE_API_BASE_URL=http://localhost:8080
 *    3. npm run dev   ← Vite proxy forwards /api/* with no CORS
 *
 *  Production (Docker/nginx):
 *    1. Set VITE_API_BASE_URL=https://your-server.com in .env.production
 *       OR leave blank — nginx proxies /predict internally (see nginx.conf)
 *    2. npm run build && docker build -t transformer-chat .
 *
 *  Vercel / Netlify:
 *    Set VITE_API_BASE_URL in the platform dashboard under Environment Variables.
 *    The build pipeline reads it at compile time via import.meta.env.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Security note
 * ════════════════════════════════════════════════════════════════════════════
 *  VITE_ variables are inlined into the JS bundle at build time.
 *  Never store API keys or secrets here.
 */

// ── API endpoint ──────────────────────────────────────────────────────────────
//
// In development:  Vite rewrites /api/* → VITE_API_BASE_URL/* (see vite.config.js).
//                  The browser always hits the same origin — no CORS.
//
// In production:   nginx rewrites /api/* → http://cpp-server:8080/* internally.
//                  Again, no CORS because requests are same-origin.
//
// The API_BASE exported here is therefore always '/api' — a relative path.
// You only need VITE_API_BASE_URL for the proxy/nginx target, not for fetch() calls.
//
export const API_BASE = '/api'

// ── Generation ────────────────────────────────────────────────────────────────
export const MAX_GEN_TOKENS   = Number(import.meta.env.VITE_MAX_GEN_TOKENS   ?? 300)
export const TOKEN_TIMEOUT_MS = Number(import.meta.env.VITE_TOKEN_TIMEOUT_MS ?? 8_000)

// ── UI ────────────────────────────────────────────────────────────────────────
export const MODEL_NAME = import.meta.env.VITE_MODEL_NAME ?? 'C++ Transformer v1.0'
export const APP_TITLE  = import.meta.env.VITE_APP_TITLE  ?? 'TransformerChat'

// ── Stop sequences ────────────────────────────────────────────────────────────
// Auto-regressive loop terminates when the generated text ends with one of these.
export const STOP_SEQUENCES = ['\n\n', '</s>']
