/**
 * @file    src/api/predict.js
 * @brief   API layer for the C++ inference server.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Design
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  predictNextToken(prompt, signal)
 *    ↓ POST /api/predict  {"prompt": "..."}
 *    ↓ C++ server → {"completion": "X"}   (ONE character)
 *    ↓ returns "X"
 *
 *  Callers (useChat.js) loop predictNextToken to auto-regressively build
 *  the full response, updating the UI after every character.  This gives a
 *  real-time streaming appearance even though the protocol is request/response.
 *
 *  Error taxonomy:
 *    TIMEOUT    — per-token request exceeded TOKEN_TIMEOUT_MS
 *    NETWORK    — server unreachable (ECONNREFUSED / CORS / DNS failure)
 *    CANCELLED  — AbortController signal from user pressing Stop
 *    SERVER     — HTTP 4xx / 5xx from C++ server
 */

import { API_BASE, TOKEN_TIMEOUT_MS } from '../config.js'

// ── Custom error class ────────────────────────────────────────────────────────

export class ApiError extends Error {
  /**
   * @param {string} message  Human-readable message shown in the Toast.
   * @param {'TIMEOUT'|'NETWORK'|'CANCELLED'|'SERVER'} code
   */
  constructor(message, code) {
    super(message)
    this.name = 'ApiError'
    this.code = code
  }
}

// ── Core function ─────────────────────────────────────────────────────────────

/**
 * @brief Call POST /api/predict and return the single predicted character.
 *
 * @param {string}       prompt  Full context string (conversation so far).
 * @param {AbortSignal}  signal  Optional — caller's cancellation signal.
 * @returns {Promise<string>}   One character from the model.
 * @throws {ApiError}
 */
export async function predictNextToken(prompt, signal, temperature = 0.8) {
  // Internal abort controller for per-token timeout.
  // We merge it with the caller's signal via a shared listener.
  const controller = new AbortController()

  const timeoutId = setTimeout(
    () => controller.abort(new DOMException('timeout', 'AbortError')),
    TOKEN_TIMEOUT_MS,
  )

  // Forward caller cancellation → our controller
  const forwardAbort = () => controller.abort(signal?.reason instanceof Error ? signal.reason : new DOMException('cancelled', 'AbortError'))
  signal?.addEventListener('abort', forwardAbort, { once: true })

  try {
    let res
    try {
      res = await fetch(`${API_BASE}/predict`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ prompt, temperature }),
        signal:  controller.signal,
      })
    } catch (err) {
      // Distinguish timeout vs user-cancel vs network error
      if (err.name === 'AbortError') {
        const reason = controller.signal.reason
        if (reason === 'timeout' || reason?.message === 'timeout') {
          throw new ApiError(
            'Inference server timed out — it may be overloaded or asleep.',
            'TIMEOUT',
          )
        }
        throw new ApiError('Generation cancelled.', 'CANCELLED')
      }
      // TypeError from fetch = network failure (ECONNREFUSED / CORS / DNS)
      throw new ApiError(
        'Cannot reach the inference server.  ' +
        'Make sure it is running and CORS headers are set ' +
        '(or use the nginx reverse-proxy — see nginx.conf).',
        'NETWORK',
      )
    }

    if (!res.ok) {
      let detail = ''
      try { detail = (await res.json()).error ?? '' } catch (_) { /* ignore */ }
      throw new ApiError(
        `Server error ${res.status}: ${detail || res.statusText}`,
        'SERVER',
      )
    }

    const data = await res.json()
    // Defensive: C++ server returns {"completion": "X"}
    return typeof data.completion === 'string' ? data.completion : ''

  } finally {
    clearTimeout(timeoutId)
    signal?.removeEventListener('abort', forwardAbort)
  }
}

// ── Health-check ──────────────────────────────────────────────────────────────

/**
 * @brief Probe GET /api/health to see if the C++ server is reachable.
 * @returns {Promise<'online'|'offline'>}
 */
export async function checkServerHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(3_000),
    })
    return res.ok ? 'online' : 'offline'
  } catch (_) {
    return 'offline'
  }
}
