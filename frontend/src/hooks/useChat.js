/**
 * @file    src/hooks/useChat.js
 * @brief   Core chat state + autoregressive generation loop.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Auto-regressive generation (how streaming is simulated)
 * ════════════════════════════════════════════════════════════════════════════
 *
 * The C++ server returns ONE character per POST /predict request.
 * To produce a full response, this hook runs a loop:
 *
 *   context = format(conversation history) + user_prompt + "\nAssistant: "
 *   generated = ""
 *   while generated < MAX_GEN_TOKENS and no stop sequence:
 *     char = await predictNextToken(context + generated)
 *     generated += char
 *     update assistant bubble in real-time         ← streaming effect!
 *
 * Each character update triggers a React state update so the user sees
 * the response building character by character — exactly like GPT streaming.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * XSS safety
 * ════════════════════════════════════════════════════════════════════════════
 * React's JSX renderer sets textContent (not innerHTML) for string children,
 * so {message.content} in MessageBubble.jsx is XSS-safe by construction.
 * No DOMPurify needed — React's virtual DOM never calls innerHTML on
 * user/model text.
 */

import { useState, useCallback, useRef } from 'react'
import { predictNextToken, ApiError } from '../api/predict.js'
import { MAX_GEN_TOKENS, STOP_SEQUENCES } from '../config.js'

// ── Helpers ───────────────────────────────────────────────────────────────────

let _msgId = 0
const newId = () => ++_msgId

/**
 * Format conversation history into a plain-text prompt for the character-level model.
 * The model was trained on raw text, so we use a simple "User: … Assistant: …" format.
 */
function formatPrompt(messages, pendingUserText) {
  const lines = messages.map((m) =>
    m.role === 'user'
      ? `User: ${m.content}`
      : `Assistant: ${m.content}`,
  )
  if (pendingUserText) lines.push(`User: ${pendingUserText}`)
  lines.push('Assistant:')
  return lines.join('\n')
}

function hasStopSequence(text) {
  return STOP_SEQUENCES.some((seq) => text.endsWith(seq))
}

// ── Hook ──────────────────────────────────────────────────────────────────────

/**
 * @param {{ addToast: Function }} opts
 */
export function useChat({ addToast }) {
  const [messages,     setMessages]     = useState(/** @type {Message[]} */ ([]))
  const [isGenerating, setIsGenerating] = useState(false)
  const abortRef = useRef(/** @type {AbortController|null} */ (null))

  // ── Stop ───────────────────────────────────────────────────────────────────
  const stopGeneration = useCallback(() => {
    abortRef.current?.abort('user_stop')
  }, [])

  // ── Clear ──────────────────────────────────────────────────────────────────
  const clearMessages = useCallback(() => {
    stopGeneration()
    setMessages([])
  }, [stopGeneration])

  // ── Send ───────────────────────────────────────────────────────────────────
  /**
   * @param {string} userText  The user's typed message.
   */
  const sendMessage = useCallback(
    async (userText) => {
      if (!userText.trim() || isGenerating) return

      // ── 1. Append user message ─────────────────────────────────────────────
      const userMsg = {
        id:      newId(),
        role:    'user',
        content: userText.trim(),
        ts:      Date.now(),
      }

      // ── 2. Create placeholder assistant message ────────────────────────────
      const asstId = newId()
      const asstMsg = {
        id:          asstId,
        role:        'assistant',
        content:     '',
        ts:          Date.now(),
        isStreaming: true,
      }

      // Use functional update to get the latest messages snapshot
      // (avoid stale closure over `messages`)
      let historySnapshot = []
      setMessages((prev) => {
        historySnapshot = prev
        return [...prev, userMsg, asstMsg]
      })

      setIsGenerating(true)

      const controller = new AbortController()
      abortRef.current = controller

      try {
        // Build initial prompt from history + new user message
        const basePrompt = formatPrompt(historySnapshot, userText.trim())
        let generated = ''

        for (let i = 0; i < MAX_GEN_TOKENS; i++) {
          if (controller.signal.aborted) break

          const char = await predictNextToken(
            basePrompt + ' ' + generated,
            controller.signal,
          )

          // Empty char = model returned nothing (shouldn't happen, but guard)
          if (!char) break

          generated += char

          // ── Live update the assistant bubble ────────────────────────────────
          // React batches these rapid updates efficiently via concurrent mode.
          setMessages((prev) =>
            prev.map((m) =>
              m.id === asstId ? { ...m, content: generated } : m,
            ),
          )

          // Stop on natural end sequences
          if (hasStopSequence(generated)) break
        }

      } catch (err) {
        if (err instanceof ApiError && err.code !== 'CANCELLED') {
          // Show user-friendly error message via toast
          const friendlyMsg = getFriendlyError(err)
          addToast(friendlyMsg, 'error', 8_000)
        }

        // Remove the empty assistant placeholder if nothing was generated
        setMessages((prev) => {
          const last = prev[prev.length - 1]
          if (last?.id === asstId && last.content === '') {
            return prev.slice(0, -1)
          }
          return prev
        })
      } finally {
        // Mark streaming as done
        setMessages((prev) =>
          prev.map((m) =>
            m.id === asstId ? { ...m, isStreaming: false } : m,
          ),
        )
        setIsGenerating(false)
        abortRef.current = null
      }
    },
    [isGenerating, addToast],
  )

  return { messages, isGenerating, sendMessage, stopGeneration, clearMessages }
}

// ── Error → user-friendly string ──────────────────────────────────────────────

function getFriendlyError(err) {
  switch (err.code) {
    case 'TIMEOUT':
      return '⏱ Inference server is taking too long — it may be waking up. Please try again in a moment.'
    case 'NETWORK':
      return '🔌 Cannot reach the inference server. Check that it is running on port 8080 (or that nginx is routing correctly).'
    case 'SERVER':
      return `🚨 Server error: ${err.message}`
    default:
      return `❌ ${err.message}`
  }
}
