/**
 * @file    src/hooks/useToast.js
 * @brief   Lightweight toast notification system.
 *
 * Usage:
 *   const { toasts, addToast, removeToast } = useToast()
 *   addToast('Something went wrong', 'error')
 *   addToast('Copied!', 'success', 2000)
 */

import { useState, useCallback, useRef } from 'react'

let _nextId = 0

/**
 * @typedef {'error'|'warning'|'success'|'info'} ToastType
 * @typedef {{ id: number, message: string, type: ToastType }} Toast
 */

export function useToast() {
  const [toasts, setToasts] = useState(/** @type {Toast[]} */ ([]))
  const timers = useRef(new Map())

  const removeToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
    clearTimeout(timers.current.get(id))
    timers.current.delete(id)
  }, [])

  /**
   * @param {string}    message
   * @param {ToastType} type      default 'info'
   * @param {number}    duration  ms before auto-dismiss, default 5000
   */
  const addToast = useCallback(
    (message, type = 'info', duration = 5_000) => {
      const id = ++_nextId
      setToasts((prev) => [...prev, { id, message, type }])
      if (duration > 0) {
        timers.current.set(id, setTimeout(() => removeToast(id), duration))
      }
      return id
    },
    [removeToast],
  )

  return { toasts, addToast, removeToast }
}
