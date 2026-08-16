/**
 * @file src/components/Toast.jsx
 * Individual dismissible toast notification.
 */

const STYLES = {
  error:   'border-red-500/40   bg-red-950/60   text-red-200',
  warning: 'border-amber-500/40 bg-amber-950/60 text-amber-200',
  success: 'border-emerald-500/40 bg-emerald-950/60 text-emerald-200',
  info:    'border-violet-500/40 bg-violet-950/60 text-violet-200',
}

const ICONS = {
  error:   '✕',
  warning: '⚠',
  success: '✓',
  info:    'ℹ',
}

export function Toast({ toast, onRemove }) {
  return (
    <div
      role="alert"
      className={`
        flex items-start gap-3 w-80 max-w-[calc(100vw-2rem)]
        px-4 py-3 rounded-xl border backdrop-blur-md
        shadow-2xl shadow-black/50
        animate-[slide-in-right_0.3s_ease-out_forwards]
        ${STYLES[toast.type] ?? STYLES.info}
      `}
    >
      {/* Icon */}
      <span className="flex-shrink-0 mt-0.5 font-bold text-sm w-4 text-center">
        {ICONS[toast.type]}
      </span>

      {/* Message */}
      <p className="flex-1 text-xs leading-relaxed">{toast.message}</p>

      {/* Dismiss */}
      <button
        onClick={() => onRemove(toast.id)}
        aria-label="Dismiss notification"
        className="flex-shrink-0 opacity-50 hover:opacity-100 transition-opacity mt-0.5 text-xs"
      >
        ✕
      </button>
    </div>
  )
}

/**
 * @file src/components/ToastContainer.jsx
 * Fixed top-right container for all active toasts.
 */
export function ToastContainer({ toasts, onRemove }) {
  if (toasts.length === 0) return null

  return (
    <div
      aria-live="polite"
      aria-atomic="false"
      className="fixed top-4 right-4 z-50 flex flex-col gap-2 pointer-events-none"
    >
      {toasts.map((t) => (
        <div key={t.id} className="pointer-events-auto">
          <Toast toast={t} onRemove={onRemove} />
        </div>
      ))}
    </div>
  )
}
