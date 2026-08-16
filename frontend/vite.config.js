import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Load .env.local / .env.production so we can read VITE_API_BASE_URL
  // even inside the config (needed for the dev proxy target).
  const env = loadEnv(mode, process.cwd(), '')
  const apiTarget = env.VITE_API_BASE_URL || 'http://localhost:8080'

  return {
    plugins: [react()],

    server: {
      port: 5173,
      // ── Dev proxy ─────────────────────────────────────────────────────────
      // The browser sends POST /api/predict → Vite rewrites to POST /predict
      // on the C++ server.  This avoids CORS entirely during development.
      // In production, nginx handles the same rewrite (see nginx.conf).
      proxy: {
        '/api': {
          target: apiTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
          configure: (proxy) => {
            proxy.on('error', (err) => {
              console.error('[vite-proxy] C++ server unreachable:', err.message)
            })
          },
        },
      },
    },

    build: {
      outDir: 'dist',
      sourcemap: false,
      // Split vendor chunk so the main bundle stays small
      rollupOptions: {
        output: {
          manualChunks: { vendor: ['react', 'react-dom'] },
        },
      },
    },
  }
})
