/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      colors: {
        // Design token palette — change here to retheme the whole app
        bg: {
          primary:  '#07071a',
          sidebar:  '#0c0c1e',
          surface:  '#13132a',
          elevated: '#1a1a35',
        },
        border: 'rgba(255,255,255,0.07)',
        accent: {
          DEFAULT: '#7c3aed',
          light:   '#a78bfa',
          dim:     'rgba(124,58,237,0.15)',
        },
      },
      keyframes: {
        // Typing dot animation
        'bounce-dot': {
          '0%, 80%, 100%': { transform: 'translateY(0)', opacity: '0.4' },
          '40%':           { transform: 'translateY(-6px)', opacity: '1' },
        },
        // Message entry
        'slide-up': {
          from: { opacity: '0', transform: 'translateY(12px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
        // Toast entry
        'slide-in-right': {
          from: { opacity: '0', transform: 'translateX(100%)' },
          to:   { opacity: '1', transform: 'translateX(0)' },
        },
        // Spinner
        'spin-slow': {
          to: { transform: 'rotate(360deg)' },
        },
      },
      animation: {
        'bounce-dot':      'bounce-dot 1.2s ease-in-out infinite',
        'slide-up':        'slide-up 0.25s ease-out forwards',
        'slide-in-right':  'slide-in-right 0.3s ease-out forwards',
        'spin-slow':       'spin-slow 1s linear infinite',
      },
    },
  },
  plugins: [],
}
