# ════════════════════════════════════════════════════════════════════════════
# transformer-chat — Dockerfile
# Multi-stage build: Node.js builder → nginx alpine server
#
# Build:
#   docker build \
#     --build-arg VITE_API_BASE_URL=https://your-server.com \
#     --build-arg VITE_MODEL_NAME="C++ Transformer v1.0" \
#     -t transformer-chat .
#
# Run locally:
#   docker run -p 80:80 \
#     -e CPP_SERVER_HOST=your-server.com \
#     -e CPP_SERVER_PORT=8080 \
#     transformer-chat
#
# ════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Build the React app ─────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies first (better layer caching)
COPY package.json package-lock.json* ./
RUN npm ci --prefer-offline

# Copy source and build
COPY . .

# Accept build-time env vars for VITE_ prefix (they're baked into the JS bundle)
ARG VITE_API_BASE_URL=""
ARG VITE_MAX_GEN_TOKENS=300
ARG VITE_TOKEN_TIMEOUT_MS=8000
ARG VITE_MODEL_NAME="C++ Transformer v1.0"
ARG VITE_APP_TITLE="TransformerChat"

# Expose them to Vite as actual env vars
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
ENV VITE_MAX_GEN_TOKENS=$VITE_MAX_GEN_TOKENS
ENV VITE_TOKEN_TIMEOUT_MS=$VITE_TOKEN_TIMEOUT_MS
ENV VITE_MODEL_NAME=$VITE_MODEL_NAME
ENV VITE_APP_TITLE=$VITE_APP_TITLE

RUN npm run build

# ── Stage 2: Serve with nginx ────────────────────────────────────────────────
FROM nginx:1.27-alpine

# envsubst is included in nginx alpine for template variable substitution
# We use it so CPP_SERVER_HOST/PORT can be set at container start (not build time)
RUN apk add --no-cache gettext

# Copy the built static files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx config template
COPY nginx.conf.template /etc/nginx/templates/default.conf.template

# Default environment variables (override with -e flags at docker run)
ENV CPP_SERVER_HOST=localhost
ENV CPP_SERVER_PORT=8080
ENV NGINX_PORT=80

EXPOSE 80

# nginx docker entrypoint processes /etc/nginx/templates/*.template → /etc/nginx/conf.d/*.conf
# then starts nginx.  This is built into the official nginx image.
CMD ["nginx", "-g", "daemon off;"]
