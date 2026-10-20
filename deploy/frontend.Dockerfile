FROM node:22-alpine AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
ARG FASTAPI_BASE_URL=http://api:8000
ARG AGENT_PROXY_TIMEOUT_MS=630000
ENV FASTAPI_BASE_URL=$FASTAPI_BASE_URL AGENT_PROXY_TIMEOUT_MS=$AGENT_PROXY_TIMEOUT_MS NEXT_TELEMETRY_DISABLED=1
RUN npm run build
FROM node:22-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 HOSTNAME=0.0.0.0 PORT=3000
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static
USER node
EXPOSE 3000
CMD ["node", "server.js"]
