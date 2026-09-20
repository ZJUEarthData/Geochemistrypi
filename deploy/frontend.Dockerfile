FROM node:22-alpine AS builder

ARG GEOCHEMISTRYPI_ONLINE_INSTANCE_ID=geochem-online
ARG GEOCHEMISTRYPI_SOURCE_REVISION=unknown
ARG GEOCHEMISTRYPI_BUILD_ID=unknown
ARG VITE_API_BASE_URL=
ARG VITE_ENABLE_AUTH_UI=false

ENV GEOCHEMISTRYPI_ONLINE_INSTANCE_ID=${GEOCHEMISTRYPI_ONLINE_INSTANCE_ID} \
    GEOCHEMISTRYPI_SOURCE_REVISION=${GEOCHEMISTRYPI_SOURCE_REVISION} \
    GEOCHEMISTRYPI_BUILD_ID=${GEOCHEMISTRYPI_BUILD_ID} \
    VITE_API_BASE_URL=${VITE_API_BASE_URL} \
    VITE_ENABLE_AUTH_UI=${VITE_ENABLE_AUTH_UI} \
    NODE_OPTIONS=--max-old-space-size=512

WORKDIR /app

RUN corepack enable && corepack prepare pnpm@9.15.5 --activate

COPY geochemistrypi/frontend/package.json \
     geochemistrypi/frontend/pnpm-lock.yaml \
     geochemistrypi/frontend/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile

COPY geochemistrypi/frontend/ ./
RUN pnpm run build

FROM nginx:1.27-alpine

LABEL org.opencontainers.image.title="Geochemistry Pi Online Web"

COPY deploy/nginx/http.conf /etc/nginx/conf.d/default.conf
COPY deploy/nginx/proxy-common.inc /etc/nginx/snippets/proxy-common.inc
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["wget", "--quiet", "--spider", "http://127.0.0.1/healthz"]
