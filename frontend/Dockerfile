# React frontend, built with Vite and served by nginx.
# nginx also proxies /api and /health to the API container, so the browser
# talks to a single origin and no CORS configuration is needed.

FROM node:24-alpine AS build

WORKDIR /build

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:1.30-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /build/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD ["wget", "-q", "--spider", "http://127.0.0.1/"]
