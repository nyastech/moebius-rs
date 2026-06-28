FROM rust:1.96.0-bookworm AS builder

RUN cargo install trunk
RUN rustup target add wasm32-unknown-unknown

WORKDIR /app

COPY . .

RUN trunk build --release

FROM caddy:2

COPY Caddyfile /etc/caddy/Caddyfile
COPY --from=builder /app/dist /srv

EXPOSE 8080
