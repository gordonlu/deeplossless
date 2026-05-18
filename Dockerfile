FROM rust:1.86-slim-bookworm AS builder
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release 2>/dev/null || true
COPY . .
RUN cargo build --release --locked

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/deeplossless /usr/local/bin/deeplossless
EXPOSE 8080
ENV DEEPSEEK_API_KEY=""
ENV ADMIN_KEY=""
ENV SUMMARIZER_MODEL="deepseek-v4-pro"
ENTRYPOINT ["deeplossless"]
