#!/usr/bin/env bash
set -euo pipefail

RUST_VERSION="1.96.0"
IMAGE_NAME="${IMAGE_NAME:-moebius-rs}"
DOCKER_TAG="${DOCKER_TAG:-0.0.1}"
WASM_SIMD="${WASM_SIMD:-0}"
WASM_OPT="${WASM_OPT:-0}"

format() {
    cargo +nightly fmt;
}

setup_rust() {
    echo "[INFO] Checking Rust installation..."
    if command -v rustc >/dev/null 2>&1; then
        current_version="$(rustc --version | awk '{print $2}')"
        echo "[INFO] Found Rust ${current_version}"
    else
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "${RUST_VERSION}"
    fi

    export PATH="$HOME/.cargo/bin:$PATH"
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env" 2>/dev/null || true
    rustup target add wasm32-unknown-unknown

    if ! command -v trunk >/dev/null 2>&1; then
        cargo install trunk
    fi
}

check() {
    cargo test
    cargo +nightly fmt -- --check
    cargo clippy -- -D warnings
    build_release_wasm
    copy_model_assets
}

build() {
    build_release_wasm
    copy_model_assets
}

run() {
    local port="${PORT:-8080}"

    build_release_wasm
    copy_model_assets
    echo "[INFO] Serving http://127.0.0.1:${port}"
    python3 -m http.server "${port}" --bind 127.0.0.1 --directory dist
}

build_release_wasm() {
    if [[ "${WASM_SIMD}" == "1" ]]; then
        export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128"
        echo "[INFO] Building release WASM with simd128 enabled"
    else
        echo "[INFO] Building release WASM"
    fi

    trunk build --release

    if [[ "${WASM_OPT}" == "1" ]]; then
        optimize_wasm
    fi
}

optimize_wasm() {
    if ! command -v wasm-opt >/dev/null 2>&1; then
        echo "[ERROR] WASM_OPT=1 requires wasm-opt from Binaryen" >&2
        return 1
    fi

    while IFS= read -r wasm_file; do
        local output_file="${wasm_file}.opt"
        echo "[INFO] Optimizing ${wasm_file}"
        wasm-opt -O3 "${wasm_file}" -o "${output_file}"
        mv "${output_file}" "${wasm_file}"
    done < <(find dist -type f -name '*.wasm')
}

copy_model_assets() {
    local source_dir="public/models/moebius-ft-places2"
    local target_dir="dist/models"

    if [[ -d "${source_dir}" ]]; then
        mkdir -p "${target_dir}"
        cp -R "${source_dir}" "${target_dir}/"
    fi
}

build_docker() {
    docker build --tag "${IMAGE_NAME}:${DOCKER_TAG}" --tag "${IMAGE_NAME}:latest" --file Dockerfile .
}

help() {
    echo "Usage: $0 [setup|format|check|build|run|build_docker|help]"
    echo "Environment: WASM_SIMD=1 enables simd128; WASM_OPT=1 runs wasm-opt -O3"
}

main() {
    case "${1:-help}" in
        setup) setup_rust ;;
        format) format ;;
        check) check ;;
        build) build ;;
        run) run ;;
        build_docker) build_docker ;;
        help|*) help ;;
    esac
}

main "$@"
