# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
# Registry digests were verified on 2026-10-08; upgrades require qualification.
FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:0.11.14@sha256:58ca64e5c85c084cb1f4b30c6bf51a1120305eb1bf3d059f5b853c963ab97f66 AS uv
FROM --platform=linux/amd64 rust:1.97.1-slim-bookworm@sha256:39f68a3e8e3ff425f8945ffa91128e60ff930d53e17fbb5214e95824bdd46f1b AS rust
FROM --platform=linux/amd64 python:3.14.7-slim-bookworm@sha256:998acd06f485adfd6890e3e15a4b542543e0cf22ff904310095da328a5e3e561 AS dependencies

ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/g-venv \
    PATH=/opt/g-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /build
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group workbench --no-install-project

FROM dependencies AS native-build
ARG NATIVE_BUILD_JOBS=2
ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    RUSTUP_TOOLCHAIN=1.97.1 \
    PYO3_PYTHON=/opt/g-venv/bin/python \
    CARGO_BUILD_JOBS=${NATIVE_BUILD_JOBS} \
    RUSTFLAGS="--allow=non-local-definitions -Ctarget-cpu=x86-64" \
    CFLAGS="-O3 -march=x86-64 -mtune=generic" \
    CXXFLAGS="-O3 -march=x86-64 -mtune=generic" \
    PATH=/opt/g-venv/bin:/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup
# Pin apt repository state; native kernels need neither mold nor a host CUDA toolkit.
RUN printf '%s\n' \
      'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20261001T000000Z bookworm main' \
      > /etc/apt/sources.list \
    && rm -f /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install --no-install-recommends -y build-essential pkg-config patchelf \
    && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY src ./src
COPY crates ./crates
COPY native ./native
COPY vendor/openxla ./vendor/openxla
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group workbench --group workbench-build --no-install-project
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/build/target \
    maturin build --release --locked --strip --out /wheels --interpreter /opt/g-venv/bin/python
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group workbench --no-install-project \
    && uv pip install --python /opt/g-venv/bin/python --no-deps /wheels/g-*.whl \
    && mkdir -p /opt/g \
    && sha256sum pyproject.toml uv.lock Cargo.toml Cargo.lock > /opt/g/build-locks.sha256

FROM --platform=linux/amd64 python:3.14.7-slim-bookworm@sha256:998acd06f485adfd6890e3e15a4b542543e0cf22ff904310095da328a5e3e561 AS runtime
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.title="GWAS Engine Workbench runner" \
      org.opencontainers.image.source="https://github.com/kirilledition/g" \
      org.opencontainers.image.revision=${SOURCE_REVISION}
ENV PATH=/opt/g-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    GWAS_ENGINE_SOURCE_REVISION=${SOURCE_REVISION} \
    PYTHONPATH=/opt/g \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAX_ENABLE_X64=true \
    XLA_PYTHON_CLIENT_PREALLOCATE=false
RUN printf '%s\n' \
      'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20261001T000000Z bookworm main' \
      > /etc/apt/sources.list \
    && rm -f /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 researcher \
    && useradd --uid 1000 --gid 1000 --create-home --no-log-init researcher \
    && mkdir /work \
    && chown researcher:researcher /work
COPY --from=native-build /opt/g-venv /opt/g-venv
COPY --from=native-build /opt/g/build-locks.sha256 /opt/g/build-locks.sha256
COPY tooling /opt/g/tooling
COPY deploy/workbench /opt/g/deploy/workbench
COPY --chmod=0755 deploy/workbench/entrypoint.sh /usr/local/bin/g-workbench
COPY README.md /opt/g/README.md
USER 1000:1000
WORKDIR /work
ENTRYPOINT ["g-workbench"]
CMD ["--help"]
