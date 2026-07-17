# Force amd64 so the x86-64 moulinette binary runs (emulated on Apple Silicon)
FROM --platform=linux/amd64 python:3.11-slim

# System deps: git for any VCS installs, build-essential for wheels that
# don't ship manylinux binaries. Cleaned up to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv, the project's package manager (the moulinette only runs `uv sync`)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy fails on different filesystems fall back to full copy; silence the warning
ENV UV_LINK_MODE=copy
# Keep the venv inside the image, not on a mount that might be a different arch
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /work

# Dependency layer first, so edits to source don't bust the install cache
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-install-project || uv sync

# Then the rest of the project
COPY . .

# Make the moulinette executable (it ships without +x sometimes)
RUN chmod +x moulinette/moulinette-* 2>/dev/null || true

CMD ["bash"]