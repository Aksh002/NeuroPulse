FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.4 /uv /uvx /bin/
WORKDIR /workspace
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY apps/__init__.py ./apps/__init__.py
COPY apps/backend ./apps/backend
COPY configs ./configs
RUN uv sync --locked --no-dev --extra research --extra backend
ENV PATH="/workspace/.venv/bin:${PATH}" \
    PYTHONPATH=/workspace \
    PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "apps.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
