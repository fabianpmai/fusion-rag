FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

WORKDIR /app

# deps first, cached independently of source changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY fusionrag/ fusionrag/
COPY data/ data/
COPY app.py smoke.py ./
COPY pages/ pages/
RUN uv sync --frozen

# bake the embedding model into the image so first use needs no download
RUN uv run python -c "from fusionrag.embedder import download; download()"

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
