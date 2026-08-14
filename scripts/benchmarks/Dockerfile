# The benchmark, containerised: the same code and the same entry point
# that runs locally, so a cloud run and a laptop run are the same
# experiment rather than two implementations of one.
FROM python:3.12-slim

# gudhi needs a C++ runtime; matplotlib needs freetype.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /topogym

# Dependencies first, so edits to the library do not re-resolve them.
COPY pyproject.toml README.md ./
COPY topogym/__init__.py topogym/__init__.py
RUN pip install --no-cache-dir ".[benchmarks,testing]" || true

COPY . .
# imageio is the [assets] extra: pods publish coverage GIFs, and a run
# without it warns once per algorithm and ships no animations.
RUN pip install --no-cache-dir --no-deps -e ".[benchmarks]" \
    && pip install --no-cache-dir "ray[rllib]>=2.9" torch rliable \
        "arch>=7.2,<8" "pandas>=2.0,<3" matplotlib "imageio>=2.31"

# The splits are committed, so the container needs no generation step
# and every task sees byte-identical instances.
ENV PYTHONUNBUFFERED=1
# The script is the first argument rather than baked in, so the same
# image runs the benchmark sweep and a single-layout study. Two images
# would be two chances for them to drift apart.
ENTRYPOINT ["python"]
CMD ["scripts/benchmarks/run_baselines_gridworld_v1_benchmark.py", \
     "--baselines", "all", "--only-missing", "--keep-going"]
