"""Manual NumPy/MLX microbenchmarks for the simulator hot paths.

Run on Apple Silicon from the repository root:

    .venv/bin/python tests/benchmark/bench_mlx.py

MLX is evaluated after every chunk.  This mirrors a streaming pipeline and avoids
timing a growing collection of lazy graphs or including their retained memory.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.simbiophys import (
    CosineEncoderSettings,
    CosineEncoderTransformer,
    DynamicColoredNoiseSettings,
    DynamicColoredNoiseTransformer,
    LineNoiseSettings,
    LineNoiseTransformer,
)

N_CHUNKS = 200


def _message(data, fs: float) -> AxisArray:
    dims = ["time"] if data.ndim == 1 else ["time", "ch"]
    return AxisArray(data, dims=dims, axes={"time": AxisArray.TimeAxis(fs=fs, offset=0.0)})


def _time_chunks(
    transformer: Callable[[AxisArray], AxisArray],
    chunks: list[AxisArray],
    evaluate: Callable[[object], None] | None = None,
) -> tuple[float, AxisArray]:
    # Warm up compilation/JIT and state initialization outside the timed region.
    warm = transformer(chunks[0])
    if evaluate is not None:
        evaluate(warm.data)

    start = time.perf_counter()
    for chunk in chunks[1:]:
        output = transformer(chunk)
        if evaluate is not None:
            evaluate(output.data)
    elapsed = time.perf_counter() - start
    return elapsed / (len(chunks) - 1), output


def _report(name: str, numpy_seconds: float, mlx_seconds: float) -> None:
    print(
        f"{name:36s} NumPy {numpy_seconds * 1e6:9.1f} us/chunk | "
        f"MLX {mlx_seconds * 1e6:9.1f} us/chunk | "
        f"speedup {numpy_seconds / mlx_seconds:5.2f}x"
    )


def benchmark_cosine(mx) -> None:
    rng = np.random.default_rng(1)
    polar = np.column_stack(
        (
            np.abs(rng.standard_normal(500)),
            rng.uniform(-np.pi, np.pi, 500),
        )
    ).astype(np.float32)
    numpy_message = _message(polar, 100.0)
    mlx_message = _message(mx.array(polar), 100.0)
    mx.eval(mlx_message.data)
    numpy_chunks = [numpy_message] * N_CHUNKS
    mlx_chunks = [mlx_message] * N_CHUNKS

    settings = CosineEncoderSettings(output_ch=256, baseline=10.0, modulation=20.0, seed=42)
    numpy_seconds, numpy_out = _time_chunks(CosineEncoderTransformer(settings), numpy_chunks)
    mlx_seconds, mlx_out = _time_chunks(CosineEncoderTransformer(settings), mlx_chunks, mx.eval)
    np.testing.assert_allclose(np.asarray(mlx_out.data), numpy_out.data, rtol=5e-3, atol=1e-4)
    _report("Cosine encoder (500 x 2 -> 256)", numpy_seconds, mlx_seconds)


def benchmark_dynamic_colored_noise(mx) -> None:
    beta = np.linspace(0.5, 1.95, 50, dtype=np.float32)[:, np.newaxis]
    beta = np.broadcast_to(beta, (50, 8)).copy()
    numpy_message = _message(beta, 100.0)
    mlx_message = _message(mx.array(beta), 100.0)
    mx.eval(mlx_message.data)
    numpy_chunks = [numpy_message] * N_CHUNKS
    mlx_chunks = [mlx_message] * N_CHUNKS

    settings = DynamicColoredNoiseSettings(output_fs=30000.0, n_poles=5, seed=42)
    numpy_seconds, numpy_out = _time_chunks(DynamicColoredNoiseTransformer(settings), numpy_chunks)
    mlx_seconds, mlx_out = _time_chunks(DynamicColoredNoiseTransformer(settings), mlx_chunks, mx.eval)
    np.testing.assert_allclose(np.asarray(mlx_out.data), numpy_out.data, rtol=1e-4, atol=1e-5)
    _report("Colored noise (50 x 8 -> 15000)", numpy_seconds, mlx_seconds)


def benchmark_line_noise(mx) -> None:
    rng = np.random.default_rng(2)
    signal = rng.standard_normal((15000, 256), dtype=np.float32)
    numpy_message = _message(signal, 30000.0)
    mlx_message = _message(mx.array(signal), 30000.0)
    mx.eval(mlx_message.data)
    numpy_chunks = [numpy_message] * N_CHUNKS
    mlx_chunks = [mlx_message] * N_CHUNKS

    settings = LineNoiseSettings(freq=60.0, amp=10.0, drift_rate=0.002, seed=42)
    numpy_seconds, numpy_out = _time_chunks(LineNoiseTransformer(settings), numpy_chunks)
    mlx_seconds, mlx_out = _time_chunks(LineNoiseTransformer(settings), mlx_chunks, mx.eval)
    np.testing.assert_allclose(np.asarray(mlx_out.data), numpy_out.data, rtol=1e-5, atol=2e-6)
    _report("Line noise (15000 x 256)", numpy_seconds, mlx_seconds)


def main() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("These benchmarks require MLX on Apple Silicon")

    import mlx.core as mx

    benchmark_cosine(mx)
    benchmark_dynamic_colored_noise(mx)
    benchmark_line_noise(mx)


if __name__ == "__main__":
    main()
