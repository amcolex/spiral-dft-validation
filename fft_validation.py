import os
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import cocotb
    from cocotb.clock import Clock
    from cocotb.triggers import ReadOnly, RisingEdge
except ImportError:  # pragma: no cover - only used when running tests in sim
    cocotb = None
    Clock = ReadOnly = RisingEdge = None

from cocotb_test.simulator import run as cocotb_run


TRANSFORM_LENGTH = 2048
SAMPLES_PER_CYCLE = 2  # two complex words per clock
DATA_WIDTH = 16
MAX_LSB_ERROR = 6

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
BUILD_DIR = PROJECT_ROOT / "build"
STIMULI = ("impulse", "constant", "tone", "square", "multitone", "ofdm")
OFDM_NUM_CARRIERS = 1200


def _twos_complement(value: int, width: int) -> int:
    mask = (1 << width) - 1
    value &= mask
    if value & (1 << (width - 1)):
        value -= 1 << width
    return value


def _ofdm_active_indices() -> np.ndarray:
    half = OFDM_NUM_CARRIERS // 2
    positive = np.arange(1, half + 1, dtype=np.int32)
    negative = np.arange(TRANSFORM_LENGTH - half, TRANSFORM_LENGTH, dtype=np.int32)
    return np.concatenate((negative, positive))


def _ofdm_qpsk_symbols() -> np.ndarray:
    rng = np.random.default_rng(2025)
    mapping = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex128) / np.sqrt(2)
    return mapping[rng.integers(0, len(mapping), size=OFDM_NUM_CARRIERS)]


def _generate_vector(stimulus: str) -> np.ndarray:
    vector = np.zeros((TRANSFORM_LENGTH, 2), dtype=np.int16)
    if stimulus == "impulse":
        vector[0, 0] = np.iinfo(np.int16).max
    elif stimulus == "constant":
        vector[:, 0] = 1
    elif stimulus == "tone":
        amp = np.iinfo(np.int16).max // 2
        tone_bin = 16
        n = np.arange(TRANSFORM_LENGTH)
        vector[:, 0] = np.round(amp * np.cos(2 * np.pi * tone_bin * n / TRANSFORM_LENGTH)).astype(np.int16)
        vector[:, 1] = np.round(amp * np.sin(2 * np.pi * tone_bin * n / TRANSFORM_LENGTH)).astype(np.int16)
    elif stimulus == "square":
        amp = np.iinfo(np.int16).max
        period = 128
        high_samples = period // 2
        for start in range(0, TRANSFORM_LENGTH, period):
            vector[start : start + high_samples, 0] = amp
    elif stimulus == "multitone":
        amp = np.iinfo(np.int16).max // 3
        n = np.arange(TRANSFORM_LENGTH)
        real = (
            0.6 * np.cos(2 * np.pi * 7 * n / TRANSFORM_LENGTH)
            + 0.4 * np.sin(2 * np.pi * 31 * n / TRANSFORM_LENGTH)
            + 0.3 * np.cos(2 * np.pi * 103 * n / TRANSFORM_LENGTH + np.pi / 4)
        )
        imag = (
            0.5 * np.sin(2 * np.pi * 13 * n / TRANSFORM_LENGTH)
            + 0.35 * np.cos(2 * np.pi * 59 * n / TRANSFORM_LENGTH)
            + 0.25 * np.sin(2 * np.pi * 211 * n / TRANSFORM_LENGTH + np.pi / 6)
        )
        vector[:, 0] = np.clip(np.round(amp * real), np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)
        vector[:, 1] = np.clip(np.round(amp * imag), np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)
    elif stimulus == "ofdm":
        indices = _ofdm_active_indices()
        symbols = _ofdm_qpsk_symbols()
        freq_bins = np.zeros(TRANSFORM_LENGTH, dtype=np.complex128)
        freq_bins[indices] = symbols
        time_domain = np.fft.ifft(freq_bins)
        max_mag = float(np.max(np.abs(time_domain)))
        scale = (0.85 * np.iinfo(np.int16).max) / max_mag if max_mag > 0 else 1.0
        time_domain *= scale
        vector[:, 0] = np.round(time_domain.real).astype(np.int16)
        vector[:, 1] = np.round(time_domain.imag).astype(np.int16)
    else:
        raise RuntimeError(f"Unsupported stimulus '{stimulus}'")
    return vector


def _reference_fft(inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    complex_inputs = inputs[:, 0].astype(np.int64) + 1j * inputs[:, 1].astype(np.int64)
    fft = np.fft.fft(complex_inputs) / TRANSFORM_LENGTH
    clip_min, clip_max = np.iinfo(np.int16).min, np.iinfo(np.int16).max
    fft_real = np.clip(np.floor(np.real(fft)), clip_min, clip_max).astype(np.int16)
    fft_imag = np.clip(np.floor(np.imag(fft)), clip_min, clip_max).astype(np.int16)
    return fft_real, fft_imag


def _check_against_reference(stimulus: str, inputs: np.ndarray, outputs: np.ndarray) -> None:
    exp_real, exp_imag = _reference_fft(inputs)
    act_real = outputs[:, 0].astype(np.int32)
    act_imag = outputs[:, 1].astype(np.int32)
    err_real = np.abs(act_real - exp_real.astype(np.int32))
    err_imag = np.abs(act_imag - exp_imag.astype(np.int32))
    max_err = int(max(err_real.max(initial=0), err_imag.max(initial=0)))
    if max_err > MAX_LSB_ERROR:
        worst_idx = int(np.argmax(np.maximum(err_real, err_imag)))
        raise AssertionError(
            f"{stimulus}: FFT mismatch exceeds {MAX_LSB_ERROR} LSB "
            f"(idx {worst_idx}, expected {exp_real[worst_idx]}+j{exp_imag[worst_idx]}, "
            f"observed {act_real[worst_idx]}+j{act_imag[worst_idx]}, error {max_err} LSB)"
        )


def _save_results(stimulus: str, inputs: np.ndarray, outputs: np.ndarray) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        RESULTS_DIR / f"{stimulus}.npz",
        input_real=inputs[:, 0],
        input_imag=inputs[:, 1],
        output_real=outputs[:, 0],
        output_imag=outputs[:, 1],
    )


def _load_results(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].astype(np.int16) for key in data.files}


def _compute_reference(case: str, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    inputs = np.column_stack((data["input_real"], data["input_imag"]))
    ref_real, ref_imag = _reference_fft(inputs)
    reference = {"ref_real": ref_real, "ref_imag": ref_imag}
    if case == "ofdm":
        reference["constellation_indices"] = _ofdm_active_indices()
    return reference


def _plot_case(case: str, data: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> Path:
    indices = np.arange(data["input_real"].size)
    num_rows = 4 if case == "ofdm" else 3
    fig, axes = plt.subplots(num_rows, 1, sharex=False, figsize=(10, 9 if case == "ofdm" else 8))
    axes = np.atleast_1d(axes)

    axes[0].plot(indices, data["input_real"], label="real")
    axes[0].plot(indices, data["input_imag"], label="imag", linestyle="--")
    axes[0].set_title(f"{case.capitalize()} input (raw 16-bit)")
    axes[0].set_ylabel("Value")
    axes[0].legend()

    axes[1].plot(indices, data["output_real"], label="real (hw)")
    axes[1].plot(indices, data["output_imag"], label="imag (hw)", linestyle="--")
    axes[1].plot(indices, reference["ref_real"], label="real (ref)", alpha=0.7)
    axes[1].plot(indices, reference["ref_imag"], label="imag (ref)", linestyle="--", alpha=0.7)
    axes[1].set_title("FFT output vs reference")
    axes[1].set_ylabel("Value")
    axes[1].legend()

    err_real = data["output_real"].astype(np.int32) - reference["ref_real"].astype(np.int32)
    err_imag = data["output_imag"].astype(np.int32) - reference["ref_imag"].astype(np.int32)
    axes[2].plot(indices, err_real, label="real error")
    axes[2].plot(indices, err_imag, label="imag error", linestyle="--")
    axes[2].set_title("LSB error (hw - ref)")
    axes[2].set_ylabel("LSB")
    axes[2].set_xlabel("Sample index")
    axes[2].legend()

    if case == "ofdm":
        const_ax = axes[3]
        freq_hw = (data["output_real"].astype(np.int32) + 1j * data["output_imag"].astype(np.int32)) * TRANSFORM_LENGTH
        freq_ref = (reference["ref_real"].astype(np.int32) + 1j * reference["ref_imag"].astype(np.int32)) * TRANSFORM_LENGTH
        active = reference.get("constellation_indices", _ofdm_active_indices())
        const_ax.scatter(freq_ref[active].real, freq_ref[active].imag, label="ref", s=12, alpha=0.6)
        const_ax.scatter(freq_hw[active].real, freq_hw[active].imag, label="hw", s=12, marker="x")
        const_ax.set_title("Constellation (active carriers)")
        const_ax.set_xlabel("In-phase")
        const_ax.set_ylabel("Quadrature")
        const_ax.grid(True, alpha=0.3)
        const_ax.legend()

    fig.tight_layout()
    image_path = RESULTS_DIR / f"{case}.png"
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    return image_path


def _reset_dut_signals(dut) -> None:
    dut.reset.value = 0
    dut.next.value = 0
    dut.X0.value = 0
    dut.X1.value = 0
    dut.X2.value = 0
    dut.X3.value = 0


async def _reset_dut(dut):
    _reset_dut_signals(dut)
    await RisingEdge(dut.clk)
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


async def _drive_transform(dut, vector: np.ndarray):
    assert vector.shape == (TRANSFORM_LENGTH, 2)
    total_cycles = TRANSFORM_LENGTH // SAMPLES_PER_CYCLE

    dut.next.value = 1
    await RisingEdge(dut.clk)
    dut.next.value = 0

    for cycle in range(total_cycles):
        sample0 = vector[cycle * 2]
        sample1 = vector[cycle * 2 + 1]

        dut.X0.value = int(sample0[0]) & 0xFFFF
        dut.X1.value = int(sample0[1]) & 0xFFFF
        dut.X2.value = int(sample1[0]) & 0xFFFF
        dut.X3.value = int(sample1[1]) & 0xFFFF

        await RisingEdge(dut.clk)

    _reset_dut_signals(dut)


async def _collect_outputs(dut) -> np.ndarray:
    total_cycles = TRANSFORM_LENGTH // SAMPLES_PER_CYCLE
    outputs = np.zeros((TRANSFORM_LENGTH, 2), dtype=np.int16)

    await RisingEdge(dut.next_out)
    for cycle in range(total_cycles):
        await RisingEdge(dut.clk)
        await ReadOnly()

        base = cycle * 2
        outputs[base, 0] = _twos_complement(int(dut.Y0.value.integer), DATA_WIDTH)
        outputs[base, 1] = _twos_complement(int(dut.Y1.value.integer), DATA_WIDTH)
        outputs[base + 1, 0] = _twos_complement(int(dut.Y2.value.integer), DATA_WIDTH)
        outputs[base + 1, 1] = _twos_complement(int(dut.Y3.value.integer), DATA_WIDTH)

    return outputs


def run_simulation(cases: Sequence[str] = STIMULI) -> list[Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for case in cases:
        for suffix in ("npz", "png"):
            path = RESULTS_DIR / f"{case}.{suffix}"
            if path.exists():
                path.unlink()

    cocotb_run(
        simulator="verilator",
        toplevel="spiral_dft_it_2048_16bit_scaled",
        module=Path(__file__).stem,
        verilog_sources=[str(PROJECT_ROOT / "spiral_dft_it_2048_16bit_scaled.v")],
        python_search=[str(PROJECT_ROOT)],
        sim_build=str(BUILD_DIR / "verilator"),
        extra_env={
            "STIMULI": ",".join(cases),
            "RESULTS_DIR": str(RESULTS_DIR),
        },
        verilog_compile_args=["-Wno-WIDTHEXPAND"],
        timescale="1ns/1ps",
    )

    return [RESULTS_DIR / f"{case}.npz" for case in cases]


def main() -> None:
    result_files = run_simulation(STIMULI)
    for case, result in zip(STIMULI, result_files):
        data = _load_results(result)
        reference = _compute_reference(case, data)
        _plot_case(case, data, reference)


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    main()


if cocotb:

    @cocotb.test()
    async def run_basic_stimulus(dut):
        clock = Clock(dut.clk, 10, units="ns")
        cocotb.start_soon(clock.start())

        stimuli_env = os.getenv("STIMULI")
        if stimuli_env:
            stimuli = [token.strip().lower() for token in stimuli_env.split(",") if token.strip()]
        else:
            stimuli = [os.getenv("STIMULUS", "impulse").lower()]

        for index, stimulus in enumerate(stimuli, start=1):
            dut._log.info("Running stimulus %d/%d: %s", index, len(stimuli), stimulus)

            if index == 1:
                await _reset_dut(dut)
            else:
                await RisingEdge(dut.clk)
                await _reset_dut(dut)

            inputs = _generate_vector(stimulus)
            collector = cocotb.start_soon(_collect_outputs(dut))
            await _drive_transform(dut, inputs)
            outputs = await collector

            _save_results(stimulus, inputs, outputs)
            _check_against_reference(stimulus, inputs, outputs)

            dut._log.info("Captured %d complex outputs", outputs.shape[0])
