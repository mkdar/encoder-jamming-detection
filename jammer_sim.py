from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


CLASS_NAMES = ["clean", "wideband", "partial_band", "sweep", "reactive"]


@dataclass
class Batch:
    x: np.ndarray
    y: np.ndarray
    snr_db: np.ndarray
    jnr_db: np.ndarray


class OFDMJammingGenerator:
    """
    A physically motivated synthetic OFDM-like data generator for jammer detection.

    Each example is a sequence of OFDM symbols. A token corresponds to one symbol and
    contains per-subcarrier real, imaginary, log-magnitude, and phase features, plus a
    DMRS indicator bit. This mirrors a practical receiver-side feature design where a
    classifier sees structured PHY observations instead of packet-level metadata.
    """

    def __init__(
        self,
        seq_len: int = 14,
        num_subcarriers: int = 72,
        nfft: int = 128,
        dmrs_symbols: Sequence[int] = (2, 11),
        num_taps: int = 4,
        num_classes: int = 5,
    ) -> None:
        self.seq_len = int(seq_len)
        self.num_subcarriers = int(num_subcarriers)
        self.nfft = int(nfft)
        self.dmrs_symbols = np.array(dmrs_symbols, dtype=np.int64)
        self.num_taps = int(num_taps)
        self.num_classes = int(num_classes)

        power_delay = np.exp(-np.arange(self.num_taps, dtype=np.float64))
        self.power_delay = power_delay / power_delay.sum()

        self.k = np.arange(self.num_subcarriers, dtype=np.float64)[None, None, :, None]
        self.l = np.arange(self.num_taps, dtype=np.float64)[None, None, None, :]
        self.s_grid = np.arange(self.seq_len, dtype=np.float64)[None, :, None]
        self.f_grid = np.arange(self.num_subcarriers, dtype=np.float64)[None, None, :]
        self.dmrs_flag = np.isin(np.arange(self.seq_len), self.dmrs_symbols).astype(np.float32)[None, :, None]

    def feature_dim(self) -> int:
        return 4 * self.num_subcarriers + 1

    def _sample_labels(self, batch_size: int, rng: np.random.Generator, force_labels: Optional[np.ndarray]) -> np.ndarray:
        if force_labels is not None:
            labels = np.asarray(force_labels, dtype=np.int64)
            if labels.shape[0] != batch_size:
                raise ValueError("force_labels must have length batch_size")
            return labels
        return rng.integers(0, self.num_classes, size=batch_size, dtype=np.int64)

    def _sample_tx_grid(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        bits = rng.integers(0, 2, size=(batch_size, self.seq_len, self.num_subcarriers, 2), dtype=np.int8)
        qpsk = ((2 * bits[..., 0] - 1) + 1j * (2 * bits[..., 1] - 1)) / math.sqrt(2.0)
        # Put higher-energy known pilots on DMRS symbols to make reactive jamming meaningful.
        qpsk[:, self.dmrs_symbols, :] = math.sqrt(2.0) + 0j
        return qpsk

    def _sample_channel(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        h0 = (rng.normal(size=(batch_size, self.num_taps)) + 1j * rng.normal(size=(batch_size, self.num_taps))) / math.sqrt(2.0)
        h0 = h0 * np.sqrt(self.power_delay)[None, :]

        doppler = rng.uniform(-0.04, 0.04, size=(batch_size, self.num_taps))
        time_phase = np.exp(1j * 2.0 * math.pi * doppler[:, None, :] * self.s_grid)
        taps = h0[:, None, :] * time_phase

        freq_response = np.sum(taps[:, :, None, :] * np.exp(-1j * 2.0 * math.pi * self.k * self.l / self.nfft), axis=-1)
        return freq_response

    def _sample_frontend_phase(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        cfo = rng.normal(0.0, 0.01, size=(batch_size, 1, 1))
        timing = rng.normal(0.0, 0.40, size=(batch_size, 1, 1))
        return np.exp(1j * (2.0 * math.pi * cfo * self.s_grid + 2.0 * math.pi * timing * self.f_grid / self.nfft))

    def _apply_jammer(
        self,
        rx_clean: np.ndarray,
        freq_response: np.ndarray,
        labels: np.ndarray,
        rng: np.random.Generator,
        jnr_db: np.ndarray,
    ) -> np.ndarray:
        batch_size, seq_len, num_subcarriers = rx_clean.shape
        jammer = np.zeros_like(rx_clean)
        signal_power = np.mean(np.abs(rx_clean) ** 2, axis=(1, 2), keepdims=True)
        base_var = signal_power * (10.0 ** (jnr_db / 10.0))

        for cls in (1, 2, 3, 4):
            idx = np.where(labels == cls)[0]
            if len(idx) == 0:
                continue

            noise_like = (
                rng.normal(size=(len(idx), seq_len, num_subcarriers))
                + 1j * rng.normal(size=(len(idx), seq_len, num_subcarriers))
            ) * np.sqrt(base_var[idx] / 2.0)

            if cls == 1:
                # Wideband jammer across almost all symbols.
                active = (rng.uniform(size=(len(idx), seq_len, 1)) < rng.uniform(0.75, 1.0, size=(len(idx), 1, 1))).astype(np.float64)
                jammer[idx] += noise_like * active

            elif cls == 2:
                # Partial-band jammer: one contiguous band, possibly bursty over time.
                mask = np.zeros((len(idx), 1, num_subcarriers), dtype=bool)
                widths = rng.integers(max(8, num_subcarriers // 8), max(9, num_subcarriers // 3), size=len(idx))
                centers = rng.integers(0, num_subcarriers, size=len(idx))
                for i, (center, width) in enumerate(zip(centers, widths)):
                    lo = max(0, center - width // 2)
                    hi = min(num_subcarriers, lo + width)
                    if hi - lo < width and lo > 0:
                        lo = max(0, hi - width)
                    mask[i, 0, lo:hi] = True
                active = (rng.uniform(size=(len(idx), seq_len, 1)) < rng.uniform(0.65, 1.0, size=(len(idx), 1, 1))).astype(np.float64)
                jammer[idx] += 1.10 * noise_like * mask * active

            elif cls == 3:
                # Sweep jammer: a narrow band that walks across frequency over time.
                mask = np.zeros((len(idx), seq_len, num_subcarriers), dtype=bool)
                widths = rng.integers(4, max(5, num_subcarriers // 10), size=len(idx))
                starts = rng.integers(0, num_subcarriers, size=len(idx))
                speeds = rng.choice([-3, -2, -1, 1, 2, 3], size=len(idx))
                for i, (start, speed, width) in enumerate(zip(starts, speeds, widths)):
                    for s in range(seq_len):
                        center = (start + speed * s) % num_subcarriers
                        lo = max(0, center - width // 2)
                        hi = min(num_subcarriers, lo + width)
                        if hi - lo < width and lo > 0:
                            lo = max(0, hi - width)
                        mask[i, s, lo:hi] = True
                jammer[idx] += 1.20 * noise_like * mask

            else:
                # Reactive jammer: strongly attacks DMRS symbols and high-gain regions.
                mask = np.zeros((len(idx), 1, num_subcarriers), dtype=bool)
                widths = rng.integers(4, max(5, num_subcarriers // 8), size=len(idx))
                centers = np.argmax(np.mean(np.abs(freq_response[idx]), axis=1), axis=1)
                for i, (center, width) in enumerate(zip(centers, widths)):
                    lo = max(0, center - width // 2)
                    hi = min(num_subcarriers, lo + width)
                    if hi - lo < width and lo > 0:
                        lo = max(0, hi - width)
                    mask[i, 0, lo:hi] = True
                active = np.zeros((len(idx), seq_len, 1), dtype=np.float64)
                active[:, self.dmrs_symbols, :] = 1.0
                active += (rng.uniform(size=(len(idx), seq_len, 1)) < 0.15).astype(np.float64)
                active = np.clip(active, 0.0, 1.0)
                jammer[idx] += 1.60 * noise_like * mask * active
                weak_fullband = (
                    rng.normal(size=(len(idx), seq_len, num_subcarriers))
                    + 1j * rng.normal(size=(len(idx), seq_len, num_subcarriers))
                ) * np.sqrt((0.25 * base_var[idx]) / 2.0)
                jammer[idx] += weak_fullband * active

        return jammer

    def generate(
        self,
        batch_size: int,
        rng: np.random.Generator,
        snr_db: Optional[float] = None,
        force_labels: Optional[np.ndarray] = None,
    ) -> Batch:
        batch_size = int(batch_size)
        labels = self._sample_labels(batch_size, rng, force_labels)
        tx_grid = self._sample_tx_grid(batch_size, rng)
        freq_response = self._sample_channel(batch_size, rng)
        frontend_phase = self._sample_frontend_phase(batch_size, rng)

        path_loss_db = rng.uniform(-5.0, 5.0, size=(batch_size, 1, 1))
        path_gain = 10.0 ** (path_loss_db / 20.0)
        rx_clean = path_gain * tx_grid * freq_response * frontend_phase

        jnr_db_arr = rng.uniform(-2.0, 18.0, size=(batch_size, 1, 1))
        jammer = self._apply_jammer(rx_clean, freq_response, labels, rng, jnr_db_arr)

        signal_power = np.mean(np.abs(rx_clean) ** 2, axis=(1, 2), keepdims=True)
        if snr_db is None:
            snr_arr = rng.uniform(0.0, 24.0, size=(batch_size, 1, 1))
        else:
            snr_arr = np.full((batch_size, 1, 1), float(snr_db), dtype=np.float64)
        noise_var = signal_power / (10.0 ** (snr_arr / 10.0))
        noise = (
            rng.normal(size=rx_clean.shape) + 1j * rng.normal(size=rx_clean.shape)
        ) * np.sqrt(noise_var / 2.0)

        rx = rx_clean + jammer + noise
        magnitude = np.abs(rx).astype(np.float32)
        features = np.concatenate(
            [
                rx.real.astype(np.float32),
                rx.imag.astype(np.float32),
                np.log1p(magnitude).astype(np.float32),
                (np.angle(rx) / math.pi).astype(np.float32),
                np.broadcast_to(self.dmrs_flag, (batch_size, self.seq_len, 1)),
            ],
            axis=-1,
        )

        return Batch(
            x=features,
            y=labels.astype(np.int64),
            snr_db=snr_arr.reshape(batch_size).astype(np.float32),
            jnr_db=jnr_db_arr.reshape(batch_size).astype(np.float32),
        )

    def make_fixed_set(
        self,
        num_examples: int,
        seed: int,
        snr_db: Optional[float] = None,
        balanced: bool = True,
    ) -> Batch:
        rng = np.random.default_rng(seed)
        labels = None
        if balanced:
            labels = np.arange(num_examples, dtype=np.int64) % self.num_classes
            rng.shuffle(labels)
        return self.generate(num_examples, rng, snr_db=snr_db, force_labels=labels)
