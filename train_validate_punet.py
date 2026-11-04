#!/usr/bin/env python3
"""Train PositionalUNet on a file range, denoise the matching validation files, and score them.

The training step follows train.py: segment length 40,000, one tenth of the windows,
FocalLoss1D, and a fresh Adam optimizer on every file. A new network is started at
each frequency-split boundary (0, 4, 10, 15) that falls inside the requested range.

If no file range is given, the four frequency bands are trained and scored separately
(0-3, 4-9, 10-14, 15-19) and the reported fine/coarse scores are the average of the
four band scores.

Examples:
    python train_validate_punet.py training validation
    python train_validate_punet.py training validation 10-15
"""

import argparse
import csv
import gc
import math
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm

from array2h5 import create_abra_file
from network import FocalLoss1D, PositionalUNet

INPUT_SIZE = 40000
SAMPLE_SIZE = 10
MAX_INDEX = 2_000_000_000
CHECKPOINTS = (0, 4, 10, 15, 20)
DEFAULT_BANDS = ((0, 3), (4, 9), (10, 14), (15, 19))
SCORE_BASE = 5.27
N_PER_SEC = 10_000_000
SECONDS_PER_FILE = 200
NOISE_FLOOR = np.finfo(np.float64).tiny
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_range(text):
    try:
        start_text, end_text = text.split("-")
        start, end = int(start_text), int(end_text)
    except ValueError:
        raise argparse.ArgumentTypeError("range must look like 0-19")
    if not (0 <= start <= end <= 19):
        raise argparse.ArgumentTypeError("range must be inside 0-19, with start <= end")
    return start, end


def file_name(kind, index):
    return f"abra_{kind}_{index:04d}.h5"


def log(msg):
    print(msg, flush=True)


def load_pair(path, random_index):
    """Return (sampling_frequency, noisy, clean) windows in 0..255.

    The files store int8 samples. Adding 128 in that dtype overflows, so the
    shift is done after casting the selected windows to int16.
    """
    with h5py.File(path, "r") as handle:
        fs = int(handle["timeseries"]["channel0001"].attrs["sampling_frequency"])
        selected = []
        for name in ("channel0001", "channel0002"):
            raw = np.asarray(handle["timeseries"][name]["timeseries"][:MAX_INDEX])
            usable = len(raw) - (len(raw) % (SAMPLE_SIZE * INPUT_SIZE))
            windows = raw[:usable].reshape(-1, SAMPLE_SIZE, INPUT_SIZE)[:, random_index]
            shifted = windows.astype(np.int16)
            shifted += 128
            selected.append(shifted)
            del raw, windows
    return fs, selected[0], selected[1]


def train(train_dir, indexes, output_dir):
    models = {}
    model = None
    criterion = None
    band_start = None
    sampling_frequency = None
    trained_any = False
    n_params = sum(p.numel() for p in PositionalUNet().parameters())

    for ifile in indexes:
        if model is None or ifile in CHECKPOINTS:
            if model is not None:
                save_model(model, band_start, ifile, output_dir)
            model = PositionalUNet().to(DEVICE)
            criterion = FocalLoss1D().to(DEVICE)
            band_start = ifile
            models[band_start] = model
            log(f"new PUNet for files {band_start} onward")

        path = os.path.join(train_dir, file_name("training", ifile))
        if not os.path.exists(path):
            log(f"missing {os.path.basename(path)}, skipping")
            continue

        random_index = np.random.randint(SAMPLE_SIZE)
        sampling_frequency, noisy, clean = load_pair(path, random_index)
        order = np.random.permutation(len(noisy))
        trained_any = True
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        model.train()
        log(f"training {os.path.basename(path)} on {len(order)} windows")
        for step, row in enumerate(order):
            input_seq = torch.from_numpy(noisy[row]).unsqueeze(0).to(DEVICE).int()
            target_seq = torch.from_numpy(clean[row]).unsqueeze(0).to(DEVICE).long()
            loss = criterion(model(input_seq), target_seq)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 500 == 50:
                log(f"Epoch: {ifile} | Batch: {step} | Loss: {loss.item():.4f}")
        del noisy, clean, optimizer
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if not trained_any:
        raise SystemExit(f"no training files found in {train_dir} for {indexes[0]}-{indexes[-1]}")
    save_model(model, band_start, indexes[-1] + 1, output_dir)
    for net in models.values():
        net.eval()
    return models, n_params, sampling_frequency


def save_model(model, start, end, output_dir):
    path = os.path.join(output_dir, f"PUNet_{start}_{end}.pth")
    torch.save(model, path)
    log(f"saved {path}")


def model_for(models, index):
    starts = [start for start in models if start <= index]
    if not starts:
        raise SystemExit(f"no trained PUNet covers validation file {index}")
    return models[max(starts)]


def denoise(valid_dir, indexes, models, output_dir):
    written = []
    batch = 25
    step = batch * INPUT_SIZE
    with torch.no_grad():
        for ifile in indexes:
            src = os.path.join(valid_dir, file_name("validation", ifile))
            if not os.path.exists(src):
                log(f"missing {os.path.basename(src)}, skipping")
                continue
            dst = os.path.join(output_dir, f"abra_validation_denoised_punet_{ifile:04d}.h5")
            model = model_for(models, ifile)
            log(f"denoising {os.path.basename(src)}")
            denoised = np.empty(MAX_INDEX, dtype=np.int8)
            injected = np.empty(MAX_INDEX, dtype=np.int8)
            with h5py.File(src, "r") as handle:
                noisy_ds = handle["timeseries"]["channel0001"]["timeseries"]
                clean_ds = handle["timeseries"]["channel0002"]["timeseries"]
                length = min(MAX_INDEX, len(noisy_ds), len(clean_ds))
                length -= length % step
                for start in tqdm(range(0, length, step), desc=os.path.basename(src)):
                    noisy = np.asarray(noisy_ds[start:start + step], dtype=np.int16)
                    clean = np.asarray(clean_ds[start:start + step], dtype=np.int16)
                    noisy += 128
                    clean += 128
                    xb = torch.from_numpy(noisy.reshape(batch, INPUT_SIZE)).to(DEVICE).long()
                    out = model(xb).argmax(dim=1).detach().cpu().numpy().reshape(-1)
                    denoised[start:start + step] = np.int8(out - 128)
                    injected[start:start + step] = np.int8(clean - 128)
            create_abra_file(dst, denoised[:length], injected[:length], indexed=False)
            written.append(os.path.basename(dst))
            del denoised, injected
            gc.collect()
    if not written:
        raise SystemExit(f"no validation files found in {valid_dir} for {indexes[0]}-{indexes[-1]}")
    return written


def one_second_psd(path, files, channel, start):
    paths = [os.path.join(path, name) for name in files]
    file_number = start // SECONDS_PER_FILE
    start_index = N_PER_SEC * (start % SECONDS_PER_FILE)
    with h5py.File(paths[file_number], "r") as handle:
        group = handle["timeseries"]["channel0001"]
        data = handle["timeseries"][f"channel000{channel}"]["timeseries"][start_index:start_index + N_PER_SEC]
        # float64 keeps the noise floor. In float32 the peak is so large that
        # peak + floor rounds back to the peak, the floor becomes 0, and SNR is inf.
        scaling = float(group.attrs["voltage_range_mV"]) / (2 * 128.0)
        series = np.asarray(data, dtype=np.float64) * scaling
        dt = 1.0 / float(group.attrs["sampling_frequency"])
    power = dt / N_PER_SEC * (np.abs(np.fft.rfft(series)) ** 2)[1:]
    freqs = np.linspace(0, 5e6, N_PER_SEC // 2)
    return freqs, power


def signal_to_noise(freqs, power, target=0):
    if target == 0:
        contrast = power[1:-1] - power[:-2] - power[2:]
        center = int(np.argmax(contrast)) + 1
    else:
        center = int(np.where(freqs == target)[0][0])
    signal = np.sum(power[center - 1:center + 2])
    noise = np.sum(power[center - 50:center + 51]) - signal
    # Guard residual underflow: a zero noise floor would make SNR infinite and
    # then poison the weighted score with NaN.
    if not np.isfinite(noise) or noise <= 0:
        noise = NOISE_FLOOR
    return float(signal / noise), freqs[center]


def denoising_score(path, files, coarse):
    n = SECONDS_PER_FILE * len(files)
    if coarse:
        n //= 10
    clean_snr = np.zeros(n, dtype=np.float64)
    denoised_snr = np.zeros(n, dtype=np.float64)
    for i in tqdm(range(n), desc="coarse score" if coarse else "fine score"):
        second = i * 10 if coarse else i
        freqs, clean_power = one_second_psd(path, files, 2, second)
        clean_snr[i], center = signal_to_noise(freqs, clean_power)
        _, denoised_power = one_second_psd(path, files, 1, second)
        denoised_snr[i] = signal_to_noise(freqs, denoised_power, center)[0]
    max_clean = np.nanmax(clean_snr)
    if not np.isfinite(max_clean) or max_clean <= 0:
        raise SystemExit("clean SNR is non-positive; cannot compute denoising score")
    weights = clean_snr / max_clean
    raw = np.round(np.nansum(weights * denoised_snr) / len(denoised_snr), decimals=2) + 1e-10
    return math.log(raw, SCORE_BASE)


def write_report(output_dir, segment_size, sampling_frequency, n_params, fine, coarse, band_rows=None):
    report_dir = os.path.join(output_dir, "csv")
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, "scores.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Segment Size", "FS", "Parameters", "Fine Score", "Coarse Score"])
        writer.writerow([segment_size, sampling_frequency, n_params, fine, coarse])
        if band_rows:
            writer.writerow([])
            writer.writerow(["Band", "Fine Score", "Coarse Score"])
            for start, end, band_fine, band_coarse in band_rows:
                writer.writerow([f"{start}-{end}", band_fine, band_coarse])
    log(f"wrote {path}")
    return path


def run_range(train_dir, valid_dir, indexes, output_dir):
    log(f"device {DEVICE} | files {indexes[0]}-{indexes[-1]} | segment {INPUT_SIZE}")
    models, n_params, sampling_frequency = train(train_dir, indexes, output_dir)
    written = denoise(valid_dir, indexes, models, output_dir)
    del models
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    fine = denoising_score(output_dir, written, coarse=False)
    coarse = denoising_score(output_dir, written, coarse=True)
    log(f"files {indexes[0]}-{indexes[-1]} | Fine Score {fine:.6f} | Coarse Score {coarse:.6f}")
    return n_params, sampling_frequency, fine, coarse


def main():
    parser = argparse.ArgumentParser(description="Train, validate, and score PUNet on a file range.")
    parser.add_argument("train_dir", help="Folder of abra_training_00XX.h5 files.")
    parser.add_argument("valid_dir", help="Folder of abra_validation_00XX.h5 files.")
    parser.add_argument(
        "file_range",
        nargs="?",
        type=parse_range,
        default=None,
        help="Inclusive file numbers, for example 10-15. Omit to run bands 0-3, 4-9, 10-14, 15-19 and average scores.",
    )
    parser.add_argument("--output_dir", "-o", default="punet_output", help="Folder for denoised files and the score csv.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.file_range is None:
        fine_scores = []
        coarse_scores = []
        band_rows = []
        n_params = None
        sampling_frequency = None
        for start, end in DEFAULT_BANDS:
            indexes = list(range(start, end + 1))
            log(f"=== band {start}-{end} ===")
            n_params, sampling_frequency, fine, coarse = run_range(
                args.train_dir, args.valid_dir, indexes, args.output_dir
            )
            fine_scores.append(fine)
            coarse_scores.append(coarse)
            band_rows.append((start, end, fine, coarse))
        fine = float(np.mean(fine_scores))
        coarse = float(np.mean(coarse_scores))
        write_report(args.output_dir, INPUT_SIZE, sampling_frequency, n_params, fine, coarse, band_rows)
        log(f"Average Fine Score {fine:.6f} | Average Coarse Score {coarse:.6f}")
        return

    start, end = args.file_range
    indexes = list(range(start, end + 1))
    n_params, sampling_frequency, fine, coarse = run_range(
        args.train_dir, args.valid_dir, indexes, args.output_dir
    )
    write_report(args.output_dir, INPUT_SIZE, sampling_frequency, n_params, fine, coarse)
    log(f"Fine Score {fine:.6f} | Coarse Score {coarse:.6f}")


if __name__ == "__main__":
    main()
