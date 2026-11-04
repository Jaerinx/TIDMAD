#!/usr/bin/env python3
"""Train FCNet (the AE in network.py) on a file range, denoise the matching validation files, and score them.

The training step follows train.py for fcnet: segment length 40,000, one tenth of the windows,
SmoothL1Loss on float inputs/targets, and a fresh Adam optimizer on every file. A new network is
started at each frequency-split boundary (0, 4, 10, 15) that falls inside the requested range.

FCNet differs from PUNet in that it is a regression model: it outputs one float per sample in
the 0..255 ADC range instead of 256 class logits, so inference rounds and clips the output rather
than taking an argmax. Each FCNet has ~320M parameters (~1.3 GB), so finished networks are parked
on the CPU and moved back to the device only while they denoise their validation files.

Example:
    python train_validate_fcnet.py "D:\\Hieu\\DSC Capstone\\training" "D:\\Hieu\\DSC Capstone\\validation" 0-19
"""

import argparse
import csv
import gc
import math
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from array2h5 import create_abra_file
from network import AE

INPUT_SIZE = 40000
SAMPLE_SIZE = 10
MAX_INDEX = 2_000_000_000
CHECKPOINTS = (0, 4, 10, 15, 20)
SCORE_BASE = 5.27
N_PER_SEC = 10_000_000
SECONDS_PER_FILE = 200
INFER_BATCH = 25
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


def free_device_memory():
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def load_pair(path, random_index):
    """Return (sampling_frequency, noisy, clean) windows in 0..255 as float32.

    The files store int8 samples. Adding 128 in that dtype overflows, so the
    shift is done after casting the selected windows to a wider dtype.
    """
    with h5py.File(path, "r") as handle:
        fs = int(handle["timeseries"]["channel0001"].attrs["sampling_frequency"])
        selected = []
        for name in ("channel0001", "channel0002"):
            raw = np.asarray(handle["timeseries"][name]["timeseries"][:MAX_INDEX])
            usable = len(raw) - (len(raw) % (SAMPLE_SIZE * INPUT_SIZE))
            windows = raw[:usable].reshape(-1, SAMPLE_SIZE, INPUT_SIZE)[:, random_index]
            shifted = windows.astype(np.float32)
            shifted += 128
            selected.append(shifted)
            del raw, windows
    return fs, selected[0], selected[1]


def save_model(model, start, end, output_dir):
    path = os.path.join(output_dir, f"FCNet_{start}_{end}.pth")
    torch.save(model, path)
    log(f"saved {path}")


def park(model):
    """Move a finished network off the GPU so the next band has room to train."""
    model.eval()
    model.cpu()
    free_device_memory()


def train(train_dir, indexes, output_dir):
    models = {}
    model = None
    criterion = None
    band_start = None
    sampling_frequency = None
    trained_any = False
    n_params = None

    for ifile in indexes:
        if model is None or ifile in CHECKPOINTS:
            if model is not None:
                save_model(model, band_start, ifile, output_dir)
                park(model)
            model = AE(INPUT_SIZE).to(DEVICE)
            criterion = nn.SmoothL1Loss().to(DEVICE)
            if n_params is None:
                n_params = sum(p.numel() for p in model.parameters())
            band_start = ifile
            models[band_start] = model
            log(f"new FCNet for files {band_start} onward ({n_params:,} parameters)")

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
            input_seq = torch.from_numpy(noisy[row]).unsqueeze(0).to(DEVICE)
            target_seq = torch.from_numpy(clean[row]).unsqueeze(0).to(DEVICE)
            loss = criterion(model(input_seq), target_seq)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 500 == 50:
                log(f"Epoch: {ifile} | Batch: {step} | Loss: {loss.item():.4f}")
        del noisy, clean, optimizer
        free_device_memory()

    if not trained_any:
        raise SystemExit(f"no training files found in {train_dir} for {indexes[0]}-{indexes[-1]}")
    save_model(model, band_start, indexes[-1] + 1, output_dir)
    park(model)
    return models, n_params, sampling_frequency


def model_start_for(models, index):
    starts = [start for start in models if start <= index]
    if not starts:
        raise SystemExit(f"no trained FCNet covers validation file {index}")
    return max(starts)


def to_adc_int8(values):
    """Map FCNet's float output (nominally 0..255) back to the stored int8 range."""
    return (np.clip(np.rint(values), 0, 255) - 128).astype(np.int8)


def denoise(valid_dir, indexes, models, output_dir):
    written = []
    step = INFER_BATCH * INPUT_SIZE
    active_start = None
    model = None
    with torch.no_grad():
        for ifile in indexes:
            src = os.path.join(valid_dir, file_name("validation", ifile))
            if not os.path.exists(src):
                log(f"missing {os.path.basename(src)}, skipping")
                continue
            dst = os.path.join(output_dir, f"abra_validation_denoised_fcnet_{ifile:04d}.h5")

            start_key = model_start_for(models, ifile)
            if start_key != active_start:
                if model is not None:
                    park(model)
                model = models[start_key].to(DEVICE).eval()
                active_start = start_key

            log(f"denoising {os.path.basename(src)} with FCNet_{start_key}")
            denoised = np.empty(MAX_INDEX, dtype=np.int8)
            injected = np.empty(MAX_INDEX, dtype=np.int8)
            with h5py.File(src, "r") as handle:
                noisy_ds = handle["timeseries"]["channel0001"]["timeseries"]
                clean_ds = handle["timeseries"]["channel0002"]["timeseries"]
                length = min(MAX_INDEX, len(noisy_ds), len(clean_ds))
                length -= length % step
                for start in tqdm(range(0, length, step), desc=os.path.basename(src)):
                    noisy = np.asarray(noisy_ds[start:start + step], dtype=np.float32)
                    noisy += 128
                    xb = torch.from_numpy(noisy.reshape(INFER_BATCH, INPUT_SIZE)).to(DEVICE)
                    out = model(xb).detach().cpu().numpy().reshape(-1)
                    denoised[start:start + step] = to_adc_int8(out)
                    injected[start:start + step] = clean_ds[start:start + step]
            create_abra_file(dst, denoised[:length], injected[:length], indexed=False)
            written.append(os.path.basename(dst))
            del denoised, injected
            gc.collect()
    if model is not None:
        park(model)
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
        scaling = np.float32(group.attrs["voltage_range_mV"] / (2 * 128.0))
        series = np.asarray(data, dtype=np.float32) * scaling
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
    return signal / noise, freqs[center]


def denoising_score(path, files, coarse):
    n = SECONDS_PER_FILE * len(files)
    if coarse:
        n //= 10
    clean_snr = np.zeros(n)
    denoised_snr = np.zeros(n)
    for i in tqdm(range(n), desc="coarse score" if coarse else "fine score"):
        second = i * 10 if coarse else i
        freqs, clean_power = one_second_psd(path, files, 2, second)
        clean_snr[i], center = signal_to_noise(freqs, clean_power)
        _, denoised_power = one_second_psd(path, files, 1, second)
        denoised_snr[i] = signal_to_noise(freqs, denoised_power, center)[0]
    weights = clean_snr / np.max(clean_snr)
    raw = np.round(np.sum(weights * denoised_snr) / len(denoised_snr), decimals=2) + 1e-10
    return math.log(raw, SCORE_BASE)


def write_report(output_dir, segment_size, sampling_frequency, n_params, fine, coarse):
    report_dir = os.path.join(output_dir, "csv")
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, "scores.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Segment Size", "FS", "Parameters", "Fine Score", "Coarse Score"])
        writer.writerow([segment_size, sampling_frequency, n_params, fine, coarse])
    log(f"wrote {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Train, validate, and score FCNet on a file range.")
    parser.add_argument("train_dir", help="Folder of abra_training_00XX.h5 files.")
    parser.add_argument("valid_dir", help="Folder of abra_validation_00XX.h5 files.")
    parser.add_argument("file_range", type=parse_range, help="Inclusive file numbers, for example 0-19.")
    parser.add_argument("--output_dir", "-o", default="fcnet_output", help="Folder for models, denoised files, and the score csv.")
    args = parser.parse_args()

    start, end = args.file_range
    indexes = list(range(start, end + 1))
    os.makedirs(args.output_dir, exist_ok=True)
    log(f"device {DEVICE} | files {start}-{end} | segment {INPUT_SIZE}")

    models, n_params, sampling_frequency = train(args.train_dir, indexes, args.output_dir)
    written = denoise(args.valid_dir, indexes, models, args.output_dir)
    del models
    free_device_memory()

    fine = denoising_score(args.output_dir, written, coarse=False)
    coarse = denoising_score(args.output_dir, written, coarse=True)
    write_report(args.output_dir, INPUT_SIZE, sampling_frequency, n_params, fine, coarse)
    log(f"Fine Score {fine:.6f} | Coarse Score {coarse:.6f}")


if __name__ == "__main__":
    main()
