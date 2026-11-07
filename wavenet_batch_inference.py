#Runs the trained model on multiple validation files and saves denoised outputs.

import h5py
import numpy as np
import torch
from network import SimpleWaveNet

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    "wavenet_progress.pth",
    map_location="cpu",
    weights_only=True
)

model = SimpleWaveNet().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Using checkpoint through file {checkpoint['next_file'] - 1:04d}", flush=True)

for file_id in range(4, 20):
    source = f"/home/bscruz/teams/d02/validation/abra_validation_{file_id:04d}.h5"
    destination = f"abra_validation_denoised_wavenet_test_{file_id:04d}.h5"

    total_samples = 100_000_000
    chunk_size = 40_000

    with h5py.File(source, "r") as src, h5py.File(destination, "w") as dst:
        group = dst.create_group("timeseries")

        noisy_ds = src["timeseries/channel0001/timeseries"]
        target_ds = src["timeseries/channel0002/timeseries"]

        output_group = group.create_group("channel0001")
        target_group = group.create_group("channel0002")

        for key, value in src["timeseries/channel0001"].attrs.items():
            output_group.attrs[key] = value

        for key, value in src["timeseries/channel0002"].attrs.items():
            target_group.attrs[key] = value

        output = output_group.create_dataset(
            "timeseries", shape=(total_samples,), dtype="i1",
            chunks=(1_000_000,)
        )
        target = target_group.create_dataset(
            "timeseries", shape=(total_samples,), dtype="i1",
            chunks=(1_000_000,)
        )

        with torch.inference_mode():
            for start in range(0, total_samples, chunk_size):
                end = min(start + chunk_size, total_samples)

                noisy = np.asarray(noisy_ds[start:end], dtype=np.int16)
                x = torch.from_numpy(
                    (noisy + 128).astype(np.int64)
                ).reshape(1, -1).to(DEVICE)

                prediction = model(x).argmax(dim=1)
                denoised = (
                    prediction.cpu().numpy().reshape(-1) - 128
                ).astype(np.int8)

                output[start:end] = denoised
                target[start:end] = target_ds[start:end]

                if (start // chunk_size) % 250 == 0:
                    print(f"Processed {end:,}/{total_samples:,} samples", flush=True)

    print("Short WaveNet inference completed!", flush=True)
