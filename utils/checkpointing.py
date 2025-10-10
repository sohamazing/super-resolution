# utils/checkpointing.py
import torch
from pathlib import Path
import glob
import os

def load_latest_checkpoint(model, optimizer, checkpoint_dir):
    latest_ckpt = None
    start_epoch = 0
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    ckpts = list(ckpt_dir.glob("latest_checkpoint_*.pth"))
    if ckpts:
        latest_ckpt = max(ckpts, key=lambda f: int(re.search(r'latest_checkpoint_(\d+).pth', f.name).group(1)))
        checkpoint = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resumed from {latest_ckpt} (epoch {start_epoch})")
    return start_epoch

def save_checkpoints(model, optimizer, epoch, best_psnr, psnr, checkpoint_dir):
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    # Delete old latest checkpoints
    for f in ckpt_dir.glob("latest_checkpoint_*.pth"):
        os.remove(f)

    latest_path = ckpt_dir / f"latest_checkpoint_{epoch}.pth"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "psnr": psnr,
    }, latest_path)

    if psnr > best_psnr:
        best_path = ckpt_dir / "best_model.pth"
        torch.save(model.state_dict(), best_path)
        print(f"New best model saved (epoch {epoch}, PSNR={psnr:.3f})")
    else:
        print(f"Checkpoint saved (epoch {epoch})")
