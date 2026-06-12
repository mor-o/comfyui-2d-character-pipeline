"""
Download pipeline models from HuggingFace into the correct ComfyUI folders.

Usage:
    python drivers/download_models.py --comfyui-root /path/to/ComfyUI --w2
    python drivers/download_models.py --comfyui-root /path/to/ComfyUI --all
    python drivers/download_models.py --comfyui-root /path/to/ComfyUI --all --token hf_xxx

You can also set the token via environment variable instead of --token:
    export HF_TOKEN="hf_xxxxxxxxxxxx"          (bash)
    $env:HF_TOKEN = "hf_xxxxxxxxxxxx"          (PowerShell)

Get a token at: https://huggingface.co/settings/tokens (Read access is enough)

Models for W3/W5 (BiRefNet, SAM3) auto-download on first ComfyUI run — not included here.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# (workflow, repo_id, filename_in_repo, dest_folder, approx_size, dest_name_or_None)
#
# dest_name: final filename in dest_folder. Required when filename_in_repo includes a
# subfolder (hf_hub_download preserves repo paths) or when the on-disk name must
# match what the workflow JSON expects but differs from the repo filename.
MODEL_LIST = [
    # ── W2: Video Generation ──────────────────────────────────────────────────
    ("W2", "jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF",
           "high_noise/wan2.2_i2v_A14b_high_noise_lightx2v_4step-Q5_K_M.gguf",
           "unet", "~10 GB",
           "wan2.2_i2v_high_noise_lightx2v_Q5_K_M.gguf"),
    ("W2", "jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF",
           "low_noise/wan2.2_i2v_A14b_low_noise_lightx2v_4step-Q5_K_M.gguf",
           "unet", "~10 GB",
           "wan2.2_i2v_low_noise_lightx2v_Q5_K_M.gguf"),
    ("W2", "city96/umt5-xxl-encoder-gguf",
           "umt5-xxl-encoder-Q5_K_M.gguf",
           "clip", "~5 GB",
           None),
    ("W2", "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
           "split_files/vae/wan_2.1_vae.safetensors",
           "vae", "~250 MB",
           "wan_2.1_vae.safetensors"),

    # ── W4: Cosmetic Inpaint (shares UMT5 + VAE with W2) ─────────────────────
    ("W4", "QuantStack/Wan2.1_14B_VACE-GGUF",
           "Wan2.1_14B_VACE-Q5_K_M.gguf",
           "unet", "~10 GB",
           None),
    ("W4", "Kijai/WanVideo_comfy",
           "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",
           "loras", "~600 MB",
           None),

    # ── W1: Pose Edit ─────────────────────────────────────────────────────────
    ("W1", "QuantStack/Qwen-Image-Edit-2509-GGUF",
           "Qwen-Image-Edit-2509-Q5_K_M.gguf",
           "unet", "~10 GB",
           None),
    ("W1", "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
           "Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf",
           "clip", "~5 GB",
           None),
    # mmproj (vision encoder) — must live beside the main CLIP GGUF with "mmproj" in name.
    # ComfyUI-GGUF auto-discovers it by matching the model name stem in the filename.
    ("W1", "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
           "mmproj-F16.gguf",
           "clip", "~1.3 GB",
           "Qwen2.5-VL-7B-Instruct-mmproj-F16.gguf"),
    ("W1", "Comfy-Org/Qwen-Image_ComfyUI",
           "split_files/vae/qwen_image_vae.safetensors",
           "vae", "~250 MB",
           "qwen_image_vae.safetensors"),
    ("W1", "lightx2v/Qwen-Image-Lightning",
           "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
           "loras", "~600 MB",
           "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"),
    ("W1", "InstantX/Qwen-Image-ControlNet-Union",
           "diffusion_pytorch_model.safetensors",
           "controlnet", "~2.5 GB",
           "Qwen-Image-InstantX-ControlNet-Union.safetensors"),
]


def check_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download  # noqa: F401
    except ImportError:
        print("[error] huggingface_hub not installed.")
        print("  Run:  pip install huggingface_hub")
        sys.exit(1)


def final_dest(dest_folder: Path, filename_in_repo: str, dest_name: str | None) -> Path:
    name = dest_name if dest_name else Path(filename_in_repo).name
    return dest_folder / name


def already_downloaded(dest_folder: Path, filename_in_repo: str, dest_name: str | None) -> bool:
    return final_dest(dest_folder, filename_in_repo, dest_name).exists()


def download_one(repo_id: str, filename_in_repo: str, dest_folder: Path,
                 size: str, token: str | None, dest_name: str | None):
    from huggingface_hub import hf_hub_download

    dest_file = final_dest(dest_folder, filename_in_repo, dest_name)

    if dest_file.exists():
        print(f"  [skip]  {dest_file.name}  (already exists)")
        return

    dest_folder.mkdir(parents=True, exist_ok=True)
    print(f"  [download]  {dest_file.name}  ({size})")
    print(f"              from {repo_id}/{filename_in_repo}")

    try:
        downloaded = Path(hf_hub_download(
            repo_id=repo_id,
            filename=filename_in_repo,
            local_dir=str(dest_folder),
            token=token or None,
        ))
        if downloaded != dest_file:
            shutil.move(str(downloaded), str(dest_file))
            try:
                downloaded.parent.rmdir()
            except OSError:
                pass
        print(f"  [done]  {dest_file}")
    except Exception as e:
        print(f"  [error]  {e}")
        if "401" in str(e) or "authentication" in str(e).lower():
            print("  This repo requires a HuggingFace token.")
            print("  Get one at: https://huggingface.co/settings/tokens")
            print("  Then run with:  --token hf_xxxxxxxxxxxx")
        print(f"  Verify filename at: https://huggingface.co/{repo_id}/tree/main")


TOKEN_FILE = Path(__file__).parent / ".tkn"


def load_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if TOKEN_FILE.exists():
        t = TOKEN_FILE.read_text().strip()
        if t:
            return t
    return None


def run(comfyui_root: Path, workflows: set[str], token: str | None):
    check_huggingface_hub()

    models_dir = comfyui_root / "models"
    if not models_dir.exists():
        print(f"[error] models directory not found: {models_dir}")
        print("  Check that --comfyui-root points to your ComfyUI installation.")
        sys.exit(1)

    token = load_token(token)
    if token:
        print(f"  Using HF token: {token[:8]}{'*' * max(0, len(token) - 8)}")

    targets = [m for m in MODEL_LIST if m[0] in workflows]

    total_new = sum(
        1 for wf, repo, fname, folder, size, dest in targets
        if not already_downloaded(models_dir / folder, fname, dest)
    )
    print(f"\nDownloading {len(targets)} file(s) for workflows: {', '.join(sorted(workflows))}")
    print(f"  {total_new} new  |  {len(targets) - total_new} already present\n")

    for wf, repo, fname, folder, size, dest in targets:
        download_one(repo, fname, models_dir / folder, size, token, dest)

    print("\nAll done.")
    print(f"Models are in: {models_dir}")


def interactive_menu():
    print("=== Pipeline Model Downloader ===\n")
    print("  1  W2 — Video Generation    (~25 GB, needed for every animation)")
    print("  2  W4 — Cosmetic Inpaint    (~11 GB extra, shares W2 encoder/VAE)")
    print("  3  W1 — Pose Edit           (~19 GB, only if you want OpenPose input)")
    print("  4  All of the above")
    print()
    choice = input("Select [1/2/3/4]: ").strip()
    mapping = {
        "1": {"W2"},
        "2": {"W2", "W4"},
        "3": {"W1"},
        "4": {"W1", "W2", "W4"},
    }
    if choice not in mapping:
        print("Invalid choice.")
        sys.exit(1)
    return mapping[choice]


def main():
    parser = argparse.ArgumentParser(description="Download pipeline models into ComfyUI")
    parser.add_argument("--comfyui-root", required=True,
                        help="Path to your ComfyUI installation (the folder containing models/)")
    parser.add_argument("--w1",    action="store_true", help="Download W1 models (pose edit)")
    parser.add_argument("--w2",    action="store_true", help="Download W2 models (video gen)")
    parser.add_argument("--w4",    action="store_true", help="Download W4 models (cosmetic inpaint)")
    parser.add_argument("--all",   action="store_true", help="Download all models")
    parser.add_argument("--token", default=None,
                        help="HuggingFace access token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    comfyui_root = Path(args.comfyui_root).expanduser().resolve()

    if args.all:
        workflows = {"W1", "W2", "W4"}
    elif args.w1 or args.w2 or args.w4:
        workflows = set()
        if args.w1: workflows.add("W1")
        if args.w2: workflows.add("W2")
        if args.w4: workflows.add("W4")
    else:
        workflows = interactive_menu()

    run(comfyui_root, workflows, args.token)


if __name__ == "__main__":
    main()
