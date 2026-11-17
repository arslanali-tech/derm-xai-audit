"""Measure how fast the expensive operations run on THIS laptop and project each phase's total time.

Uses random tensors and a randomly initialised ResNet-18 (speed does not depend on the weight
values, and this avoids downloading weights). Respects the laptop-safety settings: physical
cores - 1 threads and below-normal priority. Results: results/benchmark/cpu_benchmark.json.

Usage:  .venv\\Scripts\\python.exe scripts\\benchmark_cpu.py [--sizes 224 192 160]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from dermxai.utils import use_project_local_caches

use_project_local_caches()

import psutil  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torchvision.models import resnet18  # noqa: E402

from dermxai.utils import log_ram, prepare_long_job, project_root, setup_logging  # noqa: E402

logger = logging.getLogger("benchmark")

CACHE_ASPECT = 341 / 256  # width / height of the cached 600x450 images
N_CLASSES = 7
# Approximate workload sizes (exact numbers come from data/splits.csv once it exists).
N_ALL, N_TRAIN, N_VAL, N_TEST = 10015, 7010, 1502, 1503
STAGE_B_EPOCHS = 8
XAI_SUBSET, SANITY_SUBSET, SANITY_STATES = 300, 100, 7  # original + 6 randomised stages
TARGETS = 2  # predicted class and true class


def eval_shape(size: int) -> tuple[int, int]:
    """Full-frame evaluation input (H, W): short side = size, aspect ratio of the cache, no crop."""
    return size, round(size * CACHE_ASPECT)


def timed(fn, repeats: int) -> float:
    """Run fn once as warm-up, then `repeats` times; return mean seconds per call."""
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def bench_size(size: int, batches: int, batch_size: int, images: int) -> dict[str, float]:
    """Time each operation at one input size; return seconds per image for each."""
    torch.manual_seed(0)
    model = resnet18(weights=None, num_classes=N_CLASSES).eval()
    h, w = eval_shape(size)
    x_eval = torch.rand(batch_size, 3, h, w)
    x_train = torch.rand(batch_size, 3, size, size)
    y = torch.randint(0, N_CLASSES, (batch_size,))
    res: dict[str, float] = {}

    # 1) Stage A: pooled features for every image (one forward pass, no gradients).
    backbone = nn.Sequential(*list(model.children())[:-1])
    def features() -> None:
        with torch.inference_mode():
            backbone(x_eval)
    res["features_s_per_img"] = timed(features, batches) / batch_size

    # 2) Stage B: train only layer4 + fc. Frozen layers run without building a graph.
    frozen = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3)
    trainable = [*model.layer4.parameters(), *model.fc.parameters()]
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    def stage_b_step() -> None:
        model.layer4.train()
        with torch.no_grad():
            z = frozen(x_train)
        logits = model.fc(torch.flatten(model.avgpool(model.layer4(z)), 1))
        opt.zero_grad(set_to_none=True)
        loss_fn(logits, y).backward()
        opt.step()
    res["stage_b_train_s_per_img"] = timed(stage_b_step, batches) / batch_size
    model.eval()

    for p in model.parameters():  # explanations need input/activation gradients, not weight gradients
        p.requires_grad_(False)

    # 3) Grad-CAM: forward pass + gradient of the class score w.r.t. layer4's output.
    # With all weights frozen nothing upstream needs a gradient, so the hook marks layer4's output as
    # the start of the graph; backprop then only runs through avgpool + fc (cheap).
    acts: dict[str, torch.Tensor] = {}

    def keep(module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        acts["a"] = output.detach().requires_grad_(True)
        return acts["a"]

    handle = model.layer4.register_forward_hook(keep)
    def gradcam() -> None:
        with torch.enable_grad():
            out = model(x_eval)
            score = out.gather(1, y[:, None]).sum()
            torch.autograd.grad(score, acts["a"])
    res["gradcam_s_per_img"] = timed(gradcam, batches) / batch_size
    handle.remove()

    # 4) Integrated Gradients (Captum), 32 steps, all steps of one image in one batch.
    from captum.attr import IntegratedGradients, Occlusion
    ig = IntegratedGradients(model)
    one = x_eval[:1].clone()
    def run_ig() -> None:
        ig.attribute(one, baselines=torch.zeros_like(one), target=0, n_steps=32, internal_batch_size=32)
    res["ig32_s_per_img"] = timed(run_ig, images)

    # 5) Occlusion (Captum), 32x32 window, stride 32, 32 occluded copies per forward batch.
    occ = Occlusion(model)
    def run_occ() -> None:
        occ.attribute(one, sliding_window_shapes=(3, 32, 32), strides=(3, 32, 32), baselines=0.0,
                      target=0, perturbations_per_eval=32)
    res["occlusion_s_per_img"] = timed(run_occ, images)
    res["peak_rss_gb"] = psutil.Process().memory_info().rss / 1024**3
    return res


def project(r: dict[str, float]) -> dict[str, float]:
    """Projected minutes per phase from per-image timings."""
    stage_a = N_ALL * r["features_s_per_img"]
    stage_b_epoch = N_TRAIN * r["stage_b_train_s_per_img"] + N_VAL * r["features_s_per_img"]
    return {
        "stage_A_features_min": stage_a / 60,
        "stage_B_per_epoch_min": stage_b_epoch / 60,
        "train_total_min (A + 8 epochs B)": (stage_a + STAGE_B_EPOCHS * stage_b_epoch) / 60,
        "gradcam_full_test_2_targets_min": N_TEST * TARGETS * r["gradcam_s_per_img"] / 60,
        "ig32_subset300_2_targets_min": XAI_SUBSET * TARGETS * r["ig32_s_per_img"] / 60,
        "occlusion_subset300_2_targets_min": XAI_SUBSET * TARGETS * r["occlusion_s_per_img"] / 60,
        "sanity_100img_7states_gradcam+ig16_min":
            SANITY_SUBSET * SANITY_STATES * (r["gradcam_s_per_img"] + r["ig32_s_per_img"] / 2) / 60,
    }


def main() -> None:
    """Benchmark each input size, log a table and save JSON."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[224, 192, 160])
    ap.add_argument("--batches", type=int, default=10, help="timed batches for batched ops")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--images", type=int, default=10, help="timed images for IG / occlusion")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    out_dir = project_root() / "results" / "benchmark"
    setup_logging(out_dir / "log.txt")
    prepare_long_job(args.threads, low_priority=True, logger=logger)
    log_ram(logger, 0.5)
    logger.info("torch %s, device cpu, sizes %s", torch.__version__, args.sizes)

    results: dict[str, dict] = {}
    for size in args.sizes:
        logger.info("Benchmarking input size %d (train crop %dx%d, eval full frame %dx%d) ...",
                    size, size, size, *eval_shape(size))
        r = bench_size(size, args.batches, args.batch_size, args.images)
        results[str(size)] = {"per_image_seconds": r, "projected_minutes": project(r)}
        for k, v in r.items():
            logger.info("  %-28s %.4f", k, v)
        for k, v in results[str(size)]["projected_minutes"].items():
            logger.info("  -> %-42s %6.1f min", k, v)
        (out_dir / "cpu_benchmark.json").write_text(json.dumps(
            {"torch": torch.__version__, "threads": torch.get_num_threads(),
             "cpu": "see log", "note": "random weights + random tensors; excludes data loading",
             "results": results}, indent=2), encoding="utf-8")
    logger.info("Saved %s", out_dir / "cpu_benchmark.json")


if __name__ == "__main__":
    main()
