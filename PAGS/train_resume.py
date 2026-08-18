# -*- coding: utf-8 -*-
"""
train_resume.py  实时 L1 + 写入 txt
train_resume.py  去 compile 版
AMP + channels_last + bilinear + 线程隔离 仍在
"""
import argparse, copy, os, sys, yaml, torch, torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import autocast, GradScaler

import datasets, models, utils
from test import eval_psnr

torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"



# ===================== M1 helpers: pore pseudo target + sampling =====================
def _safe_load_state_dict(model, sd, strict=False, verbose=True):
    """
    Load state_dict but tolerate shape mismatch by:
    1) skipping unmatched shapes
    2) padding Linear weights when in_features increased by +1 (common for M2 pore_append_dec)
    Returns (missing_keys, unexpected_keys, skipped_keys).
    """
    import torch
    model_sd = model.state_dict()
    new_sd = {}
    skipped = []

    for k, v in sd.items():
        if k not in model_sd:
            continue
        mv = model_sd[k]
        if (not hasattr(v, "shape")) or (not hasattr(mv, "shape")):
            new_sd[k] = v
            continue

        if tuple(v.shape) == tuple(mv.shape):
            new_sd[k] = v
            continue

        # Handle Linear weight padding: [out, in_old] -> [out, in_old+1]
        if v.ndim == 2 and mv.ndim == 2 and v.shape[0] == mv.shape[0] and mv.shape[1] == v.shape[1] + 1:
            pad = torch.zeros((v.shape[0], 1), device=v.device, dtype=v.dtype)
            new_sd[k] = torch.cat([v, pad], dim=1)
            if verbose:
                print(f"[RESUME] padded weight: {k} {tuple(v.shape)} -> {tuple(new_sd[k].shape)}")
            continue

        skipped.append((k, tuple(v.shape), tuple(mv.shape)))

    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    if verbose and skipped:
        print("[RESUME] skipped mismatched params (ckpt -> model):")
        for k, s1, s2 in skipped[:20]:
            print(f"  - {k}: {s1} -> {s2}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

    return missing, unexpected, skipped

def _to_gray(x: torch.Tensor) -> torch.Tensor:
    """x: [B,C,H,W] -> [B,1,H,W] float"""
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if x.size(1) == 1:
        return x
    # assume RGB
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


@torch.no_grad()
def _robust_norm01(x: torch.Tensor, p1: float = 1.0, p99: float = 99.0, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample robust normalization to [0,1] using percentiles (quantiles)."""
    B = x.size(0)
    out = []
    for b in range(B):
        xb = x[b].reshape(-1).float()
        lo = torch.quantile(xb, p1 / 100.0)
        hi = torch.quantile(xb, p99 / 100.0)
        y = (x[b].float() - lo) / (hi - lo + eps)
        out.append(y.clamp(0.0, 1.0))
    return torch.stack(out, dim=0)


@torch.no_grad()
def build_pore_target_from_hr(
    hr_img: torch.Tensor,
    bright_ratio: float = 0.18,
    pore_is_dark: str = "auto",
    tau: float = 0.03,
    p1: float = 1.0,
    p99: float = 99.0,
) -> torch.Tensor:
    """
    Build a *soft* pore target map from HR patch.
    Returns: [B,1,H,W] float in [0,1]
    - bright_ratio: expected pore area fraction (approx)
    - pore_is_dark: "dark" / "bright" / "auto"
    - tau: soft threshold temperature
    """
    g = _robust_norm01(_to_gray(hr_img), p1=p1, p99=p99)  # [B,1,H,W] in [0,1]
    mode = str(pore_is_dark).lower()
    if mode in ["auto", "a"]:
        # default assumption for CT rocks: pores are dark (air) more often than bright
        is_dark = True
    elif mode in ["dark", "true", "1", "yes", "y"]:
        is_dark = True
    elif mode in ["bright", "false", "0", "no", "n"]:
        is_dark = False
    else:
        is_dark = True

    B = g.size(0)
    tgt = []
    br = float(max(1e-4, min(0.9999, bright_ratio)))
    for b in range(B):
        flat = g[b].reshape(-1).float()
        if is_dark:
            t = torch.quantile(flat, br)
            # pore=1 for low intensities
            m = torch.sigmoid((t - g[b]) / max(1e-6, tau))
        else:
            t = torch.quantile(flat, 1.0 - br)
            # pore=1 for high intensities
            m = torch.sigmoid((g[b] - t) / max(1e-6, tau))
        tgt.append(m)
    return torch.stack(tgt, dim=0)  # [B,1,H,W]


@torch.no_grad()
def sample_map_at_coord(map_b1hw: torch.Tensor, coord_bq2: torch.Tensor) -> torch.Tensor:
    """
    map: [B,1,H,W], coord: [B,Q,2] in [-1,1] (x,y)
    return: [B,Q,1]
    """
    B, Q, _ = coord_bq2.shape
    grid = coord_bq2.view(B, Q, 1, 2)
    # grid_sample output: [B,1,Q,1]
    samp = F.grid_sample(map_b1hw, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return samp.permute(0, 2, 3, 1).reshape(B, Q, 1)

# ===================== end M1 helpers =====================


def make_data_loader(spec, tag=""):
    if spec is None:
        return None, None
    dataset = datasets.make(spec["dataset"])
    dataset = datasets.make(spec["wrapper"], args={"dataset": dataset})
    print(f"[{tag}] dataset size={len(dataset)}")
    for k, v in dataset[0].items():
        if not isinstance(v, float):
            print(f"  {k}: shape={tuple(v.shape)}")
    suggested_workers = max(8, (os.cpu_count() or 8) // 2)
    num_workers = int(spec.get("num_workers", suggested_workers))
    is_train = tag == "train"
    loader = DataLoader(
        dataset,
        batch_size=spec["batch_size"],
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=is_train,
    )
    return loader, dataset


def make_data_loaders(cfg):
    train_loader, train_dataset = make_data_loader(cfg.get("train_dataset"), tag="train")
    val_loader, val_dataset = make_data_loader(cfg.get("val_dataset"), tag="val")
    return train_loader, val_loader, train_dataset, val_dataset


def build_scheduler(optimizer, cfg, epoch_start):
    ms = cfg.get("multi_step_lr")
    if ms is None:
        return None
    return MultiStepLR(
        optimizer,
        milestones=ms["milestones"],
        gamma=ms.get("gamma", 0.5),
        last_epoch=epoch_start - 2,
    )


def save_checkpoint(save_path, epoch, model, optimizer, cfg, lr_scheduler=None, suffix="last"):
    os.makedirs(save_path, exist_ok=True)
    model_ = model.module if isinstance(model, nn.DataParallel) else model
    model_spec = copy.deepcopy(cfg["model"])
    model_spec["sd"] = model_.state_dict()
    optim_spec = copy.deepcopy(cfg["optimizer"])
    optim_spec["sd"] = optimizer.state_dict()
    sv = {"model": model_spec, "optimizer": optim_spec, "epoch": epoch}
    if lr_scheduler is not None:
        sv["lr_scheduler"] = lr_scheduler.state_dict()
    out_path = os.path.join(save_path, f"epoch-{suffix}.pth" if suffix != "last" else "epoch-last.pth")
    torch.save(sv, out_path)
    print(f"[CKPT] saved -> {out_path}")


def prepare_training(cfg, resume_override=None):
    """
    Resume 逻辑（兼容从 L1 baseline ckpt -> M1/M2 新模型）：
      - 始终按 cfg["model"] 实例化当前模型
      - 从 ckpt 里只加载权重 sd（strict=False），让新增 head 随机初始化
      - optimizer 默认按 cfg 新建（避免 param group 不匹配）
    """
    resume_path = resume_override if resume_override is not None else cfg.get("resume")

    model = models.make(cfg["model"]).cuda()
    optimizer = utils.make_optimizer(model.parameters(), cfg["optimizer"])
    epoch_start = 1
    lr_scheduler = build_scheduler(optimizer, cfg, epoch_start)

    if resume_path and os.path.isfile(resume_path):
        print(f"[RESUME] loading checkpoint from: {resume_path}")
        sv_file = torch.load(resume_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        sd = None
        if isinstance(sv_file, dict) and "model" in sv_file and isinstance(sv_file["model"], dict):
            sd = sv_file["model"].get("sd", None)
        if sd is None and isinstance(sv_file, dict):
            sd = sv_file.get("sd", None)
        if sd is not None:
            missing, unexpected, _skipped = _safe_load_state_dict(model, sd, strict=False, verbose=True)
            if missing:
                print(f"[RESUME] missing keys: {len(missing)} (new params, ok)")
            if unexpected:
                print(f"[RESUME] unexpected keys: {len(unexpected)}")
        epoch_start = int(sv_file.get("epoch", 0)) + 1
        lr_scheduler = build_scheduler(optimizer, cfg, epoch_start)
        if lr_scheduler is not None and "lr_scheduler" in sv_file:
            try:
                lr_scheduler.load_state_dict(sv_file["lr_scheduler"])
            except Exception:
                pass
    else:
        print("[RESUME] start from scratch.")

    print("model: #params={}".format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler



# ----------- 实时 L1 日志 -----------
def init_l1_log(save_path):
    log_file = os.path.join(save_path, "epoch-l1.txt")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"# {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  start\n")
    return log_file


def log_l1(log_file, epoch, step, l1_val, total_val=None, bce_val=None, ap_val=None,
           w_bce=None, w_ap=None, pore_ratio=None, pore_pred_mean=None):
    parts = [
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        f"epoch {epoch}",
        f"step {step}",
        f"L1 {l1_val:.6f}",
    ]
    if total_val is not None:
        parts.append(f"Total {float(total_val):.6f}")
    if bce_val is not None:
        parts.append(f"BCE {float(bce_val):.6f}")
    if ap_val is not None:
        parts.append(f"AP {float(ap_val):.6f}")
    if w_bce is not None:
        parts.append(f"w_bce {float(w_bce):.4f}")
    if w_ap is not None:
        parts.append(f"w_ap {float(w_ap):.4f}")
    if pore_ratio is not None:
        parts.append(f"pore_ratio {float(pore_ratio):.4f}")
    if pore_pred_mean is not None:
        parts.append(f"pore_pred_mean {float(pore_pred_mean):.4f}")

    line = "  ".join(parts)
    print(line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ------------------------------------


def train_one_epoch(train_loader, model, optimizer, cfg, scaler, epoch, log_file, log_interval):
    model.train()
    loss_fn = nn.L1Loss(reduction="none")

    meter_l1 = utils.Averager()
    meter_total = utils.Averager()
    meter_bce = utils.Averager()
    meter_ap = utils.Averager()
    meter_pore_ratio = utils.Averager()
    meter_pore_pred_mean = utils.Averager()

    data_norm = cfg["data_norm"]
    t = data_norm["inp"]
    inp_sub = torch.FloatTensor(t["sub"]).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t["div"]).view(1, -1, 1, 1).cuda()
    t = data_norm["gt"]
    gt_sub = torch.FloatTensor(t["sub"]).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t["div"]).view(1, 1, -1).cuda()

    # ----- M1 config -----
    pg = cfg.get("porosity_guidance", {}) or {}
    pg_enable = bool(pg.get("enable", False))
    start_epoch = int(pg.get("start_epoch", 0))
    ramp_epochs = int(pg.get("ramp_epochs", 1))
    w_bce0 = float(pg.get("w_bce", 0.0))
    w_ap0 = float(pg.get("w_ap", 0.0))

    # ramp factor
    if (not pg_enable) or (epoch < start_epoch):
        alpha = 0.0
    else:
        alpha = min(1.0, (epoch - start_epoch + 1) / max(1, ramp_epochs))
    w_bce = w_bce0 * alpha
    w_ap = w_ap0 * alpha

    for step, batch in enumerate(tqdm(train_loader, leave=False, desc="train"), 1):
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        inp = (batch["inp"] - inp_sub) / inp_div

        with autocast():
            if pg_enable:
                pred, aux = model(inp, batch["coord"], batch["scale"], batch["cell"], return_aux=True)
            else:
                pred = model(inp, batch["coord"], batch["scale"], batch["cell"])
                aux = {}

            gt = (batch["gt"] - gt_sub) / gt_div
            l1 = loss_fn(pred, gt).mean(dtype=torch.float32)

            total = l1
            bce = None
            ap = None
            pore_ratio = None
            pore_pred_mean = None

            if pg_enable and (w_bce > 0.0 or w_ap > 0.0):
                hr_img = batch.get("hr_img", None)
                if hr_img is not None:
                    # build pore pseudo target on HR and sample at coords
                    pore_tgt_map = build_pore_target_from_hr(
                        hr_img,
                        bright_ratio=float(pg.get("bright_ratio", 0.18)),
                        pore_is_dark=str(pg.get("pore_is_dark", pg.get("pore_is_dark", "auto"))),
                        tau=float(pg.get("tau", 0.03)),
                        p1=float(pg.get("p1", 1.0)),
                        p99=float(pg.get("p99", 99.0)),
                    )
                    pore_tgt_q = sample_map_at_coord(pore_tgt_map, batch["coord"])  # [B,Q,1]
                    pore_pred_q = aux.get("pore_pred", None)
                    if pore_pred_q is None:
                        # fallback: if forward didn't return pore_pred, try query_pore
                        model_ = model.module if isinstance(model, nn.DataParallel) else model
                        if hasattr(model_, "query_pore"):
                            pore_pred_q = model_.query_pore(batch["coord"])
                    if pore_pred_q is not None:
                        pore_pred_q = pore_pred_q.clamp(1e-4, 1 - 1e-4)
                        with torch.cuda.amp.autocast(enabled=False):
                            bce = F.binary_cross_entropy(pore_pred_q.float(), pore_tgt_q.float(), reduction="mean")
                        ap = torch.abs(pore_pred_q.mean() - pore_tgt_q.mean())
                        total = total + (w_bce * bce) + (w_ap * ap)

                        pore_ratio = pore_tgt_q.mean()
                        pore_pred_mean = pore_pred_q.mean()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()

        meter_l1.add(float(l1.detach().item()))
        meter_total.add(float(total.detach().item()))
        if bce is not None:
            meter_bce.add(float(bce.detach().item()))
        if ap is not None:
            meter_ap.add(float(ap.detach().item()))
        if pore_ratio is not None:
            meter_pore_ratio.add(float(pore_ratio.detach().item()))
        if pore_pred_mean is not None:
            meter_pore_pred_mean.add(float(pore_pred_mean.detach().item()))

        # 实时打印 & 写文件
        if step % log_interval == 0 or step == len(train_loader):
            log_l1(
                log_file, epoch, step,
                meter_l1.item(),
                total_val=meter_total.item(),
                bce_val=(meter_bce.item() if meter_bce.n > 0 else None),
                ap_val=(meter_ap.item() if meter_ap.n > 0 else None),
                w_bce=w_bce if pg_enable else None,
                w_ap=w_ap if pg_enable else None,
                pore_ratio=(meter_pore_ratio.item() if meter_pore_ratio.n > 0 else None),
                pore_pred_mean=(meter_pore_pred_mean.item() if meter_pore_pred_mean.n > 0 else None),
            )

    return meter_l1.item(), meter_total.item(), (meter_bce.item() if meter_bce.n > 0 else None), (meter_ap.item() if meter_ap.n > 0 else None)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--log_interval", type=int, default=1000, help="实时打印 L1 的步频")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    save_name = args.name or "_" + os.path.splitext(os.path.basename(args.config))[0]
    if args.tag:
        save_name += "_" + args.tag
    save_path = os.path.join("./save", save_name)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    train_loader, val_loader, *_ = make_data_loaders(cfg)
    if cfg.get("data_norm") is None:
        cfg["data_norm"] = {"inp": {"sub": [0], "div": [1]}, "gt": {"sub": [0], "div": [1]}}

    model, optimizer, epoch_start, lr_scheduler = prepare_training(cfg, resume_override=args.resume)

    # ---- 仅 channels_last，无 compile ----
    model = model.to(memory_format=torch.channels_last)

    n_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    if n_gpus > 1:
        model = nn.DataParallel(model)

    epoch_max = cfg["epoch_max"]
    epoch_val = cfg.get("epoch_val")
    epoch_save = cfg.get("epoch_save")
    timer = utils.Timer()
    scaler = GradScaler()

    log_file = init_l1_log(save_path)          # 初始化日志

    try:
        for epoch in range(epoch_start, epoch_max + 1):
            t0 = timer.t()
            print(f"\n===== Epoch {epoch}/{epoch_max} =====")
            print(f"lr = {optimizer.param_groups[0]['lr']:.6g}")

            train_l1, train_total, train_bce, train_ap = train_one_epoch(train_loader, model, optimizer, cfg, scaler,
                                         epoch, log_file, args.log_interval)
            msg = f"[train] epoch L1 = {train_l1:.6f}  Total = {train_total:.6f}"
            if train_bce is not None:
                msg += f"  BCE = {train_bce:.6f}"
            if train_ap is not None:
                msg += f"  AP = {train_ap:.6f}"
            print(msg)

            if lr_scheduler is not None:
                lr_scheduler.step()

            save_checkpoint(save_path, epoch, model, optimizer, cfg, lr_scheduler, suffix="last")

            if (epoch_save is not None) and (epoch % epoch_save == 0):
                save_checkpoint(save_path, epoch, model, optimizer, cfg, lr_scheduler, suffix=str(epoch))
                # 保存孔隙概率图等可视化（M1）
                model_ = model.module if isinstance(model, nn.DataParallel) else model
                try:
                    save_porosity_vis(save_path, epoch, model_, cfg, val_loader)
                except Exception as e:
                    print(f"[vis] skipped: {e}")

            if (epoch_val is not None) and (epoch % epoch_val == 0):
                model_ = model.module if isinstance(model, nn.DataParallel) else model
                psnr = eval_psnr(val_loader, model_, data_norm=cfg["data_norm"],
                                 eval_type=cfg.get("eval_type"), eval_bsize=cfg.get("eval_bsize"))
                print(f"[val] psnr = {psnr:.4f}")

            t1 = timer.t()
            print(f"[time] epoch={utils.time_text(t1 - t0)}  elapsed={utils.time_text(t1)}")

    except KeyboardInterrupt:
        cur_epoch = max(epoch_start, locals().get("epoch", epoch_start))
        print("\n[INTERRUPT] saving ...")
        save_checkpoint(save_path, cur_epoch, model, optimizer, cfg, lr_scheduler, suffix="interrupt")
        save_checkpoint(save_path, cur_epoch, model, optimizer, cfg, lr_scheduler, suffix="last")
        sys.exit(0)


if __name__ == "__main__":
    main()