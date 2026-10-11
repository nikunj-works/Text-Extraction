#!/usr/bin/env python3
"""Detect text regions in a video with PP-OCRv6 (detection only) and pixelate them."""

import argparse
import os
import shutil
import subprocess

import cv2
import numpy as np
from paddleocr import TextDetection


def build_detector(tier, device, det_side, box_thresh, unclip):
    return TextDetection(
        model_name=f"PP-OCRv6_{tier}_det",
        device=device,
        limit_side_len=det_side,
        limit_type="max",
        box_thresh=box_thresh,
        unclip_ratio=unclip,
    )


def detect(det, frame, min_score, max_short_frac, max_area_frac):
    """Return (kept, rejected) lists of (polygon, score)."""
    h, w = frame.shape[:2]
    kept, rejected = [], []
    res = det.predict(input=frame, batch_size=1)
    if not res:
        return kept, rejected
    r = res[0]
    for poly, score in zip(r["dt_polys"], r["dt_scores"]):
        poly = np.asarray(poly, dtype=np.int32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(poly)
        too_big = min(rw, rh) > max_short_frac * h or rw * rh > max_area_frac * w * h
        if score < min_score or too_big:
            rejected.append((poly, float(score)))
        else:
            kept.append((poly, float(score)))
    return kept, rejected


def pixelate(frame, polys, pad):
    """Pixelate only the polygon area (not its bounding rectangle), in place."""
    h, w = frame.shape[:2]
    for poly in polys:
        x, y, bw, bh = cv2.boundingRect(poly)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
        local = (poly - np.array([x1, y1])).astype(np.int32)
        cv2.fillPoly(mask, [local], 255)
        if pad > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
            mask = cv2.dilate(mask, k)
        roi = frame[y1:y2, x1:x2]
        cell = max(6, min(roi.shape[:2]) // 3)
        small = cv2.resize(
            roi,
            (max(1, roi.shape[1] // cell), max(1, roi.shape[0] // cell)),
            interpolation=cv2.INTER_AREA,
        )
        pix = cv2.resize(
            small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        roi[mask > 0] = pix[mask > 0]


def main():
    ap = argparse.ArgumentParser(
        description="Blur text in a video using PP-OCRv6 detection only."
    )
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="output/blurred.mp4")
    ap.add_argument("--tier", choices=["tiny", "small", "medium"], default="medium")
    ap.add_argument("--device", default="gpu:0")
    ap.add_argument(
        "--min-score", type=float, default=0.4, help="min detection score to blur"
    )
    ap.add_argument(
        "--box-thresh",
        type=float,
        default=0.35,
        help="detector's own box threshold (lower = more boxes)",
    )
    ap.add_argument(
        "--unclip",
        type=float,
        default=1.8,
        help="box expansion ratio (higher = bigger boxes)",
    )
    ap.add_argument(
        "--max-short-frac",
        type=float,
        default=0.10,
        help="skip boxes whose short side exceeds this fraction of frame height",
    )
    ap.add_argument(
        "--max-area-frac",
        type=float,
        default=0.03,
        help="skip boxes larger than this fraction of the frame",
    )
    ap.add_argument("--every", type=int, default=1, help="run detection every N frames")
    ap.add_argument("--hold", type=int, default=2, help="frames a box stays blurred")
    ap.add_argument("--pad", type=int, default=6, help="extra pixels around each box")
    ap.add_argument(
        "--det-side",
        type=int,
        default=1920,
        help="max image side fed to the detector (lower if you run out of VRAM)",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="draw boxes instead of blurring: red = would be blurred, "
        "blue = rejected by filters",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    silent_path = os.path.splitext(args.output)[0] + "_silent.mp4"

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(silent_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    det = build_detector(
        args.tier, args.device, args.det_side, args.box_thresh, args.unclip
    )
    tracked = []  # [polygon, frames_left]
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if idx % args.every == 0:
                found, rejected = detect(
                    det, frame, args.min_score, args.max_short_frac, args.max_area_frac
                )
                if args.debug:
                    for poly, score in rejected:  # blue = rejected by filters
                        cv2.polylines(frame, [poly], True, (255, 0, 0), 1)
                    for poly, score in found:  # red = would be blurred
                        cv2.polylines(frame, [poly], True, (0, 0, 255), 2)
                        cv2.putText(
                            frame,
                            f"{score:.2f}",
                            tuple(int(v) for v in poly[0]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                else:
                    tracked.extend([p, args.hold] for p, _ in found)

            if not args.debug:
                pixelate(frame, [p for p, _ in tracked], args.pad)
                tracked = [[p, n - 1] for p, n in tracked if n > 1]

            writer.write(frame)
            idx += 1
            if idx % 100 == 0:
                print(f"{idx}/{total} frames", flush=True)
    finally:
        cap.release()
        writer.release()

    # OpenCV drops audio, so copy it back from the original with ffmpeg if available.
    if shutil.which("ffmpeg"):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                silent_path,
                "-i",
                args.input,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "medium",
                "-c:a",
                "copy",
                "-shortest",
                args.output,
            ],
            check=True,
        )
        os.remove(silent_path)
    else:
        os.replace(silent_path, args.output)
        print("ffmpeg not found on PATH, so the output has no audio.")
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
