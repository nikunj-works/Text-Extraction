#!/usr/bin/env python3
"""
blur_text.py - privacy blur for on-screen text in video (PP-OCRv6, detection only).

DESIGN (read this before changing any number)
---------------------------------------------
Goal: no readable text survives. A missed word is a privacy leak; a blurred
non-text patch is cosmetic. So every decision below favours RECALL, and the
policy contains no per-video knobs.

  Stage 1  DETECT   (GPU, slow, run once, cached on disk)
           Multi-scale: the detector must see text at native pixel size.
             - frame fits ~native in one pass  -> one pass
             - otherwise (e.g. 4K)             -> one global downscaled pass
                                                  (large text) + overlapping
                                                  native-resolution tiles
                                                  (small text)
           Model defaults are used wherever possible; only box_thresh is
           lowered (recall floor).

  Stage 2  POLICY   (CPU, fast, re-runs from the cache in seconds)
           - Hysteresis: confident boxes (>= SCORE_HI) are always kept. Weak
             boxes (SCORE_LO..SCORE_HI) are kept only if a nearby frame
             (within WINDOW_SEC) also has a box in the same place (real text
             persists, noise does not).
           - Temporal fusion: because the whole video is available offline,
             a box found in frame t also protects frames t-k..t+k. It is
             carried across frames with optical flow (moving text stays
             covered) AND kept un-moved (static overlays such as subtitles or
             watermarks stay covered even when the background moves and
             flow drifts: measured, flow-only leaked up to 24% of a static
             caption over a fast pan; both copies leak nothing).
           - Margin is proportional to text height, not a pixel count.
           - No geometry filters on text size/shape (they cannot work for all
             resolutions and fonts). Only a catastrophic-failure guard.

  Stage 3  RENDER   Pixelate the union mask, mux original audio.

Changing thresholds only needs Stage 2 again: no GPU re-run.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Policy constants. Scale-free (ratios / seconds), deliberately not CLI flags.
# ----------------------------------------------------------------------------
CACHE_VERSION = 2

SCORE_LO = 0.30  # floor passed to the detector as box_thresh (recall)
SCORE_HI = 0.60  # PaddleOCR's own default box_thresh: trusted on its own
WINDOW_SEC = 0.16  # ONE temporal horizon (seconds): a weak box needs a neighbour
# within it, and a box protects frames within it
MAX_K = 6  # cap on half-window in frames (bounds memory)
SUPPORT_CONTAIN = 0.5  # overlap needed to count as "same place"
PAD_FRAC = 0.10  # extra margin, as a fraction of text height
PAD_MIN = 3  # ... never less than this many pixels
MAX_AREA_FRAC = 0.40  # catastrophic-failure guard only (not a tuning knob)

TILE_LONG = 2048  # detector window, long side
TILE_SHORT = 1152  # detector window, short side
TILE_OVERLAP = 128  # minimum overlap between tiles (>= typical text height)
SINGLE_PASS_MIN_SCALE = 0.75  # shrink text by at most 25% before also tiling

FLOW_LONG = 640  # optical flow runs on frames downscaled to this long side


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def tile_starts(total, tile, min_overlap):
    """Start offsets so tiles of size `tile` cover [0,total) with overlap."""
    if total <= tile:
        return [0]
    n = int(math.ceil((total - tile) / float(tile - min_overlap))) + 1
    return [int(round(v)) for v in np.linspace(0, total - tile, n)]


def tile_boxes(w, h):
    """List of (x0, y0, tw, th) native-resolution tiles covering a w x h frame."""
    win_w, win_h = (TILE_LONG, TILE_SHORT) if w >= h else (TILE_SHORT, TILE_LONG)
    win_w, win_h = min(win_w, w), min(win_h, h)
    return [
        (x, y, win_w, win_h)
        for y in tile_starts(h, win_h, TILE_OVERLAP)
        for x in tile_starts(w, win_w, TILE_OVERLAP)
    ]


def poly_bbox(poly):
    x, y, w, h = cv2.boundingRect(np.asarray(poly, dtype=np.int32))
    return x, y, x + w, y + h


# ----------------------------------------------------------------------------
# Stage 1: detection
# ----------------------------------------------------------------------------
class MultiScaleDetector:
    """Runs `run(img) -> [(poly Nx2 float, score)]` at the right scales."""

    def __init__(self, run, tiling=True):
        self.run = run
        self.tiling = tiling

    def _pass(self, img, inv_scale, ox, oy):
        out = []
        for poly, score in self.run(img):
            p = np.asarray(poly, dtype=np.float64) * inv_scale + np.array([ox, oy])
            out.append((p, float(score)))
        return out

    def detect(self, frame):
        h, w = frame.shape[:2]
        s = min(1.0, TILE_LONG / float(max(w, h)))
        small = (
            frame
            if s == 1.0
            else cv2.resize(
                frame,
                (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                interpolation=cv2.INTER_AREA,
            )
        )
        dets = self._pass(small, 1.0 / s, 0, 0)
        if self.tiling and s < SINGLE_PASS_MIN_SCALE:
            for x0, y0, tw, th in tile_boxes(w, h):
                dets += self._pass(frame[y0 : y0 + th, x0 : x0 + tw], 1.0, x0, y0)
        return dets


def make_paddle_detector(tier, device, tiling=True):
    from paddleocr import TextDetection  # lazy: keeps tests/CI free of paddle

    model = TextDetection(
        model_name="PP-OCRv6_%s_det" % tier,
        device=device,
        limit_side_len=TILE_LONG,
        limit_type="max",
        box_thresh=SCORE_LO,  # only deviation from model defaults
    )

    def run(img):
        res = model.predict(input=img, batch_size=1)
        if not res:
            return []
        r = res[0]
        return list(zip(r["dt_polys"], r["dt_scores"]))

    return MultiScaleDetector(run, tiling=tiling)


def video_meta(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("Cannot open video: %s" % path)
    meta = dict(
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return meta


def detect_video(path, detector, stride=1, log=print):
    """Return list (one entry per frame) of None | [(int32 poly, score)]."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("Cannot open video: %s" % path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames, i, t0 = [], 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            h, w = frame.shape[:2]
            keep = []
            for poly, score in detector.detect(frame):
                p = np.round(np.asarray(poly)).astype(np.int32)
                if len(p) < 3 or cv2.contourArea(p) > MAX_AREA_FRAC * w * h:
                    continue
                keep.append((p, float(score)))
            frames.append(keep)
        else:
            frames.append(None)
        i += 1
        if i % 25 == 0:
            fps = i / max(1e-6, time.time() - t0)
            eta = (total - i) / fps if total else 0
            log("detect %d/%d frames  (%.1f fps, ETA %ds)" % (i, total, fps, eta))
    cap.release()
    return frames


# ----------------------------------------------------------------------------
# Detection cache (so Stage 2/3 can be re-run without the GPU)
# ----------------------------------------------------------------------------
def cache_key(path, tier, stride, tiling):
    st = os.stat(path)
    return dict(
        version=CACHE_VERSION,
        size=st.st_size,
        mtime=int(st.st_mtime),
        tier=tier,
        stride=stride,
        tiling=tiling,
        box_thresh=SCORE_LO,
        tile=[TILE_LONG, TILE_SHORT, TILE_OVERLAP],
    )


def save_cache(cache_path, key, frames):
    data = dict(
        key=key,
        frames=[
            None if f is None else [[round(s, 3), p.tolist()] for p, s in f]
            for f in frames
        ],
    )
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))


def load_cache(cache_path, key):
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if data.get("key") != key:
        return None
    return [
        None if f is None else [(np.asarray(p, dtype=np.int32), float(s)) for s, p in f]
        for f in data["frames"]
    ]


# ----------------------------------------------------------------------------
# Stage 2: policy
# ----------------------------------------------------------------------------
def _containment(a, b):
    """a: (M,4), b: (N,4) boxes x1,y1,x2,y2 -> (M,N) inter / min(area)."""
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.maximum(1, (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))
    area_b = np.maximum(1, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
    return inter / np.minimum(area_a[:, None], area_b[None, :])


def hysteresis(frames, support):
    """Split detections into kept / dropped per frame.

    Returns (kept, dropped); kept[t] = [(poly, score, reason)], reason in
    {"hi", "support"}; dropped[t] = [(poly, score)].
    """
    n = len(frames)
    boxes = [
        None
        if f is None
        else np.array([poly_bbox(p) for p, _ in f], dtype=np.float64).reshape(-1, 4)
        for f in frames
    ]
    kept, dropped = [], []
    for t, f in enumerate(frames):
        k_t, d_t = [], []
        if f:
            near = [
                boxes[u]
                for u in range(max(0, t - support), min(n, t + support + 1))
                if u != t and boxes[u] is not None and len(boxes[u])
            ]
            near = np.concatenate(near) if near else np.zeros((0, 4))
            supported = (
                _containment(boxes[t], near).max(axis=1) >= SUPPORT_CONTAIN
                if len(near)
                else np.zeros(len(f), bool)
            )
            for (p, s), sup in zip(f, supported):
                if s >= SCORE_HI:
                    k_t.append((p, s, "hi"))
                elif sup:
                    k_t.append((p, s, "support"))
                else:
                    d_t.append((p, s))
        kept.append(k_t)
        dropped.append(d_t)
    return kept, dropped


def _small_gray(frame, scale):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _flow(a, b):
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def _warp(poly, flow, inv_scale):
    """Translate poly by the median flow inside its bbox."""
    fh, fw = flow.shape[:2]
    x1, y1, x2, y2 = [v / inv_scale for v in poly_bbox(poly)]
    x1, y1 = (
        int(max(0, min(fw - 3, math.floor(x1)))),
        int(max(0, min(fh - 3, math.floor(y1)))),
    )
    x2, y2 = (
        int(max(x1 + 3, min(fw, math.ceil(x2)))),
        int(max(y1 + 3, min(fh, math.ceil(y2)))),
    )
    d = np.median(flow[y1:y2, x1:x2].reshape(-1, 2), axis=0) * inv_scale
    return np.round(poly + d).astype(np.int32)


def build_mask(shape, polys):
    """Union mask of polys, each grown by a margin proportional to text height."""
    mask = np.zeros(shape[:2], np.uint8)
    for poly in polys:
        (_, _), (rw, rh), _ = cv2.minAreaRect(poly)
        d = int(round(PAD_FRAC * min(rw, rh))) + PAD_MIN
        cv2.fillPoly(mask, [poly], 255)
        cv2.polylines(mask, [poly], True, 255, thickness=2 * d + 1)
    return mask


def pixelate_mask(frame, mask):
    """Pixelate masked pixels in place; cell size follows each blob's size."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w < 2 or h < 2:
            continue
        roi = frame[y : y + h, x : x + w]
        sel = labels[y : y + h, x : x + w] == i
        cell = max(8, min(w, h) // 3)
        small = cv2.resize(
            roi, (max(1, w // cell), max(1, h // cell)), interpolation=cv2.INTER_AREA
        )
        pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        roi[sel] = pix[sel]


def policy_params(fps, stride):
    """(support, k): both are the same temporal horizon, in frames."""
    k = min(MAX_K, max(stride, int(round(WINDOW_SEC * fps)), 1))
    return k, k


def iter_masks(path, kept, k, with_flow=True, keep_static=True):
    """Yield (idx, frame, mask, carried) for every frame of the video.

    carried = polys that came from neighbouring frames (for debug drawing).
    """
    n = len(kept)
    cap = cv2.VideoCapture(path)
    frames, grays, f_fwd, f_bwd = {}, {}, {}, {}
    scale, loaded = None, 0

    def load_upto(j):
        nonlocal scale, loaded
        while loaded <= j and loaded < n:
            ok, fr = cap.read()
            if not ok:
                raise RuntimeError(
                    "video ended early at frame %d (cache mismatch?)" % loaded
                )
            if scale is None:
                scale = FLOW_LONG / float(max(fr.shape[:2]))
            frames[loaded] = fr
            if with_flow:
                grays[loaded] = _small_gray(fr, scale)
                if loaded > 0:
                    f_fwd[loaded - 1] = _flow(grays[loaded - 1], grays[loaded])
                    f_bwd[loaded - 1] = _flow(grays[loaded], grays[loaded - 1])
            loaded += 1

    for c in range(n):
        load_upto(min(n - 1, c + k))
        own = [p for p, _, _ in kept[c]]
        carried = []
        inv = 1.0 / scale
        for d in range(1, k + 1):
            for src in (c + d, c - d):
                if src < 0 or src >= n or not kept[src]:
                    continue
                for p, _, _ in kept[src]:
                    if keep_static or not with_flow:
                        carried.append(p)  # un-moved copy (static overlays)
                    if with_flow:
                        q, step = p, src
                        while step != c:  # chain flow src -> c
                            if step > c:
                                q = _warp(q, f_bwd[step - 1], inv)
                                step -= 1
                            else:
                                q = _warp(q, f_fwd[step], inv)
                                step += 1
                        carried.append(q)
        mask = build_mask(frames[c].shape, own + carried)
        yield c, frames[c], mask, own, carried
        # free what the next centre no longer needs
        frames.pop(c, None)
        for j in [j for j in grays if j < c + 1 - k]:
            grays.pop(j, None)
            f_fwd.pop(j, None)
            f_bwd.pop(j, None)
    cap.release()


# ----------------------------------------------------------------------------
# Stage 3: output
# ----------------------------------------------------------------------------
def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class Sink:
    """Writes BGR frames; pipes to ffmpeg (H.264 + original audio) if present."""

    def __init__(self, out_path, src_path, w, h, fps, n_frames):
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        self.proc = self.writer = None
        ff = find_ffmpeg()
        if ff:
            cmd = [
                ff,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                "%dx%d" % (w, h),
                "-r",
                "%.5f" % fps,
                "-i",
                "-",
                "-i",
                src_path,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                # Length = exactly the video's length. (-shortest would drop
                # video frames whenever the audio track ends slightly early.)
                "-t",
                "%.5f" % (n_frames / fps),
                out_path,
            ]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            print(
                "WARNING: ffmpeg not found (pip install imageio-ffmpeg). "
                "Output will have no audio."
            )
            self.writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )

    def write(self, frame):
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.writer.write(frame)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError("ffmpeg failed")
        else:
            self.writer.release()


def draw_debug(frame, kept_c, dropped_c, carried):
    for p in carried:  # magenta: carried from neighbours
        cv2.polylines(frame, [p], True, (255, 0, 255), 1)
    for p, s in dropped_c:  # blue: dropped (weak, unsupported)
        cv2.polylines(frame, [p], True, (255, 0, 0), 2)
    for p, s, why in kept_c:  # red: kept hi, orange: kept by support
        col = (0, 0, 255) if why == "hi" else (0, 140, 255)
        cv2.polylines(frame, [p], True, col, 3)
        cv2.putText(
            frame,
            "%.2f" % s,
            tuple(int(v) for v in p[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            col,
            2,
        )


def render(path, out_path, frames, meta, stride, debug=False, log=print):
    support, k = policy_params(meta["fps"], stride)
    kept, dropped = hysteresis(frames, support)
    sink = Sink(out_path, path, meta["w"], meta["h"], meta["fps"], len(frames))
    t0 = time.time()
    try:
        for c, frame, mask, own, carried in iter_masks(path, kept, k):
            if debug:
                draw_debug(frame, kept[c], dropped[c], carried)
            else:
                pixelate_mask(frame, mask)
            sink.write(frame)
            if (c + 1) % 50 == 0:
                log(
                    "render %d/%d frames (%.1f fps)"
                    % (c + 1, len(kept), (c + 1) / max(1e-6, time.time() - t0))
                )
    finally:
        sink.close()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Blur all on-screen text in a video (PP-OCRv6 detection only)."
    )
    ap.add_argument("input")
    ap.add_argument("-o", "--output", help="default: output/<name>_blurred.mp4")
    ap.add_argument("--tier", choices=["tiny", "small", "medium"], default="medium")
    ap.add_argument("--device", default="gpu:0")
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="detect every Nth frame (speed knob; gaps are filled by "
        "temporal fusion). Default 1 = every frame.",
    )
    ap.add_argument(
        "--no-tiling",
        action="store_true",
        help="skip native-resolution tiles on >1080p video (faster, misses small text)",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="draw detections instead of blurring. red=kept, "
        "orange=kept via temporal support, blue=dropped, "
        "magenta=carried from neighbour frames",
    )
    ap.add_argument(
        "--redetect", action="store_true", help="ignore the cached detections"
    )
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.input))[0]
    out = args.output or os.path.join(
        "output", stem + ("_debug.mp4" if args.debug else "_blurred.mp4")
    )
    cache_path = os.path.join(
        os.path.dirname(os.path.abspath(out)), stem + ".dets.json"
    )

    meta = video_meta(args.input)
    tiling = not args.no_tiling
    key = cache_key(args.input, args.tier, args.stride, tiling)
    frames = None if args.redetect else load_cache(cache_path, key)
    if frames is None:
        print("Detecting text (cached afterwards in %s)" % cache_path)
        det = make_paddle_detector(args.tier, args.device, tiling)
        frames = detect_video(args.input, det, args.stride)
        save_cache(cache_path, key, frames)
    else:
        print("Using cached detections: %s" % cache_path)

    render(args.input, out, frames, meta, args.stride, debug=args.debug)
    print("Done: %s" % out)


if __name__ == "__main__":
    sys.exit(main())
