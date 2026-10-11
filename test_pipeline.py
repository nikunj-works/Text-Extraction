#!/usr/bin/env python3
"""
GPU-free regression tests for blur_text.py. Run:  python test_pipeline.py

These test OUR logic (tiling, coordinate mapping, hysteresis, temporal fusion,
flow carry, rendering, caching) with mock detectors that behave like a flaky
real one (random misses, jittered boxes, weak scores, false positives).
They do NOT measure PP-OCR accuracy; that needs real footage.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

import blur_text as bt


# ----------------------------------------------------------------------------
def rect_poly(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)


def coverage(gt_polys, mask):
    """Fraction of ground-truth polygon area covered by mask."""
    gt = np.zeros(mask.shape, np.uint8)
    for p in gt_polys:
        cv2.fillPoly(gt, [np.round(p).astype(np.int32)], 255)
    tot = int((gt > 0).sum())
    return 1.0 if tot == 0 else int(((gt > 0) & (mask > 0)).sum()) / tot


# ----------------------------------------------------------------------------
def test_tiles_cover_every_pixel():
    for w, h in [(3840, 2160), (2160, 3840), (7680, 4320), (2560, 1440), (3000, 1100)]:
        cov = np.zeros((h, w), np.uint8)
        boxes = bt.tile_boxes(w, h)
        for x, y, tw, th in boxes:
            assert x >= 0 and y >= 0 and x + tw <= w and y + th <= h, (
                w,
                h,
                x,
                y,
                tw,
                th,
            )
            cov[y : y + th, x : x + tw] += 1
        assert cov.min() >= 1, "uncovered pixels for %dx%d" % (w, h)
    xs = bt.tile_starts(3840, 2048, bt.TILE_OVERLAP)
    assert xs[1] - xs[0] <= 2048 - bt.TILE_OVERLAP
    print("ok  tiles cover every pixel, overlap >= %d px" % bt.TILE_OVERLAP)


def _threshold_run(calls):
    """Mock detector: finds bright rectangles. Records the image sizes it was given."""

    def run(img):
        calls.append(img.shape[:2])
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h >= 20:
                out.append((rect_poly(x, y, w, h), 0.9))
        return out

    return run


def test_multiscale_scales_and_coordinates():
    # 1080p: exactly one native pass, no resize
    calls = []
    det = bt.MultiScaleDetector(_threshold_run(calls))
    det.detect(np.full((1080, 1920, 3), 100, np.uint8))
    assert calls == [(1080, 1920)], calls

    # 4K: one global pass (long side 2048) + native tiles
    calls = []
    det = bt.MultiScaleDetector(_threshold_run(calls))
    frame = np.full((2160, 3840, 3), 100, np.uint8)
    gts = [
        rect_poly(100, 100, 200, 40),
        rect_poly(1800, 500, 160, 24),  # inside the x-overlap zone
        rect_poly(1900, 1140, 300, 30),  # straddles tile 0's bottom edge
        rect_poly(1500, 1500, 800, 300),  # wider than overlap: cut by tiles
        rect_poly(3600, 2100, 200, 40),
    ]  # bottom-right corner
    for g in gts:
        x, y = g[0].astype(int)
        w, h = (g[2] - g[0]).astype(int)
        frame[y : y + h, x : x + w] = 255
    dets = det.detect(frame)
    assert calls[0][1] == 2048 and len(calls) == 1 + len(bt.tile_boxes(3840, 2160)), (
        calls
    )
    mask = np.zeros((2160, 3840), np.uint8)
    for p, _ in dets:
        cv2.fillPoly(mask, [np.round(p).astype(np.int32)], 255)
    for g in gts:
        c = coverage([g], mask)
        assert c >= 0.98, "coordinate mapping/coverage failed for %s: %.3f" % (g[0], c)
    print(
        "ok  multi-scale: 1080p=1 pass, 4K=%d passes, all 5 targets mapped back (>=98%%)"
        % len(calls)
    )


def test_no_tiling_flag():
    calls = []
    det = bt.MultiScaleDetector(_threshold_run(calls), tiling=False)
    det.detect(np.full((2160, 3840, 3), 100, np.uint8))
    assert len(calls) == 1
    print("ok  --no-tiling runs a single pass")


# ----------------------------------------------------------------------------
def test_hysteresis():
    P = lambda x: np.array([[x, 0], [x + 50, 0], [x + 50, 20], [x, 20]], dtype=np.int32)
    frames = [[] for _ in range(12)]
    for t in (3, 4, 5):  # weak but persistent -> keep
        frames[t].append((P(100), 0.40))
    frames[9].append((P(500), 0.40))  # weak and isolated -> drop
    frames[10].append((P(900), 0.95))  # strong and isolated -> keep
    kept, dropped = bt.hysteresis(frames, support=2)
    assert all(any(np.array_equal(p, P(100)) for p, _, _ in kept[t]) for t in (3, 4, 5))
    assert not kept[9] and len(dropped[9]) == 1
    assert len(kept[10]) == 1 and kept[10][0][2] == "hi"
    print("ok  hysteresis: persistent-weak kept, isolated-weak dropped, strong kept")


def test_margin_is_proportional():
    m_small = bt.build_mask(
        (400, 800), [np.round(rect_poly(100, 100, 100, 20)).astype(np.int32)]
    )
    m_big = bt.build_mask(
        (400, 800), [np.round(rect_poly(100, 100, 400, 100)).astype(np.int32)]
    )
    ys, xs = np.where(m_small > 0)
    grow_small = (xs.max() - xs.min() + 1) - 100
    ys, xs = np.where(m_big > 0)
    grow_big = (xs.max() - xs.min() + 1) - 400
    assert grow_big > 2 * grow_small, (grow_small, grow_big)
    print(
        "ok  margin scales with text height (small +%dpx, large +%dpx)"
        % (grow_small, grow_big)
    )


# ----------------------------------------------------------------------------
W, H, FPS, N = 1280, 720, 25, 60


def scene_objects(t):
    """Ground-truth text rectangles: (id, x, y, w, h)."""
    return [
        ("moving_taxi_text", 100 + 7 * t, 120 + 2 * t, 260, 50),
        ("static_subtitle", 300, 600, 600, 60),
        ("fast_left", 900 - 9 * t, 400 + 1 * t, 240, 46),
    ]


def make_synthetic_video(path):
    rng = np.random.default_rng(0)
    canvas = rng.integers(60, 200, (H + 40, W + 400), dtype=np.uint8)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 2.5)
    canvas = cv2.normalize(canvas, None, 40, 215, cv2.NORM_MINMAX)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for t in range(N):
        off = 4 * t  # camera pan
        bg = canvas[0:H, off : off + W]
        frame = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        for name, x, y, w, h in scene_objects(t):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 200, 240), -1)
            cv2.putText(
                frame,
                "ACCT 4821-77",
                (x + 8, y + h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                h / 55.0,
                (20, 20, 20),
                2,
            )
        vw.write(frame)
    vw.release()


class FlakyDetector:
    """Behaves like an imperfect real detector."""

    def __init__(self, drop_p, fp_p, seed=1, stride=1):
        self.rng = np.random.default_rng(seed)
        self.drop_p, self.fp_p, self.t, self.stride = drop_p, fp_p, 0, stride

    def detect(self, frame):
        t, self.t = self.t * self.stride, self.t + 1  # real frame index
        out = []
        for name, x, y, w, h in scene_objects(t):
            if self.rng.random() < self.drop_p:
                continue  # missed this frame
            q = rect_poly(x, y, w, h) + self.rng.normal(0, 1.5, (4, 2))
            score = float(self.rng.uniform(0.35, 0.95))  # some are weak
            out.append((q, score))
        if self.rng.random() < self.fp_p:  # one-frame weak false positive
            fx, fy = self.rng.integers(0, W - 80), self.rng.integers(0, H - 40)
            out.append((rect_poly(fx, fy, 80, 30), float(self.rng.uniform(0.3, 0.5))))
        return out


def temporal_recall(video, drop_p, fp_p, seed=1, stride=1):
    meta = bt.video_meta(video)
    det = FlakyDetector(drop_p, fp_p, seed, stride)
    frames = bt.detect_video(video, det, stride=stride, log=lambda *_: None)
    support, k = bt.policy_params(meta["fps"], stride)
    kept, dropped = bt.hysteresis(frames, support)
    recs, fp_area = [], []
    for c, frame, mask, own, carried in bt.iter_masks(video, kept, k):
        gts = [rect_poly(x, y, w, h) for _, x, y, w, h in scene_objects(c)]
        per = [coverage([g], mask) for g in gts]
        recs.append(per)
        gt_all = np.zeros(mask.shape, np.uint8)
        for g in gts:
            cv2.fillPoly(gt_all, [np.round(g).astype(np.int32)], 255)
        fp_area.append(int(((mask > 0) & (gt_all == 0)).sum()) / float(mask.size))
    recs = np.array(recs)
    return recs, float(np.mean(fp_area))


def test_temporal_fusion_recall(video):
    recs, fp = temporal_recall(video, drop_p=0.35, fp_p=0.3)
    mean_rec, min_rec = recs.mean(), recs.min()
    per_obj = recs.mean(axis=0)
    print("    detector misses 35% of boxes/frame, 30% of frames get a false positive")
    print(
        "    recall mean=%.4f  worst frame/object=%.3f  per object=%s  blurred non-text area=%.2f%% of frame"
        % (mean_rec, min_rec, np.round(per_obj, 4), fp * 100)
    )
    assert mean_rec >= 0.99, "temporal recall too low: %.4f" % mean_rec
    assert (per_obj >= 0.985).all(), per_obj
    print("ok  temporal fusion + flow keeps recall >= 99% despite a 35%-miss detector")


def test_stress_report(video):
    print("    stress (informational): recall vs. detector miss-rate")
    for p in (0.2, 0.35, 0.5, 0.65):
        rs = [
            temporal_recall(video, drop_p=p, fp_p=0.3, seed=s)[0] for s in range(10, 16)
        ]
        print(
            "      miss=%2d%%  recall=%.4f  (avg of 6 runs)  worst frame/object=%.3f"
            % (p * 100, np.mean([r.mean() for r in rs]), min(r.min() for r in rs))
        )


def test_static_overlay_over_fast_pan(tmp):
    """Burned-in caption (fixed on screen) over a sharply textured, fast-panning
    background. Flow-only carry drifts here; the un-moved copy must cover it."""
    rng = np.random.default_rng(5)
    canvas = cv2.GaussianBlur(
        rng.integers(0, 255, (H + 40, W + 900), dtype=np.uint8), (0, 0), 0.8
    )
    canvas = cv2.normalize(canvas, None, 20, 235, cv2.NORM_MINMAX)
    path = os.path.join(tmp, "overlay.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    gt = (200, 560, 200, 30)
    for t in range(N):
        off = (25 * t) % 800  # 25 px/frame pan
        fr = cv2.cvtColor(canvas[0:H, off : off + W], cv2.COLOR_GRAY2BGR)
        for col, th in (((0, 0, 0), 6), ((255, 255, 255), 2)):
            cv2.putText(
                fr,
                "4111 2222 3333",
                (gt[0], gt[1] + gt[3] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                col,
                th,
            )
        vw.write(fr)
    vw.release()

    class Det:
        def __init__(s, seed):
            s.r = np.random.default_rng(seed)

        def detect(s, frame):
            if s.r.random() < 0.35:
                return []
            return [(rect_poly(*gt) + s.r.normal(0, 1.5, (4, 2)), 0.85)]

    covs = []
    for seed in range(1, 7):
        meta = bt.video_meta(path)
        frames = bt.detect_video(path, Det(seed), log=lambda *_: None)
        support, k = bt.policy_params(meta["fps"], 1)
        kept, _ = bt.hysteresis(frames, support)
        covs += [
            coverage([rect_poly(*gt)], m)
            for _, _, m, _, _ in bt.iter_masks(path, kept, k)
        ]
    covs = np.array(covs)
    print(
        "    static caption, 25 px/frame pan, detector misses 35%%: recall=%.4f worst frame=%.3f"
        % (covs.mean(), covs.min())
    )
    assert covs.mean() >= 0.999 and covs.min() >= 0.99, (covs.mean(), covs.min())
    print("ok  static overlay stays covered over a fast pan")


def test_stride_fills_gaps(video):
    recs, _ = temporal_recall(video, drop_p=0.1, fp_p=0.0, stride=3)
    print(
        "    stride=3 (detect every 3rd frame): recall=%.4f worst=%.3f"
        % (recs.mean(), recs.min())
    )
    assert recs.mean() >= 0.98
    print("ok  --stride 3 still keeps recall >= 98%")


# ----------------------------------------------------------------------------
def test_end_to_end_render_with_audio(video, tmp):
    ff = bt.find_ffmpeg()
    if not ff:
        print("skip end-to-end render (no ffmpeg)")
        return
    src = os.path.join(tmp, "with_audio.mp4")
    subprocess.run(
        [
            ff,
            "-y",
            "-loglevel",
            "error",
            "-i",
            video,
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            src,
        ],
        check=True,
    )
    meta = bt.video_meta(src)
    frames = bt.detect_video(src, FlakyDetector(0.2, 0.2), log=lambda *_: None)
    out = os.path.join(tmp, "out.mp4")
    bt.render(src, out, frames, meta, 1, log=lambda *_: None)
    cap = cv2.VideoCapture(out)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert (int(cap.get(3)), int(cap.get(4))) == (W, H) and n == N, (n,)
    ok, first = cap.read()
    cap.release()
    # text region of frame 0 must have changed (pixelated)
    cap0 = cv2.VideoCapture(src)
    _, orig = cap0.read()
    cap0.release()
    _, x, y, w, h = scene_objects(0)[1]
    diff = np.abs(
        first[y : y + h, x : x + w].astype(int) - orig[y : y + h, x : x + w].astype(int)
    ).mean()
    assert diff > 3, "text region not altered (diff %.2f)" % diff
    probe = subprocess.run(
        [
            ff.replace("ffmpeg", "ffprobe") if "ffmpeg" in ff else "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            out,
        ],
        capture_output=True,
        text=True,
    )
    kinds = probe.stdout.split()
    if probe.returncode == 0:
        assert "audio" in kinds and "video" in kinds, kinds
    print(
        "ok  end-to-end render: %d frames, %dx%d, text pixelated, audio kept%s"
        % (
            n,
            W,
            H,
            "" if probe.returncode == 0 else " (ffprobe unavailable, audio unchecked)",
        )
    )


def test_cache_roundtrip(video, tmp):
    key = bt.cache_key(video, "medium", 1, True)
    frames = [[(np.array([[1, 2], [3, 4], [5, 6]], np.int32), 0.7)], None, []]
    path = os.path.join(tmp, "c.json")
    bt.save_cache(path, key, frames)
    back = bt.load_cache(path, key)
    assert (
        back[1] is None
        and back[2] == []
        and np.array_equal(back[0][0][0], frames[0][0][0])
    )
    assert bt.load_cache(path, bt.cache_key(video, "small", 1, True)) is None
    print("ok  cache round-trips and is invalidated when settings change")


def test_cli_flow_cache_and_debug(video, tmp):
    """main() end to end with the Paddle detector swapped for a mock."""
    import contextlib
    import io

    calls = {"n": 0}

    def fake_make(tier, device, tiling):
        calls["n"] += 1
        return FlakyDetector(0.2, 0.1)

    real = bt.make_paddle_detector
    bt.make_paddle_detector = fake_make
    out = os.path.join(tmp, "cli", "clip_blurred.mp4")
    dbg = os.path.join(tmp, "cli", "clip_debug.mp4")
    try:

        def run(argv):
            old = sys.argv
            sys.argv = ["blur_text.py"] + argv
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    bt.main()
            finally:
                sys.argv = old
            return buf.getvalue()

        clip = os.path.join(tmp, "clip.mp4")
        shutil.copy(video, clip)
        log1 = run([clip, "-o", out])
        assert os.path.exists(out) and calls["n"] == 1 and "Detecting text" in log1
        assert os.path.exists(os.path.join(tmp, "cli", "clip.dets.json"))
        log2 = run([clip, "-o", dbg, "--debug"])  # reuses detections, no GPU
        assert (
            calls["n"] == 1
            and "Using cached detections" in log2
            and os.path.exists(dbg)
        )
        run([clip, "-o", out, "--redetect"])
        assert calls["n"] == 2
    finally:
        bt.make_paddle_detector = real
    print(
        "ok  CLI: detect once, cache reused for --debug re-render, --redetect forces GPU stage"
    )


def test_degenerate_guard(video):
    class Huge:
        def detect(self, frame):
            h, w = frame.shape[:2]
            return [(rect_poly(0, 0, w, h), 0.99), (rect_poly(10, 10, 100, 30), 0.99)]

    frames = bt.detect_video(video, Huge(), log=lambda *_: None)
    assert all(len(f) == 1 for f in frames)
    print("ok  catastrophic guard drops a frame-sized 'text' box, keeps the real one")


# ----------------------------------------------------------------------------
def main():
    tmp = tempfile.mkdtemp()
    try:
        video = os.path.join(tmp, "synthetic.mp4")
        make_synthetic_video(video)
        test_tiles_cover_every_pixel()
        test_multiscale_scales_and_coordinates()
        test_no_tiling_flag()
        test_hysteresis()
        test_margin_is_proportional()
        test_temporal_fusion_recall(video)
        test_static_overlay_over_fast_pan(tmp)
        test_stride_fills_gaps(video)
        test_stress_report(video)
        test_degenerate_guard(video)
        test_cache_roundtrip(video, tmp)
        test_cli_flow_cache_and_debug(video, tmp)
        test_end_to_end_render_with_audio(video, tmp)
        print("\nALL TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
