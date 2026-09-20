#!/usr/bin/env python3
"""Generate the map-drone synthetic training data end-to-end.

Reproduces the full pipeline that creates the map-drone-data dataset:
  1. Extract object sprites from the Helsinki reference scene
  2. Fetch aerial basemap tiles from Esri World Imagery (cached)
  3. Generate simulated drone survey scenes over Nordic terrain
  4. Export YOLO-format 960x540 view crops with train/val split

Requires the Nordic AI Cup competition repo with the Helsinki reference scene.

Usage:
    # Set the path to the competition repo's helsinki scene
    export NAIC_HELSINKI=/path/to/Nordic-AI-Cup-2026/drone-flyby/src/helsinki

    # Run the full pipeline
    python crawl_map_drone_data.py --naic-helsinki $NAIC_HELSINKI

    # Run individual steps
    python crawl_map_drone_data.py --naic-helsinki $NAIC_HELSINKI --step sprites
    python crawl_map_drone_data.py --naic-helsinki $NAIC_HELSINKI --step basemaps
    python crawl_map_drone_data.py --naic-helsinki $NAIC_HELSINKI --step scenes
    python crawl_map_drone_data.py --naic-helsinki $NAIC_HELSINKI --step export

    # Custom settings
    python crawl_map_drone_data.py \
        --naic-helsinki $NAIC_HELSINKI \
        --output map-drone-data \
        --workers 8 \
        --per-class 2 \
        --basemap-source esri \
        --tile-cache /tmp/tilecache
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import yaml
from PIL import Image

# ---------------------------------------------------------------- constants --

OBJECT_CLASSES = (
    "hangar", "helicopter", "jet_plane", "large_launcher", "large_tower",
    "medium_launcher", "medium_plane", "mine_roller", "small_launcher",
    "small_plane", "small_tower", "ta-ta", "tank", "condor", "jammer",
    "spacecraft",
)

SRC_W, SRC_H = 3840, 2160
ALTITUDE_M = 600
STEP_M = 13.8888889
PX_PER_M = 4.6033
GSD_M_PER_PX = 1.0 / PX_PER_M
FOCAL_PX = PX_PER_M * ALTITUDE_M
FOOTPRINT_M = (SRC_W * GSD_M_PER_PX, SRC_H * GSD_M_PER_PX)
STEP_PX = STEP_M * PX_PER_M
TILE_SIZE = 256
EARTH_C = 40075016.685578488
VIEW_W, VIEW_H = 960, 540
REGION = {0: (3840, 2160), 1: (1920, 1080), 2: (960, 540)}
MIN_VISIBLE_PX = 3
SPRITE_MARGIN = 8
SPRITE_UP = 4

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Default sites: rural/peri-urban Nordic locations resembling Helsinki reference
DEFAULT_SITES = [
    {"name": "fi_espoo_nuuksio", "country": "FI", "terrain": "forest + lakes", "lon": 24.51, "lat": 60.3, "bearing": 12, "frames": 60},
    {"name": "fi_sipoo_coast", "country": "FI", "terrain": "coastal farmland", "lon": 25.29, "lat": 60.26, "bearing": 168, "frames": 60},
    {"name": "fi_tuusula_fields", "country": "FI", "terrain": "farmland + lake", "lon": 25.03, "lat": 60.42, "bearing": 340, "frames": 60},
    {"name": "fi_vihti_hiidenvesi", "country": "FI", "terrain": "lake shore + forest", "lon": 24.4, "lat": 60.37, "bearing": 75, "frames": 60},
    {"name": "fi_lieto_farmland", "country": "FI", "terrain": "flat farmland", "lon": 22.44, "lat": 60.53, "bearing": 195, "frames": 60},
    {"name": "se_jarfalla_malaren", "country": "SE", "terrain": "lake shore + suburb", "lon": 17.84, "lat": 59.42, "bearing": 55, "frames": 60},
    {"name": "se_tyresta_forest", "country": "SE", "terrain": "forest + lake", "lon": 18.25, "lat": 59.19, "bearing": 135, "frames": 60},
    {"name": "se_norrkoping_fields", "country": "SE", "terrain": "open farmland", "lon": 16.17, "lat": 58.6, "bearing": 280, "frames": 60},
    {"name": "se_karlstad_klaralven", "country": "SE", "terrain": "river + farmland", "lon": 13.5, "lat": 59.4, "bearing": 350, "frames": 60},
    {"name": "se_gotland_faro", "country": "SE", "terrain": "coast + scrub", "lon": 19.15, "lat": 57.95, "bearing": 90, "frames": 60},
    {"name": "dk_amager_faelled", "country": "DK", "terrain": "reclaimed land", "lon": 12.63, "lat": 55.64, "bearing": 15, "frames": 60},
    {"name": "dk_lejre_farmland", "country": "DK", "terrain": "farmland + forest", "lon": 11.97, "lat": 55.6, "bearing": 225, "frames": 60},
    {"name": "dk_thy_dunes", "country": "DK", "terrain": "coastal dunes", "lon": 8.43, "lat": 56.97, "bearing": 310, "frames": 60},
    {"name": "dk_gribskov_forest", "country": "DK", "terrain": "dense forest", "lon": 12.35, "lat": 55.97, "bearing": 155, "frames": 60},
    {"name": "dk_bornholm_coast", "country": "DK", "terrain": "rocky coast + fields", "lon": 14.75, "lat": 55.18, "bearing": 40, "frames": 60},
    {"name": "no_asker_forest", "country": "NO", "terrain": "forest + suburb", "lon": 10.4, "lat": 59.85, "bearing": 70, "frames": 60},
    {"name": "no_nesodden_coast", "country": "NO", "terrain": "fjord coast", "lon": 10.65, "lat": 59.87, "bearing": 200, "frames": 60},
    {"name": "no_ringsaker_fields", "country": "NO", "terrain": "inland farmland", "lon": 10.82, "lat": 60.87, "bearing": 120, "frames": 60},
    {"name": "no_tromso_fjord", "country": "NO", "terrain": "fjord + birch", "lon": 19.02, "lat": 69.65, "bearing": 260, "frames": 60},
    {"name": "no_lofoten_village", "country": "NO", "terrain": "fishing village", "lon": 14.6, "lat": 68.2, "bearing": 330, "frames": 60},
    {"name": "fi_hyvinkaa_fields", "country": "FI", "terrain": "open farmland", "lon": 24.88, "lat": 60.63, "bearing": 242, "frames": 60},
    {"name": "se_vasteras_farmland", "country": "SE", "terrain": "flat farmland + lake", "lon": 16.55, "lat": 59.6, "bearing": 105, "frames": 60},
]

UA = ("NordicAICup-drone-flyby-dataset-builder/1.0 "
      "(offline training-data generation)")

BASEMAP_SOURCES = {
    "esri": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery"
        "/MapServer/tile/{z}/{y}/{x}", 18,
        "Esri World Imagery", False),
}


# ---------------------------------------------------------------- geometry --

def mercator_gsd(lat_deg: float, zoom: int) -> float:
    return EARTH_C * math.cos(math.radians(lat_deg)) / (TILE_SIZE * 2 ** zoom)


def lonlat_to_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    n = TILE_SIZE * 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    return x, y


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


# ----------------------------------------------------------- sprite extraction --

def _grabcut(big, bx):
    x0, y0, x1, y1 = bx
    m = np.full(big.shape[:2], cv2.GC_BGD, np.uint8)
    m[y0:y1, x0:x1] = cv2.GC_PR_FGD
    bw, bh = x1 - x0, y1 - y0
    cw, ch = max(1, bw // 5), max(1, bh // 5)
    for cy, cx in ((y0, x0), (y0, x1 - cw), (y1 - ch, x0), (y1 - ch, x1 - cw)):
        m[cy:cy + ch, cx:cx + cw] = cv2.GC_PR_BGD
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    cv2.ellipse(m, (cx, cy), (max(1, bw // 6), max(1, bh // 6)), 0, 0, 360, cv2.GC_FGD, -1)
    try:
        cv2.grabCut(big, m, None, np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.zeros(big.shape[:2], np.uint8)
    fg = np.where((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    n, l, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        p = l[cy, cx] or 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        fg = np.where(l == p, 255, 0).astype(np.uint8)
    return fg


def _cluster(big, bx, k, keep_rel):
    x0, y0, x1, y1 = bx
    lab = cv2.cvtColor(cv2.GaussianBlur(big, (0, 0), 0.8 * SPRITE_UP), cv2.COLOR_BGR2Lab).astype(np.float32)
    ring = np.ones(big.shape[:2], bool)
    ring[y0:y1, x0:x1] = False
    mu, sd = lab[ring].mean(0), lab[ring].std(0) + 1e-3
    bh, bw = y1 - y0, x1 - x0
    inside = lab[y0:y1, x0:x1].reshape(-1, 3)
    k = int(max(2, min(k, inside.shape[0] // 50)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, lbl, cen = cv2.kmeans(inside, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.reshape(bh, bw)
    yy, xx = np.mgrid[0:bh, 0:bw]
    core = np.hypot((yy - bh / 2) / (bh / 2), (xx - bw / 2) / (bw / 2)) < 0.40
    sc = []
    for c in range(k):
        m = lbl == c
        if m.sum() < 8:
            sc.append(-1.0)
            continue
        cov = (m & core).sum() / max(core.sum(), 1)
        dist = float(np.linalg.norm((cen[c] - mu) / sd))
        frac = float(m.mean())
        border = (m[0].mean() + m[-1].mean() + m[:, 0].mean() + m[:, -1].mean()) / 4
        sc.append(cov * (0.5 + dist) * (1 - 0.75 * frac) * (1 - 0.6 * border))
    sc = np.array(sc)
    if sc.max() <= 0:
        return None
    fg = np.isin(lbl, [c for c in range(k) if sc[c] >= keep_rel * sc.max()]).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((SPRITE_UP, SPRITE_UP), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((2 * SPRITE_UP, 2 * SPRITE_UP), np.uint8))
    n, l, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        pick = l[bh // 2, bw // 2] or 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        main_area = st[pick, cv2.CC_STAT_AREA]
        px = st[pick, cv2.CC_STAT_LEFT] + st[pick, cv2.CC_STAT_WIDTH] / 2
        py = st[pick, cv2.CC_STAT_TOP] + st[pick, cv2.CC_STAT_HEIGHT] / 2
        keep = {pick}
        for c in range(1, n):
            if c == pick:
                continue
            qx = st[c, cv2.CC_STAT_LEFT] + st[c, cv2.CC_STAT_WIDTH] / 2
            qy = st[c, cv2.CC_STAT_TOP] + st[c, cv2.CC_STAT_HEIGHT] / 2
            if st[c, cv2.CC_STAT_AREA] > 0.18 * main_area and \
               math.hypot(qx - px, qy - py) < 0.55 * max(bw, bh):
                keep.add(c)
        fg = np.where(np.isin(l, list(keep)), 255, 0).astype(np.uint8)
    ff = fg.copy()
    m2 = np.zeros((bh + 2, bw + 2), np.uint8)
    cv2.floodFill(ff, m2, (0, 0), 255)
    fg = fg | cv2.bitwise_not(ff)
    out = np.zeros(big.shape[:2], np.uint8)
    out[y0:y1, x0:x1] = fg
    return out


def _plausibility(fg_box):
    a = fg_box.astype(np.float32) / 255.0
    fill = float(a.mean())
    if fill < 0.05 or fill > 0.88:
        return -1.0
    bh, bw = a.shape
    ys, xs = np.nonzero(fg_box)
    if len(xs) == 0:
        return -1.0
    centred = 1 - min(1.0, 2.2 * math.hypot(ys.mean() / bh - 0.5, xs.mean() / bw - 0.5))
    reach = min(1.0, (xs.min() < 0.12 * bw) + (xs.max() > 0.88 * bw) +
                     (ys.min() < 0.12 * bh) + (ys.max() > 0.88 * bh))
    band = max(0.0, 1 - abs(fill - 0.34) / 0.54)
    border = (a[0].mean() + a[-1].mean() + a[:, 0].mean() + a[:, -1].mean()) / 4
    return band * centred * (1 - 0.8 * max(0.0, border - 0.35) / 0.65) * (0.6 + 0.4 * reach)


def sprite_mattes(patch, box):
    h, w = patch.shape[:2]
    big = cv2.resize(patch, (w * SPRITE_UP, h * SPRITE_UP), interpolation=cv2.INTER_LANCZOS4)
    bx = tuple(v * SPRITE_UP for v in box)
    x0, y0, x1, y1 = bx

    best, best_s = None, -1.0
    for k in (4, 5, 6, 3):
        for keep_rel in (0.80, 0.62, 0.45):
            fg = _cluster(big, bx, k, keep_rel)
            if fg is None:
                continue
            s = _plausibility(fg[y0:y1, x0:x1])
            if s > best_s:
                best, best_s = fg, s
    gc = _grabcut(big, bx)
    tight = best if best is not None and best_s > 0 else gc
    generous = np.maximum(gc, tight)
    generous = cv2.dilate((generous > 127).astype(np.uint8) * 255, np.ones((SPRITE_UP, SPRITE_UP), np.uint8))

    def down(m, lo, span):
        a = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        return cv2.GaussianBlur(np.clip((a - lo) / span, 0, 1), (5, 5), 1.0)

    return down(generous, 0.22, 0.55), down(tight, 0.32, 0.50)


def extract_sprites(helsinki_dir: Path, out_dir: Path, per_class: int = 10):
    print(f"\n{'='*60}")
    print("Step 1: Extracting sprites from Helsinki reference scene")
    print(f"{'='*60}")

    docs = {}
    for path in sorted(glob.glob(str(helsinki_dir / "annotations" / "*.json"))):
        with open(path) as f:
            d = json.load(f)
        docs[d["frame"]] = d
    print(f"  Reading {helsinki_dir} ({len(docs)} frames)")

    cands: dict[str, list] = {}
    for fr, doc in docs.items():
        for a in doc["annotations"]:
            b = a["bbox"]
            if (b[0] <= SPRITE_MARGIN or b[1] <= SPRITE_MARGIN
                    or b[2] >= SRC_W - SPRITE_MARGIN or b[3] >= SRC_H - SPRITE_MARGIN):
                continue
            cands.setdefault(a["object_id"], []).append(
                (-(b[2] - b[0]) * (b[3] - b[1]), fr, b))
    for v in cands.values():
        v.sort()

    index, cache = {}, {}
    for cls in OBJECT_CLASSES:
        outdir = out_dir / cls
        outdir.mkdir(parents=True, exist_ok=True)
        kept = []
        for _, fr, b in cands.get(cls, []):
            if len(kept) >= per_class:
                break
            if fr not in cache:
                cache[fr] = cv2.imread(str(helsinki_dir / "images" / f"frame_{fr:06d}.png"),
                                       cv2.IMREAD_COLOR)
            img = cache[fr]
            bw, bh = b[2] - b[0], b[3] - b[1]
            pad = max(6, int(0.40 * max(bw, bh)))
            px0, py0 = max(0, b[0] - pad), max(0, b[1] - pad)
            px1, py1 = min(SRC_W, b[2] + pad), min(SRC_H, b[3] + pad)
            patch = img[py0:py1, px0:px1]
            box = (b[0] - px0, b[1] - py0, b[2] - px0, b[3] - py0)
            gen, tight = sprite_mattes(patch, box)

            sl = (slice(box[1], box[3]), slice(box[0], box[2]))
            a_gen, a_tight = gen[sl], tight[sl]
            rgb = patch[sl]
            fill = float(a_gen.mean())
            if not (0.06 <= fill <= 0.97):
                continue

            name = f"{cls}_f{fr:02d}"
            cv2.imwrite(str(outdir / f"{name}.png"),
                        np.dstack([rgb, (a_gen * 255).astype(np.uint8)]),
                        [cv2.IMWRITE_PNG_COMPRESSION, 6])
            cv2.imwrite(str(outdir / f"{name}_core.png"),
                        (a_tight * 255).astype(np.uint8),
                        [cv2.IMWRITE_PNG_COMPRESSION, 6])

            kept.append({
                "file": f"{cls}/{name}.png",
                "core": f"{cls}/{name}_core.png",
                "source_frame": fr,
                "source_bbox": b,
                "w": int(bw), "h": int(bh),
                "alpha_fill": round(fill, 3),
            })
        index[cls] = kept
        print(f"  {cls:16s} {len(kept):2d} sprites")

    write_json(out_dir / "sprites_index.json", {
        "source_scene": str(helsinki_dir),
        "classes": index,
    })
    total = sum(len(v) for v in index.values())
    print(f"  {total} sprites -> {out_dir}")
    missing = [c for c, v in index.items() if not v]
    if missing:
        print(f"  WARNING: no sprites for: {', '.join(missing)}")
    return out_dir


# ----------------------------------------------------------- basemap fetching --

_throttle = threading.Semaphore(8)
_last = [0.0]
_lock = threading.Lock()


def _get_tile(url: str, cache: Path, tries: int = 4) -> bytes | None:
    if cache.exists() and cache.stat().st_size > 0:
        return cache.read_bytes()
    for attempt in range(tries):
        try:
            with _throttle:
                with _lock:
                    gap = time.time() - _last[0]
                    if gap < 0.012:
                        time.sleep(0.012 - gap)
                    _last[0] = time.time()
                raw = urlopen(Request(url, headers={"User-Agent": UA}), timeout=40).read()
            if len(raw) < 200:
                return None
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(raw)
            return raw
        except HTTPError as e:
            if e.code in (404, 400):
                return None
            time.sleep(1.5 * (attempt + 1))
        except (URLError, OSError, TimeoutError):
            time.sleep(1.5 * (attempt + 1))
    return None


def strip_extent_m(n_frames: int, bearing_deg: float, margin_m: float = 60.0):
    along = FOOTPRINT_M[1] + (n_frames - 1) * STEP_M
    across = FOOTPRINT_M[0]
    t = math.radians(bearing_deg)
    w = abs(across * math.cos(t)) + abs(along * math.sin(t))
    h = abs(across * math.sin(t)) + abs(along * math.cos(t))
    return w + 2 * margin_m, h + 2 * margin_m


def fetch_basemaps(sites: list[dict], source: str, out_dir: Path,
                   cache_dir: Path, api_key: str | None = None):
    print(f"\n{'='*60}")
    print("Step 2: Fetching aerial basemaps")
    print(f"{'='*60}")

    tmpl, zoom, attribution, needs_key = BASEMAP_SOURCES[source]
    if needs_key and not api_key:
        raise SystemExit(f"Source {source} needs --api-key")

    print(f"  {len(sites)} sites from {source} @z{zoom}")
    kept = []

    for site in sites:
        lon, lat = site["lon"], site["lat"]
        w_m, h_m = strip_extent_m(site["frames"], site["bearing"])
        gsd = mercator_gsd(lat, zoom)
        cx, cy = lonlat_to_pixel(lon, lat, zoom)
        half_w, half_h = (w_m / gsd) / 2, (h_m / gsd) / 2

        tx0 = int((cx - half_w) // TILE_SIZE)
        ty0 = int((cy - half_h) // TILE_SIZE)
        tx1 = int((cx + half_w) // TILE_SIZE)
        ty1 = int((cy + half_h) // TILE_SIZE)
        nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
        print(f"  {site['name']:<26s} {nx}x{ny} tiles", end="", flush=True)

        mosaic = np.zeros((ny * TILE_SIZE, nx * TILE_SIZE, 3), np.uint8)
        missing = 0

        def one(i, j):
            tx, ty = tx0 + i, ty0 + j
            url = tmpl.format(z=zoom, x=tx, y=ty, key=api_key or "")
            raw = _get_tile(url, cache_dir / source / str(zoom) / str(tx) / f"{ty}.img")
            if raw is None:
                return i, j, None
            try:
                return i, j, np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
            except Exception:
                return i, j, None

        with ThreadPoolExecutor(8) as ex:
            for i, j, arr in ex.map(lambda ij: one(*ij),
                                    [(i, j) for i in range(nx) for j in range(ny)]):
                if arr is None:
                    missing += 1
                    continue
                mosaic[j * TILE_SIZE:(j + 1) * TILE_SIZE,
                       i * TILE_SIZE:(i + 1) * TILE_SIZE] = arr[:TILE_SIZE, :TILE_SIZE]

        total = nx * ny
        if missing > 0.08 * total:
            print(f" SKIP ({missing}/{total} tiles missing)")
            continue

        bgr = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        detail = float(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 2.0)).mean())
        if detail < 1.2:
            print(f" SKIP (no detail)")
            continue

        d = out_dir / site["name"]
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "mosaic.png"), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        meta = {
            "name": site["name"],
            "centre_lonlat": [lon, lat],
            "bearing_deg": site["bearing"],
            "frames": site["frames"],
            "native_gsd_m_per_px": gsd,
        }
        write_json(d / "meta.json", meta)
        kept.append(meta)
        print(f" ok (detail={detail:.1f})")

    write_json(out_dir / "index.json", {"source": source, "zoom": zoom, "sites": kept})
    print(f"  {len(kept)}/{len(sites)} sites usable -> {out_dir}")
    return out_dir


# ----------------------------------------------------------- scene generation --

def build_strip(mosaic, native_gsd, bearing_deg):
    scale = native_gsd / GSD_M_PER_PX
    h, w = mosaic.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), bearing_deg, scale)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    strip = cv2.warpAffine(mosaic, M, (nw, nh), flags=cv2.INTER_LANCZOS4,
                           borderMode=cv2.BORDER_REFLECT_101)
    valid = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (nw, nh),
                           flags=cv2.INTER_NEAREST, borderValue=0)
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
    return strip, valid


def placement_weight(strip, valid):
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hf = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 3.0))
    texture = cv2.boxFilter(hf, -1, (25, 25))
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1].astype(np.int16), hsv[:, :, 2].astype(np.int16)
    water = ((texture < 2.0) & (val < 70)) | ((texture < 1.0) & (sat < 45))
    veto = (water | (val < 30)).astype(np.uint8)
    veto = cv2.dilate(veto, np.ones((31, 31), np.uint8))
    openness = np.exp(-(texture / 9.0) ** 2).astype(np.float32)
    w = openness * (1 - veto) * (valid > 0)
    return np.maximum(w, 0.03 * (1 - veto) * (valid > 0)).astype(np.float32)


def refine_alpha(rgb, alpha, core_f):
    solid = (core_f > 0.5).astype(np.uint8)
    if solid.sum() < 4:
        return alpha
    core = np.minimum(core_f, solid.astype(np.float32))
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.boxFilter(cv2.magnitude(gx, gy), -1, (5, 5))
    ref = float(np.percentile(grad[solid > 0], 55)) + 1e-3
    structure = np.clip(grad / ref, 0.0, 1.0)
    collar = cv2.dilate(solid, np.ones((3, 3), np.uint8)).astype(np.float32)
    weight = np.maximum(np.maximum(structure, 0.05), collar)
    out = np.maximum(core, alpha * weight)
    return cv2.GaussianBlur(out.astype(np.float32), (3, 3), 0.6)


def load_sprites(root):
    idx = json.loads((root / "sprites_index.json").read_text())["classes"]
    out = {}
    for cls, entries in idx.items():
        bank = []
        for e in entries:
            rgba = cv2.imread(str(root / e["file"]), cv2.IMREAD_UNCHANGED)
            core = cv2.imread(str(root / e["core"]), cv2.IMREAD_GRAYSCALE)
            if rgba is None or core is None or rgba.shape[2] != 4:
                continue
            alpha = rgba[:, :, 3].astype(np.float32) / 255.0
            core_f = core.astype(np.float32) / 255.0
            refined = refine_alpha(rgba[:, :, :3], alpha, core_f)
            bank.append({"rgb": rgba[:, :, :3], "alpha": refined, "core": core_f})
        if bank:
            out[cls] = bank
    return out


def rotate_sprite(sp, quarter_turns):
    k = quarter_turns % 4
    if k == 0:
        return sp
    return {
        "rgb": np.ascontiguousarray(np.rot90(sp["rgb"], k)),
        "alpha": np.ascontiguousarray(np.rot90(sp["alpha"], k)),
        "core": np.ascontiguousarray(np.rot90(sp["core"], k)),
    }


def harmonize(rgb, terrain_mask, dst_roi, strength=0.75):
    if terrain_mask.sum() < 12:
        return rgb
    src = cv2.cvtColor(rgb, cv2.COLOR_BGR2Lab).astype(np.float32)
    dst = cv2.cvtColor(dst_roi, cv2.COLOR_BGR2Lab).astype(np.float32).reshape(-1, 3)
    sm, ss = src[terrain_mask].mean(0), src[terrain_mask].std(0) + 1e-3
    dm, ds = dst.mean(0), dst.std(0) + 1e-3
    gain = np.clip(ds / ss, 0.6, 1.7)
    out = src * (1 + strength * (gain - 1)) + strength * (dm - sm * gain)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)


def paste(frame, sp, cx, cy):
    h, w = sp["alpha"].shape
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(SRC_W, x0 + w), min(SRC_H, y0 + h)
    if fx1 - fx0 < MIN_VISIBLE_PX or fy1 - fy0 < MIN_VISIBLE_PX:
        return None
    sx0, sy0 = fx0 - x0, fy0 - y0
    sx1, sy1 = sx0 + (fx1 - fx0), sy0 + (fy1 - fy0)
    a = sp["alpha"][sy0:sy1, sx0:sx1]
    rgb = sp["rgb"][sy0:sy1, sx0:sx1]
    core = sp["core"][sy0:sy1, sx0:sx1]
    roi = frame[fy0:fy1, fx0:fx1]
    rgb = harmonize(rgb, (core < 0.25) & (a > 0.03), roi, strength=0.85)
    a3 = a[:, :, None]
    frame[fy0:fy1, fx0:fx1] = (rgb.astype(np.float32) * a3
                               + roi.astype(np.float32) * (1 - a3)).astype(np.uint8)
    return [int(fx0), int(fy0), int(fx1), int(fy1)]


def generate_scenes(basemaps_dir: Path, sprites_dir: Path, scenes_dir: Path,
                    per_class: int = 2, workers: int = 8, seed: int = 20260918):
    print(f"\n{'='*60}")
    print("Step 3: Generating simulated drone scenes")
    print(f"{'='*60}")

    index = json.loads((basemaps_dir / "index.json").read_text())
    sites = index["sites"]
    if not sites:
        raise SystemExit("No basemaps - run step 2 first")

    sprites = load_sprites(sprites_dir)
    print(f"  {len(sites)} sites, {sum(len(v) for v in sprites.values())} sprites loaded")

    total_boxes = 0
    for i, site in enumerate(sites):
        meta = json.loads((basemaps_dir / site["name"] / "meta.json").read_text())
        mosaic = cv2.imread(str(basemaps_dir / site["name"] / "mosaic.png"), cv2.IMREAD_COLOR)
        if mosaic is None:
            print(f"  {site['name']}: SKIP (no mosaic)")
            continue

        strip, valid = build_strip(mosaic, meta["native_gsd_m_per_px"], meta["bearing_deg"])
        del mosaic
        n = site["frames"]

        sh, sw = valid.shape[:2]
        need_h = int(math.ceil((n - 1) * STEP_PX)) + SRC_H + 2
        if sw < SRC_W or sh < need_h:
            print(f"  {site['name']}: SKIP (canvas too small)")
            continue
        x_left = (sw - SRC_W) // 2
        y_off = (sh - need_h) // 2
        window = valid[y_off:y_off + need_h, x_left:x_left + SRC_W]
        if float((window > 0).mean()) < 0.999:
            print(f"  {site['name']}: SKIP (incomplete coverage)")
            continue

        weight = placement_weight(strip, valid)
        rng = random.Random(seed + i)

        # Sample placement positions
        sub = weight[y_off:y_off + need_h, x_left:x_left + SRC_W]
        small = cv2.resize(sub, (SRC_W // 16, need_h // 16), interpolation=cv2.INTER_AREA)
        flat = small.ravel().astype(np.float64)
        total = flat.sum()
        if total <= 0:
            print(f"  {site['name']}: SKIP (no placeable ground)")
            continue
        cdf = np.cumsum(flat) / total
        w_cells = small.shape[1]

        def draw():
            idx = int(np.searchsorted(cdf, rng.random()))
            cy_cell, cx_cell = divmod(min(idx, cdf.size - 1), w_cells)
            return (x_left + (cx_cell + rng.random()) * 16,
                    y_off + (cy_cell + rng.random()) * 16)

        placed, taken = [], []
        for cls in OBJECT_CLASSES:
            bank = sprites.get(cls)
            if not bank:
                continue
            for _ in range(per_class):
                for _try in range(220):
                    sp = rotate_sprite(rng.choice(bank), rng.randrange(4))
                    h, w = sp["alpha"].shape
                    sx, sy = draw()
                    sx = min(max(sx, x_left + w / 2 + 4), x_left + SRC_W - w / 2 - 4)
                    sy = min(max(sy, y_off + h / 2 + 4), y_off + need_h - h / 2 - 4)
                    fy0_s, fx0_s = int(sy - h / 2), int(sx - w / 2)
                    foot = weight[max(0, fy0_s):fy0_s + h, max(0, fx0_s):fx0_s + w]
                    if foot.size == 0 or foot.min() <= 0 or foot.mean() < 0.22:
                        continue
                    r = max(w, h) * 0.75
                    if any(math.hypot(sx - ox, sy - oy) < r + orr for ox, oy, orr in taken):
                        continue
                    taken.append((sx, sy, r))
                    placed.append({"cls": cls, "sprite": sp, "sx": sx, "sy": sy})
                    break

        if not placed:
            print(f"  {site['name']}: SKIP (no objects placed)")
            continue

        out = scenes_dir / site["name"]
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "annotations").mkdir(parents=True, exist_ok=True)

        bearing = math.radians(meta["bearing_deg"])
        y_top0 = y_off + (n - 1) * STEP_PX
        scene_boxes = 0

        for k in range(n):
            y_top = y_top0 - k * STEP_PX
            yi, xi = int(round(y_top)), int(round(x_left))
            frame = strip[yi:yi + SRC_H, xi:xi + SRC_W].copy()
            if frame.shape[:2] != (SRC_H, SRC_W):
                break

            anns = []
            for obj in placed:
                cx_obj, cy_obj = obj["sx"] - xi, obj["sy"] - yi
                h_s, w_s = obj["sprite"]["alpha"].shape
                if cx_obj + w_s / 2 < 0 or cx_obj - w_s / 2 > SRC_W or \
                   cy_obj + h_s / 2 < 0 or cy_obj - h_s / 2 > SRC_H:
                    continue
                box = paste(frame, obj["sprite"], cx_obj, cy_obj)
                if box is None:
                    continue
                anns.append({"object_id": obj["cls"], "bbox": box})

            scene_boxes += len(anns)
            dist = k * STEP_M
            cv2.imwrite(str(out / "images" / f"frame_{k:06d}.png"), frame,
                        [cv2.IMWRITE_PNG_COMPRESSION, 3])
            per = {}
            for a in anns:
                per[a["object_id"]] = per.get(a["object_id"], 0) + 1
            write_json(out / "annotations" / f"frame_{k:06d}.json", {
                "frame": k,
                "pose": {"x": dist * math.sin(bearing), "y": dist * math.cos(bearing),
                         "z": -ALTITUDE_M},
                "annotations": anns,
                "object_counts": dict(sorted(per.items())),
            })

        totals: dict[str, int] = {}
        for obj in placed:
            totals[obj["cls"]] = totals.get(obj["cls"], 0) + 1
        write_json(out / "run_metadata.json", {
            "capture": {"altitude_m": ALTITUDE_M, "num_frames": n, "step_m": STEP_M},
            "total_objects": len(placed),
            "object_totals": totals,
        })
        total_boxes += scene_boxes
        print(f"  {site['name']:<26s} {n:3d} frames  {len(placed):3d} objects  {scene_boxes:5d} boxes")

    print(f"  Total: {total_boxes} boxes across all scenes")
    return scenes_dir


# ----------------------------------------------------------- YOLO export --

def export_yolo(scenes_dir: Path, out_dir: Path, levels: list[int] = None,
                views_per_frame: int = 2, min_visible: float = 0.35,
                val_fraction: float = 0.2, seed: int = 7):
    print(f"\n{'='*60}")
    print("Step 4: Exporting YOLO dataset")
    print(f"{'='*60}")

    if levels is None:
        levels = [0, 1, 2]

    scenes = sorted(p for p in scenes_dir.iterdir()
                    if p.is_dir() and (p / "run_metadata.json").exists())
    if not scenes:
        raise SystemExit(f"No scenes under {scenes_dir}")

    rng = random.Random(seed)
    shuffled = scenes[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction))
    val = {p.name for p in shuffled[:n_val]}
    print(f"  {len(scenes)} scenes: {len(scenes) - len(val)} train / {len(val)} val")

    idx_of = {c: i for i, c in enumerate(OBJECT_CLASSES)}
    for sub in ("train", "val"):
        (out_dir / "images" / sub).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / sub).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    boxes = {"train": 0, "val": 0}

    for scene in scenes:
        sub = "val" if scene.name in val else "train"
        frames = sorted(int(p.stem.split("_")[-1])
                        for p in (scene / "images").glob("frame_*.png"))
        for fr in frames:
            img = cv2.imread(str(scene / "images" / f"frame_{fr:06d}.png"), cv2.IMREAD_COLOR)
            if img is None:
                continue
            anns = json.loads(
                (scene / "annotations" / f"frame_{fr:06d}.json").read_text())["annotations"]

            for level in levels:
                rw, rh = REGION[level]
                if level == 0:
                    view_list = [(0, 0, SRC_W, SRC_H)]
                else:
                    view_list = []
                    for _ in range(views_per_frame):
                        cx = rng.uniform(rw / 2, SRC_W - rw / 2)
                        cy = rng.uniform(rh / 2, SRC_H - rh / 2)
                        x0, y0 = int(cx - rw / 2), int(cy - rh / 2)
                        view_list.append((x0, y0, x0 + rw, y0 + rh))

                for vi, (x0, y0, x1, y1) in enumerate(view_list):
                    crop = img[y0:y1, x0:x1]
                    if crop.shape[1] != VIEW_W or crop.shape[0] != VIEW_H:
                        crop = cv2.resize(crop, (VIEW_W, VIEW_H),
                                          interpolation=cv2.INTER_AREA)
                    rows = []
                    for a in anns:
                        bx0, by0, bx1, by1 = a["bbox"]
                        ix0, iy0 = max(bx0, x0), max(by0, y0)
                        ix1, iy1 = min(bx1, x1), min(by1, y1)
                        if ix1 <= ix0 or iy1 <= iy0:
                            continue
                        area = (bx1 - bx0) * (by1 - by0)
                        if area <= 0 or ((ix1 - ix0) * (iy1 - iy0)) / area < min_visible:
                            continue
                        cx_n = ((ix0 + ix1) / 2 - x0) / (x1 - x0)
                        cy_n = ((iy0 + iy1) / 2 - y0) / (y1 - y0)
                        w_n = (ix1 - ix0) / (x1 - x0)
                        h_n = (iy1 - iy0) / (y1 - y0)
                        rows.append(f"{idx_of[a['object_id']]} "
                                    f"{cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")

                    stem = f"{scene.name}_f{fr:04d}_L{level}_{vi}"
                    cv2.imwrite(str(out_dir / "images" / sub / f"{stem}.jpg"), crop,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    (out_dir / "labels" / sub / f"{stem}.txt").write_text(
                        "\n".join(rows) + ("\n" if rows else ""))
                    counts[sub] += 1
                    boxes[sub] += len(rows)

        print(f"  {scene.name:<26s} -> {sub}", flush=True)

    yaml_lines = [f"path: {out_dir.resolve()}", "train: images/train", "val: images/val",
                  f"nc: {len(OBJECT_CLASSES)}", "names:"]
    yaml_lines += [f"  {i}: {c}" for i, c in enumerate(OBJECT_CLASSES)]
    (out_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    print(f"\n  train {counts['train']} views / {boxes['train']} boxes")
    print(f"  val   {counts['val']} views / {boxes['val']} boxes")
    print(f"  -> {out_dir}/data.yaml")
    return out_dir


# ------------------------------------------------------------------- main --

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--naic-helsinki", type=Path, required=True,
                        help="Path to Nordic-AI-Cup-2026/drone-flyby/src/helsinki")
    parser.add_argument("--output", type=Path, default=Path("map-drone-data"),
                        help="Output root directory (default: map-drone-data/)")
    parser.add_argument("--step", choices=["sprites", "basemaps", "scenes", "export", "all"],
                        default="all", help="Run a single step or all (default: all)")
    parser.add_argument("--basemap-source", default="esri", choices=sorted(BASEMAP_SOURCES))
    parser.add_argument("--api-key", default=os.environ.get("BASEMAP_API_KEY"))
    parser.add_argument("--tile-cache", type=Path,
                        default=Path(os.environ.get("TILE_CACHE", "/tmp/tilecache")))
    parser.add_argument("--per-class", type=int, default=2,
                        help="Object instances per class per scene (default: 2)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--yolo-levels", type=int, nargs="+", default=[0, 1, 2],
                        help="YOLO export view levels (default: 0 1 2)")
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output.resolve()

    sprites_dir = output / "sprites"
    basemaps_dir = output / "basemaps"
    scenes_dir = output / "scenes"
    yolo_dir = output / "yolo"

    steps = ["sprites", "basemaps", "scenes", "export"] if args.step == "all" else [args.step]

    if "sprites" in steps:
        extract_sprites(args.naic_helsinki, sprites_dir)

    if "basemaps" in steps:
        fetch_basemaps(DEFAULT_SITES, args.basemap_source, basemaps_dir,
                       args.tile_cache, args.api_key)

    if "scenes" in steps:
        generate_scenes(basemaps_dir, sprites_dir, scenes_dir,
                        per_class=args.per_class, workers=args.workers, seed=args.seed)

    if "export" in steps:
        export_yolo(scenes_dir, yolo_dir, levels=args.yolo_levels)

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"Output: {output}")
    if yolo_dir.exists():
        print(f"YOLO dataset: {yolo_dir}/data.yaml")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
