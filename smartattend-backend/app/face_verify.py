"""
Face verification — server-side, using InsightFace (buffalo_l / ArcFace).

Why this replaces the old face-api.js-on-frontend approach:
- face-api.js descriptors are generated in-browser with a lighter model.
  Fine for a demo, not accurate enough once real faculty use this daily
  under real lighting/angle variance.
- Trusting a client-supplied embedding also means the backend can't verify
  a real face was even present — a forged 128-float array would "work".
- Now the backend receives raw image frames and does detection + embedding
  itself. The client never gets to hand us a fake vector.

Enrollment stores 3 embeddings per faculty member (left / center / right
angle buckets, picked client-side via yaw estimation from a 5s video).
Verification (check-in/check-out) sends a handful of candidate frames from
a short video; the backend picks the single best frame (by detection
confidence, face size, and frontal-ness) and compares its embedding
against all 3 stored embeddings, taking the max similarity. This makes
matching robust to whatever angle the phone happens to be at that day.
"""
import base64
import json
from io import BytesIO
from typing import List, Optional

import numpy as np
import cv2
from PIL import Image

FACE_MATCH_THRESHOLD = 0.62  # cosine similarity on L2-normalized ArcFace embeddings
MIN_DETECTION_CONFIDENCE = 0.4
MIN_FACE_SIZE_PX = 80  # reject tiny/far-away faces

LIVENESS_MIN_FRAMES = 3
LIVENESS_SIMILARITY_CEIL = 0.97   # tightened from 0.98
LIVENESS_TEXTURE_FLOOR = 15.0     # raised from 12.0

# Anti-spoofing thresholds (each signal produces a 0-1 suspicion score)
MOIRE_ENERGY_CEIL = 0.32          # FFT high-freq energy ratio; screens exceed this
SKIN_RATIO_FLOOR = 0.20           # minimum YCbCr skin-tone pixels in face crop
LBP_UNIFORMITY_CEIL = 0.06        # LBP histogram uniformity; flat surfaces exceed this
COLOR_VARIANCE_FLOOR = 12.0       # per-channel saturation std; screens are more uniform
MULTI_LAP_RATIO_FLOOR = 0.30      # fine-to-coarse Laplacian ratio; prints/screens are lower
SPOOF_SCORE_THRESHOLD = 0.45      # weighted sum above this = rejected

_face_app = None


def _get_face_app():
    """
    Lazy-load InsightFace's FaceAnalysis app (buffalo_l: detection + landmarks
    + ArcFace recognition in one pass). Lazy so a cold backend process /
    simple health-check request doesn't pay the model-load cost.
    """
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis

        _face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _face_app.prepare(ctx_id=0, det_size=(320, 320))
    return _face_app


def decode_image(image_b64: str) -> np.ndarray:
    """Accepts a base64 string (with or without a data: URL prefix), returns a BGR numpy array (cv2/insightface convention)."""
    if "," in image_b64 and image_b64.strip().startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    img = Image.open(BytesIO(raw)).convert("RGB")
    arr = np.array(img)
    return arr[:, :, ::-1].copy()  # RGB -> BGR


def _face_quality_score(face) -> float:
    """
    Combines detection confidence, face size (bigger = closer/clearer), and
    frontal-ness into one score so we can pick the single best frame out of
    several candidates sent from a short video clip.
    """
    x1, y1, x2, y2 = face.bbox
    w, h = x2 - x1, y2 - y1
    if w < MIN_FACE_SIZE_PX or h < MIN_FACE_SIZE_PX:
        return -1.0

    size_score = min(w, h) / 300.0  # normalize; 300px+ face treated as "large enough", no extra bonus past that
    size_score = min(size_score, 1.0)

    det_score = float(getattr(face, "det_score", 0.5))

    # Frontal-ness from 5-point kps: compare horizontal eye-to-nose distances.
    frontal_score = 1.0
    kps = getattr(face, "kps", None)
    if kps is not None and len(kps) >= 3:
        left_eye, right_eye, nose = kps[0], kps[1], kps[2]
        left_dist = abs(nose[0] - left_eye[0])
        right_dist = abs(right_eye[0] - nose[0])
        total = left_dist + right_dist
        if total > 0:
            ratio = min(left_dist, right_dist) / total  # 0.5 = perfectly frontal, closer to 0 = profile
            frontal_score = ratio * 2.0  # rescale to 0-1

    return (det_score * 0.4) + (size_score * 0.3) + (frontal_score * 0.3)


def _make_thumbnail(img_bgr: np.ndarray, bbox, size: int = 150) -> str:
    """
    Crops the face region (with a little margin), resizes to a small square,
    and returns a compressed base64 JPEG — for admin approval review only.
    Deliberately NOT the full original frame: keeps DB rows small and avoids
    storing more of the person's photo than necessary for that purpose.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    margin_x, margin_y = int(w * 0.3), int(h * 0.3)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(img_bgr.shape[1], x2 + margin_x)
    y2 = min(img_bgr.shape[0], y2 + margin_y)

    crop_rgb = img_bgr[y1:y2, x1:x2, ::-1]  # BGR -> RGB
    pil_img = Image.fromarray(crop_rgb).convert("RGB")
    pil_img.thumbnail((size, size))

    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def get_best_face_embedding(image_b64_list: List[str], *, collect_all: bool = False) -> dict:
    """
    Given several candidate frames (base64), runs detection on each,
    scores every detected face, and returns the embedding from the single
    best face found across all frames.

    Returns {"embedding": [...], "quality": float} on success, or
    {"embedding": None, "reason": "..."} if no usable face was found.
    """
    app = _get_face_app()

    best_face = None
    best_img = None
    best_score = -1.0
    all_embeddings = []
    all_face_crops = []

    MAX_FRAMES = 10
    frames_to_process = image_b64_list[:MAX_FRAMES]

    for image_b64 in frames_to_process:
        try:
            img = decode_image(image_b64)
        except Exception:
            continue

        faces = app.get(img)
        if not faces:
            continue

        # If multiple faces in frame, only consider the largest (assume it's the person checking in)
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        candidate = faces[0]
        score = _face_quality_score(candidate)

        if collect_all and score > 0 and hasattr(candidate, 'normed_embedding'):
            all_embeddings.append(candidate.normed_embedding.copy())
            bx1, by1, bx2, by2 = [int(v) for v in candidate.bbox]
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(img.shape[1], bx2), min(img.shape[0], by2)
            all_face_crops.append(img[by1:by2, bx1:bx2])

        if score > best_score:
            best_score = score
            best_face = candidate
            best_img = img
            if best_score > 0.85 and not collect_all:
                break

    if best_face is None or best_score < 0:
        result = {"embedding": None, "quality": 0.0, "reason": "no_usable_face_detected", "thumbnail": None}
        if collect_all:
            result["all_embeddings"] = all_embeddings
            result["all_face_crops"] = all_face_crops
        return result

    if float(getattr(best_face, "det_score", 0.0)) < MIN_DETECTION_CONFIDENCE:
        result = {"embedding": None, "quality": best_score, "reason": "low_detection_confidence", "thumbnail": None}
        if collect_all:
            result["all_embeddings"] = all_embeddings
            result["all_face_crops"] = all_face_crops
        return result

    embedding = best_face.normed_embedding  # InsightFace already L2-normalizes this
    thumbnail = _make_thumbnail(best_img, best_face.bbox)
    result = {"embedding": embedding.tolist(), "quality": best_score, "reason": None, "thumbnail": thumbnail}
    if collect_all:
        result["all_embeddings"] = all_embeddings
        result["all_face_crops"] = all_face_crops
    return result


def _compute_lbp(gray: np.ndarray) -> np.ndarray:
    """Vectorized 8-neighbor Local Binary Pattern."""
    h, w = gray.shape
    center = gray[1:h-1, 1:w-1].astype(np.int16)
    lbp = np.zeros_like(center, dtype=np.uint8)
    lbp |= (gray[0:h-2, 0:w-2].astype(np.int16) >= center).astype(np.uint8) << 7
    lbp |= (gray[0:h-2, 1:w-1].astype(np.int16) >= center).astype(np.uint8) << 6
    lbp |= (gray[0:h-2, 2:w  ].astype(np.int16) >= center).astype(np.uint8) << 5
    lbp |= (gray[1:h-1, 2:w  ].astype(np.int16) >= center).astype(np.uint8) << 4
    lbp |= (gray[2:h,   2:w  ].astype(np.int16) >= center).astype(np.uint8) << 3
    lbp |= (gray[2:h,   1:w-1].astype(np.int16) >= center).astype(np.uint8) << 2
    lbp |= (gray[2:h,   0:w-2].astype(np.int16) >= center).astype(np.uint8) << 1
    lbp |= (gray[1:h-1, 0:w-2].astype(np.int16) >= center).astype(np.uint8) << 0
    return lbp


def _signal_moire(gray_crop: np.ndarray) -> float:
    """Moiré / screen pixel-grid detection via FFT. Returns suspicion 0-1."""
    if gray_crop.shape[0] < 64 or gray_crop.shape[1] < 64:
        return 0.0
    resized = cv2.resize(gray_crop, (128, 128)).astype(np.float32)
    f = np.fft.fft2(resized)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    total_energy = np.sum(magnitude)
    if total_energy == 0:
        return 0.0

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    high_mask = dist > min(h, w) * 0.4
    high_energy = np.sum(magnitude[high_mask])
    ratio = float(high_energy / total_energy)

    high_vals = magnitude[high_mask]
    mean_hf = np.mean(high_vals)
    peak_boost = 0.0
    if mean_hf > 0:
        cv_hf = float(np.std(high_vals) / mean_hf)
        if cv_hf > 2.5:
            peak_boost = 0.3

    raw = (ratio / MOIRE_ENERGY_CEIL) + peak_boost
    return min(raw, 1.0)


def _signal_skin_color(bgr_crop: np.ndarray) -> float:
    """YCbCr skin-color analysis. Returns suspicion 0-1."""
    if bgr_crop.size == 0:
        return 0.0
    ycrcb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2YCrCb)
    lower = np.array([80, 133, 77], dtype=np.uint8)
    upper = np.array([235, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    total_px = mask.shape[0] * mask.shape[1]
    skin_ratio = float(np.count_nonzero(mask) / total_px) if total_px > 0 else 0.0

    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    sat_std = float(np.std(hsv[:, :, 1].astype(np.float32)))

    score = 0.0
    if skin_ratio < SKIN_RATIO_FLOOR:
        score += 0.6 * (1.0 - skin_ratio / SKIN_RATIO_FLOOR)
    if sat_std < 15.0:
        score += 0.4 * (1.0 - sat_std / 15.0)
    return min(score, 1.0)


def _signal_lbp_texture(gray_crop: np.ndarray) -> float:
    """LBP micro-texture analysis. Returns suspicion 0-1."""
    if gray_crop.shape[0] < 32 or gray_crop.shape[1] < 32:
        return 0.0
    resized = cv2.resize(gray_crop, (96, 96))
    lbp = _compute_lbp(resized)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 1.0
    prob = hist / total
    uniformity = float(np.sum(prob ** 2))
    nonzero = prob[prob > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))

    score = 0.0
    if uniformity > LBP_UNIFORMITY_CEIL:
        score += 0.5 * min((uniformity - LBP_UNIFORMITY_CEIL) / LBP_UNIFORMITY_CEIL, 1.0)
    if entropy < 4.0:
        score += 0.5 * (1.0 - entropy / 4.0)
    return min(score, 1.0)


def _signal_multi_laplacian(gray_crop: np.ndarray) -> float:
    """Multi-scale Laplacian texture. Returns suspicion 0-1."""
    if gray_crop.shape[0] < 32 or gray_crop.shape[1] < 32:
        return 0.0
    resized = cv2.resize(gray_crop, (128, 128)).astype(np.float32)
    fine = float(cv2.Laplacian(resized, cv2.CV_32F, ksize=3).var())
    coarse = float(cv2.Laplacian(resized, cv2.CV_32F, ksize=7).var())
    if coarse == 0:
        return 0.0
    ratio = fine / coarse
    if ratio < MULTI_LAP_RATIO_FLOOR:
        return min((MULTI_LAP_RATIO_FLOOR - ratio) / MULTI_LAP_RATIO_FLOOR, 1.0)
    return 0.0


def _signal_color_uniformity(bgr_crop: np.ndarray) -> float:
    """Screens produce more spatially uniform color than real skin. Returns suspicion 0-1."""
    if bgr_crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    sat_std = float(np.std(sat))
    val_std = float(np.std(val))
    combined = (sat_std + val_std) / 2.0
    if combined < COLOR_VARIANCE_FLOOR:
        return min((COLOR_VARIANCE_FLOOR - combined) / COLOR_VARIANCE_FLOOR, 1.0)
    return 0.0


def _check_liveness(all_embeddings: List[np.ndarray], all_face_crops: List[np.ndarray]) -> dict:
    """
    Six-signal anti-spoofing gate with weighted scoring:
      1. Embedding variance  — photo replay yields near-identical embeddings
      2. Laplacian texture   — printed photos / matte screens have low variance
      3. Moiré / FFT         — screens produce periodic high-frequency artifacts
      4. YCbCr skin color    — real skin sits in a specific color-space range
      5. LBP micro-texture   — flat surfaces have fewer distinct texture patterns
      6. Multi-scale Laplacian — fine-to-coarse frequency ratio distinguishes skin
    Plus a color-uniformity bonus signal.

    Each signal produces a 0-1 suspicion score; the weighted sum is compared
    to SPOOF_SCORE_THRESHOLD. Hard-fail thresholds on embedding similarity
    and basic Laplacian are kept as instant-reject gates.
    """
    result = {"alive": True, "reason": None, "scores": {}}

    # --- Gate 1: Embedding similarity (instant reject) ---
    if len(all_embeddings) >= LIVENESS_MIN_FRAMES:
        sims = []
        for i in range(len(all_embeddings)):
            for j in range(i + 1, len(all_embeddings)):
                dot = float(np.dot(all_embeddings[i], all_embeddings[j]))
                norm = float(np.linalg.norm(all_embeddings[i]) * np.linalg.norm(all_embeddings[j]))
                sims.append(dot / norm if norm > 0 else 0.0)
        avg_sim = float(np.mean(sims))
        result["scores"]["frame_similarity"] = round(avg_sim, 4)
        if avg_sim > LIVENESS_SIMILARITY_CEIL:
            result["alive"] = False
            result["reason"] = "liveness_failed"
            return result

    if not all_face_crops:
        return result

    # --- Gate 2: Basic Laplacian texture (instant reject) ---
    textures = []
    grays = []
    for crop in all_face_crops:
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
        textures.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
    if textures:
        median_texture = float(np.median(textures))
        result["scores"]["texture"] = round(median_texture, 2)
        if median_texture < LIVENESS_TEXTURE_FLOOR:
            result["alive"] = False
            result["reason"] = "liveness_failed"
            return result

    # --- Weighted scoring across all advanced signals ---
    moire_scores = [_signal_moire(g) for g in grays]
    skin_scores = [_signal_skin_color(c) for c in all_face_crops if c.size > 0]
    lbp_scores = [_signal_lbp_texture(g) for g in grays]
    mlap_scores = [_signal_multi_laplacian(g) for g in grays]
    color_scores = [_signal_color_uniformity(c) for c in all_face_crops if c.size > 0]

    avg_moire = float(np.mean(moire_scores)) if moire_scores else 0.0
    avg_skin = float(np.mean(skin_scores)) if skin_scores else 0.0
    avg_lbp = float(np.mean(lbp_scores)) if lbp_scores else 0.0
    avg_mlap = float(np.mean(mlap_scores)) if mlap_scores else 0.0
    avg_color = float(np.mean(color_scores)) if color_scores else 0.0

    result["scores"]["moire"] = round(avg_moire, 4)
    result["scores"]["skin_color"] = round(avg_skin, 4)
    result["scores"]["lbp_texture"] = round(avg_lbp, 4)
    result["scores"]["multi_laplacian"] = round(avg_mlap, 4)
    result["scores"]["color_uniformity"] = round(avg_color, 4)

    # Weights: moiré is the strongest screen indicator, skin color catches both
    weighted = (
        avg_moire * 0.30
        + avg_skin * 0.25
        + avg_lbp * 0.15
        + avg_mlap * 0.15
        + avg_color * 0.15
    )
    result["scores"]["spoof_score"] = round(weighted, 4)

    if weighted >= SPOOF_SCORE_THRESHOLD:
        result["alive"] = False
        result["reason"] = "liveness_failed"

    return result


def embeddings_to_str(embeddings: List[List[float]]) -> str:
    """Stores multiple embeddings (e.g. 3 angle buckets) as a single JSON text column."""
    return json.dumps(embeddings)


def str_to_embeddings(s: str) -> List[np.ndarray]:
    """
    Parses the stored embeddings column. Handles both the new JSON-list
    format and the old comma-separated single-embedding format so existing
    enrolled faculty don't break — they'll just have 1 embedding until they
    re-enroll.
    """
    s = s.strip()
    if s.startswith("["):
        parsed = json.loads(s)
        # Could be a flat list (old single embedding as JSON) or list-of-lists
        if parsed and isinstance(parsed[0], (int, float)):
            return [np.array(parsed)]
        return [np.array(e) for e in parsed]
    # Legacy comma-separated single embedding
    return [np.array([float(x) for x in s.split(",")])]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def verify_face_from_frames(candidate_frames_b64: List[str], enrolled_embeddings_str: str) -> dict:
    """
    Main entry point for check-in/check-out. Takes candidate frames from the
    live capture, extracts the best embedding, and compares it against all
    stored angle-bucket embeddings for that faculty member, taking the max
    similarity (robust to whichever angle matches best today).
    """
    extraction = get_best_face_embedding(candidate_frames_b64, collect_all=True)
    if extraction["embedding"] is None:
        extraction.pop("all_embeddings", None)
        extraction.pop("all_face_crops", None)
        return {"score": 0.0, "verified": "fail", "reason": extraction["reason"]}

    liveness = _check_liveness(
        extraction.get("all_embeddings", []),
        extraction.get("all_face_crops", []),
    )

    del extraction["all_embeddings"]
    del extraction["all_face_crops"]
    extraction.pop("thumbnail", None)

    if not liveness["alive"]:
        return {
            "score": 0.0,
            "verified": "fail",
            "reason": liveness["reason"],
            "liveness_scores": liveness["scores"],
        }

    live = np.array(extraction["embedding"])
    del extraction

    enrolled_list = str_to_embeddings(enrolled_embeddings_str)

    best_score = max(cosine_similarity(live, enrolled) for enrolled in enrolled_list)
    verified = "pass" if best_score >= FACE_MATCH_THRESHOLD else "fail"
    return {"score": best_score, "verified": verified, "reason": None, "liveness_scores": liveness["scores"]}


def enroll_from_frames(image_b64_list: List[str]) -> dict:
    """
    Main entry point for enrollment / re-enrollment. Expects up to 3 frames,
    one per angle bucket (left/center/right), already selected client-side.
    Runs each through detection+embedding independently — does NOT pick a
    single "best" across all of them, since we deliberately want 3 distinct
    angles stored, not 3 attempts at the same angle.

    Returns {"embeddings": [[...], [...], [...]], "thumbnails": [...], "rejected": [indices]}
    """
    embeddings = []
    thumbnails = []
    rejected = []

    for i, image_b64 in enumerate(image_b64_list):
        result = get_best_face_embedding([image_b64])
        if result["embedding"] is None:
            rejected.append({"index": i, "reason": result["reason"]})
        else:
            embeddings.append(result["embedding"])
            thumbnails.append(result["thumbnail"])

    if not embeddings:
        return {"embeddings": [], "thumbnails": [], "rejected": rejected, "reason": "no_usable_face_in_any_frame"}

    return {"embeddings": embeddings, "thumbnails": thumbnails, "rejected": rejected, "reason": None}
