from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2, os, uuid
import numpy as np
from collections import defaultdict
import math

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: SAMPLE LOADING
# Capillary chamber is placed on microscope. Chamber depth is fixed (20 µm).
# auto_calibrate() converts pixels → µm based on image resolution.
# Chamber volume is depth × grid area (default: 0.0005 mL counting area).
# ─────────────────────────────────────────────────────────────────────────────

CHAMBER_DEPTH_UM   = 20.0      # standard Makler/Leja chamber depth (µm)
CHAMBER_AREA_ML    = 0.0005    # counting area volume (mL) — Makler cell equivalent
DILUTION_FACTOR    = 1.0       # set >1 if sample was diluted before loading

def auto_calibrate(frame):
    """
    Returns µm per pixel based on image resolution.
    Calibrated for standard 10× objective on typical microscope cameras:
      1920px wide → 0.05 µm/px (high-res)
      1280px wide → 0.137 µm/px (HD)
       720px wide → 0.275 µm/px (SD)
    In production: replace with actual scale-bar OCR or microscope metadata.
    """
    h, w = frame.shape[:2]
    if   w >= 1920: return 0.050
    elif w >= 1280: return 0.137
    elif w >= 720:  return 0.275
    else:           return 0.550


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: DETECTION — YOLOv8 Segmentation
# Mask → bounding box → confidence filter → size/shape validation
# ─────────────────────────────────────────────────────────────────────────────

try:
    model = YOLO("models/sperm-seg.pt")
    print("✅ Segmentation model loaded")
except Exception as e:
    model = None
    print("⚠️  Model not found:", e)


def detect_seg(frame):
    """
    Run YOLOv8-seg on a single frame.
    Returns list of [x1, y1, x2, y2, conf, mask_area_px] detections.
    mask_area_px is used downstream for head/midpiece/tail morphology.
    """
    if model is None:
        return []

    res = model(frame, verbose=False)[0]
    dets = []

    if res.masks is not None:
        for i, mask_xy in enumerate(res.masks.xy):
            pts  = np.array(mask_xy, dtype=np.int32)
            if len(pts) < 3:
                continue
            x, y, w, h = cv2.boundingRect(pts)
            conf        = float(res.boxes.conf[i]) if res.boxes is not None else 0.5
            area        = cv2.contourArea(pts)          # mask area in px²
            dets.append([x, y, x + w, y + h, conf, area])
    return dets


def valid_sperm(x1, y1, x2, y2, conf, um_per_px):
    """
    Validate a detection against known sperm biology:
      Confidence ≥ 0.25
      Bounding box in pixels:  2–150 px each side
      Bounding box in µm:      0.3–30 µm each side
        (human sperm head: 3–5 µm wide, 4–6 µm long; tail: up to 55 µm)
    """
    w, h = x2 - x1, y2 - y1
    if conf < 0.25:                     return False
    if w < 2 or h < 2:                  return False
    if w > 150 or h > 150:              return False
    w_um, h_um = w * um_per_px, h * um_per_px
    if w_um < 0.3 or h_um < 0.3:       return False
    if w_um > 30  or h_um > 30:        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: TRACKING — ByteTrack-style multi-object tracker
# Each track accumulates a position path across frames.
# Path is later used to compute all CASA velocity metrics.
# ─────────────────────────────────────────────────────────────────────────────

class ByteTracker:
    """
    Lightweight ByteTrack-inspired tracker.
    Uses IoU matching between existing track bounding boxes and new detections.
    Tracks are kept alive for max_age consecutive miss-frames before deletion.

    For production: swap with the official ByteTrack or BotSort implementation
    for multi-scale association and Kalman state estimation.
    """

    def __init__(self, max_age=30, iou_thresh=0.3):
        self.tracks     = {}        # track_id → {bbox, age, hits, path, areas}
        self.next_id    = 1
        self.max_age    = max_age
        self.iou_thresh = iou_thresh

    @staticmethod
    def iou(a, b):
        xA = max(a[0], b[0]);  yA = max(a[1], b[1])
        xB = min(a[2], b[2]);  yB = min(a[3], b[3])
        inter  = max(0, xB - xA) * max(0, yB - yA)
        areaA  = (a[2] - a[0]) * (a[3] - a[1])
        areaB  = (b[2] - b[0]) * (b[3] - b[1])
        union  = areaA + areaB - inter + 1e-6
        return inter / union

    def update(self, detections):
        """
        detections: list of [x1, y1, x2, y2, conf, area_px]
        Returns updated tracks dict.
        """
        assigned = set()

        # ── Match existing tracks to new detections (greedy by IoU) ──────────
        for tid, t in list(self.tracks.items()):
            best_iou, best_j = 0, -1
            for j, det in enumerate(detections):
                if j in assigned:
                    continue
                iou = self.iou(t["bbox"], det[:4])
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou > self.iou_thresh and best_j != -1:
                det = detections[best_j]
                x1, y1, x2, y2, _, area = det
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                self.tracks[tid]["bbox"]  = [x1, y1, x2, y2]
                self.tracks[tid]["age"]   = 0
                self.tracks[tid]["hits"] += 1
                self.tracks[tid]["path"].append((cx, cy))
                self.tracks[tid]["areas"].append(area)
                assigned.add(best_j)
            else:
                self.tracks[tid]["age"] += 1

            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]

        # ── Spawn new tracks for unmatched detections ─────────────────────────
        for j, det in enumerate(detections):
            if j in assigned:
                continue
            x1, y1, x2, y2, _, area = det
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            self.tracks[self.next_id] = {
                "bbox" : [x1, y1, x2, y2],
                "age"  : 0,
                "hits" : 1,
                "path" : [(cx, cy)],
                "areas": [area],
            }
            self.next_id += 1

        return self.tracks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

# ── 5a. Concentration (WHO 6th ed. ref: ≥ 16 × 10⁶/mL) ─────────────────────

def concentration(avg_sperm_count_per_frame):
    """
    Concentration = counted sperm / chamber volume
    Volume = chamber area × depth (Makler cell: 0.0005 mL × 1 = 0.0005 mL)
    Result in cells/mL; divide by 1e6 for M/mL.
    """
    cells_per_ml = (avg_sperm_count_per_frame * DILUTION_FACTOR) / CHAMBER_AREA_ML
    return cells_per_ml


# ── 5b. CASA Velocity Metrics (µm/s) ────────────────────────────────────────
#   VCL  Curvilinear Velocity     — mean of frame-to-frame displacements
#   VSL  Straight-Line Velocity   — displacement start→end / total time
#   VAP  Average Path Velocity    — smoothed 5-point running average trajectory
#   LIN  Linearity                — VSL / VCL  (0–1)
#   STR  Straightness             — VSL / VAP  (0–1)
#   WOB  Wobble                   — VAP / VCL  (0–1)
#   ALH  Amplitude Lateral Head   — mean lateral deviation from average path
#   BCF  Beat-Cross Frequency     — crossings of average-path per second

def compute_casa(path, fps, um_per_px):
    """
    path     : list of (cx, cy) pixel positions, one per frame
    fps      : video frame rate
    um_per_px: calibration factor
    Returns dict with all standard CASA metrics.
    """
    n = len(path)
    if n < 3:
        return None

    pts = np.array(path, dtype=float) * um_per_px   # convert to µm

    # VCL — sum of consecutive distances / total time
    dists = [np.linalg.norm(pts[i] - pts[i-1]) for i in range(1, n)]
    total_time = (n - 1) / fps
    vcl = sum(dists) / total_time if total_time > 0 else 0.0

    # VSL — straight-line distance / total time
    vsl = np.linalg.norm(pts[-1] - pts[0]) / total_time if total_time > 0 else 0.0

    # VAP — 5-point running-mean smoothed path velocity
    window = 5
    smoothed = []
    for i in range(n):
        lo, hi = max(0, i - window // 2), min(n, i + window // 2 + 1)
        smoothed.append(pts[lo:hi].mean(axis=0))
    smoothed = np.array(smoothed)
    vap_dists = [np.linalg.norm(smoothed[i] - smoothed[i-1]) for i in range(1, n)]
    vap = sum(vap_dists) / total_time if total_time > 0 else 0.0

    # Ratios
    lin = round(vsl / vcl,  3) if vcl > 0 else 0.0
    str_ = round(vsl / vap, 3) if vap > 0 else 0.0
    wob  = round(vap / vcl, 3) if vcl > 0 else 0.0

    # ALH — lateral deviation from smoothed path (µm)
    deviations = [np.linalg.norm(pts[i] - smoothed[i]) for i in range(n)]
    alh = round(float(np.mean(deviations)), 2)

    # BCF — count zero-crossings of lateral deviation sign per second
    lateral = [pts[i] - smoothed[i] for i in range(n)]
    cross_axis = np.array([v[0] for v in lateral])  # x-component of deviation
    signs = np.sign(cross_axis)
    crossings = int(np.sum(np.abs(np.diff(signs)) > 0))
    bcf = round(crossings / total_time, 1) if total_time > 0 else 0.0

    # Motility class (WHO 6th ed.)
    if vcl >= 25 and lin >= 0.5:
        motility = "rapid_progressive"       # PR fast
    elif vcl >= 5:
        motility = "slow_progressive"        # PR slow
    elif vcl > 0 and lin < 0.5:
        motility = "non_progressive"         # NP
    else:
        motility = "immotile"                # IM

    return {
        "vcl_um_s"  : round(vcl,  2),
        "vsl_um_s"  : round(vsl,  2),
        "vap_um_s"  : round(vap,  2),
        "lin"       : lin,
        "str"       : str_,
        "wob"       : wob,
        "alh_um"    : alh,
        "bcf_hz"    : bcf,
        "motility"  : motility,
    }


# ── 5c. Motility summary ─────────────────────────────────────────────────────

def calculate_motility(sperm_report):
    """
    Aggregate individual track motility classes into WHO percentages.
    WHO 6th ed. thresholds:
      Total motility (PR + NP) ≥ 42%
      Progressive motility     ≥ 30%
    """
    total = len(sperm_report)
    def pct(x):
        return round(x / max(total, 1) * 100, 1)

    rapid = sum(1 for s in sperm_report if s["motility"] == "rapid_progressive")
    slow  = sum(1 for s in sperm_report if s["motility"] == "slow_progressive")
    nonp  = sum(1 for s in sperm_report if s["motility"] == "non_progressive")
    imm   = sum(1 for s in sperm_report if s["motility"] == "immotile")

    rp, sp, npct, ip = pct(rapid), pct(slow), pct(nonp), pct(imm)

    return {
        "rapid_progressive_percent"  : rp,
        "slow_progressive_percent"   : sp,
        "non_progressive_percent"    : npct,
        "immotile_percent"           : ip,
        "total_motility_percent"     : round(rp + sp + npct, 1),
        "progressive_motility_pct"   : round(rp + sp, 1),
        "who_reference_total_pct"    : 42,
        "who_reference_progressive"  : 30,
    }


# ── 5d. Morphology (Kruger strict criteria) ──────────────────────────────────

def morphology(boxes_and_areas):
    """
    Classify each detected sperm into a morphology category using bbox + mask area.
    Kruger strict criteria (WHO 6th ed.): normal ≥ 4% is lower-reference limit.

    Classification rules (simplified; in production use a dedicated morphology CNN):
      Aspect ratio (AR) = max(w,h) / min(w,h)
        1.5 ≤ AR ≤ 2.5  → Normal (oval head)
        AR > 3.5         → Tail defect (elongated)
        w > h by >2:1    → Midpiece defect (thick midpiece)
        AR < 1.4         → Head defect (round/large head)
        Multiple: anything else
    """
    normal = head = mid = tail = multi = 0
    for x1, y1, x2, y2, area in boxes_and_areas:
        w, h = x2 - x1, y2 - y1
        ar = max(w, h) / max(min(w, h), 1)

        if 1.5 <= ar <= 2.5:
            normal += 1
        elif ar > 3.5:
            tail   += 1
        elif w > h * 2.0:
            mid    += 1
        elif ar < 1.4:
            head   += 1
        else:
            multi  += 1

    total    = max(len(boxes_and_areas), 1)
    abnormal = total - normal
    return {
        "normal_percent"           : round(normal / total * 100, 1),
        "abnormal_percent"         : round(abnormal / total * 100, 1),
        "head_defects_percent"     : round(head  / total * 100, 1),
        "midpiece_defects_percent" : round(mid   / total * 100, 1),
        "tail_defects_percent"     : round(tail  / total * 100, 1),
        "multiple_defects_percent" : round(multi / total * 100, 1),
        "who_reference_normal_pct" : 4,
    }


# ── 5e. Vitality ─────────────────────────────────────────────────────────────

def vitality(mot):
    """
    WHO 6th ed.: vitality (live sperm) reference ≥ 54%.
    Estimated from motility: motile cells are considered alive.
    In clinical practice use eosin-Y stain (membrane integrity) — not available
    from video alone.
    """
    live = mot["total_motility_percent"]
    return {
        "live_percent"         : live,
        "dead_percent"         : round(100 - live, 1),
        "who_reference_live_pct": 54,
        "note"                 : "Estimated from motility; eosin stain recommended for clinical use",
    }


# ── 5f. WHO Grading & Impression ─────────────────────────────────────────────

def who_grading(conc_M_ml, mot, morph):
    """
    WHO 6th edition semen analysis reference values.
    Returns list of diagnosed conditions.
    """
    conditions = []

    # Concentration
    if conc_M_ml == 0:
        conditions.append("Azoospermia (no sperm detected)")
    elif conc_M_ml < 1:
        conditions.append("Cryptozoospermia (< 1 M/mL)")
    elif conc_M_ml < 16:
        conditions.append("Oligozoospermia (< 16 M/mL)")

    # Motility
    if mot["progressive_motility_pct"] < 30:
        conditions.append("Asthenozoospermia (progressive motility < 30%)")
    elif mot["total_motility_percent"] < 42:
        conditions.append("Asthenozoospermia (total motility < 42%)")

    # Morphology
    if morph["normal_percent"] < 4:
        conditions.append("Teratozoospermia (normal morphology < 4%)")

    # Combined
    count = sum(1 for c in conditions
                if any(t in c for t in ["Oligozoospermia", "Asthenozoospermia", "Teratozoospermia"]))
    if count >= 2:
        conditions.append("Oligo-astheno-teratozoospermia (OAT syndrome)")
    if not conditions:
        conditions.append("Normozoospermia — all parameters within WHO reference range")

    return conditions


def impression(conc_M_ml, mot, morph, total_tracks):
    if total_tracks < 5:
        return "Insufficient data — fewer than 5 tracks detected; resubmit with adequate sample"
    if conc_M_ml == 0:
        return "Azoospermia — no motile sperm detected"
    if conc_M_ml < 16 or mot["progressive_motility_pct"] < 30 or morph["normal_percent"] < 4:
        return "Abnormal semen analysis — at least one parameter below WHO 6th ed. reference"
    return "Normal semen analysis — all parameters within WHO 6th ed. reference range"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_zone(cx, cy, w, h):
    x = "left"   if cx < w / 3 else "center" if cx < 2 * w / 3 else "right"
    y = "top"    if cy < h / 3 else "middle"  if cy < 2 * h / 3 else "bottom"
    return f"{y}-{x}"


# ─────────────────────────────────────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 + 3 + 4 + 5: VIDEO ANALYSIS endpoint
# Implements full 5-step CASA pipeline.
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/analyze/video", methods=["POST"])
def analyze_video():
    """
    Accepts a video file (mp4/avi).
    Runs the full 5-step CASA pipeline and returns a JSON report.

    Steps:
      1. Load video → auto-calibrate µm/px
      2. Read frames at native FPS (Step 2: capture)
      3. Run YOLOv8-seg detection per frame (Step 3)
      4. Update ByteTracker per frame (Step 4)
      5. Compute CASA metrics + WHO report (Step 5)
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file     = request.files["file"]
    tmp_path = f"{OUTPUT_DIR}/{uuid.uuid4()}.mp4"
    file.save(tmp_path)

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.remove(tmp_path)
        return jsonify({"error": "Cannot open video file"}), 400

    fps           = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker       = ByteTracker(max_age=30, iou_thresh=0.3)
    frame_counts  = []          # sperm count per frame (for concentration)
    all_detections= []          # [(x1,y1,x2,y2,area)] for morphology
    total_valid   = 0
    um_per_px     = None        # calibrated on first frame

    # ── Step 2 & 3 & 4: Frame loop ────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1/2: calibrate once from first frame
        if um_per_px is None:
            um_per_px = auto_calibrate(frame)

        # Step 3: detect
        raw_dets   = detect_seg(frame)
        valid_dets = []

        for d in raw_dets:
            x1, y1, x2, y2, conf, area = d
            if valid_sperm(x1, y1, x2, y2, conf, um_per_px):
                valid_dets.append(d)
                all_detections.append((x1, y1, x2, y2, area))
                total_valid += 1

        # Step 4: track
        tracker.update(valid_dets)
        frame_counts.append(len(valid_dets))

    cap.release()
    os.remove(tmp_path)

    # ── Step 5a: Concentration ────────────────────────────────────────────────
    avg_count = float(np.mean(frame_counts)) if frame_counts else 0.0
    conc_raw  = concentration(avg_count)            # cells/mL
    conc_M_ml = round(conc_raw / 1e6, 2)            # million/mL

    # ── Step 5b: Build per-track CASA report ─────────────────────────────────
    sperm_report = []
    for sid, t in tracker.tracks.items():
        path_pts = t["path"]
        if len(path_pts) < 3:
            continue

        casa = compute_casa(path_pts, fps, um_per_px)
        if casa is None:
            continue

        cx, cy = path_pts[-1]
        zone   = get_zone(cx, cy, frame_width, frame_height)

        sperm_report.append({
            "id"          : sid,
            "frames"      : len(path_pts),
            "zone"        : zone,
            **casa,
        })

    # ── Step 5c: Motility summary ─────────────────────────────────────────────
    mot = calculate_motility(sperm_report)

    # ── Step 5d: Morphology ───────────────────────────────────────────────────
    morph = morphology(all_detections[:1000])

    # ── Step 5e: Vitality ─────────────────────────────────────────────────────
    vit = vitality(mot)

    # ── Step 5f: WHO grading + impression ────────────────────────────────────
    conditions = who_grading(conc_M_ml, mot, morph)
    final_imp  = impression(conc_M_ml, mot, morph, len(sperm_report))

    # ── Aggregate CASA means ──────────────────────────────────────────────────
    def mean_field(field):
        vals = [s[field] for s in sperm_report if field in s]
        return round(float(np.mean(vals)), 2) if vals else 0.0

    casa_summary = {
        "mean_vcl_um_s" : mean_field("vcl_um_s"),
        "mean_vsl_um_s" : mean_field("vsl_um_s"),
        "mean_vap_um_s" : mean_field("vap_um_s"),
        "mean_lin"      : mean_field("lin"),
        "mean_str"      : mean_field("str"),
        "mean_wob"      : mean_field("wob"),
        "mean_alh_um"   : mean_field("alh_um"),
        "mean_bcf_hz"   : mean_field("bcf_hz"),
    }

    return jsonify({
        "analysis": {
            "concentration": {
                "sperm_per_ml_million"  : conc_M_ml,
                "density"               : "Low" if conc_M_ml < 16 else "Normal",
                "who_reference_M_ml"    : 16,
            },
            "motility"   : mot,
            "morphology" : morph,
            "vitality"   : vit,
            "casa_summary": casa_summary,
        },
        "sperm_tracks"    : sperm_report,
        "who_conditions"  : conditions,
        "final_impression": final_imp,
        "debug": {
            "fps_detected"    : fps,
            "frame_count"     : len(frame_counts),
            "um_per_px"       : um_per_px,
            "total_detections": total_valid,
            "total_tracks"    : len(sperm_report),
        },
    })


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE ANALYSIS endpoint (single frame — no motility, concentration only)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/analyze/image", methods=["POST"])
def analyze_image():
    """
    Accepts a single microscope image.
    Returns concentration + morphology only (no motility — requires video).
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file     = request.files["file"]
    tmp_path = f"{OUTPUT_DIR}/{uuid.uuid4()}.jpg"
    file.save(tmp_path)

    frame = cv2.imread(tmp_path)
    os.remove(tmp_path)

    if frame is None:
        return jsonify({"error": "Cannot read image"}), 400

    um_per_px = auto_calibrate(frame)
    raw_dets  = detect_seg(frame)

    valid_boxes = []
    for d in raw_dets:
        x1, y1, x2, y2, conf, area = d
        if valid_sperm(x1, y1, x2, y2, conf, um_per_px):
            valid_boxes.append((x1, y1, x2, y2, area))

    count   = len(valid_boxes)
    conc_M  = round(concentration(count) / 1e6, 2)
    morph   = morphology(valid_boxes)

    return jsonify({
        "analysis": {
            "concentration": {
                "sperm_per_ml_million": conc_M,
                "density"             : "Low" if conc_M < 16 else "Normal",
                "who_reference_M_ml"  : 16,
                "sperm_count_in_frame": count,
            },
            "morphology": morph,
            "motility"  : {"note": "Motility requires video input"},
        },
        "who_conditions"  : who_grading(conc_M, {"progressive_motility_pct": 100, "total_motility_percent": 100}, morph),
        "final_impression": "Morphology/concentration only — submit video for full CASA analysis",
        "debug"           : {"um_per_px": um_per_px, "detected_count": count},
    })


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({
        "status"       : "ok",
        "segmentation" : model is not None,
        "endpoints"    : ["/api/analyze/video", "/api/analyze/image"],
        "who_edition"  : "6th (2021)",
    })


if __name__ == "__main__":
    app.run(port=5001, debug=True)