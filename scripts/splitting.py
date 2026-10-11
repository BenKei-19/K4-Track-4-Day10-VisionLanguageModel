"""Gom ảnh theo phiên quay và gán split cố định (chống rò rỉ giữa các frame liền kề)."""
import hashlib
import re

SPLITS = ("train", "val", "test")
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}
MIN_FRAC = {"train": 0.50, "val": 0.10, "test": 0.10}
MIN_CLASS_BOXES = 20
SEED = "day10"

_PATTERNS = [
    (re.compile(r"^vlcsnap[-_](\d{4}-\d{2}-\d{2})-(\d{2})h\d{2}m\d{2}s\d+\.jpg$"),
     lambda m: f"vlc-{m.group(1)}-{m.group(2)}"),
    (re.compile(r"^(\d{8})_\d{6}\.jpg$"),
     lambda m: f"phone-{m.group(1)}"),
    (re.compile(r"^vlcsnap-(\d+)(?:-[A-Za-z]+)?\.jpg$"),
     lambda m: f"num-{int(m.group(1)) // 50}"),
]


def session_key(filename):
    for pat, fn in _PATTERNS:
        m = pat.match(filename)
        if m:
            return fn(m)
    raise ValueError(f"unrecognised image filename: {filename}")


def _attempt(groups, seed, salt):
    order = sorted(
        groups,
        key=lambda k: hashlib.sha256(f"{seed}:{salt}:{k}".encode()).hexdigest(),
    )
    total = sum(g["images"] for g in groups.values())
    cur = {s: 0 for s in SPLITS}
    out = {}
    for k in order:
        s = max(SPLITS, key=lambda s: TARGET[s] * total - cur[s])
        out[k] = s
        cur[s] += groups[k]["images"]
    return out


def _is_valid(groups, assign, min_frac, min_class_boxes):
    total = sum(g["images"] for g in groups.values())
    for s in SPLITS:
        imgs = sum(groups[k]["images"] for k, v in assign.items() if v == s)
        if imgs < min_frac[s] * total or imgs == 0:
            return False
    classes = {c for g in groups.values() for c in g["boxes"]}
    for s in ("val", "test"):
        for c in classes:
            n = sum(groups[k]["boxes"].get(c, 0) for k, v in assign.items() if v == s)
            if n < min_class_boxes:
                return False
    return True


def assign_splits(groups, seed=SEED, min_frac=MIN_FRAC,
                  min_class_boxes=MIN_CLASS_BOXES, max_salt=2000):
    """Gán mỗi phiên vào một split. Cùng (groups, seed) luôn ra cùng kết quả."""
    for salt in range(max_salt):
        a = _attempt(groups, seed, salt)
        if _is_valid(groups, a, min_frac, min_class_boxes):
            return a, salt
    raise RuntimeError(f"no valid split for seed {seed!r} within {max_salt} salts")


def dataset_version(image_splits):
    h = hashlib.sha256()
    for image_id, split in sorted(image_splits):
        h.update(f"|{image_id}:{split}".encode())
    return "d10-" + h.hexdigest()[:10]
