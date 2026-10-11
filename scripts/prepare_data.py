"""Tải dataset road damage, kiểm tra, chia split theo phiên quay (cố định). KHÔNG sửa file này."""
import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from scripts.labcommon import CLASSES, write_jsonl  # noqa: E402
from scripts.splitting import (  # noqa: E402
    MIN_CLASS_BOXES, MIN_FRAC, SEED, SPLITS, assign_splits, dataset_version, session_key,
)

KAGGLE_SLUG = "lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes"


def _parse_label_file(path, filename):
    objs = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            if len(parts) < 5:
                raise ValueError
            cid = int(parts[0])
            xc, yc, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            raise ValueError(f"{filename}: malformed label line {ln}: {line!r}") from None
        if not 0 <= cid < len(CLASSES):
            raise ValueError(f"{filename}: class id {cid} outside 0..{len(CLASSES) - 1} (line {ln})")
        if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
            raise ValueError(f"{filename}: value out of range [0,1] on line {ln}: {line!r}")
        objs.append({"label": CLASSES[cid], "yolo": [xc, yc, w, h]})
    return objs


def load_records(src):
    src = Path(src)
    images = sorted(p for p in (src / "images").iterdir() if p.suffix.lower() == ".jpg")
    if not images:
        raise ValueError(f"no .jpg under {src / 'images'}")
    records = []
    for p in images:
        label = src / "labels-YOLO" / f"{p.stem}.txt"
        if not label.exists():
            raise ValueError(f"{p.name}: missing label file {label.name}")
        with Image.open(p) as im:
            width, height = im.size
        records.append({
            "image_id": p.stem, "filename": p.name, "width": width, "height": height,
            "session": session_key(p.name), "objects": _parse_label_file(label, p.name),
        })
    return records


def _group_stats(records):
    groups = {}
    for r in records:
        g = groups.setdefault(r["session"], {"images": 0, "boxes": {c: 0 for c in CLASSES}})
        g["images"] += 1
        for o in r["objects"]:
            g["boxes"][o["label"]] += 1
    return groups


def prepare(src, out, seed=SEED, min_frac=MIN_FRAC, min_class_boxes=MIN_CLASS_BOXES):
    src, out = Path(src), Path(out)
    records = load_records(src)
    groups = _group_stats(records)
    assign, salt = assign_splits(groups, seed=seed, min_frac=min_frac,
                                 min_class_boxes=min_class_boxes)
    (out / "images").mkdir(parents=True, exist_ok=True)
    per_split = {s: [] for s in SPLITS}
    counts = {s: {"images": 0, "boxes": {c: 0 for c in CLASSES}} for s in SPLITS}
    for r in records:
        s = assign[r["session"]]
        shutil.copy2(src / "images" / r["filename"], out / "images" / r["filename"])
        per_split[s].append({
            "image_id": r["image_id"], "image_path": f"images/{r['filename']}",
            "width": r["width"], "height": r["height"],
            "session": r["session"], "objects": r["objects"],
        })
        counts[s]["images"] += 1
        for o in r["objects"]:
            counts[s]["boxes"][o["label"]] += 1
    for s in SPLITS:
        write_jsonl(out / "splits" / f"{s}.jsonl", per_split[s])
    meta = {
        "dataset_version": dataset_version([(r["image_id"], assign[r["session"]]) for r in records]),
        "seed": seed, "salt": salt, "classes": list(CLASSES), "counts": counts, "groups": assign,
    }
    (out / "splits" / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _find_root(base):
    for p in [Path(base)] + sorted(Path(base).rglob("*")):
        if p.is_dir() and (p / "images").is_dir() and (p / "labels-YOLO").is_dir():
            return p
    raise FileNotFoundError(f"cannot find images/ and labels-YOLO/ under {base}")


def _fetch(args, work):
    work.mkdir(parents=True, exist_ok=True)
    if args.source:
        return _find_root(args.source)
    if args.zip:
        z = args.zip
        if z.startswith(("http://", "https://")):
            dest = work / "dataset.zip"
            urllib.request.urlretrieve(z, dest)
            z = str(dest)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(work / "raw")
        return _find_root(work / "raw")
    subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p",
                    str(work / "raw"), "--unzip"], check=True)
    return _find_root(work / "raw")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--source", help="thư mục đã giải nén (có images/ và labels-YOLO/)")
    g.add_argument("--zip", help="đường dẫn hoặc URL file zip của dataset")
    g.add_argument("--kaggle", action="store_true",
                   help="tải bằng Kaggle CLI (cần KAGGLE_USERNAME/KAGGLE_KEY)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--work", default="work")
    args = ap.parse_args(argv)
    root = _fetch(args, Path(args.work))
    meta = prepare(root, args.out)
    print(f"dataset_version : {meta['dataset_version']}")
    for s in SPLITS:
        c = meta["counts"][s]
        print(f"  {s:5s} {c['images']:5d} ảnh  boxes={c['boxes']}")


if __name__ == "__main__":
    main()
