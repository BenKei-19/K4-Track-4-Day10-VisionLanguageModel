"""Trọng tài chấm lab Day 10: parse output, F1@IoU0.5, CI bootstrap. KHÔNG sửa file này."""
import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.labcommon import CLASSES, read_jsonl  # noqa: E402

SCORE_VERSION = 1
IOU_THR = 0.5
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse_output(raw):
    """Trả về list[{"label","bbox"}] hợp lệ, [] nếu '[]', hoặc None nếu hỏng."""
    text = (raw or "").strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        i, j = text.find("["), text.rfind("]")
        if i < 0 or j <= i:
            return None
        try:
            obj = json.loads(text[i:j + 1])
        except (ValueError, TypeError):
            return None
    if not isinstance(obj, list):
        return None
    out = []
    for o in obj:
        if not isinstance(o, dict):
            return None
        lab, bb = o.get("label"), o.get("bbox_2d")
        if not isinstance(lab, str) or not isinstance(bb, list) or len(bb) != 4:
            return None
        if not all(_is_num(v) for v in bb):
            return None
        x1, y1, x2, y2 = (float(v) for v in bb)
        if x2 <= x1 or y2 <= y1:
            return None
        out.append({"label": lab, "bbox": (x1, y1, x2, y2)})
    return out


def iou_xyxy(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _gt_boxes(entry):
    out = []
    for o in entry["objects"]:
        xc, yc, w, h = o["yolo"]
        out.append({"label": o["label"], "bbox": (
            (xc - w / 2) * 1000, (yc - h / 2) * 1000, (xc + w / 2) * 1000, (yc + h / 2) * 1000)})
    return out


def _to_norm1000(pred, coord, width, height):
    if coord == "norm1000":
        return pred
    return [{"label": p["label"], "bbox": (
        p["bbox"][0] / width * 1000, p["bbox"][1] / height * 1000,
        p["bbox"][2] / width * 1000, p["bbox"][3] / height * 1000)} for p in pred]


def _match(gt, pred):
    per = {c: [0, 0, 0] for c in CLASSES}
    per["other"] = [0, 0, 0]
    ious = []
    for lab in set(CLASSES) | {p["label"] for p in pred}:
        g = [x["bbox"] for x in gt if x["label"] == lab]
        p = [x["bbox"] for x in pred if x["label"] == lab]
        pairs = sorted(((iou_xyxy(a, b), i, j) for i, a in enumerate(g)
                        for j, b in enumerate(p)), reverse=True)
        used_g, used_p, tp = set(), set(), 0
        for v, i, j in pairs:
            if v < IOU_THR:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            tp += 1
            ious.append(v)
        key = lab if lab in per else "other"
        per[key][0] += tp
        per[key][1] += len(p) - tp
        per[key][2] += len(g) - tp
    return per, ious


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def _bootstrap_f1(rows, n_boot, seed):
    rng, n, vals = random.Random(seed), len(rows), []
    for _ in range(n_boot):
        tp = fp = fn = 0
        for _ in range(n):
            r = rows[rng.randrange(n)]
            tp, fp, fn = tp + r[0], fp + r[1], fn + r[2]
        vals.append(_prf(tp, fp, fn)["f1"])
    vals.sort()
    return [vals[int(0.025 * n_boot)], vals[max(int(0.975 * n_boot) - 1, 0)]]


def evaluate(entries, preds, coord="norm1000", n_boot=1000, seed=0):
    per_tot = {c: [0, 0, 0] for c in (*CLASSES, "other")}
    rows, ious, valid, missing = [], [], 0, 0
    for e in entries:
        raw = preds.get(e["image_id"])
        parsed = None
        if raw is None:
            missing += 1
        else:
            parsed = parse_output(raw)
        pred = []
        if parsed is not None:
            valid += 1
            pred = _to_norm1000(parsed, coord, e["width"], e["height"])
        per, i = _match(_gt_boxes(e), pred)
        ious += i
        rows.append((sum(v[0] for v in per.values()), sum(v[1] for v in per.values()),
                     sum(v[2] for v in per.values())))
        for c, v in per.items():
            for k in range(3):
                per_tot[c][k] += v[k]
    n = len(entries)
    tp, fp, fn = (sum(r[k] for r in rows) for k in range(3))
    return {
        "n_images": n,
        "parse_valid_rate": valid / n if n else 0.0,
        "missing_predictions": missing,
        "micro": _prf(tp, fp, fn),
        "per_class": {c: _prf(*v) for c, v in per_tot.items()},
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "fp_per_image": fp / n if n else 0.0,
        "f1_ci95": _bootstrap_f1(rows, n_boot, seed) if n else [0.0, 0.0],
    }


def _parse_meta(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k] = float(v) if "." in v else int(v)
        except ValueError:
            out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", action="append",
                    help="key=value (trainable_params, peak_vram_gb, train_minutes…)")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)

    sd = Path(a.splits_dir)
    entries = read_jsonl(sd / f"{a.split}.jsonl")
    meta = json.loads((sd / "meta.json").read_text(encoding="utf-8"))
    preds = {}
    for row in read_jsonl(a.pred):
        if row["image_id"] in preds:
            sys.exit(f"duplicate image_id in predictions: {row['image_id']}")
        preds[row["image_id"]] = row["raw"]
    extra = set(preds) - {e["image_id"] for e in entries}
    if extra:
        print(f"cảnh báo: bỏ qua {len(extra)} image_id không thuộc split {a.split}", file=sys.stderr)

    res = evaluate(entries, preds, coord=a.coord, n_boot=a.n_boot)
    res.update({
        "score_version": SCORE_VERSION, "name": a.name,
        "dataset_version": meta["dataset_version"], "split": a.split, "coord": a.coord,
        "meta": _parse_meta(a.meta),
    })
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    m = res["micro"]
    print(f"{a.name} [{a.split}] n={res['n_images']} valid={res['parse_valid_rate']:.3f} "
          f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
          f"CI95={res['f1_ci95'][0]:.3f}-{res['f1_ci95'][1]:.3f} meanIoU={res['mean_iou']:.3f} "
          f"FP/img={res['fp_per_image']:.2f}")
    for c, v in res["per_class"].items():
        if c != "other" or v["fp"]:
            print(f"   {c:8s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}")


if __name__ == "__main__":
    main()
