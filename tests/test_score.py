import json

import pytest

from scripts import score
from scripts.score import evaluate, iou_xyxy, parse_output


def _box(label, x1, y1, x2, y2):
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def test_parse_plain_fenced_and_surrounded():
    s = json.dumps([_box("pothole", 10, 20, 30, 40)])
    want = [{"label": "pothole", "bbox": (10.0, 20.0, 30.0, 40.0)}]
    assert parse_output(s) == want
    assert parse_output(f"```json\n{s}\n```") == want
    assert parse_output(f"Here you go: {s} Hope this helps.") == want


def test_parse_empty_list_is_valid_and_empty():
    assert parse_output("[]") == []


@pytest.mark.parametrize("raw", [
    "", "not json", "{}", '{"label":"pothole"}', "[1,2]",
    '[{"label":"pothole"}]',
    '[{"label":"pothole","bbox_2d":[1,2,3]}]',
    '[{"label":"pothole","bbox_2d":[5,5,5,9]}]',
    '[{"label":"pothole","bbox_2d":[1,9,5,9]}]',
    '[{"label":"pothole","bbox_2d":["1","2","3","4"]}]',
    '[{"label":"pothole","bbox_2d":[true,2,3,4]}]',
    '[{"label":"pothole","bbox_2d":[NaN,2,3,4]}]',
    '[{"label":"pothole","bbox_2d":[1,2,Infinity,4]}]',
    '[{"label":7,"bbox_2d":[1,2,3,4]}]',
])
def test_parse_rejects_malformed_without_raising(raw):
    assert parse_output(raw) is None


def test_parse_none_input_does_not_raise():
    assert parse_output(None) is None


def test_iou():
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def _entry(image_id, objs, w=640, h=360):
    return {"image_id": image_id, "width": w, "height": h, "objects": objs}


def _obj(label, xc, yc, w, h):
    return {"label": label, "yolo": [xc, yc, w, h]}


GT_ONE = [_obj("pothole", 0.5, 0.5, 0.2, 0.2)]          # 0-1000: (400,400,600,600)
PERFECT = json.dumps([_box("pothole", 400, 400, 600, 600)])


def test_perfect_prediction():
    r = evaluate([_entry("a", GT_ONE)], {"a": PERFECT}, n_boot=20)
    assert r["micro"]["f1"] == 1.0 and r["mean_iou"] == pytest.approx(1.0)
    assert r["parse_valid_rate"] == 1.0 and r["missing_predictions"] == 0
    assert r["per_class"]["pothole"]["tp"] == 1


def test_wrong_class_is_fp_and_fn():
    raw = json.dumps([_box("crack", 400, 400, 600, 600)])
    r = evaluate([_entry("a", GT_ONE)], {"a": raw}, n_boot=20)
    assert (r["micro"]["tp"], r["micro"]["fp"], r["micro"]["fn"]) == (0, 1, 1)


def test_duplicate_prediction_counts_one_tp_one_fp():
    raw = json.dumps([_box("pothole", 400, 400, 600, 600)] * 2)
    r = evaluate([_entry("a", GT_ONE)], {"a": raw}, n_boot=20)
    assert (r["micro"]["tp"], r["micro"]["fp"], r["micro"]["fn"]) == (1, 1, 0)


def test_invalid_and_missing_outputs():
    entries = [_entry("a", GT_ONE), _entry("b", GT_ONE), _entry("c", GT_ONE)]
    r = evaluate(entries, {"a": PERFECT, "b": "garbage"}, n_boot=20)
    assert r["parse_valid_rate"] == pytest.approx(1 / 3)
    assert r["missing_predictions"] == 1
    assert r["micro"]["recall"] == pytest.approx(1 / 3)


def test_extra_prediction_ids_ignored():
    r = evaluate([_entry("a", GT_ONE)], {"a": PERFECT, "zzz": PERFECT}, n_boot=20)
    assert r["n_images"] == 1


def test_pixel_vs_norm1000_scale_trap():
    raw = json.dumps([_box("pothole", 256, 144, 384, 216)])   # đúng nếu là pixel của ảnh 640x360
    px = evaluate([_entry("a", GT_ONE)], {"a": raw}, coord="pixel", n_boot=20)
    nm = evaluate([_entry("a", GT_ONE)], {"a": raw}, coord="norm1000", n_boot=20)
    assert px["micro"]["f1"] == 1.0
    assert nm["micro"]["f1"] == 0.0 and nm["parse_valid_rate"] == 1.0


def test_unknown_label_counts_as_fp_other():
    raw = json.dumps([_box("pothole", 400, 400, 600, 600), _box("puddle", 1, 1, 50, 50)])
    r = evaluate([_entry("a", GT_ONE)], {"a": raw}, n_boot=20)
    assert r["micro"]["fp"] == 1 and r["per_class"]["other"]["fp"] == 1


def test_bootstrap_ci_contains_point_estimate_and_is_seeded():
    entries = [_entry(str(i), GT_ONE) for i in range(20)]
    preds = {str(i): (PERFECT if i % 2 else "[]") for i in range(20)}
    r1 = evaluate(entries, preds, n_boot=200, seed=0)
    r2 = evaluate(entries, preds, n_boot=200, seed=0)
    lo, hi = r1["f1_ci95"]
    assert lo <= r1["micro"]["f1"] <= hi and r1["f1_ci95"] == r2["f1_ci95"]


def _write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_cli_end_to_end_writes_json_with_meta(tmp_path):
    sp = tmp_path / "splits"
    _write(sp / "test.jsonl", [_entry("a", GT_ONE)])
    (sp / "meta.json").write_text(json.dumps({"dataset_version": "d10-abc"}))
    pred = tmp_path / "p.jsonl"
    _write(pred, [{"image_id": "a", "raw": PERFECT}])
    out = tmp_path / "ev" / "e1.json"
    score.main(["--pred", str(pred), "--splits-dir", str(sp), "--name", "E1",
                "--out", str(out), "--meta", "trainable_params=1000",
                "--meta", "peak_vram_gb=7.5", "--meta", "note=hello"])
    d = json.loads(out.read_text())
    assert d["name"] == "E1" and d["dataset_version"] == "d10-abc"
    assert d["score_version"] == 1 and d["split"] == "test"
    assert d["meta"] == {"trainable_params": 1000, "peak_vram_gb": 7.5, "note": "hello"}


def test_cli_duplicate_image_id_is_loud(tmp_path):
    sp = tmp_path / "splits"
    _write(sp / "test.jsonl", [_entry("a", GT_ONE)])
    (sp / "meta.json").write_text(json.dumps({"dataset_version": "d"}))
    pred = tmp_path / "p.jsonl"
    _write(pred, [{"image_id": "a", "raw": "[]"}, {"image_id": "a", "raw": "[]"}])
    with pytest.raises(SystemExit) as e:
        score.main(["--pred", str(pred), "--splits-dir", str(sp), "--name", "E1",
                    "--out", str(tmp_path / "o.json")])
    assert "duplicate" in str(e.value)
