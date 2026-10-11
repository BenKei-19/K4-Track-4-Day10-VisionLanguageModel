import json
import os
from pathlib import Path

import pytest
from PIL import Image

from scripts.labcommon import read_jsonl
from scripts.prepare_data import _group_stats, load_records, prepare
from scripts.splitting import assign_splits

FILES = [
    "vlcsnap-2025-02-18-17h00m00s001.jpg", "vlcsnap-2025-02-18-18h00m00s001.jpg",
    "vlcsnap-2025-02-19-14h00m00s001.jpg", "20250219_164649.jpg",
    "20250223_101010.jpg", "vlcsnap-00001.jpg", "vlcsnap-00060.jpg", "vlcsnap-00120.jpg",
]
LABEL = "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n2 0.7 0.7 0.1 0.1\n"
NO_MIN = {"train": 0, "val": 0, "test": 0}


def _make_dataset(root: Path, files, labels):
    (root / "images").mkdir(parents=True)
    (root / "labels-YOLO").mkdir(parents=True)
    for name in files:
        Image.new("RGB", (640, 360), (10, 10, 10)).save(root / "images" / name)
    for stem, text in labels.items():
        (root / "labels-YOLO" / f"{stem}.txt").write_text(text, encoding="utf-8")


def _labels(files, text=LABEL):
    return {Path(f).stem: text for f in files}


def test_load_records_parses_objects_and_sessions(tmp_path):
    _make_dataset(tmp_path, FILES, _labels(FILES))
    recs = load_records(tmp_path)
    assert len(recs) == len(FILES)
    r = next(x for x in recs if x["filename"] == "20250219_164649.jpg")
    assert (r["width"], r["height"], r["session"]) == (640, 360, "phone-20250219")
    assert [o["label"] for o in r["objects"]] == ["pothole", "crack", "manhole"]
    assert r["objects"][0]["yolo"] == [0.5, 0.5, 0.2, 0.2]


@pytest.mark.parametrize("bad,msg", [
    ("5 0.5 0.5 0.2 0.2\n", "class"),
    ("0 1.5 0.5 0.2 0.2\n", "range"),
    ("0 0.5 0.5\n", "malformed"),
    ("0 a b c d\n", "malformed"),
])
def test_bad_label_lines_name_the_file(tmp_path, bad, msg):
    labels = _labels(FILES)
    labels[Path(FILES[0]).stem] = bad
    _make_dataset(tmp_path, FILES, labels)
    with pytest.raises(ValueError) as e:
        load_records(tmp_path)
    assert FILES[0] in str(e.value) and msg in str(e.value)


def test_missing_label_file_is_loud(tmp_path):
    labels = _labels(FILES)
    del labels[Path(FILES[1]).stem]
    _make_dataset(tmp_path, FILES, labels)
    with pytest.raises(ValueError, match=FILES[1]):
        load_records(tmp_path)


def test_unknown_filename_is_loud(tmp_path):
    files = FILES + ["IMG_0001.jpg"]
    _make_dataset(tmp_path, files, _labels(files))
    with pytest.raises(ValueError, match="IMG_0001.jpg"):
        load_records(tmp_path)


def test_prepare_writes_splits_meta_and_images_and_is_reproducible(tmp_path):
    src = tmp_path / "src"
    _make_dataset(src, FILES, _labels(FILES))
    m1 = prepare(src, tmp_path / "o1", min_frac=NO_MIN, min_class_boxes=0)
    m2 = prepare(src, tmp_path / "o2", min_frac=NO_MIN, min_class_boxes=0)
    assert m1["dataset_version"] == m2["dataset_version"]       # cố định cho cả lớp
    out = tmp_path / "o1"
    rows = {s: read_jsonl(out / "splits" / f"{s}.jsonl") for s in ("train", "val", "test")}
    assert sum(len(v) for v in rows.values()) == len(FILES)
    sessions = {}
    for s, v in rows.items():
        for r in v:
            sessions.setdefault(r["session"], set()).add(s)
            assert (out / r["image_path"]).exists()
    assert all(len(v) == 1 for v in sessions.values())          # không phiên nào ở hai split
    on_disk = json.loads((out / "splits" / "meta.json").read_text())
    assert on_disk["dataset_version"] == m1["dataset_version"]


@pytest.mark.skipif(not os.environ.get("D10_RAW"), reason="đặt D10_RAW=<thư mục dataset> để chạy")
def test_real_data_default_split_is_valid():
    groups = _group_stats(load_records(os.environ["D10_RAW"]))
    assign, _ = assign_splits(groups)        # ném RuntimeError nếu không chia được
    assert set(assign.values()) == {"train", "val", "test"}
