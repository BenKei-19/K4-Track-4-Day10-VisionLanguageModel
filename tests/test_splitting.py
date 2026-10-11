import pytest

from scripts.labcommon import CLASSES, read_jsonl, write_jsonl
from scripts.splitting import SPLITS, assign_splits, dataset_version, session_key


def test_jsonl_roundtrip(tmp_path):
    rows = [{"id": "a", "t": "đường"}, {"id": "b", "n": [1, 2]}]
    p = tmp_path / "x" / "y.jsonl"
    write_jsonl(p, rows)
    assert read_jsonl(p) == rows
    assert CLASSES == ("pothole", "crack", "manhole")


@pytest.mark.parametrize("name,expected", [
    ("vlcsnap-2025-02-18-23h08m14s528.jpg", "vlc-2025-02-18-23"),
    ("vlcsnap_2025-03-16-15h16m59s824.jpg", "vlc-2025-03-16-15"),
    ("20250219_164649.jpg", "phone-20250219"),
    ("vlcsnap-00050.jpg", "num-1"),
    ("vlcsnap-00049.jpg", "num-0"),
    ("vlcsnap-00097-MSI.jpg", "num-1"),
    ("vlcsnap-00214.jpg", "num-4"),
])
def test_session_key_known_patterns(name, expected):
    assert session_key(name) == expected


def test_session_key_unknown_name_is_loud():
    with pytest.raises(ValueError, match="IMG_001.jpg"):
        session_key("IMG_001.jpg")


def _real_like_groups():
    sizes = {
        "vlc-2025-02-18-17": 124, "vlc-2025-02-18-18": 244, "vlc-2025-02-18-23": 213,
        "vlc-2025-02-19-13": 55, "vlc-2025-02-19-14": 475, "vlc-2025-02-19-15": 116,
        "vlc-2025-02-19-16": 32, "vlc-2025-02-19-17": 83, "vlc-2025-02-26-20": 217,
        "vlc-2025-03-16-15": 136, "phone-20250216": 3, "phone-20250219": 60,
        "phone-20250223": 39, "num-0": 49, "num-1": 51, "num-2": 50, "num-3": 50, "num-4": 12,
    }
    mix = {"crack": 0.53, "pothole": 0.27, "manhole": 0.20}
    return {k: {"images": n, "boxes": {c: round(n * 2.36 * f) for c, f in mix.items()}}
            for k, n in sizes.items()}


def test_assign_is_deterministic_and_covers_all_groups():
    g = _real_like_groups()
    assert assign_splits(g) == assign_splits(g)
    a, _ = assign_splits(g)
    assert set(a) == set(g) and set(a.values()) == set(SPLITS)


def test_assign_valid_for_many_seeds():
    g = _real_like_groups()
    total = sum(v["images"] for v in g.values())
    for i in range(100):
        a, _ = assign_splits(g, seed=f"s{i}")
        for split in SPLITS:
            assert sum(g[k]["images"] for k, s in a.items() if s == split) >= 0.10 * total
        for split in ("val", "test"):
            for c in CLASSES:
                assert sum(g[k]["boxes"][c] for k, s in a.items() if s == split) >= 20


def test_assign_raises_when_impossible():
    g = {"only": {"images": 10, "boxes": {"pothole": 1, "crack": 1, "manhole": 1}}}
    with pytest.raises(RuntimeError):
        assign_splits(g, max_salt=5)


def test_dataset_version_stable_and_sensitive():
    rows = [("a", "train"), ("b", "test")]
    v = dataset_version(rows)
    assert v.startswith("d10-") and len(v) == 14
    assert v == dataset_version(list(reversed(rows)))
    assert v != dataset_version([("a", "test"), ("b", "train")])
