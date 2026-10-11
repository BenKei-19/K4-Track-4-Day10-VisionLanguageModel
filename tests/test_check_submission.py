import json
import subprocess

from scripts.check_submission import run_checks


def _ev(name, ver="d10-x", trained=True, **over):
    d = {"score_version": 1, "name": name, "dataset_version": ver,
         "split": "test", "missing_predictions": 0, "micro": {"f1": 0.5},
         "meta": {"trainable_params": 1e6, "peak_vram_gb": 9.0, "train_minutes": 12.0}
         if trained else {}}
    d.update(over)
    return d


def _setup(root, evs, report="Họ tên: A\n\nNội dung đã điền.\n", meta=True):
    ev = root / "report" / "evidence"
    ev.mkdir(parents=True)
    for fname, d in evs.items():
        (ev / fname).write_text(json.dumps(d))
    (root / "REPORT.md").write_text(report, encoding="utf-8")
    if meta:
        (root / "data" / "splits").mkdir(parents=True)
        (root / "data" / "splits" / "meta.json").write_text(json.dumps({"dataset_version": "d10-x"}))


def _good():
    return {"e0.json": _ev("E0", trained=False), "e1.json": _ev("E1"), "e2.json": _ev("E2"),
            "e3.json": _ev("E3"), "e4_rank.json": _ev("E4")}


def _has(res, status, text):
    return any(st == status and text in name for st, name, _ in res)


def test_all_good_has_no_fail(tmp_path):
    _setup(tmp_path, _good())
    res = run_checks(tmp_path)
    assert not [r for r in res if r[0] == "FAIL"], res


def test_missing_required_evidence_fails(tmp_path):
    g = _good(); del g["e2.json"]
    _setup(tmp_path, g)
    assert _has(run_checks(tmp_path), "FAIL", "e2")


def test_missing_e4_fails(tmp_path):
    g = _good(); del g["e4_rank.json"]
    _setup(tmp_path, g)
    assert _has(run_checks(tmp_path), "FAIL", "e4")


def test_dataset_version_mismatch_between_files_fails(tmp_path):
    g = _good(); g["e3.json"] = _ev("E3", ver="d10-other")
    _setup(tmp_path, g)
    assert _has(run_checks(tmp_path), "FAIL", "dataset_version")


def test_version_not_matching_local_meta_fails(tmp_path):
    g = {k: _ev(v["name"], ver="d10-zzz", trained=v["meta"] != {}) for k, v in _good().items()}
    _setup(tmp_path, g)
    assert _has(run_checks(tmp_path), "FAIL", "meta.json")


def test_missing_trained_meta_fails(tmp_path):
    g = _good(); g["e1.json"] = _ev("E1", trained=False)
    _setup(tmp_path, g)
    assert _has(run_checks(tmp_path), "FAIL", "e1 meta")


def test_missing_predictions_fails(tmp_path):
    g = _good(); g["e1.json"] = _ev("E1", missing_predictions=3)
    _setup(tmp_path, g)
    assert _has(run_checks(tmp_path), "FAIL", "e1")


def test_report_placeholder_fails(tmp_path):
    _setup(tmp_path, _good(), report="<điền câu trả lời>\n")
    assert _has(run_checks(tmp_path), "FAIL", "REPORT")


def test_corrupt_evidence_json_fails_not_crashes(tmp_path):
    _setup(tmp_path, _good())
    (tmp_path / "report" / "evidence" / "e1.json").write_text("{not json")
    assert _has(run_checks(tmp_path), "FAIL", "e1")


def test_no_git_is_warning_only(tmp_path):
    _setup(tmp_path, _good())
    assert _has(run_checks(tmp_path), "WARN", "git")


def test_git_commit_count_and_forbidden_files(tmp_path):
    _setup(tmp_path, _good())

    def run(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)

    run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
    (tmp_path / "model.safetensors").write_text("x")
    run("add", "-A"); run("commit", "-qm", "one")
    res = run_checks(tmp_path)
    assert _has(res, "FAIL", "commits") and _has(res, "FAIL", "forbidden")
