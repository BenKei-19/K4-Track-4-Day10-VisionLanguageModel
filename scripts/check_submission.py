"""Kiểm tra bài nộp Lab Day 10 trước khi nộp. KHÔNG sửa file này."""
import json
import subprocess
import sys
from pathlib import Path

REQUIRED = ("e0", "e1", "e2", "e3")
TRAINED = ("e1", "e2", "e3")
META_KEYS = ("trainable_params", "peak_vram_gb", "train_minutes")
FORBIDDEN_PREFIX = ("data/", "work/")
FORBIDDEN_NAMES = (".env",)
FORBIDDEN_SUFFIX = (".safetensors", ".bin", ".pt", ".ckpt")
MIN_COMMITS = 5


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def run_checks(root):
    root = Path(root)
    res = []

    def add(st, name, detail=""):
        res.append((st, name, detail))

    ev_dir = root / "report" / "evidence"
    ev = {}
    for key in REQUIRED:
        p = ev_dir / f"{key}.json"
        if not p.exists():
            add("FAIL", f"evidence {key}", f"thiếu report/evidence/{key}.json")
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as e:
            add("FAIL", f"evidence {key}", f"JSON hỏng: {e}")
            continue
        ev[key] = d
        problems = []
        if d.get("score_version") != 1:
            problems.append("score_version != 1")
        if d.get("split") != "test":
            problems.append("split != test")
        if d.get("missing_predictions", 1) != 0:
            problems.append(f"missing_predictions={d.get('missing_predictions')}")
        add("FAIL" if problems else "PASS", f"evidence {key}", "; ".join(problems))
    e4 = sorted(ev_dir.glob("e4*.json")) if ev_dir.exists() else []
    if e4:
        for p in e4:
            try:
                ev[p.stem] = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                add("FAIL", f"evidence {p.stem}", "JSON hỏng")
        add("PASS", "evidence e4", f"{len(e4)} file")
    else:
        add("FAIL", "evidence e4", "cần ít nhất một file e4*.json")

    for key in TRAINED:
        if key not in ev:
            continue
        meta = ev[key].get("meta", {})
        bad = [k for k in META_KEYS
               if not isinstance(meta.get(k), (int, float)) or isinstance(meta.get(k), bool)
               or meta[k] <= 0]
        add("FAIL" if bad else "PASS", f"{key} meta", ", ".join(bad))

    vers = {d.get("dataset_version") for d in ev.values()}
    add("PASS" if len(vers) <= 1 else "FAIL", "dataset_version nhất quán", str(sorted(map(str, vers))))
    meta_path = root / "data" / "splits" / "meta.json"
    if meta_path.exists():
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        ok = vers <= {m.get("dataset_version")}
        add("PASS" if ok else "FAIL", "khớp data/splits/meta.json",
            "" if ok else "bằng chứng không sinh từ split chuẩn (chạy lại prepare_data.py)")
    else:
        add("WARN", "khớp data/splits/meta.json", "không có data/splits/meta.json để đối chiếu")

    rp = root / "REPORT.md"
    if not rp.exists():
        add("FAIL", "REPORT.md", "thiếu REPORT.md")
    else:
        add("FAIL" if "<điền" in rp.read_text(encoding="utf-8") else "PASS",
            "REPORT.md đã điền", "còn chuỗi '<điền'")

    if (root / ".git").exists():
        n = _git(root, "rev-list", "--count", "HEAD")
        count = int(n.stdout.strip() or 0) if n.returncode == 0 else 0
        add("PASS" if count >= MIN_COMMITS else "FAIL", "git commits", f"{count}/{MIN_COMMITS}")
        files = _git(root, "ls-files").stdout.splitlines()
        bad = [f for f in files if f.startswith(FORBIDDEN_PREFIX)
               or f.split("/")[-1] in FORBIDDEN_NAMES or f.endswith(FORBIDDEN_SUFFIX)]
        add("FAIL" if bad else "PASS", "git forbidden files", ", ".join(bad[:5]))
    else:
        add("WARN", "git", "không phải git repo")
    return res


def main():
    root = Path(__file__).resolve().parents[1]
    res = run_checks(root)
    for st, name, detail in res:
        print(f"[{st}] {name}" + (f" — {detail}" if detail else ""))
    fails = sum(1 for r in res if r[0] == "FAIL")
    print(f"\n{fails} FAIL, {sum(1 for r in res if r[0] == 'WARN')} WARN")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
