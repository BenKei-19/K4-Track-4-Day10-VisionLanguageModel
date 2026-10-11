import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = {n: (ROOT / n).read_text(encoding="utf-8")
        for n in ("README.md", "GUIDE.md", "RUBRIC.md", "REPORT.md")}


def _cfg(n):
    return yaml.safe_load((ROOT / "configs" / f"e{n}.yaml").read_text(encoding="utf-8"))


def test_all_configs_parse_and_have_name():
    for n in range(6):
        d = _cfg(n)
        assert d["name"] == f"E{n}" and d["description"]


def test_trained_configs_differ_only_in_the_variable():
    cfgs = {n: _cfg(n) for n in (1, 2, 3)}
    for k in ("lr", "batch_size", "grad_accum", "max_steps", "seed"):
        assert len({c[k] for c in cfgs.values()}) == 1, k
    assert [cfgs[n]["train_projector"] for n in (1, 2, 3)] == [False, True, True]
    assert [cfgs[n]["lora"]["targets"] for n in (1, 2, 3)] == ["llm", "llm", "llm+vit"]


def test_notebook_sections_and_no_secrets():
    nb = json.loads((ROOT / "lab_day10.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    heads = [c["source"][0] for c in nb["cells"] if c["cell_type"] == "markdown" and c["source"]]
    for i in range(9):
        assert any(h.startswith(f"## {i}.") for h in heads), i
    text = json.dumps(nb)
    assert not re.search(r"KAGGLE_KEY'?\]?\s*=\s*['\"][0-9a-f]{20,}", text)


def test_no_placeholders_in_readme_guide_rubric():
    for n in ("README.md", "GUIDE.md", "RUBRIC.md"):
        assert not re.search(r"\bTBD\b|<điền", DOCS[n]), n


def test_report_template_has_student_markers_and_ten_questions():
    assert "<điền" in DOCS["REPORT.md"]
    assert len(re.findall(r"^### Câu \d+", DOCS["REPORT.md"], re.M)) == 10


def test_docs_mention_real_commands_and_verified_facts():
    g = DOCS["GUIDE.md"]
    for token in ("prepare_data.py", "score.py", "bbox_2d", "0–1000", "llm_int8_skip_modules",
                  "visual.merger", "visual.deepstack_merger_list", "language_model",
                  "report/evidence", "220", "252", "348"):
        assert token in g, token
    assert "check_submission.py" in DOCS["README.md"]


def test_rubric_sums_to_100():
    pts = [int(x) for x in re.findall(r"^\|\s*[A-D]\.[^|]*\|\s*(\d+)\s*\|", DOCS["RUBRIC.md"], re.M)]
    assert sum(pts) == 100, pts
    text = DOCS["RUBRIC.md"]
    for head, nxt, total in (("## A.", "## B.", 40), ("## B.", "## Bonus", 20)):
        section = text[text.index(head):text.index(nxt)]
        sub = [int(x) for x in re.findall(r"^\|[^|\n]*\|\s*(\d+)\s*\|", section, re.M)]
        assert sum(sub) == total, (head, sub)


def test_readme_split_table_matches_real_split_sizes():
    # số liệu đã đo bằng prepare_data.py trên dữ liệu thật (dataset_version d10-21e4070301)
    for n in ("1.422", "276", "311"):
        assert n in DOCS["README.md"]
