"""Mã của tôi cho Lab Day 10 (GUIDE bước 2–8): nhãn JSON, suy luận, SFT QLoRA cho Qwen3-VL-4B.

    python lab/vlm.py check                                   # bước 2 + 4: round-trip nhãn, 220 token ảnh, mask loss
    python lab/vlm.py predict --name e0                       # E0 zero-shot → work/e0.jsonl + report/evidence/e0.json
    python lab/vlm.py train --name e1 --targets llm           # E1 (thêm --projector cho E2, --targets llm+vit cho E3)
    python lab/vlm.py predict --name e1 --adapter <thư mục adapter>
"""
import argparse
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.labcommon import read_jsonl, write_jsonl  # noqa: E402

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
SPLITS = ROOT / "data" / "splits"
LLM_RX = r"model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)"
VIT_RX = r"model\.visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc[12])"
PROJECTOR = ("visual.merger", "visual.deepstack_merger_list")


# ---------- Bước 2: nhãn YOLO → JSON ----------

def yolo_to_box(label, yolo, sx=1000, sy=1000):
    """sx, sy = 1000 cho thang norm1000; = width, height cho thang pixel."""
    xc, yc, w, h = yolo
    box = [round((xc - w / 2) * sx), round((yc - h / 2) * sy),
           round((xc + w / 2) * sx), round((yc + h / 2) * sy)]
    box = [min(max(v, 0), m) for v, m in zip(box, (sx, sy, sx, sy))]
    return {"label": label, "bbox_2d": box}


def scale(entry, coord):
    return (1000, 1000) if coord == "norm1000" else (entry["width"], entry["height"])


def objects_to_answer(entry, coord="norm1000"):
    sx, sy = scale(entry, coord)
    boxes = [yolo_to_box(o["label"], o["yolo"], sx, sy) for o in entry["objects"]]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))  # trên→dưới, trái→phải
    return json.dumps(boxes, separators=(",", ":"))


def prompt(entry, coord="norm1000"):
    if coord == "norm1000":
        unit = "integers on a 0-1000 scale relative to the image width and height"
    else:
        unit = f"integer pixel coordinates in the {entry['width']}x{entry['height']} image"
    return ("Detect every road damage in this dashcam image. Classes: pothole, crack, manhole. "
            'Answer ONLY with a JSON list like [{"label":"pothole","bbox_2d":[x1,y1,x2,y2]}], '
            f"where x1,y1,x2,y2 are {unit}. Answer [] if there is no damage.")


def user_msg(entry, coord):
    return {"role": "user", "content": [
        {"type": "image", "image": str(ROOT / "data" / entry["image_path"])},
        {"type": "text", "text": prompt(entry, coord)}]}


# ---------- Bước 3/7: nạp model, suy luận ----------

def load_model(adapter=None):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                               # transformers 5.x so khớp bằng re.match (từ đầu tên) nên "visual" trơn
                               # không khớp "model.visual.*" và ViT sẽ bị lượng tử hoá
                               llm_int8_skip_modules=["model.visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant, dtype=torch.float16, device_map={"": 0})
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        proj = Path(adapter) / "projector.pt"
        if proj.exists():  # projector huấn luyện đầy đủ không nằm trong adapter LoRA
            _, unexpected = model.load_state_dict(torch.load(proj), strict=False)
            assert not unexpected, unexpected[:5]
            print("đã nạp lại projector:", proj)
    return model, processor


def predict(model, processor, rows, coord, bs=8):
    import torch
    model.eval()
    processor.tokenizer.padding_side = "left"
    out = []
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        inputs = processor.apply_chat_template(
            [[user_msg(r, coord)] for r in batch], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        texts = processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        out += [{"image_id": r["image_id"], "raw": t} for r, t in zip(batch, texts)]
        if i // bs % 10 == 0:
            print(f"predict {i + len(batch)}/{len(rows)}  {texts[0][:120]!r}", flush=True)
    return out


# ---------- Bước 4: ví dụ huấn luyện và mask loss ----------

def encode_example(processor, entry, coord="norm1000"):
    import torch
    user = user_msg(entry, coord)
    asst = {"role": "assistant", "content": [{"type": "text", "text": objects_to_answer(entry, coord)}]}
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    p = processor.apply_chat_template([user], add_generation_prompt=True, **kw)
    full = processor.apply_chat_template([user, asst], add_generation_prompt=False, **kw)
    n = p["input_ids"].shape[1]
    assert torch.equal(full["input_ids"][0, :n], p["input_ids"][0]), "prompt không phải tiền tố của chuỗi đầy đủ"
    full["labels"] = full["input_ids"].clone()
    full["labels"][:, :n] = -100  # chỉ học phần trả lời
    return full


def collate(encs, pad_id):
    import torch
    import torch.nn.functional as F
    L = max(e["input_ids"].shape[1] for e in encs)
    out = {}
    for k in encs[0].keys():
        if k in ("pixel_values", "image_grid_thw"):
            out[k] = torch.cat([e[k] for e in encs])
            continue
        fill = -100 if k == "labels" else pad_id if k == "input_ids" else 0
        out[k] = torch.cat([F.pad(e[k], (0, L - e[k].shape[1]), value=fill) for e in encs])
    return out


# ---------- Bước 5: LoRA / đóng băng ----------

def apply_setup(model, targets, train_projector, r, alpha):
    import torch
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()
    rx = LLM_RX if targets == "llm" else f"{LLM_RX}|{VIT_RX}"
    matched = [n for n, _ in model.named_modules() if re.fullmatch(rx, n)]
    expected = 252 if targets == "llm" else 348
    assert len(matched) == expected, (len(matched), expected)
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, target_modules=rx))
    if train_projector:
        for n, p in model.named_parameters():
            if any(k in n for k in PROJECTOR):
                p.data = p.data.to(torch.float32)  # GradScaler/AdamW cần master weight fp32
                p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_proj = sum(p.numel() for n, p in model.named_parameters() if any(k in n for k in PROJECTOR))
    print(f"LoRA modules={len(matched)}  trainable={n_train:,}  projector={n_proj:,}  "
          f"total={sum(p.numel() for p in model.parameters()):,}", flush=True)
    return model, n_train


# ---------- Bước 6: vòng huấn luyện ----------

def save(model, out):
    import torch
    model.save_pretrained(out)
    proj = {n: p.detach().cpu() for n, p in model.named_parameters()
            if p.requires_grad and any(k in n for k in PROJECTOR)}
    if proj:
        torch.save(proj, Path(out) / "projector.pt")


def train(a):
    import torch
    from transformers import get_cosine_schedule_with_warmup
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out_dir) / a.name
    model, processor = load_model()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model, n_train = apply_setup(model, a.targets, a.projector, a.r, a.alpha or 2 * a.r)

    rows = read_jsonl(SPLITS / "train.jsonl")
    if a.fraction < 1:
        rows = random.Random(a.seed).sample(rows, round(len(rows) * a.fraction))
    print(f"train images={len(rows)}", flush=True)
    rng, queue = random.Random(a.seed), []

    def take(n):
        nonlocal queue
        if len(queue) < n:
            queue += rng.sample(rows, len(rows))  # epoch mới, xáo lại
        b, queue = queue[:n], queue[n:]
        return b

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * a.steps), a.steps)
    scaler = torch.amp.GradScaler("cuda")
    pad_id = processor.tokenizer.pad_token_id
    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for step in range(1, a.steps + 1):
        total = 0.0
        for _ in range(a.accum):
            batch = collate([encode_example(processor, r, a.coord) for r in take(a.bs)], pad_id)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch, use_cache=False).loss / a.accum
            if not torch.isfinite(loss):
                sys.exit(f"loss không hữu hạn ở bước {step}: {loss.item()} (tràn số fp16?)")
            scaler.scale(loss).backward()
            total += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step % 10 == 0 or step == 1:
            print(f"step {step}/{a.steps} loss={total:.4f} lr={sched.get_last_lr()[0]:.2e} "
                  f"vram={torch.cuda.max_memory_allocated() / 2**30:.1f}GB "
                  f"{(time.time() - t0) / 60:.1f}min", flush=True)
        if step % 100 == 0 or step == a.steps:
            save(model, out)
    run = {"trainable_params": n_train,
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "train_minutes": round((time.time() - t0) / 60, 1),
           "args": vars(a)}
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print("saved", out, run, flush=True)


# ---------- CLI ----------

def cmd_predict(a):
    rows = read_jsonl(SPLITS / f"{a.split}.jsonl")[:a.limit]
    model, processor = load_model(a.adapter)
    pred = ROOT / "work" / f"{a.name}{'' if a.split == 'test' else '_' + a.split}.jsonl"
    write_jsonl(pred, predict(model, processor, rows, a.coord))
    if a.limit:
        return
    ev = ROOT / "report" / "evidence" / f"{a.evidence or a.name}.json"
    cmd = [sys.executable, str(ROOT / "scripts" / "score.py"), "--pred", str(pred), "--split", a.split,
           "--coord", a.coord, "--name", (a.evidence or a.name).upper(), "--out", str(ev)]
    run = Path(a.adapter or "-") / "run.json"
    if run.exists():
        meta = json.loads(run.read_text(encoding="utf-8"))
        cmd += [f"--meta={k}={meta[k]}" for k in ("trainable_params", "peak_vram_gb", "train_minutes")]
    subprocess.run(cmd, check=True)


def cmd_check(a):
    import torch
    from transformers import AutoProcessor
    e = next(r for r in read_jsonl(SPLITS / "train.jsonl") if r["objects"])
    ans = json.loads(objects_to_answer(e))
    back = sorted([b["bbox_2d"][0] / 1000 * e["width"], b["bbox_2d"][1] / 1000 * e["height"]] for b in ans)
    orig = sorted([(o["yolo"][0] - o["yolo"][2] / 2) * e["width"], (o["yolo"][1] - o["yolo"][3] / 2) * e["height"]]
                  for o in e["objects"])
    assert all(abs(x - y) < 1 for b, o in zip(back, orig) for x, y in zip(b, o)), (back, orig)
    print("round-trip OK:", objects_to_answer(e))
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    enc = encode_example(processor, e)
    img_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    n_img = int((enc["input_ids"] == img_id).sum())
    print("keys:", list(enc.keys()), "image tokens:", n_img)
    assert n_img == 220, n_img
    ans_ids = enc["input_ids"][enc["labels"] != -100]
    print("label tokens decode:", repr(processor.tokenizer.decode(ans_ids)))
    assert torch.equal(enc["input_ids"], collate([enc], processor.tokenizer.pad_token_id)["input_ids"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p = sub.add_parser("predict")
    p.add_argument("--name", required=True)
    p.add_argument("--adapter")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--limit", type=int, help="gỡ lỗi trên N ảnh đầu, không chấm")
    p.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"])
    p.add_argument("--evidence", help="tên file evidence nếu khác --name (vd. e4_data_size_0.1)")
    t = sub.add_parser("train")
    t.add_argument("--name", required=True)
    t.add_argument("--targets", default="llm", choices=["llm", "llm+vit"])
    t.add_argument("--projector", action="store_true")
    t.add_argument("--r", type=int, default=16)
    t.add_argument("--alpha", type=int)
    t.add_argument("--fraction", type=float, default=1.0)
    t.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"])
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--bs", type=int, default=2)
    t.add_argument("--accum", type=int, default=4)
    t.add_argument("--steps", type=int, default=300)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--out-dir", default=str(ROOT / "work" / "adapters"))
    a = ap.parse_args()
    {"check": cmd_check, "predict": cmd_predict, "train": train}[a.cmd](a)


if __name__ == "__main__":
    main()
