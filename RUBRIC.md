# Thang điểm Lab Day 10 (100 điểm + 10 bonus)

> Bài cá nhân, nộp qua **repo GitHub công khai**. Xem [`README.md`](README.md) và [`GUIDE.md`](GUIDE.md).

## Cách chấm

- Giảng viên clone repo (commit mới nhất **trước hạn nộp**), **ghi đè `scripts/` và `tests/` bằng bản gốc của đề**, chạy `scripts/check_submission.py`, và đọc `report/evidence/*.json` do `score.py` sinh ra.
- Giảng viên đối chiếu báo cáo **giữa các repo** và có thể gọi **vấn đáp 5 phút** để xác nhận bạn hiểu điều bạn viết.

| Phần | Điểm | Nội dung |
|---|---|---|
| A. Kết quả chạy (tự động) | 40 | Xem bảng A bên dưới |
| B. Thí nghiệm và bằng chứng | 20 | Xem bảng B bên dưới |
| C. Báo cáo | 30 | 10 câu × 3 điểm. Mỗi câu phải **đúng bản chất** và **dùng số của chính bạn** (không chép tài liệu). Câu 3, 4, 6, 8 chấm kỹ nhất |
| D. Git và trình bày | 10 | ≥ 5 commit theo bước, repo sạch (không dữ liệu/khoá/trọng số): 5 · `check_submission.py` không còn FAIL: 5 |

## A. Kết quả chạy (40 điểm, tự động)

| Tiêu chí | Điểm | Đạt khi |
|---|---|---|
| `e0.json`–`e3.json` hợp lệ | 12 | Đủ 4 file, `split` là `test`, `missing_predictions` = 0, cùng `dataset_version` và khớp `data/splits/meta.json` |
| `meta` của E1–E3 | 6 | Có `trainable_params`, `peak_vram_gb`, `train_minutes` là số dương |
| Output hợp lệ của E1 | 4 | `parse_valid_rate` của E1 ≥ 0,95 |
| E1 thật sự học được | 6 | F1 của E1 hơn F1 của E0 ít nhất 0,10 |
| E4 | 6 | Có ít nhất một file `e4*.json` hợp lệ |
| Bằng chứng nhất quán | 6 | `check_submission.py` không báo FAIL ở các mục bằng chứng |

## B. Thí nghiệm và bằng chứng (20 điểm)

| Tiêu chí | Điểm | Đạt đủ điểm khi |
|---|---|---|
| Bảng tổng hợp E0–E3 | 6 | Đủ F1, CI95, FP/ảnh, VRAM, thời gian, F1 từng lớp, khớp các file bằng chứng |
| Ảnh minh hoạ | 4 | ≥ 6 ảnh dự đoán đúng/sai có chú thích (box dự đoán và box đúng) |
| E4 sạch | 6 | Chỉ đổi **một** biến, chạy cả hai giá trị, kết luận có nhìn CI95 |
| Nhật ký tài nguyên | 4 | VRAM và thời gian trong báo cáo khớp `meta` |

## Bonus E5 (+10, tổng không vượt quá 110)

Hàm reward đúng và có test: 4 · chạy GRPO có log: 3 · phân tích reward hacking và so với SFT bằng số kèm CI95: 3.

## Trừ điểm

- Sửa `scripts/` hoặc `tests/` của đề: **−10**.
- Chép bài (số liệu hoặc văn bản giống nhau giữa các repo): **0 điểm cho cả hai bên**.
- Commit dữ liệu, khoá Kaggle/HF, hoặc trọng số: mỗi mục trừ 2 điểm ở phần D.
