# Demo Pipeline Xử Lý Dữ Liệu Bài Báo Khoa Học

Thư mục này chứa mã nguồn và dữ liệu dùng để demo quy trình xử lý dữ liệu: từ văn bản thô, chuyển thành vector ngữ nghĩa, lọc bài báo liên quan và khai phá thực thể tri thức trước khi đưa vào LLM.

## 1. Cấu trúc thư mục

- data/
  - papers.jsonl: Danh sách các bài báo nghiên cứu mục tiêu.
  - references.jsonl: Tập hợp các bài báo tham chiếu dùng để so khớp.
  - knowledge.jsonl: Cơ sở dữ liệu các thực thể tri thức đã được trích xuất.
- vectorize.py: Chuyển đổi tiêu đề và tóm tắt bài báo thành vector 768 chiều bằng mô hình SPECTER, đo độ tương đồng Cosine giữa 2 bài ngẫu nhiên.
- top10_ref.py: Chọn ngẫu nhiên 1 bài báo mục tiêu và lọc ra Top 10 bài tham chiếu có độ tương đồng cao nhất bằng Cosine Similarity và torch.topk.
- top30_ner.py: Xây dựng ma trận đồng xuất hiện và áp dụng thuật toán Bayes (Bayesian Scoring) để tìm Top 30 thực thể tri thức quan trọng nhất cho cụm bài báo.
- export_llm.py: Tổng hợp kết quả (Target Paper + Top 10 References + Top 30 Entities) và xuất ra file llm_input.json kèm prompt hoàn chỉnh để nạp vào LLM.
- knowledge_store.py: Module xử lý ma trận đồng xuất hiện và thuật toán Bayes độc lập.
- llm_input.json: File JSON mẫu kết quả đầu ra.

## 2. Cài đặt môi trường

Yêu cầu Python 3.9 trở lên và cài các thư viện sau:

```bash
pip install torch transformers tqdm
```

Lưu ý: Lần đầu tiên chạy vectorize.py hoặc top10_ref.py, máy sẽ tải mô hình allenai/specter (~440MB) từ Hugging Face về lưu vào bộ nhớ đệm. Các lần sau sẽ chạy trực tiếp từ ổ cứng nên rất nhanh.

## 3. Thứ tự chạy demo

Bước 1: Demo mã hóa bài báo thành vector ngữ nghĩa và so sánh tương đồng
```bash
python vectorize.py
```

Bước 2: Demo thuật toán truy xuất Top 10 bài tham chiếu liên quan nhất
```bash
python top10_ref.py
```

Bước 3: Demo khai phá đồ thị tri thức và lọc Top 30 thực thể bằng Bayes
```bash
python top30_ner.py
```

Bước 4: Demo đóng gói toàn bộ dữ liệu thành file JSON gửi lên LLM
```bash
python export_llm.py
```

Sau khi chạy xong Bước 4, file `llm_input.json` sẽ được tạo ra tại thư mục này. Có thể dùng nội dung trong trường `formatted_prompt` để đưa trực tiếp vào ChatGPT, Claude hoặc gọi API LLM để sinh ý tưởng nghiên cứu.
