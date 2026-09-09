import json
import torch
from vectorize import PaperVectorizer

# 1. Khởi tạo mô hình
vectorizer = PaperVectorizer()

import random

# 2. Chọn ngẫu nhiên 1 Target Paper từ kho dữ liệu
pool_papers = []
with open('./data/papers.jsonl', 'r', encoding='utf-8') as f:
    for _ in range(100):
        line = f.readline()
        if not line:
            break
        pool_papers.append(json.loads(line))

target_paper = random.choice(pool_papers)
print(f"[*] Target Paper (Ngẫu nhiên): {target_paper['title']}")
target_vec = vectorizer.encode(target_paper['title']) # (1, 768)

# 3. Lấy ngẫu nhiên 30 bài tham chiếu ứng viên từ references.jsonl
ref_pool = []
with open('./data/references.jsonl', 'r', encoding='utf-8') as f:
    for _ in range(150):
        line = f.readline()
        if not line:
            break
        ref_pool.append(json.loads(line))

candidates = random.sample(ref_pool, min(30, len(ref_pool)))
print(f"[*] Đang so sánh Cosine với {len(candidates)} bài báo tham chiếu ứng viên ngẫu nhiên...")

# 4. Biến đổi tất cả các bài tham chiếu thành vector
ref_vectors = []
for paper in candidates:
    v = vectorizer.encode(paper.get('title', ''), paper.get('abstract', ''))
    ref_vectors.append(v)

# Ghép thành một ma trận Tensor (30 x 768)
ref_matrix = torch.stack(ref_vectors)

# 5. THUẬT TOÁN TÍNH ĐỘ TƯƠNG ĐỒNG TOÀN BỘ VÀ LẤY TOP 10
# Tính Cosine Similarity giữa 1 vector với ma trận 30 vector
cos = torch.nn.CosineSimilarity(dim=1)
similarities = cos(target_vec.unsqueeze(0), ref_matrix)

# Dùng torch.topk để lấy 10 bài có điểm cao nhất
top_k_values, top_k_indices = torch.topk(similarities, k=10)

print("\n" + "=" * 70)
print("TOP 10 BÀI BÁO THAM CHIẾU LIÊN QUAN NHẤT ĐƯỢC CHỌN LỌC:")
print("=" * 70)
for rank, (idx, score) in enumerate(zip(top_k_indices, top_k_values), 1):
    best_paper = candidates[idx]
    print(f"#{rank} [Điểm tương đồng: {score:.4f}]")
    print(f"   Tiêu đề: {best_paper.get('title')}")
    print(f"   CorpusId: {best_paper.get('corpusid')}\n")
