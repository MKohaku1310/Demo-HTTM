import json
import sys
import torch
from transformers import AutoTokenizer, AutoModel

# Cấu hình hiển thị tiếng Việt trên Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

class PaperVectorizer:
    def __init__(self, model_name: str = "allenai/specter"):
        """
        Khởi tạo mô hình Transformer chuyên biệt cho bài báo khoa học.
        'allenai/specter' là mô hình gốc tạo ra vector 768 chiều cho bài báo.
        """
        print(f"[*] Đang tải mô hình Transformer: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Chuyển sang chế độ inference (dự đoán)
        print("[+] Mô hình đã sẵn sàng!\n")

    def encode(self, title: str, abstract: str = "") -> torch.Tensor:
        """
        Biến đổi (Title + Abstract) thành 1 vector 768 chiều
        """
        # Quy tắc của SPECTER: Ghép Tiêu đề + token phân cách [SEP] + Tóm tắt
        sep_token = self.tokenizer.sep_token or "[SEP]"
        text = f"{title} {sep_token} {abstract}"

        # 1. Tokenize văn bản thành tensor đầu vào cho Transformer
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # 2. Đưa qua mạng Transformer để trích xuất vector
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Token [CLS] (ở vị trí đầu tiên index 0) đại diện cho ngữ nghĩa toàn bài báo
            paper_vector = outputs.last_hidden_state[:, 0, :]
            
        return paper_vector.squeeze(0)  # Trả về vector 1D gồm 768 số thực

    @staticmethod
    def compute_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Tính độ tương đồng góc Cosine giữa 2 vector (từ -1.0 đến 1.0)
        """
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(vec1, vec2).item()


def main():
    print("=" * 70)
    print("DEMO: BIẾN ĐỔI BÀI BÁO THÀNH VECTOR NGỮ NGHĨA (SPECTER TRANSFORMER)")
    print("=" * 70)

    # 1. Khởi tạo bộ trích xuất vector
    vectorizer = PaperVectorizer(model_name="allenai/specter")

    # 2. Chọn ngẫu nhiên 2 bài báo thực tế từ kho dữ liệu references.jsonl
    print("[*] Đang chọn ngẫu nhiên 2 bài báo từ kho dữ liệu (43,000+ bài)...")
    import random
    candidates = []
    with open('./data/references.jsonl', 'r', encoding='utf-8') as f:
        # Đọc 200 dòng đầu và chọn ngẫu nhiên 2 bài để tốc độ đọc tức thì
        for _ in range(200):
            line = f.readline()
            if not line:
                break
            data = json.loads(line)
            if data.get('title') and data.get('abstract'):
                candidates.append(data)

    sampled = random.sample(candidates, 2)
    paper1, paper2 = sampled[0], sampled[1]

    # 3. Biến đổi bài báo thành Vector
    print("\n--- [BÀI BÁO 1 (NGẪU NHIÊN)] ---")
    print(f"Tiêu đề : {paper1['title']}")
    print(f"Tóm tắt : {paper1['abstract'][:180]}...")
    vec1 = vectorizer.encode(paper1['title'], paper1['abstract'])
    print(f"-> Vector thu được: {vec1.shape[0]} chiều (Chuẩn SPECTER 768 chiều)")
    print(f"-> 5 giá trị đầu tiên: {vec1[:5].tolist()}\n")

    print("--- [BÀI BÁO 2 (NGẪU NHIÊN)] ---")
    print(f"Tiêu đề : {paper2['title']}")
    print(f"Tóm tắt : {paper2['abstract'][:180]}...")
    vec2 = vectorizer.encode(paper2['title'], paper2['abstract'])
    print(f"-> Vector thu được: {vec2.shape[0]} chiều (Chuẩn SPECTER 768 chiều)")
    print(f"-> 5 giá trị đầu tiên: {vec2[:5].tolist()}\n")

    # 4. Đo độ tương đồng ngữ nghĩa giữa 2 bài
    sim_score = PaperVectorizer.compute_similarity(vec1, vec2)
    print("=" * 70)
    print(f"ĐỘ TƯƠNG ĐỒNG COSINE GIỮA 2 BÀI: {sim_score:.4f} (Thang đo -1.0 đến 1.0)")
    if sim_score > 0.7:
        print("=> Nhận xét: 2 bài có cùng hướng nghiên cứu chuyên sâu (Rất gần nhau).")
    elif sim_score > 0.5:
        print("=> Nhận xét: 2 bài có điểm giao thoa chủ đề ở mức trung bình.")
    else:
        print("=> Nhận xét: 2 bài thuộc 2 lĩnh vực nghiên cứu khác biệt nhau!")
    print("=" * 70)


if __name__ == '__main__':
    main()
