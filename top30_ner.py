import sys
import os
import json

from knowledge_store import KnowledgeStore

# Cấu hình hiển thị tiếng Việt trên Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 70)
    print("BƯỚC 2: TÌM TOP 30 THỰC THỂ TRI THỨC BẰNG THUẬT TOÁN BAYES (KNOWLEDGE STORE)")
    print("=" * 70)

    # 1. Nạp cơ sở tri thức (tính ma trận đồng xuất hiện)
    print("[*] Đang nạp knowledge.jsonl và xây dựng ma trận đồng xuất hiện...")
    ks = KnowledgeStore('./data/knowledge.jsonl')
    print(f"[+] Đã nạp thành công! Tổng số thực thể duy nhất: {len(ks.entity_counter):,}")

    import random

    # 2. Chọn ngẫu nhiên 1 bài báo mục tiêu từ papers.jsonl
    print("[*] Đang chọn ngẫu nhiên 1 bài báo nghiên cứu từ kho dữ liệu...")
    sample_papers = []
    with open('./data/papers.jsonl', 'r', encoding='utf-8') as f:
        for _ in range(200):
            line = f.readline()
            if not line:
                break
            p = json.loads(line)
            pid = p.get('corpusid')
            if pid in ks.paper2entities and len(ks.paper2entities[pid]) > 0:
                # Đảm bảo bài báo có thực thể đã từng đồng xuất hiện với các thực thể khác
                if any(len(ks.entity_cooccurrence.get(e, {})) > 0 for e in ks.paper2entities[pid]):
                    sample_papers.append(p)

    target = random.choice(sample_papers)
    target_id = target['corpusid']
    target_title = target['title']
    target_entities = list(ks.paper2entities[target_id].keys())

    print(f"\n[+] Bài báo mục tiêu được chọn ngẫu nhiên:")
    print(f"    - Corpus ID: {target_id}")
    print(f"    - Tiêu đề  : {target_title}")
    print(f"    - Thực thể cốt lõi ban đầu: {target_entities}")

    # Lấy thêm 15 bài báo tham chiếu thực tế từ references.jsonl
    ref_ids = []
    with open('./data/references.jsonl', 'r', encoding='utf-8') as f:
        for _ in range(100):
            line = f.readline()
            if not line:
                break
            r = json.loads(line)
            if r.get('corpusid') in ks.paper2entities:
                ref_ids.append(r['corpusid'])
                if len(ref_ids) >= 15:
                    break

    selected_cluster = [target_id] + ref_ids
    print(f"\n[*] Đang chạy thuật toán Bayes trên cụm {len(selected_cluster)} bài báo (Target + References)...")
    top30_entities = ks.get_relevant_entities(paper_ids=selected_cluster, top_k=30)

    # 4. In kết quả
    print("\n" + "=" * 70)
    print("TOP 30 THỰC THỂ TRI THỨC GỢI CẢM HỨNG CHO AI (ENTITIES INSPIRATION):")
    print("=" * 70)
    for i, entity in enumerate(top30_entities, 1):
        print(f"{i:2d}. {entity}")

    print("=" * 70)
    print("=> 30 thực thể này cùng với Target Paper và Top 10 References")
    print("   sẽ được đóng gói thành Prompt hoàn chỉnh đưa vào Multi-Agent LLM!")
    print("=" * 70)

if __name__ == '__main__':
    main()
