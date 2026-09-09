import json
import sys
import os

from knowledge_store import KnowledgeStore

# Cấu hình UTF-8 trên Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def build_llm_prompt(paper: dict, references: list, entities: list) -> str:
    """Tạo bản text Prompt hoàn chỉnh theo format của ResearchAgent."""
    prompt = (
        "You are an AI research assistant. Your task is to brainstorm and identify a promising, novel, "
        "and significant research problem based on the following scientific literature and knowledge entities:\n\n"
        f"=== TARGET PAPER ===\n"
        f"Title: {paper.get('title')}\n"
        f"Abstract: {paper.get('abstract', '')}\n\n"
        f"=== TOP 10 RELEVANT REFERENCES ===\n"
    )
    for i, ref in enumerate(references, 1):
        prompt += f"[{i}] {ref.get('title')}\n"
        if ref.get('abstract'):
            prompt += f"    Abstract: {ref.get('abstract')[:200]}...\n"

    prompt += f"\n=== TOP 30 KNOWLEDGE ENTITIES (INSPIRATION) ===\n"
    prompt += ", ".join(entities) + "\n\n"

    prompt += (
        "=== YOUR TASK ===\n"
        "1. Identify key limitations or gaps in the target paper and its references.\n"
        "2. Leverage cross-domain inspiration from the knowledge entities.\n"
        "3. Propose a clear, high-impact research problem and a novel methodology to solve it."
    )
    return prompt


def main():
    print("=" * 70)
    print("XUẤT DỮ LIỆU ĐẦU VÀO CHO LLM (EXPORT TO JSON)")
    print("=" * 70)

    import random

    # 1. Chọn ngẫu nhiên Target Paper
    print("[1/4] Đang chọn ngẫu nhiên Target Paper từ kho dữ liệu...")
    sample_papers = []
    with open('./data/papers.jsonl', 'r', encoding='utf-8') as f:
        for _ in range(150):
            line = f.readline()
            if not line:
                break
            sample_papers.append(json.loads(line))
    
    target_paper = random.choice(sample_papers)
    paper_info = {
        "corpus_id": target_paper.get("corpusid"),
        "title": target_paper.get("title"),
        "venue": target_paper.get("venue", "N/A"),
        "year": target_paper.get("year", 2023)
    }
    print(f"  -> Target Paper: {paper_info['title']} ({paper_info['year']})")

    # 2. Lấy ngẫu nhiên các bài tham chiếu thực tế từ references.jsonl
    print("\n[2/4] Đang trích xuất Top References thực tế từ references.jsonl...")
    ref_candidates = []
    with open('./data/references.jsonl', 'r', encoding='utf-8') as f:
        for _ in range(200):
            line = f.readline()
            if not line:
                break
            r = json.loads(line)
            if r.get('title') and r.get('abstract'):
                ref_candidates.append({
                    "corpus_id": r.get('corpusid'),
                    "title": r.get('title'),
                    "abstract": r.get('abstract')
                })

    references = random.sample(ref_candidates, min(10, len(ref_candidates)))
    print(f"  -> Đã thu thập: {len(references)} bài báo tham chiếu thực tế.")

    # 3. Lấy Top 30 Entities bằng thuật toán Bayes
    print("\n[3/4] Đang nạp KnowledgeStore và tính toán Top 30 Entities qua Bayes...")
    ks = KnowledgeStore('./data/knowledge.jsonl')
    paper_ids_for_entities = [paper_info['corpus_id']] + [r['corpus_id'] for r in references]
    # Nếu target paper chưa có trong knowledge store, lấy 1 vài paper id từ references để khởi tạo
    valid_ids = [pid for pid in paper_ids_for_entities if pid in ks.paper2entities]
    if not valid_ids:
        valid_ids = random.sample(list(ks.paper2entities.keys()), 5)
    top30_entities = ks.get_relevant_entities(paper_ids=valid_ids, top_k=30)
    print(f"  -> Đã chọn lọc: {len(top30_entities)} thực thể tri thức mở rộng.")

    # 4. Đóng gói thành cấu trúc JSON hoàn chỉnh
    print("\n[4/4] Đang đóng gói dữ liệu và xuất file JSON...")
    llm_payload = {
        "metadata": {
            "version": "1.0",
            "pipeline": "ResearchAgent Data Pipeline",
            "target_id": paper_info["corpus_id"]
        },
        "context": {
            "paper": paper_info,
            "references": references,
            "entities": top30_entities
        },
        "formatted_prompt": build_llm_prompt(paper_info, references, top30_entities)
    }

    output_path = "./llm_input.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(llm_payload, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print(f"[THÀNH CÔNG] File JSON đã được tạo tại: {output_path}")
    print(f"- Kích thước file: {os.path.getsize(output_path):,} bytes")
    print("- Cấu trúc JSON gồm 4 phần chính:")
    print("   1. context.paper       : Tiêu đề & Tóm tắt bài báo mục tiêu")
    print("   2. context.references  : Danh sách Top 10 bài báo tham chiếu liên quan")
    print("   3. context.entities    : Top 30 thực thể tri thức kích thích sáng tạo")
    print("   4. formatted_prompt    : Đoạn Prompt hoàn chỉnh sẵn sàng nạp thẳng vào LLM!")
    print("=" * 70)


if __name__ == '__main__':
    main()
