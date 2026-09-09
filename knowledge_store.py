import math
import json
from collections import defaultdict, Counter
import torch


class KnowledgeStore:
    """
    Cơ sở tri thức và thuật toán Bayesian Scoring độc lập hoàn toàn,
    không phụ thuộc vào bất kỳ thư viện ngoài nào của ResearchAgent.
    """
    def __init__(self, file_path: str):
        self.knowledge_base = self._load_jsonl(file_path)
        self.paper2entities = {item['corpusid']: item['knowledge'] for item in self.knowledge_base}
        self.entity_counter, self.entity_cooccurrence = self._build_entity_statistics()

    @staticmethod
    def _load_jsonl(file_path: str) -> list:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _build_entity_statistics(self):
        entity_counter = Counter()
        entity_cooccurrence = defaultdict(Counter)

        for instance in self.knowledge_base:
            entities = instance['knowledge']
            entity_counter.update(entities)

            for entity_name in entities.keys():
                entity_cooccurrence[entity_name].update(
                    {k: v for k, v in entities.items() if k != entity_name}
                )

        return entity_counter, entity_cooccurrence

    def get_entity_log_likelihood(self, entity: str, paper_entities: list) -> float:
        conditional_log_probabilities = [
            math.log2(
                (self.entity_cooccurrence[entity][pe] + 1e-16) /
                (sum(self.entity_cooccurrence[entity].values()) + 1e-16)
            ) for pe in paper_entities
        ]
        return sum(conditional_log_probabilities)

    def get_entity_probability(self, entity: str) -> float:
        return self.entity_counter[entity] / sum(self.entity_counter.values())

    def get_relevant_entities(self, paper_ids: list, top_k: int = 30) -> list:
        paper_entities = sum(
            [
                Counter(self.paper2entities[pid]) for pid in paper_ids 
                if pid in self.paper2entities
            ], 
            start=Counter()
        )
        paper_entities = list(paper_entities.elements())

        candidate_entities = sum(
            [self.entity_cooccurrence[entity] for entity in paper_entities],
            start=Counter()
        )
        candidate_entities = [entity for entity, count in candidate_entities.items() if count >= 3]

        if not candidate_entities:
            return []

        candidate_entities_probs = [
            (
                self.get_entity_log_likelihood(entity, paper_entities) + 
                math.log2(self.get_entity_probability(entity) + 1e-16)
            )
            for entity in candidate_entities
        ]

        _, indices = torch.topk(
            torch.tensor(candidate_entities_probs), 
            k=min(top_k, len(candidate_entities)), 
            axis=-1
        )
        return [candidate_entities[idx] for idx in indices]
