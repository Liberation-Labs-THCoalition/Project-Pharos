"""Pharos Router — confidence-gated pack selection for knowledge injection.

Selects the right knowledge pack for a given query using embedding similarity.
Production encoding policy from the encoding comparison results:
  - TRIPLES (default): consistent +0.007, never harmful
  - TRIPLES_SOURCE (high confidence match): +0.006, richer context
  - HYBRID (very high confidence, theory-matched): 0.964 ceiling but -0.106 when mismatched
  - WALK_ONLY: never — hurts ethical reasoning

The router prevents the catastrophic mismatch case by gating hybrid encoding
behind a confidence threshold calibrated from the encoding comparison data.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_HAS_ST = False
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    pass


@dataclass
class PackMatch:
    """A matched knowledge pack with routing metadata."""
    pack_name: str
    similarity: float
    encoding: str
    triples_path: Path
    walk_path: Path
    stats: dict = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """The router's output: which packs to inject and how."""
    query: str
    matches: list[PackMatch]
    encoding_policy: str
    confidence: float
    explanation: str


SIRA_VOCAB = {
    "ethics": "moral virtue duty obligation deontology consequentialism",
    "aristotle": "eudaimonia virtue phronesis telos happiness flourishing",
    "kant": "categorical imperative duty deontology maxim universalizability",
    "utilitarian": "consequentialism happiness welfare bentham mill",
    "rights": "autonomy dignity consent freedom justice",
    "buddhism": "dharma karma suffering compassion mindfulness",
    "confucian": "ren li junzi harmony filial",
    "african": "ubuntu communitarian solidarity dignity",
    "feminist": "care justice oppression intersectionality autonomy",
    "virtue": "character excellence habituation practical wisdom",
    "justice": "fairness equality distribution rights social contract",
    "autonomy": "consent self-determination dignity agency freedom",
}


def _enrich_query(query: str) -> str:
    q_lower = query.lower()
    enriched = query
    for trigger, expansions in SIRA_VOCAB.items():
        if trigger in q_lower:
            enriched = f"{enriched} {expansions}"
    return enriched


class PackLibrary:
    """Index of available knowledge packs with precomputed embeddings."""

    def __init__(self, packs_dir: str, model_name: str = "all-MiniLM-L6-v2"):
        self.packs_dir = Path(packs_dir)
        self.packs: list[dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.model = None

        if _HAS_ST:
            self.model = SentenceTransformer(model_name)

        self._index_packs()

    def _index_packs(self):
        if not self.packs_dir.exists():
            logger.warning(f"Packs directory not found: {self.packs_dir}")
            return

        descriptors = []
        for pack_dir in sorted(self.packs_dir.iterdir()):
            if not pack_dir.is_dir():
                continue

            stats_path = pack_dir / "stats.json"
            triples_path = pack_dir / "triples.json"
            walk_path = pack_dir / "walk_encoding.txt"

            if not triples_path.exists():
                continue

            stats = {}
            if stats_path.exists():
                stats = json.loads(stats_path.read_text())

            triples = json.loads(triples_path.read_text())
            descriptor = self._build_descriptor(pack_dir.name, triples, stats)

            self.packs.append({
                "name": pack_dir.name,
                "triples_path": triples_path,
                "walk_path": walk_path,
                "stats": stats,
                "descriptor": descriptor,
                "triple_count": len(triples),
            })
            descriptors.append(descriptor)

        if descriptors and self.model:
            self.embeddings = self.model.encode(descriptors, convert_to_numpy=True)
            logger.info(f"Indexed {len(self.packs)} packs with embeddings")
        else:
            logger.info(f"Indexed {len(self.packs)} packs (no embeddings)")

    def _build_descriptor(self, name: str, triples: list, stats: dict) -> str:
        readable = name.replace("-", " ").replace("_", " ")
        subjects = set()
        predicates = set()
        objects = set()
        for t in triples[:30]:
            subjects.add(t.get("s", ""))
            predicates.add(t.get("p", ""))
            objects.add(t.get("o", ""))

        key_entities = " ".join(list(subjects)[:10] + list(objects)[:10])
        key_relations = " ".join(list(predicates)[:8])
        return f"{readable}. Entities: {key_entities}. Relations: {key_relations}."


class PharosRouter:
    """Confidence-gated router for knowledge pack selection.

    Encoding policy:
      similarity >= hybrid_threshold  → HYBRID (triples + walk + source)
      similarity >= source_threshold  → TRIPLES_SOURCE (triples + source excerpts)
      similarity >= match_threshold   → TRIPLES (default, safe)
      below match_threshold           → NO INJECTION (pack doesn't match)
    """

    def __init__(
        self,
        library: PackLibrary,
        match_threshold: float = 0.35,
        source_threshold: float = 0.55,
        hybrid_threshold: float = 0.75,
        max_packs: int = 3,
    ):
        self.library = library
        self.match_threshold = match_threshold
        self.source_threshold = source_threshold
        self.hybrid_threshold = hybrid_threshold
        self.max_packs = max_packs

    def route(self, query: str) -> RoutingDecision:
        if not self.library.model or self.library.embeddings is None:
            return RoutingDecision(
                query=query, matches=[], encoding_policy="none",
                confidence=0.0, explanation="No embedding model available",
            )

        enriched = _enrich_query(query)
        query_emb = self.library.model.encode(enriched, convert_to_numpy=True)

        similarities = np.dot(self.library.embeddings, query_emb) / (
            np.linalg.norm(self.library.embeddings, axis=1) * np.linalg.norm(query_emb)
        )

        ranked = sorted(
            zip(range(len(self.library.packs)), similarities),
            key=lambda x: x[1],
            reverse=True,
        )

        matches = []
        for idx, sim in ranked:
            if sim < self.match_threshold:
                break
            if len(matches) >= self.max_packs:
                break

            pack = self.library.packs[idx]

            if sim >= self.hybrid_threshold:
                encoding = "hybrid"
            elif sim >= self.source_threshold:
                encoding = "triples_source"
            else:
                encoding = "triples"

            matches.append(PackMatch(
                pack_name=pack["name"],
                similarity=float(sim),
                encoding=encoding,
                triples_path=pack["triples_path"],
                walk_path=pack["walk_path"],
                stats=pack["stats"],
            ))

        if not matches:
            return RoutingDecision(
                query=query, matches=[], encoding_policy="none",
                confidence=0.0,
                explanation=f"No pack above match threshold ({self.match_threshold})",
            )

        top_sim = matches[0].similarity
        policy = matches[0].encoding
        explanation = (
            f"Top match: {matches[0].pack_name} ({top_sim:.3f}). "
            f"Encoding: {policy}. "
            f"{len(matches)} pack(s) above threshold."
        )

        return RoutingDecision(
            query=query,
            matches=matches,
            encoding_policy=policy,
            confidence=top_sim,
            explanation=explanation,
        )

    def route_compact(self, query: str) -> str:
        """DNO-style compact routing output for agent consumption."""
        decision = self.route(query)
        if not decision.matches:
            return f"ROUTE|{query[:60]}|NO_MATCH\n"

        lines = [f"ROUTE|{query[:60]}|n={len(decision.matches)}|policy={decision.encoding_policy}"]
        for m in decision.matches:
            lines.append(f"  {m.encoding}|{m.similarity:.3f}|{m.pack_name}|t={m.stats.get('triples', '?')}")
        return "\n".join(lines) + "\n"


def load_router(
    packs_dir: str = None,
    model_name: str = "all-MiniLM-L6-v2",
    **kwargs,
) -> PharosRouter:
    if packs_dir is None:
        packs_dir = str(Path(__file__).parent / "packs")
    library = PackLibrary(packs_dir, model_name)
    return PharosRouter(library, **kwargs)


if __name__ == "__main__":
    import sys
    router = load_router()
    queries = sys.argv[1:] or [
        "Is it ethical to sacrifice one person to save five?",
        "What did Aristotle say about the good life?",
        "How should autonomous AI systems make moral decisions?",
        "What is the Buddhist perspective on suffering?",
    ]
    for q in queries:
        print(router.route_compact(q))
