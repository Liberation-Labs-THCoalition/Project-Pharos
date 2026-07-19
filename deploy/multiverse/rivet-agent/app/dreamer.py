"""Dreamer — H-MEM Temporal Consolidation Service.

Runs as a sidecar process. Periodically:
1. Triggers Mnemosyne consolidation (significance decay)
2. Extracts entities from high-significance interactions
3. Indexes extracted knowledge into HippoRAG graph

Reference: Project-Mnemosyne H-MEM module
"""

import os
import time
import logging

import httpx

logging.basicConfig(level=logging.INFO, format="[dreamer] %(message)s")
log = logging.getLogger("dreamer")

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://rivet:8000")
HIPPORAG_URL = os.environ.get("HIPPORAG_URL", "http://hipporag:11235")
CYCLE_HOURS = int(os.environ.get("CONSOLIDATION_INTERVAL_HOURS", "6"))
CYCLE_SECONDS = CYCLE_HOURS * 3600


def consolidate():
    """Trigger Mnemosyne consolidation via gateway."""
    try:
        r = httpx.post(f"{GATEWAY_URL}/api/consolidate", timeout=30)
        result = r.json()
        log.info(f"Consolidation: {result}")
        return result
    except Exception as e:
        log.error(f"Consolidation failed: {e}")
        return {}


def enrich_graph():
    """Extract entities from recent significant memories, index into HippoRAG."""
    try:
        r = httpx.post(
            f"{GATEWAY_URL}/api/search/memory",
            json={"query": "code deploy build error fix", "limit": 20},
            timeout=30,
        )
        memories = r.json().get("results", [])

        triples = []
        for m in memories:
            content = m.get("content", "")
            significance = m.get("significance", 0)
            if significance >= 0.6:
                words = [w.strip(".,!?()[]") for w in content.split() if len(w) > 4]
                unique_words = list(dict.fromkeys(words))
                if len(unique_words) >= 2:
                    triples.append({
                        "subject": unique_words[0],
                        "predicate": "mentioned_with",
                        "object": unique_words[1],
                    })

        if triples:
            r = httpx.post(
                f"{HIPPORAG_URL}/index",
                json={"triples": triples, "source_id": "dreamer-enrichment"},
                timeout=30,
            )
            log.info(f"Graph enrichment: {r.json()}")
    except Exception as e:
        log.error(f"Enrichment failed: {e}")


if __name__ == "__main__":
    log.info(f"Dreamer starting — H-MEM consolidation every {CYCLE_HOURS}h")
    time.sleep(60)

    while True:
        time.sleep(CYCLE_SECONDS)
        consolidate()
        enrich_graph()
        log.info("Cycle complete")
