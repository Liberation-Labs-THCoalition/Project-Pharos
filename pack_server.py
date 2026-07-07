#!/usr/bin/env python3
"""Pharos Pack Server — serve any knowledge pack as a queryable MCP-style API.

Turns static triple files into live, searchable knowledge endpoints.
Supports: exact match, fuzzy search, relationship traversal, bulk query.

Usage:
  # Serve all packs on port 8300
  python pack_server.py --port 8300

  # Serve specific packs
  python pack_server.py --packs statistical-pitfalls,causal-inference

  # Query from code
  curl localhost:8300/query -d '{"pack": "statistical-pitfalls", "query": "p-hacking"}'
  curl localhost:8300/traverse -d '{"pack": "causal-inference", "entity": "confounding", "depth": 2}'
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

PACKS_DIR = Path(os.path.expanduser("~/lab/projects/pharos/packs"))


class KnowledgeIndex:
    """In-memory index over a Pharos pack's triples."""

    def __init__(self, pack_name: str, pack_dir: Path):
        self.name = pack_name
        self.triples = []
        self.description = ""
        self.sources = []

        # Indexes
        self.by_subject = defaultdict(list)
        self.by_predicate = defaultdict(list)
        self.by_object = defaultdict(list)
        self.all_entities = set()
        self.search_index = {}  # lowered text -> triple index

        self._load(pack_dir)

    def _load(self, pack_dir: Path):
        triple_file = pack_dir / "triples.json"
        if not triple_file.exists():
            return

        data = json.load(open(triple_file))
        if isinstance(data, list):
            self.triples = data
        elif isinstance(data, dict):
            self.description = data.get("description", "")
            self.sources = data.get("sources", [])
            self.triples = data.get("triples", [])

        for i, t in enumerate(self.triples):
            subj = t.get("subject", "")
            pred = t.get("predicate", "")
            obj = t.get("object", "")

            self.by_subject[subj.lower()].append(i)
            self.by_predicate[pred.lower()].append(i)
            self.by_object[obj.lower()].append(i)
            self.all_entities.add(subj.lower())
            self.all_entities.add(obj.lower())

            text = f"{subj} {pred} {obj}".lower()
            for word in text.split():
                word = re.sub(r'[^\w]', '', word)
                if word and len(word) > 2:
                    if word not in self.search_index:
                        self.search_index[word] = []
                    self.search_index[word].append(i)

    def query(self, text: str, max_results: int = 20) -> list:
        """Fuzzy search across all triples."""
        text = text.lower().strip()
        words = [re.sub(r'[^\w]', '', w) for w in text.split() if len(w) > 2]

        if not words:
            return self.triples[:max_results]

        scores = defaultdict(float)
        for word in words:
            for idx_word, indices in self.search_index.items():
                if word in idx_word or idx_word in word:
                    weight = 1.0 if word == idx_word else 0.5
                    for idx in indices:
                        scores[idx] += weight

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [self.triples[idx] for idx, _ in ranked[:max_results]]

    def lookup(self, entity: str, role: str = "any") -> list:
        """Exact entity lookup."""
        entity = entity.lower().strip()
        results = []

        if role in ("subject", "any"):
            results.extend(self.by_subject.get(entity, []))
        if role in ("object", "any"):
            results.extend(self.by_object.get(entity, []))
        if role in ("predicate", "any"):
            results.extend(self.by_predicate.get(entity, []))

        return [self.triples[i] for i in sorted(set(results))]

    def traverse(self, entity: str, depth: int = 1, max_results: int = 50) -> dict:
        """Graph traversal from an entity."""
        visited = set()
        frontier = {entity.lower()}
        layers = {}

        for d in range(depth):
            next_frontier = set()
            layer_triples = []

            for ent in frontier:
                if ent in visited:
                    continue
                visited.add(ent)

                for idx in self.by_subject.get(ent, []):
                    t = self.triples[idx]
                    layer_triples.append(t)
                    next_frontier.add(t["object"].lower())

                for idx in self.by_object.get(ent, []):
                    t = self.triples[idx]
                    layer_triples.append(t)
                    next_frontier.add(t["subject"].lower())

            layers[f"depth_{d}"] = layer_triples[:max_results]
            frontier = next_frontier - visited

        return layers

    def entities(self) -> list:
        """List all unique entities."""
        subjects = set()
        objects = set()
        for t in self.triples:
            subjects.add(t.get("subject", ""))
            objects.add(t.get("object", ""))
        return sorted(subjects | objects)

    def stats(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "triple_count": len(self.triples),
            "entity_count": len(self.all_entities),
            "predicate_count": len(self.by_predicate),
            "sources": self.sources[:5],
        }


class PackRegistry:
    """Registry of all loaded packs."""

    def __init__(self, packs_dir: Path, filter_packs: Optional[list] = None):
        self.packs = {}
        self._load_all(packs_dir, filter_packs)

    def _load_all(self, packs_dir: Path, filter_packs: Optional[list]):
        for entry in sorted(packs_dir.iterdir()):
            if not entry.is_dir():
                continue
            if (entry / "triples.json").exists():
                name = entry.name
                if filter_packs and name not in filter_packs:
                    continue
                self.packs[name] = KnowledgeIndex(name, entry)

    def get(self, name: str) -> Optional[KnowledgeIndex]:
        return self.packs.get(name)

    def list_packs(self) -> list:
        return [p.stats() for p in self.packs.values()]

    def search_all(self, query: str, max_per_pack: int = 5) -> dict:
        """Search across all packs."""
        results = {}
        for name, pack in self.packs.items():
            hits = pack.query(query, max_results=max_per_pack)
            if hits:
                results[name] = hits
        return results


class PackHandler(BaseHTTPRequestHandler):
    registry: PackRegistry = None

    def do_GET(self):
        if self.path == "/health":
            self._json_response({"status": "ok", "packs": len(self.registry.packs)})
        elif self.path == "/packs":
            self._json_response(self.registry.list_packs())
        elif self.path.startswith("/packs/"):
            name = self.path.split("/")[2]
            pack = self.registry.get(name)
            if pack:
                self._json_response(pack.stats())
            else:
                self._json_response({"error": f"pack not found: {name}"}, 404)
        elif self.path.startswith("/entities/"):
            name = self.path.split("/")[2]
            pack = self.registry.get(name)
            if pack:
                self._json_response(pack.entities())
            else:
                self._json_response({"error": f"pack not found: {name}"}, 404)
        else:
            self._json_response({
                "endpoints": {
                    "GET /packs": "List all loaded packs",
                    "GET /packs/<name>": "Pack stats",
                    "GET /entities/<name>": "List entities in a pack",
                    "GET /health": "Health check",
                    "POST /query": "Search triples {pack, query, max_results}",
                    "POST /lookup": "Entity lookup {pack, entity, role}",
                    "POST /traverse": "Graph walk {pack, entity, depth}",
                    "POST /search_all": "Search all packs {query}",
                    "POST /ask": "Natural language question {pack, question}",
                },
            })

    def do_POST(self):
        body = self._read_body()
        if not body:
            self._json_response({"error": "invalid JSON body"}, 400)
            return

        if self.path == "/query":
            pack = self.registry.get(body.get("pack", ""))
            if not pack:
                self._json_response({"error": f"pack not found: {body.get('pack')}"}, 404)
                return
            results = pack.query(body.get("query", ""), body.get("max_results", 20))
            self._json_response({"pack": pack.name, "query": body.get("query"), "results": results})

        elif self.path == "/lookup":
            pack = self.registry.get(body.get("pack", ""))
            if not pack:
                self._json_response({"error": f"pack not found: {body.get('pack')}"}, 404)
                return
            results = pack.lookup(body.get("entity", ""), body.get("role", "any"))
            self._json_response({"pack": pack.name, "entity": body.get("entity"), "results": results})

        elif self.path == "/traverse":
            pack = self.registry.get(body.get("pack", ""))
            if not pack:
                self._json_response({"error": f"pack not found: {body.get('pack')}"}, 404)
                return
            results = pack.traverse(
                body.get("entity", ""),
                body.get("depth", 1),
                body.get("max_results", 50),
            )
            self._json_response({"pack": pack.name, "entity": body.get("entity"), "layers": results})

        elif self.path == "/search_all":
            results = self.registry.search_all(
                body.get("query", ""),
                body.get("max_per_pack", 5),
            )
            self._json_response({"query": body.get("query"), "packs": results})

        elif self.path == "/ask":
            pack = self.registry.get(body.get("pack", ""))
            if not pack:
                self._json_response({"error": f"pack not found: {body.get('pack')}"}, 404)
                return
            question = body.get("question", "")
            triples = pack.query(question, max_results=10)
            context = "\n".join(
                f"- {t['subject']} [{t['predicate']}] {t['object']}"
                for t in triples
            )
            self._json_response({
                "pack": pack.name,
                "question": question,
                "relevant_triples": triples,
                "context": context,
                "note": "Use context as grounding for LLM-based answering",
            })

        else:
            self._json_response({"error": f"unknown endpoint: {self.path}"}, 404)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return None
        try:
            return json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, ValueError):
            return None

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def log_message(self, format, *args):
        pass  # Suppress default logging


def main():
    parser = argparse.ArgumentParser(description="Pharos Pack Server")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--packs", help="Comma-separated pack names (default: all)")
    parser.add_argument("--packs-dir", default=str(PACKS_DIR))
    args = parser.parse_args()

    filter_packs = args.packs.split(",") if args.packs else None
    packs_dir = Path(args.packs_dir)

    print(f"Loading packs from {packs_dir}...")
    registry = PackRegistry(packs_dir, filter_packs)
    print(f"Loaded {len(registry.packs)} packs:")
    for name, pack in sorted(registry.packs.items()):
        print(f"  {name}: {len(pack.triples)} triples")

    PackHandler.registry = registry

    server = HTTPServer(("0.0.0.0", args.port), PackHandler)
    server.socket.setsockopt(__import__('socket').SOL_SOCKET, __import__('socket').SO_REUSEADDR, 1)
    total_triples = sum(len(p.triples) for p in registry.packs.values())
    print(f"\nPharos Pack Server listening on http://localhost:{args.port}")
    print(f"  {len(registry.packs)} packs, {total_triples} triples indexed")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
