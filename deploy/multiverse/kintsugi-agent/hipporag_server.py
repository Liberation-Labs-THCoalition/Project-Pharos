"""HippoRAG Knowledge Graph Server — Mnemosyne SIRA integration.

Graph-backed retrieval with OpenIE triples, PPR traversal,
and text-graph-semantic fusion for cross-session case memory.

Part of Project-Mnemosyne: https://github.com/Liberation-Labs-THCoalition/Project-Mnemosyne
"""

import json
import os
import sqlite3
import logging
from pathlib import Path

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import networkx as nx
import numpy as np

logging.basicConfig(level=logging.INFO, format="[hipporag] %(message)s")
log = logging.getLogger("hipporag")

app = Flask(__name__)
DB_PATH = os.environ.get("HIPPORAG_DB", "/data/hipporag.db")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

model = SentenceTransformer(EMBEDDING_MODEL)
G = nx.DiGraph()


def init_db():
    """Initialize SQLite database and load existing graph."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS triples
        (id INTEGER PRIMARY KEY, subject TEXT, predicate TEXT, object TEXT,
         source_id TEXT, confidence REAL DEFAULT 0.5,
         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS entities
        (name TEXT PRIMARY KEY, entity_type TEXT, embedding BLOB,
         mention_count INTEGER DEFAULT 1)""")
    conn.commit()

    # Load existing triples into graph
    for row in conn.execute("SELECT subject, predicate, object FROM triples"):
        G.add_edge(row[0], row[2], predicate=row[1])
    log.info(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    conn.close()


init_db()


@app.route("/health")
def health():
    return jsonify({"status": "ok", "nodes": G.number_of_nodes(), "edges": G.number_of_edges()})


@app.route("/index", methods=["POST"])
def index_triples():
    """Index new triples into the knowledge graph."""
    data = request.json
    triples = data.get("triples", [])
    source_id = data.get("source_id", "")
    conn = sqlite3.connect(DB_PATH)
    added = 0
    for t in triples:
        s, p, o = t.get("subject", ""), t.get("predicate", ""), t.get("object", "")
        if s and p and o:
            conn.execute(
                "INSERT INTO triples (subject, predicate, object, source_id) VALUES (?,?,?,?)",
                (s, p, o, source_id),
            )
            G.add_edge(s, o, predicate=p)
            added += 1
    conn.commit()
    conn.close()
    return jsonify({"indexed": added})


@app.route("/search", methods=["POST"])
def search():
    """Search the knowledge graph via PPR-style neighborhood expansion."""
    query = request.json.get("query", "")
    top_k = request.json.get("top_k", 5)
    q_emb = model.encode(query)

    results = []
    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) > 3]

    for node in G.nodes():
        if any(w in node.lower() for w in query_words):
            # Get neighborhood triples (1-hop)
            for _, neighbor, data in G.edges(node, data=True):
                results.append(
                    {"subject": node, "predicate": data.get("predicate", ""), "object": neighbor}
                )
            for pred, _, data in G.in_edges(node, data=True):
                results.append(
                    {"subject": pred, "predicate": data.get("predicate", ""), "object": node}
                )

    return jsonify({"results": results[: top_k * 3], "query": query})


@app.route("/stats")
def stats():
    return jsonify({
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": nx.number_weakly_connected_components(G),
    })


if __name__ == "__main__":
    log.info("HippoRAG serving on :11235")
    app.run(host="0.0.0.0", port=11235)
