"""Knowledge graph built from wiki page cross-references.

Supports 1-hop expansion for Chat RAG enrichment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.wiki.frontmatter import parse_frontmatter


@dataclass
class GraphNode:
    page_id: str
    title: str
    page_type: str
    path: str
    neighbors: list[str] = field(default_factory=list)


@dataclass
class WikiGraph:
    nodes: dict[str, GraphNode]
    edges: list[tuple[str, str]]


def _extract_wikilinks(text: str) -> list[str]:
    """Extract [[wikilink]] targets from markdown body."""
    return re.findall(r"\[\[([^\]]+)\]\]", text)


def build_wiki_graph(project_dir: str) -> WikiGraph:
    """Build a graph from wiki pages' frontmatter `related` + wikilinks."""
    wiki_dir = Path(project_dir) / "wiki"
    if not wiki_dir.exists():
        return WikiGraph(nodes={}, edges=[])

    nodes: dict[str, GraphNode] = {}
    edges: list[tuple[str, str]] = []

    # First pass: collect all pages
    for md_file in wiki_dir.rglob("*.md"):
        rel = str(md_file.relative_to(Path(project_dir)))
        page_id = md_file.stem
        content = md_file.read_text(encoding="utf-8", errors="replace")
        meta, body = parse_frontmatter(content)

        nodes[page_id] = GraphNode(
            page_id=page_id,
            title=meta.title or page_id,
            page_type=meta.type,
            path=rel,
        )

        # Edges from frontmatter `related`
        for rel_slug in meta.related:
            edges.append((page_id, rel_slug))

        # Edges from [[wikilinks]] in body
        for link_target in _extract_wikilinks(body):
            slug = link_target.strip().lower().replace(" ", "-")
            edges.append((page_id, slug))

    # Second pass: populate neighbor lists
    for src, dst in edges:
        if src in nodes:
            nodes[src].neighbors.append(dst)
        if dst in nodes:
            nodes[dst].neighbors.append(src)

    # Deduplicate neighbors
    for node in nodes.values():
        node.neighbors = sorted(set(node.neighbors))

    return WikiGraph(nodes=nodes, edges=edges)


def graph_expand(
    seed_page_ids: list[str],
    graph: WikiGraph,
    *,
    max_related: int = 3,
    min_hops: int = 1,
) -> list[str]:
    """1-hop expansion: return related page_ids not already in seed set."""
    seed_set = set(seed_page_ids)
    expanded: dict[str, int] = {}

    for pid in seed_page_ids:
        node = graph.nodes.get(pid)
        if not node:
            continue
        for neighbor in node.neighbors:
            if neighbor not in seed_set and neighbor in graph.nodes:
                expanded[neighbor] = expanded.get(neighbor, 0) + 1

    # Sort by how many seed pages link to them (relevance proxy)
    ranked = sorted(expanded.items(), key=lambda x: -x[1])
    return [pid for pid, _ in ranked[:max_related * len(seed_page_ids)]]


def graph_to_json(graph: WikiGraph) -> dict:
    """Serialize graph for API response."""
    return {
        "nodes": [
            {
                "id": n.page_id,
                "title": n.title,
                "type": n.page_type,
                "path": n.path,
                "neighbors": n.neighbors,
            }
            for n in graph.nodes.values()
        ],
        "edges": [{"source": s, "target": t} for s, t in graph.edges],
    }
