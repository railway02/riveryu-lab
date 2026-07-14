#!/usr/bin/env python3
"""Validate the generated RiverYu's Lab knowledge graph JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ALLOWED_RELATION_TYPES = {
    "builds-on",
    "extends",
    "related",
    "contrasts-with",
    "applies-to",
}

ALLOWED_TOPICS = {
    "aigc-authenticity",
    "generative-representations",
    "deep-learning-foundations",
    "3d-vision",
    "research-engineering",
}


def nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def load_graph(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as graph_file:
            data = json.load(graph_file)
    except FileNotFoundError as error:
        raise ValueError(f"graph file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid JSON in {path} at line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(f"graph root must be an object: {path}")
    return data


def validate_graph(data: dict) -> list[str]:
    errors: list[str] = []
    nodes = data.get("nodes")
    edges = data.get("edges")

    if not isinstance(nodes, list):
        errors.append("nodes must be an array")
        nodes = []
    if not isinstance(edges, list):
        errors.append("edges must be an array")
        edges = []

    known_uids: set[str] = set()
    duplicate_uids: set[str] = set()

    for index, node in enumerate(nodes):
        location = f"nodes[{index}]"
        if not isinstance(node, dict):
            errors.append(f"{location}: node must be an object")
            continue

        uid = node.get("uid")
        if not nonempty_string(uid):
            errors.append(f"{location}: uid is missing or empty")
            continue

        uid = uid.strip()
        if uid in known_uids:
            duplicate_uids.add(uid)
        known_uids.add(uid)

        if not nonempty_string(node.get("title")):
            errors.append(f"node {uid}: title is missing or empty")
        if not nonempty_string(node.get("url")):
            errors.append(f"node {uid}: URL is missing or empty")

        topics = node.get("topics")
        if not isinstance(topics, list):
            errors.append(f"node {uid}: topics must be an array")
        else:
            if not topics:
                errors.append(f"node {uid}: must have at least one topic")
            if len(topics) > 2:
                errors.append(f"node {uid}: has more than two topics")
            seen_topics: set[str] = set()
            for topic in topics:
                if not isinstance(topic, str) or topic not in ALLOWED_TOPICS:
                    errors.append(f"node {uid}: unknown topic {topic!r}")
                    continue
                if topic in seen_topics:
                    errors.append(f"node {uid}: duplicate topic {topic!r}")
                seen_topics.add(topic)

    for uid in sorted(duplicate_uids):
        errors.append(f"duplicate uid: {uid}")

    seen_edges: set[tuple[str, str]] = set()
    seen_symmetric_edges: set[tuple[str, str, str]] = set()
    for index, edge in enumerate(edges):
        location = f"edges[{index}]"
        if not isinstance(edge, dict):
            errors.append(f"{location}: edge must be an object")
            continue

        source = edge.get("source")
        target = edge.get("target")
        relation_type = edge.get("type")
        note = edge.get("note")

        if not nonempty_string(source):
            errors.append(f"{location}: source is missing or empty")
            continue
        if not nonempty_string(target):
            errors.append(f"edge from {source}: target is missing or empty")
            continue

        source = source.strip()
        target = target.strip()

        if source not in known_uids:
            errors.append(f"edge {source} -> {target}: source uid does not exist")
        if target not in known_uids:
            errors.append(f"edge {source} -> {target}: target uid does not exist")
        if source == target:
            errors.append(f"edge {source} -> {target}: self-connection is not allowed")
        if relation_type not in ALLOWED_RELATION_TYPES:
            errors.append(
                f"edge {source} -> {target}: unknown relation type {relation_type!r}"
            )
        if not nonempty_string(note):
            errors.append(f"edge {source} -> {target}: note is missing or empty")

        edge_key = (source, target)
        duplicate_directed_edge = edge_key in seen_edges
        if duplicate_directed_edge:
            errors.append(f"duplicate edge: {source} -> {target}")
        seen_edges.add(edge_key)

        if relation_type in {"related", "contrasts-with"}:
            first, second = sorted((source, target))
            symmetric_key = (relation_type, first, second)
            if symmetric_key in seen_symmetric_edges and not duplicate_directed_edge:
                errors.append(
                    "duplicate bidirectional relation: "
                    f"{source} {relation_type} {target}"
                )
            seen_symmetric_edges.add(symmetric_key)

    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} PATH_TO_GRAPH_JSON", file=sys.stderr)
        return 2

    graph_path = Path(sys.argv[1])
    try:
        graph = load_graph(graph_path)
    except ValueError as error:
        print(f"graph validation failed: {error}", file=sys.stderr)
        return 1

    errors = validate_graph(graph)
    if errors:
        print(f"graph validation failed with {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        "graph validation passed: "
        f"{len(graph.get('nodes', []))} nodes, {len(graph.get('edges', []))} edges"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
