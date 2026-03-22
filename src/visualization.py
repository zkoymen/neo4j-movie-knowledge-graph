"""Visualization helpers for Phase 1."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def visualize_schema() -> Path:
    """
    Draw conceptual schema as an image.

    This plot is not database data. It is just the model diagram.
    """
    graph = nx.DiGraph()

    # Node colors are fixed so students can read quickly.
    node_colors = {
        "Movie": "#4C8BF5",
        "Actor": "#34A853",
        "Director": "#FB8C00",
        "User": "#8E24AA",
        "Genre": "#EA4335",
        "Country": "#00897B",
    }
    for node in node_colors:
        graph.add_node(node)

    # Keep edges explicit for clarity.
    edges = [
        ("Actor", "Movie", "ACTED_IN"),
        ("Director", "Movie", "DIRECTED"),
        ("User", "Movie", "RATED"),
        ("Movie", "Genre", "IN_GENRE"),
        ("Movie", "Country", "IN_COUNTRY"),
    ]
    for source, target, label in edges:
        graph.add_edge(source, target, label=label)

    fig, ax = plt.subplots(figsize=(11, 7))
    pos = nx.spring_layout(graph, seed=42)
    colors = [node_colors[node] for node in graph.nodes()]

    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color=colors,
        node_size=3000,
        font_size=12,
        font_weight="bold",
        arrows=True,
        arrowsize=18,
        ax=ax,
    )

    # Draw relationship names on edges.
    edge_labels = nx.get_edge_attributes(graph, "label")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=9, ax=ax)

    ax.set_title("Knowledge Graph Schema")
    fig.tight_layout()

    output_path = OUTPUT_DIR / "schema_diagram.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
