"""Conceptual schema for the movie knowledge graph."""
from __future__ import annotations

GRAPH_SCHEMA = {
    "nodes": {
        "Movie": {
            "properties": ["title", "year", "rating", "tagline"],
            "description": "A movie entity.",
        },
        "Actor": {
            "properties": ["name", "born"],
            "description": "An actor node.",
        },
        "Director": {
            "properties": ["name", "born"],
            "description": "A director node.",
        },
        "User": {
            "properties": ["name"],
            "description": "A user who rates movies.",
        },
        "Genre": {
            "properties": ["name"],
            "description": "A movie genre label.",
        },
    },
    "relationships": {
        "ACTED_IN": {"from": "Actor", "to": "Movie", "properties": ["roles"]},
        "DIRECTED": {"from": "Director", "to": "Movie", "properties": []},
        "RATED": {"from": "User", "to": "Movie", "properties": ["rating"]},
        "IN_GENRE": {"from": "Movie", "to": "Genre", "properties": []},
    },
}


def print_schema() -> None:
    """Pretty print the graph schema in terminal."""
    print("=" * 60)
    print("KNOWLEDGE GRAPH SCHEMA")
    print("=" * 60)

    print("\n--- NODE TYPES ---")
    for node_type, info in GRAPH_SCHEMA["nodes"].items():
        print(f"\n[{node_type}]")
        print(f"  Properties: {', '.join(info['properties'])}")
        print(f"  Description: {info['description']}")

    print("\n--- RELATIONSHIP TYPES ---")
    for rel_type, info in GRAPH_SCHEMA["relationships"].items():
        print(f"\n({info['from']})-[:{rel_type}]->({info['to']})")
        if info["properties"]:
            print(f"  Properties: {', '.join(info['properties'])}")


if __name__ == "__main__":
    print_schema()
