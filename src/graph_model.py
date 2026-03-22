"""Conceptual schema for the movie knowledge graph."""
from __future__ import annotations

GRAPH_SCHEMA = {
    "nodes": {
        "Movie": {
            "properties": ["title", "released", "tagline"],
            "description": "A movie entity.",
        },
        "Person": {
            "properties": ["name", "born"],
            "description": "A person entity (actor, director, writer...).",
        },
        "Genre": {
            "properties": ["name"],
            "description": "A movie genre label.",
        },
    },
    "relationships": {
        "ACTED_IN": {"from": "Person", "to": "Movie", "properties": ["roles"]},
        "DIRECTED": {"from": "Person", "to": "Movie", "properties": []},
        "PRODUCED": {"from": "Person", "to": "Movie", "properties": []},
        "WROTE": {"from": "Person", "to": "Movie", "properties": []},
        "REVIEWED": {"from": "Person", "to": "Movie", "properties": ["summary", "rating"]},
        "BELONGS_TO": {"from": "Movie", "to": "Genre", "properties": []},
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
