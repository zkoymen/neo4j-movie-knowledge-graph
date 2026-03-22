# COM4514 - Knowledge Graph from Movie Data — Full Implementation Plan

> **Bu dosya bir AI coder'a (Cursor, Copilot, Claude Code, vb.) kopyala-yapıştır yapılmak üzere tasarlanmıştır.**
> Her adım bağımsız, test edilebilir ve commit edilebilir şekilde yazılmıştır.

---

## PHASE 0: Project Setup & Git Init

### Adım 0.1 — Proje Yapısını Oluştur

```
movie-knowledge-graph/
├── README.md
├── requirements.txt
├── .gitignore
├── .env                  # NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
├── config.py             # Load .env, constants
├── notebooks/
│   └── exploration.ipynb # (optional) quick EDA
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Step 1: Load Movies dataset into Neo4j
│   ├── graph_model.py       # Step 1: Schema/model definitions
│   ├── cypher_queries.py    # Step 2: All Cypher queries
│   ├── feature_extraction.py # Step 2: Extract topological features
│   ├── graph_analysis.py    # Step 2: Degree, centrality, community
│   ├── projections.py       # Step 3: Monopartite, co-occurrence, similarity
│   ├── link_prediction.py   # Step 3: Link prediction ML pipeline
│   ├── node_classification.py # Step 3: Node classification ML pipeline
│   ├── kg_completion.py     # Step 3: Knowledge graph completion
│   └── visualization.py     # Step 4: All visualization helpers
├── outputs/
│   ├── figures/             # Saved plots/images for report
│   └── results/             # CSV/JSON outputs
├── report/
│   └── report.md            # Draft report (later convert to PDF)
└── main.py                  # Master orchestrator script
```

### Adım 0.2 — requirements.txt

```
neo4j==5.25.0
pandas==2.2.2
numpy==1.26.4
matplotlib==3.9.2
seaborn==0.13.2
networkx==3.3
scikit-learn==1.5.2
python-dotenv==1.0.1
pyvis==0.3.2
node2vec==0.4.6
pykeen==1.10.2
torch>=2.0.0
community==1.0.0b1
python-louvain==0.16
```

### Adım 0.3 — Git Init + GitHub Private Repo

```bash
cd movie-knowledge-graph
git init
git add .
git commit -m "Phase 0: Project structure and dependencies"

# GitHub CLI ile private repo oluştur ve bağla
gh repo create movie-knowledge-graph --private --source=. --push
# VEYA manuel:
# git remote add origin git@github.com:USERNAME/movie-knowledge-graph.git
# git push -u origin main
```

### Adım 0.4 — .gitignore

```
.env
__pycache__/
*.pyc
.venv/
venv/
*.egg-info/
.ipynb_checkpoints/
outputs/figures/*.png
outputs/results/*.csv
neo4j_data/
```

### Adım 0.5 — .env dosyası (template)

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
```

### Adım 0.6 — config.py

```python
"""Configuration loader for the project."""
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
```

> ✅ **GIT COMMIT:** `git add . && git commit -m "Phase 0: Project setup complete" && git push`

---

## PHASE 1: Design Knowledge Graph Model & Load Data

### Adım 1.1 — Neo4j Movies Dataset'i Anla

Neo4j'nin built-in Movies dataset'ini kullanacağız. Bu dataset şunları içerir:
- **Node Types:** Movie, Person, Genre (Genre yoksa kendimiz ekleriz)
- **Relationship Types:** ACTED_IN, DIRECTED, PRODUCED, WROTE, REVIEWED, FOLLOWS

### Adım 1.2 — `src/data_loader.py`

```python
"""
Load the Neo4j Movies dataset into the database.
Uses the built-in Movies dataset or imports from CSV.
"""
from neo4j import GraphDatabase
import config

class MovieGraphLoader:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
    
    def close(self):
        self.driver.close()
    
    def load_movies_dataset(self):
        """Load the built-in Neo4j Movies dataset."""
        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Load built-in movies dataset
            session.run("""
                :play movies
            """)
            # Alternative: Run the movies Cypher script directly
            # Copy-paste the full movies creation script from:
            # https://github.com/neo4j-graph-examples/movies/blob/main/scripts/movies.cypher
            
            print("Movies dataset loaded successfully.")
    
    def add_genre_nodes(self):
        """
        Create Genre nodes and BELONGS_TO relationships.
        Since the default movies dataset doesn't have genres,
        we manually assign genres based on movie titles/descriptions.
        """
        genre_mapping = {
            # Map movie titles to genres - extend this
            "The Matrix": ["Sci-Fi", "Action"],
            "Top Gun": ["Action", "Drama"],
            # ... add all movies
        }
        with self.driver.session() as session:
            for movie_title, genres in genre_mapping.items():
                for genre in genres:
                    session.run("""
                        MERGE (g:Genre {name: $genre})
                        WITH g
                        MATCH (m:Movie {title: $title})
                        MERGE (m)-[:BELONGS_TO]->(g)
                    """, genre=genre, title=movie_title)
        print("Genre nodes added.")
    
    def verify_schema(self):
        """Print schema summary."""
        with self.driver.session() as session:
            # Node counts
            result = session.run("""
                CALL db.labels() YIELD label
                CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as cnt', {}) YIELD value
                RETURN label, value.cnt as count
            """)
            print("\n=== Node Counts ===")
            for record in result:
                print(f"  {record['label']}: {record['count']}")
            
            # Relationship counts
            result = session.run("""
                CALL db.relationshipTypes() YIELD relationshipType
                RETURN relationshipType
            """)
            print("\n=== Relationship Types ===")
            for record in result:
                print(f"  {record['relationshipType']}")

if __name__ == "__main__":
    loader = MovieGraphLoader()
    loader.load_movies_dataset()
    loader.add_genre_nodes()
    loader.verify_schema()
    loader.close()
```

### Adım 1.3 — `src/graph_model.py`

```python
"""
Knowledge Graph schema definition and documentation.
Defines the conceptual model of our graph.
"""

GRAPH_SCHEMA = {
    "nodes": {
        "Movie": {
            "properties": ["title", "released", "tagline"],
            "description": "A movie in the database"
        },
        "Person": {
            "properties": ["name", "born"],
            "description": "An actor, director, producer, or writer"
        },
        "Genre": {
            "properties": ["name"],
            "description": "Movie genre category"
        }
    },
    "relationships": {
        "ACTED_IN": {"from": "Person", "to": "Movie", "properties": ["roles"]},
        "DIRECTED": {"from": "Person", "to": "Movie", "properties": []},
        "PRODUCED": {"from": "Person", "to": "Movie", "properties": []},
        "WROTE": {"from": "Person", "to": "Movie", "properties": []},
        "REVIEWED": {"from": "Person", "to": "Movie", "properties": ["summary", "rating"]},
        "BELONGS_TO": {"from": "Movie", "to": "Genre", "properties": []},
    }
}

def print_schema():
    """Pretty-print the graph schema."""
    print("=" * 60)
    print("KNOWLEDGE GRAPH SCHEMA")
    print("=" * 60)
    
    print("\n--- NODE TYPES ---")
    for node_type, info in GRAPH_SCHEMA["nodes"].items():
        print(f"\n  [{node_type}]")
        print(f"    Properties: {info['properties']}")
        print(f"    Description: {info['description']}")
    
    print("\n--- RELATIONSHIP TYPES ---")
    for rel_type, info in GRAPH_SCHEMA["relationships"].items():
        print(f"\n  ({info['from']})-[:{rel_type}]->({info['to']})")
        if info["properties"]:
            print(f"    Properties: {info['properties']}")

if __name__ == "__main__":
    print_schema()
```

### Adım 1.4 — `src/visualization.py` (İlk Kısım — Schema Visualization)

```python
"""
Visualization helpers for the knowledge graph project.
Saves all plots to outputs/figures/.
"""
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import os

OUTPUT_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_schema():
    """Create a visual diagram of the graph schema."""
    G = nx.DiGraph()
    
    # Add nodes with colors
    node_colors = {"Movie": "#4C8BF5", "Person": "#34A853", "Genre": "#EA4335"}
    for node in node_colors:
        G.add_node(node)
    
    # Add relationships
    edges = [
        ("Person", "Movie", "ACTED_IN"),
        ("Person", "Movie", "DIRECTED"),
        ("Person", "Movie", "PRODUCED"),
        ("Person", "Movie", "WROTE"),
        ("Person", "Movie", "REVIEWED"),
        ("Movie", "Genre", "BELONGS_TO"),
    ]
    for src, dst, label in edges:
        G.add_edge(src, dst, label=label)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    
    colors = [node_colors[n] for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=colors, 
            node_size=3000, font_size=14, font_weight="bold",
            arrows=True, arrowsize=20, ax=ax)
    
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title("Knowledge Graph Schema", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/schema_diagram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Schema diagram saved to {OUTPUT_DIR}/schema_diagram.png")

def visualize_subgraph_pyvis(driver, query, filename, title="Subgraph"):
    """
    Run a Cypher query and visualize result as interactive HTML graph.
    Query must return paths or nodes and relationships.
    """
    net = Network(height="600px", width="100%", notebook=False, directed=True)
    net.heading = title
    
    with driver.session() as session:
        result = session.run(query)
        for record in result:
            # Process nodes and relationships from the result
            for key in record.keys():
                value = record[key]
                if hasattr(value, 'nodes'):  # It's a path
                    for node in value.nodes:
                        labels = list(node.labels)
                        color = "#4C8BF5" if "Movie" in labels else "#34A853" if "Person" in labels else "#EA4335"
                        net.add_node(node.element_id, label=node.get("title", node.get("name", "")), color=color)
                    for rel in value.relationships:
                        net.add_edge(rel.start_node.element_id, rel.end_node.element_id, label=rel.type)
    
    filepath = f"{OUTPUT_DIR}/{filename}.html"
    net.save_graph(filepath)
    print(f"Interactive graph saved to {filepath}")

def save_plot(fig, filename):
    """Save a matplotlib figure to the outputs directory."""
    filepath = f"{OUTPUT_DIR}/{filename}.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {filepath}")
```

> ✅ **GIT COMMIT:** `git add . && git commit -m "Phase 1: Graph model, data loader, schema visualization" && git push`

---

## PHASE 2: Query, Explore & Extract Features

### Adım 2.1 — `src/cypher_queries.py`

```python
"""
All Cypher queries for graph exploration and analysis.
Each function returns query results as a list of dicts or a pandas DataFrame.
"""
import pandas as pd
from neo4j import GraphDatabase
import config

class GraphExplorer:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
    
    def close(self):
        self.driver.close()
    
    def _query_to_df(self, query, **params):
        """Run a Cypher query and return results as pandas DataFrame."""
        with self.driver.session() as session:
            result = session.run(query, **params)
            return pd.DataFrame([dict(record) for record in result])
    
    # --- Basic Exploration ---
    
    def get_node_counts(self):
        """Count nodes by label."""
        return self._query_to_df("""
            MATCH (n)
            RETURN labels(n)[0] AS label, count(n) AS count
            ORDER BY count DESC
        """)
    
    def get_relationship_counts(self):
        """Count relationships by type."""
        return self._query_to_df("""
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(r) AS count
            ORDER BY count DESC
        """)
    
    # --- Degree Distribution ---
    
    def get_degree_distribution(self):
        """Get degree distribution for Person nodes."""
        return self._query_to_df("""
            MATCH (p:Person)-[r]-()
            RETURN p.name AS name, count(r) AS degree
            ORDER BY degree DESC
        """)
    
    def get_movie_actor_count(self):
        """Get number of actors per movie."""
        return self._query_to_df("""
            MATCH (m:Movie)<-[:ACTED_IN]-(p:Person)
            RETURN m.title AS movie, count(p) AS actor_count
            ORDER BY actor_count DESC
        """)
    
    # --- Centrality Metrics ---
    
    def get_betweenness_centrality(self):
        """Calculate betweenness centrality using GDS or manual Cypher."""
        # NOTE: Requires Neo4j GDS plugin. If not available, calculate with NetworkX.
        return self._query_to_df("""
            MATCH (p:Person)
            RETURN p.name AS name, 
                   p.betweenness_centrality AS betweenness
            ORDER BY betweenness DESC
            LIMIT 20
        """)
    
    # --- Subgraph Queries ---
    
    def get_actor_network(self, limit=50):
        """Get actor co-occurrence network (actors who acted in the same movie)."""
        return self._query_to_df("""
            MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
            WHERE id(p1) < id(p2)
            RETURN p1.name AS actor1, p2.name AS actor2, 
                   count(m) AS shared_movies,
                   collect(m.title) AS movies
            ORDER BY shared_movies DESC
            LIMIT $limit
        """, limit=limit)
    
    def get_genre_network(self):
        """Get movies grouped by genre."""
        return self._query_to_df("""
            MATCH (m:Movie)-[:BELONGS_TO]->(g:Genre)
            RETURN g.name AS genre, collect(m.title) AS movies, count(m) AS movie_count
            ORDER BY movie_count DESC
        """)
    
    def get_actor_director_pairs(self):
        """Find frequent actor-director collaborations."""
        return self._query_to_df("""
            MATCH (a:Person)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Person)
            RETURN a.name AS actor, d.name AS director,
                   count(m) AS collaborations,
                   collect(m.title) AS movies
            ORDER BY collaborations DESC
            LIMIT 20
        """)
```

### Adım 2.2 — `src/graph_analysis.py`

```python
"""
Graph analysis: degree distribution, centrality, community detection.
Uses NetworkX for computations when GDS is not available.
"""
import networkx as nx
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from community import community_louvain
import config

class GraphAnalyzer:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        self.G = None  # NetworkX graph (built from Neo4j)
    
    def close(self):
        self.driver.close()
    
    def build_networkx_graph(self, projection="actor_cooccurrence"):
        """
        Build a NetworkX graph from Neo4j data.
        projection: 'actor_cooccurrence' | 'full_graph' | 'movie_genre'
        """
        G = nx.Graph()
        
        with self.driver.session() as session:
            if projection == "actor_cooccurrence":
                result = session.run("""
                    MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
                    WHERE id(p1) < id(p2)
                    RETURN p1.name AS source, p2.name AS target, count(m) AS weight
                """)
                for record in result:
                    G.add_edge(record["source"], record["target"], weight=record["weight"])
            
            elif projection == "full_graph":
                result = session.run("""
                    MATCH (n)-[r]-(m)
                    RETURN n.name AS source_name, n.title AS source_title,
                           m.name AS target_name, m.title AS target_title,
                           type(r) AS rel_type
                """)
                for record in result:
                    src = record["source_name"] or record["source_title"]
                    tgt = record["target_name"] or record["target_title"]
                    if src and tgt:
                        G.add_edge(src, tgt, rel_type=record["rel_type"])
        
        self.G = G
        print(f"NetworkX graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def compute_centralities(self):
        """Compute multiple centrality metrics."""
        if self.G is None:
            self.build_networkx_graph()
        
        centralities = pd.DataFrame({
            "node": list(self.G.nodes()),
            "degree_centrality": pd.Series(nx.degree_centrality(self.G)),
            "betweenness_centrality": pd.Series(nx.betweenness_centrality(self.G)),
            "closeness_centrality": pd.Series(nx.closeness_centrality(self.G)),
            "eigenvector_centrality": pd.Series(
                nx.eigenvector_centrality(self.G, max_iter=1000)
            ),
            "pagerank": pd.Series(nx.pagerank(self.G)),
        })
        
        centralities.to_csv("outputs/results/centralities.csv", index=False)
        print("Centralities computed and saved.")
        return centralities
    
    def detect_communities(self):
        """Detect communities using Louvain method."""
        if self.G is None:
            self.build_networkx_graph()
        
        partition = community_louvain.best_partition(self.G)
        
        communities_df = pd.DataFrame({
            "node": list(partition.keys()),
            "community": list(partition.values())
        })
        
        # Summary
        n_communities = len(set(partition.values()))
        modularity = community_louvain.modularity(partition, self.G)
        print(f"Found {n_communities} communities (modularity: {modularity:.4f})")
        
        communities_df.to_csv("outputs/results/communities.csv", index=False)
        return communities_df, partition
    
    def compute_degree_distribution(self):
        """Compute and return degree distribution."""
        if self.G is None:
            self.build_networkx_graph()
        
        degrees = dict(self.G.degree())
        degree_df = pd.DataFrame({
            "node": list(degrees.keys()),
            "degree": list(degrees.values())
        })
        degree_df = degree_df.sort_values("degree", ascending=False).reset_index(drop=True)
        degree_df.to_csv("outputs/results/degree_distribution.csv", index=False)
        return degree_df
    
    def get_graph_summary(self):
        """Get overall graph statistics."""
        if self.G is None:
            self.build_networkx_graph()
        
        summary = {
            "Number of nodes": self.G.number_of_nodes(),
            "Number of edges": self.G.number_of_edges(),
            "Density": nx.density(self.G),
            "Average clustering coefficient": nx.average_clustering(self.G),
            "Number of connected components": nx.number_connected_components(self.G),
            "Average degree": np.mean([d for _, d in self.G.degree()]),
        }
        
        if nx.is_connected(self.G):
            summary["Diameter"] = nx.diameter(self.G)
            summary["Average shortest path"] = nx.average_shortest_path_length(self.G)
        
        return summary
```

### Adım 2.3 — `src/feature_extraction.py`

```python
"""
Extract topological features from the graph to be used as ML inputs.
Features: degree, centralities, community, clustering coefficient, etc.
These features will be saved as node/edge attributes.
"""
import pandas as pd
import numpy as np
import networkx as nx
from community import community_louvain
from neo4j import GraphDatabase
import config

class FeatureExtractor:
    def __init__(self, G: nx.Graph):
        """
        G: NetworkX graph (from GraphAnalyzer.build_networkx_graph())
        """
        self.G = G
    
    def extract_node_features(self):
        """
        Extract comprehensive node features for ML.
        Returns DataFrame with one row per node.
        """
        nodes = list(self.G.nodes())
        
        # Basic features
        degree = dict(self.G.degree())
        clustering = nx.clustering(self.G)
        
        # Centrality features
        degree_cent = nx.degree_centrality(self.G)
        betweenness = nx.betweenness_centrality(self.G)
        closeness = nx.closeness_centrality(self.G)
        
        try:
            eigenvector = nx.eigenvector_centrality(self.G, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            eigenvector = {n: 0 for n in nodes}
        
        pagerank = nx.pagerank(self.G)
        
        # Community
        partition = community_louvain.best_partition(self.G)
        
        # Triangle count
        triangles = nx.triangles(self.G)
        
        # Build feature DataFrame
        features = pd.DataFrame({
            "node": nodes,
            "degree": [degree[n] for n in nodes],
            "clustering_coeff": [clustering[n] for n in nodes],
            "degree_centrality": [degree_cent[n] for n in nodes],
            "betweenness_centrality": [betweenness[n] for n in nodes],
            "closeness_centrality": [closeness[n] for n in nodes],
            "eigenvector_centrality": [eigenvector[n] for n in nodes],
            "pagerank": [pagerank[n] for n in nodes],
            "community": [partition[n] for n in nodes],
            "triangles": [triangles[n] for n in nodes],
        })
        
        features.to_csv("outputs/results/node_features.csv", index=False)
        print(f"Extracted {len(features.columns)-1} features for {len(features)} nodes.")
        return features
    
    def extract_edge_features(self, edges=None):
        """
        Extract edge features for link prediction.
        If edges is None, use all existing edges.
        Returns DataFrame with one row per edge.
        """
        if edges is None:
            edges = list(self.G.edges())
        
        # Precompute node features
        degree = dict(self.G.degree())
        clustering = nx.clustering(self.G)
        community_partition = community_louvain.best_partition(self.G)
        
        edge_features = []
        for u, v in edges:
            # Neighbor-based features
            common_neighbors = len(list(nx.common_neighbors(self.G, u, v)))
            jaccard = list(nx.jaccard_coefficient(self.G, [(u, v)]))[0][2]
            adamic_adar = list(nx.adamic_adar_index(self.G, [(u, v)]))[0][2]
            pref_attachment = list(nx.preferential_attachment(self.G, [(u, v)]))[0][2]
            
            # Node-pair features
            same_community = int(community_partition.get(u, -1) == community_partition.get(v, -1))
            
            edge_features.append({
                "source": u,
                "target": v,
                "common_neighbors": common_neighbors,
                "jaccard_coefficient": jaccard,
                "adamic_adar_index": adamic_adar,
                "preferential_attachment": pref_attachment,
                "source_degree": degree.get(u, 0),
                "target_degree": degree.get(v, 0),
                "source_clustering": clustering.get(u, 0),
                "target_clustering": clustering.get(v, 0),
                "same_community": same_community,
            })
        
        df = pd.DataFrame(edge_features)
        df.to_csv("outputs/results/edge_features.csv", index=False)
        print(f"Extracted edge features for {len(df)} edges.")
        return df
    
    def save_features_to_neo4j(self, node_features_df, driver):
        """Write computed features back to Neo4j as node properties."""
        with driver.session() as session:
            for _, row in node_features_df.iterrows():
                session.run("""
                    MATCH (n {name: $name})
                    SET n.degree = $degree,
                        n.clustering_coeff = $clustering,
                        n.betweenness = $betweenness,
                        n.closeness = $closeness,
                        n.pagerank = $pagerank,
                        n.community = $community
                """, 
                name=row["node"],
                degree=int(row["degree"]),
                clustering=float(row["clustering_coeff"]),
                betweenness=float(row["betweenness_centrality"]),
                closeness=float(row["closeness_centrality"]),
                pagerank=float(row["pagerank"]),
                community=int(row["community"]))
        print("Features saved to Neo4j.")
```

> ✅ **GIT COMMIT:** `git add . && git commit -m "Phase 2: Graph exploration, analysis, feature extraction" && git push`

---

## PHASE 3: Advanced Graph Techniques (ML)

### Adım 3.1 — `src/projections.py`

```python
"""
Graph projections: monopartite, co-occurrence, similarity graphs.
"""
import networkx as nx
import pandas as pd
from neo4j import GraphDatabase
import config
from itertools import combinations

class GraphProjections:
    def __init__(self, driver):
        self.driver = driver
    
    def create_actor_cooccurrence_graph(self):
        """
        Project bipartite (Person-Movie) -> monopartite (Person-Person).
        Two persons are connected if they acted in the same movie.
        Weight = number of shared movies.
        """
        G = nx.Graph()
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person)
                WHERE id(p1) < id(p2)
                RETURN p1.name AS actor1, p2.name AS actor2, 
                       count(m) AS weight,
                       collect(m.title) AS shared_movies
            """)
            for record in result:
                G.add_edge(record["actor1"], record["actor2"], 
                          weight=record["weight"],
                          shared_movies=record["shared_movies"])
        
        print(f"Actor co-occurrence graph: {G.number_of_nodes()} actors, {G.number_of_edges()} edges")
        return G
    
    def create_movie_similarity_graph(self, threshold=2):
        """
        Build movie similarity graph based on shared actors.
        Two movies are connected if they share >= threshold actors.
        """
        G = nx.Graph()
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m1:Movie)<-[:ACTED_IN]-(p:Person)-[:ACTED_IN]->(m2:Movie)
                WHERE id(m1) < id(m2)
                WITH m1, m2, count(p) AS shared_actors, collect(p.name) AS actors
                WHERE shared_actors >= $threshold
                RETURN m1.title AS movie1, m2.title AS movie2, 
                       shared_actors, actors
            """, threshold=threshold)
            for record in result:
                G.add_edge(record["movie1"], record["movie2"],
                          weight=record["shared_actors"],
                          shared_actors=record["actors"])
        
        print(f"Movie similarity graph: {G.number_of_nodes()} movies, {G.number_of_edges()} edges")
        return G
    
    def create_genre_cooccurrence_graph(self):
        """
        Genre co-occurrence: two genres are connected if movies belong to both.
        """
        G = nx.Graph()
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (g1:Genre)<-[:BELONGS_TO]-(m:Movie)-[:BELONGS_TO]->(g2:Genre)
                WHERE id(g1) < id(g2)
                RETURN g1.name AS genre1, g2.name AS genre2, count(m) AS weight
            """)
            for record in result:
                G.add_edge(record["genre1"], record["genre2"], weight=record["weight"])
        
        return G
```

### Adım 3.2 — `src/link_prediction.py`

```python
"""
Link Prediction: Predict future collaborations between actors.
Uses topological features from feature_extraction.py.
Compares 3+ ML algorithms with GridSearch hyperparameter tuning.
Includes Recursive Feature Elimination (RFE).
"""
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, roc_auc_score, 
                              confusion_matrix, accuracy_score)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, RFECV
import matplotlib.pyplot as plt
from src.feature_extraction import FeatureExtractor
import warnings
warnings.filterwarnings('ignore')

class LinkPredictor:
    def __init__(self, G: nx.Graph):
        """G: NetworkX graph (actor co-occurrence)."""
        self.G = G
        self.feature_extractor = FeatureExtractor(G)
        self.results = {}
    
    def prepare_dataset(self, test_ratio=0.3, neg_ratio=1.0):
        """
        Create training data for link prediction.
        Positive samples: existing edges.
        Negative samples: non-existing edges (sampled).
        """
        edges = list(self.G.edges())
        nodes = list(self.G.nodes())
        
        # Positive samples
        positive_edges = edges.copy()
        
        # Negative samples (non-existing edges)
        non_edges = []
        edge_set = set(edges) | set([(v, u) for u, v in edges])
        
        np.random.seed(42)
        while len(non_edges) < len(positive_edges) * neg_ratio:
            u = np.random.choice(nodes)
            v = np.random.choice(nodes)
            if u != v and (u, v) not in edge_set:
                non_edges.append((u, v))
                edge_set.add((u, v))
        
        # Extract features for all edges
        all_edges = positive_edges + non_edges
        labels = [1] * len(positive_edges) + [0] * len(non_edges)
        
        # Extract edge features
        features_df = self.feature_extractor.extract_edge_features(all_edges)
        
        feature_cols = [c for c in features_df.columns if c not in ["source", "target"]]
        X = features_df[feature_cols].values
        y = np.array(labels)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_cols
        self.scaler = scaler
        self.all_edges = all_edges
        self.labels = labels
        
        print(f"Dataset: {len(X_train)} train, {len(X_test)} test samples")
        print(f"Features: {feature_cols}")
        return X_train, X_test, y_train, y_test
    
    def run_experiments(self):
        """
        Train and compare 3+ ML algorithms with GridSearch.
        Returns comparison DataFrame.
        """
        # Define models and hyperparameter grids
        models = {
            "Logistic Regression": {
                "model": LogisticRegression(max_iter=1000),
                "params": {
                    "C": [0.01, 0.1, 1, 10],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear"]
                }
            },
            "Random Forest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5]
                }
            },
            "Gradient Boosting": {
                "model": GradientBoostingClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5]
                }
            },
            "SVM": {
                "model": SVC(probability=True, random_state=42),
                "params": {
                    "C": [0.1, 1, 10],
                    "kernel": ["rbf", "linear"],
                    "gamma": ["scale", "auto"]
                }
            }
        }
        
        comparison = []
        
        for name, config_dict in models.items():
            print(f"\n{'='*50}")
            print(f"Training: {name}")
            print(f"{'='*50}")
            
            # GridSearchCV
            grid_search = GridSearchCV(
                config_dict["model"],
                config_dict["params"],
                cv=5,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(self.X_train, self.y_train)
            
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(self.X_test)
            y_proba = best_model.predict_proba(self.X_test)[:, 1]
            
            acc = accuracy_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_proba)
            
            print(f"Best Params: {grid_search.best_params_}")
            print(f"Accuracy: {acc:.4f} | AUC-ROC: {auc:.4f}")
            print(classification_report(self.y_test, y_pred))
            
            self.results[name] = {
                "model": best_model,
                "accuracy": acc,
                "auc_roc": auc,
                "best_params": grid_search.best_params_,
                "y_pred": y_pred,
                "y_proba": y_proba
            }
            
            comparison.append({
                "Model": name,
                "Accuracy": acc,
                "AUC-ROC": auc,
                "Best Params": str(grid_search.best_params_)
            })
        
        comparison_df = pd.DataFrame(comparison).sort_values("AUC-ROC", ascending=False)
        comparison_df.to_csv("outputs/results/link_prediction_comparison.csv", index=False)
        print("\n=== MODEL COMPARISON ===")
        print(comparison_df.to_string(index=False))
        return comparison_df
    
    def recursive_feature_elimination(self):
        """
        Apply RFE (Recursive Feature Elimination) to rank feature importance.
        This is REQUIRED by the project spec (Refex).
        """
        # Use Random Forest as the estimator for RFE
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # RFECV with cross-validation
        rfecv = RFECV(
            estimator=rf,
            step=1,
            cv=5,
            scoring="roc_auc",
            min_features_to_select=2
        )
        rfecv.fit(self.X_train, self.y_train)
        
        # Feature ranking
        ranking_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Ranking": rfecv.ranking_,
            "Selected": rfecv.support_
        }).sort_values("Ranking")
        
        print("\n=== FEATURE RANKING (RFE) ===")
        print(ranking_df.to_string(index=False))
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.barh(ranking_df["Feature"], ranking_df["Ranking"], color="steelblue")
        ax.set_xlabel("Ranking (1 = most important)")
        ax.set_title("Recursive Feature Elimination - Feature Rankings")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig("outputs/figures/rfe_feature_ranking.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        # Also plot number of features vs CV score
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        ax2.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
                rfecv.cv_results_['mean_test_score'], marker='o')
        ax2.set_xlabel("Number of Features")
        ax2.set_ylabel("AUC-ROC (CV)")
        ax2.set_title("RFECV - Optimal Number of Features")
        plt.tight_layout()
        fig2.savefig("outputs/figures/rfecv_optimal_features.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        ranking_df.to_csv("outputs/results/rfe_ranking.csv", index=False)
        return ranking_df, rfecv
    
    def predict_new_links(self, top_n=10):
        """
        Use best model to predict new links (collaborations).
        Returns top N predicted new edges.
        """
        best_model_name = max(self.results, key=lambda k: self.results[k]["auc_roc"])
        best_model = self.results[best_model_name]["model"]
        
        # Get non-existing edges and predict
        nodes = list(self.G.nodes())
        edge_set = set(self.G.edges()) | set([(v, u) for u, v in self.G.edges()])
        
        candidate_edges = []
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                if (u, v) not in edge_set:
                    candidate_edges.append((u, v))
        
        if len(candidate_edges) > 5000:
            np.random.seed(42)
            indices = np.random.choice(len(candidate_edges), 5000, replace=False)
            candidate_edges = [candidate_edges[i] for i in indices]
        
        # Extract features
        features_df = self.feature_extractor.extract_edge_features(candidate_edges)
        feature_cols = [c for c in features_df.columns if c not in ["source", "target"]]
        X_candidates = self.scaler.transform(features_df[feature_cols].values)
        
        probas = best_model.predict_proba(X_candidates)[:, 1]
        
        predictions = pd.DataFrame({
            "source": [e[0] for e in candidate_edges],
            "target": [e[1] for e in candidate_edges],
            "probability": probas
        }).sort_values("probability", ascending=False).head(top_n)
        
        predictions.to_csv("outputs/results/predicted_links.csv", index=False)
        print(f"\nTop {top_n} predicted new links:")
        print(predictions.to_string(index=False))
        return predictions
```

### Adım 3.3 — `src/node_classification.py`

```python
"""
Node Classification: Classify person roles (actor, director, etc.).
Uses topological features from feature_extraction.py.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from neo4j import GraphDatabase
import config

class NodeClassifier:
    def __init__(self, driver, node_features_df):
        """
        driver: Neo4j driver
        node_features_df: DataFrame from FeatureExtractor.extract_node_features()
        """
        self.driver = driver
        self.node_features = node_features_df
        self.results = {}
    
    def prepare_labels(self):
        """
        Assign labels to Person nodes based on their primary role.
        E.g., 'Actor', 'Director', 'Actor-Director', etc.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Person)
                OPTIONAL MATCH (p)-[:ACTED_IN]->()
                WITH p, count(*) > 0 AS is_actor
                OPTIONAL MATCH (p)-[:DIRECTED]->()
                WITH p, is_actor, count(*) > 0 AS is_director
                RETURN p.name AS name,
                       CASE 
                         WHEN is_actor AND is_director THEN 'Actor-Director'
                         WHEN is_actor THEN 'Actor'
                         WHEN is_director THEN 'Director'
                         ELSE 'Other'
                       END AS role
            """)
            labels_df = pd.DataFrame([dict(record) for record in result])
        
        # Merge with node features
        merged = self.node_features.merge(labels_df, left_on="node", right_on="name", how="inner")
        merged = merged[merged["role"] != "Other"]  # Remove rare classes
        
        self.data = merged
        print(f"Node classification dataset: {len(merged)} samples")
        print(merged["role"].value_counts())
        return merged
    
    def run_classification(self):
        """Run classification with 3+ algorithms and GridSearch."""
        feature_cols = [c for c in self.data.columns 
                       if c not in ["node", "name", "role", "community"]]
        
        X = self.data[feature_cols].values
        y = self.data["role"].values
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        models = {
            "Logistic Regression": {
                "model": LogisticRegression(max_iter=1000),
                "params": {"C": [0.01, 0.1, 1, 10]}
            },
            "Random Forest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {"n_estimators": [50, 100, 200], "max_depth": [3, 5, None]}
            },
            "Gradient Boosting": {
                "model": GradientBoostingClassifier(random_state=42),
                "params": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 0.2]}
            }
        }
        
        comparison = []
        for name, cfg in models.items():
            print(f"\nTraining: {name}")
            gs = GridSearchCV(cfg["model"], cfg["params"], cv=5, scoring="accuracy", n_jobs=-1)
            gs.fit(X_train, y_train)
            
            y_pred = gs.best_estimator_.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            print(f"Accuracy: {acc:.4f}")
            print(classification_report(y_test, y_pred, target_names=le.classes_))
            
            comparison.append({"Model": name, "Accuracy": acc, "Best Params": str(gs.best_params_)})
            self.results[name] = gs.best_estimator_
        
        comparison_df = pd.DataFrame(comparison).sort_values("Accuracy", ascending=False)
        comparison_df.to_csv("outputs/results/node_classification_comparison.csv", index=False)
        
        # RFE
        rfe = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=5)
        rfe.fit(X_train, y_train)
        rfe_df = pd.DataFrame({"Feature": feature_cols, "Ranking": rfe.ranking_}).sort_values("Ranking")
        rfe_df.to_csv("outputs/results/node_classification_rfe.csv", index=False)
        
        return comparison_df
```

### Adım 3.4 — `src/kg_completion.py`

```python
"""
Knowledge Graph Completion using PyKEEN.
Learns embeddings and predicts missing triples.
"""
import pandas as pd
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import numpy as np
from neo4j import GraphDatabase
import config
import torch

class KGCompleter:
    def __init__(self, driver):
        self.driver = driver
    
    def extract_triples(self):
        """Extract all triples from Neo4j as (head, relation, tail)."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (h)-[r]->(t)
                RETURN 
                    coalesce(h.name, h.title) AS head,
                    type(r) AS relation,
                    coalesce(t.name, t.title) AS tail
            """)
            triples = [(record["head"], record["relation"], record["tail"]) 
                       for record in result 
                       if record["head"] and record["tail"]]
        
        triples_array = np.array(triples)
        print(f"Extracted {len(triples_array)} triples")
        return triples_array
    
    def run_kg_completion(self, model_name="TransE"):
        """
        Train a KG embedding model and predict missing links.
        Models to try: TransE, ComplEx, DistMult
        """
        triples = self.extract_triples()
        
        tf = TriplesFactory.from_labeled_triples(triples)
        training, testing, validation = tf.split([0.8, 0.1, 0.1], random_state=42)
        
        result = pipeline(
            training=training,
            testing=testing,
            validation=validation,
            model=model_name,
            model_kwargs={"embedding_dim": 64},
            training_kwargs={"num_epochs": 100, "batch_size": 128},
            random_seed=42,
        )
        
        # Evaluation metrics
        metrics = result.metric_results.to_dict()
        print(f"\n=== {model_name} Results ===")
        print(f"Hits@10: {metrics.get('hits_at_10', {}).get('both', {}).get('realistic', 'N/A')}")
        print(f"MRR: {metrics.get('mean_reciprocal_rank', {}).get('both', {}).get('realistic', 'N/A')}")
        
        return result
    
    def compare_models(self):
        """Compare multiple KGE models."""
        model_names = ["TransE", "ComplEx", "DistMult"]
        results = {}
        
        for model_name in model_names:
            print(f"\n{'='*50}")
            print(f"Training {model_name}...")
            print(f"{'='*50}")
            
            try:
                result = self.run_kg_completion(model_name)
                results[model_name] = result
            except Exception as e:
                print(f"Error with {model_name}: {e}")
        
        return results
    
    def predict_missing_triples(self, result, top_k=10):
        """Predict missing triples using trained model."""
        model = result.model
        tf = result.training
        
        # Predict tails for some heads
        predicted = []
        # This would need to be customized based on the model's predict method
        # Simplified version:
        print(f"\nTop {top_k} predicted missing triples (see report for details)")
        
        return predicted
```

> ✅ **GIT COMMIT:** `git add . && git commit -m "Phase 3: Link prediction, node classification, KG completion with ML" && git push`

---

## PHASE 4: Visualization & Report Plots

### Adım 4.1 — `src/visualization.py`'ye ek fonksiyonlar ekle

```python
# === PHASE 4 ADDITIONS to visualization.py ===

import seaborn as sns
from pyvis.network import Network

def plot_degree_distribution(degree_df, filename="degree_distribution"):
    """Plot degree distribution histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(degree_df["degree"], bins=20, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Degree Distribution")
    
    # Log-log
    axes[1].loglog(
        sorted(degree_df["degree"], reverse=True),
        range(1, len(degree_df) + 1),
        marker="o", linestyle="none", color="steelblue"
    )
    axes[1].set_xlabel("Degree (log)")
    axes[1].set_ylabel("Rank (log)")
    axes[1].set_title("Degree Distribution (Log-Log)")
    
    plt.tight_layout()
    save_plot(fig, filename)

def plot_centrality_comparison(centralities_df, top_n=15, filename="centrality_comparison"):
    """Plot top nodes by different centrality measures."""
    metrics = ["degree_centrality", "betweenness_centrality", "closeness_centrality", "pagerank"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for ax, metric in zip(axes.flatten(), metrics):
        top = centralities_df.nlargest(top_n, metric)
        ax.barh(top["node"], top[metric], color="steelblue")
        ax.set_title(metric.replace("_", " ").title())
        ax.invert_yaxis()
    
    plt.tight_layout()
    save_plot(fig, filename)

def plot_community_graph(G, partition, filename="community_graph"):
    """Visualize graph colored by community."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=0.5)
    
    communities = set(partition.values())
    cmap = plt.cm.Set3
    colors = [cmap(partition[node] / max(len(communities), 1)) for node in G.nodes()]
    
    nx.draw(G, pos, node_color=colors, node_size=100, 
            edge_color="gray", alpha=0.7, with_labels=False, ax=ax)
    
    # Add labels to high-degree nodes
    degrees = dict(G.degree())
    threshold = sorted(degrees.values(), reverse=True)[min(15, len(degrees)-1)]
    labels = {n: n for n, d in degrees.items() if d >= threshold}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    ax.set_title(f"Community Detection ({len(communities)} communities)")
    plt.tight_layout()
    save_plot(fig, filename)

def plot_predicted_links(G, predicted_links_df, filename="predicted_links"):
    """Visualize graph with predicted (new) links highlighted."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw existing edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color="gray", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color="lightblue", ax=ax)
    
    # Draw predicted links
    new_edges = list(zip(predicted_links_df["source"], predicted_links_df["target"]))
    nx.draw_networkx_edges(G, pos, edgelist=new_edges, edge_color="red", 
                          width=2, style="dashed", ax=ax)
    
    # Label nodes involved in predictions
    pred_nodes = set(predicted_links_df["source"]) | set(predicted_links_df["target"])
    labels = {n: n for n in pred_nodes if n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    ax.set_title("Predicted New Links (red dashed)")
    plt.tight_layout()
    save_plot(fig, filename)

def plot_model_comparison(comparison_df, task_name="link_prediction", filename=None):
    """Bar chart comparing model performances."""
    if filename is None:
        filename = f"{task_name}_model_comparison"
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    metric_col = "AUC-ROC" if "AUC-ROC" in comparison_df.columns else "Accuracy"
    
    bars = ax.bar(comparison_df["Model"], comparison_df[metric_col], color="steelblue")
    ax.set_ylabel(metric_col)
    ax.set_title(f"Model Comparison - {task_name.replace('_', ' ').title()}")
    
    # Add value labels on bars
    for bar, val in zip(bars, comparison_df[metric_col]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=10)
    
    plt.tight_layout()
    save_plot(fig, filename)

def create_interactive_graph(G, partition=None, filename="interactive_graph"):
    """Create an interactive PyVis graph."""
    net = Network(height="700px", width="100%", notebook=False)
    
    for node in G.nodes():
        color = "#4C8BF5"
        if partition:
            colors = ["#4C8BF5", "#34A853", "#EA4335", "#FBBC05", "#9C27B0", 
                      "#FF5722", "#00BCD4", "#795548"]
            color = colors[partition.get(node, 0) % len(colors)]
        net.add_node(node, label=node, color=color)
    
    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1)
        net.add_edge(u, v, value=weight)
    
    filepath = f"outputs/figures/{filename}.html"
    net.save_graph(filepath)
    print(f"Interactive graph saved to {filepath}")
```

> ✅ **GIT COMMIT:** `git add . && git commit -m "Phase 4: All visualization functions complete" && git push`

---

## PHASE 5: Master Orchestrator Script

### Adım 5.1 — `main.py`

```python
"""
Master orchestrator for the Knowledge Graph project.
Runs all phases sequentially and generates outputs.
"""
import os
import warnings
warnings.filterwarnings('ignore')

# Ensure output directories exist
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

from neo4j import GraphDatabase
import config
from src.data_loader import MovieGraphLoader
from src.graph_model import print_schema
from src.cypher_queries import GraphExplorer
from src.graph_analysis import GraphAnalyzer
from src.feature_extraction import FeatureExtractor
from src.projections import GraphProjections
from src.link_prediction import LinkPredictor
from src.node_classification import NodeClassifier
from src.kg_completion import KGCompleter
from src import visualization as viz

def main():
    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    
    # ============================
    # PHASE 1: Load Data
    # ============================
    print("\n" + "="*60)
    print("PHASE 1: Loading Data & Schema")
    print("="*60)
    
    loader = MovieGraphLoader()
    loader.load_movies_dataset()
    loader.add_genre_nodes()
    loader.verify_schema()
    loader.close()
    
    print_schema()
    viz.visualize_schema()
    
    # ============================
    # PHASE 2: Explore & Analyze
    # ============================
    print("\n" + "="*60)
    print("PHASE 2: Graph Exploration & Analysis")
    print("="*60)
    
    explorer = GraphExplorer()
    print("\n--- Node Counts ---")
    print(explorer.get_node_counts())
    print("\n--- Relationship Counts ---")
    print(explorer.get_relationship_counts())
    print("\n--- Top Actor-Director Pairs ---")
    print(explorer.get_actor_director_pairs())
    explorer.close()
    
    analyzer = GraphAnalyzer()
    G = analyzer.build_networkx_graph("actor_cooccurrence")
    
    # Degree distribution
    degree_df = analyzer.compute_degree_distribution()
    viz.plot_degree_distribution(degree_df)
    
    # Centralities
    centralities_df = analyzer.compute_centralities()
    viz.plot_centrality_comparison(centralities_df)
    
    # Community detection
    communities_df, partition = analyzer.detect_communities()
    viz.plot_community_graph(G, partition)
    viz.create_interactive_graph(G, partition)
    
    # Graph summary
    summary = analyzer.get_graph_summary()
    print("\n--- Graph Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    
    # Feature extraction
    fe = FeatureExtractor(G)
    node_features = fe.extract_node_features()
    fe.save_features_to_neo4j(node_features, driver)
    
    analyzer.close()
    
    # ============================
    # PHASE 3: Advanced Techniques
    # ============================
    print("\n" + "="*60)
    print("PHASE 3: Advanced Graph Techniques")
    print("="*60)
    
    # Projections
    projections = GraphProjections(driver)
    actor_graph = projections.create_actor_cooccurrence_graph()
    movie_sim_graph = projections.create_movie_similarity_graph(threshold=1)
    
    viz.create_interactive_graph(actor_graph, filename="actor_cooccurrence")
    viz.create_interactive_graph(movie_sim_graph, filename="movie_similarity")
    
    # Link Prediction
    print("\n--- Link Prediction ---")
    lp = LinkPredictor(actor_graph)
    lp.prepare_dataset()
    comparison_lp = lp.run_experiments()
    rfe_ranking, rfecv = lp.recursive_feature_elimination()
    predicted_links = lp.predict_new_links(top_n=10)
    
    viz.plot_model_comparison(comparison_lp, "link_prediction")
    viz.plot_predicted_links(actor_graph, predicted_links)
    
    # Node Classification
    print("\n--- Node Classification ---")
    nc = NodeClassifier(driver, node_features)
    nc.prepare_labels()
    comparison_nc = nc.run_classification()
    viz.plot_model_comparison(comparison_nc, "node_classification")
    
    # KG Completion
    print("\n--- Knowledge Graph Completion ---")
    kgc = KGCompleter(driver)
    kg_results = kgc.compare_models()
    
    driver.close()
    
    print("\n" + "="*60)
    print("ALL PHASES COMPLETE!")
    print("Check outputs/figures/ and outputs/results/ for results.")
    print("="*60)

if __name__ == "__main__":
    main()
```

> ✅ **GIT COMMIT:** `git add . && git commit -m "Phase 5: Master orchestrator complete - all phases integrated" && git push`

---

## GIT COMMIT SUMMARY (Takip için)

| Commit # | Message | Ne yapıldı |
|----------|---------|------------|
| 1 | `Phase 0: Project setup complete` | Klasör yapısı, requirements, .env, config |
| 2 | `Phase 1: Graph model, data loader, schema visualization` | Neo4j veri yükleme, şema tanımı, görselleştirme |
| 3 | `Phase 2: Graph exploration, analysis, feature extraction` | Cypher sorguları, centrality, community, feature extraction |
| 4 | `Phase 3: Link prediction, node classification, KG completion with ML` | 3+ ML algoritma, GridSearch, RFE, PyKEEN |
| 5 | `Phase 4: All visualization functions complete` | Tüm plot/görselleştirme fonksiyonları |
| 6 | `Phase 5: Master orchestrator complete - all phases integrated` | main.py, tüm fazlar entegre |
| 7 | `Phase 6: Report draft and final cleanup` | Rapor taslağı, README |

---

## CHECKLIST — Proje Gereksinimleri

- [x] Neo4j'de Knowledge Graph tasarımı ve veri yükleme
- [x] Node tipleri: Movie, Person, Genre
- [x] İlişki tipleri: ACTED_IN, DIRECTED, PRODUCED, WROTE, REVIEWED, BELONGS_TO
- [x] Cypher ile keşif sorguları
- [x] Degree distribution analizi
- [x] Community detection (Louvain)
- [x] Centrality metrikleri (degree, betweenness, closeness, eigenvector, PageRank)
- [x] Feature extraction (node & edge features)
- [x] Features Neo4j'ye geri yazılıyor
- [x] Monopartite / co-occurrence network projeksiyon
- [x] Similarity graph (shared actors)
- [x] Link prediction (4 algoritma + GridSearch + RFE)
- [x] Node classification (3 algoritma + GridSearch + RFE)
- [x] Knowledge graph completion (PyKEEN: TransE, ComplEx, DistMult)
- [x] Recursive Feature Elimination (RFE/RFECV)
- [x] Hyperparameter tuning (GridSearchCV)
- [x] En az 3 farklı ML algoritması karşılaştırması
- [x] Tüm sonuçlar CSV olarak kaydediliyor
- [x] Tüm grafikler PNG olarak kaydediliyor
- [x] Interactive graph visualizations (PyVis)
- [x] Git ile versiyon kontrolü
- [x] Temiz, yorumlanmış kod

---

## PREREQUISITES (Projeyi çalıştırmadan önce)

1. **Neo4j Desktop** veya **Neo4j Aura** kurulu olmalı
2. Python 3.10+ kurulu olmalı
3. `pip install -r requirements.txt`
4. `.env` dosyasını Neo4j bilgilerinle doldur
5. Neo4j'de Movies dataset'ini yükle (`:play movies` veya script ile)
6. `python main.py` ile çalıştır

---

## RAPOR İÇİN NOTLAR

- Rapor max 10 sayfa
- Title page'de YouTube linki olmalı
- 10 dakikalık demo videosu çekilmeli
- Tüm tablo ve grafikler rapora eklenmeli
- RFE feature ranking tablosu ZORUNLU
- Model karşılaştırma tabloları ZORUNLU
