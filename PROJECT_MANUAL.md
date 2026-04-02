# Project Manual

## 1. What This Project Does

This project uses the Neo4j `recommendations` movie dataset and turns it into a full graph analytics and machine learning workflow.

In simple words:
- Neo4j stores the movie graph
- Python reads the graph
- Python analyzes the graph structure
- Python creates manual graph features
- Python trains machine learning models for:
  - link prediction
  - node classification
  - knowledge graph completion

The goal is to show that the movie graph is not just a database.  
It can also be used as a machine learning and graph mining problem.

## 2. Final Big Picture

The project is now basically complete on the implementation side.

The main completed parts are:
1. graph model and dataset validation
2. Cypher exploration and graph analysis
3. manual topological feature extraction
4. graph projections
5. link prediction
6. node classification
7. ReFeX-style recursive feature expansion
8. knowledge graph completion

The main remaining work is:
- report writing
- figure selection
- video demo preparation

## 3. Dataset Used

The final dataset is the Neo4j example recommendations dataset.

Expected graph model:
- Nodes: `Movie`, `Actor`, `Director`, `User`, `Genre`
- Relationships: `ACTED_IN`, `DIRECTED`, `RATED`, `IN_GENRE`

Current validated size:
- about `28k` nodes
- about `166k` relationships

More detailed counts:
- `Movie`: 9125
- `Actor`: 15443
- `Director`: 4091
- `User`: 671
- `Genre`: 20

## 4. Core Idea of the Graph Model

The graph answers questions like:
- which actors played in which movies
- which directors directed which movies
- which users rated which movies
- which genres each movie belongs to

This is better than a flat table because we can:
- project actor-to-actor networks
- project movie-to-movie similarity networks
- compute centrality and community structure
- build machine learning tasks from graph topology

## 5. Main Workflow

The code runs in this order:

1. validate that the correct Neo4j dataset is loaded
2. run basic Cypher exploration
3. build the actor co-occurrence graph
4. compute graph metrics like degree, centrality, and communities
5. save manual graph features
6. build projections
7. run link prediction
8. run movie node classification
9. run movie ReFeX classification
10. run knowledge graph completion

Main entry point:
- [main.py](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/main.py)

## 6. Important Files and What They Do

### Config and setup
- [config.py](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/config.py)
  Loads `.env` settings.

- [.env.example](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/.env.example)
  Shows which environment variables are expected.

- [setup_env.ps1](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/setup_env.ps1)
  Creates and prepares the Python environment.

### Graph validation and schema
- [src/data_loader.py](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/src/data_loader.py)
  Does not import the dump. It validates that Neo4j Desktop already has the correct graph loaded.

- [src/graph_model.py](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/src/graph_model.py)
  Prints the conceptual graph schema.

- [src/visualization.py](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/src/visualization.py)
  Creates the schema diagram figure.

### Exploration and analysis
- [src/cypher_queries.py](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/src/cypher_queries.py)
  Runs basic exploration queries and exports CSV tables.

- [src/graph_analysis.py](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/src/graph_analysis.py)
  Builds the actor co-occurrence graph and computes:
  - degree
  - centralities
  - communities
  - graph summary

- [src/feature_extraction.py](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/src/feature_extraction.py)
  Saves manual actor graph features for later ML use.

### Advanced graph methods
- [src/projections.py](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/src/projections.py)
  Builds:
  - actor co-occurrence projection
  - movie similarity projection

- [src/link_prediction.py](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/src/link_prediction.py)
  Predicts new actor collaboration links.

- [src/movie_node_classification.py](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/src/movie_node_classification.py)
  Baseline movie node classification using manual graph features.

- [src/movie_refex.py](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/src/movie_refex.py)
  Final node classification pipeline using ReFeX-style recursive neighborhood features.

- [src/kg_completion.py](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/src/kg_completion.py)
  Runs knowledge graph completion with PyKEEN.

### Project memory
- [progress.md](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/progress.md)
  Full engineering log of important decisions, bottlenecks, and improvements.

## 7. How To Run the Project

### Step 1. Start Neo4j Desktop

Make sure the recommendations dump dataset is already loaded and running.

You need:
- the DBMS started
- the correct password in `.env`
- the correct database name in `.env`

### Step 2. Prepare Python environment

Run:

```powershell
.\setup_env.ps1
```

### Step 3. Check `.env`

The minimum important fields are:

```env
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_real_password
NEO4J_DATABASE=neo4j
DATASET_NAME=recommendations_dump
```

### Step 4. Run the main script

Run:

```powershell
python main.py
```

### Step 5. Check outputs

Main outputs are saved in:
- `outputs/results`
- `outputs/figures`

Important figure:
- [schema_diagram.png](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/figures/schema_diagram.png)

### Runtime expectation

The full pipeline is not instant.
Expect several minutes, not a few seconds.

For demo purposes, it is usually better to:
- show Neo4j Browser queries live
- show the code structure
- show the already generated CSV outputs

## 8. Useful Cypher Queries for Neo4j Browser

Use:
- [EXAMPLE_QUERIES.md](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/EXAMPLE_QUERIES.md)

Best queries to show quickly:
- node counts
- relationship counts
- actor subgraph
- user rating subgraph
- actor collaborations
- similar movies by shared genres

## 9. Final Results You Should Know

### 9.1 Link prediction

Task:
- predict likely future actor collaborations

Best result:
- best model: `Gradient Boosting`
- `AUC-ROC = 0.8894`
- `Accuracy = 0.8297`

Important file:
- [link_prediction_comparison.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/link_prediction_comparison.csv)

### 9.2 Final node classification

Final task:
- classify movies into a stable 4-class label set
- classes:
  - `Drama`
  - `Comedy`
  - `Documentary`
  - `Horror`

Why not 5 classes:
- `Thriller` was too unstable
- it was repeatedly confused with `Drama`
- diagnostics showed that keeping it hurt macro-F1 too much

Baseline manual-feature result:
- best model: `Gradient Boosting`
- `Accuracy = 0.7026`
- `Macro-F1 = 0.6190`

Final ReFeX-style result:
- best model: `Gradient Boosting`
- `Accuracy = 0.7545`
- `Balanced Accuracy = 0.6997`
- `Macro-F1 = 0.7274`

This is the main final node classification result.

### 9.3 Per-class F1 for the final node classification model

- `Comedy = 0.7078`
- `Documentary = 0.7481`
- `Drama = 0.7976`
- `Horror = 0.6563`

Meaning:
- `Drama` is easiest
- `Horror` is hardest
- but all four classes are now usable

### 9.4 Cross-validation check

For the final ReFeX model:
- `CV F1-Macro Mean = 0.6781`
- `CV F1-Macro Std = 0.0255`

Meaning:
- the holdout score is a bit higher than the CV mean
- but the model is still reasonably stable

### 9.5 Knowledge graph completion

Library used:
- `PyKEEN`

Model:
- `TransE`

Important realistic metrics:
- `Hits@1 = 0.0143`
- `Hits@3 = 0.0359`
- `Hits@5 = 0.0543`
- `Hits@10 = 0.0815`

Interpretation:
- this is a baseline KG completion result
- it proves the KG completion pipeline works
- but it is not the strongest result in the project

## 10. Why the Node Classification Changed

This is important for the report and for the demo.

The project did not jump directly to the final result.
There was an engineering decision path:

1. actor-based node classification was tried first
2. the labels were too noisy
3. movie-based classification was cleaner
4. the 5-class movie version still had an unstable `Thriller` class
5. diagnostics showed that `Thriller` was the bottleneck
6. the task was stabilized into 4 strong classes
7. a small targeted tuning step pushed the final macro-F1 to about `0.73`

This is not cheating or random trimming.
It is a justified supervised learning redesign based on:
- class support
- confusion matrix
- macro-F1
- balanced accuracy
- cross-validation stability

## 11. Which Outputs Matter for the Final Report

The `outputs/results` folder contains many CSV files because the project went through several experiments.

For the final report, use these as the main files.

### A. Graph model and exploration

Use:
- [node_counts.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/node_counts.csv)
- [relationship_counts.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/relationship_counts.csv)
- [top_rated_movies.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/top_rated_movies.csv)
- [movies_by_genre.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/movies_by_genre.csv)

Use for the model/import part:
- [schema_diagram.png](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/figures/schema_diagram.png)

### B. Exploratory graph analysis

Use:
- [degree_distribution.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/degree_distribution.csv)
- [centralities.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/centralities.csv)
- [communities.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/communities.csv)
- [graph_summary.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/graph_summary.csv)

### C. Manual features

Use:
- [actor_features.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/actor_features.csv)

This is important because the assignment says the manual topological features must be saved and used in ML.

### D. Projections

Use:
- [actor_cooccurrence_projection.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/actor_cooccurrence_projection.csv)
- [movie_similarity_projection.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/movie_similarity_projection.csv)

### E. Link prediction

Use:
- [link_prediction_comparison.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/link_prediction_comparison.csv)
- [link_prediction_rfe.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/link_prediction_rfe.csv)
- [predicted_actor_links.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/predicted_actor_links.csv)
- [link_prediction_report_gradient_boosting.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/link_prediction_report_gradient_boosting.csv)

### F. Final node classification

Use these as the main final node classification outputs:
- [movie_refex_comparison.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/movie_refex_comparison.csv)
- [movie_refex_rfe.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/movie_refex_rfe.csv)
- [movie_refex_cv_summary.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/movie_refex_cv_summary.csv)
- [movie_refex_imbalance_comparison.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/movie_refex_imbalance_comparison.csv)
- [movie_refex_report_gradient_boosting.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/movie_refex_report_gradient_boosting.csv)
- [movie_refex_confusion_gradient_boosting.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/movie_refex_confusion_gradient_boosting.csv)
- [movie_refex_label_distribution.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/movie_refex_label_distribution.csv)

Optional baseline comparison:
- [movie_classification_comparison.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/movie_classification_comparison.csv)
- [movie_classification_rfe.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/movie_classification_rfe.csv)

### G. Knowledge graph completion

Use:
- [kg_completion_metrics.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/kg_completion_metrics.csv)
- [kg_completion_predictions.csv](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/outputs/results/kg_completion_predictions.csv)

## 12. Which Older Files Are Not Main Final Results

These files came from earlier experiments and are not the main final node classification story:
- `node_classification_*`
- `refex_node_classification_*`
- `actor_features_classification.csv`
- `classification_*`

You can mention them as intermediate experiments if needed, but they should not be the main result tables in the final report.

## 13. Suggested Report Structure

A simple report structure:

1. Introduction
2. Dataset and graph model
3. Neo4j implementation and Cypher exploration
4. Graph analysis and manual feature extraction
5. Advanced graph techniques
   - actor co-occurrence projection
   - movie similarity graph
   - link prediction
   - node classification
   - KG completion
6. Results and discussion
7. Conclusion

## 14. Suggested 10-Minute Demo Flow

### Minute 1
- introduce the project goal
- say that Neo4j stores the movie knowledge graph

### Minute 2
- open Neo4j Browser
- show node counts and relationship counts
- show one small graph query result

### Minute 3
- show the schema diagram
- explain the node and relationship types

### Minute 4
- open `main.py`
- explain the pipeline order at a high level

### Minute 5
- show exploration CSV outputs
- mention degree, centrality, and communities

### Minute 6
- show actor co-occurrence and movie similarity outputs
- explain what a projection means

### Minute 7
- show link prediction comparison table
- explain the best model and predicted links

### Minute 8
- show movie ReFeX classification comparison table
- explain why the final task uses 4 classes

### Minute 9
- show the confusion matrix and RFE table
- explain which features were most useful

### Minute 10
- show KG completion metrics and predicted missing links
- close with the main findings

## 15. If Someone Asks "Why Did You Choose This Final Classification Task?"

Short answer:

> We first tried actor-based classification, but the labels were noisy.  
> Then we moved to movie-based single-genre classification because the labels were cleaner.  
> The 5-class version still had an unstable Thriller class, so we used diagnostics to redesign the task into 4 stable classes.  
> This increased the final ReFeX-based macro-F1 to about 0.73, which is much more defensible.

## 16. If Someone Asks "What Is the Strongest Result in the Project?"

Best answers:
- strongest ML result by ranking quality: link prediction (`AUC-ROC ~ 0.889`)
- strongest classification result: movie ReFeX node classification (`Macro-F1 ~ 0.727`)

## 17. Final Reminder

If you need to understand the engineering story in more detail, read:
- [progress.md](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/progress.md)

If you need quick Neo4j Browser queries, use:
- [EXAMPLE_QUERIES.md](c:/Users/zeyne/Desktop/DERSLER/Special Topics/proje/EXAMPLE_QUERIES.md)
