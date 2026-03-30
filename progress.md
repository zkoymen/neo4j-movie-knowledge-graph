# Project Progress Log

Last updated: 2026-03-30
Project: COM4514 Special Topics II - Building and Analyzing a Knowledge Graph from Movie Data
Dataset in current use: Neo4j recommendations dump dataset
Current status: Phase 1 completed, Phase 2 completed, Phase 3 implemented and runtime-validated

## 1. Project Goal

This project builds a movie knowledge graph in Neo4j and analyzes it with Python.
The required tasks from the course are:
- design the graph model
- query and explore the graph with Cypher
- extract manual graph features
- build advanced graph projections
- run link prediction and node classification
- attempt knowledge graph completion
- compare machine learning models
- use feature selection / recursive feature elimination
- create visuals, tables, and report-ready outputs

## 2. Important Dataset Decision

### Initial wrong direction
At first, the project started with a small custom OMDb-based pipeline.
This was useful only for early scaffolding and testing.
It was not enough for the real assignment because:
- the graph was too small
- it did not match the instructor's intended dataset closely enough
- it was weak for machine learning tasks
- it depended on repeated API calls

### Correct direction
Later, the project was moved to the Neo4j example dataset from:
- `neo4j-graph-examples/recommendations`

This was the correct choice because it already matches the course idea very well:
- movie nodes
- actor nodes
- director nodes
- user ratings
- genres
- enough scale for graph analytics and machine learning

### Current dataset scale
Current Neo4j dataset size after validation:
- Actor: 15443
- Director: 4091
- Genre: 20
- Movie: 9125
- Person: 19047
- User: 671
- ACTED_IN: 35910
- DIRECTED: 10007
- IN_GENRE: 20340
- RATED: 100004

This is approximately:
- 28k nodes
- 166k relationships

## 3. Graph Model Used

### Node labels
- `Movie`
- `Actor`
- `Director`
- `User`
- `Genre`

### Relationship types
- `ACTED_IN`
- `DIRECTED`
- `RATED`
- `IN_GENRE`

### Main properties used
- `Movie.title`
- `Movie.year`
- `Movie.plot`
- `Actor.name`
- `Director.name`
- `User.name`
- `Genre.name`
- `RATED.rating`

### Note
The dataset also contains `Person` nodes in Neo4j. The main project logic currently focuses on the labels listed above because they match the course example more directly.

## 4. Milestones Completed

### Milestone A - Base Python project scaffold
Completed:
- project folder structure
- `main.py`
- config loader
- schema printing
- schema visualization
- output folders
- commented starter modules

Reason:
This created a stable base before working with the real dataset.

### Milestone B - Recommendations dataset validation
Completed:
- Neo4j connection test
- database selection from `.env`
- expected label validation
- expected relationship validation
- minimum size validation

Reason:
This prevents running analysis on the wrong database by mistake.

### Milestone C - Phase 2 exploration queries
Completed:
- node counts
- relationship counts
- top rated movies
- movies by genre
- actor movie counts
- actor collaborations
- director movie counts
- actor-director pairs

Outputs are saved as CSV files in `outputs/results`.

### Milestone D - Graph analysis on actor co-occurrence network
Completed:
- actor co-occurrence graph construction
- degree distribution
- centrality metrics
- community detection
- graph summary table

Interpretation:
This part turns the movie graph into an actor-to-actor network.
If two actors appear in the same movie, they become connected.
This is the main monopartite projection used for analysis.

### Milestone E - Manual feature extraction
Completed:
- degree
- degree centrality
- betweenness centrality
- closeness centrality
- pagerank
- community id
- movie count
- average movie rating
- director count
- genre diversity

These features are saved to:
- CSV file
- Neo4j Actor nodes

Reason:
These manual topological features are needed as machine learning inputs later.

### Milestone F - Advanced graph projections
Completed:
- actor co-occurrence projection
- movie similarity projection

Actor projection meaning:
- nodes are actors
- edges show shared movies

Movie similarity meaning:
- nodes are movies
- edges show shared genres, actors, or directors

Similarity score design used now:
- shared genres: weight 3
- shared actors: weight 2
- shared directors: weight 4

### Milestone G - Link prediction pipeline
Completed:
- training dataset creation
- test dataset creation
- negative edge sampling
- feature engineering for candidate links
- model comparison with GridSearchCV
- recursive feature elimination
- new predicted actor links

Current ML models used:
- Logistic Regression
- Random Forest
- Gradient Boosting

Current best result:
- best model: Gradient Boosting
- AUC-ROC: 0.8927

This is a good first result for the project.

## 5. Important Bottlenecks and How They Were Solved

### Bottleneck 1 - Full graph analysis was too slow
Problem:
The earlier version used exact graph metrics on a much larger graph.
This caused long execution time and timeouts.

Observation:
Basic exploration queries finished in seconds, but graph analysis did not finish in time.
This showed the bottleneck was not Neo4j connection or Cypher basics. The expensive part was the advanced graph computation.

Fix:
- switched from exact betweenness centrality to approximate betweenness centrality
- added configurable limits in `config.py`
- moved analysis to a more meaningful core actor subgraph instead of all actors

Current core actor settings:
- minimum movie count per actor: 3
- maximum actors in core graph: 1500
- approximate betweenness sample size: 200

Reason:
This keeps the analysis meaningful but makes it computationally realistic.

### Bottleneck 2 - Need to locate the true hot spot before changing code blindly
Problem:
The full pipeline was too heavy, but the exact failing part was not obvious at first.

Fix:
The code was tested in smaller isolated steps.
Examples:
- run only exploration
- run only graph analysis
- run only projections
- run only link prediction
- run only movie similarity

Reason:
This made it possible to identify which stage was expensive and which stage was already fine.
This was an important milestone because it changed the process from guessing to measuring.

### Bottleneck 3 - Movie similarity query caused memory pressure
Problem:
A naive all-pairs movie similarity approach was too expensive for the full movie set.

Fix:
- only compare movies that already share at least one trait
- use a stronger movie subset first
- limit movie similarity projection to a top subset by rating activity

Current movie similarity settings:
- subset size: 1200 movies
- minimum similarity score: 3

Reason:
The project needs a useful similarity graph, not an unnecessarily huge one that crashes memory.

### Bottleneck 4 - Link prediction repeated expensive computations
Problem:
The link prediction code recomputed the same node-level graph metrics multiple times.
It also used exact betweenness internally.

Fix:
- compute node-level helper features once per graph
- reuse them across positive and negative edge feature generation
- switch internal betweenness to approximate mode
- keep candidate actor set limited when predicting new links

Reason:
Without this fix, time would be wasted repeating the same expensive work.

### Bottleneck 5 - Testing with smaller samples first
Problem:
Running every heavy step on the full possible graph was risky and slow.

Fix:
Small but representative subsets were used first to verify correctness.
Examples:
- test movie similarity on 500 movies first
- then scale to 1200 movies
- use core actor graph for link prediction instead of the full actor space

Reason:
This made debugging faster and protected the machine from unnecessary overload.

## 6. Current Runtime Behavior

After optimization, the full `main.py` pipeline completed successfully.

Latest successful full run:
- total runtime: about 413.6 seconds
- about 6 minutes 54 seconds

This is acceptable for a project pipeline of this size.

## 7. Current Important Outputs

Important generated files in `outputs/results`:
- `node_counts.csv`
- `relationship_counts.csv`
- `top_rated_movies.csv`
- `movies_by_genre.csv`
- `actor_movie_counts.csv`
- `actor_collaborations.csv`
- `director_movie_counts.csv`
- `actor_director_pairs.csv`
- `degree_distribution.csv`
- `centralities.csv`
- `communities.csv`
- `graph_summary.csv`
- `actor_features.csv`
- `actor_cooccurrence_projection.csv`
- `movie_similarity_projection.csv`
- `link_prediction_train.csv`
- `link_prediction_test.csv`
- `link_prediction_comparison.csv`
- `link_prediction_rfe.csv`
- `predicted_actor_links.csv`

## 8. Current Quantitative Results

### Core actor graph
- nodes: 1495
- edges: 13889

### Movie similarity graph
- nodes: 1195
- edges: 390203

### Link prediction result
- best model: Gradient Boosting
- best AUC-ROC: 0.8927

### RFE-selected top features
Current strongest link prediction features include:
- Adamic-Adar
- Jaccard
- Preferential Attachment
- Resource Allocation
- Source Closeness
- Target Closeness

Interpretation:
The local network neighborhood and structural similarity are very important for predicting future actor collaborations.

## 9. What Is Still Missing

The project is not finished yet.
Main remaining tasks:
- more visualization for report-ready figures
- final comparison tables for the report
- final explanation text for methods and findings
- video demo planning

## 9A. New Milestone on 2026-03-24

### Milestone H - Node classification pipeline implemented
Completed in code:
- created `src/node_classification.py`
- built a supervised node classification task
- target label: dominant actor genre
- merged graph-derived labels with manual actor features
- added train/test split
- added three classifiers
- added GridSearchCV
- added RFE
- added prediction export for the test set

Current model family set:
- Logistic Regression
- Random Forest
- Gradient Boosting

Important note:
This step was later runtime-tested successfully on 2026-03-30.

### Milestone I - Knowledge graph completion pipeline implemented
Completed in code:
- created `src/kg_completion.py`
- exported a semantic triple set from Neo4j
- added PyKEEN `TransE` experiment
- added train / test / validation split
- added metrics export
- added predicted missing `IN_GENRE` links export

Important note:
This step was later runtime-tested successfully on 2026-03-30.

### Milestone J - Main orchestration updated
Completed:
- `main.py` now includes
  - exploration
  - graph analysis
  - feature extraction
  - projections
  - link prediction
  - node classification
  - knowledge graph completion

Reason:
The project now has one main pipeline file that is much closer to the full course deliverable.

## 9B. Runtime Validation on 2026-03-30

### Node classification validation
Node classification was run successfully on the live Neo4j dataset.

Task definition used:
- classify actors by dominant genre
- use manually extracted graph features as inputs

Final filtered class distribution used:
- Comedy: 91
- Drama: 58
- Documentary: 8

Reason for filtering:
- very small classes were removed because they were unstable for cross-validation
- this made the comparison more defensible

Result summary:
- dataset rows: 157
- best model: Random Forest
- best F1-Macro: 0.6993

Important selected features from RFE:
- avg_movie_rating
- betweenness_centrality
- closeness_centrality
- director_count
- pagerank
- purity

### Knowledge graph completion validation
Knowledge graph completion was also run successfully on the live Neo4j dataset.

Task definition used:
- export a semantic triple set
- train a PyKEEN TransE model
- predict possible missing `IN_GENRE` links

Run summary:
- triples used: 36772
- relation types: 3
- genre candidates restricted to real genre labels only
- top predictions saved to CSV

Important correction:
- early prediction output mixed non-genre entities into `IN_GENRE` suggestions
- this was fixed by restricting prediction targets to the true set of `Genre` entities

This was an important milestone because it made the KG completion output semantically correct.

## 9C. Broader Node Classification Rebuild on 2026-03-30

### Why this rebuild was needed
The first working node classification pipeline was too narrow for a strong final report.

Main issue:
- feature extraction only covered the old 1495-actor core subset
- this made the classification dataset shrink too much after merging labels and features

This was not mainly a timeout issue.
It was mostly a coverage issue.

### What changed
The node classification workflow was rebuilt on a broader actor scope.

New classification scope:
- minimum movies per actor: 3
- maximum actors for classification features: 3000
- actual actors covered in the new feature table: 2957

New graph summary for the classification scope:
- nodes: 2957
- edges: 21798
- connected components: 8

### New output files
New or refreshed files related to this broader classification run:
- `outputs/results/actor_features_classification.csv`
- `outputs/results/classification_degree_distribution.csv`
- `outputs/results/classification_centralities.csv`
- `outputs/results/classification_communities.csv`
- `outputs/results/classification_graph_summary.csv`
- `outputs/results/node_classification_dataset.csv`
- `outputs/results/node_classification_label_distribution.csv`
- `outputs/results/node_classification_train.csv`
- `outputs/results/node_classification_test.csv`
- `outputs/results/node_classification_comparison.csv`
- `outputs/results/node_classification_rfe.csv`
- `outputs/results/node_classification_predictions.csv`

### New label strategy
The task is still:
- predict dominant actor genre

But the label handling is now more defensible:
- stable genres are kept directly
- the rare genre tail is grouped into `Other`

Reason:
- tiny one-digit classes are very weak for cross-validation
- grouping them is more honest than pretending they are stable standalone classes

### Final class distribution after rebuild
- Comedy: 230
- Drama: 191
- Documentary: 16
- Other: 26

Total final rows:
- 463

### New model result
Best model after the broader rebuild:
- Random Forest
- Accuracy: 0.7097
- F1-Macro: 0.4500

Important note:
- F1-Macro is lower than the old narrow result
- but this new result is more realistic because the task is harder and the data coverage is broader

This makes it more defensible for the report.

## 9D. ReFeX-Style Feature Expansion on 2026-03-30

### Why this step was added
The course note says feature selection / Refex must be used alongside the chosen algorithms.

To be safe, the project now includes:
- manual topological features
- classic model-side feature ranking
- a separate ReFeX-style recursive neighborhood feature expansion

### Important implementation note
The `graphrole` package was tested first, because it is a known ReFeX-related library.
It did not install cleanly in the current environment because it depends on older `pandas` behavior.

Because of that, a simple in-project ReFeX-style implementation was added instead.

### What the ReFeX-style pipeline does
Starting from the broader actor feature table, it:
- keeps local structural features
- adds recursive neighborhood mean features
- adds recursive neighborhood sum features
- repeats this expansion for 2 iterations
- drops constant or near-duplicate columns

This produces a richer feature table for node classification.

### New ReFeX output files
- `outputs/results/refex_actor_features.csv`
- `outputs/results/refex_feature_metadata.csv`
- `outputs/results/refex_feature_summary.csv`
- `outputs/results/refex_node_classification_dataset.csv`
- `outputs/results/refex_node_classification_label_distribution.csv`
- `outputs/results/refex_node_classification_train.csv`
- `outputs/results/refex_node_classification_test.csv`
- `outputs/results/refex_node_classification_comparison.csv`
- `outputs/results/refex_node_classification_rfe.csv`
- `outputs/results/refex_node_classification_predictions.csv`

### ReFeX run summary
- rows: 463
- final feature columns used for classification: 30
- best model: Logistic Regression
- best F1-Macro: 0.4884

### Comparison with the broader manual-feature run
Broader manual-feature node classification:
- best F1-Macro: 0.4500

ReFeX-style feature expansion:
- best F1-Macro: 0.4884

Interpretation:
- the recursive neighborhood features helped
- the improvement is not huge, but it is real
- this is useful for the report because it shows that richer graph-context features improved prediction quality

## 9E. Task Redesign for Better Node Classification on 2026-03-30

### New bottleneck discovered
Even after broadening actor coverage and adding ReFeX-style features, the actor-based node classification task was still limited.

Main bottleneck:
- `actor dominant genre` is a noisy label
- many actors have mixed genre history
- minority actor classes remained weak

Conclusion:
- the task definition itself was holding performance back
- this was not only a model-tuning issue

### New decision
The node classification task was redesigned as:
- `single-genre movie classification`

Reason:
- a movie with exactly one genre has a cleaner label
- this removes much of the ambiguity in the actor-based task
- it is more suitable for a stronger supervised learning benchmark

### Important leakage control
The movie graph for classification was built without using genre edges as input features.

Used for movie-movie links:
- shared actors
- shared directors

Not used in the movie feature graph:
- `IN_GENRE` edges

Reason:
- using genre edges to predict genre would leak the label into the input

### Important data integrity fix
An additional bug was found during this redesign:
- movie nodes were initially keyed by title
- duplicate movie titles caused accidental merges

This was fixed by switching to a unique movie key based on:
- `movieId`
- or fallback IDs
- or `title + year` if needed

This was an important correction because it made the classification results more trustworthy.

### Final movie classification label distribution
Using single-genre movies with at least 50 examples per class:
- Drama: 1156
- Comedy: 804
- Documentary: 359
- Horror: 182
- Thriller: 74

Total final rows:
- 2575

### Movie classification result with manual features
Best model:
- Random Forest

Result:
- Accuracy: 0.6816
- F1-Macro: 0.4976

### Movie classification result with ReFeX-style features
Best model:
- Gradient Boosting

Result:
- Accuracy: 0.7204
- F1-Macro: 0.5727

### Why this result is better
Old actor-based ReFeX node classification:
- F1-Macro: 0.4884

New movie-based ReFeX node classification:
- F1-Macro: 0.5727

The improvement is meaningful because:
- the labels are cleaner
- the dataset is larger
- the task is more defensible
- the score is clearly above the majority-class baseline

### Baseline comparison
Movie task majority baseline:
- Accuracy: 0.4485
- F1-Macro: 0.1239

This means the final movie ReFeX model is much better than a trivial baseline.

## 10. Practical Notes for the Final Report

Useful story for the report:
1. We first built and validated the graph model.
2. We explored the dataset with Cypher and exported summary tables.
3. We projected the heterogeneous movie graph into analysis-friendly networks.
4. We extracted manual graph features.
5. We optimized the pipeline after identifying real computational bottlenecks.
6. We trained and compared machine learning models for link prediction.
7. We used RFE to identify the most informative manual graph features.

This story is strong because it shows both technical implementation and reasoning.

## 11. Good Technical Decisions Made So Far

- moved from a too-small custom dataset to the proper recommendations dump
- validated dataset structure before running analysis
- used configurable limits instead of hard-coded expensive full-graph runs
- tested heavy stages separately before merging them into the main pipeline
- switched exact expensive graph metrics to approximate versions when scale required it
- saved outputs to CSV so they can be reused in the report and later ML tasks
- kept comments simple and readable in the code

## 12. Suggested Next Steps

Recommended next implementation steps:
1. build node classification pipeline
2. define labels for node classification carefully
3. add knowledge graph completion workflow
4. create report-ready charts from the CSV outputs
5. write a short methods summary for each phase

## 13. Reminder About Why This File Exists

This file is meant to help with:
- writing the final report
- remembering important decisions
- tracking bottlenecks and fixes
- giving future LLM prompts accurate project context
- avoiding repeated confusion about what has already been done

