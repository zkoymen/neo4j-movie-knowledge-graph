# Example Cypher Queries

These queries are written for the Neo4j `recommendations` dataset.

Graph model:
- Nodes: `Movie`, `Actor`, `Director`, `User`, `Genre`
- Relationships: `ACTED_IN`, `DIRECTED`, `RATED`, `IN_GENRE`

## 1. Count Nodes by Label

```cypher
MATCH (n)
UNWIND labels(n) AS label
RETURN label, count(*) AS count
ORDER BY count DESC, label
```

## 2. Count Relationships by Type

```cypher
MATCH ()-[r]->()
RETURN type(r) AS type, count(*) AS count
ORDER BY count DESC, type
```

## 3. Show Movies With Ratings

```cypher
MATCH (u:User)-[r:RATED]->(m:Movie)
RETURN m.title AS movie,
       count(r) AS rating_count,
       round(avg(r.rating) * 10) / 10.0 AS avg_rating
ORDER BY avg_rating DESC, rating_count DESC
LIMIT 20
```

## 4. Show Movies by Genre

```cypher
MATCH (m:Movie)-[:IN_GENRE]->(g:Genre)
RETURN g.name AS genre,
       count(m) AS movie_count,
       collect(m.title)[0..10] AS example_movies
ORDER BY movie_count DESC, genre
```

## 5. Show Actor Subgraph

Good for graph view.

```cypher
MATCH p=(a:Actor)-[:ACTED_IN]->(m:Movie)
RETURN p
LIMIT 50
```

## 6. Show Director Subgraph

```cypher
MATCH p=(d:Director)-[:DIRECTED]->(m:Movie)
RETURN p
LIMIT 50
```

## 7. Show User Rating Subgraph

```cypher
MATCH p=(u:User)-[:RATED]->(m:Movie)
RETURN p
LIMIT 50
```

## 8. Actor Movie Counts

```cypher
MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)
RETURN a.name AS actor,
       count(m) AS movie_count,
       collect(m.title)[0..10] AS movies
ORDER BY movie_count DESC, actor
LIMIT 25
```

## 9. Director Movie Counts

```cypher
MATCH (d:Director)-[:DIRECTED]->(m:Movie)
RETURN d.name AS director,
       count(m) AS movie_count,
       collect(m.title)[0..10] AS movies
ORDER BY movie_count DESC, director
LIMIT 25
```

## 10. Actor Collaborations

```cypher
MATCH (a1:Actor)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(a2:Actor)
WHERE a1.name < a2.name
RETURN a1.name AS actor_1,
       a2.name AS actor_2,
       count(m) AS shared_movies,
       collect(m.title)[0..10] AS movies
ORDER BY shared_movies DESC, actor_1, actor_2
LIMIT 25
```

## 11. Actor-Director Collaborations

```cypher
MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Director)
RETURN a.name AS actor,
       d.name AS director,
       count(m) AS collaborations,
       collect(m.title)[0..10] AS movies
ORDER BY collaborations DESC, actor, director
LIMIT 25
```

## 12. One Movie Neighborhood

Good for graph view.

```cypher
MATCH p=(m:Movie {title: "The Matrix"})-[*1..1]-(n)
RETURN p
LIMIT 50
```

## 13. Count Ratings for Matrix Movies

This one is from the Neo4j guide style.

```cypher
MATCH (m:Movie)<-[:RATED]-(u:User)
WHERE m.title CONTAINS "Matrix"
WITH m, count(*) AS reviews
RETURN m.title AS movie, reviews
ORDER BY reviews DESC
LIMIT 10
```

## 14. Similar Movies by Common Genres

```cypher
MATCH (m:Movie)-[:IN_GENRE]->(g:Genre)<-[:IN_GENRE]-(rec:Movie)
WHERE m.title = "Inception"
WITH rec, collect(g.name) AS genres, count(*) AS commonGenres
RETURN rec.title, genres, commonGenres
ORDER BY commonGenres DESC
LIMIT 10
```

## 15. Simple Collaborative Recommendation

```cypher
MATCH (m:Movie {title: "Crimson Tide"})<-[:RATED]-(u:User)-[:RATED]->(rec:Movie)
WITH rec, COUNT(*) AS usersWhoAlsoWatched
RETURN rec.title AS recommendation, usersWhoAlsoWatched
ORDER BY usersWhoAlsoWatched DESC
LIMIT 25
```
