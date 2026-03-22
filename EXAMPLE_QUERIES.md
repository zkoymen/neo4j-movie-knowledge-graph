# Example Cypher Queries

These queries are written for the current graph model:
- Nodes: `Movie`, `Actor`, `Director`, `User`, `Genre`, `Country`
- Relationships: `ACTED_IN`, `DIRECTED`, `RATED`, `IN_GENRE`, `IN_COUNTRY`

You can run them in Neo4j Browser.

## 1. Show Some Nodes

```cypher
MATCH (n)
RETURN n
LIMIT 25
```

## 2. Count Nodes by Label

```cypher
MATCH (n)
UNWIND labels(n) AS label
RETURN label, count(*) AS count
ORDER BY count DESC, label
```

## 3. Count Relationships by Type

```cypher
MATCH ()-[r]->()
RETURN type(r) AS type, count(*) AS count
ORDER BY count DESC, type
```

## 4. Show Movie, Genre, Country Graph

This one is nice in graph view.

```cypher
MATCH p=(m:Movie)-[:IN_GENRE|IN_COUNTRY]->()
RETURN p
LIMIT 50
```

## 5. Show Actors and Their Movies

```cypher
MATCH p=(a:Actor)-[:ACTED_IN]->(m:Movie)
RETURN p
LIMIT 50
```

## 6. Show Directors and Their Movies

```cypher
MATCH p=(d:Director)-[:DIRECTED]->(m:Movie)
RETURN p
LIMIT 50
```

## 7. Show Rating Sources and Movies

```cypher
MATCH p=(u:User)-[r:RATED]->(m:Movie)
RETURN p
LIMIT 50
```

## 8. Top Rated Movies

`m.rating` is the IMDb rating from OMDb.

```cypher
MATCH (m:Movie)
RETURN m.title AS title,
       m.year AS year,
       m.rating AS imdb_rating
ORDER BY imdb_rating DESC, title
```

## 9. Average Rating from Rating Sources

This uses `RATED` relationships.

```cypher
MATCH (u:User)-[r:RATED]->(m:Movie)
RETURN m.title AS title,
       round(avg(r.rating) * 10) / 10.0 AS avg_source_rating,
       count(r) AS rating_count
ORDER BY avg_source_rating DESC, rating_count DESC, title
```

## 10. Movies Grouped by Genre

```cypher
MATCH (m:Movie)-[:IN_GENRE]->(g:Genre)
RETURN g.name AS genre,
       collect(m.title) AS movies,
       count(m) AS movie_count
ORDER BY movie_count DESC, genre
```

## 11. Movies Grouped by Country

```cypher
MATCH (m:Movie)-[:IN_COUNTRY]->(c:Country)
RETURN c.name AS country,
       collect(m.title) AS movies,
       count(m) AS movie_count
ORDER BY movie_count DESC, country
```

## 12. Movies With Full Basic Info

```cypher
MATCH (m:Movie)
RETURN m.title AS title,
       m.year AS year,
       m.rated AS rated,
       m.runtime AS runtime,
       m.rating AS imdb_rating,
       m.released AS released
ORDER BY m.year, title
```

## 13. Actor Movie Counts

```cypher
MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)
RETURN a.name AS actor,
       count(m) AS movie_count,
       collect(m.title) AS movies
ORDER BY movie_count DESC, actor
```

## 14. Director Movie Counts

```cypher
MATCH (d:Director)-[:DIRECTED]->(m:Movie)
RETURN d.name AS director,
       count(m) AS movie_count,
       collect(m.title) AS movies
ORDER BY movie_count DESC, director
```

## 15. Actor Collaborations

Actors who acted in the same movie.

```cypher
MATCH (a1:Actor)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(a2:Actor)
WHERE a1.name < a2.name
RETURN a1.name AS actor_1,
       a2.name AS actor_2,
       count(m) AS shared_movies,
       collect(m.title) AS movies
ORDER BY shared_movies DESC, actor_1, actor_2
```

## 16. Actor-Director Collaborations

```cypher
MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Director)
RETURN a.name AS actor,
       d.name AS director,
       count(m) AS collaborations,
       collect(m.title) AS movies
ORDER BY collaborations DESC, actor, director
```

## 17. Movies for a Given Actor

Example with Tom Cruise.

```cypher
MATCH (a:Actor {name: "Tom Cruise"})-[:ACTED_IN]->(m:Movie)
RETURN a.name AS actor, collect(m.title) AS movies
```

## 18. Movies for a Given Director

Example with Lana Wachowski.

```cypher
MATCH (d:Director {name: "Lana Wachowski"})-[:DIRECTED]->(m:Movie)
RETURN d.name AS director, collect(m.title) AS movies
```

## 19. Show One Movie Neighborhood

Good for graph view.

```cypher
MATCH p=(m:Movie {title: "The Matrix"})-[*1..1]-(n)
RETURN p
```

## 20. Show Movie With Actor and Director Together

Also good for graph view.

```cypher
MATCH p=(a:Actor)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Director)
RETURN p
LIMIT 50
```
