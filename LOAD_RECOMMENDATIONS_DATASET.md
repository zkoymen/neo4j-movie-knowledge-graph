# Load Recommendations Dump In Neo4j Desktop

Use a new DBMS. Do not load this dump into the OMDb test database.

## Which dump file?

Choose:
- `recommendations-5.26.dump`

Reason:
- `aligned` is the safer default choice.
- `block` is only worth choosing if you already know your DBMS/store format should use block.

## Short Steps

1. Open Neo4j Desktop.
2. Create a new project if you want, or open your current project.
3. Drag `recommendations-5.26.dump` into the project `Files` area.
4. Click the file options menu.
5. Choose `Create new DBMS from dump`.
6. Give the DBMS a clear name such as `recommendations-dbms`.
7. Set username/password if Desktop asks.
8. Start the DBMS.
9. Open the DBMS details and check:
   - Bolt URI
   - username
   - password
   - default database name

## .env Example

```env
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
NEO4J_DATABASE=neo4j
DATASET_NAME=recommendations_dump
```

If your default database name is not `neo4j`, change `NEO4J_DATABASE`.

## Quick Validation

After the DBMS is running, test these in Neo4j Browser:

```cypher
MATCH (n)
UNWIND labels(n) AS label
RETURN label, count(*) AS count
ORDER BY count DESC, label
```

```cypher
MATCH ()-[r]->()
RETURN type(r) AS type, count(*) AS count
ORDER BY count DESC, type
```

Expected model:
- Labels include `Movie`, `Actor`, `Director`, `User`, `Genre`
- Relationship types include `ACTED_IN`, `DIRECTED`, `RATED`, `IN_GENRE`
