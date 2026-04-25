
## Folder Conventions
- `raw/` – Immutable source documents. Claude reads from here but NEVER modifies.
- `wiki/` – All LLM-generated pages go here. Claude owns this directory.
- `index.md` – Catalog of all wiki pages with one-line summaries. Claude updates on every ingest.
- `log.md` – Append-only record of ingests, queries, lint passes.

## Page Templates
### Source Summary Page (in `wiki/`)
```markdown
# [Source Title]
- **Source:** `raw/[filename]`
- **Date ingested:** YYYY-MM-DD
- **Type:** article / paper / book chapter / meeting notes
- **Key takeaways:**
  - Point 1
  - Point 2
- **Related pages:** [[Entity A]], [[Concept B]]