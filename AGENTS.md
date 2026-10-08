# Project instructions

Read and write text as UTF-8, preferably without BOM. Verify Chinese text after edits.

The research workflow is self-contained in `.agents/skills/autoresearch/SKILL.md`. Keep its references, assets and scripts within that directory so it can be installed independently.

When changing Python tools, review their interfaces, failure handling and tests. Run `python -m unittest discover -s tests -v`. Preserve failed experiment records and distinguish tool tests from actual research results. Do not add credentials, private data or runtime artifacts to the repository.
