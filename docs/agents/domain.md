# Domain docs

How to read this repo's domain language.

## Before exploring, read these

- `GLOSSARY.md` at the repo root（论文三档：首次分配 / 无消耗再分配 / 有消耗再分配）.
- `docs/adr/` for any decision that touches the area you are about to change（含 `0002-three-tier-schedulers.md`）.

If a file is missing, proceed. Create a glossary entry or an ADR only when a term or a decision is actually resolved.

## Layout

```
/
├── GLOSSARY.md
├── docs/adr/
└── src/
```

## Vocabulary

Name a claim, a knob, or a migration with the term in `GLOSSARY.md`. When a term you need is missing, either stop inventing a synonym or record the gap as a new glossary entry.

## ADR conflicts

If a change contradicts an ADR, say so in the write-up before overriding it.
