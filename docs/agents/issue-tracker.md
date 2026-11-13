# Issue Tracker (Multica)

Issues and task orchestration for `acam-infra/satsim` and `acam-infra/satcosim` live in the **Multica** platform (`gae.gjam.tech`, workspace `ccds-lab`), **not** in GitHub Issues. Pull requests still live on GitHub (`acam-infra/*`).

## CLI Invocation

Always invoke the Multica CLI via the daemon-injected `$MULTICA_CLI_PATH` (quoted):

```bash
"$MULTICA_CLI_PATH" issue get <issue-id> --output json
"$MULTICA_CLI_PATH" issue comment list <issue-id> --roots-only --summary --compact --output json
"$MULTICA_CLI_PATH" issue children <parent-issue-id> --output json
```

## Publishing Specs and Tickets (`/to-spec`, `/to-tickets`, `/implement-spec`)

- **Do NOT run `gh issue create`.** Never duplicate tickets onto GitHub Issues.
- **Sub-issues & Blocking Edges**: Express dependency stages on Multica using `--parent <parent-issue-id>` and `--stage <N>`:
  - Stage 1 ready frontier (starts immediately on the assigned concurrent worker):
    ```bash
    "$MULTICA_CLI_PATH" issue create \
      --title "Satellite · S3 · <Task ID>：<Title>" \
      --parent <parent-issue-id> \
      --stage 1 \
      --assignee-id <worker-agent-uuid> \
      --status todo \
      --description-file ./child_desc.md \
      --output json
    ```
  - Later blocked stages (parked until the Stage 1 barrier closes and is audited):
    ```bash
    "$MULTICA_CLI_PATH" issue create \
      --title "Satellite · S3 · <Task ID>：<Title>" \
      --parent <parent-issue-id> \
      --stage 2 \
      --assignee-id <worker-agent-uuid> \
      --status backlog \
      --description-file ./child_desc.md \
      --output json
    ```
- **Orthogonal Slicing Rule**: Sibling tickets in the same `--stage` MUST touch disjoint code files so concurrent `hydrogen-4` workers on separate branches/worktrees merge cleanly into the stage integration branch (`int/step<n>-*`) with zero rebase conflicts.

## Linking Pull Requests

- Every worker branch pushes to `acam-infra/satsim` (or `acam-infra/satcosim`) targeting the stage integration branch (`int/step<n>-*`), opening a **Draft PR** on first push.
- Prefix PR titles with the Multica issue identifier so Multica links the PR automatically, e.g.:
  `LAB-1867: satsim: DP-1A discrete power tiers in CoWorld`
- PRs as a request surface: **off** (triage operates on Multica issues only).
