# Triage Labels

The engineering skills speak in terms of five canonical triage roles. On our Multica tracker (`ccds-lab`), those roles map to labels and workflow statuses as follows:

| Role in `mattpocock/skills` | Label / Status in Multica | Meaning |
| --- | --- | --- |
| `needs-triage` | `needs-triage` (`backlog`) | Strong planner (`carbon-4`) or maintainer needs to evaluate scope and physical dimensions |
| `needs-info` | `needs-info` (`blocked` / `todo`) | Waiting on user decision on parameter bounds, physical assumptions, or go/no-go gate |
| `ready-for-agent` | `ready-for-agent` (`todo` on active stage, `backlog` on parked stage) | Orthogonal vertical slice with frozen interface contract and zero-regression gate, ready for `hydrogen-4` worker |
| `ready-for-human` | `ready-for-human` (`in_review`) | Integration audit passed; Ready PR awaiting human research lead merge into `main` |
| `wontfix` | `已归档` (`cancelled`) | Negative result recorded or out-of-scope engineering item archived |

When a skill instructs you to apply a triage role, use the corresponding Multica status/label convention above without reopening archived tickets.
