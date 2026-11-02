# satsim

## Layout

`input/` is the scenario data. `src/satsim/method/` is the event loop that runs it. `src/satsim/output/` is how a run becomes the two claims.

## Language

Before naming a claim, a knob, or a migration, read `GLOSSARY.md`. Before changing the scenario surface, the event loop, or how a claim is scored, read `docs/adr/`. How those files are consumed is in `docs/agents/domain.md`.

## Lab

A research result or a go / no-go goes to the self-hosted tracker at `api.gjam.tech`, workspace `ccds-lab`, entry `LAB-1466`.
