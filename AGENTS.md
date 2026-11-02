# satsim

## Layout

`input/` is scenario data + offline generators. `src/satsim/method/` is engine + interfaces + policies. `src/satsim/output/` turns run rows into the two claims. Method hooks are a single global Scheduler (place + maybe_rebalance) plus link_cost. See `docs/SPEC_PLUGIN_FRAMEWORK.md`.

## Language

Before naming a claim, a knob, or a migration, read `GLOSSARY.md`. Before changing the scenario surface, the event loop, or how a claim is scored, read `docs/adr/`. How those files are consumed is in `docs/agents/domain.md`.

## Lab

A research result or a go / no-go goes to the self-hosted tracker at `api.gjam.tech`, workspace `ccds-lab`, entry `LAB-1466`.
