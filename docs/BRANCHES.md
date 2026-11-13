# Repository branches

The owner explicitly chose separate, reproducible experiments. There is no
canonical merged experimental model and no new handoff branch. Shared package
names describe responsibilities; their implementation is extracted from each
branch's own code. Do not assume identical behavior because names match.

| Branch | Independent role | Pre-cleanup commit |
| --- | --- | --- |
| `main` | Original Simple-BEV reference, unchanged | `be46f0ef71960c233341852f3d9bc3677558ab6d` |
| `dd2414-mini-baseline` | Workstation/scaling and official full-data results | `0225bbef5f69ae04e0d52adbd4ee9d393cb1678e` |
| `experiment/hybrid-teacher-10sweep-cluster` | Historical hybrid teacher and independent RPL setup | `33f2d86d70286c514cede6539f991825dde726ee` |
| `experiment/rpl-full-data-double-distill` | Official-RGB full-data double distillation | `f9f333f8607f22514df216d5969f23a0ecbf5917` |
| `experiment/rpl-depth-10sweep-velocity-study` | Ten-sweep depth/velocity DDP experiments | `0f91e222d658439d253dde4babf9a353056eecd3` |

Every original branch remains available. The rejected
`codex/team-integration-20261113` and temporary snapshot branch were deleted
locally/remotely only after their exact heads were saved as verified tags.
Their commits remain recoverable, not force-rewritten or merged into experiments.

For development, select the required experiment and use its own checkout,
environment, manifest and checkpoint provenance. Avoid one editable installation
silently pointing at a different experiment. `DD2414_PROJECT_ROOT` can nominate
the matching source checkout; actual package bytes are checked before training.
Existing remote jobs keep their original immutable execution roots.
