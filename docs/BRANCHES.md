# Experiment branches

Choose the branch for the experiment you want to reproduce. Common package
names are used across branches, but model implementations and settings differ.

| Branch | Independent role | Original revision |
| --- | --- | --- |
| `main` | Original Simple-BEV reference | `be46f0ef71960c233341852f3d9bc3677558ab6d` |
| `experiment/workstation-radar-baseline` | Workstation/scaling and official full-data results | `0225bbef5f69ae04e0d52adbd4ee9d393cb1678e` |
| `experiment/hybrid-teacher-10sweep-cluster` | Historical hybrid teacher and independent RPL setup | `33f2d86d70286c514cede6539f991825dde726ee` |
| `experiment/rpl-full-data-double-distill` | Official-RGB full-data double distillation | `f9f333f8607f22514df216d5969f23a0ecbf5917` |
| `experiment/rpl-depth-10sweep-velocity-study` | Ten-sweep depth/velocity DDP experiments | `0f91e222d658439d253dde4babf9a353056eecd3` |

Use a separate environment for each experiment, or reinstall the selected
checkout before running it. An editable install points to one checkout at a
time. `DD2414_PROJECT_ROOT` selects the matching source directory; training
checks that the installed package matches those source files.

Historical checkpoints use the source revision recorded with the run.
[Recovery instructions](REPRODUCIBILITY.md) explain how to open an older version
in a separate worktree.

## Workstation branch name

`dd2414-mini-baseline` was renamed to `experiment/workstation-radar-baseline`
on 2026-11-14. The old name came from the initial mini-dataset reproduction;
this branch also contains full-data experiments. The rename preserves its history.
Snapshot tags and recorded provenance retain the old name deliberately.

For an idle local checkout currently on `dd2414-mini-baseline`:

```bash
git branch -m experiment/workstation-radar-baseline
git fetch origin
git branch --set-upstream-to=origin/experiment/workstation-radar-baseline
```

Keep checkouts used by running jobs at their recorded source revision. Migrate
their local branch and update source only when those jobs no longer depend on them.
