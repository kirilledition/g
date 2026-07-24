# Output Transaction BeeGFS Qualification

| Status | Applies to | Owner |
| --- | --- | --- |
| Transition-protocol requalification required | Gauss BeeGFS mount; primitive evidence from 2026-07-23 | Development maintainers |

This handoff records the cross-node qualification of the append-only output
transaction primitives on `/mnt/beegfs/kirill`. The Rust test binary was
compiled once on `ramanujan` with the portable fixed target
`-Ctarget-cpu=x86-64-v3` and executed from the shared filesystem on both nodes.

## Current Result

| Primitive | Producer | Contender/consumer | Result |
| --- | --- | --- | --- |
| Immutable hard-link no-replace publication | Slurm 46630, `ramanujan` | Slurm 46631, `shannon` | Pass |
| Live durable owner-claim exclusion | Slurm 46682, `ramanujan` | Slurm 46693, `landau` | Pass |
| Surviving claim after owner `SIGKILL` | Slurm 46682 cancelled | Slurm 46694, `landau` | Pass: contender remained blocked |

The immutable-publication consumer observed the same device/inode and the same
SHA-256 on both names:

```text
9df830d529b8d403e5d780904942c444b5dcf55dd5d2b877b6aafe1533fc8d90
```

Creating the destination a second time failed without replacement. The owner
claim uses the same hard-link no-replace primitive at
`.g-output/session.claim.json`. While job 46682 held that claim, job 46693
observed the typed blocked-owner result on the other node. After the holder was
killed, job 46694 still observed the claim and remained blocked. This is the
required fail-closed behavior: process death never authorizes takeover.
The earlier 46695/46696 exercise tested a retired unlink-based release and is
not qualification evidence for the current protocol. Before release, the
current implementation still requires a cross-node test in which graceful
release, fenced takeover, and reacquisition contend through immutable
predecessor transition slots and both readers resolve the same final leaf.

## Rejected Locking Mechanisms

The deployed mount failed the locking mechanisms evaluated before the durable
claim design:

| Rejected primitive | Owner | Cross-node contender | Result |
| --- | --- | --- | --- |
| Rust `File::try_lock` | Slurm 46616, `ramanujan` | Slurm 46619, `shannon` | Fail: contender acquired |
| POSIX `fcntl` probe | Slurm 46622, `ramanujan` | Slurm 46626, `shannon` | Fail: contender acquired |

Those failures are evidence for excluding advisory locks from correctness, not
an open blocker in the current design. The implementation does not use either
mechanism.

This deliberately differs from the original roadmap's `File::try_lock`
lease semantics. A process-scoped advisory lease would release automatically
after `SIGKILL`, but the cross-node probes show that it does not exclude another
BeeGFS client and therefore cannot be the ownership authority. The hard-link
claim provides the required exclusion and visibility, at the cost of surviving
process death.

## Operational Decision

Cross-node initial ownership exclusion is qualified through durable hard-link
compare-and-set. The permanent root claim and each immutable owner transition
use that primitive, but the composed transition protocol remains a release
blocker until its current cross-node qualification is recorded here. A live or
crashed owner leaves an Active authority leaf, so automatic stale detection,
timeout takeover, authority deletion, and unlink-on-open are forbidden.
Recovery after process death requires an external coordinator to fence the
recorded host/process so it cannot write or publish a graceful release. The
operator then supplies the exact current Active identifier from the typed error through
`--fenced-output-owner-claim CLAIM_ID` or
`[output].fenced_owner_claim_id`. The output manager verifies that the same
leaf is still current and publishes one no-replace fenced-takeover transition.
An absent, different, or historical identifier fails without changing
authority. A nonterminal leaf additionally requires its exact
`recover_attempt`. Claim age and PID liveness are not fencing evidence, and
manual deletion is never the supported workflow.

Re-run this qualification after a BeeGFS mount, client, metadata service, or
filesystem configuration change. Failure of hard-link no-replace visibility,
live-claim exclusion, post-kill persistence, transition-slot contention, or
cross-node final-leaf agreement is a release blocker.

The raw qualification logs are retained under the ignored local build
directory:

```text
target/beegfs-qualification.KkLSuh/
```

They are diagnostic artifacts, not repository or output data.
