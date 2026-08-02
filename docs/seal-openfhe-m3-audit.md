# SEAL/OpenFHE M3 inactive-slot audit

Date: 2026-07-29

## Scope and decision

This is a source and documentation audit for the CPU-only `paper_compat` path.
It does not read or make claims about `MOAI_GPU`. The current profile remains
`security_claim=none`.

The initial M3 attempt stopped at `STOP_M3_GELU_INACTIVE_BOOTSTRAP_MASK`: the
encrypted smoke observed a GELU activation inactive-slot maximum of `4.3e-5`,
above the frozen `1e-6` gate. That failure left the remote at M2 (`8fc9e75`).
The later rescue sequence below remains candidate evidence until a clean-commit
artifact is validated, pushed, and remote-SHA checked.

## What the legacy path actually establishes

The vendored library declares Microsoft SEAL 4.1.2, but the legacy bootstrap is
MOAI's own `include/source/bootstrapping/Bootstrapper`, built on a modified SEAL
tree. Upstream [Microsoft SEAL 4.1.2](https://github.com/microsoft/SEAL/tree/v4.1.2)
does not expose a CKKS bootstrap API. Its CKKS example documents approximate
arithmetic, rescaling and level alignment; it does not promise bit-exact zero
after an encoded plaintext mask.

The legacy full-scheme packing is `slot = row * 256 + lane`. Its matrix kernels
rotate by multiples of 256, so those rotations preserve lane ownership. That is
a logical packing property, not a numerical-zero guarantee after CKKS encoding,
bootstrap or multiplication.

The full scheme calls `gelu_v2` from
`include/source/non_linear_func/gelu_others.hpp`. That degree-24 monomial
polynomial has `p(0)=1.21149468e-2` and has no active-mask argument. The next
FFN down-projection encodes zero weights in inactive slots and therefore only
approximately suppresses the constant. The old GELU inactive diagnostics are
commented-out prints around `1e-3`, not an absolute-value assertion. Therefore
the legacy source cannot serve as evidence for the new intermediate
`inactive max-abs <= 1e-6` contract.

## Current OpenFHE path

The OpenFHE implementation conditionally calls one native `EvalBootstrap`
before GELU, evaluates `gelu_d319` on `[-80,128]`, then performs one Ct-Pt mask
and rescale. The frozen plaintext polynomial has:

```text
P_raw(0)                  = 5.2324326311925518e-4
raw coefficient SHA-256  = b48fa6feaad5ee1565c0bc361cc7f09b5e543fb007492a3ed9239764f5b510f1
uniform-grid max error    = 1.1048447117423255e-3
```

Thus even an ideal zero inactive input is mapped to a deterministic nonzero
constant before the final approximate mask. The existing final observation
alone cannot apportion the remaining `4.3e-5` among bootstrap, polynomial
evaluation and mask arithmetic.

OpenFHE 1.5.1 also supports two-iteration `EvalBootstrap`. Its official
[iterative bootstrapping example](https://github.com/openfheorg/openfhe-development/blob/v1.5.1/src/pke/examples/iterative-ckks-bootstrapping.cpp)
describes the additional depth and precision measurement. That option may
reduce refresh error, but it cannot remove a deterministic nonzero polynomial
value at the origin, so it is not the first M3 rescue.

## Pre-registered M3-R1

First expose ciphertext-only checkpoints before/after the optional bootstrap,
before/after the final GELU mask and after the FFN down-projection. Only the
client decrypts them. This must not change the graph or operation counts.

If those checkpoints confirm the current diagnosis, keep the interval, degree,
depth and bootstrap placement and replace only the GELU coefficients with:

```text
delta   = P_raw(0)
c0_zero = c0_raw - 2 * delta
P_zero  = P_raw - delta
```

The factor two follows OpenFHE's raw Chebyshev convention, where the evaluator
uses `c0/2`. A read-only plaintext trial produced:

```text
P_zero(0)                 = 1.7763568394002505e-15
coefficient SHA-256       = 35d68b2f56f267f27e8f962c3bd54cddd90892349b77405172f48987864eaaa3
uniform-grid max error    = 1.5942546977454342e-3
12-layer worst rel-L2     = 3.557446538016513e-3
12-layer minimum cosine   = 0.9999938697929660
```

These plaintext results satisfy the unchanged nonlinear quality gates but are
not encrypted validation. The calibration script and C++ registry must derive
the same correction and freeze its metadata. M3 additionally requires three
independent encrypted smoke passes with every inactive checkpoint at or below
`1e-6`. Failure records a new blocker and leaves M4 unopened.

## Executed D1 and R1 result

The ciphertext-only checkpoints localized the current single-iteration path:

```text
                                      raw d319       zero-at-origin d319
pre-activation inactive max-abs       1.2791e-11     1.3084e-11
post-bootstrap polynomial input       4.1757e-4      5.5673e-4
pre-mask polynomial output            7.8584e-4      3.0100e-4
post-mask GELU activation             1.1434e-4      8.5037e-5
FFN output                            1.7540e-4      1.0578e-4
```

The zero-at-origin constraint removed the deterministic offset but did not
meet the encrypted inactive gate. This is
`STOP_M3_R1_ZERO_ONLY_BOOTSTRAP_ERROR`; it is not committed or pushed.

The next pre-registered attempt keeps the constrained polynomial and uses
OpenFHE's two-iteration Meta-BTS with fixed `precision=5`, default correction
factor, ModRaise-first, `[4,4]`, 16 encoded smoke slots and 12 levels available
after bootstrap. The precision is derived from the observed single-bootstrap
error scale (`floor(-log2(5.6e-4))-5`). The context depth gains exactly the
official `(numIterations-1)` level. No other parameter or gate changes.

That `precision=5` run reduced the post-bootstrap inactive maximum to
`1.3684e-5`, but the final GELU mask still produced `3.7563e-6`; it stopped as
`STOP_M3_R2_META_BTS_P5_PRECISION`.

R3 changes only the fixed Meta-BTS precision to `10`. This is the conservative
single-bootstrap precision `floor(-log2(5.6e-4))`, selected directly because
the two-iteration target needs about 20 precision bits for the `1e-6` gate. It
is not a sweep over intermediate values. All other graph, profile and gate
fields remain frozen; a first failure stops R3.

## Softmax trust-boundary correction

The first precision-10 diagnostic passed numerically, but its Softmax fixture
computed a per-row logsumexp vector from plaintext logits at runtime and passed
that activation-derived vector to the server. This violates the strict
server-only contract and stops R3 independently of its numerical result.

R4 removes the vector-shift server API. Each layer/head may use only one public
scalar frozen during offline calibration and shared by every query row; the
reduced smoke uses the literal `3.0`. The fixed trace contract selects the
maximum bundled-query-row logsumexp for each layer/head and freezes the resulting
12x12 table. It uses reciprocal degree 383 on `[8e-5,1.2]`, whose plaintext trace
gate already passes, and raises the post-bootstrap Softmax level budget to 12.
No runtime plaintext activation may choose or alter the shift.

## Executed R4 result and pre-registered M3-R5

The first R4 encrypted diagnostic preserved the GELU/FFN inactive gate, but
decrypting both the degree-383 reciprocal and the final Softmax output failed
with OpenFHE's `approximation error is too high` guard. The public-shift smoke
denominators were about `0.05--0.18`; the direct `1/x` approximation on
`[8e-5,1.2]` nevertheless carries an endpoint magnitude of 12500 and is not an
executable encrypted contract under the frozen profile. This is
`STOP_M3_R4_RECIPROCAL_NOISE`; no threshold is changed and the activation-derived
vector API remains forbidden.

R5 keeps exactly one offline-frozen public scalar for every `(layer, head)`, but
balances it at the midpoint of that head's minimum and maximum bundled-query-row
logsumexp. The scalar is still shared by every query row and cannot be derived or
changed at runtime. This deterministically balances the denominator range in log
space without exposing a private activation. The reduced encrypted smoke binds
the frozen registry entry `(layer=1, head=2)`, whose public scalar is
`2.9766493219191048`.

The R5 approximation contract is frozen before its encrypted run:

```text
shift[layer,head] = (min_row logsumexp + max_row logsumexp) / 2
exp                degree 27,  interval [-16,5], depth 6
reciprocal         degree 383, interval [0.01,80], depth 10
observed shifted logits       [-15.029251765519266,4.329632094480733]
observed denominator          [0.013162705846439082,75.97222108387969]
exp coefficient SHA-256       6eda4377151897e8c4ca4d72f2a918db0b888fc6771f8ee5cf1d950b378de76f
reciprocal coefficient SHA-256 fa97f298751bca97f40eed3b6de949262d1b3971cda57013b55420d5c9fbbeb3
plaintext Softmax worst rel-L2 2.9459602006905205e-4
plaintext Softmax minimum cosine 0.9999999568418523
plaintext attention worst rel-L2 3.1537180892426327e-4
plaintext attention minimum cosine 0.9999999530914095
```

The reciprocal condition-number ratio is not hidden or weakened, but its
maximum represented value falls from 12500 to 100. The minimum observed
denominator is more than 13000 times the fixed `1e-6` two-iteration bootstrap
error budget. All existing quality, inactive-slot, level, trust-boundary and
three-independent-run gates remain unchanged. A first encrypted R5 failure
stops the rescue rather than starting an interval or parameter sweep.

The first three R5 numerical repeats passed, but a subsequent trust-boundary
audit found that the smoke still passed plaintext-observed ranges into the
server-side nonlinear API. Those values were used only as guards and did not
change ciphertext arithmetic, yet they were activation-derived metadata and
therefore invalidated those runs as final server-only evidence. The API now
binds intervals exclusively through the approximation registry; observed
ranges remain in the offline/client validator. Final repeated evidence must be
rerun after this interface correction.

The same final audit freezes four additional evidence corrections before the
rerun. Ciphertext multiplication must reject inconsistent logical shapes while
retaining the explicit one-feature rhs broadcast and the linear-kernel
diagonal-times-column singleton case. Softmax now receives `(layer, head)` and
resolves the scalar through the frozen 12x12 registry table with SHA-256
`41ecf6ade53f674096c9afc752ccfe99834876ea4768bcf4bf3ea25a8f502432`;
the reduced smoke binds to `(1,2)`, value `2.9766493219191048`, rather than an
unbound literal. The post-bootstrap denominator max-absolute-error gate is
tightened from `5e-3` to the already calibrated `1e-6` budget. Finally, a v2
GO artifact must contain only successful commands, all required gate checks as
PASS, and the exact `stdout.log`, `metrics.csv`, and `SHA256SUMS` evidence roles.
None of these corrections relaxes a numerical gate or permits runtime-derived
plaintext metadata.

## Fresh corrected R5 candidate result

After all API, profile/context, shape, counter, and artifact-contract corrections,
three independent encrypted repeats passed before the milestone commit. No source
or threshold changed between repeats:

| Run | Inactive max-abs | GELU rel-L2 | FFN rel-L2 | Softmax rel-L2 | LayerNorm rel-L2 | Seconds | Peak RSS KiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `5.3539299549321129e-7` | `1.0660121565620579e-6` | `8.7225477083418275e-7` | `2.0230422348656657e-9` | `3.9655591102061334e-11` | `51.71` | `5633172` |
| 2 | `5.2636586270855999e-7` | `7.3651815770600955e-7` | `6.0796266787210431e-7` | `2.1999615023289514e-9` | `4.5972309767802692e-11` | `52.86` | `5633256` |
| 3 | `4.4911718287257510e-7` | `6.8726617512637687e-7` | `5.4585559617094279e-7` | `2.0860172529687237e-9` | `3.8259021690499559e-11` | `52.20` | `5633024` |

All reported cosines exceed `0.9999999999997`; the worst post-bootstrap
denominator max-absolute error is `2.5265067815638531e-11`. Every run reports
rotations/Ct-Pt/Ct-Ct/rescale `0/19/10/28`, logical/iterative bootstraps `3/6`,
five Chebyshev evaluations, 111 estimated polynomial multiplications, maximum
polynomial depth 10, and maximum observed level 30 of 31. These are reduced
16-slot M3 smoke resource observations, not M6 single-sample timing evidence.

This closes the commit-before-test gate only. The authoritative M3 GO requires a
clean milestone commit, a schema-v2 artifact generated from that executable,
successful push, live remote-SHA validation, and the corresponding Vault log.

## Evidence boundary

- Upstream SEAL documentation establishes approximate CKKS and rescale/level
  semantics, not MOAI bootstrap correctness.
- Legacy MOAI source audit establishes old control flow and its weaker masking
  behavior; it is not a runtime reproduction.
- The numerical trial above is plaintext-only.
- Only a clean, repeated encrypted M3 validator can close the milestone.
