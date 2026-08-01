# OpenFHE CPU 12-layer encoder trace contract

## Evidence boundary

`config/moai_encoder_trace.json` freezes the complete 12-layer, five-token
BERT-base encoder fixture used by the `paper_compat` migration. It is a plaintext
trace and polynomial-oracle contract with `security_claim=none`; it is not evidence
that a 12-layer ciphertext execution has run. Passing it is a prerequisite for the
M5 server-only OpenFHE run, not the M5 completion decision itself.

The contract is fixture-only. It cannot be generalized to a new sentence, a new
token count, or a different model. It excludes GPU work, Discrete CKKS/FBT, QDQ,
tokenization, classification, and task-level end-to-end inference. Runtime
activation-derived approximation metadata is forbidden.

## Frozen evidence

The manifest binds both source contracts by SHA-256:

- `config/moai_trace_channel_scales.json`, including every sparse FFN channel
  scale;
- `config/openfhe_approximations.json`, including the 12-by-12 public Softmax
  shifts, polynomial degrees, intervals, coefficient hashes, LayerNorm epsilon,
  and the two public variance scales.

Each `data/layer_0` through `data/layer_11` directory contributes all 37 CSV files:

| Trace group | Files per layer | Contracted content |
|---|---:|---|
| Self-attention results | 8 | token ids, embedded input, Q/K/V, QK^T/8, exact Softmax, attention output |
| Self-attention parameters | 6 | Q/K/V weights and biases |
| Self-output results | 5 | attention input, dense output, residual operands/sum, LayerNorm output |
| Self-output parameters | 4 | dense and LayerNorm weights/biases |
| Intermediate results | 3 | FFN input, scaled pre-activation, GELU output |
| Intermediate parameters | 2 | first FFN weight and bias |
| Final-output results | 5 | GELU snapshot, dense output, residual operands/sum, LayerNorm output |
| Final-output parameters | 4 | second FFN and LayerNorm weights/biases |

That is 444 file hashes. Every path, shape, orientation, finite-value requirement,
and digest is fail-closed. The five token ids must be nonnegative integers and
byte-identical across the 12 layer snapshots. The attention and FFN residual
operands must be byte-identical to their layer inputs, and each layer output must
be byte-identical to the next layer input.

## Static trace equations

CSV matrices are row-oriented while stored dense weights are `output_by_input`.
For the five token rows, the validator checks:

```text
Q = X Wq^T + bq                       K = X Wk^T + bk
V = X Wv^T + bv                       L = Q K^T / sqrt(64)
P = stable_softmax(L)                  A = P V
S = A Wo^T + bo + X                   H = exact_LN(S; gamma1, beta1)
Z = (H W1^T + b1) * scale             G = exact_GELU(Z)
Y = G (W2 / scale)^T + b2 + H         O = exact_LN(Y; gamma2, beta2)
```

QK and attention are reshaped as
`[query=5, head=12, key=5]` and `[token=5, head=12, feature=64]`.
LayerNorm uses population variance and epsilon `1e-12`. The sparse `scale` vector
defaults to one; its integer overrides are copied from and cryptographically bound
to the existing channel-scale contract. Dividing W2 columns by the same vector
preserves the reconstructed FFN relation.

Seventeen relation-specific gates are locked in the manifest. They are trace
parity gates, not encrypted tolerances. They cover Q/K/V, QK^T/8, exact Softmax,
attention, every snapshot link, both dense/residual paths, exact GELU, and both
exact LayerNorm sites.

## Chained frozen-polynomial oracle

The second pass starts only from layer 0 `embedded_inputs` and feeds each computed
output directly into the next layer. It never resets from an intermediate trace
checkpoint. It uses the exact OpenFHE coefficient-generation convention recorded
by M3 (DCT-II nodes, doubled raw `c0`, evaluator consumes `c0/2`) and independently
regenerates every coefficient hash before replay:

| Operator | Degree | Hard interval |
|---|---:|---:|
| zero-preserving GELU | 319 | `[-80, 128]` |
| exponential | 27 | `[-16, 5]` |
| reciprocal | 383 | `[0.01, 80]` |
| inverse square root | 159 | `[0.5, 1536]` |

Softmax subtracts one pre-registered public scalar per layer/head, shared across
all five query rows. LayerNorm applies `D_s=64` at the first site and `D_s=1` at
the second site. Any polynomial input outside its hard interval stops validation;
there is no clipping or epsilon repair.

The frozen chained replay passes all 12 per-layer gates (`rel-L2 <= 0.012`,
`cosine >= 0.9999`, `max-abs <= 0.1`). Its final layer-11 metrics are approximately
`rel-L2=0.0105192`, `cosine=0.999944943`, and `max-abs=0.0761826`, inside the
unchanged final acceptance gate (`rel-L2 <= 0.05`, `cosine >= 0.99`, no NaN/Inf).
The manifest preserves the exact machine-readable per-layer metrics and observed
ranges; the validator recomputes and compares them with only last-bit BLAS
tolerance.

These plaintext figures must not be reported as ciphertext correctness, latency,
security, bootstrap success, or a complete server-only encoder result.

## Run and failure decisions

Use Python 3.10 or 3.11 with NumPy:

```bash
/home/shawnsheep/miniconda3/envs/fhe-inference/bin/python3.10 \
  scripts/validate_openfhe_encoder_trace.py
```

A successful run reports every chained layer, `validated_files=444`, and:

```text
DECISION=PASS_OPENFHE_ENCODER_TRACE_CONTRACT
```

Missing/extra manifest fields, duplicate JSON keys, altered source contracts,
digest/shape/orientation drift, NaN/Inf, noninteger token ids, broken residual or
layer chaining, changed sparse scales, coefficient drift, approximation range
violations, relation-gate failures, or frozen-summary drift all return nonzero and:

```text
DECISION=FAIL_OPENFHE_ENCODER_TRACE_CONTRACT
```

`--emit-config` emits a measured candidate manifest to stdout and never writes a
file. It exists for an explicit future contract revision; replacing the committed
manifest is not an automatic threshold-update mechanism.

Run the fail-closed tests with:

```bash
/home/shawnsheep/miniconda3/envs/fhe-inference/bin/python3.10 \
  tests/test_validate_openfhe_encoder_trace.py
```

The validator and its fail-closed tests are registered with CTest by the M4
integration build.

## M4 encrypted layer gate

M4 replays layer 1 from its frozen five-token input at ciphertext level 29. The
server receives five 1024-slot ciphertexts, public layer weights, and the
evaluation-key bundle. It performs no decryption and owns no private key or
plaintext activation. The historical artifact established `attention_output` as the raw
weighted-V boundary before its native bootstrap and public-prefix cleanup, matching the
trace boundary consumed by the self projection. A current run must preserve that
boundary: the client also decrypts and gates the cleaned copy without aliasing over or
hiding the raw checkpoint. The other output checkpoints cover the self-attention
LayerNorm output, FFN output, and final encoder output.

The final encrypted output is compared to the single-layer frozen-polynomial
oracle, not directly to the exact BERT trace. Exact-trace parity remains a
diagnostic only. The hard gates are `rel-L2 <= 1e-2`, `cosine >= 0.999`, and
inactive/cross-lane maximum absolute value `<= 1e-6`; none may be relaxed. The current
depth-12/post-inverse-cleanup graph does not yet have a sealed M4 metadata schedule. The
reviewed M5 two-layer calibration uses a different layer position and chained-input
contract, so its layer-0 tuples and counts must not be relabelled as an M4 pass. A new
five-token server-only M4 run must independently measure and exact-gate its checkpoints,
operation counts, maximum level, and numerical quality before v5 can be claimed or
reported as current M4 evidence. Under `FLEXIBLEAUTO`, an explicit rescale request is a
runtime API count and is not claimed to be a physical modulus reduction.

The M4 client also decrypts the Softmax denominator and both LayerNorm
normalized-variance checkpoints. All values remain subject to their frozen reciprocal
or inverse-square-root intervals. The Softmax inactive sentinel keeps the reciprocal
interval gate. Each LayerNorm inactive guard must be finite and inside `[0.5,1536]`;
its deviation from one is diagnostic only. The final LayerNorm output inactive tail,
like the other zero-valued encoded checkpoints, retains the hard `1e-6` gate.

The feature-packed runtime uses two-iteration native OpenFHE bootstrap precision
14 with a separately hashed 50-bit scaling modulus and 55-bit first modulus;
M1-M3 retain the frozen 46/51-bit profile. The effective feature profile SHA-256
is `94f30e628e21f02146ce7ed9820194eabba3820f6e1e17176a31f8c5acf8b0be`, and
still carries `security_claim=none`. Its independent 1024-slot bootstrap smoke
measured active/inactive max-abs `8.55e-9/9.40e-9`.

In the two-token 12-head diagnostic, the old 46/51-bit candidate produced
attention-output inactive max-abs `1.65e-6`; the frozen 50/55-bit candidate
produced probability/raw-output/cleaned-output inactive max-abs
`8.62e-9/1.00e-7/9.74e-8` without changing the four-bootstrap attention
schedule. This is a narrow diagnostic, not the M4 encoder decision. The
feature-packed MultiHead Softmax evaluates the frozen exp
polynomial, then subtracts the public constant
`P_exp(0)=1.000000000011056` only from public inactive slots. This is a Ct-Pt
addition and does not consume a multiplication level. The path uses 10 reciprocal
levels and one normalization level after bootstrap, for post-bootstrap depth 11;
the ordinary M3 Softmax contract remains 12. Weighted-V is refreshed with native
bootstrap and then cleaned by one public 768-prefix mask. The three FFN W2 blocks
retain ciphertext diagnostic contributions but share one final explicit rescale
and one public output-bias addition.

The v5 schema/runner/validator code contract is aligned with the depth-12 graph. Only
after a new live M4 run passes, the milestone commit is pushed, and local/remote SHAs
match should the intended runner be used:

```bash
/home/shawnsheep/miniconda3/envs/fhe-inference/bin/python3.10 \
  scripts/run_openfhe_encoder_artifact_v5.py
```

The versioned v5 runner is intentionally M4-only. The published v2
schema/runner/validator remain byte-preserved for the historical M4 artifact and must
not be reinterpreted. The aligned v5 code contract is not new M4 evidence by itself. A
formal run must fix the repository data and build paths,
verify all 37 layer-1 CSV hashes, perform a fresh configure and clean-first build,
verify OpenFHE 1.5.1 package/linkage, rerun the fast contracts, and execute one warm-up
plus five measured runs. The binary and all frozen inputs must be rehashed before and
after every execution; any drift or failed gate returns nonzero.

When aligned, the v5 and v6 runners use the same scrubbed subprocess environment for configure,
build, CTest, `ldd`, workload, and validator execution. Their manifests bind the fresh
`CMakeCache.txt`, the four OpenFHE CMake package files, a canonical digest over every
regular file in `include/openfhe`, and the exact resolved
`libOPENFHE{binfhe,core,pke}.so.1.5.1` files by path, byte count, and SHA-256. They
rehash this provenance before and after the workload; inherited compiler/linker/include
flags, a stale toolchain, a changed symlink target, or library/package/header byte drift
fails closed.

## M5 encrypted 12-layer gate

M5 starts from the same client-encrypted five-token layer-0 input and calls the
server-side encoder once with exactly 12 ordered public weight sets. It never resets an
activation from the plaintext trace. Layers 0 through 10 pass their raw ciphertext
output through one native bootstrap and a public 768-prefix mask before the next layer;
layer 11 returns its raw ciphertext output for final client decryption. Therefore the
formal chain has 12 server layer evaluations and exactly 11 inter-layer refreshes.

The server target owns only the ciphertexts, public weights, and evaluation-key bundle.
It has no private-key, decryptor, or plaintext-activation API. The correctness executable
also links the client runtime so a client-owned observer can decrypt cloned ciphertext
checkpoints and validate them; that composition does not move decryption across the
server API and is not a claim of operating-system process isolation.

The reviewed live two-layer calibration supplied the original schedule candidate below.
The approved exact-three run
`20260801T124248+0900-m5-runnable-prototype-exact3-339daa5-r23` subsequently evaluated
the second handoff (layer 1 raw output to layer 2 refreshed input), rechecked the
later-layer tuple at layer 2, and transitioned the profile to `sealed_exact3`. The
shortened evidence remains `artifact_eligible=false`: it seals only the schedule
prerequisite and cannot establish M5 completion. Exact one and exact two can never claim
the schedule is sealed. Every tuple below is `(level, noise-scale degree, remaining
levels, canonical scale bits, ciphertext count)`. Runtime `log2(scale)` must be finite
and within `1e-3` of the canonical value.

| Position | Layer 0 | Layers 1-11 |
|---|---|---|
| Layer input | `(29,1,18,50,5)` | `(19,2,27,100,5)` |
| Softmax denominator after bootstrap | `(18,2,28,100,5)` | `(18,2,28,100,5)` |
| Attention LayerNorm variance after bootstrap restore | `(19,2,27,100,5)` | `(19,2,27,100,5)` |
| Output LayerNorm variance after bootstrap restore | `(19,2,27,100,5)` | `(19,2,27,100,5)` |
| Raw attention output | `(40,2,6,100,5)` | `(31,2,15,100,5)` |
| First LayerNorm output | `(32,2,14,100,5)` | `(30,2,16,100,5)` |
| FFN output projection | `(45,2,1,100,5)` | `(43,2,3,100,5)` |
| Raw encoder output | `(30,2,16,100,5)` | `(30,2,16,100,5)` |

The relative used-level delta tuple is ordered as input-to-Softmax recovery,
Softmax-to-raw-attention consumption, raw-attention-to-LN1 recovery,
LN1-checkpoint-to-LN1-output consumption, LN1-output-to-FFN consumption,
FFN-to-LN2 recovery, LN2-checkpoint-to-raw-output consumption, and
previous-raw-output-to-refreshed-input recovery. It is
`(10,22,21,13,13,26,11,null)` for layer 0 and
`(1,13,12,11,13,24,11,11)` for the candidate later regime. The exact path fails
closed on these position-specific tuples and deltas; only the explicitly diagnostic
calibration mode remains permissive for collecting new live evidence. Every candidate
inter-layer refresh recovers 11 used levels. The production encoder checks the complete
`(19,2,27,2^100,5)` handoff before starting every later layer, even when no diagnostic
observer is attached.

The live two-layer output measured the per-layer, per-refresh, and cumulative candidate
counts below. The 12-layer row is a deterministic projection of those candidates, not
executed full-chain evidence:

| Scope | Rotations | Ct-Pt mul | Ct-Ct mul | Rescale requests | Chebyshev evals | Estimated polynomial mul | Bootstraps | Bootstrap iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| One encoder layer | 6300 | 51885 | 95 | 810 | 55 | 1150 | 25 | 50 |
| One inter-layer refresh | 0 | 5 | 0 | 5 | 0 | 0 | 5 | 10 |
| Observed two-layer cumulative | 12600 | 103775 | 190 | 1625 | 110 | 2300 | 55 | 110 |
| Projected 12-layer total | 75600 | 622675 | 1140 | 9775 | 660 | 13800 | 355 | 710 |

The client validates every encrypted layer input and output against the corresponding
chained frozen-polynomial oracle and also gates the exact-trace diagnostic. Every such
gate requires rel-L2 `<= 5e-2`, cosine `>= 0.99`, and finite values. The client decrypts
the Softmax shifted logits and denominator, both LayerNorm normalized variances, and all
GELU inputs and requires them to remain inside the M3 frozen polynomial intervals.
Each layer must expose exactly 28 unique zero-valued encoded logical checkpoints. The
16 base labels are `encoder_input`, `query`, `key`, `value`, `scaled_scores`,
`shifted_logits`, `probabilities`, the attention output before and after bootstrap
cleanup, `self_projection`, `attention_residual`, `attention_layernorm`,
`output_projection`, the output residual before and after bootstrap, and
`encoder_output`. Each of the three FFN blocks additionally contributes
`intermediate_pre_activation_i`, `intermediate_polynomial_output_i`,
`intermediate_activation_i`, and `output_contribution_i`. Missing, extra, or duplicate
labels fail closed. Under the user-authorized
`moai_observability_compatible_prototype_v1` contract, their encoded
inactive/cross-lane maximum absolute value must be `<= 1e-3`. This threshold applies
only to M5 exact-three and full-12 runtime hygiene; M2 packing, M3 nonlinear, and M4
single-layer gates remain at `1e-6`. Legacy MOAI did not assert or report an inactive
threshold, so `1e-3` is a new prototype engineering bound rather than a claimed MOAI
value or strict numerical-parity criterion. The 16 base tensors and three
`output_contribution_i` tensors freeze active
width 768; the other nine intermediate tensors freeze active width 1024. A wrong width
fails closed. The nine full-width tensors have an empty encoded inactive tail, so their
presence proves registry completeness but is not evidence about physical CKKS slots
beyond the 1024 encoded values. This contract does not claim inspection of the unused
remainder of the 32768-slot ring capacity.

The Softmax denominator and the two LayerNorm normalized-variance tensors are three
separate public sentinel channels. Their actual inactive values must remain finite and
inside the corresponding frozen reciprocal or inverse-square-root interval. Deviation
from one is recorded as a diagnostic maximum and is not required to satisfy the
zero-inactive M5 prototype threshold. For both LayerNorm sites, the active public epsilon is
multiplied by the selected public trace-scale factor `D` and added before native
bootstrap. The variance branch is preconditioned by 2048: active slots carry `u/2048`,
and inactive guard slots carry `1/2048`. After bootstrap, one uniform public
multiplication restores all 1024 slots by 2048 before inverse-square-root evaluation.

After the inverse-square-root polynomial, one public first-768-active/last-256-zero
Ct-Pt mask and one explicit `ServerRuntime::Rescale` clean the inverse branch before the
centered Ct-Ct merge. The centered branch receives the per-token public
`gamma*sqrt(D)` Ct-Pt factor. The post-bootstrap required depth is 12. This is reflected
in the candidate per-layer count of 51885 Ct-Pt multiplications and 810 explicit
rescale requests. Weighted-V cleanup and each of the 11 inter-layer refreshes still use
one public 768-prefix mask. The restored LayerNorm inactive guard has only the finite
`[0.5,1536]` interval hard gate; deviation from one is diagnostic. The final LayerNorm
output inactive tail remains part of the encoded zero-valued M5 prototype gate.

The plaintext preflight proves only fixture/oracle consistency and negative API
contracts. Metadata calibration and every shortened exact-prefix mode are explicitly
marked `artifact_eligible=false`; their stdout cannot be mixed into a formal artifact.
The registered `openfhe_encoder_3_layer_exact_smoke` CTest is the minimum schedule
sealing diagnostic because it validates the second handoff and the later steady-state
regime. Exact one and exact two always report `formal_schedule_sealed=false`; exact three
may report true only in its successful final summary. The approved exact-three run above
has passed and is bound by the profile. The live M4 regression and formal 12-layer
prototype CTest remain independent required gates.

```bash
ctest --test-dir build-openfhe --output-on-failure \
  -R '^openfhe_encoder_3_layer_exact_smoke$'
```

The v6 runner/schema/validator code contract is aligned with the sealed tuple/counts,
the exact-three evidence binding, and the M5 prototype threshold. The exact-three
prerequisite is satisfied; M5 still lacks milestone evidence until live M4 and full
12-layer prototype validation pass, the milestone commit is pushed, and the local,
tracking, and live remote SHAs agree. The M5 entry point is:

```bash
/home/shawnsheep/miniconda3/envs/fhe-inference/bin/python3.10 \
  scripts/run_openfhe_encoder12_artifact_v6.py
```

The versioned v6 M5 runner performs one untimed-claim prototype execution, binds all
frozen inputs plus the executable by SHA-256 before and after the run, requires 12
position-specific metadata/count records, and invokes the independent schema and
semantic validator. Those artifact-side contracts are aligned, but no passing
full12-plus-validator milestone artifact has yet been produced. The M4 intermediate
gates must still pass before the full ciphertext chain starts; the 12-layer output gate
is not a substitute for that regression. The runner must emit `timing_claim=false`. A shortened diagnostic,
calibration run, dirty-tree run, remote-SHA mismatch, input or binary hash drift, or any
threshold/schema failure is not M5 evidence.
