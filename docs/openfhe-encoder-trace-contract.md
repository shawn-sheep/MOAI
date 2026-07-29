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
plaintext activation. The sealed `attention_output` artifact checkpoint is the
raw weighted-V value before its native bootstrap and public-prefix cleanup,
matching the trace boundary consumed by the self projection. The client also
decrypts and gates the cleaned copy in the same run; it is not aliased over or
used to hide the raw checkpoint. The remaining sealed checkpoints cover the
self-attention LayerNorm output, FFN output, and final encoder output.

The final encrypted output is compared to the single-layer frozen-polynomial
oracle, not directly to the exact BERT trace. Exact-trace parity remains a
diagnostic only. The hard gates are `rel-L2 <= 1e-2`, `cosine >= 0.999`, and
inactive/cross-lane maximum absolute value `<= 1e-6`; none may be relaxed. The
frozen schedule is input level 29, output level 29, 17 remaining levels, a
maximum observed level of 45, 25 ciphertext bootstraps, and 50 bootstrap
iterations. The four client-decrypted artifact checkpoints freeze
`(level, noise-scale degree, remaining levels)` as `attention_output=(40,2,6)`,
`self_layernorm_output=(32,2,14)`, `ffn_output=(45,2,1)`, and
`encoder_output=(29,2,17)`. Every checkpoint has `scale_bits=100`: the runtime
requires finite `log2(scale)` within `1e-3` of 100, then emits the deterministic
integer 100 rather than preserving floating-point noise. Their canonical
metadata SHA-256 is
`c4c1c85e52154215784b9fa93a584d824dde94a644e5af997882486aac01a6db`.
Its exact one-layer operation contract is 6300 rotations, 51865 Ct-Pt
multiplications, 95 Ct-Ct
multiplications, 800 explicit `ServerRuntime::Rescale` requests, 55 Chebyshev
evaluations, and 1150 estimated polynomial multiplications. Under the frozen
`FLEXIBLEAUTO` scaling technique, an explicit rescale request is not claimed to
be a physical modulus reduction.

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

After the milestone commit is pushed and the local and remote SHAs match, seal
the evidence with:

```bash
/home/shawnsheep/miniconda3/envs/fhe-inference/bin/python3.10 \
  scripts/run_openfhe_encoder_artifact.py
```

The runner is intentionally M4-only. It fixes the repository data and build
paths, verifies all 37 layer-1 CSV hashes against this trace contract, performs a
clean-first build, verifies the OpenFHE 1.5.1 package/linkage, reruns the fast
contracts, then executes one warm-up plus five measured runs. The fixed binary
and all 41 frozen inputs are rehashed before and after every execution. Any dirty tree,
remote SHA mismatch, binary drift, failed child process, threshold violation, or
artifact-validator rejection removes the unvalidated run directory and returns
nonzero.
