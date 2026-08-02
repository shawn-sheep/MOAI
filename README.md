# MOAI

## OpenFHE CPU migration

The default build is the OpenFHE v1.5.1 CPU backend. It includes the pushed M4
server-only replay of layer 1 and the completed M5 gate that chains all 12 layers of the
frozen five-token BERT-base encoder trace without plaintext activation resets. Each
token uses one 1024-slot ciphertext inside the full 32768-slot CKKS ring capacity. The
client owns the private key; the server receives only ciphertexts, public model weights,
and an evaluation-key bundle. The only accepted parameter profile is `paper_compat`,
which always reports `security_claim=none`. These research-reproduction parameters do
not support a 128-bit security claim.

The approved M5 v6 artifact is
`20260801T224242+0900-m5-runnable-prototype-v6-534f582-r27`, bound to pushed commit
`534f582a655669bafee9d9098cb54efbf66d2cd5`. It passed the formal client-validated
12-layer correctness gate at rel-L2 `0.0017283753946057928`, cosine
`0.9999985073522254`, and inactive maximum `3.415363197201149e-06`. M6 adds a distinct
single-sample timing contract; it does not promote the M5 single execution or shortened
schedule diagnostics into timing evidence.

Configure, build, and run the fast gates:

```bash
cmake -S . -B build-openfhe \
  -DOpenFHE_DIR=/home/shawnsheep/opt/openfhe_v1_5_1/lib/OpenFHE \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build build-openfhe -j
ctest --test-dir build-openfhe --output-on-failure -LE slow
```

The M2 gate uses all 32768 slots and 256 distinct interleaved lanes. It checks inactive
and cross-lane error at `1e-6`, then checks Ct-Pt column, Ct-Ct column-to-diagonal,
and Ct-Ct diagonal-column BSGS kernels at rel-L2 `1e-4` and cosine `0.99999`:

```bash
ctest --test-dir build-openfhe --output-on-failure \
  -R openfhe_packing_linear_smoke
```

The M3 nonlinear gate evaluates GELU, fixed-public-shift masked softmax, LayerNorm, and
a reduced two-layer FFN against the same frozen Chebyshev-series oracle. Softmax and
scaled LayerNorm variance use OpenFHE's native CKKS bootstrap; a depleted FFN fixture
also verifies the conditional pre-GELU bootstrap. Frozen usable-level budgets are 12
from GELU pre-activation through the final affine, 7/12 before/after the softmax
bootstrap, and 4/11 before/after the LayerNorm bootstrap. The smoke uses an
explicit test-only 16-slot encoding with four active and twelve masked inactive slots;
masked outputs plus every GELU checkpoint retain the 1e-6 inactive-slot hard gate.
Softmax resolves a single offline-calibrated public shift from a frozen 12-by-12
layer/head registry; callers cannot inject approximation contracts or activation-derived
ranges. The denominator intentionally uses identity values in inactive slots. The gate
also freezes the reduced graph's rotations, ciphertext multiplications, rescale,
bootstrap, Chebyshev-evaluation, and polynomial-depth counts. This is not a full
32768-slot nonlinear workload result:

```bash
ctest --test-dir build-openfhe --output-on-failure \
  -R openfhe_nonlinear_smoke
```

The frozen plaintext approximation and five-token trace contracts are also registered
with CTest. Their Python interpreter and data root are configurable:

```bash
cmake -S . -B build-openfhe \
  -DMOAI_PYTHON_EXECUTABLE=/path/to/python3.10 \
  -DMOAI_TRACE_DATA_ROOT=/path/to/MOAI/data
ctest --test-dir build-openfhe --output-on-failure \
  -R 'openfhe_nonlinear_contract|openfhe_artifact_schema_contract|moai_trace_contract'
```

The native CKKS bootstrap API/level-refresh smoke is deliberately separate and uses an
explicit test-only 8-slot encoding. It is not a 32768-slot workload result:

```bash
ctest --test-dir build-openfhe --output-on-failure -R openfhe_bootstrap_smoke
```

The M4 full-layer correctness gate replays only layer 1 of the fixed five-token trace.
It validates raw and bootstrap-cleaned attention checkpoints, both LayerNorm sites, the
three-block FFN, the final polynomial-oracle output, inactive slots, exact operation
counts, and level/scale metadata. The server target contains no private key, decryptor,
or plaintext activation path:

```bash
ctest --test-dir build-openfhe --output-on-failure \
  -R 'openfhe_(profile_contract|profile_validator_contract|encoder_fixture_contract|encoder_plaintext_oracle_smoke)'
./build-openfhe/openfhe_encoder_layer_smoke \
  --data-root data --layer 1 --input-level 29
```

On the development host the full correctness gate uses roughly 31 GiB peak RSS and
takes about 31 minutes. This is a correctness observation, not M6 timing evidence. The M4
live regression evidence is already part of the validated M5 prerequisite chain. Its
runner is:

```bash
/home/shawnsheep/miniconda3/envs/fhe-inference/bin/python3.10 \
  scripts/run_openfhe_encoder_artifact_v5.py
```

The historical v2 contract remains byte-preserved. Any M4 artifact establishes only the
one-layer result; it does not by itself establish 12-layer ciphertext execution,
task-level inference, or a speedup over the optional SEAL reference.

Both active artifact runners discard the ignored build-tree configuration with a fixed
`/usr/bin/cmake --fresh` configure before their clean-first build. Configure, build,
CTest, linkage inspection, workload, and validator subprocesses share one scrubbed
environment. The manifest binds the resulting `CMakeCache.txt`, four OpenFHE CMake
package files, the canonical `include/openfhe` tree, and the exact three resolved
OpenFHE 1.5.1 shared libraries by path, byte count, and SHA-256. Any byte or linkage
drift before or after a workload fails closed.

The M5 target consumes the client-encrypted layer-0 input once, evaluates all 12 ordered
public weight sets, and passes each layer output to the next only after the frozen native
bootstrap plus public 768-prefix mask. The integration executable contains a client
observer solely to decrypt and validate ciphertext checkpoints. The server library and
API do not receive a private key, a decryptor, or plaintext activations; this is an API
trust boundary, not a claim of process isolation.

Each FeaturePacked LayerNorm site selects the public trace-scale factor `D`, adds the
public active epsilon, and preconditions the variance by 2048 before native bootstrap:
active slots carry `u/2048` and inactive guard slots carry `1/2048`. One uniform public
multiplication restores all 1024 slots by 2048 before inverse-square-root evaluation.
After that polynomial, one public 768-prefix Ct-Pt mask plus an explicit rescale cleans
the inverse branch before the centered Ct-Ct merge; the centered branch receives the
per-token public `gamma*sqrt(D)` factor. The post-bootstrap required depth is 12.
Restored inactive guards need only be finite and inside `[0.5,1536]`; deviation from one
is diagnostic. The standalone FeaturePacked and M4 single-layer output tail retains the
hard `1e-6` gate; only M5 exact-three/full-12 uses the top-level `1e-3` prototype gate.
The candidate per-layer operation tuple is
`6300/51885/95/810/55/1150/25/50`. This does not change the single-mask weighted-V
cleanup or inter-layer refresh contract.

Run the fail-closed contracts and the focused LayerNorm HE gate first:

```bash
ctest --test-dir build-openfhe --output-on-failure \
  -R 'openfhe_(feature_layernorm_smoke|m5_v6_artifact_(schema|validator)_contract|m5_v6_encoder12_artifact_runner_contract|encoder_12_layer_(preflight|crypto_preflight))'
```

The registered exact-three diagnostic validates both layer-0-to-layer-1 and
layer-1-to-layer-2 handoffs, including the later-layer steady-state tuple. Exact one- and
two-layer diagnostics can never emit `formal_schedule_sealed=true`; exact three may do so
only in its successful final summary. Every shortened run remains
`artifact_eligible=false` and cannot replace the full gate:

```bash
ctest --test-dir build-openfhe --output-on-failure \
  -R '^openfhe_encoder_3_layer_exact_smoke$'
```

The runnable-prototype correctness gate evaluates all 12 ciphertext layers, checks every
layer against the chained frozen-polynomial oracle and exact-trace diagnostic, validates every
encrypted polynomial input range on the client, and requires exactly 28 uniquely named
zero-valued encoded logical checkpoint tensors per layer to pass the inactive/cross-lane
`1e-3` M5 prototype gate. This is a user-authorized engineering bound: legacy MOAI did
not assert inactive values, so it is not a claimed legacy threshold or strict numerical
parity. M2/M3/M4 keep their existing `1e-6` gates. Label-to-width mapping is also frozen:
19 checkpoints expose a 768-active/
256-inactive tail, while nine full-width intermediate checkpoints have no encoded
inactive tail. This gate covers the 1024 explicitly encoded slots, not the unused
remainder of the 32768-slot ring capacity. The three public sentinel channels are
separate: all must stay in their frozen approximation intervals. For LayerNorm, restored
inactive-guard deviation from one is diagnostic only; the downstream output inactive
tail remains covered by the M5 `1e-3` prototype gate. The Softmax denominator deviation also
remains diagnostic. Final acceptance bounds are rel-L2
`<= 5e-2`, cosine `>= 0.99`, and no NaN/Inf. It is a
very-slow correctness test and does not make a latency claim:

```bash
ctest --test-dir build-openfhe --output-on-failure \
  -R '^openfhe_encoder_12_layer_smoke$'
```

The exact-three schedule prerequisite, live M4 regression, and formal 12-layer gate are
satisfied and bound into the approved r27 evidence. The non-benchmark correctness runner
is retained for reproducibility:

```bash
/home/shawnsheep/miniconda3/envs/fhe-inference/bin/python3.10 \
  scripts/run_openfhe_encoder12_artifact_v6.py
```

The M5 v6 runner rehashes the executable and frozen inputs, requires the
layer-1/level-29 M4 regression, requires 12 ordered layer records and 11 inter-layer
refreshes, rejects diagnostic or calibration stdout, and delegates the sealed directory
to the independent schema and semantic validator. M5 still excludes task-level
inference and any SEAL speedup claim.

M6 measures the same five-token, 12-layer OpenFHE workload with checkpoint observation
disabled inside the server-online window. Every execution still performs a client-owned
final decryption and correctness gate. By explicit user amendment, the prototype harness
uses one discarded warm-up and exactly one measured execution. It reports the observed
phase and wall latency, batch and five-token amortized latency, process high-water RSS,
and OpenFHE BINARY archive component bytes for keys and ciphertexts. Compatibility
summary fields have median/minimum/maximum equal to the one observation and MAD equal to
zero; they do not estimate repeatability or dispersion. Size measurement is outside
server-online timing and writes no key material:

```bash
/home/shawnsheep/miniconda3/envs/fhe-inference/bin/python3.10 \
  scripts/run_openfhe_m6_benchmark.py
```

The full 1+1 protocol is expected to take roughly 14–15 hours on the development host,
depending on host load. The runner requires a clean pushed commit with matching local,
tracking, and live-remote SHAs, and binds the approved M5 r27 artifact before either
execution. Legacy SEAL uses a different parameter/packing/precision/workload contract,
so it is recorded as non-comparable and no speedup is calculated.

## Optional legacy SEAL reference

Install dependencies and the vendored SEAL fork:

```
sudo apt update
sudo apt install cmake g++ git libntl-dev libssl-dev libgmp-dev pkg-config
cd thirdparty/SEAL-4.1-bs
cmake -S . -B build
cmake --build build
sudo cmake --install build
```

Then enable the reference target explicitly:

```bash
cmake -S . -B build-seal -DMOAI_ENABLE_SEAL_REFERENCE=ON
cmake --build build-seal --target moai_seal_reference -j
```

The legacy default entrypoint includes the full source graph and is not registered as an
automated correctness test. Its historical output reports total packed-batch time for a
capacity of 256 inputs; that capacity must not be described as 256 distinct measured
samples without a nonzero multi-lane fixture.

## Citation

```
@misc{cryptoeprint:2025/991,
      author = {Linru Zhang and Xiangning Wang and Jun Jie Sim and Zhicong Huang and Jiahao Zhong and Huaxiong Wang and Pu Duan and Kwok Yan Lam},
      title = {{MOAI}: Module-Optimizing Architecture for Non-Interactive Secure Transformer Inference},
      howpublished = {Cryptology {ePrint} Archive, Paper 2025/991},
      year = {2025},
      url = {https://eprint.iacr.org/2025/991}
}
```
