# MOAI

## OpenFHE CPU migration

The default build is the staged OpenFHE v1.5.1 CPU migration. It currently provides an
M4 server-only replay of layer 1 from the frozen five-token BERT-base encoder trace, in
addition to the M1-M3 client/server runtime, packing, linear-kernel, nonlinear, and
native-bootstrap gates. Each token uses one 1024-slot ciphertext. The client owns the
private key and decrypts the gated checkpoints; the server receives only ciphertexts,
public model weights, and an evaluation-key bundle. The only accepted parameter profile
is `paper_compat`, which always reports `security_claim=none`. These research
reproduction parameters do not support a 128-bit security claim.

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
takes about 31 minutes. This is a correctness observation, not an M6 benchmark. After
the milestone commit has been pushed and the local and remote branch SHAs match, seal
one warm-up plus five measured runs with:

```bash
/home/shawnsheep/miniconda3/envs/fhe-inference/bin/python3.10 \
  scripts/run_openfhe_encoder_artifact.py
```

M4 does not establish 12-layer ciphertext execution, task-level inference, or a
speedup over the optional SEAL reference.

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
