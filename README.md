# MOAI

## OpenFHE CPU migration

The default build is the staged OpenFHE v1.5.1 CPU migration. It currently provides the
M2 client/server runtime boundary, full 32768-slot interleaved packing for 256 distinct
lanes, and OpenFHE column/diagonal linear kernels. The only accepted parameter profile
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

The native CKKS bootstrap API/level-refresh smoke is deliberately separate and uses an
explicit test-only 8-slot encoding. It is not a 32768-slot workload result:

```bash
ctest --test-dir build-openfhe --output-on-failure -R openfhe_bootstrap_smoke
```

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
