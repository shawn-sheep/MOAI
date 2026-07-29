# MOAI OpenFHE CPU migration contract

## Scope

The target is a server-only, 12-layer encoder trace replay for the existing
BERT-base-shaped MOAI workload. This is not a tokenizer-to-classifier application and
must not be described as task-level end-to-end private inference.

Only ordinary OpenFHE CKKS is in scope. Discrete CKKS, functional bootstrapping, QDQ,
integer quantization, GPU execution, and MOAI_GPU are out of scope.

## Security posture

The only supported profile is `paper_compat`. It has `security_claim=none` and is for
research reproduction only. Every executable and run manifest must surface that warning.
No result from this profile supports a 128-bit security claim.

## Trust boundary

- The client owns plaintext activations and the OpenFHE `PrivateKey`.
- The server receives ciphertexts, plaintext model weights, the public key, and only the
  evaluation keys needed by the declared operator graph.
- Server libraries and operator signatures must not contain `PrivateKey` or `Decryptor`.
- A test server may return encrypted checkpoints. Decryption and comparison remain client
  responsibilities and cannot feed plaintext back into server computation.

## Gates

| Gate | Requirement |
|---|---|
| Packing | Logical slot ownership is exact; inactive and cross-lane absolute error <= 1e-6. |
| Linear | Relative L2 <= 1e-4 and cosine similarity >= 0.99999. |
| Nonlinear / one layer | Relative L2 against the same polynomial oracle <= 1e-2 and cosine >= 0.999. |
| Twelve layers | Final relative L2 <= 5e-2, cosine >= 0.99, and no NaN/Inf at any checkpoint. |

Correctness precedes timing. A benchmark uses one warm-up and five measured repetitions,
reports median plus MAD/range, and separates setup/key generation, client work, and server
online work. Thresholds may be tightened; relaxing one requires a reviewed contract change.

## Evidence boundary at M0

The repository contains source entrypoints for basic CKKS, packing, linear operators, a
single layer, and a 12-layer graph. M0 does not execute the clean-tree full scheme. The
legacy full path uses disabled security checks, carries secret-key material into operator
calls, and performs diagnostic decryption, so source presence is not runtime or
server-only validation.
