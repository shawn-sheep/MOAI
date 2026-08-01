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
- `ClientRuntime` constructs the OpenFHE context from the validated effective profile;
  callers cannot inject a raw context. The opaque evaluation-key bundle carries the
  effective-profile hash, and `ServerRuntime` revalidates both that binding and the
  observable CKKS context parameters before accepting work.
- Softmax may receive one offline-frozen public scalar per layer/head as model/fixture
  metadata. A scalar or vector derived from a runtime plaintext activation is forbidden.
- Approximation intervals are registry-bound. Observed trace ranges are checked before
  encryption by the offline/client validator and are never passed into a server API as
  activation-derived metadata.
- Server libraries and operator signatures must not contain `PrivateKey` or `Decryptor`.
- A test server may return encrypted checkpoints. Decryption and comparison remain client
  responsibilities and cannot feed plaintext back into server computation.

For contiguous and column tensors, `PackingSpec.logical_shape` is `{logical_rows,
features}` and `features` equals the ciphertext count. A reduction over feature
ciphertexts produces `{logical_rows,1}`. Add/Sub may broadcast only such an explicit
one-feature right operand. Elementwise Multiply uses the same rhs broadcast; its only
cross-layout exception is the explicitly validated diagonal-singleton times
column-singleton step in the BSGS linear kernel. Level and scale metadata must always
reflect the underlying OpenFHE ciphertexts.

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

## M5 runnable-prototype amendment (2026-08-01)

The user explicitly authorized a correctness-contract revision that prioritizes a runnable
MOAI-aligned prototype. Only the M5 exact-three and full-12 encoder-runtime inactive/cross-lane
gate changes from `1e-6` to `1e-3`. The active per-layer and final rel-L2/cosine gates,
finite/range checks, server-only boundary, and all M2/M3/M4 gates remain unchanged.

Legacy MOAI does not assert or report an inactive/cross-lane threshold, so `1e-3` is a new
engineering bound that is stricter than legacy observability. It must be labelled
`prototype_only` and `MOAI-observability-compatible`; it is not a claim that legacy MOAI used
`1e-3`, and it does not establish strict SEAL/OpenFHE numerical parity.

## Evidence boundary at M0

The repository contains source entrypoints for basic CKKS, packing, linear operators, a
single layer, and a 12-layer graph. M0 does not execute the clean-tree full scheme. The
legacy full path uses disabled security checks, carries secret-key material into operator
calls, and performs diagnostic decryption, so source presence is not runtime or
server-only validation.
