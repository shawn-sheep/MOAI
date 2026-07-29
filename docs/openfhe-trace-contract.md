# MOAI five-token FFN trace contract

## Decision scope

This contract makes one narrow claim: the 12 checked-in five-row FFN trace fixtures
are internally consistent with the declared per-channel scaling reconstruction. It
does not validate OpenFHE execution, encrypted correctness, attention, a complete
encoder, tokenizer/classifier behavior, or production inputs.

The channel scales in `config/moai_trace_channel_scales.json` were reverse-engineered
only from the existing five-token trace. Runtime inference of scales is forbidden,
and production generalization is explicitly forbidden. `paper_compat` remains
`security_claim=none`.

## Contracted files

Each `data/layer_0` through `data/layer_11` directory contributes 13 files: the
embedded layer input, FFN input, both dense weights and biases, intermediate and
final trace checkpoints, the residual operand and pre-LayerNorm sum, and the real
layer output. The manifest fixes every relative path, shape, and SHA-256 digest.
Other trace files are outside this validator's evidence surface.

The validator reads the existing data tree in place through `--data-root`; it never
copies the approximately 1.1 GB data directory.

## Reconstructed FFN relations

For layer l, let X be the 5x768 FFN input, W1 and b1 the stored intermediate
parameters, W2 and b2 the stored output parameters, and s_l the 3072-element
scale vector declared by the manifest. Its default is one, with sparse integer
overrides. With row-oriented CSV tensors, the fixture relations are:

```text
Z_l = (X W1^T + b1) * s_l
H_l = 0.5 Z_l (1 + erf(Z_l / sqrt(2)))
Y_l = H_l (W2 / s_l)^T + b2
B_l = Y_l + residual_l
```

Here `*` and `/` are per-channel broadcast operations. The validator also checks
that `H_l` matches `final_output_inputs`, and requires each layer's
`real_final_output` to be byte-identical to the next layer's `embedded_inputs`.

The equations describe the checked-in fixture reconstruction; they are not a claim
that the sparse scales are a model-wide rule or can be recovered for new inputs.

## Validation

Run with Python 3.10 or 3.11 and NumPy:

```bash
python3 scripts/validate_moai_trace_contract.py \
  --data-root /home/shawnsheep/MOAI/data
```

A passing run prints per-layer maximum absolute errors, all 11 cross-layer checks,
global maxima, and:

```text
DECISION=PASS_TRACE_PARITY_FIXTURE
```

The command fails closed on a missing contracted file, shape mismatch, NaN/Inf,
SHA-256 mismatch, altered contract flags or thresholds, missing/tampered scale,
broken FFN relation, residual mismatch, or broken layer chain. A failure prints
`DECISION=FAIL_TRACE_PARITY_FIXTURE` and exits nonzero.
