#pragma once

#include "support/moai_encoder_fixture.hpp"

namespace moai::openfhe::test {

struct PlaintextOracleRange {
    double minimum{0.0};
    double maximum{0.0};
};

struct EncoderPlaintextOracleRanges {
    PlaintextOracleRange softmax_shifted_logits;
    PlaintextOracleRange softmax_denominator;
    PlaintextOracleRange attention_layernorm_normalized_variance;
    PlaintextOracleRange gelu_input;
    PlaintextOracleRange output_layernorm_normalized_variance;
};

struct EncoderPlaintextOracleResult {
    // Five rows of one complete feature block. The first 768 values are the
    // logical BERT-base output and the remaining lanes are exactly zero.
    PlainMatrix output;
    EncoderPlaintextOracleRanges ranges;
};

// Replays one frozen paper_compat encoder layer without CKKS noise. The public
// weights are input-major and already contain the channel-scale transform used
// by the encrypted three-block FFN. Every polynomial input is checked against
// the registry interval before evaluation; clipping and range repair are
// deliberately forbidden.
[[nodiscard]] EncoderPlaintextOracleResult
EvaluateEncoderLayerPlaintextOracle(
    const PlainMatrix& input,
    const EncoderLayerWeights& weights);

}  // namespace moai::openfhe::test
