#pragma once

#include "moai/openfhe/encoder_layer.hpp"

#include <cstddef>
#include <filesystem>
#include <vector>

namespace moai::openfhe::test {

using PlainMatrix = std::vector<std::vector<double>>;

struct EncoderTraceExpectations {
    PlainMatrix input;
    PlainMatrix query;
    PlainMatrix key;
    PlainMatrix value;
    PlainMatrix scaled_scores;
    PlainMatrix probabilities;
    PlainMatrix attention;
    PlainMatrix self_projection;
    PlainMatrix attention_residual;
    PlainMatrix attention_layernorm;
    PlainMatrix intermediate_pre_activation;
    PlainMatrix intermediate_activation;
    PlainMatrix output_projection;
    PlainMatrix output_residual;
    PlainMatrix output;
};

struct EncoderLayerFixture {
    EncoderLayerWeights weights;
    EncoderTraceExpectations expected;
};

// Reads only public checked-in trace/model CSVs. It is linked to client-side
// test executables, never to the server library.
[[nodiscard]] EncoderLayerFixture LoadEncoderLayerFixture(
    const std::filesystem::path& data_root,
    std::size_t layer_index);

[[nodiscard]] PlainMatrix PadFeatureRows(
    const PlainMatrix& input,
    std::size_t active_features,
    std::size_t block_dimension = kPaperCompatFeatureBlock);

[[nodiscard]] PlainMatrix SliceIntermediateBlock(
    const PlainMatrix& input,
    std::size_t block_index);

}  // namespace moai::openfhe::test
