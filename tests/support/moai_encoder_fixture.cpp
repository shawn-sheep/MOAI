#include "support/moai_encoder_fixture.hpp"

#include "utils/hashutil.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace moai::openfhe::test {
namespace {

constexpr const char* kScaleContractFileSha256 =
    "d6b47766e893bec002abc8f07915090ab2365a6b0f59eb5509f1a63d9c954521";

using ScaleOverride = std::pair<std::size_t, double>;

const std::array<std::vector<ScaleOverride>, kPaperCompatEncoderLayers>&
ScaleOverrides() {
    static const std::array<std::vector<ScaleOverride>,
                            kPaperCompatEncoderLayers> overrides{{
        {},
        {},
        {{1072, 6.0}, {1642, 2.0}},
        {{565, 2.0},
         {838, 2.0},
         {910, 2.0},
         {1131, 2.0},
         {1378, 2.0},
         {1693, 2.0},
         {1793, 5.0},
         {2217, 2.0},
         {2267, 2.0},
         {3046, 3.0}},
        {{59, 2.0},
         {116, 2.0},
         {184, 3.0},
         {747, 2.0},
         {867, 2.0},
         {884, 2.0},
         {983, 4.0},
         {1449, 2.0},
         {1843, 4.0},
         {1936, 2.0},
         {2856, 5.0}},
        {{1360, 2.0},
         {1517, 4.0},
         {1754, 2.0},
         {2577, 2.0},
         {2739, 4.0},
         {2837, 2.0}},
        {{313, 2.0}, {2089, 2.0}},
        {{294, 2.0}, {520, 2.0}},
        {},
        {{2184, 3.0}, {2189, 4.0}, {2958, 3.0}},
        {{1046, 15.0}, {1942, 3.0}},
        {},
    }};
    return overrides;
}

std::string ReadText(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open fixture file: " + path.string());
    }
    return {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
}

void VerifyScaleContract(const std::filesystem::path& data_root) {
    const auto path =
        data_root.parent_path() / "config" / "moai_trace_channel_scales.json";
    if (lbcrypto::HashUtil::HashString(ReadText(path)) !=
        kScaleContractFileSha256) {
        throw std::runtime_error(
            "five-token FFN channel-scale contract SHA-256 drifted");
    }
}

std::vector<double> ParseCsvRow(
    const std::string& line,
    const std::filesystem::path& path,
    std::size_t line_number) {
    std::string normalized = line;
    std::replace(normalized.begin(), normalized.end(), ',', ' ');
    std::istringstream stream(normalized);
    std::vector<double> values;
    double value = 0.0;
    while (stream >> value) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "non-finite CSV value in " + path.string());
        }
        values.push_back(value);
    }
    if (!stream.eof() || values.empty()) {
        throw std::runtime_error(
            "malformed CSV row " + std::to_string(line_number) +
            " in " + path.string());
    }
    return values;
}

PlainMatrix ReadMatrix(
    const std::filesystem::path& path,
    std::size_t rows,
    std::size_t columns) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open fixture CSV: " + path.string());
    }
    PlainMatrix matrix;
    matrix.reserve(rows);
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            throw std::runtime_error("empty CSV row in " + path.string());
        }
        auto values = ParseCsvRow(line, path, matrix.size() + 1);
        if (values.size() != columns) {
            throw std::runtime_error(
                "CSV column count drifted in " + path.string());
        }
        matrix.push_back(std::move(values));
    }
    if (matrix.size() != rows) {
        throw std::runtime_error("CSV row count drifted in " + path.string());
    }
    return matrix;
}

std::vector<double> ReadVector(
    const std::filesystem::path& path,
    std::size_t elements) {
    const auto rows = ReadMatrix(path, elements, 1);
    std::vector<double> values;
    values.reserve(elements);
    for (const auto& row : rows) {
        values.push_back(row.front());
    }
    return values;
}

FeaturePackedWeights TransposeOutputByInput(
    const PlainMatrix& output_by_input) {
    if (output_by_input.empty() || output_by_input.front().empty()) {
        throw std::invalid_argument("cannot transpose an empty weight matrix");
    }
    const std::size_t outputs = output_by_input.size();
    const std::size_t inputs = output_by_input.front().size();
    FeaturePackedWeights input_by_output(
        inputs,
        std::vector<double>(outputs));
    for (std::size_t output = 0; output < outputs; ++output) {
        if (output_by_input[output].size() != inputs) {
            throw std::invalid_argument("weight matrix is ragged");
        }
        for (std::size_t input = 0; input < inputs; ++input) {
            input_by_output[input][output] = output_by_input[output][input];
        }
    }
    return input_by_output;
}

std::vector<double> ChannelScales(std::size_t layer_index) {
    if (layer_index >= kPaperCompatEncoderLayers) {
        throw std::out_of_range("channel-scale layer index is out of range");
    }
    std::vector<double> scales(kPaperCompatIntermediateSize, 1.0);
    for (const auto& [channel, scale] : ScaleOverrides()[layer_index]) {
        if (channel >= scales.size() || !std::isfinite(scale) || scale <= 0.0) {
            throw std::logic_error("compiled channel-scale registry is invalid");
        }
        scales[channel] = scale;
    }
    return scales;
}

std::filesystem::path LayerRoot(
    const std::filesystem::path& data_root,
    std::size_t layer_index) {
    if (layer_index >= kPaperCompatEncoderLayers) {
        throw std::out_of_range("fixture layer index is out of range");
    }
    return data_root / ("layer_" + std::to_string(layer_index));
}

}  // namespace

PlainMatrix PadFeatureRows(
    const PlainMatrix& input,
    std::size_t active_features,
    std::size_t block_dimension) {
    if (input.empty() || active_features == 0 ||
        active_features > block_dimension) {
        throw std::invalid_argument("feature-row padding dimensions are invalid");
    }
    PlainMatrix output(
        input.size(),
        std::vector<double>(block_dimension, 0.0));
    for (std::size_t row = 0; row < input.size(); ++row) {
        if (input[row].size() != active_features) {
            throw std::invalid_argument("feature-row padding input is ragged");
        }
        std::copy(input[row].begin(), input[row].end(), output[row].begin());
    }
    return output;
}

PlainMatrix SliceIntermediateBlock(
    const PlainMatrix& input,
    std::size_t block_index) {
    if (block_index >= kPaperCompatIntermediateBlocks || input.empty()) {
        throw std::invalid_argument("intermediate block index is invalid");
    }
    const std::size_t begin = block_index * kPaperCompatFeatureBlock;
    PlainMatrix output(
        input.size(),
        std::vector<double>(kPaperCompatFeatureBlock));
    for (std::size_t row = 0; row < input.size(); ++row) {
        if (input[row].size() != kPaperCompatIntermediateSize) {
            throw std::invalid_argument("intermediate trace matrix is ragged");
        }
        std::copy_n(
            input[row].begin() + static_cast<std::ptrdiff_t>(begin),
            kPaperCompatFeatureBlock,
            output[row].begin());
    }
    return output;
}

EncoderLayerFixture LoadEncoderLayerFixture(
    const std::filesystem::path& data_root,
    std::size_t layer_index) {
    VerifyScaleContract(data_root);
    const auto root = LayerRoot(data_root, layer_index);
    const auto attention_results =
        root / "Attention" / "BertSelfAttention" / "allresults";
    const auto attention_parameters =
        root / "Attention" / "BertSelfAttention" / "parms";
    const auto self_results =
        root / "Attention" / "SelfOutput" / "allresults";
    const auto self_parameters =
        root / "Attention" / "SelfOutput" / "parms";
    const auto intermediate_results = root / "Intermediate" / "allresults";
    const auto intermediate_parameters = root / "Intermediate" / "parms";
    const auto output_results = root / "Output" / "allresults";
    const auto output_parameters = root / "Output" / "parms";

    EncoderLayerFixture fixture;
    auto& weights = fixture.weights;
    weights.layer_index = layer_index;
    weights.query_weights = TransposeOutputByInput(ReadMatrix(
        attention_parameters / "query_weight.csv",
        kPaperCompatHiddenSize,
        kPaperCompatHiddenSize));
    weights.query_bias = ReadVector(
        attention_parameters / "query_bias.csv",
        kPaperCompatHiddenSize);
    weights.key_weights = TransposeOutputByInput(ReadMatrix(
        attention_parameters / "key_weight.csv",
        kPaperCompatHiddenSize,
        kPaperCompatHiddenSize));
    weights.key_bias = ReadVector(
        attention_parameters / "key_bias.csv",
        kPaperCompatHiddenSize);
    weights.value_weights = TransposeOutputByInput(ReadMatrix(
        attention_parameters / "value_weight.csv",
        kPaperCompatHiddenSize,
        kPaperCompatHiddenSize));
    weights.value_bias = ReadVector(
        attention_parameters / "value_bias.csv",
        kPaperCompatHiddenSize);
    weights.self_output_weights = TransposeOutputByInput(ReadMatrix(
        self_parameters / "self_output_dense_weight.csv",
        kPaperCompatHiddenSize,
        kPaperCompatHiddenSize));
    weights.self_output_bias = ReadVector(
        self_parameters / "self_output_dense_bias.csv",
        kPaperCompatHiddenSize);
    weights.attention_layernorm_gamma = ReadVector(
        self_parameters / "self_output_LayerNorm_weight.csv",
        kPaperCompatHiddenSize);
    weights.attention_layernorm_beta = ReadVector(
        self_parameters / "self_output_LayerNorm_bias.csv",
        kPaperCompatHiddenSize);

    const auto scales = ChannelScales(layer_index);
    const auto intermediate_output_by_input = ReadMatrix(
        intermediate_parameters / "intermediate_dense_weight.csv",
        kPaperCompatIntermediateSize,
        kPaperCompatHiddenSize);
    const auto intermediate_bias = ReadVector(
        intermediate_parameters / "intermediate_dense_bias.csv",
        kPaperCompatIntermediateSize);
    const auto output_output_by_input = ReadMatrix(
        output_parameters / "final_output_dense_weight.csv",
        kPaperCompatHiddenSize,
        kPaperCompatIntermediateSize);
    for (std::size_t block = 0;
         block < kPaperCompatIntermediateBlocks;
         ++block) {
        weights.intermediate_weight_blocks[block].assign(
            kPaperCompatHiddenSize,
            std::vector<double>(kPaperCompatFeatureBlock));
        weights.intermediate_bias_blocks[block].resize(
            kPaperCompatFeatureBlock);
        weights.output_weight_blocks[block].assign(
            kPaperCompatFeatureBlock,
            std::vector<double>(kPaperCompatHiddenSize));
        for (std::size_t local = 0;
             local < kPaperCompatFeatureBlock;
             ++local) {
            const std::size_t channel =
                block * kPaperCompatFeatureBlock + local;
            const double scale = scales[channel];
            weights.intermediate_bias_blocks[block][local] =
                intermediate_bias[channel] * scale;
            for (std::size_t input = 0;
                 input < kPaperCompatHiddenSize;
                 ++input) {
                weights.intermediate_weight_blocks[block][input][local] =
                    intermediate_output_by_input[channel][input] * scale;
            }
            for (std::size_t output = 0;
                 output < kPaperCompatHiddenSize;
                 ++output) {
                weights.output_weight_blocks[block][local][output] =
                    output_output_by_input[output][channel] / scale;
            }
        }
    }
    weights.output_bias = ReadVector(
        output_parameters / "final_output_dense_bias.csv",
        kPaperCompatHiddenSize);
    weights.output_layernorm_gamma = ReadVector(
        output_parameters / "final_output_LayerNorm_weight.csv",
        kPaperCompatHiddenSize);
    weights.output_layernorm_beta = ReadVector(
        output_parameters / "final_output_LayerNorm_bias.csv",
        kPaperCompatHiddenSize);

    auto& expected = fixture.expected;
    expected.input = PadFeatureRows(ReadMatrix(
        attention_results / "embedded_inputs.csv",
        kPaperCompatTraceTokens,
        kPaperCompatHiddenSize),
        kPaperCompatHiddenSize);
    expected.query = PadFeatureRows(ReadMatrix(
        attention_results / "Q.csv",
        kPaperCompatTraceTokens,
        kPaperCompatHiddenSize),
        kPaperCompatHiddenSize);
    expected.key = PadFeatureRows(ReadMatrix(
        attention_results / "K.csv",
        kPaperCompatTraceTokens,
        kPaperCompatHiddenSize),
        kPaperCompatHiddenSize);
    expected.value = PadFeatureRows(ReadMatrix(
        attention_results / "V.csv",
        kPaperCompatTraceTokens,
        kPaperCompatHiddenSize),
        kPaperCompatHiddenSize);
    expected.scaled_scores = ReadMatrix(
        attention_results / "QKT.csv",
        kPaperCompatTraceTokens,
        kPaperCompatAttentionHeadCount * kPaperCompatTraceTokens);
    expected.probabilities = ReadMatrix(
        attention_results / "aftsoftmax.csv",
        kPaperCompatTraceTokens,
        kPaperCompatAttentionHeadCount * kPaperCompatTraceTokens);
    expected.attention = PadFeatureRows(ReadMatrix(
        attention_results / "real_attention.csv",
        kPaperCompatTraceTokens,
        kPaperCompatHiddenSize),
        kPaperCompatHiddenSize);
    expected.self_projection = PadFeatureRows(ReadMatrix(
        self_results / "self_output_after_linear.csv",
        kPaperCompatTraceTokens,
        kPaperCompatHiddenSize),
        kPaperCompatHiddenSize);
    expected.attention_residual = PadFeatureRows(ReadMatrix(
        self_results / "self_output_residual_connection_before_layernorm.csv",
        kPaperCompatTraceTokens,
        kPaperCompatHiddenSize),
        kPaperCompatHiddenSize);
    expected.attention_layernorm = PadFeatureRows(ReadMatrix(
        self_results / "real_self_output.csv",
        kPaperCompatTraceTokens,
        kPaperCompatHiddenSize),
        kPaperCompatHiddenSize);
    expected.intermediate_pre_activation = ReadMatrix(
        intermediate_results / "intermediate_output_after_linear.csv",
        kPaperCompatTraceTokens,
        kPaperCompatIntermediateSize);
    expected.intermediate_activation = ReadMatrix(
        intermediate_results / "real_intermediate_output.csv",
        kPaperCompatTraceTokens,
        kPaperCompatIntermediateSize);
    expected.output_projection = PadFeatureRows(ReadMatrix(
        output_results / "final_output_after_linear.csv",
        kPaperCompatTraceTokens,
        kPaperCompatHiddenSize),
        kPaperCompatHiddenSize);
    expected.output_residual = PadFeatureRows(ReadMatrix(
        output_results / "final_output_residual_connection_before_layernorm.csv",
        kPaperCompatTraceTokens,
        kPaperCompatHiddenSize),
        kPaperCompatHiddenSize);
    expected.output = PadFeatureRows(ReadMatrix(
        output_results / "real_final_output.csv",
        kPaperCompatTraceTokens,
        kPaperCompatHiddenSize),
        kPaperCompatHiddenSize);
    return fixture;
}

}  // namespace moai::openfhe::test
