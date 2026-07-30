#include "moai/openfhe/approximation_registry.hpp"
#include "moai/openfhe/client_runtime.hpp"
#include "moai/openfhe/context_factory.hpp"
#include "moai/openfhe/encoder_layer.hpp"
#include "support/moai_encoder_fixture.hpp"
#include "support/moai_encoder_plaintext_oracle.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using moai::openfhe::CipherTensor;
using moai::openfhe::EncoderLayerCiphertextTrace;
using moai::openfhe::RunMetrics;
using moai::openfhe::test::EncoderPlaintextOracleRanges;
using moai::openfhe::test::PlainMatrix;

constexpr const char* kExpectedProfileSha256 =
    "94f30e628e21f02146ce7ed9820194eabba3820f6e1e17176a31f8c5acf8b0be";
constexpr double kSingleScaleBits = 50.0;
constexpr double kDoubleScaleBits = 100.0;
constexpr double kScaleBitsTolerance = 1e-3;
constexpr double kPerLayerRelativeL2Maximum = 5e-2;
constexpr double kPerLayerCosineMinimum = 0.99;
constexpr double kInactiveMaximum = 1e-6;
constexpr uint32_t kMetadataMultiplicativeDepth = 47;
constexpr std::size_t kInactiveZeroCheckpointCount = 28;

enum class MetadataMode {
    kExact,
    kCalibration,
};

enum class RunKind {
    kFormalFull,
    kExactPrefix,
    kMetadataCalibration,
};

struct Arguments {
    std::filesystem::path data_root{"data"};
    bool preflight_only{false};
    bool crypto_preflight_only{false};
    std::optional<std::size_t> requested_layer_count;
    bool metadata_calibration{false};
};

struct QualityMetrics {
    double relative_l2{0.0};
    double cosine{0.0};
    double maximum_absolute{0.0};
};

struct ObservedRange {
    double minimum{std::numeric_limits<double>::infinity()};
    double maximum{-std::numeric_limits<double>::infinity()};

    void Observe(double value, const char* label) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(std::string(label) + " contains NaN or Inf");
        }
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
};

struct OperationCounts {
    uint64_t rotations{0};
    uint64_t ct_pt_multiplications{0};
    uint64_t ct_ct_multiplications{0};
    uint64_t explicit_rescale_requests{0};
    uint64_t chebyshev_evaluations{0};
    uint64_t estimated_polynomial_multiplications{0};
    uint64_t bootstraps{0};
    uint64_t bootstrap_iterations{0};

    bool operator==(const OperationCounts& other) const noexcept {
        return rotations == other.rotations &&
            ct_pt_multiplications == other.ct_pt_multiplications &&
            ct_ct_multiplications == other.ct_ct_multiplications &&
            explicit_rescale_requests == other.explicit_rescale_requests &&
            chebyshev_evaluations == other.chebyshev_evaluations &&
            estimated_polynomial_multiplications ==
                other.estimated_polynomial_multiplications &&
            bootstraps == other.bootstraps &&
            bootstrap_iterations == other.bootstrap_iterations;
    }
};

struct TensorMetadata {
    uint32_t level{0};
    uint32_t noise_scale_degree{0};
    uint32_t remaining_levels{0};
    double scale_bits{0.0};
    double expected_scale_bits{0.0};
    std::size_t ciphertext_count{0};
};

struct MetadataUsedLevels {
    uint32_t input{0};
    uint32_t softmax_denominator{0};
    uint32_t attention_output{0};
    uint32_t ln1_variance{0};
    uint32_t ln1_output{0};
    uint32_t ffn_output{0};
    uint32_t ln2_variance{0};
    uint32_t raw_output{0};
};

struct MetadataUsedLevelDeltas {
    uint32_t input_to_softmax_checkpoint_net_recovered{0};
    uint32_t softmax_checkpoint_to_attention_output_consumed{0};
    uint32_t attention_output_to_ln1_checkpoint_net_recovered{0};
    uint32_t ln1_checkpoint_to_ln1_output_consumed{0};
    uint32_t ln1_output_to_ffn_output_consumed{0};
    uint32_t ffn_output_to_ln2_checkpoint_net_recovered{0};
    uint32_t ln2_checkpoint_to_raw_output_consumed{0};
    std::optional<uint32_t> previous_raw_output_to_input_recovered;
};

constexpr TensorMetadata kLayer0InputMetadata{
    29, 1, 18, kSingleScaleBits, kSingleScaleBits, 5};
constexpr TensorMetadata kLayerHandoffInputMetadata{
    19, 2, 27, kDoubleScaleBits, kDoubleScaleBits, 5};
constexpr TensorMetadata kPolynomialCheckpointMetadata{
    18, 2, 28, kDoubleScaleBits, kDoubleScaleBits, 5};
constexpr TensorMetadata kLayer0AttentionOutputMetadata{
    40, 2, 6, kDoubleScaleBits, kDoubleScaleBits, 5};
constexpr TensorMetadata kLaterLayerAttentionOutputMetadata{
    31, 2, 15, kDoubleScaleBits, kDoubleScaleBits, 5};
constexpr TensorMetadata kLayer0LayerNorm1OutputMetadata{
    32, 2, 14, kDoubleScaleBits, kDoubleScaleBits, 5};
constexpr TensorMetadata kLaterLayerNorm1OutputMetadata{
    29, 2, 17, kDoubleScaleBits, kDoubleScaleBits, 5};
constexpr TensorMetadata kLayer0FeedForwardOutputMetadata{
    45, 2, 1, kDoubleScaleBits, kDoubleScaleBits, 5};
constexpr TensorMetadata kLaterLayerFeedForwardOutputMetadata{
    42, 2, 4, kDoubleScaleBits, kDoubleScaleBits, 5};
constexpr TensorMetadata kEncoderRawOutputMetadata{
    29, 2, 17, kDoubleScaleBits, kDoubleScaleBits, 5};

struct EncryptedRanges {
    ObservedRange softmax_shifted_logits;
    ObservedRange softmax_denominator;
    ObservedRange attention_layernorm_normalized_variance;
    ObservedRange gelu_input;
    ObservedRange output_layernorm_normalized_variance;
};

struct InactiveSentinelRanges {
    ObservedRange softmax_denominator;
    ObservedRange attention_layernorm_normalized_variance;
    ObservedRange output_layernorm_normalized_variance;
};

struct LayerRecord {
    std::size_t layer_id{0};
    bool handoff_refresh_performed{false};
    QualityMetrics input_quality;
    QualityMetrics output_quality;
    QualityMetrics exact_trace_quality;
    double inactive_maximum{0.0};
    std::size_t inactive_zero_checkpoint_count{0};
    double inactive_sentinel_maximum{0.0};
    TensorMetadata input_metadata;
    TensorMetadata output_metadata;
    TensorMetadata denominator_metadata;
    TensorMetadata ln1_variance_metadata;
    TensorMetadata ln2_variance_metadata;
    TensorMetadata attention_output_metadata;
    TensorMetadata ln1_output_metadata;
    TensorMetadata ffn_output_metadata;
    MetadataUsedLevels metadata_used_levels;
    MetadataUsedLevelDeltas metadata_used_level_deltas;
    EncryptedRanges ranges;
    InactiveSentinelRanges inactive_sentinel_ranges;
    OperationCounts refresh_counts;
    OperationCounts layer_counts;
    OperationCounts cumulative_counts;
};

struct FrozenInputs {
    PlainMatrix initial_input;
    std::vector<moai::openfhe::EncoderLayerWeights> weights;
    std::vector<PlainMatrix> chained_inputs;
    std::vector<PlainMatrix> polynomial_outputs;
    std::vector<PlainMatrix> exact_outputs;
    std::vector<EncoderPlaintextOracleRanges> plaintext_ranges;
};

double ElapsedMilliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

Arguments ParseArgumentTokens(const std::vector<std::string>& tokens);

Arguments ParseArguments(int argc, char** argv) {
    std::vector<std::string> tokens;
    tokens.reserve(static_cast<std::size_t>(std::max(0, argc - 1)));
    for (int index = 1; index < argc; ++index) {
        tokens.emplace_back(argv[index]);
    }

    return ParseArgumentTokens(tokens);
}

std::size_t ParseDiagnosticLayerCount(const std::string& token) {
    std::size_t value = 0;
    const auto result = std::from_chars(
        token.data(),
        token.data() + token.size(),
        value);
    if (token.empty() || result.ec != std::errc{} ||
        result.ptr != token.data() + token.size() || value == 0 ||
        value >= moai::openfhe::kPaperCompatEncoderLayers) {
        throw std::invalid_argument(
            "diagnostic layer count must be an integer in [1,11]");
    }
    return value;
}

Arguments ParseArgumentTokens(const std::vector<std::string>& tokens) {
    Arguments arguments;
    bool data_root_seen = false;
    bool preflight_seen = false;
    bool crypto_preflight_seen = false;
    bool layer_count_seen = false;
    bool calibration_seen = false;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const std::string& argument = tokens[index];
        if (argument == "--data-root") {
            if (data_root_seen || index + 1 >= tokens.size()) {
                throw std::invalid_argument(
                    "duplicate or incomplete argument: --data-root");
            }
            data_root_seen = true;
            const std::string& data_root = tokens[++index];
            if (data_root.empty() || data_root.rfind("--", 0) == 0) {
                throw std::invalid_argument(
                    "--data-root requires a non-option path");
            }
            arguments.data_root = data_root;
        }
        else if (argument == "--preflight-only") {
            if (preflight_seen) {
                throw std::invalid_argument("duplicate argument: --preflight-only");
            }
            preflight_seen = true;
            arguments.preflight_only = true;
        }
        else if (argument == "--crypto-preflight-only") {
            if (crypto_preflight_seen) {
                throw std::invalid_argument(
                    "duplicate argument: --crypto-preflight-only");
            }
            crypto_preflight_seen = true;
            arguments.crypto_preflight_only = true;
        }
        else if (argument == "--diagnostic-layer-count") {
            if (layer_count_seen || index + 1 >= tokens.size()) {
                throw std::invalid_argument(
                    "duplicate or incomplete argument: --diagnostic-layer-count");
            }
            layer_count_seen = true;
            arguments.requested_layer_count =
                ParseDiagnosticLayerCount(tokens[++index]);
        }
        else if (argument == "--metadata-calibration") {
            if (calibration_seen) {
                throw std::invalid_argument(
                    "duplicate argument: --metadata-calibration");
            }
            calibration_seen = true;
            arguments.metadata_calibration = true;
        }
        else {
            throw std::invalid_argument(
                "unknown or incomplete argument: " + argument);
        }
    }
    if ((arguments.preflight_only || arguments.crypto_preflight_only) &&
        (arguments.requested_layer_count.has_value() ||
         arguments.metadata_calibration)) {
        throw std::invalid_argument(
            "preflight-only modes cannot be combined with a runtime diagnostic");
    }
    if (arguments.preflight_only && arguments.crypto_preflight_only) {
        throw std::invalid_argument(
            "plaintext and crypto preflight-only modes are mutually exclusive");
    }
    if (arguments.metadata_calibration &&
        !arguments.requested_layer_count.has_value()) {
        throw std::invalid_argument(
            "metadata calibration requires --diagnostic-layer-count");
    }
    if (arguments.metadata_calibration &&
        *arguments.requested_layer_count < 2) {
        throw std::invalid_argument(
            "metadata calibration requires at least a two-layer prefix");
    }
    return arguments;
}

std::size_t EvaluatedLayerCount(const Arguments& arguments) {
    return arguments.requested_layer_count.value_or(
        moai::openfhe::kPaperCompatEncoderLayers);
}

MetadataMode ResolveMetadataMode(const Arguments& arguments) {
    return arguments.metadata_calibration
        ? MetadataMode::kCalibration
        : MetadataMode::kExact;
}

RunKind ResolveRunKind(const Arguments& arguments) {
    if (!arguments.requested_layer_count.has_value()) {
        return RunKind::kFormalFull;
    }
    return arguments.metadata_calibration
        ? RunKind::kMetadataCalibration
        : RunKind::kExactPrefix;
}

const char* LayerTestName(RunKind run_kind) {
    switch (run_kind) {
        case RunKind::kFormalFull:
            return "openfhe_encoder_12_layer_layer";
        case RunKind::kExactPrefix:
            return "openfhe_encoder_exact_prefix_layer";
        case RunKind::kMetadataCalibration:
            return "openfhe_encoder_metadata_calibration_layer";
    }
    throw std::runtime_error("unknown run kind for layer test name");
}

const char* SummaryTestName(RunKind run_kind) {
    switch (run_kind) {
        case RunKind::kFormalFull:
            return "openfhe_encoder_12_layer";
        case RunKind::kExactPrefix:
            return "openfhe_encoder_exact_prefix";
        case RunKind::kMetadataCalibration:
            return "openfhe_encoder_metadata_calibration";
    }
    throw std::runtime_error("unknown run kind for summary test name");
}

const char* ClaimScope(RunKind run_kind) {
    switch (run_kind) {
        case RunKind::kFormalFull:
            return "m5_12_layer_correctness";
        case RunKind::kExactPrefix:
            return "diagnostic_prefix_exact_schedule";
        case RunKind::kMetadataCalibration:
            return "diagnostic_metadata_calibration_only";
    }
    throw std::runtime_error("unknown run kind for claim scope");
}

QualityMetrics MeasureQuality(
    const PlainMatrix& actual,
    const PlainMatrix& expected,
    std::size_t active_features) {
    if (actual.size() != expected.size() || actual.empty()) {
        throw std::runtime_error("quality row count changed");
    }
    long double squared_error = 0.0;
    long double squared_actual = 0.0;
    long double squared_expected = 0.0;
    long double dot = 0.0;
    double maximum_absolute = 0.0;
    for (std::size_t row = 0; row < actual.size(); ++row) {
        if (actual[row].size() < active_features ||
            expected[row].size() < active_features) {
            throw std::runtime_error("quality feature width changed");
        }
        for (std::size_t feature = 0; feature < active_features; ++feature) {
            const long double observed = actual[row][feature];
            const long double reference = expected[row][feature];
            if (!std::isfinite(static_cast<double>(observed)) ||
                !std::isfinite(static_cast<double>(reference))) {
                throw std::runtime_error("quality input contains NaN or Inf");
            }
            const long double error = observed - reference;
            squared_error += error * error;
            squared_actual += observed * observed;
            squared_expected += reference * reference;
            dot += observed * reference;
            maximum_absolute = std::max(
                maximum_absolute,
                std::abs(static_cast<double>(error)));
        }
    }
    if (squared_actual == 0.0 || squared_expected == 0.0) {
        throw std::runtime_error("quality norm is zero");
    }
    return {
        std::sqrt(static_cast<double>(squared_error / squared_expected)),
        static_cast<double>(dot / std::sqrt(squared_actual * squared_expected)),
        maximum_absolute};
}

void RequireQuality(const QualityMetrics& quality, const std::string& label) {
    if (!std::isfinite(quality.relative_l2) ||
        !std::isfinite(quality.cosine) ||
        !std::isfinite(quality.maximum_absolute) ||
        quality.relative_l2 > kPerLayerRelativeL2Maximum ||
        quality.cosine < kPerLayerCosineMinimum) {
        std::ostringstream message;
        message << std::setprecision(17) << label
                << " failed the frozen M5 quality gate: rel_l2="
                << quality.relative_l2 << " cosine=" << quality.cosine
                << " max_absolute=" << quality.maximum_absolute;
        throw std::runtime_error(message.str());
    }
}

void RequirePlaintextOracleQuality(
    const QualityMetrics& quality,
    const std::string& label) {
    if (!std::isfinite(quality.relative_l2) ||
        !std::isfinite(quality.cosine) ||
        !std::isfinite(quality.maximum_absolute) ||
        quality.relative_l2 > 0.012 || quality.cosine < 0.9999 ||
        quality.maximum_absolute > 0.1) {
        throw std::runtime_error(label + " failed the chained oracle gate");
    }
}

double MaximumInactiveDeviation(
    const PlainMatrix& values,
    std::size_t active_features,
    double expected_value) {
    double maximum = 0.0;
    for (const auto& row : values) {
        if (row.size() != moai::openfhe::kPaperCompatFeatureBlock ||
            active_features > row.size()) {
            throw std::runtime_error("inactive checkpoint width changed");
        }
        for (std::size_t index = active_features; index < row.size(); ++index) {
            if (!std::isfinite(row[index])) {
                throw std::runtime_error("inactive checkpoint contains NaN or Inf");
            }
            maximum = std::max(
                maximum,
                std::abs(row[index] - expected_value));
        }
    }
    return maximum;
}

double MaximumInactive(const PlainMatrix& values, std::size_t active_features) {
    return MaximumInactiveDeviation(values, active_features, 0.0);
}

const std::vector<std::string>& ExpectedInactiveZeroCheckpointLabels() {
    static const std::vector<std::string> labels = []() {
        std::vector<std::string> result{
            "encoder_input",
            "query",
            "key",
            "value",
            "scaled_scores",
            "shifted_logits",
            "probabilities",
            "attention_output_before_bootstrap_cleanup",
            "attention_output_after_bootstrap_cleanup",
            "self_projection",
            "attention_residual",
            "attention_layernorm",
            "output_projection",
            "output_residual_before_bootstrap",
            "output_residual_after_bootstrap",
            "encoder_output"};
        for (std::size_t block = 0;
             block < moai::openfhe::kPaperCompatIntermediateBlocks;
             ++block) {
            const auto suffix = std::to_string(block);
            result.push_back("intermediate_pre_activation_" + suffix);
            result.push_back("intermediate_polynomial_output_" + suffix);
            result.push_back("intermediate_activation_" + suffix);
            result.push_back("output_contribution_" + suffix);
        }
        return result;
    }();
    return labels;
}

const std::set<std::string>& FullWidthInactiveZeroCheckpointLabels() {
    static const std::set<std::string> labels = []() {
        std::set<std::string> result;
        for (std::size_t block = 0;
             block < moai::openfhe::kPaperCompatIntermediateBlocks;
             ++block) {
            const auto suffix = std::to_string(block);
            result.insert("intermediate_pre_activation_" + suffix);
            result.insert("intermediate_polynomial_output_" + suffix);
            result.insert("intermediate_activation_" + suffix);
        }
        return result;
    }();
    return labels;
}

std::size_t ExpectedInactiveZeroCheckpointActiveFeatures(
    const std::string& label) {
    const auto& expected = ExpectedInactiveZeroCheckpointLabels();
    if (std::find(expected.begin(), expected.end(), label) == expected.end()) {
        throw std::runtime_error(
            "unexpected inactive checkpoint label: " + label);
    }
    return FullWidthInactiveZeroCheckpointLabels().count(label) == 1
        ? moai::openfhe::kPaperCompatFeatureBlock
        : moai::openfhe::kPaperCompatHiddenSize;
}

void RequireInactiveZeroCheckpointActiveFeatures(
    const std::string& label,
    std::size_t actual) {
    const auto expected =
        ExpectedInactiveZeroCheckpointActiveFeatures(label);
    if (actual != expected) {
        throw std::runtime_error(
            "inactive checkpoint active-feature width drifted: " + label);
    }
}

void RequireExactInactiveZeroCheckpointLabels(
    const std::vector<std::string>& observed_labels) {
    const auto& expected_labels = ExpectedInactiveZeroCheckpointLabels();
    if (expected_labels.size() != kInactiveZeroCheckpointCount) {
        throw std::logic_error("frozen inactive checkpoint label count drifted");
    }
    if (observed_labels.size() != kInactiveZeroCheckpointCount) {
        throw std::runtime_error(
            "inactive checkpoint observation count must be exactly 28");
    }
    const std::set<std::string> observed(
        observed_labels.begin(),
        observed_labels.end());
    if (observed.size() != observed_labels.size()) {
        throw std::runtime_error("inactive checkpoint labels contain a duplicate");
    }
    const std::set<std::string> expected(
        expected_labels.begin(),
        expected_labels.end());
    if (observed != expected) {
        throw std::runtime_error(
            "inactive checkpoint labels contain a missing or extra label");
    }
}

class InactiveZeroCheckpointAccumulator {
public:
    void Observe(
        const std::string& label,
        const PlainMatrix& values,
        std::size_t active_features) {
        RequireInactiveZeroCheckpointActiveFeatures(label, active_features);
        if (std::find(observed_labels_.begin(), observed_labels_.end(), label) !=
            observed_labels_.end()) {
            throw std::runtime_error(
                "duplicate inactive checkpoint label: " + label);
        }
        observed_labels_.push_back(label);
        maximum_ = std::max(
            maximum_,
            MaximumInactive(values, active_features));
    }

    [[nodiscard]] double Finish() const {
        RequireExactInactiveZeroCheckpointLabels(observed_labels_);
        return maximum_;
    }

    [[nodiscard]] std::size_t count() const noexcept {
        return observed_labels_.size();
    }

private:
    std::vector<std::string> observed_labels_;
    double maximum_{0.0};
};

OperationCounts Counts(const RunMetrics& metrics) {
    return {
        metrics.rotations,
        metrics.ct_pt_multiplications,
        metrics.ct_ct_multiplications,
        metrics.rescale_operations,
        metrics.chebyshev_evaluations,
        metrics.estimated_polynomial_multiplications,
        metrics.bootstraps,
        metrics.bootstrap_iterations};
}

uint64_t CheckedDifference(uint64_t after, uint64_t before, const char* label) {
    if (after < before) {
        throw std::runtime_error(std::string(label) + " counter decreased");
    }
    return after - before;
}

OperationCounts Difference(const RunMetrics& after, const RunMetrics& before) {
    return {
        CheckedDifference(after.rotations, before.rotations, "rotations"),
        CheckedDifference(
            after.ct_pt_multiplications,
            before.ct_pt_multiplications,
            "Ct-Pt"),
        CheckedDifference(
            after.ct_ct_multiplications,
            before.ct_ct_multiplications,
            "Ct-Ct"),
        CheckedDifference(
            after.rescale_operations,
            before.rescale_operations,
            "rescale"),
        CheckedDifference(
            after.chebyshev_evaluations,
            before.chebyshev_evaluations,
            "Chebyshev"),
        CheckedDifference(
            after.estimated_polynomial_multiplications,
            before.estimated_polynomial_multiplications,
            "polynomial"),
        CheckedDifference(after.bootstraps, before.bootstraps, "bootstrap"),
        CheckedDifference(
            after.bootstrap_iterations,
            before.bootstrap_iterations,
            "bootstrap iterations")};
}

OperationCounts MultiplyCounts(OperationCounts counts, std::size_t multiplier) {
    counts.rotations *= multiplier;
    counts.ct_pt_multiplications *= multiplier;
    counts.ct_ct_multiplications *= multiplier;
    counts.explicit_rescale_requests *= multiplier;
    counts.chebyshev_evaluations *= multiplier;
    counts.estimated_polynomial_multiplications *= multiplier;
    counts.bootstraps *= multiplier;
    counts.bootstrap_iterations *= multiplier;
    return counts;
}

OperationCounts AddCounts(OperationCounts left, const OperationCounts& right) {
    left.rotations += right.rotations;
    left.ct_pt_multiplications += right.ct_pt_multiplications;
    left.ct_ct_multiplications += right.ct_ct_multiplications;
    left.explicit_rescale_requests += right.explicit_rescale_requests;
    left.chebyshev_evaluations += right.chebyshev_evaluations;
    left.estimated_polynomial_multiplications +=
        right.estimated_polynomial_multiplications;
    left.bootstraps += right.bootstraps;
    left.bootstrap_iterations += right.bootstrap_iterations;
    return left;
}

void PrintCounts(const OperationCounts& counts) {
    std::cout
        << "{\"rotations\":" << counts.rotations
        << ",\"ct_pt_multiplications\":" << counts.ct_pt_multiplications
        << ",\"ct_ct_multiplications\":" << counts.ct_ct_multiplications
        << ",\"explicit_rescale_requests\":"
        << counts.explicit_rescale_requests
        << ",\"chebyshev_evaluations\":" << counts.chebyshev_evaluations
        << ",\"estimated_polynomial_multiplications\":"
        << counts.estimated_polynomial_multiplications
        << ",\"bootstraps\":" << counts.bootstraps
        << ",\"bootstrap_iterations\":" << counts.bootstrap_iterations << '}';
}

void RequireCounts(
    const OperationCounts& actual,
    const OperationCounts& expected,
    const std::string& label) {
    if (!(actual == expected)) {
        std::ostringstream message;
        message << label << " operation-count contract drifted";
        throw std::runtime_error(message.str());
    }
}

TensorMetadata InspectTensor(
    const CipherTensor& tensor,
    const moai::openfhe::ServerRuntime& server,
    std::size_t expected_count,
    double expected_scale_bits,
    const std::string& label) {
    if (tensor.empty() || tensor.size() != expected_count ||
        tensor.packing.layout != moai::openfhe::PackingLayout::kContiguous ||
        tensor.packing.batch_lanes != 1 ||
        tensor.packing.slot_count != 32768 ||
        tensor.packing.active_slots !=
            moai::openfhe::kPaperCompatFeatureBlock ||
        tensor.packing.encoded_slots !=
            moai::openfhe::kPaperCompatFeatureBlock ||
        tensor.packing.logical_shape != std::vector<std::size_t>{
            moai::openfhe::kPaperCompatFeatureBlock,
            expected_count}) {
        throw std::runtime_error(label + " packing contract drifted");
    }
    const double scale_bits = std::log2(tensor.packing.scaling_factor);
    if (!std::isfinite(scale_bits) ||
        std::abs(scale_bits - expected_scale_bits) > kScaleBitsTolerance) {
        std::ostringstream message;
        message << std::setprecision(17) << label << " scale_bits="
                << scale_bits << " differs from expected="
                << expected_scale_bits;
        throw std::runtime_error(message.str());
    }
    for (const auto& ciphertext : tensor.ciphertexts) {
        const double ciphertext_scale_bits = ciphertext == nullptr
            ? std::numeric_limits<double>::quiet_NaN()
            : std::log2(ciphertext->GetScalingFactor());
        if (!ciphertext || ciphertext->GetLevel() != tensor.packing.level ||
            ciphertext->GetNoiseScaleDeg() !=
                tensor.packing.noise_scale_degree ||
            !std::isfinite(ciphertext_scale_bits) ||
            std::abs(ciphertext_scale_bits - scale_bits) >
                kScaleBitsTolerance) {
            throw std::runtime_error(label + " ciphertext metadata is stale");
        }
    }
    return {
        tensor.packing.level,
        tensor.packing.noise_scale_degree,
        server.RemainingLevels(tensor),
        scale_bits,
        expected_scale_bits,
        tensor.size()};
}

uint32_t UsedLevels(
    const TensorMetadata& metadata,
    const std::string& label) {
    if (metadata.noise_scale_degree == 0) {
        throw std::runtime_error(
            label + " noise_scale_degree must be positive");
    }
    const uint64_t used = static_cast<uint64_t>(metadata.level) +
        static_cast<uint64_t>(metadata.noise_scale_degree - 1);
    if (used >= kMetadataMultiplicativeDepth ||
        metadata.remaining_levels !=
            kMetadataMultiplicativeDepth - used) {
        std::ostringstream message;
        message << label << " violates the depth-"
                << kMetadataMultiplicativeDepth
                << " schedule: level=" << metadata.level
                << " noise_scale_degree=" << metadata.noise_scale_degree
                << " used=" << used
                << " remaining=" << metadata.remaining_levels;
        throw std::runtime_error(message.str());
    }
    return static_cast<uint32_t>(used);
}

const TensorMetadata& ExpectedAttentionOutputMetadata(std::size_t layer) {
    if (layer >= moai::openfhe::kPaperCompatEncoderLayers) {
        throw std::runtime_error("attention metadata layer is out of range");
    }
    return layer == 0
        ? kLayer0AttentionOutputMetadata
        : kLaterLayerAttentionOutputMetadata;
}

const TensorMetadata& ExpectedInputMetadata(std::size_t layer) {
    if (layer >= moai::openfhe::kPaperCompatEncoderLayers) {
        throw std::runtime_error("input metadata layer is out of range");
    }
    return layer == 0
        ? kLayer0InputMetadata
        : kLayerHandoffInputMetadata;
}

const TensorMetadata& ExpectedLayerNorm1OutputMetadata(std::size_t layer) {
    if (layer >= moai::openfhe::kPaperCompatEncoderLayers) {
        throw std::runtime_error("LN1 metadata layer is out of range");
    }
    return layer == 0
        ? kLayer0LayerNorm1OutputMetadata
        : kLaterLayerNorm1OutputMetadata;
}

const TensorMetadata& ExpectedFeedForwardOutputMetadata(std::size_t layer) {
    if (layer >= moai::openfhe::kPaperCompatEncoderLayers) {
        throw std::runtime_error("FFN metadata layer is out of range");
    }
    return layer == 0
        ? kLayer0FeedForwardOutputMetadata
        : kLaterLayerFeedForwardOutputMetadata;
}

uint32_t ExpectedLayerNorm1UsedLevelDelta(std::size_t layer) {
    if (layer >= moai::openfhe::kPaperCompatEncoderLayers) {
        throw std::runtime_error("LN1 metadata delta layer is out of range");
    }
    return layer == 0 ? 14 : 11;
}

void RequireMetadata(
    const TensorMetadata& actual,
    const TensorMetadata& expected,
    const std::string& label) {
    static_cast<void>(UsedLevels(actual, label));
    static_cast<void>(UsedLevels(expected, label + " expected"));
    if (actual.level != expected.level ||
        actual.noise_scale_degree != expected.noise_scale_degree ||
        actual.remaining_levels != expected.remaining_levels ||
        !std::isfinite(actual.scale_bits) ||
        !std::isfinite(actual.expected_scale_bits) ||
        !std::isfinite(expected.scale_bits) ||
        !std::isfinite(expected.expected_scale_bits) ||
        std::abs(actual.scale_bits - expected.expected_scale_bits) >
            kScaleBitsTolerance ||
        actual.expected_scale_bits != expected.expected_scale_bits ||
        expected.scale_bits != expected.expected_scale_bits ||
        actual.ciphertext_count != expected.ciphertext_count) {
        std::ostringstream message;
        message << label << " level/scale schedule drifted: actual=("
                << actual.level << ',' << actual.noise_scale_degree << ','
                << actual.remaining_levels << ',' << actual.ciphertext_count
                << ',' << actual.scale_bits << ") expected=("
                << expected.level << ',' << expected.noise_scale_degree << ','
                << expected.remaining_levels << ','
                << expected.ciphertext_count << ','
                << expected.expected_scale_bits << ')';
        throw std::runtime_error(message.str());
    }
}

void RequireLiveObservationEquivalent(
    const TensorMetadata& actual,
    const TensorMetadata& observed,
    const std::string& label) {
    static_cast<void>(UsedLevels(actual, label));
    static_cast<void>(UsedLevels(observed, label + " observed"));
    const bool finite_scales =
        std::isfinite(actual.scale_bits) &&
        std::isfinite(actual.expected_scale_bits) &&
        std::isfinite(observed.scale_bits) &&
        std::isfinite(observed.expected_scale_bits);
    if (actual.level != observed.level ||
        actual.noise_scale_degree != observed.noise_scale_degree ||
        actual.remaining_levels != observed.remaining_levels ||
        actual.ciphertext_count != observed.ciphertext_count ||
        !finite_scales ||
        std::abs(
            actual.expected_scale_bits - observed.expected_scale_bits) >
            kScaleBitsTolerance ||
        std::abs(actual.scale_bits - actual.expected_scale_bits) >
            kScaleBitsTolerance ||
        std::abs(observed.scale_bits - observed.expected_scale_bits) >
            kScaleBitsTolerance ||
        std::abs(actual.scale_bits - observed.scale_bits) >
            kScaleBitsTolerance) {
        std::ostringstream message;
        message << std::setprecision(17) << label
                << " live metadata observation drifted: actual=("
                << actual.level << ',' << actual.noise_scale_degree << ','
                << actual.remaining_levels << ',' << actual.ciphertext_count
                << ',' << actual.scale_bits << ','
                << actual.expected_scale_bits << ") observed=("
                << observed.level << ',' << observed.noise_scale_degree << ','
                << observed.remaining_levels << ','
                << observed.ciphertext_count << ',' << observed.scale_bits
                << ',' << observed.expected_scale_bits << ')';
        throw std::runtime_error(message.str());
    }
}

void RequireUsedLevelDelta(
    const TensorMetadata& before,
    const TensorMetadata& after,
    uint32_t expected_delta,
    const std::string& label) {
    const uint32_t before_used = UsedLevels(before, label + " input");
    const uint32_t after_used = UsedLevels(after, label + " output");
    if (after_used < before_used ||
        after_used - before_used != expected_delta) {
        std::ostringstream message;
        message << label << " used-level delta drifted: before="
                << before_used << " after=" << after_used
                << " expected_delta=" << expected_delta;
        throw std::runtime_error(message.str());
    }
}

uint32_t RequireNonnegativeDifference(
    uint32_t minuend,
    uint32_t subtrahend,
    const std::string& label) {
    if (minuend < subtrahend) {
        std::ostringstream message;
        message << label << " has an unexplained negative used-level delta: "
                << minuend << " - " << subtrahend;
        throw std::runtime_error(message.str());
    }
    return minuend - subtrahend;
}

void RequireMetadataFlow(
    LayerRecord& record,
    const LayerRecord* previous_record) {
    auto& used = record.metadata_used_levels;
    used.input = UsedLevels(record.input_metadata, "layer input");
    used.softmax_denominator = UsedLevels(
        record.denominator_metadata,
        "Softmax denominator checkpoint");
    used.attention_output = UsedLevels(
        record.attention_output_metadata,
        "attention output");
    used.ln1_variance = UsedLevels(
        record.ln1_variance_metadata,
        "LN1 variance checkpoint");
    used.ln1_output = UsedLevels(record.ln1_output_metadata, "LN1 output");
    used.ffn_output = UsedLevels(record.ffn_output_metadata, "FFN output");
    used.ln2_variance = UsedLevels(
        record.ln2_variance_metadata,
        "LN2 variance checkpoint");
    used.raw_output = UsedLevels(record.output_metadata, "encoder raw output");

    auto& deltas = record.metadata_used_level_deltas;
    deltas.input_to_softmax_checkpoint_net_recovered =
        RequireNonnegativeDifference(
            used.input,
            used.softmax_denominator,
            "input to Softmax bootstrap checkpoint net recovery");
    deltas.softmax_checkpoint_to_attention_output_consumed =
        RequireNonnegativeDifference(
            used.attention_output,
            used.softmax_denominator,
            "Softmax bootstrap checkpoint to attention output consumption");
    deltas.attention_output_to_ln1_checkpoint_net_recovered =
        RequireNonnegativeDifference(
            used.attention_output,
            used.ln1_variance,
            "attention output to LN1 bootstrap checkpoint net recovery");
    deltas.ln1_checkpoint_to_ln1_output_consumed =
        RequireNonnegativeDifference(
            used.ln1_output,
            used.ln1_variance,
            "LN1 bootstrap checkpoint to LN1 output consumption");
    deltas.ln1_output_to_ffn_output_consumed =
        RequireNonnegativeDifference(
            used.ffn_output,
            used.ln1_output,
            "LN1 output to FFN output consumption");
    deltas.ffn_output_to_ln2_checkpoint_net_recovered =
        RequireNonnegativeDifference(
            used.ffn_output,
            used.ln2_variance,
            "FFN output to LN2 bootstrap checkpoint net recovery");
    deltas.ln2_checkpoint_to_raw_output_consumed =
        RequireNonnegativeDifference(
            used.raw_output,
            used.ln2_variance,
            "LN2 bootstrap checkpoint to raw output consumption");

    if (record.layer_id == 0) {
        if (previous_record != nullptr) {
            throw std::runtime_error("layer zero unexpectedly has a previous record");
        }
        deltas.previous_raw_output_to_input_recovered.reset();
        return;
    }
    if (previous_record == nullptr ||
        previous_record->layer_id + 1 != record.layer_id ||
        !previous_record->handoff_refresh_performed) {
        throw std::runtime_error(
            "inter-layer metadata flow is missing its preceding refresh");
    }
    deltas.previous_raw_output_to_input_recovered =
        RequireNonnegativeDifference(
            previous_record->metadata_used_levels.raw_output,
            used.input,
            "previous raw output to refreshed layer input recovery");
}

void RequireLayerMetadataSchedule(
    std::size_t layer,
    const LayerRecord& record) {
    RequireMetadata(
        record.input_metadata,
        ExpectedInputMetadata(layer),
        "layer input");
    RequireMetadata(
        record.denominator_metadata,
        kPolynomialCheckpointMetadata,
        "Softmax denominator checkpoint");
    RequireMetadata(
        record.ln1_variance_metadata,
        kPolynomialCheckpointMetadata,
        "LN1 variance checkpoint");
    RequireMetadata(
        record.ln2_variance_metadata,
        kPolynomialCheckpointMetadata,
        "LN2 variance checkpoint");
    RequireMetadata(
        record.attention_output_metadata,
        ExpectedAttentionOutputMetadata(layer),
        "attention output");
    RequireUsedLevelDelta(
        record.input_metadata,
        record.attention_output_metadata,
        12,
        "layer input to attention output");

    RequireMetadata(
        record.ln1_output_metadata,
        ExpectedLayerNorm1OutputMetadata(layer),
        "LN1 output");
    RequireUsedLevelDelta(
        record.ln1_variance_metadata,
        record.ln1_output_metadata,
        ExpectedLayerNorm1UsedLevelDelta(layer),
        "LN1 bootstrap checkpoint to output");

    RequireMetadata(
        record.ffn_output_metadata,
        ExpectedFeedForwardOutputMetadata(layer),
        "FFN output");
    RequireUsedLevelDelta(
        record.ln1_output_metadata,
        record.ffn_output_metadata,
        13,
        "LN1 output to FFN output");

    RequireMetadata(
        record.output_metadata,
        kEncoderRawOutputMetadata,
        "encoder output");
    RequireUsedLevelDelta(
        record.ln2_variance_metadata,
        record.output_metadata,
        11,
        "LN2 bootstrap checkpoint to encoder output");
}

void RequireCalibrationMetadataSchedule(
    std::size_t layer,
    const LayerRecord& record) {
    RequireMetadata(
        record.attention_output_metadata,
        ExpectedAttentionOutputMetadata(layer),
        "attention output");
    if (layer == 0) {
        RequireUsedLevelDelta(
            record.input_metadata,
            record.attention_output_metadata,
            12,
            "layer-0 input to attention output");
        RequireMetadata(
            record.ln1_output_metadata,
            kLayer0LayerNorm1OutputMetadata,
            "layer-0 LN1 output");
        RequireMetadata(
            record.ffn_output_metadata,
            kLayer0FeedForwardOutputMetadata,
            "layer-0 FFN output");
        RequireUsedLevelDelta(
            record.ln1_output_metadata,
            record.ffn_output_metadata,
            13,
            "layer-0 LN1 output to FFN output");
        RequireMetadata(
            record.output_metadata,
            kEncoderRawOutputMetadata,
            "layer-0 encoder output");
    }
}

LayerRecord MakeSyntheticMetadataRecord(std::size_t layer) {
    LayerRecord record;
    record.layer_id = layer;
    record.input_metadata = layer == 0
        ? kLayer0InputMetadata
        : kLayerHandoffInputMetadata;
    record.denominator_metadata = kPolynomialCheckpointMetadata;
    record.ln1_variance_metadata = kPolynomialCheckpointMetadata;
    record.ln2_variance_metadata = kPolynomialCheckpointMetadata;
    record.attention_output_metadata =
        ExpectedAttentionOutputMetadata(layer);
    record.ln1_output_metadata = ExpectedLayerNorm1OutputMetadata(layer);
    record.ffn_output_metadata = ExpectedFeedForwardOutputMetadata(layer);
    record.output_metadata = kEncoderRawOutputMetadata;
    return record;
}

template <typename Operation>
void RequireScheduleRejection(
    Operation&& operation,
    const std::string& label) {
    bool rejected = false;
    try {
        operation();
    }
    catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error(label + " was not rejected");
    }
}

template <typename Operation>
void RequireArgumentRejection(
    Operation&& operation,
    const std::string& label) {
    bool rejected = false;
    try {
        operation();
    }
    catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error(label + " was not rejected");
    }
}

void ValidateInactiveZeroCheckpointContract() {
    const auto expected = ExpectedInactiveZeroCheckpointLabels();
    RequireExactInactiveZeroCheckpointLabels(expected);

    auto missing = expected;
    missing.pop_back();
    RequireScheduleRejection(
        [&missing]() { RequireExactInactiveZeroCheckpointLabels(missing); },
        "missing inactive checkpoint label");

    auto extra = expected;
    extra.push_back("unexpected_checkpoint");
    RequireScheduleRejection(
        [&extra]() { RequireExactInactiveZeroCheckpointLabels(extra); },
        "extra inactive checkpoint label");

    auto duplicate = expected;
    duplicate.back() = duplicate.front();
    RequireScheduleRejection(
        [&duplicate]() {
            RequireExactInactiveZeroCheckpointLabels(duplicate);
        },
        "duplicate inactive checkpoint label");

    auto substituted = expected;
    substituted.back() = "unexpected_checkpoint";
    RequireScheduleRejection(
        [&substituted]() {
            RequireExactInactiveZeroCheckpointLabels(substituted);
        },
        "missing-and-extra inactive checkpoint labels");

    RequireScheduleRejection(
        []() {
            RequireInactiveZeroCheckpointActiveFeatures(
                "query",
                moai::openfhe::kPaperCompatFeatureBlock);
        },
        "hidden checkpoint widened to the full feature block");
    RequireScheduleRejection(
        []() {
            RequireInactiveZeroCheckpointActiveFeatures(
                "intermediate_pre_activation_0",
                moai::openfhe::kPaperCompatHiddenSize);
        },
        "full-width intermediate checkpoint narrowed to hidden width");
}

void RequireRefreshPlacement(
    std::size_t layer,
    std::size_t evaluated_layer_count,
    bool actual_refresh) {
    if (evaluated_layer_count == 0 || layer >= evaluated_layer_count) {
        throw std::runtime_error("refresh placement indices are out of range");
    }
    const bool expected_refresh = layer + 1 < evaluated_layer_count;
    if (actual_refresh != expected_refresh) {
        throw std::runtime_error("inter-layer refresh placement drifted");
    }
}

void ValidateDiagnosticArgumentContract() {
    const auto exact_prefix = ParseArgumentTokens(
        {"--diagnostic-layer-count", "2"});
    if (exact_prefix.requested_layer_count != 2 ||
        exact_prefix.metadata_calibration ||
        EvaluatedLayerCount(exact_prefix) != 2 ||
        ResolveMetadataMode(exact_prefix) != MetadataMode::kExact ||
        ResolveRunKind(exact_prefix) != RunKind::kExactPrefix) {
        throw std::runtime_error("exact-prefix argument mode drifted");
    }
    const auto calibration = ParseArgumentTokens(
        {"--diagnostic-layer-count", "2", "--metadata-calibration"});
    if (calibration.requested_layer_count != 2 ||
        !calibration.metadata_calibration ||
        EvaluatedLayerCount(calibration) != 2 ||
        ResolveMetadataMode(calibration) != MetadataMode::kCalibration ||
        ResolveRunKind(calibration) != RunKind::kMetadataCalibration) {
        throw std::runtime_error("metadata-calibration argument mode drifted");
    }
    const auto formal = ParseArgumentTokens({});
    if (EvaluatedLayerCount(formal) !=
            moai::openfhe::kPaperCompatEncoderLayers ||
        ResolveMetadataMode(formal) != MetadataMode::kExact ||
        ResolveRunKind(formal) != RunKind::kFormalFull ||
        std::string(LayerTestName(RunKind::kFormalFull)) !=
            "openfhe_encoder_12_layer_layer" ||
        std::string(SummaryTestName(RunKind::kFormalFull)) !=
            "openfhe_encoder_12_layer" ||
        std::string(LayerTestName(RunKind::kExactPrefix)) ==
            LayerTestName(RunKind::kFormalFull) ||
        std::string(LayerTestName(RunKind::kMetadataCalibration)) ==
            LayerTestName(RunKind::kFormalFull) ||
        std::string(ClaimScope(RunKind::kMetadataCalibration)) !=
            "diagnostic_metadata_calibration_only") {
        throw std::runtime_error("formal full-run argument mode drifted");
    }

    const std::vector<std::vector<std::string>> rejected{
        {"--diagnostic-layer-count", "0"},
        {"--diagnostic-layer-count", "12"},
        {"--diagnostic-layer-count", "-1"},
        {"--diagnostic-layer-count", "2junk"},
        {"--diagnostic-layer-count", "2", "--diagnostic-layer-count", "3"},
        {"--metadata-calibration"},
        {"--metadata-calibration", "--metadata-calibration"},
        {"--diagnostic-layer-count", "1", "--metadata-calibration"},
        {"--preflight-only", "--diagnostic-layer-count", "2"},
        {"--crypto-preflight-only", "--metadata-calibration",
         "--diagnostic-layer-count", "2"},
        {"--preflight-only", "--preflight-only"},
        {"--data-root", "data", "--data-root", "data"},
        {"--data-root", "--preflight-only"},
    };
    for (std::size_t index = 0; index < rejected.size(); ++index) {
        RequireArgumentRejection(
            [&rejected, index]() {
                static_cast<void>(ParseArgumentTokens(rejected[index]));
            },
            "invalid diagnostic argument case " + std::to_string(index));
    }
}

void ValidateMetadataScheduleContract() {
    for (std::size_t layer = 0;
         layer < moai::openfhe::kPaperCompatEncoderLayers;
         ++layer) {
        auto record = MakeSyntheticMetadataRecord(layer);
        if (layer + 1 < moai::openfhe::kPaperCompatEncoderLayers) {
            record.handoff_refresh_performed = true;
        }
        RequireLayerMetadataSchedule(layer, record);
    }

    auto layer0 = MakeSyntheticMetadataRecord(0);
    layer0.handoff_refresh_performed = true;
    RequireMetadataFlow(layer0, nullptr);
    const auto layer1 = MakeSyntheticMetadataRecord(1);
    RequireScheduleRejection(
        [&layer0]() { RequireLayerMetadataSchedule(1, layer0); },
        "layer-0 attention tuple at a later layer");
    RequireScheduleRejection(
        [&layer1]() { RequireLayerMetadataSchedule(0, layer1); },
        "later-layer attention tuple at layer 0");

    auto wrong_later_ln1_regime = layer1;
    wrong_later_ln1_regime.ln1_output_metadata =
        kLayer0LayerNorm1OutputMetadata;
    RequireScheduleRejection(
        [&wrong_later_ln1_regime]() {
            RequireLayerMetadataSchedule(1, wrong_later_ln1_regime);
        },
        "layer-0 LN1 tuple at a later layer");

    auto wrong_later_ffn_regime = layer1;
    wrong_later_ffn_regime.ffn_output_metadata =
        kLayer0FeedForwardOutputMetadata;
    RequireScheduleRejection(
        [&wrong_later_ffn_regime]() {
            RequireLayerMetadataSchedule(1, wrong_later_ffn_regime);
        },
        "layer-0 FFN tuple at a later layer");

    auto wrong_layer0_ln1_regime = layer0;
    wrong_layer0_ln1_regime.ln1_output_metadata =
        kLaterLayerNorm1OutputMetadata;
    RequireScheduleRejection(
        [&wrong_layer0_ln1_regime]() {
            RequireLayerMetadataSchedule(0, wrong_layer0_ln1_regime);
        },
        "later-layer LN1 tuple at layer 0");

    auto wrong_layer0_ffn_regime = layer0;
    wrong_layer0_ffn_regime.ffn_output_metadata =
        kLaterLayerFeedForwardOutputMetadata;
    RequireScheduleRejection(
        [&wrong_layer0_ffn_regime]() {
            RequireLayerMetadataSchedule(0, wrong_layer0_ffn_regime);
        },
        "later-layer FFN tuple at layer 0");

    auto bad_handoff_tuple = layer1;
    bad_handoff_tuple.input_metadata.level = 18;
    bad_handoff_tuple.input_metadata.noise_scale_degree = 3;
    RequireScheduleRejection(
        [&bad_handoff_tuple]() {
            RequireLayerMetadataSchedule(1, bad_handoff_tuple);
        },
        "inter-layer handoff integer tuple drift");

    auto bad_handoff_count = layer1;
    --bad_handoff_count.input_metadata.ciphertext_count;
    RequireScheduleRejection(
        [&bad_handoff_count]() {
            RequireLayerMetadataSchedule(1, bad_handoff_count);
        },
        "inter-layer handoff ciphertext-count drift");

    auto bad_handoff_scale = layer1;
    bad_handoff_scale.input_metadata.scale_bits += 0.01;
    RequireScheduleRejection(
        [&bad_handoff_scale]() {
            RequireLayerMetadataSchedule(1, bad_handoff_scale);
        },
        "inter-layer handoff scale drift");

    auto bad_handoff_remaining = layer1;
    --bad_handoff_remaining.input_metadata.remaining_levels;
    RequireScheduleRejection(
        [&bad_handoff_remaining]() {
            RequireLayerMetadataSchedule(1, bad_handoff_remaining);
        },
        "inter-layer handoff remaining-level drift");

    auto inconsistent_remaining = layer1.attention_output_metadata;
    ++inconsistent_remaining.remaining_levels;
    RequireScheduleRejection(
        [&inconsistent_remaining]() {
            static_cast<void>(UsedLevels(
                inconsistent_remaining,
                "synthetic inconsistent remaining levels"));
        },
        "inconsistent remaining-level tuple");

    auto bad_attention_delta = kLaterLayerAttentionOutputMetadata;
    --bad_attention_delta.level;
    ++bad_attention_delta.remaining_levels;
    RequireScheduleRejection(
        [&layer1, &bad_attention_delta]() {
            RequireUsedLevelDelta(
                layer1.input_metadata,
                bad_attention_delta,
                12,
                "synthetic attention delta");
        },
        "attention used-level delta drift");

    auto bad_ln1_delta = kLaterLayerNorm1OutputMetadata;
    --bad_ln1_delta.level;
    ++bad_ln1_delta.remaining_levels;
    RequireScheduleRejection(
        [&layer1, &bad_ln1_delta]() {
            RequireUsedLevelDelta(
                layer1.ln1_variance_metadata,
                bad_ln1_delta,
                11,
                "synthetic LN1 delta");
        },
        "LN1 used-level delta drift");

    auto bad_ffn_delta = kLaterLayerFeedForwardOutputMetadata;
    --bad_ffn_delta.level;
    ++bad_ffn_delta.remaining_levels;
    RequireScheduleRejection(
        [&layer1, &bad_ffn_delta]() {
            RequireUsedLevelDelta(
                layer1.ln1_output_metadata,
                bad_ffn_delta,
                13,
                "synthetic FFN delta");
        },
        "FFN used-level delta drift");

    auto bad_output_delta = kEncoderRawOutputMetadata;
    --bad_output_delta.level;
    ++bad_output_delta.remaining_levels;
    RequireScheduleRejection(
        [&layer1, &bad_output_delta]() {
            RequireUsedLevelDelta(
                layer1.ln2_variance_metadata,
                bad_output_delta,
                11,
                "synthetic encoder-output delta");
        },
        "encoder-output used-level delta drift");

    RequireScheduleRejection(
        []() {
            static_cast<void>(ExpectedAttentionOutputMetadata(
                moai::openfhe::kPaperCompatEncoderLayers));
        },
        "out-of-range attention metadata layer");

    const TensorMetadata exhausted_depth{
        46, 2, 0, kDoubleScaleBits, kDoubleScaleBits, 5};
    RequireScheduleRejection(
        [&exhausted_depth]() {
            static_cast<void>(UsedLevels(exhausted_depth, "exhausted depth"));
        },
        "used level equal to multiplicative depth");

    auto live_final_observation = kEncoderRawOutputMetadata;
    live_final_observation.scale_bits = 100.00000015989157;
    auto live_final_reread = kEncoderRawOutputMetadata;
    live_final_reread.scale_bits = 100.00000007994579;
    RequireLiveObservationEquivalent(
        live_final_reread,
        live_final_observation,
        "synthetic live final observation");

    auto live_tuple_drift = live_final_reread;
    --live_tuple_drift.level;
    ++live_tuple_drift.remaining_levels;
    RequireScheduleRejection(
        [&live_tuple_drift, &live_final_observation]() {
            RequireLiveObservationEquivalent(
                live_tuple_drift,
                live_final_observation,
                "synthetic live tuple drift");
        },
        "live metadata integer tuple drift");

    auto live_scale_drift = live_final_reread;
    live_scale_drift.scale_bits += 0.01;
    RequireScheduleRejection(
        [&live_scale_drift, &live_final_observation]() {
            RequireLiveObservationEquivalent(
                live_scale_drift,
                live_final_observation,
                "synthetic live scale drift");
        },
        "live metadata scale drift");

    auto live_expected_scale_drift = live_final_observation;
    live_expected_scale_drift.expected_scale_bits += 0.01;
    RequireScheduleRejection(
        [&live_final_reread, &live_expected_scale_drift]() {
            RequireLiveObservationEquivalent(
                live_final_reread,
                live_expected_scale_drift,
                "synthetic live expected-scale drift");
        },
        "live metadata expected-scale drift");

    auto live_count_drift = live_final_reread;
    ++live_count_drift.ciphertext_count;
    RequireScheduleRejection(
        [&live_count_drift, &live_final_observation]() {
            RequireLiveObservationEquivalent(
                live_count_drift,
                live_final_observation,
                "synthetic live ciphertext-count drift");
        },
        "live metadata ciphertext-count drift");

    auto bad_denominator_remaining = MakeSyntheticMetadataRecord(0);
    ++bad_denominator_remaining.denominator_metadata.remaining_levels;
    RequireScheduleRejection(
        [&bad_denominator_remaining]() {
            RequireMetadataFlow(bad_denominator_remaining, nullptr);
        },
        "denominator remaining-level drift");

    auto negative_softmax_recovery = MakeSyntheticMetadataRecord(0);
    negative_softmax_recovery.denominator_metadata =
        TensorMetadata{29, 2, 17, kDoubleScaleBits, kDoubleScaleBits, 5};
    RequireScheduleRejection(
        [&negative_softmax_recovery]() {
            RequireMetadataFlow(negative_softmax_recovery, nullptr);
        },
        "negative Softmax checkpoint recovery");

    auto relaxed_layer1 = MakeSyntheticMetadataRecord(1);
    relaxed_layer1.input_metadata =
        TensorMetadata{20, 2, 26, kDoubleScaleBits, kDoubleScaleBits, 5};
    relaxed_layer1.denominator_metadata =
        TensorMetadata{19, 2, 27, kDoubleScaleBits, kDoubleScaleBits, 5};
    relaxed_layer1.ln1_variance_metadata = relaxed_layer1.denominator_metadata;
    relaxed_layer1.ln2_variance_metadata = relaxed_layer1.denominator_metadata;
    RequireMetadataFlow(relaxed_layer1, &layer0);
    RequireCalibrationMetadataSchedule(1, relaxed_layer1);
    RequireScheduleRejection(
        [&relaxed_layer1]() {
            RequireLayerMetadataSchedule(1, relaxed_layer1);
        },
        "unsealed calibration tuple in exact mode");
    auto changed_known_output = relaxed_layer1;
    --changed_known_output.attention_output_metadata.level;
    ++changed_known_output.attention_output_metadata.remaining_levels;
    RequireScheduleRejection(
        [&changed_known_output]() {
            RequireCalibrationMetadataSchedule(1, changed_known_output);
        },
        "sealed later-attention tuple in calibration mode");

    auto missing_refresh_layer0 = layer0;
    missing_refresh_layer0.handoff_refresh_performed = false;
    auto layer1_missing_refresh = MakeSyntheticMetadataRecord(1);
    RequireScheduleRejection(
        [&layer1_missing_refresh, &missing_refresh_layer0]() {
            RequireMetadataFlow(layer1_missing_refresh, &missing_refresh_layer0);
        },
        "missing prior handoff refresh");

    auto negative_handoff = MakeSyntheticMetadataRecord(1);
    negative_handoff.input_metadata =
        TensorMetadata{30, 2, 16, kDoubleScaleBits, kDoubleScaleBits, 5};
    RequireScheduleRejection(
        [&negative_handoff, &layer0]() {
            RequireMetadataFlow(negative_handoff, &layer0);
        },
        "negative inter-layer refresh recovery");

    RequireRefreshPlacement(0, 2, true);
    RequireRefreshPlacement(1, 2, false);
    RequireScheduleRejection(
        []() { RequireRefreshPlacement(0, 2, false); },
        "missing first-layer prefix refresh");
    RequireScheduleRejection(
        []() { RequireRefreshPlacement(1, 2, true); },
        "unexpected final-prefix refresh");
}

void PrintMetadata(const TensorMetadata& metadata) {
    std::cout
        << "{\"level\":" << metadata.level
        << ",\"noise_scale_degree\":" << metadata.noise_scale_degree
        << ",\"remaining_levels\":" << metadata.remaining_levels
        << ",\"scale_bits\":" << metadata.scale_bits
        << ",\"expected_scale_bits\":" << metadata.expected_scale_bits
        << ",\"ciphertext_count\":"
        << metadata.ciphertext_count << '}';
}

void PrintMetadataUsedLevels(const MetadataUsedLevels& used) {
    std::cout
        << "{\"input\":" << used.input
        << ",\"softmax_denominator\":" << used.softmax_denominator
        << ",\"attention_output\":" << used.attention_output
        << ",\"ln1_variance\":" << used.ln1_variance
        << ",\"ln1_output\":" << used.ln1_output
        << ",\"ffn_output\":" << used.ffn_output
        << ",\"ln2_variance\":" << used.ln2_variance
        << ",\"raw_output\":" << used.raw_output << '}';
}

void PrintMetadataUsedLevelDeltas(const MetadataUsedLevelDeltas& deltas) {
    std::cout
        << "{\"input_to_softmax_checkpoint_net_recovered\":"
        << deltas.input_to_softmax_checkpoint_net_recovered
        << ",\"softmax_checkpoint_to_attention_output_consumed\":"
        << deltas.softmax_checkpoint_to_attention_output_consumed
        << ",\"attention_output_to_ln1_checkpoint_net_recovered\":"
        << deltas.attention_output_to_ln1_checkpoint_net_recovered
        << ",\"ln1_checkpoint_to_ln1_output_consumed\":"
        << deltas.ln1_checkpoint_to_ln1_output_consumed
        << ",\"ln1_output_to_ffn_output_consumed\":"
        << deltas.ln1_output_to_ffn_output_consumed
        << ",\"ffn_output_to_ln2_checkpoint_net_recovered\":"
        << deltas.ffn_output_to_ln2_checkpoint_net_recovered
        << ",\"ln2_checkpoint_to_raw_output_consumed\":"
        << deltas.ln2_checkpoint_to_raw_output_consumed
        << ",\"previous_raw_output_to_input_recovered\":";
    if (deltas.previous_raw_output_to_input_recovered.has_value()) {
        std::cout << *deltas.previous_raw_output_to_input_recovered;
    }
    else {
        std::cout << "null";
    }
    std::cout << '}';
}

void ObserveActive(
    const PlainMatrix& values,
    std::size_t active_features,
    ObservedRange& observed,
    const char* label) {
    if (values.empty()) {
        throw std::runtime_error(std::string(label) + " is empty");
    }
    for (const auto& row : values) {
        if (row.size() < active_features) {
            throw std::runtime_error(std::string(label) + " width changed");
        }
        for (std::size_t index = 0; index < row.size(); ++index) {
            if (!std::isfinite(row[index])) {
                throw std::runtime_error(
                    std::string(label) + " contains NaN or Inf");
            }
            if (index < active_features) {
                observed.Observe(row[index], label);
            }
        }
    }
}

void ObserveInactive(
    const PlainMatrix& values,
    std::size_t active_features,
    ObservedRange& observed,
    const char* label) {
    if (values.empty()) {
        throw std::runtime_error(std::string(label) + " is empty");
    }
    for (const auto& row : values) {
        if (row.size() != moai::openfhe::kPaperCompatFeatureBlock ||
            active_features > row.size()) {
            throw std::runtime_error(std::string(label) + " width changed");
        }
        for (std::size_t index = active_features; index < row.size(); ++index) {
            observed.Observe(row[index], label);
        }
    }
}

void RequireRange(
    const ObservedRange& observed,
    const moai::openfhe::DeclaredRange& interval,
    const std::string& label) {
    if (!std::isfinite(observed.minimum) ||
        !std::isfinite(observed.maximum) ||
        observed.minimum < interval.minimum ||
        observed.maximum > interval.maximum) {
        std::ostringstream message;
        message << std::setprecision(17) << label << " range ["
                << observed.minimum << ',' << observed.maximum
                << "] escaped [" << interval.minimum << ','
                << interval.maximum << ']';
        throw std::runtime_error(message.str());
    }
}

void RequirePlaintextRange(
    const moai::openfhe::test::PlaintextOracleRange& observed,
    const moai::openfhe::DeclaredRange& interval,
    const std::string& label) {
    if (!std::isfinite(observed.minimum) ||
        !std::isfinite(observed.maximum) ||
        observed.minimum < interval.minimum ||
        observed.maximum > interval.maximum) {
        throw std::runtime_error(label + " plaintext range escaped its interval");
    }
}

void PrintRange(const ObservedRange& range) {
    std::cout << '[' << range.minimum << ',' << range.maximum << ']';
}

void PrintRanges(const EncryptedRanges& ranges) {
    std::cout << "{\"softmax_shifted_logits\":";
    PrintRange(ranges.softmax_shifted_logits);
    std::cout << ",\"softmax_denominator\":";
    PrintRange(ranges.softmax_denominator);
    std::cout << ",\"ln1_normalized_variance\":";
    PrintRange(ranges.attention_layernorm_normalized_variance);
    std::cout << ",\"gelu_input\":";
    PrintRange(ranges.gelu_input);
    std::cout << ",\"ln2_normalized_variance\":";
    PrintRange(ranges.output_layernorm_normalized_variance);
    std::cout << '}';
}

void PrintInactiveSentinelRanges(const InactiveSentinelRanges& ranges) {
    std::cout << "{\"softmax_denominator\":";
    PrintRange(ranges.softmax_denominator);
    std::cout << ",\"ln1_normalized_variance\":";
    PrintRange(ranges.attention_layernorm_normalized_variance);
    std::cout << ",\"ln2_normalized_variance\":";
    PrintRange(ranges.output_layernorm_normalized_variance);
    std::cout << '}';
}

FrozenInputs LoadFrozenInputs(const std::filesystem::path& data_root) {
    FrozenInputs inputs;
    const auto contracts = moai::openfhe::MakePaperCompatNonlinearContracts();
    inputs.weights.reserve(moai::openfhe::kPaperCompatEncoderLayers);
    inputs.chained_inputs.reserve(moai::openfhe::kPaperCompatEncoderLayers);
    inputs.polynomial_outputs.reserve(moai::openfhe::kPaperCompatEncoderLayers);
    inputs.exact_outputs.reserve(moai::openfhe::kPaperCompatEncoderLayers);
    inputs.plaintext_ranges.reserve(moai::openfhe::kPaperCompatEncoderLayers);

    PlainMatrix current;
    for (std::size_t layer = 0;
         layer < moai::openfhe::kPaperCompatEncoderLayers;
         ++layer) {
        auto fixture =
            moai::openfhe::test::LoadEncoderLayerFixture(data_root, layer);
        if (layer == 0) {
            current = fixture.expected.input;
            inputs.initial_input = current;
        }
        inputs.chained_inputs.push_back(current);
        const auto oracle =
            moai::openfhe::test::EvaluateEncoderLayerPlaintextOracle(
                current,
                fixture.weights);
        const auto quality = MeasureQuality(
            oracle.output,
            fixture.expected.output,
            moai::openfhe::kPaperCompatHiddenSize);
        RequirePlaintextOracleQuality(
            quality,
            "plaintext chained layer " + std::to_string(layer));
        RequirePlaintextRange(
            oracle.ranges.softmax_shifted_logits,
            contracts.softmax_exponential.interval,
            "Softmax shifted logits");
        RequirePlaintextRange(
            oracle.ranges.softmax_denominator,
            contracts.softmax_reciprocal.interval,
            "Softmax denominator");
        RequirePlaintextRange(
            oracle.ranges.attention_layernorm_normalized_variance,
            contracts.layernorm_inverse_sqrt.interval,
            "attention LayerNorm variance");
        RequirePlaintextRange(
            oracle.ranges.gelu_input,
            contracts.gelu.interval,
            "GELU input");
        RequirePlaintextRange(
            oracle.ranges.output_layernorm_normalized_variance,
            contracts.layernorm_inverse_sqrt.interval,
            "output LayerNorm variance");
        inputs.polynomial_outputs.push_back(oracle.output);
        inputs.exact_outputs.push_back(fixture.expected.output);
        inputs.plaintext_ranges.push_back(oracle.ranges);
        inputs.weights.push_back(std::move(fixture.weights));
        current = oracle.output;
    }
    return inputs;
}

void RequireNoHomomorphicWork(
    const RunMetrics& metrics,
    const std::string& label) {
    if (!(Counts(metrics) == OperationCounts{}) ||
        metrics.max_observed_level != 0 ||
        metrics.max_polynomial_depth != 0 ||
        metrics.multiplicative_depth != kMetadataMultiplicativeDepth) {
        throw std::runtime_error(
            label + " violated the no-homomorphic-work metrics contract");
    }
}

void ValidateNoHomomorphicWorkContract() {
    RunMetrics clean;
    clean.multiplicative_depth = kMetadataMultiplicativeDepth;
    RequireNoHomomorphicWork(clean, "synthetic clean metrics");

    auto additive_only = clean;
    additive_only.max_observed_level = 29;
    RequireScheduleRejection(
        [&additive_only]() {
            RequireNoHomomorphicWork(
                additive_only,
                "synthetic additive-only metrics");
        },
        "synthetic max_observed_level=29 no-work contract");
}

class ClientCheckpointValidator final
    : public moai::openfhe::EncoderCiphertextObserver {
public:
    ClientCheckpointValidator(
        moai::openfhe::ClientRuntime& client,
        moai::openfhe::ServerRuntime& server,
        const FrozenInputs& inputs,
        std::size_t evaluated_layer_count,
        MetadataMode metadata_mode,
        RunKind run_kind)
        : client_(client), server_(server), inputs_(inputs),
          contracts_(moai::openfhe::MakePaperCompatNonlinearContracts()),
          evaluated_layer_count_(evaluated_layer_count),
          metadata_mode_(metadata_mode),
          run_kind_(run_kind) {}

    void Observe(const EncoderLayerCiphertextTrace& trace) override {
        const auto begin = Clock::now();
        if (trace.layer_index != records_.size() ||
            trace.layer_index >= inputs_.polynomial_outputs.size()) {
            throw std::runtime_error("encoder callback layer order drifted");
        }
        const std::size_t layer = trace.layer_index;
        const bool expected_refresh = layer + 1 < evaluated_layer_count_;
        RequireRefreshPlacement(
            layer,
            evaluated_layer_count_,
            trace.inter_layer_refresh);

        LayerRecord record;
        record.layer_id = layer;
        record.handoff_refresh_performed = trace.inter_layer_refresh;
        record.input_metadata = InspectTensor(
            trace.input,
            server_,
            moai::openfhe::kPaperCompatTraceTokens,
            layer == 0 ? kSingleScaleBits : kDoubleScaleBits,
            "layer input");
        if (layer == 0) {
            RequireMetadata(
                record.input_metadata,
                kLayer0InputMetadata,
                "layer-0 input");
        }
        else if (metadata_mode_ == MetadataMode::kExact) {
            RequireMetadata(
                record.input_metadata,
                kLayerHandoffInputMetadata,
                "layer handoff");
        }

        const auto& checkpoints = trace.result.checkpoints;
        record.denominator_metadata = InspectTensor(
            checkpoints.attention.denominator_after_bootstrap,
            server_,
            5,
            kDoubleScaleBits,
            "Softmax denominator");
        record.attention_output_metadata = InspectTensor(
            checkpoints.attention.output_before_bootstrap_cleanup,
            server_,
            5,
            kDoubleScaleBits,
            "attention output");
        record.ln1_variance_metadata = InspectTensor(
            checkpoints.attention_layernorm_normalized_variance,
            server_,
            5,
            kDoubleScaleBits,
            "attention LayerNorm variance");
        record.ln1_output_metadata = InspectTensor(
            checkpoints.attention_layernorm,
            server_,
            5,
            kDoubleScaleBits,
            "attention LayerNorm output");
        record.ffn_output_metadata = InspectTensor(
            checkpoints.output_projection,
            server_,
            5,
            kDoubleScaleBits,
            "FFN output");
        record.ln2_variance_metadata = InspectTensor(
            checkpoints.output_layernorm_normalized_variance,
            server_,
            5,
            kDoubleScaleBits,
            "output LayerNorm variance");
        record.output_metadata = InspectTensor(
            trace.result.output,
            server_,
            5,
            kDoubleScaleBits,
            "encoder output");

        if (metadata_mode_ == MetadataMode::kExact) {
            RequireMetadata(
                record.denominator_metadata,
                kPolynomialCheckpointMetadata,
                "denominator");
            RequireMetadata(
                record.ln1_variance_metadata,
                kPolynomialCheckpointMetadata,
                "LN1 variance");
            RequireMetadata(
                record.ln2_variance_metadata,
                kPolynomialCheckpointMetadata,
                "LN2 variance");
        }
        RequireMetadataFlow(
            record,
            records_.empty() ? nullptr : &records_.back());
        if (metadata_mode_ == MetadataMode::kExact) {
            RequireLayerMetadataSchedule(layer, record);
        }
        else {
            RequireCalibrationMetadataSchedule(layer, record);
        }

        const auto input = client_.Decrypt(trace.input);
        const auto output = client_.Decrypt(trace.result.output);
        const auto exact_reference = inputs_.exact_outputs[layer];
        record.input_quality = MeasureQuality(
            input,
            inputs_.chained_inputs[layer],
            moai::openfhe::kPaperCompatHiddenSize);
        record.output_quality = MeasureQuality(
            output,
            inputs_.polynomial_outputs[layer],
            moai::openfhe::kPaperCompatHiddenSize);
        record.exact_trace_quality = MeasureQuality(
            output,
            exact_reference,
            moai::openfhe::kPaperCompatHiddenSize);
        RequireQuality(record.input_quality, "encrypted layer input");
        RequireQuality(record.output_quality, "encrypted layer output");
        RequireQuality(record.exact_trace_quality, "encrypted exact-trace diagnostic");

        const auto shifted_logits =
            client_.Decrypt(checkpoints.attention.shifted_logits);
        if (shifted_logits.size() != 25) {
            throw std::runtime_error("attention shifted-logit count drifted");
        }
        for (const auto& row : shifted_logits) {
            if (row.size() != moai::openfhe::kPaperCompatFeatureBlock ||
                !std::all_of(row.begin(), row.end(), [](double value) {
                    return std::isfinite(value);
                })) {
                throw std::runtime_error("attention score width drifted");
            }
            for (std::size_t head = 0;
                 head < moai::openfhe::kPaperCompatAttentionHeads;
                 ++head) {
                const std::size_t begin = head *
                    moai::openfhe::kPaperCompatAttentionHeadDimension;
                const std::size_t end = begin +
                    moai::openfhe::kPaperCompatAttentionHeadDimension;
                for (std::size_t slot = begin; slot < end; ++slot) {
                    record.ranges.softmax_shifted_logits.Observe(
                        row[slot],
                        "Softmax shifted logits");
                }
            }
        }
        const auto softmax_denominator =
            client_.Decrypt(checkpoints.attention.denominator_after_bootstrap);
        ObserveActive(
            softmax_denominator,
            moai::openfhe::kPaperCompatHiddenSize,
            record.ranges.softmax_denominator,
            "Softmax denominator");
        ObserveInactive(
            softmax_denominator,
            moai::openfhe::kPaperCompatHiddenSize,
            record.inactive_sentinel_ranges.softmax_denominator,
            "inactive Softmax denominator sentinel");
        const auto attention_layernorm_variance =
            client_.Decrypt(checkpoints.attention_layernorm_normalized_variance);
        ObserveActive(
            attention_layernorm_variance,
            moai::openfhe::kPaperCompatHiddenSize,
            record.ranges.attention_layernorm_normalized_variance,
            "attention LayerNorm variance");
        ObserveInactive(
            attention_layernorm_variance,
            moai::openfhe::kPaperCompatHiddenSize,
            record.inactive_sentinel_ranges
                .attention_layernorm_normalized_variance,
            "inactive attention LayerNorm variance sentinel");
        for (const auto& gelu_input : checkpoints.intermediate_pre_activation) {
            ObserveActive(
                client_.Decrypt(gelu_input),
                moai::openfhe::kPaperCompatFeatureBlock,
                record.ranges.gelu_input,
                "GELU input");
        }
        const auto output_layernorm_variance =
            client_.Decrypt(checkpoints.output_layernorm_normalized_variance);
        ObserveActive(
            output_layernorm_variance,
            moai::openfhe::kPaperCompatHiddenSize,
            record.ranges.output_layernorm_normalized_variance,
            "output LayerNorm variance");
        ObserveInactive(
            output_layernorm_variance,
            moai::openfhe::kPaperCompatHiddenSize,
            record.inactive_sentinel_ranges
                .output_layernorm_normalized_variance,
            "inactive output LayerNorm variance sentinel");

        RequireRange(
            record.ranges.softmax_shifted_logits,
            contracts_.softmax_exponential.interval,
            "Softmax shifted logits");
        RequireRange(
            record.ranges.softmax_denominator,
            contracts_.softmax_reciprocal.interval,
            "Softmax denominator");
        RequireRange(
            record.ranges.attention_layernorm_normalized_variance,
            contracts_.layernorm_inverse_sqrt.interval,
            "attention LayerNorm variance");
        RequireRange(record.ranges.gelu_input, contracts_.gelu.interval, "GELU input");
        RequireRange(
            record.ranges.output_layernorm_normalized_variance,
            contracts_.layernorm_inverse_sqrt.interval,
            "output LayerNorm variance");
        RequireRange(
            record.inactive_sentinel_ranges.softmax_denominator,
            contracts_.softmax_reciprocal.interval,
            "inactive Softmax denominator sentinel");
        RequireRange(
            record.inactive_sentinel_ranges
                .attention_layernorm_normalized_variance,
            contracts_.layernorm_inverse_sqrt.interval,
            "inactive attention LayerNorm variance sentinel");
        RequireRange(
            record.inactive_sentinel_ranges
                .output_layernorm_normalized_variance,
            contracts_.layernorm_inverse_sqrt.interval,
            "inactive output LayerNorm variance sentinel");

        const auto attention_output = client_.Decrypt(
            checkpoints.attention.output_before_bootstrap_cleanup);
        const auto attention_layernorm =
            client_.Decrypt(checkpoints.attention_layernorm);
        const auto output_projection =
            client_.Decrypt(checkpoints.output_projection);
        InactiveZeroCheckpointAccumulator inactive_zero;
        const auto observe_ciphertext = [this, &inactive_zero](
            const std::string& label,
            const CipherTensor& tensor,
            std::size_t active_features) {
            const auto values = client_.Decrypt(tensor);
            inactive_zero.Observe(label, values, active_features);
        };
        const auto hidden = moai::openfhe::kPaperCompatHiddenSize;
        inactive_zero.Observe("encoder_input", input, hidden);
        observe_ciphertext("query", checkpoints.query, hidden);
        observe_ciphertext("key", checkpoints.key, hidden);
        observe_ciphertext("value", checkpoints.value, hidden);
        observe_ciphertext(
            "scaled_scores",
            checkpoints.attention.scaled_scores,
            hidden);
        inactive_zero.Observe("shifted_logits", shifted_logits, hidden);
        observe_ciphertext(
            "probabilities",
            checkpoints.attention.probabilities,
            hidden);
        inactive_zero.Observe(
            "attention_output_before_bootstrap_cleanup",
            attention_output,
            hidden);
        observe_ciphertext(
            "attention_output_after_bootstrap_cleanup",
            checkpoints.attention.output,
            hidden);
        observe_ciphertext(
            "self_projection",
            checkpoints.self_projection,
            hidden);
        observe_ciphertext(
            "attention_residual",
            checkpoints.attention_residual,
            hidden);
        inactive_zero.Observe(
            "attention_layernorm",
            attention_layernorm,
            hidden);
        inactive_zero.Observe("output_projection", output_projection, hidden);
        observe_ciphertext(
            "output_residual_before_bootstrap",
            checkpoints.output_residual_before_bootstrap,
            hidden);
        observe_ciphertext(
            "output_residual_after_bootstrap",
            checkpoints.output_residual_after_bootstrap,
            hidden);
        inactive_zero.Observe("encoder_output", output, hidden);
        for (std::size_t block = 0;
             block < moai::openfhe::kPaperCompatIntermediateBlocks;
             ++block) {
            const auto suffix = std::to_string(block);
            observe_ciphertext(
                "intermediate_pre_activation_" + suffix,
                checkpoints.intermediate_pre_activation[block],
                moai::openfhe::kPaperCompatFeatureBlock);
            observe_ciphertext(
                "intermediate_polynomial_output_" + suffix,
                checkpoints.intermediate_polynomial_output[block],
                moai::openfhe::kPaperCompatFeatureBlock);
            observe_ciphertext(
                "intermediate_activation_" + suffix,
                checkpoints.intermediate_activation[block],
                moai::openfhe::kPaperCompatFeatureBlock);
            observe_ciphertext(
                "output_contribution_" + suffix,
                checkpoints.output_contributions[block],
                hidden);
        }
        record.inactive_maximum = inactive_zero.Finish();
        record.inactive_zero_checkpoint_count = inactive_zero.count();
        // These three polynomial-input checkpoints deliberately use 1.0 in
        // inactive slots as the public reciprocal/inverse-square-root identity.
        // Gate deviation from that frozen sentinel rather than incorrectly
        // treating the slots as zero-valued activations.
        record.inactive_sentinel_maximum = std::max({
            MaximumInactiveDeviation(
                softmax_denominator,
                moai::openfhe::kPaperCompatHiddenSize,
                1.0),
            MaximumInactiveDeviation(
                attention_layernorm_variance,
                moai::openfhe::kPaperCompatHiddenSize,
                1.0),
            MaximumInactiveDeviation(
                output_layernorm_variance,
                moai::openfhe::kPaperCompatHiddenSize,
                1.0)});
        if (record.inactive_maximum > kInactiveMaximum) {
            std::ostringstream message;
            message << std::setprecision(17)
                    << "encrypted inactive/cross-lane gate exceeded 1e-6: "
                    << record.inactive_maximum;
            throw std::runtime_error(message.str());
        }

        record.refresh_counts = Difference(
            trace.metrics_after_refresh,
            trace.metrics_after_layer);
        record.layer_counts = Difference(
            trace.metrics_after_layer,
            trace.metrics_before_layer);
        record.cumulative_counts = Counts(trace.metrics_after_refresh);
        const OperationCounts expected_layer{
            6300, 51865, 95, 800, 55, 1150, 25, 50};
        const OperationCounts expected_refresh_counts = expected_refresh
            ? OperationCounts{0, 5, 0, 5, 0, 0, 5, 10}
            : OperationCounts{};
        RequireCounts(record.layer_counts, expected_layer, "encoder layer");
        RequireCounts(
            record.refresh_counts,
            expected_refresh_counts,
            "inter-layer refresh");
        const std::size_t completed_refreshes =
            layer + (expected_refresh ? 1 : 0);
        const auto expected_cumulative = AddCounts(
            MultiplyCounts(expected_layer, layer + 1),
            MultiplyCounts(
                OperationCounts{0, 5, 0, 5, 0, 0, 5, 10},
                completed_refreshes));
        RequireCounts(
            record.cumulative_counts,
            expected_cumulative,
            "encoder cumulative");
        if (trace.metrics_after_layer.multiplicative_depth != 47 ||
            trace.metrics_after_layer.max_observed_level != 45 ||
            trace.metrics_after_layer.max_polynomial_depth != 10) {
            throw std::runtime_error("encoder depth metadata drifted");
        }

        records_.push_back(record);
        const auto end = Clock::now();
        validation_ms_ += ElapsedMilliseconds(begin, end);
        PrintLayerRecord(records_.back(), run_kind_);
    }

    [[nodiscard]] const std::vector<LayerRecord>& records() const noexcept {
        return records_;
    }

    [[nodiscard]] double validation_ms() const noexcept {
        return validation_ms_;
    }

private:
    static void PrintQuality(const QualityMetrics& quality) {
        std::cout
            << "{\"relative_l2\":" << quality.relative_l2
            << ",\"cosine\":" << quality.cosine
            << ",\"max_absolute\":" << quality.maximum_absolute << '}';
    }

    static void PrintLayerRecord(const LayerRecord& record, RunKind run_kind) {
        std::cout
            << "{\"test\":\"" << LayerTestName(run_kind) << "\","
            << "\"profile\":\"paper_compat\",\"security_claim\":\"none\","
            << "\"layer_id\":" << record.layer_id;
        if (run_kind != RunKind::kFormalFull) {
            std::cout
                << ",\"claim_scope\":\""
                << ClaimScope(run_kind)
                << "\",\"artifact_eligible\":false,"
                << "\"formal_schedule_sealed\":"
                << (run_kind == RunKind::kMetadataCalibration ? "false" : "true")
                << ",\"metadata_validation_mode\":\""
                << (run_kind == RunKind::kMetadataCalibration
                        ? "calibration"
                        : "exact")
                << "\"";
        }
        std::cout
            << ",\"weights_layer_id\":" << record.layer_id
            << ",\"handoff_refresh_performed\":"
            << (record.handoff_refresh_performed ? "true" : "false")
            << ",\"chain_input_source\":\""
            << (record.layer_id == 0
                    ? "client_encrypted_trace_input"
                    : "previous_ciphertext_output_after_refresh")
            << "\",\"input_metadata\":";
        PrintMetadata(record.input_metadata);
        std::cout << ",\"raw_output_metadata\":";
        PrintMetadata(record.output_metadata);
        std::cout << ",\"softmax_denominator_metadata\":";
        PrintMetadata(record.denominator_metadata);
        std::cout << ",\"ln1_variance_metadata\":";
        PrintMetadata(record.ln1_variance_metadata);
        std::cout << ",\"ln2_variance_metadata\":";
        PrintMetadata(record.ln2_variance_metadata);
        std::cout << ",\"attention_output_metadata\":";
        PrintMetadata(record.attention_output_metadata);
        std::cout << ",\"ln1_output_metadata\":";
        PrintMetadata(record.ln1_output_metadata);
        std::cout << ",\"ffn_output_metadata\":";
        PrintMetadata(record.ffn_output_metadata);
        if (run_kind != RunKind::kFormalFull) {
            std::cout << ",\"metadata_used_levels\":";
            PrintMetadataUsedLevels(record.metadata_used_levels);
            std::cout << ",\"metadata_used_level_deltas\":";
            PrintMetadataUsedLevelDeltas(record.metadata_used_level_deltas);
        }
        std::cout << ",\"input_quality\":";
        PrintQuality(record.input_quality);
        std::cout << ",\"output_quality\":";
        PrintQuality(record.output_quality);
        std::cout << ",\"exact_trace_diagnostic\":";
        PrintQuality(record.exact_trace_quality);
        std::cout
            << ",\"inactive_max_abs\":" << record.inactive_maximum
            << ",\"inactive_zero_checkpoint_count\":"
            << record.inactive_zero_checkpoint_count
            << ",\"inactive_sentinel_max_error\":"
            << record.inactive_sentinel_maximum
            << ",\"inactive_polynomial_sentinel_ranges\":";
        PrintInactiveSentinelRanges(record.inactive_sentinel_ranges);
        std::cout
            << ",\"inactive_sentinel_range_status\":\"passed\""
            << ",\"encrypted_polynomial_input_ranges\":";
        PrintRanges(record.ranges);
        std::cout << ",\"refresh_operation_counts\":";
        PrintCounts(record.refresh_counts);
        std::cout << ",\"layer_operation_counts\":";
        PrintCounts(record.layer_counts);
        std::cout << ",\"cumulative_operation_counts\":";
        PrintCounts(record.cumulative_counts);
        std::cout
            << ",\"range_validation_owner\":\"client\","
            << "\"checkpoint_decryption_owner\":\"client\","
            << "\"server_decryptions\":0,\"server_plaintext_activations\":false,"
            << "\"finite\":true,\"range_status\":\"passed\"";
        if (run_kind != RunKind::kFormalFull) {
            std::cout << ",\"diagnostic_gates_passed\":true";
        }
        std::cout << '}'
            << std::endl;
    }

    moai::openfhe::ClientRuntime& client_;
    moai::openfhe::ServerRuntime& server_;
    const FrozenInputs& inputs_;
    moai::openfhe::PaperCompatNonlinearContracts contracts_;
    std::size_t evaluated_layer_count_{0};
    MetadataMode metadata_mode_{MetadataMode::kExact};
    RunKind run_kind_{RunKind::kFormalFull};
    std::vector<LayerRecord> records_;
    double validation_ms_{0.0};
};

void ValidateCompositePreflight(
    moai::openfhe::FeaturePackedEncoder& encoder,
    const CipherTensor& encrypted_input,
    std::vector<moai::openfhe::EncoderLayerWeights>& weights,
    moai::openfhe::ServerRuntime& server) {
    RequireArgumentRejection(
        [&encoder, &encrypted_input, &weights]() {
            static_cast<void>(encoder.EvaluatePrefixForDiagnostics(
                encrypted_input,
                weights,
                moai::openfhe::kPaperCompatEncoderLayers));
        },
        "12-layer diagnostic prefix API");
    RequireNoHomomorphicWork(
        server.metrics(),
        "12-layer diagnostic API preflight");

    auto missing_layer = std::move(weights.back());
    weights.pop_back();
    bool missing_layer_rejected = false;
    try {
        static_cast<void>(encoder.EvaluatePrefixForDiagnostics(
            encrypted_input,
            weights,
            1));
    }
    catch (const std::invalid_argument&) {
        missing_layer_rejected = true;
    }
    weights.push_back(std::move(missing_layer));
    if (!missing_layer_rejected) {
        throw std::runtime_error("11-layer weight bundle was not rejected");
    }
    RequireNoHomomorphicWork(server.metrics(), "weight-count preflight");

    weights.back().layer_index = 10;
    bool bad_order_rejected = false;
    try {
        static_cast<void>(encoder.EvaluatePrefixForDiagnostics(
            encrypted_input,
            weights,
            1));
    }
    catch (const std::invalid_argument&) {
        bad_order_rejected = true;
    }
    weights.back().layer_index = 11;
    if (!bad_order_rejected) {
        throw std::runtime_error("late duplicate layer id was not rejected");
    }
    RequireNoHomomorphicWork(server.metrics(), "late layer-id preflight");

    weights.back().query_bias.push_back(0.0);
    bool bad_shape_rejected = false;
    try {
        static_cast<void>(encoder.EvaluatePrefixForDiagnostics(
            encrypted_input,
            weights,
            1));
    }
    catch (const std::invalid_argument&) {
        bad_shape_rejected = true;
    }
    weights.back().query_bias.pop_back();
    if (!bad_shape_rejected) {
        throw std::runtime_error("late malformed weights were not rejected");
    }
    RequireNoHomomorphicWork(server.metrics(), "late weight-shape preflight");

    auto stale_input = encrypted_input;
    ++stale_input.packing.level;
    bool stale_input_rejected = false;
    try {
        static_cast<void>(encoder.EvaluatePrefixForDiagnostics(
            stale_input,
            weights,
            1));
    }
    catch (const std::invalid_argument&) {
        stale_input_rejected = true;
    }
    if (!stale_input_rejected) {
        throw std::runtime_error("stale initial ciphertext metadata was not rejected");
    }
    RequireNoHomomorphicWork(server.metrics(), "initial metadata preflight");

    auto wrong_slot_count = encrypted_input;
    --wrong_slot_count.packing.slot_count;
    bool wrong_slot_count_rejected = false;
    try {
        static_cast<void>(encoder.EvaluatePrefixForDiagnostics(
            wrong_slot_count,
            weights,
            1));
    }
    catch (const std::invalid_argument&) {
        wrong_slot_count_rejected = true;
    }
    if (!wrong_slot_count_rejected) {
        throw std::runtime_error("wrong encoder slot count was not rejected");
    }
    RequireNoHomomorphicWork(server.metrics(), "slot-count preflight");

    bool fresh_input_as_handoff_rejected = false;
    try {
        moai::openfhe::RequirePaperCompatInterLayerHandoff(
            encrypted_input,
            server);
    }
    catch (const std::runtime_error&) {
        fresh_input_as_handoff_rejected = true;
    }
    if (!fresh_input_as_handoff_rejected) {
        throw std::runtime_error(
            "fresh layer-0 input was accepted as an inter-layer handoff");
    }
    RequireNoHomomorphicWork(server.metrics(), "inter-layer handoff preflight");
}

void PrintPreflight(const FrozenInputs& inputs, double elapsed_ms) {
    const auto final_quality = MeasureQuality(
        inputs.polynomial_outputs.back(),
        inputs.exact_outputs.back(),
        moai::openfhe::kPaperCompatHiddenSize);
    std::cout
        << "{\"test\":\"openfhe_encoder_12_layer_preflight\","
        << "\"profile\":\"paper_compat\",\"security_claim\":\"none\","
        << "\"encoder_layers\":12,\"layer_ids\":[0,1,2,3,4,5,6,7,8,9,10,11],"
        << "\"chain_mode\":\"plaintext_output_to_next_oracle_input\","
        << "\"expected_fixture_files\":444,"
        << "\"metadata_schedule_preflight\":"
        << "\"synthetic_self_consistency_passed\","
        << "\"formal_metadata_schedule_source\":"
        << "\"r6_two_layer_live_metadata_calibration\","
        << "\"he_metadata_verified_by_preflight\":false,"
        << "\"trace_hash_gate\":\"separate_fail_closed_validator_required\","
        << "\"finite_and_in_range\":true,"
        << "\"final_relative_l2\":" << final_quality.relative_l2
        << ",\"final_cosine\":" << final_quality.cosine
        << ",\"final_max_absolute\":" << final_quality.maximum_absolute
        << ",\"load_and_oracle_ms\":" << elapsed_ms << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::setprecision(17);
        const auto arguments = ParseArguments(argc, argv);
        const auto fixture_begin = Clock::now();
        auto inputs = LoadFrozenInputs(arguments.data_root);
        const auto fixture_end = Clock::now();
        if (arguments.preflight_only) {
            ValidateDiagnosticArgumentContract();
            ValidateMetadataScheduleContract();
            ValidateInactiveZeroCheckpointContract();
            ValidateNoHomomorphicWorkContract();
            PrintPreflight(
                inputs,
                ElapsedMilliseconds(fixture_begin, fixture_end));
            return 0;
        }

        auto profile = moai::openfhe::MakePaperCompatFeaturePackedProfile();
        if (profile.parameter_sha256 != kExpectedProfileSha256 ||
            profile.security_claim != "none" ||
            profile.multiplicative_depth != 47) {
            throw std::runtime_error("M5 effective profile drifted");
        }
        moai::openfhe::PrintSecurityDisclosure(profile, std::cout);

        moai::openfhe::PackingSpec packing;
        packing.layout = moai::openfhe::PackingLayout::kContiguous;
        packing.logical_shape = {
            moai::openfhe::kPaperCompatFeatureBlock,
            moai::openfhe::kPaperCompatTraceTokens};
        packing.batch_lanes = 1;
        packing.slot_count = profile.slot_count;
        packing.active_slots = moai::openfhe::kPaperCompatFeatureBlock;
        packing.encoded_slots = moai::openfhe::kPaperCompatFeatureBlock;
        packing.level = 29;
        packing.noise_scale_degree = 1;
        packing.scaling_factor =
            std::ldexp(1.0, static_cast<int>(profile.scaling_modulus_bits));

        const auto setup_begin = Clock::now();
        moai::openfhe::ClientRuntime client(profile);
        client.GenerateEvaluationKeys(
            moai::openfhe::FeaturePackedEncoderRotationIndices(),
            true);
        const auto setup_end = Clock::now();
        const auto encrypt_begin = Clock::now();
        const auto encrypted_input = client.Encrypt(inputs.initial_input, packing);
        const auto encrypt_end = Clock::now();

        moai::openfhe::ServerRuntime server(
            client.ExportServerKeyBundle(),
            profile);
        moai::openfhe::FeaturePackedEncoder encoder(server);
        ValidateCompositePreflight(
            encoder,
            encrypted_input,
            inputs.weights,
            server);
        if (arguments.crypto_preflight_only) {
            const auto input_metadata = InspectTensor(
                encrypted_input,
                server,
                moai::openfhe::kPaperCompatTraceTokens,
                kSingleScaleBits,
                "initial encrypted input");
            RequireMetadata(
                input_metadata,
                kLayer0InputMetadata,
                "initial encrypted input");
            RequireNoHomomorphicWork(server.metrics(), "crypto preflight");
            std::cout
                << "{\"test\":\"openfhe_encoder_12_layer_crypto_preflight\","
                << "\"profile\":\"paper_compat\",\"security_claim\":\"none\","
                << "\"input_metadata\":";
            PrintMetadata(input_metadata);
            std::cout
                << ",\"server_private_key_present\":false,"
                << "\"server_decryptions\":0,\"server_plaintext_activations\":false,"
                << "\"additive_he_operations\":0,\"passed\":true}"
                << std::endl;
            return 0;
        }
        const std::size_t layer_count = EvaluatedLayerCount(arguments);
        const RunKind run_kind = ResolveRunKind(arguments);
        const MetadataMode metadata_mode = ResolveMetadataMode(arguments);
        ClientCheckpointValidator validator(
            client,
            server,
            inputs,
            layer_count,
            metadata_mode,
            run_kind);

        const auto evaluation_begin = Clock::now();
        auto result = run_kind == RunKind::kFormalFull
            ? encoder.Evaluate(encrypted_input, inputs.weights, &validator)
            : encoder.EvaluatePrefixForDiagnostics(
                encrypted_input,
                inputs.weights,
                layer_count,
                &validator);
        const auto evaluation_end = Clock::now();
        if (validator.records().size() != layer_count || result.output.empty()) {
            throw std::runtime_error("encoder returned an incomplete ciphertext chain");
        }
        const auto final_metadata = InspectTensor(
            result.output,
            server,
            5,
            kDoubleScaleBits,
            "final output");
        const auto& final_record = validator.records().back();
        if (metadata_mode == MetadataMode::kExact) {
            RequireMetadata(
                final_metadata,
                kEncoderRawOutputMetadata,
                "final output");
        }
        else {
            RequireLiveObservationEquivalent(
                final_metadata,
                final_record.output_metadata,
                "final output");
        }
        RequireQuality(final_record.output_quality, "final encrypted output");
        double global_inactive_maximum = 0.0;
        double global_inactive_sentinel_maximum = 0.0;
        for (const auto& record : validator.records()) {
            global_inactive_maximum = std::max(
                global_inactive_maximum,
                record.inactive_maximum);
            global_inactive_sentinel_maximum = std::max(
                global_inactive_sentinel_maximum,
                record.inactive_sentinel_maximum);
        }
        if (global_inactive_maximum > kInactiveMaximum) {
            std::ostringstream message;
            message << std::setprecision(17)
                    << "global inactive gate failed: "
                    << global_inactive_maximum;
            throw std::runtime_error(message.str());
        }

        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) != 0) {
            throw std::runtime_error("getrusage failed");
        }
        const double evaluation_ms =
            ElapsedMilliseconds(evaluation_begin, evaluation_end);
        const double server_online_ms = evaluation_ms - validator.validation_ms();
        if (server_online_ms <= 0.0) {
            throw std::runtime_error("server diagnostic time is not positive");
        }
        const auto total_counts = Counts(server.metrics());
        const OperationCounts expected_layer{
            6300, 51865, 95, 800, 55, 1150, 25, 50};
        const OperationCounts expected_refresh{0, 5, 0, 5, 0, 0, 5, 10};
        const auto expected_total = AddCounts(
            MultiplyCounts(expected_layer, layer_count),
            MultiplyCounts(expected_refresh, layer_count - 1));
        RequireCounts(total_counts, expected_total, "final encoder");

        std::cout
            << "{\"test\":\"" << SummaryTestName(run_kind) << "\","
            << "\"profile\":\"paper_compat\",\"security_claim\":\"none\","
            << "\"parameter_sha256\":\"" << profile.parameter_sha256 << "\","
            << "\"execution_mode\":\"server-only\","
            << "\"claim_scope\":\"" << ClaimScope(run_kind) << "\"";
        if (run_kind != RunKind::kFormalFull) {
            std::cout
                << ",\"artifact_eligible\":false,"
                << "\"formal_schedule_sealed\":"
                << (run_kind == RunKind::kMetadataCalibration ? "false" : "true")
                << ",\"metadata_validation_mode\":\""
                << (run_kind == RunKind::kMetadataCalibration
                        ? "calibration"
                        : "exact")
                << "\"";
        }
        std::cout
            << ",\"encoder_layers\":" << layer_count
            << ",\"chain_mode\":\"ciphertext_output_to_next_input\","
            << "\"client_encrypt_calls\":1,\"encrypted_input_ciphertexts\":5,"
            << "\"plaintext_activation_resets\":0,\"server_layer_evaluations\":"
            << layer_count << ",\"inter_layer_refreshes\":"
            << (layer_count - 1)
            << ",\"checkpoint_decryption_owner\":\"client\","
            << "\"final_decryption_owner\":\"client\","
            << "\"server_private_key_present\":false,\"server_decryptions\":0,"
            << "\"server_plaintext_activations\":false,"
            << "\"approximation_range_status\":\"all_client_validated\","
            << "\"fixture_load_oracle_ms\":"
            << ElapsedMilliseconds(fixture_begin, fixture_end)
            << ",\"setup_keygen_ms\":"
            << ElapsedMilliseconds(setup_begin, setup_end)
            << ",\"client_encrypt_ms\":"
            << ElapsedMilliseconds(encrypt_begin, encrypt_end)
            << ",\"server_online_diagnostic_ms\":" << server_online_ms
            << ",\"client_checkpoint_validate_ms\":"
            << validator.validation_ms()
            << ",\"relative_l2\":" << final_record.output_quality.relative_l2
            << ",\"cosine\":" << final_record.output_quality.cosine
            << ",\"max_absolute\":"
            << final_record.output_quality.maximum_absolute
            << ",\"inactive_max_abs\":" << global_inactive_maximum
            << ",\"inactive_sentinel_max_error\":"
            << global_inactive_sentinel_maximum
            << ",\"inactive_sentinel_range_status\":"
            << "\"all_client_validated\""
            << ",\"final_metadata\":";
        PrintMetadata(final_metadata);
        std::cout << ",\"operation_counts\":";
        PrintCounts(total_counts);
        std::cout
            << ",\"multiplicative_depth\":47,\"max_observed_level\":45,"
            << "\"max_polynomial_depth\":10,\"peak_rss_bytes\":"
            << static_cast<unsigned long long>(usage.ru_maxrss) * 1024ULL
            << ",\"timing_claim\":false,"
            << "\"latency_kind\":\"non_benchmark_diagnostic\","
            << "\"finite\":true,\"passed\":true";
        if (run_kind != RunKind::kFormalFull) {
            std::cout << ",\"diagnostic_gates_passed\":true";
        }
        std::cout << '}' << std::endl;
        return 0;
    }
    catch (const std::exception& exception) {
        std::cerr << "openfhe_encoder_12_layer_smoke failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
