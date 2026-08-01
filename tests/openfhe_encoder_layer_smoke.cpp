#include "moai/openfhe/approximation_registry.hpp"
#include "moai/openfhe/client_runtime.hpp"
#include "moai/openfhe/context_factory.hpp"
#include "moai/openfhe/encoder_layer.hpp"
#include "support/moai_encoder_fixture.hpp"
#include "support/moai_encoder_plaintext_oracle.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace moai::openfhe {

class ServerKeyBundleTestAccess {
public:
    static void RemoveDeclaredRotation(
        ServerKeyBundle& bundle,
        int32_t rotation) {
        const auto iterator = std::lower_bound(
            bundle.rotation_indices.begin(),
            bundle.rotation_indices.end(),
            rotation);
        if (iterator == bundle.rotation_indices.end() ||
            *iterator != rotation) {
            throw std::runtime_error(
                "encoder preflight test rotation is not declared");
        }
        bundle.rotation_indices.erase(iterator);
    }

    static void RemoveBootstrapCapability(ServerKeyBundle& bundle) {
        if (bundle.bootstrap_required_indices.empty()) {
            throw std::runtime_error(
                "encoder preflight test bootstrap capability is absent");
        }
        bundle.bootstrap_required_indices.clear();
    }
};

}  // namespace moai::openfhe

namespace {

using Clock = std::chrono::steady_clock;
using PlainMatrix = moai::openfhe::test::PlainMatrix;

constexpr const char* kExpectedProfileSha256 =
    "94f30e628e21f02146ce7ed9820194eabba3820f6e1e17176a31f8c5acf8b0be";
constexpr int kExpectedArtifactScaleBits = 100;
constexpr double kArtifactScaleBitsTolerance = 1e-3;

struct QualityMetrics {
    double relative_l2{0.0};
    double cosine{0.0};
    double max_absolute{0.0};
};

void RequireQuality(
    const QualityMetrics& quality,
    double maximum_relative_l2,
    double minimum_cosine,
    const std::string& label);

double ElapsedMilliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void RequireNoHomomorphicWork(
    const moai::openfhe::RunMetrics& metrics,
    const std::string& label) {
    // multiplicative_depth is immutable profile metadata, not executed work.
    if (metrics.relative_l2 != 0.0 ||
        metrics.cosine_similarity != 0.0 ||
        metrics.max_absolute_error != 0.0 ||
        metrics.latency_ms != 0.0 ||
        metrics.peak_rss_bytes != 0 ||
        metrics.rotations != 0 ||
        metrics.ct_pt_multiplications != 0 ||
        metrics.ct_ct_multiplications != 0 ||
        metrics.rescale_operations != 0 ||
        metrics.bootstraps != 0 ||
        metrics.bootstrap_iterations != 0 ||
        metrics.chebyshev_evaluations != 0 ||
        metrics.estimated_polynomial_multiplications != 0 ||
        metrics.max_observed_level != 0 ||
        metrics.max_polynomial_depth != 0) {
        throw std::runtime_error(
            label + " did not fail before homomorphic work");
    }
}

QualityMetrics MeasureQuality(
    const PlainMatrix& actual,
    const PlainMatrix& expected,
    std::size_t active_features) {
    if (actual.size() != expected.size() || actual.empty()) {
        throw std::runtime_error("encoder checkpoint row count drifted");
    }
    long double squared_error = 0.0;
    long double squared_actual = 0.0;
    long double squared_expected = 0.0;
    long double dot = 0.0;
    double max_absolute = 0.0;
    for (std::size_t row = 0; row < actual.size(); ++row) {
        if (actual[row].size() != expected[row].size() ||
            active_features > actual[row].size()) {
            throw std::runtime_error("encoder checkpoint column count drifted");
        }
        for (std::size_t feature = 0; feature < active_features; ++feature) {
            if (!std::isfinite(actual[row][feature]) ||
                !std::isfinite(expected[row][feature])) {
                throw std::runtime_error("encoder checkpoint contains NaN or Inf");
            }
            const long double observed = actual[row][feature];
            const long double reference = expected[row][feature];
            const long double error = observed - reference;
            squared_error += error * error;
            squared_actual += observed * observed;
            squared_expected += reference * reference;
            dot += observed * reference;
            max_absolute = std::max(
                max_absolute,
                std::abs(static_cast<double>(error)));
        }
    }
    if (squared_actual == 0.0 || squared_expected == 0.0) {
        throw std::runtime_error("encoder checkpoint quality norm is zero");
    }
    return {
        std::sqrt(static_cast<double>(squared_error / squared_expected)),
        static_cast<double>(
            dot / std::sqrt(squared_actual * squared_expected)),
        max_absolute};
}

PlainMatrix DenseOracle(
    const PlainMatrix& input,
    const moai::openfhe::FeaturePackedWeights& weights,
    const std::vector<double>& bias) {
    if (input.empty() || weights.empty() || bias.empty() ||
        weights.front().size() != bias.size()) {
        throw std::invalid_argument("plaintext dense oracle dimensions are invalid");
    }
    PlainMatrix output(input.size(), bias);
    for (std::size_t row = 0; row < input.size(); ++row) {
        if (input[row].size() < weights.size()) {
            throw std::invalid_argument("plaintext dense oracle input is too short");
        }
        for (std::size_t in = 0; in < weights.size(); ++in) {
            if (weights[in].size() != bias.size()) {
                throw std::invalid_argument("plaintext dense oracle weights are ragged");
            }
            for (std::size_t out = 0; out < bias.size(); ++out) {
                output[row][out] += input[row][in] * weights[in][out];
            }
        }
    }
    return output;
}

void AddInPlace(PlainMatrix& output, const PlainMatrix& contribution) {
    if (output.size() != contribution.size()) {
        throw std::invalid_argument("plaintext contribution row count drifted");
    }
    for (std::size_t row = 0; row < output.size(); ++row) {
        if (output[row].size() != contribution[row].size()) {
            throw std::invalid_argument("plaintext contribution width drifted");
        }
        for (std::size_t feature = 0; feature < output[row].size(); ++feature) {
            output[row][feature] += contribution[row][feature];
        }
    }
}

void ValidateFixturePreflight(
    const moai::openfhe::test::EncoderLayerFixture& fixture) {
    const auto& weights = fixture.weights;
    const auto& expected = fixture.expected;
    RequireQuality(
        MeasureQuality(
            moai::openfhe::test::PadFeatureRows(
                DenseOracle(
                    expected.input,
                    weights.query_weights,
                    weights.query_bias),
                moai::openfhe::kPaperCompatHiddenSize),
            expected.query,
            moai::openfhe::kPaperCompatHiddenSize),
        5e-7,
        0.999999,
        "fixture query projection");
    RequireQuality(
        MeasureQuality(
            moai::openfhe::test::PadFeatureRows(
                DenseOracle(
                    expected.input,
                    weights.key_weights,
                    weights.key_bias),
                moai::openfhe::kPaperCompatHiddenSize),
            expected.key,
            moai::openfhe::kPaperCompatHiddenSize),
        5e-7,
        0.999999,
        "fixture key projection");
    RequireQuality(
        MeasureQuality(
            moai::openfhe::test::PadFeatureRows(
                DenseOracle(
                    expected.input,
                    weights.value_weights,
                    weights.value_bias),
                moai::openfhe::kPaperCompatHiddenSize),
            expected.value,
            moai::openfhe::kPaperCompatHiddenSize),
        5e-7,
        0.999999,
        "fixture value projection");
    RequireQuality(
        MeasureQuality(
            moai::openfhe::test::PadFeatureRows(
                DenseOracle(
                    expected.attention,
                    weights.self_output_weights,
                    weights.self_output_bias),
                moai::openfhe::kPaperCompatHiddenSize),
            expected.self_projection,
            moai::openfhe::kPaperCompatHiddenSize),
        1e-6,
        0.999999,
        "fixture self projection");

    for (std::size_t block = 0;
         block < moai::openfhe::kPaperCompatIntermediateBlocks;
         ++block) {
        RequireQuality(
            MeasureQuality(
                DenseOracle(
                    expected.attention_layernorm,
                    weights.intermediate_weight_blocks[block],
                    weights.intermediate_bias_blocks[block]),
                moai::openfhe::test::SliceIntermediateBlock(
                    expected.intermediate_pre_activation,
                    block),
                moai::openfhe::kPaperCompatFeatureBlock),
            4e-7,
            0.999999,
            "fixture intermediate projection");
    }

    PlainMatrix output_projection(
        moai::openfhe::kPaperCompatTraceTokens,
        weights.output_bias);
    for (std::size_t block = 0;
         block < moai::openfhe::kPaperCompatIntermediateBlocks;
         ++block) {
        AddInPlace(
            output_projection,
            DenseOracle(
                moai::openfhe::test::SliceIntermediateBlock(
                    expected.intermediate_activation,
                    block),
                weights.output_weight_blocks[block],
                std::vector<double>(
                    moai::openfhe::kPaperCompatHiddenSize,
                    0.0)));
    }
    RequireQuality(
        MeasureQuality(
            moai::openfhe::test::PadFeatureRows(
                output_projection,
                moai::openfhe::kPaperCompatHiddenSize),
            expected.output_projection,
            moai::openfhe::kPaperCompatHiddenSize),
        5e-7,
        0.999999,
        "fixture output projection");
}

double MaximumInactive(
    const PlainMatrix& values,
    std::size_t active_features) {
    double maximum = 0.0;
    for (const auto& row : values) {
        if (active_features > row.size()) {
            throw std::runtime_error("inactive-slot boundary exceeds row size");
        }
        for (std::size_t slot = active_features; slot < row.size(); ++slot) {
            if (!std::isfinite(row[slot])) {
                throw std::runtime_error("inactive encoder slot contains NaN or Inf");
            }
            maximum = std::max(maximum, std::abs(row[slot]));
        }
    }
    return maximum;
}

struct SentinelRange {
    double minimum{std::numeric_limits<double>::infinity()};
    double maximum{-std::numeric_limits<double>::infinity()};
};

SentinelRange MeasureInactiveSentinelRange(
    const PlainMatrix& values,
    std::size_t active_features,
    const std::string& label) {
    if (values.empty()) {
        throw std::runtime_error(label + " is empty");
    }
    SentinelRange result;
    for (const auto& row : values) {
        if (row.size() != moai::openfhe::kPaperCompatFeatureBlock ||
            active_features >= row.size()) {
            throw std::runtime_error(label + " width changed");
        }
        for (std::size_t slot = active_features; slot < row.size(); ++slot) {
            if (!std::isfinite(row[slot])) {
                throw std::runtime_error(label + " contains NaN or Inf");
            }
            result.minimum = std::min(result.minimum, row[slot]);
            result.maximum = std::max(result.maximum, row[slot]);
        }
    }
    return result;
}

void RequireValuesInDeclaredRange(
    const PlainMatrix& values,
    const moai::openfhe::DeclaredRange& interval,
    const std::string& label) {
    if (values.empty() || !std::isfinite(interval.minimum) ||
        !std::isfinite(interval.maximum) || interval.minimum > interval.maximum) {
        throw std::runtime_error(label + " range contract is invalid");
    }
    for (const auto& row : values) {
        if (row.size() != moai::openfhe::kPaperCompatFeatureBlock) {
            throw std::runtime_error(label + " width changed");
        }
        for (double value : row) {
            if (!std::isfinite(value) || value < interval.minimum ||
                value > interval.maximum) {
                throw std::runtime_error(label + " escaped its frozen interval");
            }
        }
    }
}

double MeasureLayerNormInactiveGuardDeviation(
    const PlainMatrix& values,
    const std::string& label) {
    if (values.empty()) {
        throw std::runtime_error(label + " is empty");
    }
    double maximum = 0.0;
    for (const auto& row : values) {
        if (row.size() != moai::openfhe::kPaperCompatFeatureBlock) {
            throw std::runtime_error(label + " width changed");
        }
        for (std::size_t slot = moai::openfhe::kPaperCompatHiddenSize;
             slot < row.size();
             ++slot) {
            if (!std::isfinite(row[slot])) {
                throw std::runtime_error(label + " contains NaN or Inf");
            }
            maximum = std::max(maximum, std::abs(row[slot] - 1.0));
        }
    }
    return maximum;
}

template <typename Operation>
void RequireRuntimeRejection(Operation&& operation, const std::string& label) {
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

void ValidateInactiveSentinelContract() {
    const auto contracts =
        moai::openfhe::MakePaperCompatNonlinearContracts();
    PlainMatrix layernorm_guard(
        1,
        std::vector<double>(
            moai::openfhe::kPaperCompatFeatureBlock,
            17.0));
    RequireValuesInDeclaredRange(
        layernorm_guard,
        contracts.layernorm_inverse_sqrt.interval,
        "synthetic LayerNorm inactive guard");
    static_cast<void>(MeasureLayerNormInactiveGuardDeviation(
        layernorm_guard,
        "synthetic LayerNorm inactive guard"));

    auto below_domain = layernorm_guard;
    below_domain.front().back() =
        std::nextafter(
            contracts.layernorm_inverse_sqrt.interval.minimum,
            -std::numeric_limits<double>::infinity());
    RequireRuntimeRejection(
        [&below_domain, &contracts]() {
            RequireValuesInDeclaredRange(
                below_domain,
                contracts.layernorm_inverse_sqrt.interval,
                "synthetic low LayerNorm inactive guard");
        },
        "low LayerNorm inactive guard");

    auto above_domain = layernorm_guard;
    above_domain.front().back() =
        std::nextafter(
            contracts.layernorm_inverse_sqrt.interval.maximum,
            std::numeric_limits<double>::infinity());
    RequireRuntimeRejection(
        [&above_domain, &contracts]() {
            RequireValuesInDeclaredRange(
                above_domain,
                contracts.layernorm_inverse_sqrt.interval,
                "synthetic high LayerNorm inactive guard");
        },
        "high LayerNorm inactive guard");

    auto non_finite_domain = layernorm_guard;
    non_finite_domain.front().back() =
        std::numeric_limits<double>::quiet_NaN();
    RequireRuntimeRejection(
        [&non_finite_domain, &contracts]() {
            RequireValuesInDeclaredRange(
                non_finite_domain,
                contracts.layernorm_inverse_sqrt.interval,
                "synthetic non-finite LayerNorm inactive guard");
        },
        "non-finite LayerNorm inactive guard");

    auto wrong_width_domain = layernorm_guard;
    wrong_width_domain.front().pop_back();
    RequireRuntimeRejection(
        [&wrong_width_domain, &contracts]() {
            RequireValuesInDeclaredRange(
                wrong_width_domain,
                contracts.layernorm_inverse_sqrt.interval,
                "synthetic wrong-width LayerNorm inactive guard");
        },
        "wrong-width LayerNorm inactive guard");

    PlainMatrix identity(
        1,
        std::vector<double>(
            moai::openfhe::kPaperCompatFeatureBlock,
            1.0));
    RequireValuesInDeclaredRange(
        identity,
        contracts.softmax_reciprocal.interval,
        "synthetic Softmax denominator");
    auto escaped_reciprocal = identity;
    escaped_reciprocal.front().front() =
        contracts.softmax_reciprocal.interval.minimum / 2.0;
    RequireRuntimeRejection(
        [&escaped_reciprocal, &contracts]() {
            RequireValuesInDeclaredRange(
                escaped_reciprocal,
                contracts.softmax_reciprocal.interval,
                "synthetic escaped Softmax denominator");
        },
        "Softmax reciprocal interval escape");
}

void RequireQuality(
    const QualityMetrics& quality,
    double maximum_relative_l2,
    double minimum_cosine,
    const std::string& label) {
    if (!std::isfinite(quality.relative_l2) ||
        !std::isfinite(quality.cosine) ||
        !std::isfinite(quality.max_absolute) ||
        quality.relative_l2 > maximum_relative_l2 ||
        quality.cosine < minimum_cosine) {
        throw std::runtime_error(
            label + " failed rel-L2/cosine acceptance gate");
    }
}

PlainMatrix ExpandHeadKeyTrace(const PlainMatrix& compact) {
    if (compact.size() != moai::openfhe::kPaperCompatTraceTokens) {
        throw std::invalid_argument("compact attention trace query count drifted");
    }
    PlainMatrix expanded(
        moai::openfhe::kPaperCompatTraceTokens *
            moai::openfhe::kPaperCompatTraceTokens,
        std::vector<double>(moai::openfhe::kPaperCompatFeatureBlock, 0.0));
    for (std::size_t query = 0;
         query < moai::openfhe::kPaperCompatTraceTokens;
         ++query) {
        if (compact[query].size() !=
            moai::openfhe::kPaperCompatAttentionHeadCount *
                moai::openfhe::kPaperCompatTraceTokens) {
            throw std::invalid_argument("compact attention trace width drifted");
        }
        for (std::size_t key = 0;
             key < moai::openfhe::kPaperCompatTraceTokens;
             ++key) {
            auto& row = expanded[
                query * moai::openfhe::kPaperCompatTraceTokens + key];
            for (std::size_t head = 0;
                 head < moai::openfhe::kPaperCompatAttentionHeadCount;
                 ++head) {
                const double value = compact[query][
                    head * moai::openfhe::kPaperCompatTraceTokens + key];
                std::fill_n(
                    row.begin() + static_cast<std::ptrdiff_t>(
                        head * moai::openfhe::kPaperCompatAttentionHeadDimension),
                    moai::openfhe::kPaperCompatAttentionHeadDimension,
                    value);
            }
        }
    }
    return expanded;
}

void RequirePacking(
    const moai::openfhe::CipherTensor& tensor,
    std::size_t ciphertexts,
    const std::string& label) {
    if (tensor.size() != ciphertexts ||
        tensor.packing.layout != moai::openfhe::PackingLayout::kContiguous ||
        tensor.packing.logical_shape != std::vector<std::size_t>{
            moai::openfhe::kPaperCompatFeatureBlock,
            ciphertexts} ||
        tensor.packing.active_slots !=
            moai::openfhe::kPaperCompatFeatureBlock ||
        tensor.packing.encoded_slots !=
            moai::openfhe::kPaperCompatFeatureBlock ||
        tensor.packing.noise_scale_degree == 0 ||
        !std::isfinite(tensor.packing.scaling_factor) ||
        tensor.packing.scaling_factor <= 0.0) {
        throw std::runtime_error(label + " packing metadata drifted");
    }
}

void PrintQuality(const std::string& label, const QualityMetrics& quality) {
    std::cout
        << "{\"checkpoint\":\"" << label << "\","
        << "\"relative_l2\":" << quality.relative_l2 << ','
        << "\"cosine\":" << quality.cosine << ','
        << "\"max_absolute\":" << quality.max_absolute << "}\n";
}

void PrintPacking(
    const std::string& label,
    const moai::openfhe::CipherTensor& tensor,
    const moai::openfhe::ServerRuntime& server) {
    std::cout
        << "{\"packing_checkpoint\":\"" << label << "\","
        << "\"ciphertexts\":" << tensor.size() << ','
        << "\"level\":" << tensor.packing.level << ','
        << "\"noise_scale_degree\":"
        << tensor.packing.noise_scale_degree << ','
        << "\"scaling_factor\":" << tensor.packing.scaling_factor << ','
        << "\"remaining_levels\":" << server.RemainingLevels(tensor)
        << "}\n";
}

void RequireCheckpointMetadata(
    const std::string& name,
    const moai::openfhe::CipherTensor& tensor,
    const moai::openfhe::ServerRuntime& server,
    uint32_t expected_level,
    uint32_t expected_noise_scale_degree,
    uint32_t expected_remaining_levels,
    std::size_t expected_ciphertext_count) {
    const double scale_bits = std::log2(tensor.packing.scaling_factor);
    const uint32_t remaining_levels = server.RemainingLevels(tensor);
    if (!std::isfinite(scale_bits) ||
        std::abs(scale_bits - kExpectedArtifactScaleBits) >
            kArtifactScaleBitsTolerance ||
        tensor.packing.level != expected_level ||
        tensor.packing.noise_scale_degree != expected_noise_scale_degree ||
        remaining_levels != expected_remaining_levels ||
        tensor.size() != expected_ciphertext_count) {
        std::ostringstream message;
        message << std::setprecision(17) << name
                << " artifact checkpoint metadata drifted: actual=("
                << tensor.packing.level << ','
                << tensor.packing.noise_scale_degree << ','
                << remaining_levels << ',' << scale_bits << ','
                << tensor.size() << ") expected=(" << expected_level << ','
                << expected_noise_scale_degree << ','
                << expected_remaining_levels << ','
                << kExpectedArtifactScaleBits << ','
                << expected_ciphertext_count << ')';
        throw std::runtime_error(message.str());
    }
}

void PrintArtifactCheckpoint(
    const std::string& name,
    const moai::openfhe::CipherTensor& tensor,
    const moai::openfhe::ServerRuntime& server,
    uint32_t expected_level,
    uint32_t expected_noise_scale_degree,
    uint32_t expected_remaining_levels,
    std::size_t expected_ciphertext_count) {
    RequireCheckpointMetadata(
        name,
        tensor,
        server,
        expected_level,
        expected_noise_scale_degree,
        expected_remaining_levels,
        expected_ciphertext_count);
    const uint32_t remaining_levels = server.RemainingLevels(tensor);
    std::cout
        << "{\"name\":\"" << name << "\","
        << "\"level\":" << tensor.packing.level << ','
        << "\"noise_scale_degree\":"
        << tensor.packing.noise_scale_degree << ','
        << "\"remaining_levels\":" << remaining_levels << ','
        << "\"scale_bits\":" << kExpectedArtifactScaleBits << ','
        << "\"ciphertext_count\":" << tensor.size() << ','
        << "\"decryption_owner\":\"client\"}";
}

struct Arguments {
    std::filesystem::path data_root{"data"};
    std::size_t layer{0};
    uint32_t input_level{0};
    bool preflight_only{false};
    bool attention_only{false};
};

Arguments ParseArguments(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--data-root" && index + 1 < argc) {
            arguments.data_root = argv[++index];
        }
        else if (argument == "--layer" && index + 1 < argc) {
            arguments.layer = static_cast<std::size_t>(
                std::stoul(argv[++index]));
        }
        else if (argument == "--input-level" && index + 1 < argc) {
            arguments.input_level = static_cast<uint32_t>(
                std::stoul(argv[++index]));
        }
        else if (argument == "--preflight-only") {
            arguments.preflight_only = true;
        }
        else if (argument == "--attention-only") {
            arguments.attention_only = true;
        }
        else {
            throw std::invalid_argument("unknown or incomplete argument: " + argument);
        }
    }
    return arguments;
}

void RunAttentionOnlyDiagnostic(
    moai::openfhe::ClientRuntime& client,
    moai::openfhe::ServerRuntime& server,
    const moai::openfhe::CipherTensor& encrypted_input,
    const moai::openfhe::test::EncoderLayerFixture& fixture) {
    const moai::openfhe::FeaturePackedAffineSpec hidden_spec{
        moai::openfhe::kPaperCompatFeatureBlock,
        moai::openfhe::kPaperCompatHiddenSize,
        moai::openfhe::kPaperCompatHiddenSize,
        moai::openfhe::kPaperCompatAffineBabyStep};
    std::vector<double> hidden_mask(
        moai::openfhe::kPaperCompatFeatureBlock,
        0.0);
    std::fill_n(
        hidden_mask.begin(),
        moai::openfhe::kPaperCompatHiddenSize,
        1.0);
    server.RequireEvaluationKeys(
        moai::openfhe::FeaturePackedEncoderRotationIndices(),
        true);
    moai::openfhe::FeaturePackedOps affine(server);
    const auto query = affine.DenseAffine(
        encrypted_input,
        fixture.weights.query_weights,
        fixture.weights.query_bias,
        hidden_mask,
        hidden_spec);
    const auto key = affine.DenseAffine(
        encrypted_input,
        fixture.weights.key_weights,
        fixture.weights.key_bias,
        hidden_mask,
        hidden_spec);
    const auto value = affine.DenseAffine(
        encrypted_input,
        fixture.weights.value_weights,
        fixture.weights.value_bias,
        hidden_mask,
        hidden_spec);
    moai::openfhe::FeaturePackedAttention attention(server);
    const auto result = attention.Evaluate(
        query,
        key,
        value,
        fixture.weights.layer_index,
        std::vector<double>(moai::openfhe::kPaperCompatTraceTokens, 1.0),
        moai::openfhe::MakePaperCompatFeaturePackedAttentionSpec());

    const auto decrypted_query = client.Decrypt(query);
    const auto decrypted_key = client.Decrypt(key);
    const auto decrypted_value = client.Decrypt(value);
    const auto decrypted_scores = client.Decrypt(result.scaled_scores);
    const auto decrypted_probabilities = client.Decrypt(result.probabilities);
    const auto decrypted_raw_output = client.Decrypt(
        result.output_before_bootstrap_cleanup);
    const auto decrypted_output = client.Decrypt(result.output);
    const auto query_quality = MeasureQuality(
        decrypted_query,
        fixture.expected.query,
        moai::openfhe::kPaperCompatHiddenSize);
    const auto key_quality = MeasureQuality(
        decrypted_key,
        fixture.expected.key,
        moai::openfhe::kPaperCompatHiddenSize);
    const auto value_quality = MeasureQuality(
        decrypted_value,
        fixture.expected.value,
        moai::openfhe::kPaperCompatHiddenSize);
    const auto score_quality = MeasureQuality(
        decrypted_scores,
        ExpandHeadKeyTrace(fixture.expected.scaled_scores),
        moai::openfhe::kPaperCompatHiddenSize);
    const auto probability_quality = MeasureQuality(
        decrypted_probabilities,
        ExpandHeadKeyTrace(fixture.expected.probabilities),
        moai::openfhe::kPaperCompatHiddenSize);
    const auto raw_output_quality = MeasureQuality(
        decrypted_raw_output,
        fixture.expected.attention,
        moai::openfhe::kPaperCompatHiddenSize);
    const auto output_quality = MeasureQuality(
        decrypted_output,
        fixture.expected.attention,
        moai::openfhe::kPaperCompatHiddenSize);
    RequireQuality(query_quality, 1e-4, 0.99999, "attention-only query");
    RequireQuality(key_quality, 1e-4, 0.99999, "attention-only key");
    RequireQuality(value_quality, 1e-4, 0.99999, "attention-only value");
    RequireQuality(score_quality, 1e-4, 0.99999, "attention-only scores");
    RequireQuality(
        probability_quality,
        1e-2,
        0.999,
        "attention-only probabilities");
    RequireQuality(
        raw_output_quality,
        1e-2,
        0.999,
        "attention-only output before bootstrap cleanup");
    RequireQuality(output_quality, 1e-2, 0.999, "attention-only output");

    const double query_inactive = MaximumInactive(
        decrypted_query,
        moai::openfhe::kPaperCompatHiddenSize);
    const double key_inactive = MaximumInactive(
        decrypted_key,
        moai::openfhe::kPaperCompatHiddenSize);
    const double value_inactive = MaximumInactive(
        decrypted_value,
        moai::openfhe::kPaperCompatHiddenSize);
    const double score_inactive = MaximumInactive(
        decrypted_scores,
        moai::openfhe::kPaperCompatHiddenSize);
    const double probability_inactive = MaximumInactive(
        decrypted_probabilities,
        moai::openfhe::kPaperCompatHiddenSize);
    const double raw_output_inactive = MaximumInactive(
        decrypted_raw_output,
        moai::openfhe::kPaperCompatHiddenSize);
    const double output_inactive = MaximumInactive(
        decrypted_output,
        moai::openfhe::kPaperCompatHiddenSize);
    const double inactive_maximum = std::max({
        query_inactive,
        key_inactive,
        value_inactive,
        score_inactive,
        probability_inactive,
        raw_output_inactive,
        output_inactive});
    if (inactive_maximum > 1e-6) {
        std::ostringstream message;
        message << std::setprecision(17)
                << "attention-only actual-trace inactive-slot gate failed: "
                << "query=" << query_inactive
                << " key=" << key_inactive
                << " value=" << value_inactive
                << " scores=" << score_inactive
                << " probabilities=" << probability_inactive
                << " raw_output=" << raw_output_inactive
                << " cleaned_output=" << output_inactive;
        throw std::runtime_error(message.str());
    }
    const auto& metrics = server.metrics();
    std::cout
        << "{\"test\":\"openfhe_encoder_attention_trace_diagnostic\","
        << "\"profile\":\"paper_compat\",\"security_claim\":\"none\","
        << "\"layer\":" << fixture.weights.layer_index << ','
        << "\"query_inactive_max_abs\":" << query_inactive << ','
        << "\"key_inactive_max_abs\":" << key_inactive << ','
        << "\"value_inactive_max_abs\":" << value_inactive << ','
        << "\"score_inactive_max_abs\":" << score_inactive << ','
        << "\"probability_inactive_max_abs\":" << probability_inactive << ','
        << "\"raw_output_inactive_max_abs\":" << raw_output_inactive << ','
        << "\"output_inactive_max_abs\":" << output_inactive << ','
        << "\"probability_relative_l2\":"
        << probability_quality.relative_l2 << ','
        << "\"probability_cosine\":" << probability_quality.cosine << ','
        << "\"raw_output_relative_l2\":"
        << raw_output_quality.relative_l2 << ','
        << "\"raw_output_cosine\":" << raw_output_quality.cosine << ','
        << "\"output_relative_l2\":" << output_quality.relative_l2 << ','
        << "\"output_cosine\":" << output_quality.cosine << ','
        << "\"rotations\":" << metrics.rotations << ','
        << "\"ct_pt_multiplications\":"
        << metrics.ct_pt_multiplications << ','
        << "\"ct_ct_multiplications\":"
        << metrics.ct_ct_multiplications << ','
        << "\"explicit_rescale_requests\":"
        << metrics.rescale_operations << ','
        << "\"bootstraps\":" << metrics.bootstraps << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::setprecision(17);
        const auto arguments = ParseArguments(argc, argv);
        const auto fixture_begin = Clock::now();
        auto fixture = moai::openfhe::test::LoadEncoderLayerFixture(
            arguments.data_root,
            arguments.layer);
        ValidateFixturePreflight(fixture);
        ValidateInactiveSentinelContract();
        const auto fixture_end = Clock::now();
        if (arguments.preflight_only) {
            std::cout
                << "{\"test\":\"openfhe_encoder_fixture_contract\","
                << "\"profile\":\"paper_compat\","
                << "\"security_claim\":\"none\","
                << "\"layer\":" << arguments.layer
                << ",\"tokens\":5,\"hidden_size\":768,"
                << "\"intermediate_size\":3072,"
                << "\"fixture_load_validate_ms\":"
                << ElapsedMilliseconds(fixture_begin, fixture_end) << "}\n";
            return 0;
        }

        const auto polynomial_oracle =
            moai::openfhe::test::EvaluateEncoderLayerPlaintextOracle(
                fixture.expected.input,
                fixture.weights);

        auto profile =
            moai::openfhe::MakePaperCompatFeaturePackedProfile();
        if (profile.parameter_sha256 != kExpectedProfileSha256 ||
            profile.security_claim != "none" ||
            profile.multiplicative_depth != 47) {
            throw std::runtime_error("M4 paper_compat effective profile drifted");
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
        packing.level = arguments.input_level;
        packing.noise_scale_degree = 1;
        packing.scaling_factor =
            std::ldexp(1.0, static_cast<int>(profile.scaling_modulus_bits));

        const auto rotations =
            moai::openfhe::FeaturePackedEncoderRotationIndices();
        if (rotations.size() != 62 ||
            !std::binary_search(rotations.begin(), rotations.end(), 512)) {
            throw std::runtime_error("M4 encoder rotation-key contract drifted");
        }

        const auto setup_begin = Clock::now();
        moai::openfhe::ClientRuntime client(profile);
        client.GenerateEvaluationKeys(rotations, true);
        const auto setup_end = Clock::now();
        const auto encrypt_begin = Clock::now();
        const auto encrypted_input = client.Encrypt(fixture.expected.input, packing);
        const auto encrypt_end = Clock::now();

        const auto server_bundle = client.ExportServerKeyBundle();
        moai::openfhe::ServerRuntime server(server_bundle, profile);
        moai::openfhe::FeaturePackedEncoderLayer encoder(server);

        if (arguments.attention_only) {
            RunAttentionOnlyDiagnostic(
                client,
                server,
                encrypted_input,
                fixture);
            return 0;
        }

        auto missing_rotation_bundle = server_bundle;
        moai::openfhe::ServerKeyBundleTestAccess::RemoveDeclaredRotation(
            missing_rotation_bundle,
            512);
        moai::openfhe::ServerRuntime missing_rotation_server(
            missing_rotation_bundle,
            profile);
        moai::openfhe::FeaturePackedEncoderLayer missing_rotation_encoder(
            missing_rotation_server);
        bool missing_rotation_rejected = false;
        try {
            static_cast<void>(missing_rotation_encoder.Evaluate(
                encrypted_input,
                fixture.weights));
        }
        catch (const std::logic_error&) {
            missing_rotation_rejected = true;
        }
        if (!missing_rotation_rejected) {
            throw std::runtime_error(
                "encoder accepted a missing late rotation declaration");
        }
        RequireNoHomomorphicWork(
            missing_rotation_server.metrics(),
            "missing encoder rotation preflight");

        auto missing_bootstrap_bundle = server_bundle;
        moai::openfhe::ServerKeyBundleTestAccess::RemoveBootstrapCapability(
            missing_bootstrap_bundle);
        moai::openfhe::ServerRuntime missing_bootstrap_server(
            missing_bootstrap_bundle,
            profile);
        moai::openfhe::FeaturePackedEncoderLayer missing_bootstrap_encoder(
            missing_bootstrap_server);
        bool missing_bootstrap_rejected = false;
        try {
            static_cast<void>(missing_bootstrap_encoder.Evaluate(
                encrypted_input,
                fixture.weights));
        }
        catch (const std::logic_error&) {
            missing_bootstrap_rejected = true;
        }
        if (!missing_bootstrap_rejected) {
            throw std::runtime_error(
                "encoder accepted a missing bootstrap capability");
        }
        RequireNoHomomorphicWork(
            missing_bootstrap_server.metrics(),
            "missing encoder bootstrap preflight");

        auto rejected_packing = packing;
        rejected_packing.level = 30;
        const auto rejected_input = client.Encrypt(
            fixture.expected.input,
            rejected_packing);
        bool late_input_rejected = false;
        try {
            static_cast<void>(encoder.Evaluate(
                rejected_input,
                fixture.weights));
        }
        catch (const std::runtime_error&) {
            late_input_rejected = true;
        }
        const auto& preflight_metrics = server.metrics();
        if (!late_input_rejected || preflight_metrics.rotations != 0 ||
            preflight_metrics.ct_pt_multiplications != 0 ||
            preflight_metrics.ct_ct_multiplications != 0 ||
            preflight_metrics.rescale_operations != 0 ||
            preflight_metrics.bootstraps != 0 ||
            preflight_metrics.chebyshev_evaluations != 0) {
            throw std::runtime_error(
                "level-30 encoder input did not fail before homomorphic work");
        }
        const auto server_begin = Clock::now();
        const auto encrypted_result = encoder.Evaluate(
            encrypted_input,
            fixture.weights);
        const auto server_end = Clock::now();
        const auto& checkpoints = encrypted_result.checkpoints;

        RequirePacking(checkpoints.query, 5, "query");
        RequirePacking(checkpoints.key, 5, "key");
        RequirePacking(checkpoints.value, 5, "value");
        RequirePacking(checkpoints.attention.scaled_scores, 25, "scaled scores");
        RequirePacking(checkpoints.attention.probabilities, 25, "probabilities");
        RequirePacking(
            checkpoints.attention.output_before_bootstrap_cleanup,
            5,
            "attention output before bootstrap cleanup");
        RequirePacking(checkpoints.attention.output, 5, "attention output");
        RequirePacking(checkpoints.attention_after_bootstrap, 5, "attention refresh");
        RequirePacking(
            checkpoints.attention.denominator_after_bootstrap,
            5,
            "Softmax denominator");
        RequirePacking(
            checkpoints.attention_layernorm_normalized_variance,
            5,
            "attention LayerNorm normalized variance");
        RequirePacking(checkpoints.attention_layernorm, 5, "attention LayerNorm");
        RequirePacking(
            checkpoints.output_layernorm_normalized_variance,
            5,
            "output LayerNorm normalized variance");
        RequireCheckpointMetadata(
            "Softmax denominator",
            checkpoints.attention.denominator_after_bootstrap,
            server,
            18,
            2,
            28,
            5);
        RequireCheckpointMetadata(
            "attention LayerNorm normalized variance",
            checkpoints.attention_layernorm_normalized_variance,
            server,
            19,
            2,
            27,
            5);
        RequireCheckpointMetadata(
            "output LayerNorm normalized variance",
            checkpoints.output_layernorm_normalized_variance,
            server,
            19,
            2,
            27,
            5);
        RequirePacking(checkpoints.output_layernorm, 5, "output LayerNorm");

        double inactive_maximum = 0.0;
        std::string inactive_source;
        const auto observe_inactive = [&inactive_maximum, &inactive_source](
                                          const std::string& label,
                                          const PlainMatrix& values,
                                          std::size_t active_features) {
            const double observed = MaximumInactive(values, active_features);
            std::cout
                << "{\"inactive_checkpoint\":\"" << label << "\","
                << "\"max_absolute\":" << observed << "}\n";
            if (observed > inactive_maximum) {
                inactive_maximum = observed;
                inactive_source = label;
            }
        };
        const auto decrypt_begin = Clock::now();
        const auto query = client.Decrypt(checkpoints.query);
        const auto key = client.Decrypt(checkpoints.key);
        const auto value = client.Decrypt(checkpoints.value);
        const auto scores = client.Decrypt(checkpoints.attention.scaled_scores);
        const auto probabilities = client.Decrypt(
            checkpoints.attention.probabilities);
        const auto attention = client.Decrypt(
            checkpoints.attention.output_before_bootstrap_cleanup);
        const auto attention_refreshed = client.Decrypt(
            checkpoints.attention_after_bootstrap);
        const auto self_projection = client.Decrypt(checkpoints.self_projection);
        const auto attention_residual = client.Decrypt(
            checkpoints.attention_residual);
        const auto attention_layernorm = client.Decrypt(
            checkpoints.attention_layernorm);
        const auto softmax_denominator = client.Decrypt(
            checkpoints.attention.denominator_after_bootstrap);
        const auto attention_layernorm_normalized_variance = client.Decrypt(
            checkpoints.attention_layernorm_normalized_variance);

        const auto query_quality = MeasureQuality(
            query,
            fixture.expected.query,
            moai::openfhe::kPaperCompatHiddenSize);
        const auto key_quality = MeasureQuality(
            key,
            fixture.expected.key,
            moai::openfhe::kPaperCompatHiddenSize);
        const auto value_quality = MeasureQuality(
            value,
            fixture.expected.value,
            moai::openfhe::kPaperCompatHiddenSize);
        const auto score_quality = MeasureQuality(
            scores,
            ExpandHeadKeyTrace(fixture.expected.scaled_scores),
            moai::openfhe::kPaperCompatHiddenSize);
        const auto probability_quality = MeasureQuality(
            probabilities,
            ExpandHeadKeyTrace(fixture.expected.probabilities),
            moai::openfhe::kPaperCompatHiddenSize);
        const auto attention_quality = MeasureQuality(
            attention,
            fixture.expected.attention,
            moai::openfhe::kPaperCompatHiddenSize);
        const auto attention_refresh_quality = MeasureQuality(
            attention_refreshed,
            fixture.expected.attention,
            moai::openfhe::kPaperCompatHiddenSize);
        const auto self_quality = MeasureQuality(
            self_projection,
            fixture.expected.self_projection,
            moai::openfhe::kPaperCompatHiddenSize);
        const auto attention_residual_quality = MeasureQuality(
            attention_residual,
            fixture.expected.attention_residual,
            moai::openfhe::kPaperCompatHiddenSize);
        const auto attention_layernorm_quality = MeasureQuality(
            attention_layernorm,
            fixture.expected.attention_layernorm,
            moai::openfhe::kPaperCompatHiddenSize);

        RequireQuality(query_quality, 1e-4, 0.99999, "query projection");
        RequireQuality(key_quality, 1e-4, 0.99999, "key projection");
        RequireQuality(value_quality, 1e-4, 0.99999, "value projection");
        RequireQuality(score_quality, 1e-4, 0.99999, "attention scores");
        RequireQuality(probability_quality, 1e-2, 0.999, "attention probabilities");
        RequireQuality(
            attention_quality,
            1e-2,
            0.999,
            "attention output before bootstrap cleanup");
        RequireQuality(
            attention_refresh_quality,
            1e-2,
            0.999,
            "attention output after bootstrap cleanup");
        RequireQuality(self_quality, 1e-2, 0.999, "self projection");
        RequireQuality(
            attention_residual_quality,
            1e-2,
            0.999,
            "attention residual");
        RequireQuality(
            attention_layernorm_quality,
            1e-2,
            0.999,
            "attention LayerNorm");

        observe_inactive(
            "query",
            query,
            moai::openfhe::kPaperCompatHiddenSize);
        observe_inactive(
            "key",
            key,
            moai::openfhe::kPaperCompatHiddenSize);
        observe_inactive(
            "value",
            value,
            moai::openfhe::kPaperCompatHiddenSize);
        observe_inactive(
            "scores",
            scores,
            moai::openfhe::kPaperCompatHiddenSize);
        observe_inactive(
            "probabilities",
            probabilities,
            moai::openfhe::kPaperCompatHiddenSize);
        observe_inactive(
            "attention_before_bootstrap_cleanup",
            attention,
            moai::openfhe::kPaperCompatHiddenSize);
        observe_inactive(
            "attention_after_bootstrap_cleanup",
            attention_refreshed,
            moai::openfhe::kPaperCompatHiddenSize);
        observe_inactive(
            "self_projection",
            self_projection,
            moai::openfhe::kPaperCompatHiddenSize);
        observe_inactive(
            "attention_residual",
            attention_residual,
            moai::openfhe::kPaperCompatHiddenSize);
        observe_inactive(
            "attention_layernorm",
            attention_layernorm,
            moai::openfhe::kPaperCompatHiddenSize);

        const auto nonlinear_contracts =
            moai::openfhe::MakePaperCompatNonlinearContracts();
        RequireValuesInDeclaredRange(
            softmax_denominator,
            nonlinear_contracts.softmax_reciprocal.interval,
            "Softmax denominator");
        RequireValuesInDeclaredRange(
            attention_layernorm_normalized_variance,
            nonlinear_contracts.layernorm_inverse_sqrt.interval,
            "attention LayerNorm normalized variance");
        const auto softmax_denominator_sentinel_range =
            MeasureInactiveSentinelRange(
                softmax_denominator,
                moai::openfhe::kPaperCompatHiddenSize,
                "Softmax denominator inactive sentinel");
        const auto attention_layernorm_sentinel_range =
            MeasureInactiveSentinelRange(
                attention_layernorm_normalized_variance,
                moai::openfhe::kPaperCompatHiddenSize,
                "attention LayerNorm inactive sentinel");
        const double attention_layernorm_guard_deviation =
            MeasureLayerNormInactiveGuardDeviation(
                attention_layernorm_normalized_variance,
                "attention LayerNorm normalized variance");

        QualityMetrics worst_intermediate_pre;
        QualityMetrics worst_intermediate_activation;
        QualityMetrics worst_output_contribution;
        for (std::size_t block = 0;
             block < moai::openfhe::kPaperCompatIntermediateBlocks;
             ++block) {
            const auto pre = client.Decrypt(
                checkpoints.intermediate_pre_activation[block]);
            const auto polynomial = client.Decrypt(
                checkpoints.intermediate_polynomial_output[block]);
            const auto activation = client.Decrypt(
                checkpoints.intermediate_activation[block]);
            const auto output_contribution = client.Decrypt(
                checkpoints.output_contributions[block]);
            const auto expected_pre =
                moai::openfhe::test::SliceIntermediateBlock(
                    fixture.expected.intermediate_pre_activation,
                    block);
            const auto expected_activation =
                moai::openfhe::test::SliceIntermediateBlock(
                    fixture.expected.intermediate_activation,
                    block);
            const auto pre_quality = MeasureQuality(
                pre,
                expected_pre,
                moai::openfhe::kPaperCompatFeatureBlock);
            const auto activation_quality = MeasureQuality(
                activation,
                expected_activation,
                moai::openfhe::kPaperCompatFeatureBlock);
            const auto expected_output_contribution =
                moai::openfhe::test::PadFeatureRows(
                    DenseOracle(
                        moai::openfhe::test::SliceIntermediateBlock(
                            fixture.expected.intermediate_activation,
                            block),
                        fixture.weights.output_weight_blocks[block],
                        std::vector<double>(
                            moai::openfhe::kPaperCompatHiddenSize,
                            0.0)),
                    moai::openfhe::kPaperCompatHiddenSize);
            const auto output_contribution_quality = MeasureQuality(
                output_contribution,
                expected_output_contribution,
                moai::openfhe::kPaperCompatHiddenSize);
            RequireQuality(pre_quality, 1e-2, 0.999, "FFN pre-activation");
            RequireQuality(activation_quality, 1e-2, 0.999, "GELU activation");
            RequireQuality(
                output_contribution_quality,
                1e-2,
                0.999,
                "raw W2 output contribution");
            if (pre_quality.relative_l2 > worst_intermediate_pre.relative_l2) {
                worst_intermediate_pre = pre_quality;
            }
            if (activation_quality.relative_l2 >
                worst_intermediate_activation.relative_l2) {
                worst_intermediate_activation = activation_quality;
            }
            if (output_contribution_quality.relative_l2 >
                worst_output_contribution.relative_l2) {
                worst_output_contribution = output_contribution_quality;
            }
            observe_inactive(
                "gelu_polynomial_" + std::to_string(block),
                polynomial,
                moai::openfhe::kPaperCompatFeatureBlock);
            observe_inactive(
                "output_contribution_" + std::to_string(block),
                output_contribution,
                moai::openfhe::kPaperCompatHiddenSize);
        }

        const auto output_projection = client.Decrypt(
            checkpoints.output_projection);
        const auto output_residual = client.Decrypt(
            checkpoints.output_residual_before_bootstrap);
        const auto output_residual_refreshed = client.Decrypt(
            checkpoints.output_residual_after_bootstrap);
        const auto output_layernorm_normalized_variance = client.Decrypt(
            checkpoints.output_layernorm_normalized_variance);
        const auto output = client.Decrypt(encrypted_result.output);
        const auto decrypt_end = Clock::now();

        RequireValuesInDeclaredRange(
            output_layernorm_normalized_variance,
            nonlinear_contracts.layernorm_inverse_sqrt.interval,
            "output LayerNorm normalized variance");
        const auto output_layernorm_sentinel_range =
            MeasureInactiveSentinelRange(
                output_layernorm_normalized_variance,
                moai::openfhe::kPaperCompatHiddenSize,
                "output LayerNorm inactive sentinel");
        const double output_layernorm_guard_deviation =
            MeasureLayerNormInactiveGuardDeviation(
                output_layernorm_normalized_variance,
                "output LayerNorm normalized variance");
        const double layernorm_inactive_guard_max_error =
            std::max(
                attention_layernorm_guard_deviation,
                output_layernorm_guard_deviation);

        const auto output_projection_quality = MeasureQuality(
            output_projection,
            fixture.expected.output_projection,
            moai::openfhe::kPaperCompatHiddenSize);
        const auto output_residual_quality = MeasureQuality(
            output_residual,
            fixture.expected.output_residual,
            moai::openfhe::kPaperCompatHiddenSize);
        const auto output_residual_refresh_quality = MeasureQuality(
            output_residual_refreshed,
            fixture.expected.output_residual,
            moai::openfhe::kPaperCompatHiddenSize);
        const auto output_exact_trace_quality = MeasureQuality(
            output,
            fixture.expected.output,
            moai::openfhe::kPaperCompatHiddenSize);
        const auto output_quality = MeasureQuality(
            output,
            polynomial_oracle.output,
            moai::openfhe::kPaperCompatHiddenSize);
        RequireQuality(output_projection_quality, 1e-2, 0.999, "output projection");
        RequireQuality(output_residual_quality, 1e-2, 0.999, "output residual");
        RequireQuality(
            output_residual_refresh_quality,
            1e-2,
            0.999,
            "output residual refresh");
        RequireQuality(
            output_quality,
            1e-2,
            0.999,
            "encoder layer polynomial-oracle output");
        observe_inactive(
            "output_projection",
            output_projection,
            moai::openfhe::kPaperCompatHiddenSize);
        observe_inactive(
            "output_residual",
            output_residual,
            moai::openfhe::kPaperCompatHiddenSize);
        observe_inactive(
            "output_residual_refreshed",
            output_residual_refreshed,
            moai::openfhe::kPaperCompatHiddenSize);
        observe_inactive(
            "output",
            output,
            moai::openfhe::kPaperCompatHiddenSize);
        if (inactive_maximum > 1e-6) {
            std::ostringstream message;
            message << std::setprecision(17)
                    << "M4 inactive/cross-lane max-abs exceeded 1e-6 at "
                    << inactive_source << ": " << inactive_maximum;
            throw std::runtime_error(message.str());
        }

        const auto& metrics = server.metrics();
        if (metrics.bootstraps != 25 || metrics.bootstrap_iterations != 50 ||
            metrics.rotations != 6300 ||
            metrics.ct_pt_multiplications != 51885 ||
            metrics.ct_ct_multiplications != 95 ||
            metrics.rescale_operations != 810 ||
            metrics.chebyshev_evaluations != 55 ||
            metrics.estimated_polynomial_multiplications != 1150 ||
            metrics.multiplicative_depth != 47 ||
            metrics.max_polynomial_depth != 10 ||
            encrypted_result.output.packing.level != 30 ||
            encrypted_result.output.packing.noise_scale_degree != 2 ||
            server.RemainingLevels(encrypted_result.output) != 16 ||
            metrics.max_observed_level != 45) {
            std::ostringstream message;
            message << "M4 operation/depth schedule drifted: rotations="
                    << metrics.rotations
                    << " ct_pt=" << metrics.ct_pt_multiplications
                    << " ct_ct=" << metrics.ct_ct_multiplications
                    << " explicit_rescale_requests=" << metrics.rescale_operations
                    << " chebyshev_evaluations=" << metrics.chebyshev_evaluations
                    << " estimated_polynomial_multiplications="
                    << metrics.estimated_polynomial_multiplications
                    << " bootstraps=" << metrics.bootstraps
                    << " bootstrap_iterations=" << metrics.bootstrap_iterations
                    << " multiplicative_depth=" << metrics.multiplicative_depth
                    << " max_observed_level=" << metrics.max_observed_level
                    << " max_polynomial_depth=" << metrics.max_polynomial_depth
                    << " output_level=" << encrypted_result.output.packing.level
                    << " output_remaining_levels="
                    << server.RemainingLevels(encrypted_result.output);
            throw std::runtime_error(message.str());
        }

        PrintQuality("query", query_quality);
        PrintQuality("key", key_quality);
        PrintQuality("value", value_quality);
        PrintQuality("scores", score_quality);
        PrintQuality("probabilities", probability_quality);
        PrintQuality("attention_before_bootstrap_cleanup", attention_quality);
        PrintQuality("attention_after_bootstrap_cleanup", attention_refresh_quality);
        PrintQuality("self_projection", self_quality);
        PrintQuality("attention_residual", attention_residual_quality);
        PrintQuality("attention_layernorm", attention_layernorm_quality);
        PrintQuality("intermediate_pre_worst", worst_intermediate_pre);
        PrintQuality("gelu_worst", worst_intermediate_activation);
        PrintQuality("raw_w2_contribution_worst", worst_output_contribution);
        PrintQuality("output_projection", output_projection_quality);
        PrintQuality("output_residual", output_residual_quality);
        PrintQuality("output_residual_refresh", output_residual_refresh_quality);
        PrintQuality("output_exact_trace", output_exact_trace_quality);
        PrintQuality("output_polynomial_oracle", output_quality);
        PrintPacking("scores", checkpoints.attention.scaled_scores, server);
        PrintPacking("probabilities", checkpoints.attention.probabilities, server);
        PrintPacking(
            "attention_before_bootstrap_cleanup",
            checkpoints.attention.output_before_bootstrap_cleanup,
            server);
        PrintPacking(
            "attention_after_bootstrap_cleanup",
            checkpoints.attention_after_bootstrap,
            server);
        PrintPacking("attention_layernorm", checkpoints.attention_layernorm, server);
        PrintPacking(
            "ffn_pre_activation_0",
            checkpoints.intermediate_pre_activation[0],
            server);
        PrintPacking(
            "ffn_activation_0",
            checkpoints.intermediate_activation[0],
            server);
        PrintPacking(
            "output_residual_before_bootstrap",
            checkpoints.output_residual_before_bootstrap,
            server);
        PrintPacking(
            "output_residual_after_bootstrap",
            checkpoints.output_residual_after_bootstrap,
            server);
        PrintPacking("output", encrypted_result.output, server);

        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) != 0) {
            throw std::runtime_error("getrusage failed");
        }
        std::cout
            << "{\"test\":\"openfhe_encoder_layer_smoke\","
            << "\"profile\":\"paper_compat\","
            << "\"security_claim\":\"none\","
            << "\"parameter_sha256\":\"" << profile.parameter_sha256 << "\","
            << "\"layer\":" << arguments.layer << ','
            << "\"input_level\":" << arguments.input_level << ','
            << "\"tokens\":5,\"hidden_size\":768,"
            << "\"intermediate_size\":3072,\"feature_block\":1024,"
            << "\"fixture_load_ms\":"
            << ElapsedMilliseconds(fixture_begin, fixture_end) << ','
            << "\"setup_keygen_ms\":"
            << ElapsedMilliseconds(setup_begin, setup_end) << ','
            << "\"client_encrypt_ms\":"
            << ElapsedMilliseconds(encrypt_begin, encrypt_end) << ','
            << "\"server_online_ms\":"
            << ElapsedMilliseconds(server_begin, server_end) << ','
            << "\"client_decrypt_validate_ms\":"
            << ElapsedMilliseconds(decrypt_begin, decrypt_end) << ','
            << "\"relative_l2\":" << output_quality.relative_l2 << ','
            << "\"cosine\":" << output_quality.cosine << ','
            << "\"max_absolute\":" << output_quality.max_absolute << ','
            << "\"exact_trace_relative_l2\":"
            << output_exact_trace_quality.relative_l2 << ','
            << "\"exact_trace_cosine\":"
            << output_exact_trace_quality.cosine << ','
            << "\"exact_trace_max_absolute\":"
            << output_exact_trace_quality.max_absolute << ','
            << "\"inactive_max_absolute\":" << inactive_maximum << ','
            << "\"peak_rss_bytes\":"
            << static_cast<unsigned long long>(usage.ru_maxrss) * 1024ULL << ','
            << "\"rotations\":" << metrics.rotations << ','
            << "\"ct_pt_multiplications\":"
            << metrics.ct_pt_multiplications << ','
            << "\"ct_ct_multiplications\":"
            << metrics.ct_ct_multiplications << ','
            << "\"explicit_rescale_requests\":"
            << metrics.rescale_operations << ','
            << "\"chebyshev_evaluations\":"
            << metrics.chebyshev_evaluations << ','
            << "\"estimated_polynomial_multiplications\":"
            << metrics.estimated_polynomial_multiplications << ','
            << "\"bootstraps\":" << metrics.bootstraps << ','
            << "\"bootstrap_iterations\":"
            << metrics.bootstrap_iterations << ','
            << "\"multiplicative_depth\":"
            << metrics.multiplicative_depth << ','
            << "\"max_observed_level\":"
            << metrics.max_observed_level << ','
            << "\"max_polynomial_depth\":"
            << metrics.max_polynomial_depth << ','
            << "\"final_level\":" << encrypted_result.output.packing.level << ','
            << "\"final_remaining_levels\":"
            << server.RemainingLevels(encrypted_result.output) << "}\n";
        std::cout
            << "{\"test\":\"openfhe_encoder_layer\","
            << "\"profile\":\"paper_compat\","
            << "\"security_claim\":\"none\","
            << "\"parameter_sha256\":\"" << profile.parameter_sha256 << "\","
            << "\"execution_mode\":\"server-only\","
            << "\"encoder_layers\":1,"
            << "\"layer_id\":" << arguments.layer << ','
            << "\"input_level\":" << arguments.input_level << ','
            << "\"output_level\":"
            << encrypted_result.output.packing.level << ','
            << "\"remaining_levels\":"
            << server.RemainingLevels(encrypted_result.output) << ','
            << "\"actual_trace_shape\":[5,768],"
            << "\"actual_trace_value_count\":3840,"
            << "\"feature_block_size\":1024,"
            << "\"checkpoint_decryption_owner\":\"client\","
            << "\"server_private_key_present\":false,"
            << "\"server_decryptions\":0,"
            << "\"server_plaintext_activations\":false,"
            << "\"relative_l2\":" << output_quality.relative_l2 << ','
            << "\"cosine\":" << output_quality.cosine << ','
            << "\"inactive_max_abs\":" << inactive_maximum << ','
            << "\"inactive_polynomial_sentinel_ranges\":{"
            << "\"softmax_denominator\":{\"minimum\":"
            << softmax_denominator_sentinel_range.minimum
            << ",\"maximum\":"
            << softmax_denominator_sentinel_range.maximum << "},"
            << "\"attention_layernorm_normalized_variance\":{"
            << "\"minimum\":"
            << attention_layernorm_sentinel_range.minimum
            << ",\"maximum\":"
            << attention_layernorm_sentinel_range.maximum << "},"
            << "\"output_layernorm_normalized_variance\":{"
            << "\"minimum\":"
            << output_layernorm_sentinel_range.minimum
            << ",\"maximum\":"
            << output_layernorm_sentinel_range.maximum << "}},"
            << "\"layernorm_inactive_guard_max_error\":"
            << layernorm_inactive_guard_max_error << ','
            << "\"multiplicative_depth\":"
            << metrics.multiplicative_depth << ','
            << "\"max_observed_level\":"
            << metrics.max_observed_level << ','
            << "\"max_polynomial_depth\":"
            << metrics.max_polynomial_depth << ','
            << "\"checkpoints\":[";
        PrintArtifactCheckpoint(
            "attention_output",
            checkpoints.attention.output_before_bootstrap_cleanup,
            server,
            40,
            2,
            6,
            5);
        std::cout << ',';
        PrintArtifactCheckpoint(
            "self_layernorm_output",
            checkpoints.attention_layernorm,
            server,
            32,
            2,
            14,
            5);
        std::cout << ',';
        PrintArtifactCheckpoint(
            "ffn_output",
            checkpoints.output_projection,
            server,
            45,
            2,
            1,
            5);
        std::cout << ',';
        PrintArtifactCheckpoint(
            "encoder_output",
            encrypted_result.output,
            server,
            30,
            2,
            16,
            5);
        std::cout
            << "],\"operation_counts\":{"
            << "\"rotations\":" << metrics.rotations << ','
            << "\"ct_pt_multiplications\":"
            << metrics.ct_pt_multiplications << ','
            << "\"ct_ct_multiplications\":"
            << metrics.ct_ct_multiplications << ','
            << "\"explicit_rescale_requests\":"
            << metrics.rescale_operations << ','
            << "\"chebyshev_evaluations\":"
            << metrics.chebyshev_evaluations << ','
            << "\"estimated_polynomial_multiplications\":"
            << metrics.estimated_polynomial_multiplications << ','
            << "\"bootstraps\":" << metrics.bootstraps << ','
            << "\"bootstrap_iterations\":"
            << metrics.bootstrap_iterations << "}}\n";
        return 0;
    }
    catch (const std::exception& exception) {
        std::cerr << "openfhe_encoder_layer_smoke failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
