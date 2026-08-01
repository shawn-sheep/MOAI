#include "moai/openfhe/approximation_registry.hpp"
#include "moai/openfhe/client_runtime.hpp"
#include "moai/openfhe/context_factory.hpp"
#include "moai/openfhe/nonlinear_ops.hpp"
#include "moai/openfhe/server_runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kFeatureSlots = 1024;
constexpr std::size_t kHiddenSize = 768;
constexpr double kInactiveMaximum = 1e-6;
constexpr double kActiveVarianceMaximumError = 1.0;
constexpr double kScaleBitsTolerance = 1e-3;
constexpr const char* kExpectedProfileSha256 =
    "94f30e628e21f02146ce7ed9820194eabba3820f6e1e17176a31f8c5acf8b0be";

using PlainMatrix = std::vector<std::vector<double>>;

struct QualityMetrics {
    double relative_l2{0.0};
    double cosine{0.0};
    double maximum_absolute{0.0};
};

struct SiteFixture {
    const char* label;
    moai::openfhe::PaperCompatLayerNormSite site;
    std::size_t layer;
    std::array<double, moai::openfhe::kPaperCompatLayerNormTraceTokens>
        input_variances;
    std::array<double, moai::openfhe::kPaperCompatLayerNormTraceTokens>
        expected_variance_scales;
    uint32_t input_level;
    uint32_t expected_output_level;
    uint32_t expected_output_remaining_levels;
};

struct SiteObservation {
    double normalized_variance_expected_minimum{0.0};
    double normalized_variance_expected_maximum{0.0};
    double normalized_variance_active_minimum{0.0};
    double normalized_variance_active_maximum{0.0};
    double normalized_variance_active_maximum_error{0.0};
    double normalized_variance_inactive_maximum_error{0.0};
    double output_inactive_maximum{0.0};
    QualityMetrics output_quality;
    uint32_t normalized_variance_level{0};
    uint32_t normalized_variance_remaining_levels{0};
    uint32_t output_level{0};
    uint32_t output_remaining_levels{0};
};

std::vector<int32_t> LayerNormRotations() {
    std::vector<int32_t> rotations;
    for (uint32_t step = 1; step < kFeatureSlots; step <<= 1) {
        rotations.push_back(static_cast<int32_t>(step));
    }
    return rotations;
}

std::vector<double> AlternatingInput(double variance) {
    if (!std::isfinite(variance) || variance <= 0.0) {
        throw std::invalid_argument("LayerNorm fixture variance must be positive");
    }
    std::vector<double> result(kFeatureSlots, 0.0);
    const double amplitude = std::sqrt(variance);
    for (std::size_t slot = 0; slot < kHiddenSize; ++slot) {
        result[slot] = slot % 2 == 0 ? amplitude : -amplitude;
    }
    return result;
}

PlainMatrix AlternatingInputs(
    const std::array<
        double,
        moai::openfhe::kPaperCompatLayerNormTraceTokens>& variances) {
    PlainMatrix result;
    result.reserve(variances.size());
    for (const double variance : variances) {
        result.push_back(AlternatingInput(variance));
    }
    return result;
}

double ActiveMean(const std::vector<double>& values) {
    if (values.size() != kFeatureSlots) {
        throw std::runtime_error("LayerNorm fixture width changed");
    }
    long double sum = 0.0;
    for (std::size_t slot = 0; slot < kHiddenSize; ++slot) {
        sum += values[slot];
    }
    return static_cast<double>(sum / static_cast<long double>(kHiddenSize));
}

double ActiveVariance(const std::vector<double>& values, double mean) {
    long double sum = 0.0;
    for (std::size_t slot = 0; slot < kHiddenSize; ++slot) {
        const long double centered = values[slot] - mean;
        sum += centered * centered;
    }
    return static_cast<double>(sum / static_cast<long double>(kHiddenSize));
}

QualityMetrics MeasureActiveQuality(
    const PlainMatrix& actual,
    const PlainMatrix& expected) {
    if (actual.size() != moai::openfhe::kPaperCompatLayerNormTraceTokens ||
        expected.size() != actual.size()) {
        throw std::runtime_error("LayerNorm output shape changed");
    }
    long double squared_error = 0.0;
    long double squared_actual = 0.0;
    long double squared_expected = 0.0;
    long double dot = 0.0;
    double maximum_absolute = 0.0;
    for (std::size_t row = 0; row < actual.size(); ++row) {
        if (actual[row].size() != kFeatureSlots ||
            expected[row].size() != kFeatureSlots) {
            throw std::runtime_error("LayerNorm output width changed");
        }
        for (std::size_t slot = 0; slot < kHiddenSize; ++slot) {
            const double observed = actual[row][slot];
            const double reference = expected[row][slot];
            if (!std::isfinite(observed) || !std::isfinite(reference)) {
                throw std::runtime_error(
                    "LayerNorm active output contains NaN or Inf");
            }
            const long double error = observed - reference;
            squared_error += error * error;
            squared_actual += static_cast<long double>(observed) * observed;
            squared_expected += static_cast<long double>(reference) * reference;
            dot += static_cast<long double>(observed) * reference;
            maximum_absolute = std::max(
                maximum_absolute,
                std::abs(observed - reference));
        }
    }
    if (squared_actual == 0.0 || squared_expected == 0.0) {
        throw std::runtime_error("LayerNorm active output has a zero norm");
    }
    return {
        std::sqrt(static_cast<double>(squared_error / squared_expected)),
        static_cast<double>(dot / std::sqrt(squared_actual * squared_expected)),
        maximum_absolute};
}

double RequireInactiveGuard(
    const PlainMatrix& values,
    const moai::openfhe::DeclaredRange& interval,
    const std::string& label) {
    if (values.empty() || !std::isfinite(interval.minimum) ||
        !std::isfinite(interval.maximum) ||
        interval.minimum >= interval.maximum) {
        throw std::runtime_error(label + " contract is invalid");
    }
    double maximum = 0.0;
    for (const auto& row : values) {
        if (row.size() != kFeatureSlots) {
            throw std::runtime_error(label + " width changed");
        }
        for (std::size_t slot = kHiddenSize; slot < row.size(); ++slot) {
            if (!std::isfinite(row[slot]) || row[slot] < interval.minimum ||
                row[slot] > interval.maximum) {
                throw std::runtime_error(label + " escaped the polynomial interval");
            }
            maximum = std::max(maximum, std::abs(row[slot] - 1.0));
        }
    }
    return maximum;
}

double MeasureInactiveZero(
    const PlainMatrix& values,
    const std::string& label) {
    if (values.empty()) {
        throw std::runtime_error(label + " is empty");
    }
    double maximum = 0.0;
    for (const auto& row : values) {
        if (row.size() != kFeatureSlots) {
            throw std::runtime_error(label + " width changed");
        }
        for (std::size_t slot = kHiddenSize; slot < row.size(); ++slot) {
            if (!std::isfinite(row[slot])) {
                throw std::runtime_error(label + " contains NaN or Inf");
            }
            maximum = std::max(maximum, std::abs(row[slot]));
        }
    }
    return maximum;
}

template <typename Operation>
void RequireRejection(Operation&& operation, const std::string& label) {
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
void RequireInvalidArgument(Operation&& operation, const std::string& label) {
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

void ValidateInactiveGuardContract() {
    PlainMatrix valid(1, std::vector<double>(kFeatureSlots, 1.0));
    const auto interval =
        moai::openfhe::MakePaperCompatNonlinearContracts()
            .layernorm_inverse_sqrt.interval;
    static_cast<void>(RequireInactiveGuard(
        valid,
        interval,
        "synthetic inverse-square-root guard"));

    auto escaped = valid;
    escaped.front()[kHiddenSize] = std::nextafter(
        interval.minimum,
        -std::numeric_limits<double>::infinity());
    RequireRejection(
        [&escaped, &interval]() {
            static_cast<void>(RequireInactiveGuard(
                escaped,
                interval,
                "escaped inverse-square-root guard"));
        },
        "escaped LayerNorm inverse-square-root guard");

    auto non_finite = valid;
    non_finite.front()[kHiddenSize] = std::numeric_limits<double>::quiet_NaN();
    RequireRejection(
        [&non_finite, &interval]() {
            static_cast<void>(RequireInactiveGuard(
                non_finite,
                interval,
                "non-finite inverse-square-root guard"));
        },
        "non-finite LayerNorm inverse-square-root guard");

    PlainMatrix wrong_width(1, std::vector<double>(kFeatureSlots - 1, 1.0));
    RequireRejection(
        [&wrong_width, &interval]() {
            static_cast<void>(RequireInactiveGuard(
                wrong_width,
                interval,
                "wrong-width inverse-square-root guard"));
        },
        "wrong-width LayerNorm inverse-square-root guard");

    RequireRejection(
        [&interval]() {
            static_cast<void>(RequireInactiveGuard(
                {},
                interval,
                "empty inverse-square-root guard"));
        },
        "empty LayerNorm inverse-square-root guard");
}

void ValidateFixedTraceShapeContract(
    const moai::openfhe::PackingSpec& packing,
    moai::openfhe::ClientRuntime& client,
    moai::openfhe::NonlinearOps& nonlinear) {
    auto shape_packing = packing;
    shape_packing.level = 15;
    shape_packing.noise_scale_degree = 2;
    shape_packing.scaling_factor = std::ldexp(1.0, 100);
    const PlainMatrix zeros(
        moai::openfhe::kPaperCompatLayerNormTraceTokens,
        std::vector<double>(kFeatureSlots, 0.0));
    const auto encrypted = client.Encrypt(zeros, shape_packing);
    const std::vector<double> gamma(kHiddenSize, 1.0);
    const std::vector<double> beta(kHiddenSize, 0.0);

    RequireInvalidArgument(
        [&]() {
            static_cast<void>(nonlinear.FeaturePackedLayerNormWithCheckpoints(
                encrypted,
                std::vector<double>(kHiddenSize - 1, 1.0),
                std::vector<double>(kHiddenSize - 1, 0.0),
                moai::openfhe::PaperCompatLayerNormSite::kAttentionResidual,
                0));
        },
        "non-768 feature-packed LayerNorm shape");

    auto wrong_token_count = encrypted;
    wrong_token_count.ciphertexts.pop_back();
    wrong_token_count.packing.logical_shape[1] =
        wrong_token_count.ciphertexts.size();
    RequireInvalidArgument(
        [&]() {
            static_cast<void>(nonlinear.FeaturePackedLayerNormWithCheckpoints(
                wrong_token_count,
                gamma,
                beta,
                moai::openfhe::PaperCompatLayerNormSite::kAttentionResidual,
                0));
        },
        "non-five-token feature-packed LayerNorm shape");

    auto wrong_slot_count = encrypted;
    wrong_slot_count.packing.active_slots = kFeatureSlots / 2;
    wrong_slot_count.packing.encoded_slots = kFeatureSlots / 2;
    wrong_slot_count.packing.logical_shape[0] = kFeatureSlots / 2;
    RequireInvalidArgument(
        [&]() {
            static_cast<void>(nonlinear.FeaturePackedLayerNormWithCheckpoints(
                wrong_slot_count,
                gamma,
                beta,
                moai::openfhe::PaperCompatLayerNormSite::kAttentionResidual,
                0));
        },
        "non-1024-slot feature-packed LayerNorm shape");

    auto wrong_batch_lanes = encrypted;
    wrong_batch_lanes.packing.batch_lanes = 2;
    RequireInvalidArgument(
        [&]() {
            static_cast<void>(nonlinear.FeaturePackedLayerNormWithCheckpoints(
                wrong_batch_lanes,
                gamma,
                beta,
                moai::openfhe::PaperCompatLayerNormSite::kAttentionResidual,
                0));
        },
        "multi-lane feature-packed LayerNorm shape");
}

void RequireMetadata(
    const moai::openfhe::CipherTensor& tensor,
    const moai::openfhe::ServerRuntime& server,
    uint32_t expected_level,
    uint32_t expected_noise_scale_degree,
    uint32_t expected_remaining_levels,
    const std::string& label) {
    if (tensor.size() != moai::openfhe::kPaperCompatLayerNormTraceTokens ||
        tensor.packing.layout != moai::openfhe::PackingLayout::kContiguous ||
        tensor.packing.logical_shape !=
            std::vector<std::size_t>{
                kFeatureSlots,
                moai::openfhe::kPaperCompatLayerNormTraceTokens} ||
        tensor.packing.batch_lanes != 1 ||
        tensor.packing.active_slots != kFeatureSlots ||
        tensor.packing.encoded_slots != kFeatureSlots ||
        tensor.packing.level != expected_level ||
        tensor.packing.noise_scale_degree != expected_noise_scale_degree ||
        server.RemainingLevels(tensor) != expected_remaining_levels) {
        std::ostringstream message;
        message << label << " metadata changed: size=" << tensor.size()
                << " level=" << tensor.packing.level
                << " noise_scale_degree=" << tensor.packing.noise_scale_degree
                << " remaining_levels=" << server.RemainingLevels(tensor)
                << " expected=("
                << moai::openfhe::kPaperCompatLayerNormTraceTokens << ','
                << expected_level << ','
                << expected_noise_scale_degree << ',' << expected_remaining_levels
                << ')';
        throw std::runtime_error(message.str());
    }
    const double expected_scale_bits = 100.0;
    const double packing_scale_bits = std::log2(tensor.packing.scaling_factor);
    if (!std::isfinite(packing_scale_bits) ||
        std::abs(packing_scale_bits - expected_scale_bits) > kScaleBitsTolerance) {
        throw std::runtime_error(label + " scale changed");
    }
    for (const auto& ciphertext : tensor.ciphertexts) {
        if (!ciphertext || ciphertext->GetLevel() != expected_level ||
            ciphertext->GetNoiseScaleDeg() != expected_noise_scale_degree ||
            std::abs(
                std::log2(ciphertext->GetScalingFactor()) - expected_scale_bits) >
                kScaleBitsTolerance) {
            throw std::runtime_error(label + " ciphertext metadata is stale");
        }
    }
}

SiteObservation RunSite(
    const SiteFixture& fixture,
    const moai::openfhe::PackingSpec& packing,
    const moai::openfhe::ApproximationContract& inverse_sqrt_contract,
    moai::openfhe::ClientRuntime& client,
    moai::openfhe::ServerRuntime& server,
    moai::openfhe::NonlinearOps& nonlinear) {
    const auto input = AlternatingInputs(fixture.input_variances);
    const auto variance_scales =
        moai::openfhe::PaperCompatLayerNormVarianceScales(
            fixture.site,
            fixture.layer);
    if (variance_scales != fixture.expected_variance_scales) {
        throw std::runtime_error(
            std::string(fixture.label) + " variance-scale registry changed");
    }
    std::array<
        double,
        moai::openfhe::kPaperCompatLayerNormTraceTokens>
        expected_normalized_variances{};
    std::array<
        double,
        moai::openfhe::kPaperCompatLayerNormTraceTokens>
        means{};
    double expected_minimum = std::numeric_limits<double>::infinity();
    double expected_maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t token = 0; token < input.size(); ++token) {
        means[token] = ActiveMean(input[token]);
        const double variance = ActiveVariance(input[token], means[token]);
        expected_normalized_variances[token] = variance_scales[token] *
            (variance + moai::openfhe::kPaperCompatLayerNormEpsilon);
        expected_minimum = std::min(
            expected_minimum,
            expected_normalized_variances[token]);
        expected_maximum = std::max(
            expected_maximum,
            expected_normalized_variances[token]);
        if (expected_normalized_variances[token] <
                inverse_sqrt_contract.interval.minimum ||
            expected_normalized_variances[token] >
                inverse_sqrt_contract.interval.maximum) {
            throw std::runtime_error(
                std::string(fixture.label) +
                " plaintext variance leaves the interval");
        }
    }

    const std::vector<double> gamma(kHiddenSize, 1.0);
    const std::vector<double> beta(kHiddenSize, 0.0);
    auto site_packing = packing;
    site_packing.level = fixture.input_level;
    site_packing.noise_scale_degree = 2;
    site_packing.scaling_factor = std::ldexp(1.0, 100);
    const auto encrypted = client.Encrypt({input}, site_packing);
    const auto result = nonlinear.FeaturePackedLayerNormWithCheckpoints(
        encrypted,
        gamma,
        beta,
        fixture.site,
        fixture.layer);

    RequireMetadata(
        result.normalized_variance,
        server,
        19,
        2,
        27,
        std::string(fixture.label) + " normalized variance");
    RequireMetadata(
        result.output,
        server,
        fixture.expected_output_level,
        2,
        fixture.expected_output_remaining_levels,
        std::string(fixture.label) + " output");

    const auto normalized_variance = client.Decrypt(result.normalized_variance);
    if (normalized_variance.size() !=
        moai::openfhe::kPaperCompatLayerNormTraceTokens) {
        throw std::runtime_error(
            std::string(fixture.label) + " normalized variance shape changed");
    }
    double active_minimum = std::numeric_limits<double>::infinity();
    double active_maximum = -std::numeric_limits<double>::infinity();
    double active_maximum_error = 0.0;
    for (std::size_t token = 0; token < normalized_variance.size(); ++token) {
        if (normalized_variance[token].size() != kFeatureSlots) {
            throw std::runtime_error(
                std::string(fixture.label) +
                " normalized variance width changed");
        }
        for (std::size_t slot = 0; slot < kHiddenSize; ++slot) {
            const double value = normalized_variance[token][slot];
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    std::string(fixture.label) +
                    " active variance is not finite");
            }
            active_minimum = std::min(active_minimum, value);
            active_maximum = std::max(active_maximum, value);
            active_maximum_error = std::max(
                active_maximum_error,
                std::abs(
                    value - expected_normalized_variances[token]));
        }
    }
    if (active_minimum < inverse_sqrt_contract.interval.minimum ||
        active_maximum > inverse_sqrt_contract.interval.maximum ||
        active_maximum_error > kActiveVarianceMaximumError) {
        std::ostringstream message;
        message << std::setprecision(17) << fixture.label
                << " active normalized variance gate failed: expected=["
                << expected_minimum << ',' << expected_maximum << "] observed=["
                << active_minimum << ',' << active_maximum << "] max_error="
                << active_maximum_error;
        throw std::runtime_error(message.str());
    }
    const double inactive_guard_maximum_error =
        RequireInactiveGuard(
            normalized_variance,
            inverse_sqrt_contract.interval,
            std::string(fixture.label) + " normalized variance");

    PlainMatrix expected_output(
        moai::openfhe::kPaperCompatLayerNormTraceTokens,
        std::vector<double>(kFeatureSlots, 0.0));
    for (std::size_t token = 0; token < input.size(); ++token) {
        const double inverse = moai::openfhe::EvaluateChebyshevContractAt(
            inverse_sqrt_contract,
            expected_normalized_variances[token]);
        const double scale_compensation = std::sqrt(variance_scales[token]);
        for (std::size_t slot = 0; slot < kHiddenSize; ++slot) {
            expected_output[token][slot] =
                (input[token][slot] - means[token]) *
                inverse *
                scale_compensation;
        }
    }
    const auto output = client.Decrypt(result.output);
    const auto output_quality = MeasureActiveQuality(output, expected_output);
    const double output_inactive_maximum = MeasureInactiveZero(
        output,
        std::string(fixture.label) + " output");

    return {
        expected_minimum,
        expected_maximum,
        active_minimum,
        active_maximum,
        active_maximum_error,
        inactive_guard_maximum_error,
        output_inactive_maximum,
        output_quality,
        result.normalized_variance.packing.level,
        server.RemainingLevels(result.normalized_variance),
        result.output.packing.level,
        server.RemainingLevels(result.output)};
}

void RequireExactCounts(const moai::openfhe::RunMetrics& metrics) {
    if (metrics.rotations != 300 || metrics.ct_pt_multiplications != 90 ||
        metrics.ct_ct_multiplications != 30 ||
        metrics.rescale_operations != 105 ||
        metrics.chebyshev_evaluations != 15 ||
        metrics.estimated_polynomial_multiplications != 345 ||
        metrics.bootstraps != 15 || metrics.bootstrap_iterations != 30 ||
        metrics.multiplicative_depth != 47 || metrics.max_polynomial_depth != 9 ||
        metrics.max_observed_level != 30) {
        throw std::runtime_error(
            "feature-packed LayerNorm operation-count or depth contract changed");
    }
}

void PrintObservation(const SiteFixture& fixture, const SiteObservation& observation) {
    std::cout
        << "{\"diagnostic\":\"openfhe_feature_layernorm_site\","
        << "\"site\":\"" << fixture.label << "\","
        << "\"layer\":" << fixture.layer << ','
        << "\"normalized_variance_expected_range\":["
        << observation.normalized_variance_expected_minimum << ','
        << observation.normalized_variance_expected_maximum << "],"
        << "\"normalized_variance_active_range\":["
        << observation.normalized_variance_active_minimum << ','
        << observation.normalized_variance_active_maximum << "],"
        << "\"normalized_variance_active_maximum_error\":"
        << observation.normalized_variance_active_maximum_error << ','
        << "\"normalized_variance_inactive_maximum_error\":"
        << observation.normalized_variance_inactive_maximum_error << ','
        << "\"output_inactive_maximum\":"
        << observation.output_inactive_maximum << ','
        << "\"output_quality_role\":\"diagnostic_only\","
        << "\"output_relative_l2\":" << observation.output_quality.relative_l2
        << ",\"output_cosine\":" << observation.output_quality.cosine
        << ",\"output_maximum_absolute\":"
        << observation.output_quality.maximum_absolute << ','
        << "\"normalized_variance_level\":"
        << observation.normalized_variance_level << ','
        << "\"normalized_variance_remaining_levels\":"
        << observation.normalized_variance_remaining_levels << ','
        << "\"output_level\":" << observation.output_level << ','
        << "\"output_remaining_levels\":"
        << observation.output_remaining_levels << "}\n";
}

}  // namespace

int main() {
    try {
        ValidateInactiveGuardContract();

        const auto profile = moai::openfhe::MakePaperCompatFeaturePackedProfile();
        if (profile.parameter_sha256 != kExpectedProfileSha256 ||
            profile.multiplicative_depth != 47 ||
            profile.bootstrap_slots != kFeatureSlots) {
            throw std::runtime_error(
                "feature-packed effective profile hash/depth drifted");
        }
        moai::openfhe::PrintSecurityDisclosure(profile, std::cout);

        moai::openfhe::PackingSpec packing;
        packing.layout = moai::openfhe::PackingLayout::kContiguous;
        packing.logical_shape = {
            kFeatureSlots,
            moai::openfhe::kPaperCompatLayerNormTraceTokens};
        packing.batch_lanes = 1;
        packing.slot_count = profile.slot_count;
        packing.active_slots = kFeatureSlots;
        packing.encoded_slots = kFeatureSlots;
        packing.level = 0;
        packing.noise_scale_degree = 1;
        packing.scaling_factor =
            std::ldexp(1.0, static_cast<int>(profile.scaling_modulus_bits));

        // These are focused synthetic reproductions of the frozen trace
        // variances.  They cover the smallest attention-residual variance and
        // the layer whose feed-forward residual contains both the minimum and
        // maximum token variances; full-graph absolute metadata remains owned
        // by the encoder prefix calibration.
        const std::vector<SiteFixture> fixtures{
            {"attention_residual_layer_11",
             moai::openfhe::PaperCompatLayerNormSite::kAttentionResidual,
             11,
             {1.0193692320351013,
              0.9567431685552465,
              0.9071676363888731,
              0.692131851520362,
              0.024353731416330723},
             {64.0, 64.0, 64.0, 128.0, 2048.0},
             15,
             30,
             16},
            {"feed_forward_residual_layer_10",
             moai::openfhe::PaperCompatLayerNormSite::kFeedForwardResidual,
             10,
             {1.5625182490122846,
              1.3017295668242683,
              1.3972406224727247,
              1.3893696129617965,
              1276.3021834465007},
             {32.0, 64.0, 64.0, 64.0, 0.0625},
             15,
             30,
             16},
            {"feed_forward_residual_layer_11",
             moai::openfhe::PaperCompatLayerNormSite::kFeedForwardResidual,
             11,
             {1.2098167566385978,
              0.9543693587449972,
              1.0961904804100302,
              1.0243742914278844,
              0.768945298026667},
             {64.0, 64.0, 64.0, 64.0, 64.0},
             15,
             30,
             16}};

        std::vector<SiteObservation> observations;
        moai::openfhe::RunMetrics metrics;
        {
            moai::openfhe::ClientRuntime client(profile);
            client.GenerateEvaluationKeys(LayerNormRotations(), true);
            moai::openfhe::ServerRuntime server(
                client.ExportServerKeyBundle(),
                profile);
            moai::openfhe::NonlinearOps nonlinear(server);
            const auto contracts =
                moai::openfhe::MakePaperCompatNonlinearContracts();
            ValidateFixedTraceShapeContract(packing, client, nonlinear);
            for (const auto& fixture : fixtures) {
                observations.push_back(RunSite(
                    fixture,
                    packing,
                    contracts.layernorm_inverse_sqrt,
                    client,
                    server,
                    nonlinear));
            }
            metrics = server.metrics();
        }
        RequireExactCounts(metrics);

        std::cout << std::setprecision(17);
        for (std::size_t index = 0; index < fixtures.size(); ++index) {
            PrintObservation(fixtures[index], observations[index]);
        }
        double worst_output_inactive = 0.0;
        for (const auto& observation : observations) {
            worst_output_inactive = std::max(
                worst_output_inactive,
                observation.output_inactive_maximum);
        }
        if (worst_output_inactive > kInactiveMaximum) {
            std::ostringstream message;
            message << std::setprecision(17)
                    << "feature-packed LayerNorm output inactive zero exceeds 1e-6: "
                    << worst_output_inactive;
            throw std::runtime_error(message.str());
        }
        std::cout
            << "{\"test\":\"openfhe_feature_layernorm_smoke\","
            << "\"profile\":\"paper_compat\","
            << "\"security_claim\":\"none\","
            << "\"claim_scope\":"
            << "\"preconditioned_inverse_sqrt_guard_diagnostic_only\","
            << "\"artifact_eligible\":false,"
            << "\"active_output_quality_gate\":false,"
            << "\"variance_scale_contract_id\":\""
            << moai::openfhe::kPaperCompatLayerNormScaleContractId << "\","
            << "\"variance_scale_sha256\":\""
            << moai::openfhe::kPaperCompatLayerNormScaleSha256 << "\","
            << "\"parameter_sha256\":\"" << profile.parameter_sha256 << "\","
            << "\"sites\":" << fixtures.size()
            << ",\"encoded_slots\":" << kFeatureSlots << ','
            << "\"hidden_size\":" << kHiddenSize << ','
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
            << metrics.bootstrap_iterations << ",\"passed\":true}\n";
        return 0;
    }
    catch (const std::exception& exception) {
        std::cerr << "openfhe_feature_layernorm_smoke failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
