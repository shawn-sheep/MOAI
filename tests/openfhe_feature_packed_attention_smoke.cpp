#include "moai/openfhe/approximation_registry.hpp"
#include "moai/openfhe/client_runtime.hpp"
#include "moai/openfhe/context_factory.hpp"
#include "moai/openfhe/feature_packed_attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
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
        if (iterator == bundle.rotation_indices.end() || *iterator != rotation) {
            throw std::runtime_error(
                "attention preflight test rotation is not declared");
        }
        bundle.rotation_indices.erase(iterator);
    }

    static void RemoveBootstrapCapability(ServerKeyBundle& bundle) {
        if (bundle.bootstrap_required_indices.empty()) {
            throw std::runtime_error(
                "attention preflight test bootstrap capability is absent");
        }
        bundle.bootstrap_required_indices.clear();
    }
};

}  // namespace moai::openfhe

namespace {

struct QualityMetrics {
    double relative_l2{0.0};
    double cosine{0.0};
    double max_absolute{0.0};
};

double EvaluateChebyshev(
    const moai::openfhe::ApproximationContract& contract,
    double input) {
    const double normalized =
        (2.0 * input - contract.interval.minimum - contract.interval.maximum) /
        (contract.interval.maximum - contract.interval.minimum);
    double previous = 1.0;
    double current = normalized;
    double result = contract.coefficients[0] / 2.0;
    if (contract.degree >= 1) {
        result += contract.coefficients[1] * current;
    }
    for (uint32_t degree = 2; degree <= contract.degree; ++degree) {
        const double next = 2.0 * normalized * current - previous;
        result += contract.coefficients[degree] * next;
        previous = current;
        current = next;
    }
    return result;
}

QualityMetrics MeasureQuality(
    const std::vector<std::vector<double>>& actual,
    const std::vector<std::vector<double>>& expected,
    std::size_t active_features) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("attention checkpoint tensor count mismatch");
    }
    long double squared_error = 0.0;
    long double squared_actual = 0.0;
    long double squared_expected = 0.0;
    long double dot_product = 0.0;
    double maximum = 0.0;
    for (std::size_t tensor = 0; tensor < actual.size(); ++tensor) {
        if (actual[tensor].size() != expected[tensor].size() ||
            active_features > actual[tensor].size()) {
            throw std::runtime_error("attention checkpoint shape mismatch");
        }
        for (std::size_t slot = 0; slot < active_features; ++slot) {
            if (!std::isfinite(actual[tensor][slot])) {
                throw std::runtime_error("attention checkpoint contains NaN or Inf");
            }
            const long double observed = actual[tensor][slot];
            const long double reference = expected[tensor][slot];
            const long double error = observed - reference;
            squared_error += error * error;
            squared_actual += observed * observed;
            squared_expected += reference * reference;
            dot_product += observed * reference;
            maximum = std::max(
                maximum,
                std::abs(static_cast<double>(error)));
        }
    }
    if (squared_actual == 0.0 || squared_expected == 0.0) {
        throw std::runtime_error("attention quality norm is zero");
    }
    return {
        std::sqrt(static_cast<double>(squared_error / squared_expected)),
        static_cast<double>(
            dot_product / std::sqrt(squared_actual * squared_expected)),
        maximum};
}

double MaximumInactive(
    const std::vector<std::vector<double>>& values,
    std::size_t active_features) {
    double maximum = 0.0;
    for (const auto& tensor : values) {
        for (std::size_t slot = active_features; slot < tensor.size(); ++slot) {
            if (!std::isfinite(tensor[slot])) {
                throw std::runtime_error("attention inactive checkpoint is non-finite");
            }
            maximum = std::max(maximum, std::abs(tensor[slot]));
        }
    }
    return maximum;
}

template <typename Function>
void RequireInvalidArgument(Function&& function, const std::string& label) {
    bool rejected = false;
    try {
        function();
    }
    catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error(label + " did not fail closed");
    }
}

void RequireTensorShape(
    const moai::openfhe::CipherTensor& tensor,
    std::size_t block_dimension,
    std::size_t ciphertext_count,
    const std::string& label) {
    if (tensor.size() != ciphertext_count ||
        tensor.packing.layout != moai::openfhe::PackingLayout::kContiguous ||
        tensor.packing.logical_shape != std::vector<std::size_t>{
            block_dimension,
            ciphertext_count}) {
        throw std::runtime_error(label + " packing metadata drifted");
    }
}

void RequireNoHomomorphicWork(
    const moai::openfhe::RunMetrics& metrics,
    const std::string& label) {
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
        throw std::runtime_error(label + " did not fail before homomorphic work");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 1) {
            throw std::invalid_argument(
                "openfhe_feature_packed_attention_smoke accepts no arguments");
        }
        const std::size_t kBlockDimension = 1024;
        const std::size_t kTokenCount = 2;
        const std::size_t kHeadCount = 12;
        const std::size_t kHeadDimension = 64;
        const std::size_t kFeatureDimension =
            kHeadCount * kHeadDimension;
        constexpr std::size_t kLayer = 0;
        const moai::openfhe::FeaturePackedAttentionSpec spec{
            kBlockDimension,
            kTokenCount,
            kHeadCount,
            kHeadDimension};

        const auto paper_spec =
            moai::openfhe::MakePaperCompatFeaturePackedAttentionSpec();
        if (paper_spec.block_dimension != 1024 ||
            paper_spec.token_count != 5 ||
            paper_spec.head_count != 12 ||
            paper_spec.head_dimension != 64 ||
            moai::openfhe::kPaperCompatAttentionFeatureDimension != 768 ||
            moai::openfhe::kPaperCompatAttentionScoreScale != 0.125) {
            throw std::runtime_error("paper_compat attention contract drifted");
        }

        std::vector<int32_t> expected_rotations;
        for (std::size_t step = 1; step < kBlockDimension; step *= 2) {
            expected_rotations.push_back(static_cast<int32_t>(step));
        }
        const auto rotations =
            moai::openfhe::FeaturePackedAttentionRotationIndices(spec);
        if (rotations != expected_rotations) {
            throw std::runtime_error(
                "attention isolated-sum rotation contract drifted");
        }
        RequireInvalidArgument(
            [] {
                static_cast<void>(
                    moai::openfhe::FeaturePackedAttentionRotationIndices(
                        {32, 2, 11, 2}));
            },
            "non-12-head attention spec");
        RequireInvalidArgument(
            [] {
                static_cast<void>(
                    moai::openfhe::FeaturePackedAttentionRotationIndices(
                        {16, 2, 12, 2}));
            },
            "attention feature span outside block");

        const auto contracts =
            moai::openfhe::MakePaperCompatNonlinearContracts();
        auto profile = moai::openfhe::MakePaperCompatFeaturePackedProfile();
        constexpr const char* kExpectedFeatureProfileSha256 =
            "94f30e628e21f02146ce7ed9820194eabba3820f6e1e17176a31f8c5acf8b0be";
        if (profile.parameter_sha256 != kExpectedFeatureProfileSha256 ||
            profile.multiplicative_depth != 47 ||
            profile.levels_available_after_bootstrap != 28 ||
            profile.bootstrap_slots != 1024 ||
            profile.bootstrap_iterations != 2 ||
            profile.bootstrap_precision != 14) {
            throw std::runtime_error(
                "paper_compat feature-packed profile drifted");
        }
        moai::openfhe::PrintSecurityDisclosure(profile, std::cout);

        moai::openfhe::PackingSpec packing;
        packing.layout = moai::openfhe::PackingLayout::kContiguous;
        packing.logical_shape = {kBlockDimension, kTokenCount};
        packing.batch_lanes = 1;
        packing.slot_count = profile.slot_count;
        packing.active_slots = kBlockDimension;
        packing.encoded_slots = kBlockDimension;
        // M4 starts the encoder input at level 29; Q/K/V each consume one
        // DenseAffine level before entering attention.
        packing.level = 30;
        packing.noise_scale_degree = 1;
        packing.scaling_factor =
            std::ldexp(1.0, static_cast<int>(profile.scaling_modulus_bits));

        std::vector<std::vector<double>> q(
            kTokenCount,
            std::vector<double>(kBlockDimension));
        std::vector<std::vector<double>> k = q;
        std::vector<std::vector<double>> v = q;
        for (std::size_t token = 0; token < kTokenCount; ++token) {
            for (std::size_t head = 0; head < kHeadCount; ++head) {
                for (std::size_t feature = 0;
                     feature < kHeadDimension;
                     ++feature) {
                    const std::size_t slot =
                        head * kHeadDimension + feature;
                    // Adjacent heads have deliberately different sentinels.
                    // A cross-head reduction changes every expected score.
                    const double feature_weight =
                        feature == 0 ? 1.0 : 0.6;
                    const double q_base =
                        0.8 + 0.025 * static_cast<double>(head);
                    const double shifted_target = token == 0 ? -0.25 : 0.20;
                    const double score_target =
                        contracts.softmax_shifts.At(kLayer, head) + shifted_target;
                    const double feature_energy =
                        1.0 + 0.36 * static_cast<double>(kHeadDimension - 1);
                    const double k_base =
                        score_target * std::sqrt(static_cast<double>(kHeadDimension)) /
                        (q_base * feature_energy);
                    q[token][slot] = q_base * feature_weight;
                    k[token][slot] = k_base * feature_weight;
                    v[token][slot] =
                        0.021 * static_cast<double>((head + 2) * (feature + 1)) -
                        0.035 * static_cast<double>(token + 1);
                }
            }
            for (std::size_t slot = kFeatureDimension;
                 slot < kBlockDimension;
                 ++slot) {
                q[token][slot] =
                    20.0 + static_cast<double>(3 * token + slot);
                k[token][slot] =
                    -17.0 - static_cast<double>(5 * token + slot);
                v[token][slot] = 0.0;
            }
        }

        std::vector<std::vector<double>> score_oracle(
            kTokenCount * kTokenCount,
            std::vector<double>(kBlockDimension));
        std::vector<std::vector<double>> probability_oracle = score_oracle;
        std::vector<std::vector<double>> output_oracle(
            kTokenCount,
            std::vector<double>(kBlockDimension));
        const double score_scale =
            1.0 / std::sqrt(static_cast<double>(kHeadDimension));
        for (std::size_t query = 0; query < kTokenCount; ++query) {
            for (std::size_t head = 0; head < kHeadCount; ++head) {
                std::vector<double> exponentials(kTokenCount);
                double denominator = 0.0;
                for (std::size_t key = 0; key < kTokenCount; ++key) {
                    double score = 0.0;
                    for (std::size_t feature = 0;
                         feature < kHeadDimension;
                         ++feature) {
                        const std::size_t slot =
                            head * kHeadDimension + feature;
                        score += q[query][slot] * k[key][slot];
                    }
                    score *= score_scale;
                    for (std::size_t feature = 0;
                         feature < kHeadDimension;
                         ++feature) {
                        score_oracle[query * kTokenCount + key]
                                    [head * kHeadDimension + feature] = score;
                    }
                    const double shifted = score -
                        contracts.softmax_shifts.At(kLayer, head);
                    exponentials[key] = EvaluateChebyshev(
                        contracts.softmax_exponential,
                        shifted);
                    denominator += exponentials[key];
                }
                const double reciprocal = EvaluateChebyshev(
                    contracts.softmax_reciprocal,
                    denominator);
                for (std::size_t key = 0; key < kTokenCount; ++key) {
                    const double probability = exponentials[key] * reciprocal;
                    for (std::size_t feature = 0;
                         feature < kHeadDimension;
                         ++feature) {
                        const std::size_t slot =
                            head * kHeadDimension + feature;
                        probability_oracle[query * kTokenCount + key][slot] =
                            probability;
                        output_oracle[query][slot] +=
                            probability * v[key][slot];
                    }
                }
            }
        }

        QualityMetrics score_quality;
        QualityMetrics probability_quality;
        QualityMetrics raw_output_quality;
        QualityMetrics output_quality;
        double inactive_maximum = 0.0;
        double probability_inactive_maximum = 0.0;
        double raw_output_inactive_maximum = 0.0;
        double output_inactive_maximum = 0.0;
        double normalization_error = 0.0;
        moai::openfhe::RunMetrics operation_counts;
        {
            moai::openfhe::ClientRuntime client(profile);
            client.GenerateEvaluationKeys(rotations, true);
            const auto server_bundle = client.ExportServerKeyBundle();
            auto missing_rotation_bundle = server_bundle;
            moai::openfhe::ServerKeyBundleTestAccess::RemoveDeclaredRotation(
                missing_rotation_bundle,
                rotations.back());
            moai::openfhe::ServerRuntime missing_rotation_server(
                missing_rotation_bundle,
                profile);
            moai::openfhe::FeaturePackedAttention missing_rotation_attention(
                missing_rotation_server);
            auto missing_bootstrap_bundle = server_bundle;
            moai::openfhe::ServerKeyBundleTestAccess::RemoveBootstrapCapability(
                missing_bootstrap_bundle);
            moai::openfhe::ServerRuntime missing_bootstrap_server(
                missing_bootstrap_bundle,
                profile);
            moai::openfhe::FeaturePackedAttention missing_bootstrap_attention(
                missing_bootstrap_server);
            moai::openfhe::ServerRuntime server(
                server_bundle,
                profile);
            moai::openfhe::FeaturePackedAttention attention(server);
            const auto encrypted_q = client.Encrypt(q, packing);
            const auto encrypted_k = client.Encrypt(k, packing);
            const auto encrypted_v = client.Encrypt(v, packing);

            bool missing_rotation_rejected = false;
            try {
                static_cast<void>(missing_rotation_attention.Evaluate(
                    encrypted_q,
                    encrypted_k,
                    encrypted_v,
                    kLayer,
                    {1.0, 1.0},
                    spec));
            }
            catch (const std::logic_error&) {
                missing_rotation_rejected = true;
            }
            if (!missing_rotation_rejected) {
                throw std::runtime_error(
                    "attention accepted a missing late rotation declaration");
            }
            RequireNoHomomorphicWork(
                missing_rotation_server.metrics(),
                "missing attention rotation preflight");

            bool missing_bootstrap_rejected = false;
            try {
                static_cast<void>(missing_bootstrap_attention.Evaluate(
                    encrypted_q,
                    encrypted_k,
                    encrypted_v,
                    kLayer,
                    {1.0, 1.0},
                    spec));
            }
            catch (const std::logic_error&) {
                missing_bootstrap_rejected = true;
            }
            if (!missing_bootstrap_rejected) {
                throw std::runtime_error(
                    "attention accepted a missing bootstrap capability");
            }
            RequireNoHomomorphicWork(
                missing_bootstrap_server.metrics(),
                "missing attention bootstrap preflight");

            RequireInvalidArgument(
                [&] {
                    static_cast<void>(attention.Evaluate(
                        encrypted_q,
                        encrypted_k,
                        encrypted_v,
                        kLayer,
                        {1.0, 0.0},
                        spec));
                },
                "activation-dependent/padded key mask");
            auto wrong_layout = encrypted_q;
            wrong_layout.packing.layout =
                moai::openfhe::PackingLayout::kColumn;
            RequireInvalidArgument(
                [&] {
                    static_cast<void>(attention.Evaluate(
                        wrong_layout,
                        encrypted_k,
                        encrypted_v,
                        kLayer,
                        {1.0, 1.0},
                        spec));
                },
                "non-feature-packed Q");

            const auto result = attention.Evaluate(
                encrypted_q,
                encrypted_k,
                encrypted_v,
                kLayer,
                {1.0, 1.0},
                spec);
            RequireTensorShape(
                result.scaled_scores,
                kBlockDimension,
                kTokenCount * kTokenCount,
                "attention score checkpoint");
            RequireTensorShape(
                result.probabilities,
                kBlockDimension,
                kTokenCount * kTokenCount,
                "attention probability checkpoint");
            RequireTensorShape(
                result.output_before_bootstrap_cleanup,
                kBlockDimension,
                kTokenCount,
                "attention output before bootstrap cleanup");
            RequireTensorShape(
                result.output,
                kBlockDimension,
                kTokenCount,
                "attention output");

            const auto scores = client.Decrypt(result.scaled_scores);
            const auto probabilities = client.Decrypt(result.probabilities);
            const auto raw_output = client.Decrypt(
                result.output_before_bootstrap_cleanup);
            const auto output = client.Decrypt(result.output);
            score_quality = MeasureQuality(
                scores,
                score_oracle,
                kFeatureDimension);
            probability_quality = MeasureQuality(
                probabilities,
                probability_oracle,
                kFeatureDimension);
            raw_output_quality = MeasureQuality(
                raw_output,
                output_oracle,
                kFeatureDimension);
            output_quality = MeasureQuality(
                output,
                output_oracle,
                kFeatureDimension);
            probability_inactive_maximum =
                MaximumInactive(probabilities, kFeatureDimension);
            raw_output_inactive_maximum =
                MaximumInactive(raw_output, kFeatureDimension);
            output_inactive_maximum =
                MaximumInactive(output, kFeatureDimension);
            inactive_maximum = std::max({
                MaximumInactive(scores, kFeatureDimension),
                probability_inactive_maximum,
                raw_output_inactive_maximum,
                output_inactive_maximum,
            });
            for (std::size_t query = 0; query < kTokenCount; ++query) {
                for (std::size_t head = 0; head < kHeadCount; ++head) {
                    for (std::size_t feature = 0;
                         feature < kHeadDimension;
                         ++feature) {
                        const std::size_t slot =
                            head * kHeadDimension + feature;
                        double sum = 0.0;
                        for (std::size_t key = 0; key < kTokenCount; ++key) {
                            sum += probabilities[
                                query * kTokenCount + key][slot];
                        }
                        normalization_error = std::max(
                            normalization_error,
                            std::abs(sum - 1.0));
                    }
                }
            }
            operation_counts = server.metrics();
        }

        if (score_quality.relative_l2 > 1e-4 ||
            score_quality.cosine < 0.99999 ||
            score_quality.max_absolute > 1e-6) {
            throw std::runtime_error(
                "isolated per-head encrypted score gate failed");
        }
        if (probability_quality.relative_l2 > 1e-2 ||
            probability_quality.cosine < 0.999 ||
            raw_output_quality.relative_l2 > 1e-2 ||
            raw_output_quality.cosine < 0.999 ||
            output_quality.relative_l2 > 1e-2 ||
            output_quality.cosine < 0.999 ||
            normalization_error > 1e-2 ||
            inactive_maximum > 1e-6) {
            std::cerr
                << "attention diagnostic probability_rel_l2="
                << probability_quality.relative_l2
                << " probability_cosine=" << probability_quality.cosine
                << " raw_output_rel_l2=" << raw_output_quality.relative_l2
                << " raw_output_cosine=" << raw_output_quality.cosine
                << " output_rel_l2=" << output_quality.relative_l2
                << " output_cosine=" << output_quality.cosine
                << " normalization_max_abs=" << normalization_error
                << " probabilities_inactive_max_abs="
                << probability_inactive_maximum
                << " raw_attention_output_inactive_max_abs="
                << raw_output_inactive_maximum
                << " attention_output_inactive_max_abs="
                << output_inactive_maximum << '\n';
            throw std::runtime_error(
                "paper_compat encrypted attention correctness gate failed");
        }
        if (operation_counts.bootstraps != 2 * kTokenCount) {
            throw std::runtime_error(
                "attention must perform one Softmax and one output bootstrap per query");
        }
        const uint64_t expected_rotation_count =
            kTokenCount * kTokenCount * kHeadCount * rotations.size();
        if (operation_counts.rotations != expected_rotation_count) {
            throw std::runtime_error(
                "attention isolated-head rotation count drifted");
        }
        if (operation_counts.ct_pt_multiplications != 98 ||
            operation_counts.ct_ct_multiplications != 12 ||
            operation_counts.rescale_operations != 110) {
            throw std::runtime_error(
                "feature-packed attention multiplication schedule drifted");
        }

        std::cout
            << "{\"test\":\"openfhe_feature_packed_attention_smoke\","
            << "\"profile\":\"paper_compat\","
            << "\"profile_parameter_sha256\":\""
            << profile.parameter_sha256 << "\","
            << "\"security_claim\":\"none\","
            << "\"block_dimension\":" << kBlockDimension << ","
            << "\"token_count\":" << kTokenCount << ","
            << "\"head_count\":" << kHeadCount << ","
            << "\"head_dimension\":" << kHeadDimension << ","
            << "\"score_rel_l2\":" << score_quality.relative_l2 << ","
            << "\"score_cosine\":" << score_quality.cosine << ","
            << "\"probability_rel_l2\":"
            << probability_quality.relative_l2 << ","
            << "\"probability_cosine\":" << probability_quality.cosine << ","
            << "\"raw_output_rel_l2\":" << raw_output_quality.relative_l2 << ","
            << "\"raw_output_cosine\":" << raw_output_quality.cosine << ","
            << "\"output_rel_l2\":" << output_quality.relative_l2 << ","
            << "\"output_cosine\":" << output_quality.cosine << ","
            << "\"normalization_max_abs\":" << normalization_error << ","
            << "\"inactive_max_abs\":" << inactive_maximum << ","
            << "\"probabilities_inactive_max_abs\":"
            << probability_inactive_maximum << ","
            << "\"raw_attention_output_inactive_max_abs\":"
            << raw_output_inactive_maximum << ","
            << "\"attention_output_inactive_max_abs\":"
            << output_inactive_maximum << ","
            << "\"rotations\":" << operation_counts.rotations << ","
            << "\"ct_pt\":" << operation_counts.ct_pt_multiplications << ","
            << "\"ct_ct\":" << operation_counts.ct_ct_multiplications << ","
            << "\"rescale\":" << operation_counts.rescale_operations << ","
            << "\"bootstraps\":" << operation_counts.bootstraps << ","
            << "\"multiplicative_depth\":"
            << operation_counts.multiplicative_depth << ","
            << "\"max_observed_level\":"
            << operation_counts.max_observed_level << ","
            << "\"max_polynomial_depth\":"
            << operation_counts.max_polynomial_depth
            << "}\n";
        return 0;
    }
    catch (const std::exception& exception) {
        std::cerr << "openfhe_feature_packed_attention_smoke failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
