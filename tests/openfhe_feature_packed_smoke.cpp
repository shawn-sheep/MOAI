#include "moai/openfhe/client_runtime.hpp"
#include "moai/openfhe/context_factory.hpp"
#include "moai/openfhe/feature_packed_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct QualityMetrics {
    double relative_l2{0.0};
    double cosine{0.0};
    double max_absolute{0.0};
};

QualityMetrics MeasureQuality(
    const std::vector<std::vector<double>>& actual,
    const std::vector<std::vector<double>>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("feature-packed tensor count mismatch");
    }
    long double squared_error = 0.0;
    long double squared_actual = 0.0;
    long double squared_expected = 0.0;
    long double dot_product = 0.0;
    double maximum = 0.0;
    for (std::size_t token = 0; token < actual.size(); ++token) {
        if (actual[token].size() != expected[token].size()) {
            throw std::runtime_error("feature-packed vector size mismatch");
        }
        for (std::size_t slot = 0; slot < actual[token].size(); ++slot) {
            if (!std::isfinite(actual[token][slot])) {
                throw std::runtime_error(
                    "feature-packed affine output contains NaN or Inf");
            }
            const long double observed = actual[token][slot];
            const long double reference = expected[token][slot];
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
        throw std::runtime_error("feature-packed quality norm is zero");
    }
    return {
        std::sqrt(static_cast<double>(squared_error / squared_expected)),
        static_cast<double>(
            dot_product / std::sqrt(squared_actual * squared_expected)),
        maximum};
}

double MaximumInactive(
    const std::vector<std::vector<double>>& values,
    const std::vector<double>& active_mask) {
    double maximum = 0.0;
    for (const auto& token : values) {
        if (token.size() != active_mask.size()) {
            throw std::runtime_error(
                "feature-packed inactive-mask dimension mismatch");
        }
        for (std::size_t slot = 0; slot < token.size(); ++slot) {
            if (active_mask[slot] < 0.5) {
                maximum = std::max(maximum, std::abs(token[slot]));
            }
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

void RequireNoOperationCountChange(
    const moai::openfhe::RunMetrics& before,
    const moai::openfhe::RunMetrics& after,
    const std::string& label) {
    if (after.rotations != before.rotations ||
        after.ct_pt_multiplications != before.ct_pt_multiplications ||
        after.ct_ct_multiplications != before.ct_ct_multiplications ||
        after.rescale_operations != before.rescale_operations ||
        after.bootstraps != before.bootstraps ||
        after.bootstrap_iterations != before.bootstrap_iterations ||
        after.chebyshev_evaluations != before.chebyshev_evaluations ||
        after.estimated_polynomial_multiplications !=
            before.estimated_polynomial_multiplications) {
        throw std::runtime_error(label + " performed homomorphic work before rejection");
    }
}

}  // namespace

int main() {
    try {
        constexpr std::size_t kBlockDimension = 16;
        constexpr std::size_t kInputDimension = 5;
        constexpr std::size_t kOutputDimension = 7;
        constexpr std::size_t kTokenCount = 2;
        constexpr std::size_t kBabyStep = 4;
        const moai::openfhe::FeaturePackedAffineSpec spec{
            kBlockDimension,
            kInputDimension,
            kOutputDimension,
            kBabyStep};
        const std::vector<int32_t> expected_rotations{-4, 1, 2, 3, 4, 8};
        const auto rotations =
            moai::openfhe::FeaturePackedAffineRotationIndices(spec);
        if (rotations != expected_rotations) {
            throw std::runtime_error(
                "feature-packed exact rotation-key enumeration drifted");
        }

        const auto target_rotations =
            moai::openfhe::FeaturePackedAffineRotationIndices(
                {4096, 768, 3072, 64});
        const auto has_target_rotation = [&](int32_t index) {
            return std::binary_search(
                target_rotations.begin(),
                target_rotations.end(),
                index);
        };
        if (target_rotations.size() != 122 ||
            !has_target_rotation(-1984) ||
            !has_target_rotation(-64) ||
            !has_target_rotation(1) ||
            !has_target_rotation(63) ||
            !has_target_rotation(64) ||
            !has_target_rotation(704) ||
            !has_target_rotation(1024) ||
            !has_target_rotation(2048) ||
            has_target_rotation(768) ||
            has_target_rotation(960)) {
            throw std::runtime_error(
                "target 4096 feature block rotation-key enumeration drifted");
        }

        const auto encoder_rotations =
            moai::openfhe::FeaturePackedAffineRotationIndices(
                {1024, 768, 768, 32});
        const auto has_encoder_rotation = [&](int32_t index) {
            return std::binary_search(
                encoder_rotations.begin(),
                encoder_rotations.end(),
                index);
        };
        if (encoder_rotations.size() != 62 ||
            !has_encoder_rotation(-480) ||
            !has_encoder_rotation(-32) ||
            !has_encoder_rotation(1) ||
            !has_encoder_rotation(31) ||
            !has_encoder_rotation(32) ||
            !has_encoder_rotation(512) ||
            has_encoder_rotation(-512) ||
            has_encoder_rotation(33)) {
            throw std::runtime_error(
                "M4 1024-slot encoder rotation-key enumeration drifted");
        }

        RequireInvalidArgument(
            [] {
                static_cast<void>(
                    moai::openfhe::FeaturePackedAffineRotationIndices(
                        {12, 5, 7, 4}));
            },
            "non-power-of-two block");
        RequireInvalidArgument(
            [] {
                static_cast<void>(
                    moai::openfhe::FeaturePackedAffineRotationIndices(
                        {16, 5, 17, 4}));
            },
            "oversized output dimension");
        RequireInvalidArgument(
            [] {
                static_cast<void>(
                    moai::openfhe::FeaturePackedAffineRotationIndices(
                        {16, 5, 7, 3}));
            },
            "invalid baby step");

        auto profile = moai::openfhe::MakePaperCompatProfile();
        moai::openfhe::PrintSecurityDisclosure(profile, std::cout);
        moai::openfhe::PackingSpec packing;
        packing.layout = moai::openfhe::PackingLayout::kContiguous;
        packing.logical_shape = {kBlockDimension, kTokenCount};
        packing.batch_lanes = 1;
        packing.slot_count = profile.slot_count;
        packing.active_slots = kBlockDimension;
        packing.encoded_slots = kBlockDimension;
        packing.level = 0;
        packing.noise_scale_degree = 1;
        packing.scaling_factor =
            std::ldexp(1.0, static_cast<int>(profile.scaling_modulus_bits));

        std::vector<std::vector<double>> input(
            kTokenCount,
            std::vector<double>(kBlockDimension));
        for (std::size_t token = 0; token < kTokenCount; ++token) {
            for (std::size_t feature = 0;
                 feature < kInputDimension;
                 ++feature) {
                input[token][feature] =
                    0.13 * static_cast<double>((token + 1) * (feature + 2)) -
                    0.41;
            }
            for (std::size_t feature = kInputDimension;
                 feature < kBlockDimension;
                 ++feature) {
                // Nonzero sentinels prove that slots outside input_dimension
                // cannot enter the rectangular affine result.
                input[token][feature] =
                    10.0 + static_cast<double>(token + feature);
            }
        }

        moai::openfhe::FeaturePackedWeights weights(
            kInputDimension,
            std::vector<double>(kOutputDimension));
        for (std::size_t input_feature = 0;
             input_feature < kInputDimension;
             ++input_feature) {
            for (std::size_t output_feature = 0;
                 output_feature < kOutputDimension;
                 ++output_feature) {
                weights[input_feature][output_feature] =
                    0.025 * static_cast<double>(
                        (input_feature + 1) * (output_feature + 2)) -
                    0.17;
            }
        }
        std::vector<double> bias(kOutputDimension);
        for (std::size_t output_feature = 0;
             output_feature < kOutputDimension;
             ++output_feature) {
            bias[output_feature] =
                0.03 * static_cast<double>(output_feature) - 0.07;
        }
        std::vector<double> output_mask(kBlockDimension, 0.0);
        for (std::size_t feature = 0;
             feature < kOutputDimension;
             ++feature) {
            output_mask[feature] = 1.0;
        }
        output_mask[3] = 0.0;

        std::vector<std::vector<double>> expected(
            kTokenCount,
            std::vector<double>(kBlockDimension));
        for (std::size_t token = 0; token < kTokenCount; ++token) {
            for (std::size_t output_feature = 0;
                 output_feature < kOutputDimension;
                 ++output_feature) {
                if (output_mask[output_feature] < 0.5) {
                    continue;
                }
                expected[token][output_feature] = bias[output_feature];
                for (std::size_t input_feature = 0;
                     input_feature < kInputDimension;
                     ++input_feature) {
                    expected[token][output_feature] +=
                        input[token][input_feature] *
                        weights[input_feature][output_feature];
                }
            }
        }

        constexpr std::size_t kFusedTerms = 3;
        std::vector<std::vector<std::vector<double>>> fused_inputs(
            kFusedTerms,
            input);
        std::vector<moai::openfhe::FeaturePackedWeights> fused_weights(
            kFusedTerms,
            weights);
        for (std::size_t term = 0; term < kFusedTerms; ++term) {
            for (std::size_t token = 0; token < kTokenCount; ++token) {
                for (std::size_t feature = 0;
                     feature < kInputDimension;
                     ++feature) {
                    fused_inputs[term][token][feature] +=
                        0.017 * static_cast<double>(
                            (term + 1) * (token + 1) * (feature + 1));
                }
            }
            for (std::size_t input_feature = 0;
                 input_feature < kInputDimension;
                 ++input_feature) {
                for (std::size_t output_feature = 0;
                     output_feature < kOutputDimension;
                     ++output_feature) {
                    fused_weights[term][input_feature][output_feature] +=
                        0.004 * static_cast<double>(
                            (term + 1) * (input_feature + output_feature + 1));
                }
            }
        }

        std::vector<std::vector<std::vector<double>>>
            fused_expected_contributions(
                kFusedTerms,
                std::vector<std::vector<double>>(
                    kTokenCount,
                    std::vector<double>(kBlockDimension)));
        std::vector<std::vector<double>> fused_expected(
            kTokenCount,
            std::vector<double>(kBlockDimension));
        for (std::size_t token = 0; token < kTokenCount; ++token) {
            for (std::size_t output_feature = 0;
                 output_feature < kOutputDimension;
                 ++output_feature) {
                if (output_mask[output_feature] < 0.5) {
                    continue;
                }
                fused_expected[token][output_feature] = bias[output_feature];
                for (std::size_t term = 0; term < kFusedTerms; ++term) {
                    for (std::size_t input_feature = 0;
                         input_feature < kInputDimension;
                         ++input_feature) {
                        fused_expected_contributions[term][token][output_feature] +=
                            fused_inputs[term][token][input_feature] *
                            fused_weights[term][input_feature][output_feature];
                    }
                    fused_expected[token][output_feature] +=
                        fused_expected_contributions[term][token][output_feature];
                }
            }
        }

        QualityMetrics quality;
        QualityMetrics fused_quality;
        QualityMetrics worst_raw_contribution_quality;
        double fused_inactive_maximum = 0.0;
        moai::openfhe::RunMetrics operation_counts;
        moai::openfhe::RunMetrics single_operation_counts;
        {
            moai::openfhe::ClientRuntime client(profile);
            client.GenerateEvaluationKeys(rotations, false);
            moai::openfhe::ServerRuntime server(
                client.ExportServerKeyBundle(),
                profile);
            moai::openfhe::FeaturePackedOps ops(server);
            const auto encrypted = client.Encrypt(input, packing);
            std::vector<moai::openfhe::CipherTensor> fused_encrypted;
            fused_encrypted.reserve(kFusedTerms);
            for (const auto& term_input : fused_inputs) {
                fused_encrypted.push_back(client.Encrypt(term_input, packing));
            }
            std::vector<moai::openfhe::FeaturePackedAffineTermView> fused_terms;
            fused_terms.reserve(kFusedTerms);
            for (std::size_t term = 0; term < kFusedTerms; ++term) {
                fused_terms.push_back({
                    &fused_encrypted[term],
                    &fused_weights[term]});
            }

            const auto metrics_before_sum_rejections = server.metrics();
            RequireInvalidArgument(
                [&] {
                    static_cast<void>(ops.DenseAffineSum(
                        {},
                        bias,
                        output_mask,
                        spec));
                },
                "empty affine sum");
            auto null_terms = fused_terms;
            null_terms[1].input = nullptr;
            RequireInvalidArgument(
                [&] {
                    static_cast<void>(ops.DenseAffineSum(
                        null_terms,
                        bias,
                        output_mask,
                        spec));
                },
                "null affine-sum term");
            auto ragged_sum_weights = fused_weights[1];
            ragged_sum_weights.front().pop_back();
            auto ragged_terms = fused_terms;
            ragged_terms[1].weights = &ragged_sum_weights;
            RequireInvalidArgument(
                [&] {
                    static_cast<void>(ops.DenseAffineSum(
                        ragged_terms,
                        bias,
                        output_mask,
                        spec));
                },
                "ragged later affine-sum term");
            auto stale_sum_input = fused_encrypted[1];
            ++stale_sum_input.packing.level;
            auto stale_terms = fused_terms;
            stale_terms[1].input = &stale_sum_input;
            RequireInvalidArgument(
                [&] {
                    static_cast<void>(ops.DenseAffineSum(
                        stale_terms,
                        bias,
                        output_mask,
                        spec));
                },
                "mismatched later affine-sum level");
            RequireNoOperationCountChange(
                metrics_before_sum_rejections,
                server.metrics(),
                "affine-sum contract rejection");

            auto null_ciphertext = encrypted;
            null_ciphertext.ciphertexts.front().reset();
            RequireInvalidArgument(
                [&] {
                    static_cast<void>(
                        server.RemainingLevels(null_ciphertext));
                },
                "null-ciphertext remaining-level inspection");

            auto stale_level = encrypted;
            ++stale_level.packing.level;
            RequireInvalidArgument(
                [&] {
                    static_cast<void>(server.RemainingLevels(stale_level));
                },
                "stale-level remaining-level inspection");

            auto wrong_layout = encrypted;
            wrong_layout.packing.layout =
                moai::openfhe::PackingLayout::kColumn;
            RequireInvalidArgument(
                [&] {
                    static_cast<void>(ops.DenseAffine(
                        wrong_layout,
                        weights,
                        bias,
                        output_mask,
                        spec));
                },
                "wrong feature-packed layout");

            auto wrong_shape = encrypted;
            wrong_shape.packing.logical_shape = {
                kBlockDimension,
                kTokenCount + 1};
            RequireInvalidArgument(
                [&] {
                    static_cast<void>(ops.DenseAffine(
                        wrong_shape,
                        weights,
                        bias,
                        output_mask,
                        spec));
                },
                "wrong feature-packed logical shape");

            auto wrong_slot_count = encrypted;
            wrong_slot_count.packing.slot_count /= 2;
            RequireInvalidArgument(
                [&] {
                    static_cast<void>(ops.DenseAffine(
                        wrong_slot_count,
                        weights,
                        bias,
                        output_mask,
                        spec));
                },
                "profile-mismatched slot count");

            auto invalid_mask = output_mask;
            invalid_mask[kOutputDimension] = 1.0;
            RequireInvalidArgument(
                [&] {
                    static_cast<void>(ops.DenseAffine(
                        encrypted,
                        weights,
                        bias,
                        invalid_mask,
                        spec));
                },
                "out-of-range output mask");

            auto ragged_weights = weights;
            ragged_weights.front().pop_back();
            RequireInvalidArgument(
                [&] {
                    static_cast<void>(ops.DenseAffine(
                        encrypted,
                        ragged_weights,
                        bias,
                        output_mask,
                        spec));
                },
                "ragged weights");

            const auto result = ops.DenseAffine(
                encrypted,
                weights,
                bias,
                output_mask,
                spec);
            if (result.packing.layout !=
                    moai::openfhe::PackingLayout::kContiguous ||
                result.packing.logical_shape !=
                    std::vector<std::size_t>{
                        kBlockDimension,
                        kTokenCount} ||
                result.size() != kTokenCount) {
                throw std::runtime_error(
                    "feature-packed affine output metadata drifted");
            }
            quality = MeasureQuality(client.Decrypt(result), expected);
            single_operation_counts = server.metrics();

            const auto fused_result = ops.DenseAffineSum(
                fused_terms,
                bias,
                output_mask,
                spec);
            if (fused_result.raw_contributions.size() != kFusedTerms ||
                fused_result.output.size() != kTokenCount ||
                fused_result.output.packing.logical_shape !=
                    std::vector<std::size_t>{
                        kBlockDimension,
                        kTokenCount}) {
                throw std::runtime_error(
                    "feature-packed affine-sum output metadata drifted");
            }
            for (std::size_t term = 0; term < kFusedTerms; ++term) {
                const auto& raw = fused_result.raw_contributions[term];
                if (raw.size() != kTokenCount ||
                    raw.packing.logical_shape !=
                        std::vector<std::size_t>{
                            kBlockDimension,
                            kTokenCount}) {
                    throw std::runtime_error(
                        "feature-packed raw contribution metadata drifted");
                }
                const auto raw_quality = MeasureQuality(
                    client.Decrypt(raw),
                    fused_expected_contributions[term]);
                if (raw_quality.relative_l2 >
                    worst_raw_contribution_quality.relative_l2) {
                    worst_raw_contribution_quality = raw_quality;
                }
            }
            const auto fused_plaintext = client.Decrypt(fused_result.output);
            fused_quality = MeasureQuality(fused_plaintext, fused_expected);
            fused_inactive_maximum = MaximumInactive(
                fused_plaintext,
                output_mask);
            operation_counts = server.metrics();
        }

        if (quality.relative_l2 > 1e-4 || quality.cosine < 0.99999 ||
            quality.max_absolute > 1e-6) {
            throw std::runtime_error(
                "feature-packed affine encrypted quality gate failed");
        }
        if (single_operation_counts.rotations != 12 ||
            single_operation_counts.ct_pt_multiplications != 22 ||
            single_operation_counts.ct_ct_multiplications != 0 ||
            single_operation_counts.rescale_operations != 2) {
            throw std::runtime_error(
                "feature-packed affine operation counts drifted");
        }
        if (fused_quality.relative_l2 > 1e-4 ||
            fused_quality.cosine < 0.99999 ||
            worst_raw_contribution_quality.relative_l2 > 1e-4 ||
            worst_raw_contribution_quality.cosine < 0.99999 ||
            fused_inactive_maximum > 1e-6) {
            throw std::runtime_error(
                "feature-packed affine-sum encrypted quality gate failed");
        }
        if (operation_counts.rotations - single_operation_counts.rotations != 36 ||
            operation_counts.ct_pt_multiplications -
                single_operation_counts.ct_pt_multiplications != 66 ||
            operation_counts.ct_ct_multiplications -
                single_operation_counts.ct_ct_multiplications != 0 ||
            operation_counts.rescale_operations -
                single_operation_counts.rescale_operations != 2) {
            throw std::runtime_error(
                "feature-packed affine-sum operation counts drifted");
        }

        std::cout
            << "{\"test\":\"openfhe_feature_packed_smoke\","
            << "\"profile\":\"paper_compat\","
            << "\"security_claim\":\"none\","
            << "\"block_dimension\":" << kBlockDimension << ","
            << "\"input_dimension\":" << kInputDimension << ","
            << "\"output_dimension\":" << kOutputDimension << ","
            << "\"token_count\":" << kTokenCount << ","
            << "\"rotation_key_count\":" << rotations.size() << ","
            << "\"relative_l2\":" << quality.relative_l2 << ","
            << "\"cosine\":" << quality.cosine << ","
            << "\"max_abs\":" << quality.max_absolute << ","
            << "\"rotations\":" << single_operation_counts.rotations << ","
            << "\"ct_pt\":"
            << single_operation_counts.ct_pt_multiplications << ","
            << "\"ct_ct\":"
            << single_operation_counts.ct_ct_multiplications << ","
            << "\"rescale\":"
            << single_operation_counts.rescale_operations
            << ",\"fused_relative_l2\":" << fused_quality.relative_l2
            << ",\"fused_cosine\":" << fused_quality.cosine
            << ",\"fused_inactive_max_abs\":" << fused_inactive_maximum
            << ",\"fused_raw_worst_relative_l2\":"
            << worst_raw_contribution_quality.relative_l2
            << ",\"fused_rotations\":"
            << operation_counts.rotations - single_operation_counts.rotations
            << ",\"fused_ct_pt\":"
            << operation_counts.ct_pt_multiplications -
                single_operation_counts.ct_pt_multiplications
            << ",\"fused_rescale\":"
            << operation_counts.rescale_operations -
                single_operation_counts.rescale_operations
            << "}\n";
        return 0;
    }
    catch (const std::exception& exception) {
        std::cerr << "openfhe_feature_packed_smoke failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
