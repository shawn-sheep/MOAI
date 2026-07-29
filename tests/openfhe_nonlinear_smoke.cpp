#include "moai/openfhe/approximation_registry.hpp"
#include "moai/openfhe/client_runtime.hpp"
#include "moai/openfhe/context_factory.hpp"
#include "moai/openfhe/nonlinear_ops.hpp"


#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <initializer_list>
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

moai::openfhe::DeclaredRange ActiveRange(
    const std::vector<std::vector<double>>& values,
    const std::vector<double>& active_mask) {
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (const auto& tensor : values) {
        if (tensor.size() != active_mask.size()) {
            throw std::runtime_error("range tensor does not match active mask");
        }
        for (std::size_t slot = 0; slot < tensor.size(); ++slot) {
            if (active_mask[slot] > 0.5) {
                minimum = std::min(minimum, tensor[slot]);
                maximum = std::max(maximum, tensor[slot]);
            }
        }
    }
    if (!std::isfinite(minimum) || !std::isfinite(maximum)) {
        throw std::runtime_error("range calculation found no active value");
    }
    return {minimum, maximum};
}

void RequireRangeWithinContract(
    const moai::openfhe::DeclaredRange& observed,
    const moai::openfhe::ApproximationContract& contract,
    const std::string& label) {
    constexpr double kTolerance = 1e-12;
    if (observed.minimum < contract.interval.minimum - kTolerance ||
        observed.maximum > contract.interval.maximum + kTolerance) {
        throw std::runtime_error(
            label + " observed range leaves the offline-frozen interval");
    }
}

QualityMetrics MeasureQuality(
    const std::vector<std::vector<double>>& actual,
    const std::vector<std::vector<double>>& expected,
    const std::vector<double>& active_mask) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("quality tensor count mismatch");
    }
    long double squared_error = 0.0;
    long double squared_actual = 0.0;
    long double squared_expected = 0.0;
    long double dot_product = 0.0;
    double maximum = 0.0;
    for (std::size_t tensor = 0; tensor < actual.size(); ++tensor) {
        if (actual[tensor].size() != expected[tensor].size() ||
            actual[tensor].size() != active_mask.size()) {
            throw std::runtime_error("quality vector size mismatch");
        }
        for (std::size_t slot = 0; slot < actual[tensor].size(); ++slot) {
            if (active_mask[slot] < 0.5) {
                continue;
            }
            if (!std::isfinite(actual[tensor][slot])) {
                throw std::runtime_error("nonlinear output is not finite");
            }
            const long double actual_value = actual[tensor][slot];
            const long double expected_value = expected[tensor][slot];
            const long double error = actual_value - expected_value;
            squared_error += error * error;
            squared_actual += actual_value * actual_value;
            squared_expected += expected_value * expected_value;
            dot_product += actual_value * expected_value;
            maximum = std::max(maximum, std::abs(static_cast<double>(error)));
        }
    }
    if (squared_actual == 0.0 || squared_expected == 0.0) {
        throw std::runtime_error("quality metric received a zero-norm tensor");
    }
    QualityMetrics metrics;
    metrics.relative_l2 =
        std::sqrt(static_cast<double>(squared_error / squared_expected));
    metrics.cosine = static_cast<double>(
        dot_product / std::sqrt(squared_actual * squared_expected));
    metrics.max_absolute = maximum;
    return metrics;
}

void RequireQuality(
    const QualityMetrics& quality,
    const std::string& label) {
    if (quality.relative_l2 > 1e-2) {
        throw std::runtime_error(
            label + " rel-L2 exceeds 1e-2: " +
            std::to_string(quality.relative_l2));
    }
    if (quality.cosine < 0.999) {
        throw std::runtime_error(
            label + " cosine is below 0.999: " +
            std::to_string(quality.cosine));
    }
}

double MaximumInactive(
    const std::vector<std::vector<double>>& values,
    const std::vector<double>& active_mask) {
    double maximum = 0.0;
    for (const auto& tensor : values) {
        if (tensor.size() != active_mask.size()) {
            throw std::runtime_error(
                "inactive-slot tensor does not match active mask");
        }
        for (std::size_t slot = 0; slot < active_mask.size(); ++slot) {
            if (!std::isfinite(tensor[slot])) {
                throw std::runtime_error(
                    "inactive-slot checkpoint contains NaN or Inf");
            }
            if (active_mask[slot] < 0.5) {
                maximum = std::max(maximum, std::abs(tensor[slot]));
            }
        }
    }
    return maximum;
}

double RequireInactiveBound(
    const std::vector<std::vector<double>>& values,
    const std::vector<double>& active_mask,
    const std::string& label) {
    const double maximum = MaximumInactive(values, active_mask);
    if (maximum > 1e-6) {
        throw std::runtime_error(
            label + " inactive-slot max-abs exceeds 1e-6: " +
            std::to_string(maximum));
    }
    return maximum;
}

std::vector<double> PadSlots(
    std::initializer_list<double> values,
    std::size_t slot_count) {
    if (values.size() > slot_count) {
        throw std::invalid_argument("fixture values exceed smoke slot count");
    }
    std::vector<double> result(values);
    result.resize(slot_count, 0.0);
    return result;
}

void RequireAccuratePackingLevel(
    const moai::openfhe::CipherTensor& tensor,
    const std::string& label) {
    if (tensor.empty()) {
        throw std::runtime_error(label + " is empty");
    }
    for (const auto& ciphertext : tensor.ciphertexts) {
        if (ciphertext->GetLevel() != tensor.packing.level ||
            ciphertext->GetNoiseScaleDeg() !=
                tensor.packing.noise_scale_degree ||
            !std::isfinite(tensor.packing.scaling_factor) ||
            tensor.packing.scaling_factor <= 0.0 ||
            std::abs(
                ciphertext->GetScalingFactor() -
                tensor.packing.scaling_factor) >
                1e-12 * std::max(
                    1.0,
                    std::max(
                        std::abs(ciphertext->GetScalingFactor()),
                        std::abs(tensor.packing.scaling_factor)))) {
            throw std::runtime_error(label + " PackingSpec level/scale is stale");
        }
    }
}

void RequireLogicalShape(
    const moai::openfhe::CipherTensor& tensor,
    std::size_t logical_rows,
    std::size_t logical_features,
    const std::string& label) {
    if (tensor.packing.logical_shape !=
            std::vector<std::size_t>{logical_rows, logical_features} ||
        tensor.size() != logical_features) {
        throw std::runtime_error(
            label + " logical shape does not match its ciphertext count");
    }
}

std::vector<std::vector<double>> AffineOracle(
    const std::vector<std::vector<double>>& input,
    const moai::openfhe::DenseWeights& weights,
    const std::vector<double>& bias,
    const std::vector<double>& active_mask) {
    std::vector<std::vector<double>> output(
        bias.size(),
        std::vector<double>(active_mask.size()));
    for (std::size_t out = 0; out < bias.size(); ++out) {
        for (std::size_t slot = 0; slot < active_mask.size(); ++slot) {
            if (active_mask[slot] < 0.5) {
                continue;
            }
            output[out][slot] = bias[out];
            for (std::size_t in = 0; in < input.size(); ++in) {
                output[out][slot] += input[in][slot] * weights[in][out];
            }
        }
    }
    return output;
}

}  // namespace

int main() {
    try {
        const auto contracts =
            moai::openfhe::MakePaperCompatNonlinearContracts();
        const auto& gelu_contract = contracts.gelu;
        const auto& exponential_contract = contracts.softmax_exponential;
        const auto& reciprocal_contract = contracts.softmax_reciprocal;
        const auto& inverse_sqrt_contract =
            contracts.layernorm_inverse_sqrt;
        if (contracts.softmax_shifts.contract_id !=
                moai::openfhe::kPaperCompatSoftmaxShiftContractId ||
            contracts.softmax_shifts.values_sha256 !=
                moai::openfhe::kPaperCompatSoftmaxShiftSha256 ||
            contracts.softmax_shifts.layer_count !=
                moai::openfhe::kPaperCompatEncoderLayers ||
            contracts.softmax_shifts.head_count !=
                moai::openfhe::kPaperCompatAttentionHeads ||
            moai::openfhe::ComputeCoefficientSha256(
                contracts.softmax_shifts.values) !=
                moai::openfhe::kPaperCompatSoftmaxShiftSha256) {
            throw std::runtime_error(
                "paper_compat Softmax public-shift contract drifted");
        }
        const auto require_contract = [](
                                          const moai::openfhe::ApproximationContract&
                                              contract,
                                          const std::string& expected_id,
                                          double expected_minimum,
                                          double expected_maximum,
                                          uint32_t expected_degree,
                                          uint32_t expected_depth,
                                          uint64_t expected_multiplications,
                                          const std::string& expected_hash) {
            if (contract.contract_id != expected_id ||
                contract.interval.minimum != expected_minimum ||
                contract.interval.maximum != expected_maximum ||
                contract.degree != expected_degree ||
                contract.required_depth != expected_depth ||
                contract.estimated_multiplications !=
                    expected_multiplications ||
                contract.coefficient_sha256 != expected_hash ||
                moai::openfhe::ComputeCoefficientSha256(
                    contract.coefficients) != expected_hash) {
                throw std::runtime_error(
                    expected_id + " approximation contract drifted");
            }
        };
        require_contract(
            gelu_contract,
            "gelu_d319_zero_at_origin",
            -80.0,
            128.0,
            319,
            10,
            33,
            "35d68b2f56f267f27e8f962c3bd54cddd90892349b77405172f48987864eaaa3");
        require_contract(
            exponential_contract,
            "softmax_exp_d27",
            -16.0,
            5.0,
            27,
            6,
            10,
            "6eda4377151897e8c4ca4d72f2a918db0b888fc6771f8ee5cf1d950b378de76f");
        require_contract(
            reciprocal_contract,
            "softmax_reciprocal_d383",
            0.01,
            80.0,
            383,
            10,
            35,
            "fa97f298751bca97f40eed3b6de949262d1b3971cda57013b55420d5c9fbbeb3");
        require_contract(
            inverse_sqrt_contract,
            "layernorm_inverse_sqrt_d159",
            0.5,
            1536.0,
            159,
            9,
            23,
            "28d0ffd36436228da4ee6023e0aad7641be995492a4314375b5a1aafe68465be");
        if (
            std::abs(EvaluateChebyshev(gelu_contract, 0.0)) > 1e-12) {
            throw std::runtime_error(
                "paper_compat GELU zero-at-origin contract drifted");
        }
        if (contracts.depth_budget
                    .gelu_from_preactivation_through_output_affine != 12 ||
            contracts.depth_budget.softmax_pre_bootstrap != 7 ||
            contracts.depth_budget.softmax_post_bootstrap != 12 ||
            contracts.depth_budget.layernorm_pre_bootstrap != 4 ||
            contracts.depth_budget.layernorm_post_bootstrap != 11) {
            throw std::runtime_error(
                "paper_compat nonlinear depth budget drifted");
        }

        auto profile = moai::openfhe::MakePaperCompatProfile();
        constexpr uint32_t kSmokeSlots = 16;
        const uint32_t kPostBootstrapLevels = std::max({
            contracts.depth_budget
                .gelu_from_preactivation_through_output_affine,
            contracts.depth_budget.softmax_post_bootstrap,
            contracts.depth_budget.layernorm_post_bootstrap,
        });
        moai::openfhe::ConfigureBootstrap(
            profile,
            kSmokeSlots,
            kPostBootstrapLevels,
            2,
            10,
            0,
            false);
        if (profile.bootstrap_iterations != 2 ||
            profile.bootstrap_precision != 10 ||
            profile.bootstrap_correction_factor != 0 ||
            profile.bootstrap_slots_to_coefficients_first) {
            throw std::runtime_error("paper_compat Meta-BTS contract drifted");
        }
        constexpr const char* kEffectiveParameterSha256 =
            "965ded1a76b064c738f55b87b151df60c605d62eaeaad55f6cc753e149f8af59";
        if (profile.parameter_sha256 != kEffectiveParameterSha256) {
            throw std::runtime_error(
                "paper_compat effective parameter hash drifted");
        }
        moai::openfhe::PrintSecurityDisclosure(profile, std::cout);

        moai::openfhe::PackingSpec packing;
        packing.layout = moai::openfhe::PackingLayout::kContiguous;
        packing.logical_shape = {kSmokeSlots, 1};
        packing.batch_lanes = 1;
        packing.slot_count = profile.slot_count;
        packing.active_slots = kSmokeSlots;
        packing.encoded_slots = kSmokeSlots;
        packing.level = 0;
        packing.noise_scale_degree = 1;
        packing.scaling_factor =
            std::ldexp(1.0, static_cast<int>(profile.scaling_modulus_bits));

        const auto active_mask =
            PadSlots({1.0, 1.0, 1.0, 1.0}, kSmokeSlots);

        QualityMetrics gelu_quality;
        QualityMetrics ffn_quality;
        QualityMetrics softmax_quality;
        QualityMetrics denominator_quality;
        QualityMetrics layernorm_quality;
        double inactive_maximum = 0.0;
        double denominator_post_max_abs = 0.0;
        moai::openfhe::RunMetrics operation_counts;

        {
            moai::openfhe::ClientRuntime client(profile);
            client.GenerateEvaluationKeys({}, true);
            moai::openfhe::ServerRuntime server(
                client.ExportServerKeyBundle(),
                profile);
            moai::openfhe::NonlinearOps nonlinear(server);

            const std::vector<std::vector<double>> ffn_input{
                PadSlots({-2.0, -0.8, 0.3, 1.1}, kSmokeSlots),
                PadSlots({1.5, -1.0, 0.7, 2.0}, kSmokeSlots),
            };
            const moai::openfhe::DenseWeights input_weights{
                {0.8},
                {-0.4}};
            const std::vector<double> input_bias{0.15};
            const moai::openfhe::DenseWeights output_weights{
                {1.1, -0.7}};
            const std::vector<double> output_bias{0.05, -0.02};

            const auto ffn_pre_expected = AffineOracle(
                ffn_input,
                input_weights,
                input_bias,
                active_mask);
            auto ffn_activation_expected = ffn_pre_expected;
            for (std::size_t slot = 0; slot < active_mask.size(); ++slot) {
                if (active_mask[slot] > 0.5) {
                    ffn_activation_expected[0][slot] =
                        EvaluateChebyshev(
                            gelu_contract,
                            ffn_pre_expected[0][slot]);
                }
            }
            const auto ffn_output_expected = AffineOracle(
                ffn_activation_expected,
                output_weights,
                output_bias,
                active_mask);
            RequireRangeWithinContract(
                ActiveRange(ffn_pre_expected, active_mask),
                gelu_contract,
                "FFN pre-activation");

            if (profile.multiplicative_depth <=
                contracts.depth_budget
                    .gelu_from_preactivation_through_output_affine) {
                throw std::runtime_error(
                    "profile cannot exercise the conditional GELU bootstrap");
            }
            auto depleted_ffn_packing = packing;
            depleted_ffn_packing.logical_shape = {
                kSmokeSlots,
                ffn_input.size()};
            depleted_ffn_packing.level =
                profile.multiplicative_depth -
                contracts.depth_budget
                    .gelu_from_preactivation_through_output_affine;
            const uint64_t bootstraps_before_ffn = server.metrics().bootstraps;
            const uint64_t bootstrap_iterations_before_ffn =
                server.metrics().bootstrap_iterations;
            const auto encrypted_ffn_input =
                client.Encrypt(ffn_input, depleted_ffn_packing);
            RequireLogicalShape(
                encrypted_ffn_input,
                kSmokeSlots,
                ffn_input.size(),
                "encrypted FFN input");
            const auto ffn = nonlinear.FeedForward(
                encrypted_ffn_input,
                input_weights,
                input_bias,
                output_weights,
                output_bias,
                active_mask);
            if (server.metrics().bootstraps != bootstraps_before_ffn + 1) {
                throw std::runtime_error(
                    "depleted FFN input did not trigger exactly one "
                    "pre-GELU bootstrap");
            }
            if (server.metrics().bootstrap_iterations !=
                bootstrap_iterations_before_ffn + 2) {
                throw std::runtime_error(
                    "depleted FFN input did not trigger exactly two "
                    "Meta-BTS iterations");
            }
            RequireAccuratePackingLevel(
                ffn.pre_activation,
                "FFN pre-activation");
            RequireLogicalShape(
                ffn.pre_activation,
                kSmokeSlots,
                input_bias.size(),
                "FFN pre-activation");
            RequireAccuratePackingLevel(
                ffn.gelu.polynomial_input,
                "GELU polynomial input");
            RequireAccuratePackingLevel(
                ffn.gelu.polynomial_output,
                "GELU polynomial output");
            RequireAccuratePackingLevel(ffn.gelu.output, "GELU activation");
            RequireAccuratePackingLevel(ffn.output, "FFN output");
            RequireLogicalShape(
                ffn.gelu.polynomial_input,
                kSmokeSlots,
                input_bias.size(),
                "GELU polynomial input");
            RequireLogicalShape(
                ffn.gelu.polynomial_output,
                kSmokeSlots,
                input_bias.size(),
                "GELU polynomial output");
            RequireLogicalShape(
                ffn.gelu.output,
                kSmokeSlots,
                input_bias.size(),
                "GELU activation");
            RequireLogicalShape(
                ffn.output,
                kSmokeSlots,
                output_bias.size(),
                "FFN output");
            const auto ffn_pre_activation_actual =
                client.Decrypt(ffn.pre_activation);
            const auto ffn_polynomial_input_actual =
                client.Decrypt(ffn.gelu.polynomial_input);
            const auto ffn_polynomial_output_actual =
                client.Decrypt(ffn.gelu.polynomial_output);
            const auto ffn_activation_actual =
                client.Decrypt(ffn.gelu.output);
            const auto ffn_output_actual = client.Decrypt(ffn.output);
            std::cout
                << std::setprecision(17)
                << "{\"diagnostic\":\"m3_gelu_checkpoints\","
                << "\"bootstrapped\":"
                << (ffn.gelu.bootstrapped ? "true" : "false") << ","
                << "\"pre_activation_inactive_max_abs\":"
                << MaximumInactive(ffn_pre_activation_actual, active_mask) << ","
                << "\"polynomial_input_inactive_max_abs\":"
                << MaximumInactive(ffn_polynomial_input_actual, active_mask) << ","
                << "\"polynomial_output_inactive_max_abs\":"
                << MaximumInactive(ffn_polynomial_output_actual, active_mask) << ","
                << "\"activation_inactive_max_abs\":"
                << MaximumInactive(ffn_activation_actual, active_mask) << ","
                << "\"ffn_output_inactive_max_abs\":"
                << MaximumInactive(ffn_output_actual, active_mask) << ","
                << "\"pre_activation_level\":"
                << ffn.pre_activation.packing.level << ","
                << "\"polynomial_input_level\":"
                << ffn.gelu.polynomial_input.packing.level << ","
                << "\"polynomial_input_noise_scale_degree\":"
                << ffn.gelu.polynomial_input.packing.noise_scale_degree << ","
                << "\"polynomial_output_level\":"
                << ffn.gelu.polynomial_output.packing.level << ","
                << "\"polynomial_output_noise_scale_degree\":"
                << ffn.gelu.polynomial_output.packing.noise_scale_degree << ","
                << "\"activation_level\":"
                << ffn.gelu.output.packing.level << ","
                << "\"activation_noise_scale_degree\":"
                << ffn.gelu.output.packing.noise_scale_degree
                << "}\n";
            inactive_maximum = std::max(
                inactive_maximum,
                RequireInactiveBound(
                    ffn_pre_activation_actual,
                    active_mask,
                    "FFN pre-activation"));
            inactive_maximum = std::max(
                inactive_maximum,
                RequireInactiveBound(
                    ffn_polynomial_input_actual,
                    active_mask,
                    "GELU polynomial input"));
            inactive_maximum = std::max(
                inactive_maximum,
                RequireInactiveBound(
                    ffn_polynomial_output_actual,
                    active_mask,
                    "GELU polynomial output"));
            gelu_quality = MeasureQuality(
                ffn_activation_actual,
                ffn_activation_expected,
                active_mask);
            ffn_quality = MeasureQuality(
                ffn_output_actual,
                ffn_output_expected,
                active_mask);
            RequireQuality(gelu_quality, "GELU");
            RequireQuality(ffn_quality, "reduced FFN");
            inactive_maximum = std::max(
                inactive_maximum,
                RequireInactiveBound(
                    ffn_activation_actual,
                    active_mask,
                    "GELU activation"));
            inactive_maximum = std::max(
                inactive_maximum,
                RequireInactiveBound(
                    ffn_output_actual,
                    active_mask,
                    "FFN output"));

            const std::vector<std::vector<double>> logits{
                PadSlots({-2.0, 0.5, 1.2, -0.7}, kSmokeSlots),
                PadSlots({0.0, -0.2, -1.5, 0.4}, kSmokeSlots),
            };
            constexpr std::size_t kSoftmaxLayer = 1;
            constexpr std::size_t kSoftmaxHead = 2;
            const double public_softmax_shift =
                contracts.softmax_shifts.At(kSoftmaxLayer, kSoftmaxHead);
            if (std::abs(public_softmax_shift - 2.9766493219191048) > 1e-15) {
                throw std::runtime_error(
                    "reduced Softmax fixture shift is not registry-bound");
            }
            std::vector<std::vector<double>> shifted_logits(
                logits.size(),
                std::vector<double>(kSmokeSlots));
            for (std::size_t slot = 0; slot < kSmokeSlots; ++slot) {
                if (active_mask[slot] < 0.5) {
                    continue;
                }
                for (std::size_t logit = 0; logit < logits.size(); ++logit) {
                    shifted_logits[logit][slot] =
                        logits[logit][slot] - public_softmax_shift;
                }
            }

            std::vector<std::vector<double>> exponential_expected(
                logits.size(),
                std::vector<double>(kSmokeSlots));
            std::vector<double> denominator_expected(kSmokeSlots, 1.0);
            for (std::size_t slot = 0; slot < kSmokeSlots; ++slot) {
                if (active_mask[slot] < 0.5) {
                    continue;
                }
                denominator_expected[slot] = 0.0;
                for (std::size_t logit = 0; logit < logits.size(); ++logit) {
                    exponential_expected[logit][slot] =
                        EvaluateChebyshev(
                            exponential_contract,
                            shifted_logits[logit][slot]);
                    denominator_expected[slot] +=
                        exponential_expected[logit][slot];
                }
            }
            std::vector<std::vector<double>> softmax_expected(
                logits.size(),
                std::vector<double>(kSmokeSlots));
            for (std::size_t slot = 0; slot < kSmokeSlots; ++slot) {
                if (active_mask[slot] < 0.5) {
                    continue;
                }
                const double inverse = EvaluateChebyshev(
                    reciprocal_contract,
                    denominator_expected[slot]);
                for (std::size_t logit = 0; logit < logits.size(); ++logit) {
                    softmax_expected[logit][slot] =
                        exponential_expected[logit][slot] * inverse;
                }
            }

            auto logits_packing = packing;
            logits_packing.logical_shape = {kSmokeSlots, logits.size()};
            const auto encrypted_logits = client.Encrypt(logits, logits_packing);
            RequireLogicalShape(
                encrypted_logits,
                kSmokeSlots,
                logits.size(),
                "encrypted Softmax logits");
            RequireRangeWithinContract(
                ActiveRange(shifted_logits, active_mask),
                exponential_contract,
                "Softmax shifted logits");
            RequireRangeWithinContract(
                ActiveRange({denominator_expected}, active_mask),
                reciprocal_contract,
                "Softmax denominator");
            const auto softmax = nonlinear.SoftmaxWithCheckpoints(
                encrypted_logits,
                kSoftmaxLayer,
                kSoftmaxHead,
                active_mask);
            RequireAccuratePackingLevel(
                softmax.denominator_before_bootstrap,
                "softmax denominator before bootstrap");
            RequireAccuratePackingLevel(
                softmax.denominator_after_bootstrap,
                "softmax denominator after bootstrap");
            RequireAccuratePackingLevel(
                softmax.output,
                "softmax output");
            RequireLogicalShape(
                softmax.exponentials,
                kSmokeSlots,
                logits.size(),
                "Softmax exponentials");
            RequireLogicalShape(
                softmax.denominator_before_bootstrap,
                kSmokeSlots,
                1,
                "Softmax denominator before bootstrap");
            RequireLogicalShape(
                softmax.denominator_after_bootstrap,
                kSmokeSlots,
                1,
                "Softmax denominator after bootstrap");
            RequireLogicalShape(
                softmax.reciprocal,
                kSmokeSlots,
                1,
                "Softmax reciprocal");
            RequireLogicalShape(
                softmax.output,
                kSmokeSlots,
                logits.size(),
                "Softmax output");
            if (server.RemainingLevels(
                    softmax.denominator_after_bootstrap) <
                reciprocal_contract.required_depth + 2) {
                throw std::runtime_error(
                    "post-bootstrap denominator lacks evaluator depth plus reserve");
            }

            const auto decrypt_softmax_checkpoint = [&client](
                                                        const moai::openfhe::CipherTensor&
                                                            tensor,
                                                        const std::string& label) {
                try {
                    return client.Decrypt(tensor);
                }
                catch (const std::exception& exception) {
                    throw std::runtime_error(
                        label + " decryption failed: " + exception.what());
                }
            };
            const auto exponential_actual = decrypt_softmax_checkpoint(
                softmax.exponentials,
                "Softmax exponentials");
            const auto denominator_before = decrypt_softmax_checkpoint(
                softmax.denominator_before_bootstrap,
                "Softmax denominator before bootstrap");
            static_cast<void>(denominator_before);
            const auto denominator_after = decrypt_softmax_checkpoint(
                softmax.denominator_after_bootstrap,
                "Softmax denominator after bootstrap");
            const std::vector<std::vector<double>> denominator_oracle{
                denominator_expected};
            denominator_quality = MeasureQuality(
                denominator_after,
                denominator_oracle,
                active_mask);
            RequireQuality(
                denominator_quality,
                "softmax denominator after bootstrap");
            for (std::size_t slot = 0; slot < kSmokeSlots; ++slot) {
                if (active_mask[slot] > 0.5) {
                    denominator_post_max_abs = std::max(
                        denominator_post_max_abs,
                        std::abs(
                            denominator_after[0][slot] -
                            denominator_expected[slot]));
                    if (denominator_after[0][slot] <
                            reciprocal_contract.interval.minimum ||
                        denominator_after[0][slot] >
                            reciprocal_contract.interval.maximum) {
                        throw std::runtime_error(
                            "post-bootstrap denominator left reciprocal interval");
                    }
                }
            }
            if (denominator_post_max_abs > 1e-6) {
                throw std::runtime_error(
                    "post-bootstrap denominator max-abs exceeds 1e-6");
            }

            const auto reciprocal_actual = decrypt_softmax_checkpoint(
                softmax.reciprocal,
                "Softmax reciprocal");
            static_cast<void>(reciprocal_actual);
            const auto softmax_actual = decrypt_softmax_checkpoint(
                softmax.output,
                "Softmax output");
            softmax_quality = MeasureQuality(
                softmax_actual,
                softmax_expected,
                active_mask);
            RequireQuality(softmax_quality, "softmax");
            inactive_maximum = std::max(
                inactive_maximum,
                RequireInactiveBound(
                    exponential_actual,
                    active_mask,
                    "Softmax exponentials"));
            inactive_maximum = std::max(
                inactive_maximum,
                RequireInactiveBound(
                    softmax_actual,
                    active_mask,
                    "Softmax output"));

            const std::vector<std::vector<double>> layernorm_input{
                PadSlots({-1.5, -0.5, 0.5, 1.5}, kSmokeSlots),
                PadSlots({-0.8, 0.1, 1.0, 1.8}, kSmokeSlots),
                PadSlots({2.0, -1.0, 0.0, 1.0}, kSmokeSlots),
                PadSlots({0.5, 1.5, -1.5, -0.5}, kSmokeSlots),
            };
            const std::vector<double> gamma{1.1, 0.9, 1.2, 0.8};
            const std::vector<double> beta{0.1, -0.05, 0.02, -0.08};
            constexpr double kEpsilon =
                moai::openfhe::kPaperCompatLayerNormEpsilon;
            constexpr double kVarianceScale =
                moai::openfhe::kPaperCompatLayerNorm1VarianceScale;
            std::vector<std::vector<double>> layernorm_expected(
                layernorm_input.size(),
                std::vector<double>(kSmokeSlots));
            std::vector<double> scaled_variances(kSmokeSlots);
            for (std::size_t slot = 0; slot < kSmokeSlots; ++slot) {
                if (active_mask[slot] < 0.5) {
                    continue;
                }
                double mean = 0.0;
                for (const auto& feature : layernorm_input) {
                    mean += feature[slot];
                }
                mean /= static_cast<double>(layernorm_input.size());
                double variance = 0.0;
                for (const auto& feature : layernorm_input) {
                    const double centered = feature[slot] - mean;
                    variance += centered * centered;
                }
                variance /= static_cast<double>(layernorm_input.size());
                scaled_variances[slot] =
                    (variance + kEpsilon) * kVarianceScale;
                const double inverse_scaled = EvaluateChebyshev(
                    inverse_sqrt_contract,
                    scaled_variances[slot]);
                for (std::size_t feature = 0;
                     feature < layernorm_input.size();
                     ++feature) {
                    layernorm_expected[feature][slot] =
                        (layernorm_input[feature][slot] - mean) *
                        std::sqrt(kVarianceScale) *
                        inverse_scaled *
                        gamma[feature] +
                        beta[feature];
                }
            }
            const auto scaled_variance_range = ActiveRange(
                {scaled_variances},
                active_mask);
            RequireRangeWithinContract(
                scaled_variance_range,
                inverse_sqrt_contract,
                "LayerNorm scaled variance");

            auto layernorm_packing = packing;
            layernorm_packing.logical_shape = {
                kSmokeSlots,
                layernorm_input.size()};
            const auto encrypted_layernorm =
                client.Encrypt(layernorm_input, layernorm_packing);
            RequireLogicalShape(
                encrypted_layernorm,
                kSmokeSlots,
                layernorm_input.size(),
                "encrypted LayerNorm input");
            bool invalid_layernorm_site_rejected = false;
            try {
                static_cast<void>(nonlinear.LayerNorm(
                    encrypted_layernorm,
                    gamma,
                    beta,
                    active_mask,
                    static_cast<moai::openfhe::PaperCompatLayerNormSite>(99)));
            }
            catch (const std::invalid_argument&) {
                invalid_layernorm_site_rejected = true;
            }
            if (!invalid_layernorm_site_rejected) {
                throw std::runtime_error(
                    "invalid LayerNorm site did not fail the paper_compat contract");
            }
            const auto layernorm = nonlinear.LayerNorm(
                encrypted_layernorm,
                gamma,
                beta,
                active_mask,
                moai::openfhe::PaperCompatLayerNormSite::kAttentionResidual);
            RequireAccuratePackingLevel(layernorm, "LayerNorm output");
            RequireLogicalShape(
                layernorm,
                kSmokeSlots,
                layernorm_input.size(),
                "LayerNorm output");
            const auto layernorm_actual = client.Decrypt(layernorm);
            layernorm_quality = MeasureQuality(
                layernorm_actual,
                layernorm_expected,
                active_mask);
            RequireQuality(layernorm_quality, "LayerNorm");
            inactive_maximum = std::max(
                inactive_maximum,
                RequireInactiveBound(
                    layernorm_actual,
                    active_mask,
                    "LayerNorm output"));

            if (inactive_maximum > 1e-6) {
                throw std::runtime_error(
                    "nonlinear inactive-slot max-abs exceeds 1e-6");
            }

            operation_counts = server.metrics();
            if (operation_counts.rotations != 0 ||
                operation_counts.ct_pt_multiplications != 19 ||
                operation_counts.ct_ct_multiplications != 10 ||
                operation_counts.rescale_operations != 28 ||
                operation_counts.bootstraps != 3 ||
                operation_counts.bootstrap_iterations != 6 ||
                operation_counts.chebyshev_evaluations != 5 ||
                operation_counts.estimated_polynomial_multiplications != 111 ||
                operation_counts.max_polynomial_depth != 10 ||
                operation_counts.max_observed_level >
                    profile.multiplicative_depth) {
                throw std::runtime_error(
                    "nonlinear operation/depth counters violate the M3 contract");
            }
        }

        std::cout
            << "{\"test\":\"openfhe_nonlinear_smoke\","
            << "\"profile\":\"paper_compat\","
            << "\"security_claim\":\"none\","
            << "\"parameter_sha256\":\"" << profile.parameter_sha256
            << "\","
            << "\"effective_profile_schema_version\":"
            << profile.effective_profile_schema_version << ","
            << "\"ring_dimension\":" << profile.ring_dimension << ","
            << "\"slot_count\":" << profile.slot_count << ","
            << "\"multiplicative_depth\":"
            << profile.multiplicative_depth << ","
            << "\"levels_available_after_bootstrap\":"
            << profile.levels_available_after_bootstrap << ","
            << "\"encoded_slots\":" << kSmokeSlots << ","
            << "\"packing_active_slots\":" << kSmokeSlots << ","
            << "\"logical_active_slots\":4,"
            << "\"bootstrap_iterations_per_call\":2,"
            << "\"bootstrap_precision\":10,"
            << "\"gelu_degree\":319,"
            << "\"gelu_interval_min\":"
            << gelu_contract.interval.minimum << ","
            << "\"gelu_interval_max\":"
            << gelu_contract.interval.maximum << ","
            << "\"gelu_required_depth\":"
            << gelu_contract.required_depth << ","
            << "\"gelu_estimated_multiplications\":"
            << gelu_contract.estimated_multiplications << ","
            << "\"softmax_exp_degree\":27,"
            << "\"softmax_exp_interval_min\":"
            << exponential_contract.interval.minimum << ","
            << "\"softmax_exp_interval_max\":"
            << exponential_contract.interval.maximum << ","
            << "\"softmax_exp_required_depth\":"
            << exponential_contract.required_depth << ","
            << "\"softmax_exp_estimated_multiplications\":"
            << exponential_contract.estimated_multiplications << ","
            << "\"softmax_reciprocal_degree\":383,"
            << "\"softmax_reciprocal_interval_min\":"
            << reciprocal_contract.interval.minimum << ","
            << "\"softmax_reciprocal_interval_max\":"
            << reciprocal_contract.interval.maximum << ","
            << "\"softmax_reciprocal_required_depth\":"
            << reciprocal_contract.required_depth << ","
            << "\"softmax_reciprocal_estimated_multiplications\":"
            << reciprocal_contract.estimated_multiplications << ","
            << "\"softmax_shift_layer\":1,"
            << "\"softmax_shift_head\":2,"
            << "\"softmax_shift_sha256\":\""
            << moai::openfhe::kPaperCompatSoftmaxShiftSha256 << "\","
            << "\"layernorm_invsqrt_degree\":159,"
            << "\"layernorm_invsqrt_interval_min\":"
            << inverse_sqrt_contract.interval.minimum << ","
            << "\"layernorm_invsqrt_interval_max\":"
            << inverse_sqrt_contract.interval.maximum << ","
            << "\"layernorm_invsqrt_required_depth\":"
            << inverse_sqrt_contract.required_depth << ","
            << "\"layernorm_invsqrt_estimated_multiplications\":"
            << inverse_sqrt_contract.estimated_multiplications << ","
            << "\"gelu_rel_l2\":" << gelu_quality.relative_l2 << ","
            << "\"gelu_cosine\":" << gelu_quality.cosine << ","
            << "\"ffn_rel_l2\":" << ffn_quality.relative_l2 << ","
            << "\"ffn_cosine\":" << ffn_quality.cosine << ","
            << "\"softmax_rel_l2\":" << softmax_quality.relative_l2 << ","
            << "\"softmax_cosine\":" << softmax_quality.cosine << ","
            << "\"denominator_rel_l2\":"
            << denominator_quality.relative_l2 << ","
            << "\"denominator_post_max_abs\":"
            << denominator_post_max_abs << ","
            << "\"layernorm_rel_l2\":"
            << layernorm_quality.relative_l2 << ","
            << "\"layernorm_cosine\":"
            << layernorm_quality.cosine << ","
            << "\"inactive_max_abs\":" << inactive_maximum << ","
            << "\"rotations\":" << operation_counts.rotations << ","
            << "\"ct_pt_multiplications\":"
            << operation_counts.ct_pt_multiplications << ","
            << "\"ct_ct_multiplications\":"
            << operation_counts.ct_ct_multiplications << ","
            << "\"rescale_operations\":"
            << operation_counts.rescale_operations << ","
            << "\"bootstraps\":" << operation_counts.bootstraps << ","
            << "\"bootstrap_iterations\":"
            << operation_counts.bootstrap_iterations << ","
            << "\"chebyshev_evaluations\":"
            << operation_counts.chebyshev_evaluations << ","
            << "\"estimated_polynomial_multiplications\":"
            << operation_counts.estimated_polynomial_multiplications << ","
            << "\"max_polynomial_depth\":"
            << operation_counts.max_polynomial_depth << ","
            << "\"max_observed_level\":"
            << operation_counts.max_observed_level
            << "}\n";
        return 0;
    }
    catch (const std::exception& exception) {
        std::cerr << "openfhe_nonlinear_smoke failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
