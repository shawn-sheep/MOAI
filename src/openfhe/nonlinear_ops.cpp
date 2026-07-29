#include "moai/openfhe/nonlinear_ops.hpp"
#include "moai/openfhe/approximation_registry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace moai::openfhe {
namespace {

std::vector<double> MaskedConstant(
    const std::vector<double>& active_mask,
    double value) {
    std::vector<double> result(active_mask.size());
    for (std::size_t slot = 0; slot < active_mask.size(); ++slot) {
        result[slot] = active_mask[slot] * value;
    }
    return result;
}

void AppendSingle(CipherTensor& output, CipherTensor&& single) {
    if (single.size() != 1) {
        throw std::logic_error("nonlinear affine path expected one ciphertext");
    }
    if (output.empty()) {
        output.packing = single.packing;
    }
    output.ciphertexts.push_back(std::move(single.ciphertexts.front()));
}

}  // namespace

NonlinearOps::NonlinearOps(ServerRuntime& server)
    : server_(server),
      contracts_(MakePaperCompatNonlinearContracts()) {}

void NonlinearOps::ValidateMask(
    const CipherTensor& input,
    const std::vector<double>& active_mask) const {
    if (input.empty() || input.packing.active_slots == 0 ||
        active_mask.size() != input.packing.active_slots) {
        throw std::invalid_argument(
            "active mask must match a non-empty CipherTensor packing");
    }
    bool any_active = false;
    for (const double value : active_mask) {
        if (!std::isfinite(value) ||
            (std::abs(value) > 1e-12 && std::abs(value - 1.0) > 1e-12)) {
            throw std::invalid_argument("active mask must contain only zero or one");
        }
        any_active = any_active || value > 0.5;
    }
    if (!any_active) {
        throw std::invalid_argument("active mask must contain at least one active slot");
    }
}

CipherTensor NonlinearOps::ApplyMask(
    const CipherTensor& input,
    const std::vector<double>& active_mask) {
    ValidateMask(input, active_mask);
    server_.RequireUsableLevels(input, 1, "active-mask multiplication");
    const auto mask = server_.EncodeModelVector(active_mask, input.packing);
    return server_.Rescale(server_.MultiplyPlain(input, {mask}));
}

CipherTensor NonlinearOps::EvaluatePolynomial(
    const CipherTensor& input,
    const ApproximationContract& contract,
    uint32_t reserve_levels) {
    if (reserve_levels >
        std::numeric_limits<uint32_t>::max() - contract.required_depth) {
        throw std::overflow_error("polynomial level reserve overflows uint32");
    }
    server_.RequireUsableLevels(
        input,
        contract.required_depth + reserve_levels,
        contract.contract_id + " including downstream reserve");
    return server_.EvaluateChebyshev(input, contract);
}

CipherTensor NonlinearOps::Gelu(
    const CipherTensor& input,
    const std::vector<double>& active_mask) {
    return GeluWithCheckpointsForDownstream(
        input,
        active_mask,
        0)
        .output;
}

GeluResult NonlinearOps::GeluWithCheckpoints(
    const CipherTensor& input,
    const std::vector<double>& active_mask) {
    return GeluWithCheckpointsForDownstream(input, active_mask, 0);
}

GeluResult NonlinearOps::GeluWithCheckpointsForDownstream(
    const CipherTensor& input,
    const std::vector<double>& active_mask,
    uint32_t downstream_reserve_levels) {
    ValidateMask(input, active_mask);
    const uint64_t total_required_levels =
        static_cast<uint64_t>(contracts_.gelu.required_depth) + 1 +
        downstream_reserve_levels;
    if (total_required_levels > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("GELU level reserve overflows uint32");
    }
    GeluResult result;
    result.polynomial_input = input;
    if (server_.RemainingLevels(result.polynomial_input) < total_required_levels) {
        result.polynomial_input = server_.Bootstrap(result.polynomial_input);
        result.bootstrapped = true;
    }
    result.polynomial_output = EvaluatePolynomial(
        result.polynomial_input,
        contracts_.gelu,
        1 + downstream_reserve_levels);
    result.output = ApplyMask(result.polynomial_output, active_mask);
    return result;
}

SoftmaxResult NonlinearOps::SoftmaxWithCheckpoints(
    const CipherTensor& logits,
    std::size_t layer,
    std::size_t head,
    const std::vector<double>& active_mask) {
    ValidateMask(logits, active_mask);
    const double public_shift =
        PaperCompatSoftmaxPublicShift(layer, head);
    if (!std::isfinite(public_shift)) {
        throw std::invalid_argument(
            "softmax offline-calibrated public scalar shift must be finite");
    }

    SoftmaxResult result;
    const auto shift_slots = MaskedConstant(active_mask, public_shift);
    auto shifted = server_.SubtractPlain(logits, {shift_slots});
    result.exponentials = EvaluatePolynomial(
        shifted,
        contracts_.softmax_exponential,
        1);
    result.exponentials = ApplyMask(result.exponentials, active_mask);

    result.denominator_before_bootstrap = server_.Sum(result.exponentials);
    std::vector<double> inactive_identity(active_mask.size());
    for (std::size_t slot = 0; slot < active_mask.size(); ++slot) {
        inactive_identity[slot] = 1.0 - active_mask[slot];
    }
    result.denominator_before_bootstrap = server_.AddPlain(
        result.denominator_before_bootstrap,
        {inactive_identity});
    result.denominator_after_bootstrap =
        server_.Bootstrap(result.denominator_before_bootstrap);

    result.reciprocal = EvaluatePolynomial(
        result.denominator_after_bootstrap,
        contracts_.softmax_reciprocal,
        2);
    auto probabilities = server_.Rescale(
        server_.Multiply(result.exponentials, result.reciprocal));
    result.output = ApplyMask(probabilities, active_mask);
    return result;
}

CipherTensor NonlinearOps::Softmax(
    const CipherTensor& logits,
    std::size_t layer,
    std::size_t head,
    const std::vector<double>& active_mask) {
    return SoftmaxWithCheckpoints(
        logits,
        layer,
        head,
        active_mask)
        .output;
}

SoftmaxResult NonlinearOps::MultiHeadSoftmaxWithCheckpoints(
    const CipherTensor& logits_by_key,
    std::size_t layer,
    std::size_t head_width) {
    if (logits_by_key.empty() ||
        logits_by_key.packing.layout != PackingLayout::kContiguous ||
        logits_by_key.packing.logical_shape.size() != 2 ||
        logits_by_key.packing.logical_shape[0] !=
            logits_by_key.packing.active_slots ||
        logits_by_key.packing.logical_shape[1] != logits_by_key.size() ||
        head_width == 0 ||
        head_width > logits_by_key.packing.active_slots ||
        kPaperCompatAttentionHeads >
            logits_by_key.packing.active_slots / head_width) {
        throw std::invalid_argument(
            "multi-head Softmax requires feature-packed key logits");
    }
    if (layer >= kPaperCompatEncoderLayers) {
        throw std::out_of_range(
            "multi-head Softmax layer index is out of range");
    }

    const std::size_t active_features =
        kPaperCompatAttentionHeads * head_width;
    std::vector<double> active_mask(
        logits_by_key.packing.active_slots,
        0.0);
    std::vector<double> shift_slots(
        logits_by_key.packing.active_slots,
        0.0);
    for (std::size_t head = 0;
         head < kPaperCompatAttentionHeads;
         ++head) {
        const double shift = PaperCompatSoftmaxPublicShift(layer, head);
        if (!std::isfinite(shift)) {
            throw std::logic_error(
                "multi-head Softmax public-shift registry is non-finite");
        }
        const std::size_t begin = head * head_width;
        const std::size_t end = begin + head_width;
        for (std::size_t slot = begin; slot < end; ++slot) {
            active_mask[slot] = 1.0;
            shift_slots[slot] = shift;
        }
    }
    if (active_features == 0 ||
        active_features > logits_by_key.packing.active_slots) {
        throw std::logic_error(
            "multi-head Softmax active feature span is invalid");
    }

    SoftmaxResult result;
    auto shifted = server_.SubtractPlain(logits_by_key, {shift_slots});
    if (contracts_.softmax_exponential.required_depth !=
            kPaperCompatFeaturePackedSoftmaxPreBootstrapDepth ||
        contracts_.softmax_exponential.coefficient_sha256 !=
            kPaperCompatFeaturePackedSoftmaxExpCoefficientSha256) {
        throw std::logic_error(
            "feature-packed Softmax exponential contract drifted");
    }
    const double exponential_at_zero = EvaluateChebyshevContractAt(
        contracts_.softmax_exponential,
        0.0);
    if (std::abs(
            exponential_at_zero -
            kPaperCompatFeaturePackedSoftmaxExpAtZero) > 2e-14) {
        throw std::logic_error(
            "feature-packed Softmax P_exp(0) contract drifted");
    }
    result.exponentials = EvaluatePolynomial(
        shifted,
        contracts_.softmax_exponential,
        1);
    // Inactive score slots are publicly fixed to zero. Multiplying P_exp by a
    // CKKS zero mask introduced a measured full-width cross-slot residual.
    // Subtracting the frozen public P_exp(0) only from inactive slots is
    // algebraically identical on the declared layout, keeps active slots
    // unchanged, and consumes no multiplication level.
    std::vector<double> inactive_exponential_constant(active_mask.size(), 0.0);
    for (std::size_t slot = 0; slot < active_mask.size(); ++slot) {
        if (active_mask[slot] < 0.5) {
            inactive_exponential_constant[slot] = exponential_at_zero;
        }
    }
    result.exponentials = server_.SubtractPlain(
        result.exponentials,
        {inactive_exponential_constant});

    result.denominator_before_bootstrap = server_.Sum(result.exponentials);
    std::vector<double> inactive_identity(active_mask.size());
    for (std::size_t slot = 0; slot < active_mask.size(); ++slot) {
        inactive_identity[slot] = 1.0 - active_mask[slot];
    }
    result.denominator_before_bootstrap = server_.AddPlain(
        result.denominator_before_bootstrap,
        {inactive_identity});
    result.denominator_after_bootstrap =
        server_.Bootstrap(result.denominator_before_bootstrap);

    if (contracts_.softmax_reciprocal.required_depth + 1 !=
        kPaperCompatFeaturePackedSoftmaxPostBootstrapDepth) {
        throw std::logic_error(
            "feature-packed Softmax post-bootstrap depth drifted");
    }
    result.reciprocal = EvaluatePolynomial(
        result.denominator_after_bootstrap,
        contracts_.softmax_reciprocal,
        1);
    result.output = server_.Rescale(
        server_.Multiply(result.exponentials, result.reciprocal));
    return result;
}

CipherTensor NonlinearOps::MultiHeadSoftmax(
    const CipherTensor& logits_by_key,
    std::size_t layer,
    std::size_t head_width) {
    return MultiHeadSoftmaxWithCheckpoints(
        logits_by_key,
        layer,
        head_width)
        .output;
}

CipherTensor NonlinearOps::LayerNorm(
    const CipherTensor& input,
    const std::vector<double>& gamma,
    const std::vector<double>& beta,
    const std::vector<double>& active_mask,
    PaperCompatLayerNormSite site) {
    ValidateMask(input, active_mask);
    if (gamma.size() != input.size() || beta.size() != input.size() ||
        input.size() == 0) {
        throw std::invalid_argument(
            "LayerNorm gamma/beta dimensions must match feature ciphertexts");
    }
    if (!std::all_of(gamma.begin(), gamma.end(), [](double value) {
            return std::isfinite(value);
        }) ||
        !std::all_of(beta.begin(), beta.end(), [](double value) {
            return std::isfinite(value);
        })) {
        throw std::invalid_argument("LayerNorm public parameters are invalid");
    }
    double variance_scale = 0.0;
    switch (site) {
        case PaperCompatLayerNormSite::kAttentionResidual:
            variance_scale = kPaperCompatLayerNorm1VarianceScale;
            break;
        case PaperCompatLayerNormSite::kFeedForwardResidual:
            variance_scale = kPaperCompatLayerNorm2VarianceScale;
            break;
        default:
            throw std::invalid_argument(
                "LayerNorm site violates the frozen paper_compat contract");
    }

    server_.RequireUsableLevels(
        input,
        4,
        "LayerNorm pre-bootstrap path");
    auto masked_input = ApplyMask(input, active_mask);
    auto mean = server_.Sum(masked_input);
    const auto inverse_feature_count = MaskedConstant(
        active_mask,
        1.0 / static_cast<double>(input.size()));
    mean = server_.Rescale(server_.MultiplyPlain(
        mean,
        {server_.EncodeModelVector(inverse_feature_count, mean.packing)}));

    auto centered = server_.Subtract(masked_input, mean);
    auto squared = server_.Rescale(server_.Multiply(centered, centered));
    auto variance = server_.Sum(squared);
    const auto variance_multiplier = MaskedConstant(
        active_mask,
        variance_scale / static_cast<double>(input.size()));
    variance = server_.Rescale(server_.MultiplyPlain(
        variance,
        {server_.EncodeModelVector(
            variance_multiplier,
            variance.packing)}));

    std::vector<double> safe_variance_add(active_mask.size());
    for (std::size_t slot = 0; slot < active_mask.size(); ++slot) {
        safe_variance_add[slot] = active_mask[slot] > 0.5
            ? kPaperCompatLayerNormEpsilon * variance_scale
            : 1.0;
    }
    variance = server_.AddPlain(variance, {safe_variance_add});
    variance = server_.Bootstrap(variance);

    auto inverse_scaled = EvaluatePolynomial(
        variance,
        contracts_.layernorm_inverse_sqrt,
        2);
    auto normalized =
        server_.Rescale(server_.Multiply(centered, inverse_scaled));

    const double scale_compensation = std::sqrt(variance_scale);
    std::vector<lbcrypto::Plaintext> gamma_plaintexts;
    gamma_plaintexts.reserve(gamma.size());
    for (const double value : gamma) {
        gamma_plaintexts.push_back(server_.EncodeModelVector(
            MaskedConstant(active_mask, value * scale_compensation),
            normalized.packing));
    }
    auto output =
        server_.Rescale(server_.MultiplyPlain(normalized, gamma_plaintexts));

    std::vector<std::vector<double>> beta_vectors;
    beta_vectors.reserve(beta.size());
    for (const double value : beta) {
        beta_vectors.push_back(MaskedConstant(active_mask, value));
    }
    return server_.AddPlain(output, beta_vectors);
}

CipherTensor NonlinearOps::FeaturePackedLayerNorm(
    const CipherTensor& input,
    const std::vector<double>& gamma,
    const std::vector<double>& beta,
    PaperCompatLayerNormSite site) {
    if (input.empty() || input.packing.layout != PackingLayout::kContiguous ||
        input.packing.logical_shape.size() != 2 ||
        input.packing.logical_shape[0] != input.packing.active_slots ||
        input.packing.logical_shape[1] != input.size() ||
        gamma.empty() || gamma.size() != beta.size() ||
        gamma.size() > input.packing.active_slots) {
        throw std::invalid_argument(
            "feature-packed LayerNorm dimensions do not match");
    }
    if (!std::all_of(gamma.begin(), gamma.end(), [](double value) {
            return std::isfinite(value);
        }) ||
        !std::all_of(beta.begin(), beta.end(), [](double value) {
            return std::isfinite(value);
        })) {
        throw std::invalid_argument(
            "feature-packed LayerNorm parameters are non-finite");
    }

    double variance_scale = 0.0;
    switch (site) {
        case PaperCompatLayerNormSite::kAttentionResidual:
            variance_scale = kPaperCompatLayerNorm1VarianceScale;
            break;
        case PaperCompatLayerNormSite::kFeedForwardResidual:
            variance_scale = kPaperCompatLayerNorm2VarianceScale;
            break;
        default:
            throw std::invalid_argument(
                "feature-packed LayerNorm site violates paper_compat");
    }

    std::vector<double> active_mask(input.packing.active_slots, 0.0);
    std::fill_n(active_mask.begin(), gamma.size(), 1.0);
    server_.RequireUsableLevels(
        input,
        contracts_.depth_budget.layernorm_pre_bootstrap,
        "feature-packed LayerNorm pre-bootstrap path");
    const auto masked_input = ApplyMask(input, active_mask);
    auto mean = server_.SumSlots(masked_input);
    const auto inverse_feature_count = MaskedConstant(
        active_mask,
        1.0 / static_cast<double>(gamma.size()));
    mean = server_.Rescale(server_.MultiplyPlain(
        mean,
        {server_.EncodeModelVector(inverse_feature_count, mean.packing)}));

    const auto centered = server_.Subtract(masked_input, mean);
    const auto squared = server_.Rescale(server_.Multiply(centered, centered));
    auto variance = server_.SumSlots(squared);
    const auto variance_multiplier = MaskedConstant(
        active_mask,
        variance_scale / static_cast<double>(gamma.size()));
    variance = server_.Rescale(server_.MultiplyPlain(
        variance,
        {server_.EncodeModelVector(
            variance_multiplier,
            variance.packing)}));

    std::vector<double> safe_variance_add(active_mask.size());
    for (std::size_t slot = 0; slot < active_mask.size(); ++slot) {
        safe_variance_add[slot] = active_mask[slot] > 0.5
            ? kPaperCompatLayerNormEpsilon * variance_scale
            : 1.0;
    }
    variance = server_.AddPlain(variance, {safe_variance_add});
    variance = server_.Bootstrap(variance);

    const auto inverse_scaled = EvaluatePolynomial(
        variance,
        contracts_.layernorm_inverse_sqrt,
        2);
    const auto normalized =
        server_.Rescale(server_.Multiply(centered, inverse_scaled));

    const double scale_compensation = std::sqrt(variance_scale);
    std::vector<double> gamma_slots(input.packing.active_slots, 0.0);
    std::vector<double> beta_slots(input.packing.active_slots, 0.0);
    for (std::size_t feature = 0; feature < gamma.size(); ++feature) {
        gamma_slots[feature] = gamma[feature] * scale_compensation;
        beta_slots[feature] = beta[feature];
    }
    auto output = server_.Rescale(server_.MultiplyPlain(
        normalized,
        {server_.EncodeModelVector(gamma_slots, normalized.packing)}));
    return server_.AddPlain(output, {beta_slots});
}

CipherTensor NonlinearOps::Affine(
    const CipherTensor& input,
    const DenseWeights& weights,
    const std::vector<double>& bias,
    const std::vector<double>& active_mask) {
    ValidateMask(input, active_mask);
    if (weights.size() != input.size() || weights.empty() ||
        weights.front().empty() || bias.size() != weights.front().size()) {
        throw std::invalid_argument("affine matrix dimensions do not match");
    }
    const std::size_t output_features = weights.front().size();
    for (const auto& row : weights) {
        if (row.size() != output_features ||
            !std::all_of(row.begin(), row.end(), [](double value) {
                return std::isfinite(value);
            })) {
            throw std::invalid_argument("affine weights are ragged or non-finite");
        }
    }
    if (!std::all_of(bias.begin(), bias.end(), [](double value) {
            return std::isfinite(value);
        })) {
        throw std::invalid_argument("affine bias contains a non-finite value");
    }

    CipherTensor output;
    for (std::size_t output_feature = 0;
         output_feature < output_features;
         ++output_feature) {
        std::vector<lbcrypto::Plaintext> encoded_weights;
        encoded_weights.reserve(input.size());
        for (std::size_t input_feature = 0;
             input_feature < input.size();
             ++input_feature) {
            encoded_weights.push_back(server_.EncodeModelVector(
                MaskedConstant(
                    active_mask,
                    weights[input_feature][output_feature]),
                input.packing));
        }
        auto products = server_.MultiplyPlain(input, encoded_weights);
        auto summed = server_.Rescale(server_.Sum(products));
        summed = server_.AddPlain(
            summed,
            {MaskedConstant(active_mask, bias[output_feature])});
        AppendSingle(output, std::move(summed));
    }
    if (input.packing.logical_shape.size() == 2) {
        output.packing.logical_shape[1] = output_features;
    }
    else {
        output.packing.logical_shape = {
            input.packing.active_slots,
            output_features};
    }
    return output;
}

FeedForwardResult NonlinearOps::FeedForward(
    const CipherTensor& input,
    const DenseWeights& input_weights,
    const std::vector<double>& input_bias,
    const DenseWeights& output_weights,
    const std::vector<double>& output_bias,
    const std::vector<double>& active_mask) {
    FeedForwardResult result;
    result.pre_activation =
        Affine(input, input_weights, input_bias, active_mask);
    result.gelu = GeluWithCheckpointsForDownstream(
        result.pre_activation,
        active_mask,
        1);
    result.output = Affine(
        result.gelu.output,
        output_weights,
        output_bias,
        active_mask);
    return result;
}

}  // namespace moai::openfhe
