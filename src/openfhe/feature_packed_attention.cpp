#include "moai/openfhe/feature_packed_attention.hpp"

#include "moai/openfhe/approximation_registry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace moai::openfhe {
namespace {

bool IsPowerOfTwo(std::size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

bool ScalesMatch(double lhs, double rhs) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs) || lhs <= 0.0 || rhs <= 0.0) {
        return false;
    }
    const double magnitude =
        std::max(1.0, std::max(std::abs(lhs), std::abs(rhs)));
    return std::abs(lhs - rhs) <= 1e-12 * magnitude;
}

std::size_t FeatureDimension(const FeaturePackedAttentionSpec& spec) {
    if (spec.head_count != 0 &&
        spec.head_dimension >
            std::numeric_limits<std::size_t>::max() / spec.head_count) {
        throw std::overflow_error("attention feature dimension overflows size_t");
    }
    return spec.head_count * spec.head_dimension;
}

void ValidateSpec(const FeaturePackedAttentionSpec& spec) {
    if (!IsPowerOfTwo(spec.block_dimension) || spec.block_dimension < 2 ||
        spec.block_dimension >
            static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument(
            "attention block_dimension must be a power of two in int32 range");
    }
    if (spec.token_count == 0 ||
        spec.token_count > kPaperCompatAttentionTokenCount) {
        throw std::invalid_argument(
            "attention token_count must be in the frozen range [1, 5]");
    }
    if (spec.head_count != kPaperCompatAttentionHeadCount) {
        throw std::invalid_argument(
            "attention head_count must equal the frozen 12-head shift registry");
    }
    if (spec.head_dimension == 0 ||
        spec.head_dimension > spec.block_dimension ||
        FeatureDimension(spec) > spec.block_dimension) {
        throw std::invalid_argument(
            "attention head spans must fit within the feature block");
    }
}

void ValidateValidKeyMask(
    const std::vector<double>& valid_key_mask,
    const FeaturePackedAttentionSpec& spec) {
    if (valid_key_mask.size() != spec.token_count) {
        throw std::invalid_argument(
            "attention valid-key mask length must equal token_count");
    }
    for (const double value : valid_key_mask) {
        if (!std::isfinite(value) || std::abs(value - 1.0) > 1e-12) {
            throw std::invalid_argument(
                "the frozen attention trace requires every declared key to be valid");
        }
    }
}

void ValidateTensor(
    const CipherTensor& tensor,
    const FeaturePackedAttentionSpec& spec,
    const char* label) {
    if (tensor.empty() || tensor.size() != spec.token_count ||
        tensor.packing.layout != PackingLayout::kContiguous ||
        tensor.packing.batch_lanes != 1 ||
        tensor.packing.active_slots != spec.block_dimension ||
        tensor.packing.encoded_slots != spec.block_dimension ||
        tensor.packing.slot_count < spec.block_dimension ||
        tensor.packing.logical_shape != std::vector<std::size_t>{
            spec.block_dimension,
            spec.token_count}) {
        throw std::invalid_argument(
            std::string(label) +
            " must contain one contiguous full-block ciphertext per token");
    }
    for (const auto& ciphertext : tensor.ciphertexts) {
        if (!ciphertext ||
            ciphertext->GetLevel() != tensor.packing.level ||
            ciphertext->GetNoiseScaleDeg() != tensor.packing.noise_scale_degree ||
            !ScalesMatch(
                ciphertext->GetScalingFactor(),
                tensor.packing.scaling_factor)) {
            throw std::invalid_argument(
                std::string(label) + " has stale or inconsistent packing metadata");
        }
    }
}

void ValidateMatchingQkv(
    const CipherTensor& q,
    const CipherTensor& k,
    const CipherTensor& v) {
    const auto matches = [&q](const CipherTensor& other) {
        return q.packing.layout == other.packing.layout &&
            q.packing.logical_shape == other.packing.logical_shape &&
            q.packing.batch_lanes == other.packing.batch_lanes &&
            q.packing.slot_count == other.packing.slot_count &&
            q.packing.active_slots == other.packing.active_slots &&
            q.packing.encoded_slots == other.packing.encoded_slots &&
            q.packing.level == other.packing.level &&
            q.packing.noise_scale_degree == other.packing.noise_scale_degree &&
            ScalesMatch(
                q.packing.scaling_factor,
                other.packing.scaling_factor);
    };
    if (!matches(k) || !matches(v)) {
        throw std::invalid_argument(
            "attention Q, K, and V must start with identical packing and scale metadata");
    }
}

CipherTensor SelectCiphertext(
    const CipherTensor& input,
    std::size_t index,
    std::size_t block_dimension) {
    if (index >= input.size()) {
        throw std::out_of_range("attention ciphertext index is out of range");
    }
    CipherTensor result;
    result.packing = input.packing;
    result.packing.logical_shape = {block_dimension, 1};
    result.ciphertexts.push_back(input.ciphertexts[index]);
    return result;
}

void AppendTensor(
    CipherTensor& output,
    const CipherTensor& input,
    std::size_t block_dimension,
    const char* label) {
    if (input.empty()) {
        throw std::logic_error(std::string(label) + " checkpoint is empty");
    }
    if (output.empty()) {
        output.packing = input.packing;
    }
    else if (
        output.packing.layout != input.packing.layout ||
        output.packing.batch_lanes != input.packing.batch_lanes ||
        output.packing.slot_count != input.packing.slot_count ||
        output.packing.active_slots != input.packing.active_slots ||
        output.packing.encoded_slots != input.packing.encoded_slots ||
        output.packing.level != input.packing.level ||
        output.packing.noise_scale_degree != input.packing.noise_scale_degree ||
        !ScalesMatch(
            output.packing.scaling_factor,
            input.packing.scaling_factor)) {
        throw std::logic_error(
            std::string(label) + " checkpoint members do not share packing metadata");
    }
    output.ciphertexts.insert(
        output.ciphertexts.end(),
        input.ciphertexts.begin(),
        input.ciphertexts.end());
    output.packing.logical_shape = {block_dimension, output.size()};
}

std::vector<double> HeadMask(
    const FeaturePackedAttentionSpec& spec,
    std::size_t head,
    double value) {
    if (head >= spec.head_count || !std::isfinite(value)) {
        throw std::invalid_argument("attention head mask request is invalid");
    }
    std::vector<double> mask(spec.block_dimension, 0.0);
    const std::size_t begin = head * spec.head_dimension;
    std::fill(
        mask.begin() + begin,
        mask.begin() + begin + spec.head_dimension,
        value);
    return mask;
}

CipherTensor MaskAndRescale(
    ServerRuntime& server,
    const CipherTensor& input,
    const std::vector<double>& mask,
    const char* operation) {
    if (server.RemainingLevels(input) < 1) {
        throw std::runtime_error(
            std::string(operation) + " requires one usable level");
    }
    const auto plaintext = server.EncodeModelVector(mask, input.packing);
    return server.Rescale(server.MultiplyPlain(input, {plaintext}));
}

CipherTensor IsolatedHeadScores(
    ServerRuntime& server,
    const CipherTensor& q,
    const CipherTensor& k,
    const FeaturePackedAttentionSpec& spec) {
    const auto product = server.Rescale(server.Multiply(q, k));
    const double score_scale =
        1.0 / std::sqrt(static_cast<double>(spec.head_dimension));
    CipherTensor scores;
    for (std::size_t head = 0; head < spec.head_count; ++head) {
        // Masking before the full-block sum is the critical isolation step:
        // no rotation can import a sentinel from an adjacent 64-slot head.
        const auto selected = MaskAndRescale(
            server,
            product,
            HeadMask(spec, head, 1.0),
            "attention head selection");
        const auto replicated = server.SumSlots(selected);
        const auto scaled_head = MaskAndRescale(
            server,
            replicated,
            HeadMask(spec, head, score_scale),
            "attention replicated-score selection");
        scores = scores.empty() ? scaled_head : server.Add(scores, scaled_head);
    }
    return scores;
}

}  // namespace

std::vector<int32_t> FeaturePackedAttentionRotationIndices(
    const FeaturePackedAttentionSpec& spec) {
    ValidateSpec(spec);
    std::set<int32_t> indices;
    for (std::size_t step = 1; step < spec.block_dimension; step *= 2) {
        indices.insert(static_cast<int32_t>(step));
    }
    return {indices.begin(), indices.end()};
}

FeaturePackedAttentionResult FeaturePackedAttention::Evaluate(
    const CipherTensor& q,
    const CipherTensor& k,
    const CipherTensor& v,
    std::size_t layer,
    const std::vector<double>& valid_key_mask,
    const FeaturePackedAttentionSpec& spec) {
    ValidateSpec(spec);
    if (layer >= kPaperCompatEncoderLayers) {
        throw std::invalid_argument(
            "attention layer index is outside the frozen 12-layer registry");
    }
    ValidateValidKeyMask(valid_key_mask, spec);
    ValidateTensor(q, spec, "attention Q");
    ValidateTensor(k, spec, "attention K");
    ValidateTensor(v, spec, "attention V");
    ValidateMatchingQkv(q, k, v);
    server_.RequireEvaluationKeys(
        FeaturePackedAttentionRotationIndices(spec),
        true);

    const auto contracts = MakePaperCompatNonlinearContracts();
    // Q*K, head selection, and replicated-score selection consume three
    // levels. P_exp consumes six; normalization and weighted V consume two
    // more. Inactive P_exp(0) cancellation is Ct-Pt addition and consumes no
    // multiplication level.
    const uint32_t minimum_qk_levels =
        3 + kPaperCompatFeaturePackedSoftmaxPreBootstrapDepth + 2;
    if (server_.RemainingLevels(q) < minimum_qk_levels ||
        server_.RemainingLevels(k) < minimum_qk_levels) {
        throw std::runtime_error(
            "attention Q/K do not have enough levels for isolated scores and softmax");
    }
    if (server_.RemainingLevels(v) < 1) {
        throw std::runtime_error(
            "attention V lacks the weighted-value multiplication level");
    }

    FeaturePackedAttentionResult result;
    for (std::size_t query = 0; query < spec.token_count; ++query) {
        const auto query_ciphertext =
            SelectCiphertext(q, query, spec.block_dimension);
        CipherTensor query_scores;
        for (std::size_t key = 0; key < spec.token_count; ++key) {
            const auto key_ciphertext =
                SelectCiphertext(k, key, spec.block_dimension);
            const auto score = IsolatedHeadScores(
                server_,
                query_ciphertext,
                key_ciphertext,
                spec);
            AppendTensor(
                query_scores,
                score,
                spec.block_dimension,
                "attention score");
        }
        AppendTensor(
            result.scaled_scores,
            query_scores,
            spec.block_dimension,
            "attention score");

        // Every head span already contains a replicated score. This one call
        // applies all 12 offline-frozen head shifts and performs exactly one
        // denominator bootstrap for the query.
        auto softmax = nonlinear_.MultiHeadSoftmaxWithCheckpoints(
            query_scores,
            layer,
            spec.head_dimension);
        AppendTensor(
            result.probabilities,
            softmax.output,
            spec.block_dimension,
            "attention probability");

        if (server_.RemainingLevels(softmax.output) < 1) {
            throw std::runtime_error(
                "attention probabilities lack the weighted-V multiplication level");
        }
        const auto weighted_per_key = server_.Rescale(
            server_.Multiply(softmax.output, v));
        const auto weighted_values = server_.Sum(weighted_per_key);
        AppendTensor(
            result.output_before_bootstrap_cleanup,
            weighted_values,
            spec.block_dimension,
            "attention output before bootstrap cleanup");
        std::vector<double> active_output_mask(spec.block_dimension, 0.0);
        std::fill_n(
            active_output_mask.begin(),
            spec.head_count * spec.head_dimension,
            1.0);
        const auto refreshed_weighted_values =
            server_.Bootstrap(weighted_values);
        const auto cleaned_weighted_values = MaskAndRescale(
            server_,
            refreshed_weighted_values,
            active_output_mask,
            "attention output cleanup");
        AppendTensor(
            result.output,
            cleaned_weighted_values,
            spec.block_dimension,
            "attention output");
    }
    return result;
}

}  // namespace moai::openfhe
