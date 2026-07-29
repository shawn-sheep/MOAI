#include "moai/openfhe/feature_packed_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
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

void ValidateSpec(const FeaturePackedAffineSpec& spec) {
    if (!IsPowerOfTwo(spec.block_dimension) || spec.block_dimension < 2) {
        throw std::invalid_argument(
            "feature-packed block_dimension must be a power of two >= 2");
    }
    if (spec.block_dimension >
        static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument(
            "feature-packed block_dimension exceeds the rotation-index range");
    }
    if (spec.input_dimension == 0 ||
        spec.input_dimension > spec.block_dimension ||
        spec.output_dimension == 0 ||
        spec.output_dimension > spec.block_dimension) {
        throw std::invalid_argument(
            "feature-packed affine dimensions must be in [1, block_dimension]");
    }
    if (!IsPowerOfTwo(spec.baby_step) ||
        spec.baby_step > spec.block_dimension ||
        spec.block_dimension % spec.baby_step != 0) {
        throw std::invalid_argument(
            "feature-packed baby_step must be a power-of-two divisor of the block");
    }
}

std::vector<uint32_t> DenseDiagonalOffsets(
    const FeaturePackedAffineSpec& spec) {
    ValidateSpec(spec);
    std::vector<bool> used(spec.block_dimension, false);
    // For a dense input-by-output rectangle, i - j covers every integer from
    // -(output_dimension - 1) through input_dimension - 1.
    for (std::size_t offset = 0;
         offset < spec.input_dimension;
         ++offset) {
        used[offset] = true;
    }
    for (std::size_t negative = 1;
         negative < spec.output_dimension;
         ++negative) {
        used[spec.block_dimension - negative] = true;
    }

    std::vector<uint32_t> offsets;
    offsets.reserve(std::min(
        spec.block_dimension,
        spec.input_dimension + spec.output_dimension - 1));
    for (std::size_t offset = 0; offset < used.size(); ++offset) {
        if (used[offset]) {
            offsets.push_back(static_cast<uint32_t>(offset));
        }
    }
    return offsets;
}

int32_t NormalizeRotation(
    std::size_t offset,
    std::size_t block_dimension) {
    const std::size_t cyclic = offset % block_dimension;
    if (cyclic == 0) {
        return 0;
    }
    if (cyclic > block_dimension / 2) {
        return static_cast<int32_t>(
            static_cast<int64_t>(cyclic) -
            static_cast<int64_t>(block_dimension));
    }
    return static_cast<int32_t>(cyclic);
}

void ValidateTensor(
    const CipherTensor& input,
    const FeaturePackedAffineSpec& spec) {
    if (input.empty()) {
        throw std::invalid_argument(
            "feature-packed affine requires a non-empty CipherTensor");
    }
    if (input.packing.layout != PackingLayout::kContiguous ||
        input.packing.batch_lanes != 1 ||
        input.packing.active_slots != spec.block_dimension ||
        input.packing.encoded_slots != spec.block_dimension ||
        input.packing.slot_count < spec.block_dimension ||
        input.packing.noise_scale_degree == 0 ||
        !std::isfinite(input.packing.scaling_factor) ||
        input.packing.scaling_factor <= 0.0 ||
        input.packing.logical_shape != std::vector<std::size_t>{
            spec.block_dimension,
            input.size()}) {
        throw std::invalid_argument(
            "feature-packed tensor must be contiguous with one token per "
            "ciphertext and a full power-of-two feature block");
    }
    for (const auto& ciphertext : input.ciphertexts) {
        if (!ciphertext ||
            ciphertext->GetLevel() != input.packing.level ||
            ciphertext->GetNoiseScaleDeg() !=
                input.packing.noise_scale_degree ||
            !ScalesMatch(
                ciphertext->GetScalingFactor(),
                input.packing.scaling_factor)) {
            throw std::invalid_argument(
                "feature-packed tensor has stale ciphertext level/scale metadata");
        }
    }
}

void ValidateWeights(
    const FeaturePackedWeights& weights,
    const FeaturePackedAffineSpec& spec) {
    if (weights.size() != spec.input_dimension) {
        throw std::invalid_argument(
            "feature-packed affine weight dimensions do not match");
    }
    for (const auto& row : weights) {
        if (row.size() != spec.output_dimension ||
            !std::all_of(row.begin(), row.end(), [](double value) {
                return std::isfinite(value);
            })) {
            throw std::invalid_argument(
                "feature-packed affine weights are ragged or non-finite");
        }
    }
}

void ValidateBiasAndMask(
    const std::vector<double>& bias,
    const std::vector<double>& output_mask,
    const FeaturePackedAffineSpec& spec) {
    if (bias.size() != spec.output_dimension ||
        output_mask.size() != spec.block_dimension) {
        throw std::invalid_argument(
            "feature-packed affine public parameter dimensions do not match");
    }
    if (!std::all_of(bias.begin(), bias.end(), [](double value) {
            return std::isfinite(value);
        })) {
        throw std::invalid_argument(
            "feature-packed affine bias contains a non-finite value");
    }

    bool any_output = false;
    for (std::size_t slot = 0; slot < output_mask.size(); ++slot) {
        const double value = output_mask[slot];
        if (!std::isfinite(value) ||
            (std::abs(value) > 1e-12 &&
             std::abs(value - 1.0) > 1e-12)) {
            throw std::invalid_argument(
                "feature-packed output mask must contain only zero or one");
        }
        if (slot >= spec.output_dimension && value > 0.5) {
            throw std::invalid_argument(
                "feature-packed output mask activates a slot outside output_dimension");
        }
        any_output = any_output || value > 0.5;
    }
    if (!any_output) {
        throw std::invalid_argument(
            "feature-packed output mask must retain at least one output");
    }
}

bool PackingContractsMatch(
    const CipherTensor& lhs,
    const CipherTensor& rhs) {
    return lhs.size() == rhs.size() &&
        lhs.packing.layout == rhs.packing.layout &&
        lhs.packing.logical_shape == rhs.packing.logical_shape &&
        lhs.packing.batch_lanes == rhs.packing.batch_lanes &&
        lhs.packing.slot_count == rhs.packing.slot_count &&
        lhs.packing.active_slots == rhs.packing.active_slots &&
        lhs.packing.encoded_slots == rhs.packing.encoded_slots &&
        lhs.packing.level == rhs.packing.level &&
        lhs.packing.noise_scale_degree == rhs.packing.noise_scale_degree &&
        ScalesMatch(
            lhs.packing.scaling_factor,
            rhs.packing.scaling_factor);
}

std::vector<double> PreRotatedDiagonal(
    const FeaturePackedWeights& weights,
    const std::vector<double>& output_mask,
    const FeaturePackedAffineSpec& spec,
    std::size_t diagonal_offset,
    std::size_t giant_offset) {
    std::vector<double> diagonal(spec.block_dimension, 0.0);
    for (std::size_t output_feature = 0;
         output_feature < spec.output_dimension;
         ++output_feature) {
        const std::size_t input_feature =
            (output_feature + diagonal_offset) % spec.block_dimension;
        if (input_feature < spec.input_dimension) {
            diagonal[output_feature] =
                weights[input_feature][output_feature] *
                output_mask[output_feature];
        }
    }

    std::vector<double> pre_rotated(spec.block_dimension);
    for (std::size_t slot = 0; slot < spec.block_dimension; ++slot) {
        const std::size_t source =
            (slot + spec.block_dimension - giant_offset) %
            spec.block_dimension;
        pre_rotated[slot] = diagonal[source];
    }
    return pre_rotated;
}

CipherTensor EvaluateDenseAffineRaw(
    ServerRuntime& server,
    const CipherTensor& input,
    const FeaturePackedWeights& weights,
    const std::vector<double>& output_mask,
    const FeaturePackedAffineSpec& spec,
    const std::vector<uint32_t>& offsets) {
    std::vector<bool> baby_used(spec.baby_step, false);
    for (const uint32_t offset : offsets) {
        baby_used[offset % spec.baby_step] = true;
    }

    // Process one term at a time. This retains only its reusable baby rotations
    // plus the previously completed raw totals, rather than all terms' BSGS
    // working sets at once.
    std::vector<CipherTensor> baby_rotations(spec.baby_step);
    baby_rotations[0] = input;
    for (std::size_t baby = 1; baby < spec.baby_step; ++baby) {
        if (baby_used[baby]) {
            baby_rotations[baby] = server.Rotate(
                input,
                NormalizeRotation(baby, spec.block_dimension));
        }
    }

    CipherTensor total;
    bool has_total = false;
    const std::size_t giant_count =
        spec.block_dimension / spec.baby_step;
    std::size_t offset_index = 0;
    for (std::size_t giant = 0; giant < giant_count; ++giant) {
        const std::size_t giant_offset = giant * spec.baby_step;
        CipherTensor group;
        bool has_group = false;
        while (offset_index < offsets.size() &&
               offsets[offset_index] < giant_offset + spec.baby_step) {
            const std::size_t offset = offsets[offset_index++];
            if (offset < giant_offset) {
                throw std::logic_error(
                    "feature-packed diagonal offsets are not sorted");
            }
            const std::size_t baby = offset - giant_offset;
            const auto plaintext = server.EncodeModelVector(
                PreRotatedDiagonal(
                    weights,
                    output_mask,
                    spec,
                    offset,
                    giant_offset),
                input.packing);
            auto product = server.MultiplyPlain(
                baby_rotations[baby],
                {plaintext});
            group = has_group
                ? server.Add(group, product)
                : std::move(product);
            has_group = true;
        }
        if (!has_group) {
            continue;
        }
        if (giant_offset != 0) {
            group = server.Rotate(
                group,
                NormalizeRotation(giant_offset, spec.block_dimension));
        }
        total = has_total
            ? server.Add(total, group)
            : std::move(group);
        has_total = true;
    }
    if (offset_index != offsets.size() || !has_total) {
        throw std::logic_error(
            "feature-packed BSGS did not consume every dense diagonal");
    }
    total.packing.layout = PackingLayout::kContiguous;
    total.packing.logical_shape = {
        spec.block_dimension,
        input.size()};
    return total;
}

}  // namespace

std::vector<int32_t> FeaturePackedAffineRotationIndices(
    const FeaturePackedAffineSpec& spec) {
    const auto offsets = DenseDiagonalOffsets(spec);
    std::vector<bool> baby_used(spec.baby_step, false);
    std::vector<bool> giant_used(
        spec.block_dimension / spec.baby_step,
        false);
    for (const uint32_t offset : offsets) {
        baby_used[offset % spec.baby_step] = true;
        giant_used[offset / spec.baby_step] = true;
    }

    std::set<int32_t> indices;
    for (std::size_t baby = 1; baby < baby_used.size(); ++baby) {
        if (baby_used[baby]) {
            indices.insert(NormalizeRotation(baby, spec.block_dimension));
        }
    }
    for (std::size_t giant = 1; giant < giant_used.size(); ++giant) {
        if (giant_used[giant]) {
            indices.insert(NormalizeRotation(
                giant * spec.baby_step,
                spec.block_dimension));
        }
    }
    return {indices.begin(), indices.end()};
}

CipherTensor FeaturePackedOps::DenseAffine(
    const CipherTensor& input,
    const FeaturePackedWeights& weights,
    const std::vector<double>& bias,
    const std::vector<double>& output_mask,
    const FeaturePackedAffineSpec& spec) {
    return DenseAffineSum(
        {{&input, &weights}},
        bias,
        output_mask,
        spec).output;
}

FeaturePackedAffineSumResult FeaturePackedOps::DenseAffineSum(
    const std::vector<FeaturePackedAffineTermView>& terms,
    const std::vector<double>& bias,
    const std::vector<double>& output_mask,
    const FeaturePackedAffineSpec& spec) {
    ValidateSpec(spec);
    ValidateBiasAndMask(bias, output_mask, spec);
    if (terms.empty()) {
        throw std::invalid_argument(
            "feature-packed affine sum requires at least one term");
    }

    // Validate the complete multi-term contract before the first rotation can
    // mutate RunMetrics. This includes failures in later terms.
    const CipherTensor* canonical_input = nullptr;
    for (const auto& term : terms) {
        if (term.input == nullptr || term.weights == nullptr) {
            throw std::invalid_argument(
                "feature-packed affine sum contains a null term view");
        }
        ValidateTensor(*term.input, spec);
        ValidateWeights(*term.weights, spec);
        if (canonical_input == nullptr) {
            canonical_input = term.input;
        }
        else if (!PackingContractsMatch(*canonical_input, *term.input)) {
            throw std::invalid_argument(
                "feature-packed affine sum terms have mismatched packing, "
                "level, or scale contracts");
        }
    }

    for (const auto& term : terms) {
        // Preflight profile-dependent slot, level, scale, and remaining-depth
        // metadata for every term before any homomorphic work begins.
        static_cast<void>(server_.EncodeModelVector(
            std::vector<double>(spec.block_dimension, 0.0),
            term.input->packing));
        if (server_.RemainingLevels(*term.input) < 1) {
            throw std::runtime_error(
                "feature-packed affine sum requires one usable level per term");
        }
    }
    server_.RequireEvaluationKeys(
        FeaturePackedAffineRotationIndices(spec),
        false);

    const auto offsets = DenseDiagonalOffsets(spec);
    FeaturePackedAffineSumResult result;
    result.raw_contributions.reserve(terms.size());
    CipherTensor raw_sum;
    for (const auto& term : terms) {
        auto raw = EvaluateDenseAffineRaw(
            server_,
            *term.input,
            *term.weights,
            output_mask,
            spec,
            offsets);
        raw_sum = result.raw_contributions.empty()
            ? raw
            : server_.Add(raw_sum, raw);
        result.raw_contributions.push_back(std::move(raw));
    }
    result.output = server_.Rescale(raw_sum);
    std::vector<double> masked_bias(spec.block_dimension, 0.0);
    for (std::size_t feature = 0;
         feature < spec.output_dimension;
         ++feature) {
        masked_bias[feature] = bias[feature] * output_mask[feature];
    }
    result.output = server_.AddPlain(result.output, {masked_bias});
    result.output.packing.layout = PackingLayout::kContiguous;
    result.output.packing.logical_shape = {
        spec.block_dimension,
        canonical_input->size()};
    return result;
}

}  // namespace moai::openfhe
