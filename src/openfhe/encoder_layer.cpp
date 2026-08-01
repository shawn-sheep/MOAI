#include "moai/openfhe/encoder_layer.hpp"

#include "moai/openfhe/approximation_registry.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace moai::openfhe {
namespace {

const FeaturePackedAffineSpec kHiddenAffineSpec{
    kPaperCompatFeatureBlock,
    kPaperCompatHiddenSize,
    kPaperCompatHiddenSize,
    kPaperCompatAffineBabyStep};
const FeaturePackedAffineSpec kIntermediateAffineSpec{
    kPaperCompatFeatureBlock,
    kPaperCompatHiddenSize,
    kPaperCompatFeatureBlock,
    kPaperCompatAffineBabyStep};
const FeaturePackedAffineSpec kOutputAffineSpec{
    kPaperCompatFeatureBlock,
    kPaperCompatFeatureBlock,
    kPaperCompatHiddenSize,
    kPaperCompatAffineBabyStep};

std::vector<double> PrefixMask(std::size_t active) {
    if (active == 0 || active > kPaperCompatFeatureBlock) {
        throw std::invalid_argument("encoder prefix mask dimension is invalid");
    }
    std::vector<double> mask(kPaperCompatFeatureBlock, 0.0);
    std::fill_n(mask.begin(), active, 1.0);
    return mask;
}

void ValidateVector(
    const std::vector<double>& values,
    std::size_t expected,
    const std::string& label) {
    if (values.size() != expected ||
        !std::all_of(values.begin(), values.end(), [](double value) {
            return std::isfinite(value);
        })) {
        throw std::invalid_argument(label + " has an invalid public-vector contract");
    }
}

void ValidateMatrix(
    const FeaturePackedWeights& matrix,
    std::size_t rows,
    std::size_t columns,
    const std::string& label) {
    if (matrix.size() != rows) {
        throw std::invalid_argument(label + " has an invalid public-matrix row count");
    }
    for (const auto& row : matrix) {
        if (row.size() != columns ||
            !std::all_of(row.begin(), row.end(), [](double value) {
                return std::isfinite(value);
            })) {
            throw std::invalid_argument(
                label + " is ragged or contains a non-finite public weight");
        }
    }
}

void ValidateWeights(const EncoderLayerWeights& weights) {
    if (weights.layer_index >= kPaperCompatEncoderLayers) {
        throw std::invalid_argument("encoder layer index is outside paper_compat");
    }
    ValidateMatrix(
        weights.query_weights,
        kPaperCompatHiddenSize,
        kPaperCompatHiddenSize,
        "query weights");
    ValidateVector(weights.query_bias, kPaperCompatHiddenSize, "query bias");
    ValidateMatrix(
        weights.key_weights,
        kPaperCompatHiddenSize,
        kPaperCompatHiddenSize,
        "key weights");
    ValidateVector(weights.key_bias, kPaperCompatHiddenSize, "key bias");
    ValidateMatrix(
        weights.value_weights,
        kPaperCompatHiddenSize,
        kPaperCompatHiddenSize,
        "value weights");
    ValidateVector(weights.value_bias, kPaperCompatHiddenSize, "value bias");
    ValidateMatrix(
        weights.self_output_weights,
        kPaperCompatHiddenSize,
        kPaperCompatHiddenSize,
        "self-output weights");
    ValidateVector(
        weights.self_output_bias,
        kPaperCompatHiddenSize,
        "self-output bias");
    ValidateVector(
        weights.attention_layernorm_gamma,
        kPaperCompatHiddenSize,
        "attention LayerNorm gamma");
    ValidateVector(
        weights.attention_layernorm_beta,
        kPaperCompatHiddenSize,
        "attention LayerNorm beta");

    for (std::size_t block = 0;
         block < kPaperCompatIntermediateBlocks;
         ++block) {
        ValidateMatrix(
            weights.intermediate_weight_blocks[block],
            kPaperCompatHiddenSize,
            kPaperCompatFeatureBlock,
            "intermediate weight block " + std::to_string(block));
        ValidateVector(
            weights.intermediate_bias_blocks[block],
            kPaperCompatFeatureBlock,
            "intermediate bias block " + std::to_string(block));
        ValidateMatrix(
            weights.output_weight_blocks[block],
            kPaperCompatFeatureBlock,
            kPaperCompatHiddenSize,
            "output weight block " + std::to_string(block));
    }
    ValidateVector(weights.output_bias, kPaperCompatHiddenSize, "output bias");
    ValidateVector(
        weights.output_layernorm_gamma,
        kPaperCompatHiddenSize,
        "output LayerNorm gamma");
    ValidateVector(
        weights.output_layernorm_beta,
        kPaperCompatHiddenSize,
        "output LayerNorm beta");
}

void ValidateInput(const CipherTensor& input) {
    if (input.empty() || input.size() != kPaperCompatTraceTokens ||
        input.packing.layout != PackingLayout::kContiguous ||
        input.packing.batch_lanes != 1 ||
        input.packing.slot_count != 32768 ||
        input.packing.active_slots != kPaperCompatFeatureBlock ||
        input.packing.encoded_slots != kPaperCompatFeatureBlock ||
        input.packing.logical_shape != std::vector<std::size_t>{
            kPaperCompatFeatureBlock,
            kPaperCompatTraceTokens}) {
        throw std::invalid_argument(
            "encoder input must be five 1024-slot feature-packed ciphertexts");
    }
    for (const auto& ciphertext : input.ciphertexts) {
        if (!ciphertext) {
            throw std::invalid_argument(
                "encoder input contains a null ciphertext");
        }
        const double scale_magnitude = std::max(
            1.0,
            std::max(
                std::abs(ciphertext->GetScalingFactor()),
                std::abs(input.packing.scaling_factor)));
        if (ciphertext->GetLevel() != input.packing.level ||
            ciphertext->GetNoiseScaleDeg() !=
                input.packing.noise_scale_degree ||
            !std::isfinite(ciphertext->GetScalingFactor()) ||
            std::abs(
                ciphertext->GetScalingFactor() -
                input.packing.scaling_factor) >
                1e-12 * scale_magnitude) {
            throw std::invalid_argument(
                "encoder input ciphertext metadata is stale");
        }
    }
}

CipherTensor MaskAndRescale(
    ServerRuntime& server,
    const CipherTensor& input,
    const std::vector<double>& mask,
    const std::string& placement) {
    if (server.RemainingLevels(input) < 1) {
        throw std::runtime_error(placement + " lacks one mask level");
    }
    return server.Rescale(server.MultiplyPlain(
        input,
        {server.EncodeModelVector(mask, input.packing)}));
}

CipherTensor BootstrapAndMask(
    ServerRuntime& server,
    const CipherTensor& input,
    const std::vector<double>& mask,
    const std::string& placement) {
    auto output = server.Bootstrap(input);
    return MaskAndRescale(server, output, mask, placement);
}

CipherTensor CloneCipherTensorForObserver(const CipherTensor& input) {
    ValidateInput(input);
    CipherTensor snapshot;
    snapshot.packing = input.packing;
    snapshot.ciphertexts.reserve(input.size());
    for (const auto& ciphertext : input.ciphertexts) {
        snapshot.ciphertexts.push_back(ciphertext->Clone());
    }
    return snapshot;
}

}  // namespace

void RequirePaperCompatInterLayerHandoff(
    const CipherTensor& input,
    const ServerRuntime& server) {
    ValidateInput(input);
    const double scale_bits = std::log2(input.packing.scaling_factor);
    if (input.packing.level != 19 ||
        input.packing.noise_scale_degree != 2 ||
        server.RemainingLevels(input) != 27 ||
        !std::isfinite(scale_bits) ||
        std::abs(scale_bits - 100.0) > 1e-3) {
        throw std::runtime_error(
            "inter-layer encoder handoff must match the frozen "
            "(19,2,27,2^100,5) schedule");
    }
}

std::vector<int32_t> FeaturePackedEncoderRotationIndices() {
    std::set<int32_t> indices;
    const auto append = [&indices](const std::vector<int32_t>& values) {
        indices.insert(values.begin(), values.end());
    };
    append(FeaturePackedAffineRotationIndices(kHiddenAffineSpec));
    append(FeaturePackedAffineRotationIndices(kIntermediateAffineSpec));
    append(FeaturePackedAffineRotationIndices(kOutputAffineSpec));
    append(FeaturePackedAttentionRotationIndices(
        MakePaperCompatFeaturePackedAttentionSpec()));
    return {indices.begin(), indices.end()};
}

EncoderLayerResult FeaturePackedEncoderLayer::Evaluate(
    const CipherTensor& input,
    const EncoderLayerWeights& weights) {
    // Validate the complete public contract before the first homomorphic
    // operation, so malformed weights cannot leave partial server metrics.
    ValidateInput(input);
    ValidateWeights(weights);
    server_.RequireEvaluationKeys(
        FeaturePackedEncoderRotationIndices(),
        true);
    if (server_.RemainingLevels(input) < 18) {
        throw std::runtime_error(
            "encoder input requires at least 18 usable levels for the frozen graph");
    }

    const auto hidden_mask = PrefixMask(kPaperCompatHiddenSize);
    const auto intermediate_mask = PrefixMask(kPaperCompatFeatureBlock);
    const std::vector<double> valid_keys(kPaperCompatTraceTokens, 1.0);

    EncoderLayerResult result;
    auto& checkpoints = result.checkpoints;
    checkpoints.query = affine_.DenseAffine(
        input,
        weights.query_weights,
        weights.query_bias,
        hidden_mask,
        kHiddenAffineSpec);
    checkpoints.key = affine_.DenseAffine(
        input,
        weights.key_weights,
        weights.key_bias,
        hidden_mask,
        kHiddenAffineSpec);
    checkpoints.value = affine_.DenseAffine(
        input,
        weights.value_weights,
        weights.value_bias,
        hidden_mask,
        kHiddenAffineSpec);

    checkpoints.attention = attention_.Evaluate(
        checkpoints.query,
        checkpoints.key,
        checkpoints.value,
        weights.layer_index,
        valid_keys,
        MakePaperCompatFeaturePackedAttentionSpec());
    // FeaturePackedAttention retains both the pre-cleanup diagnostic and the
    // frozen bootstrap-and-cleanup output. This alias keeps the downstream
    // checkpoint name without duplicating homomorphic work.
    checkpoints.attention_after_bootstrap = checkpoints.attention.output;

    checkpoints.self_projection = affine_.DenseAffine(
        checkpoints.attention_after_bootstrap,
        weights.self_output_weights,
        weights.self_output_bias,
        hidden_mask,
        kHiddenAffineSpec);
    checkpoints.attention_residual =
        server_.Add(checkpoints.self_projection, input);
    auto attention_layernorm =
        nonlinear_.FeaturePackedLayerNormWithCheckpoints(
        checkpoints.attention_residual,
        weights.attention_layernorm_gamma,
        weights.attention_layernorm_beta,
        PaperCompatLayerNormSite::kAttentionResidual,
        weights.layer_index);
    checkpoints.attention_layernorm_normalized_variance =
        std::move(attention_layernorm.normalized_variance);
    checkpoints.attention_layernorm = std::move(attention_layernorm.output);

    for (std::size_t block = 0;
         block < kPaperCompatIntermediateBlocks;
         ++block) {
        checkpoints.intermediate_pre_activation[block] = affine_.DenseAffine(
            checkpoints.attention_layernorm,
            weights.intermediate_weight_blocks[block],
            weights.intermediate_bias_blocks[block],
            intermediate_mask,
            kIntermediateAffineSpec);
        server_.RequireUsableLevels(
            checkpoints.intermediate_pre_activation[block],
            MakePaperCompatNonlinearContracts()
                .depth_budget
                .gelu_from_preactivation_through_output_affine,
            "encoder GELU plus output-affine path");
        auto gelu = nonlinear_.GeluWithCheckpoints(
            checkpoints.intermediate_pre_activation[block],
            intermediate_mask);
        if (gelu.bootstrapped) {
            throw std::runtime_error(
                "paper_compat encoder depth schedule unexpectedly bootstrapped GELU");
        }
        checkpoints.intermediate_polynomial_output[block] =
            std::move(gelu.polynomial_output);
        checkpoints.intermediate_activation[block] = std::move(gelu.output);
    }

    std::vector<FeaturePackedAffineTermView> output_terms;
    output_terms.reserve(kPaperCompatIntermediateBlocks);
    for (std::size_t block = 0;
         block < kPaperCompatIntermediateBlocks;
         ++block) {
        output_terms.push_back({
            &checkpoints.intermediate_activation[block],
            &weights.output_weight_blocks[block]});
    }
    auto output_affine = affine_.DenseAffineSum(
        output_terms,
        weights.output_bias,
        hidden_mask,
        kOutputAffineSpec);
    if (output_affine.raw_contributions.size() !=
        kPaperCompatIntermediateBlocks) {
        throw std::logic_error(
            "encoder output affine returned an unexpected raw checkpoint count");
    }
    for (std::size_t block = 0;
         block < kPaperCompatIntermediateBlocks;
         ++block) {
        checkpoints.output_contributions[block] =
            std::move(output_affine.raw_contributions[block]);
    }
    checkpoints.output_projection = std::move(output_affine.output);
    checkpoints.output_residual_before_bootstrap = server_.Add(
        checkpoints.output_projection,
        checkpoints.attention_layernorm);
    checkpoints.output_residual_after_bootstrap = BootstrapAndMask(
        server_,
        checkpoints.output_residual_before_bootstrap,
        hidden_mask,
        "pre-output-LayerNorm bootstrap mask");
    auto output_layernorm = nonlinear_.FeaturePackedLayerNormWithCheckpoints(
        checkpoints.output_residual_after_bootstrap,
        weights.output_layernorm_gamma,
        weights.output_layernorm_beta,
        PaperCompatLayerNormSite::kFeedForwardResidual,
        weights.layer_index);
    checkpoints.output_layernorm_normalized_variance =
        std::move(output_layernorm.normalized_variance);
    checkpoints.output_layernorm = std::move(output_layernorm.output);
    result.output = checkpoints.output_layernorm;
    return result;
}

EncoderStackResult FeaturePackedEncoder::Evaluate(
    CipherTensor input,
    const std::vector<EncoderLayerWeights>& weights,
    EncoderCiphertextObserver* observer) {
    return EvaluateImpl(
        std::move(input),
        weights,
        kPaperCompatEncoderLayers,
        observer);
}

EncoderStackResult FeaturePackedEncoder::EvaluatePrefixForDiagnostics(
    CipherTensor input,
    const std::vector<EncoderLayerWeights>& weights,
    std::size_t layer_count,
    EncoderCiphertextObserver* observer) {
    if (layer_count == 0 || layer_count >= kPaperCompatEncoderLayers) {
        throw std::invalid_argument(
            "diagnostic encoder prefix must contain between 1 and 11 layers");
    }
    return EvaluateImpl(std::move(input), weights, layer_count, observer);
}

EncoderStackResult FeaturePackedEncoder::EvaluateImpl(
    CipherTensor input,
    const std::vector<EncoderLayerWeights>& weights,
    std::size_t layer_count,
    EncoderCiphertextObserver* observer) {
    ValidateInput(input);
    if (weights.size() != kPaperCompatEncoderLayers) {
        throw std::invalid_argument(
            "paper_compat encoder requires exactly 12 public weight sets");
    }
    for (std::size_t index = 0; index < weights.size(); ++index) {
        ValidateWeights(weights[index]);
        if (weights[index].layer_index != index) {
            throw std::invalid_argument(
                "paper_compat encoder weights must be ordered layers 0 through 11");
        }
    }
    server_.RequireEvaluationKeys(FeaturePackedEncoderRotationIndices(), true);
    const double input_scale_bits = std::log2(input.packing.scaling_factor);
    if (input.packing.level != 29 ||
        input.packing.noise_scale_degree != 1 ||
        server_.RemainingLevels(input) != 18 ||
        !std::isfinite(input_scale_bits) ||
        std::abs(input_scale_bits - 50.0) > 1e-3) {
        throw std::runtime_error(
            "initial encoder input must match the frozen (29,1,18,2^50) schedule");
    }

    const auto hidden_mask = PrefixMask(kPaperCompatHiddenSize);
    auto current = std::move(input);
    for (std::size_t index = 0; index < layer_count; ++index) {
        if (index > 0) {
            RequirePaperCompatInterLayerHandoff(current, server_);
        }
        // OpenFHE may adjust shared ciphertext metadata while automatically
        // aligning operands.  Preserve a deep ciphertext-only pre-layer
        // snapshot for client validation; a shared_ptr copy is not a snapshot.
        const auto input_snapshot = observer == nullptr
            ? CipherTensor{}
            : CloneCipherTensorForObserver(current);
        const RunMetrics metrics_before_layer = server_.metrics();
        auto layer_result = layer_.Evaluate(current, weights[index]);
        const RunMetrics metrics_after_layer = server_.metrics();
        const bool refreshed = index + 1 < layer_count;
        CipherTensor next;
        if (refreshed) {
            next = BootstrapAndMask(
                server_,
                layer_result.output,
                hidden_mask,
                "inter-layer encoder refresh mask");
        }
        const RunMetrics metrics_after_refresh = server_.metrics();
        if (observer != nullptr) {
            observer->Observe({
                index,
                refreshed,
                input_snapshot,
                layer_result,
                metrics_before_layer,
                metrics_after_layer,
                metrics_after_refresh});
        }
        current = refreshed
            ? std::move(next)
            : std::move(layer_result.output);
    }
    return {std::move(current)};
}

}  // namespace moai::openfhe
