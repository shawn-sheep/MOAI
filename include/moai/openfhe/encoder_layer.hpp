#pragma once

#include "moai/openfhe/feature_packed_attention.hpp"
#include "moai/openfhe/feature_packed_ops.hpp"
#include "moai/openfhe/nonlinear_ops.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace moai::openfhe {

inline constexpr std::size_t kPaperCompatFeatureBlock = 1024;
inline constexpr std::size_t kPaperCompatTraceTokens = 5;
inline constexpr std::size_t kPaperCompatHiddenSize = 768;
inline constexpr std::size_t kPaperCompatIntermediateSize = 3072;
inline constexpr std::size_t kPaperCompatIntermediateBlocks = 3;
inline constexpr std::size_t kPaperCompatAffineBabyStep = 32;

// Public model parameters are input-major. The three FFN blocks are already
// bound to the frozen five-token channel-scale contract: W1 and b1 are
// multiplied by each channel scale, while the matching W2 input row is divided
// by that scale. Runtime activation-derived scaling is forbidden.
struct EncoderLayerWeights {
    std::size_t layer_index{0};

    FeaturePackedWeights query_weights;
    std::vector<double> query_bias;
    FeaturePackedWeights key_weights;
    std::vector<double> key_bias;
    FeaturePackedWeights value_weights;
    std::vector<double> value_bias;

    FeaturePackedWeights self_output_weights;
    std::vector<double> self_output_bias;
    std::vector<double> attention_layernorm_gamma;
    std::vector<double> attention_layernorm_beta;

    std::array<FeaturePackedWeights, kPaperCompatIntermediateBlocks>
        intermediate_weight_blocks;
    std::array<std::vector<double>, kPaperCompatIntermediateBlocks>
        intermediate_bias_blocks;
    std::array<FeaturePackedWeights, kPaperCompatIntermediateBlocks>
        output_weight_blocks;
    std::vector<double> output_bias;
    std::vector<double> output_layernorm_gamma;
    std::vector<double> output_layernorm_beta;
};

struct EncoderLayerCheckpoints {
    CipherTensor query;
    CipherTensor key;
    CipherTensor value;
    FeaturePackedAttentionResult attention;
    CipherTensor attention_after_bootstrap;
    CipherTensor self_projection;
    CipherTensor attention_residual;
    CipherTensor attention_layernorm;

    std::array<CipherTensor, kPaperCompatIntermediateBlocks>
        intermediate_pre_activation;
    std::array<CipherTensor, kPaperCompatIntermediateBlocks>
        intermediate_polynomial_output;
    std::array<CipherTensor, kPaperCompatIntermediateBlocks>
        intermediate_activation;
    // Per-block W2 BSGS totals before their one shared explicit Rescale request.
    // They are ciphertext-only client diagnostic checkpoints, not standalone
    // DenseAffine outputs and therefore have a different scale contract.
    std::array<CipherTensor, kPaperCompatIntermediateBlocks>
        output_contributions;
    CipherTensor output_projection;
    CipherTensor output_residual_before_bootstrap;
    CipherTensor output_residual_after_bootstrap;
    CipherTensor output_layernorm;
};

struct EncoderLayerResult {
    CipherTensor output;
    EncoderLayerCheckpoints checkpoints;
};

// Exact rotation-key union for the 1024-slot BERT-base layer. The result is
// sorted and duplicate-free and includes every SumSlots power-of-two key.
[[nodiscard]] std::vector<int32_t> FeaturePackedEncoderRotationIndices();

class FeaturePackedEncoderLayer {
public:
    explicit FeaturePackedEncoderLayer(ServerRuntime& server)
        : server_(server),
          affine_(server),
          attention_(server),
          nonlinear_(server) {}

    // The server receives only ciphertext activations, public model parameters,
    // and the evaluation-key-backed ServerRuntime. Every checkpoint remains a
    // ciphertext; only ClientRuntime may decrypt copies for validation.
    [[nodiscard]] EncoderLayerResult Evaluate(
        const CipherTensor& input,
        const EncoderLayerWeights& weights);

private:
    ServerRuntime& server_;
    FeaturePackedOps affine_;
    FeaturePackedAttention attention_;
    NonlinearOps nonlinear_;
};

}  // namespace moai::openfhe
