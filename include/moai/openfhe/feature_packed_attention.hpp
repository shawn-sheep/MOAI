#pragma once

#include "moai/openfhe/nonlinear_ops.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace moai::openfhe {

inline constexpr std::size_t kPaperCompatAttentionTokenCount = 5;
inline constexpr std::size_t kPaperCompatAttentionHeadCount = 12;
inline constexpr std::size_t kPaperCompatAttentionHeadDimension = 64;
inline constexpr std::size_t kPaperCompatAttentionFeatureDimension =
    kPaperCompatAttentionHeadCount * kPaperCompatAttentionHeadDimension;
inline constexpr std::size_t kPaperCompatAttentionBlockDimension = 1024;
inline constexpr double kPaperCompatAttentionScoreScale = 1.0 / 8.0;

// paper_compat stores one token in each ciphertext. The first 768 slots are 12
// contiguous 64-feature heads and the remaining 256 slots are inactive. Tests
// may reduce token count and head width, but the 12-head shift registry stays
// fixed; reduced contracts carry no additional security or accuracy claim.
struct FeaturePackedAttentionSpec {
    std::size_t block_dimension{kPaperCompatAttentionBlockDimension};
    std::size_t token_count{kPaperCompatAttentionTokenCount};
    std::size_t head_count{kPaperCompatAttentionHeadCount};
    std::size_t head_dimension{kPaperCompatAttentionHeadDimension};
};

[[nodiscard]] constexpr FeaturePackedAttentionSpec
MakePaperCompatFeaturePackedAttentionSpec() noexcept {
    return {};
}

// Exact positive rotation keys used by the isolated per-head SumSlots path.
// The returned indices are sorted and duplicate-free.
[[nodiscard]] std::vector<int32_t> FeaturePackedAttentionRotationIndices(
    const FeaturePackedAttentionSpec& spec);

struct FeaturePackedAttentionResult {
    // Query-major [query * token_count + key]. Every slot in a head contains
    // the encrypted Q*K/sqrt(head_dimension) score for that head.
    CipherTensor scaled_scores;

    // Query-major [query * token_count + key]. Every slot in a head contains
    // that key's encrypted probability under the frozen per-layer/head shift
    // and exp/reciprocal contracts.
    CipherTensor probabilities;

    // One ciphertext per query containing the encrypted weighted V sum before
    // the native bootstrap and public prefix cleanup. This remains an explicit
    // client-decrypted diagnostic checkpoint and must pass the same inactive-slot
    // and quality gates as the cleaned output.
    CipherTensor output_before_bootstrap_cleanup;

    // One ciphertext per query containing the encrypted weighted V sum after
    // the frozen native bootstrap and public prefix cleanup.
    CipherTensor output;
};

class FeaturePackedAttention {
public:
    explicit FeaturePackedAttention(ServerRuntime& server)
        : server_(server), nonlinear_(server) {}

    // q, k, and v contain one ciphertext per token. valid_key_mask is public,
    // fixed before execution, and must contain token_count ones. The frozen
    // trace has exactly five valid keys and does not permit padding selected
    // from ciphertext values or activation-derived plaintext metadata.
    //
    // Native-bootstrap evaluation keys must match block_dimension. Missing
    // profile/key support fails closed in ServerRuntime; there is no decrypt or
    // plaintext-activation fallback in this server target.
    [[nodiscard]] FeaturePackedAttentionResult Evaluate(
        const CipherTensor& q,
        const CipherTensor& k,
        const CipherTensor& v,
        std::size_t layer,
        const std::vector<double>& valid_key_mask,
        const FeaturePackedAttentionSpec& spec =
            MakePaperCompatFeaturePackedAttentionSpec());

private:
    ServerRuntime& server_;
    NonlinearOps nonlinear_;
};

}  // namespace moai::openfhe
