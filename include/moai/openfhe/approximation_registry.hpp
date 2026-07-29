#pragma once

#include "moai/openfhe/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace moai::openfhe {

inline constexpr double kPaperCompatLayerNormEpsilon = 1e-12;
inline constexpr double kPaperCompatLayerNorm1VarianceScale = 64.0;
inline constexpr double kPaperCompatLayerNorm2VarianceScale = 1.0;
inline constexpr std::size_t kPaperCompatEncoderLayers = 12;
inline constexpr std::size_t kPaperCompatAttentionHeads = 12;
inline constexpr const char* kPaperCompatSoftmaxShiftContractId =
    "softmax_shift_layer_head_logspace_midpoint_v1";
inline constexpr const char* kPaperCompatSoftmaxShiftSha256 =
    "41ecf6ade53f674096c9afc752ccfe99834876ea4768bcf4bf3ea25a8f502432";

struct PaperCompatSoftmaxShiftContract {
    std::string contract_id;
    std::string values_sha256;
    std::size_t layer_count{0};
    std::size_t head_count{0};
    std::vector<double> values;

    [[nodiscard]] double At(std::size_t layer, std::size_t head) const;
};

enum class PaperCompatLayerNormSite {
    kAttentionResidual,
    kFeedForwardResidual,
};

struct PaperCompatNonlinearDepthBudget {
    uint32_t gelu_from_preactivation_through_output_affine{12};
    uint32_t softmax_pre_bootstrap{7};
    uint32_t softmax_post_bootstrap{12};
    uint32_t layernorm_pre_bootstrap{4};
    uint32_t layernorm_post_bootstrap{11};
};

struct PaperCompatNonlinearContracts {
    PaperCompatNonlinearDepthBudget depth_budget;
    PaperCompatSoftmaxShiftContract softmax_shifts;
    ApproximationContract gelu;
    ApproximationContract softmax_exponential;
    ApproximationContract softmax_reciprocal;
    ApproximationContract layernorm_inverse_sqrt;
};

[[nodiscard]] PaperCompatNonlinearContracts
MakePaperCompatNonlinearContracts();

[[nodiscard]] PaperCompatSoftmaxShiftContract
MakePaperCompatSoftmaxShiftContract();

[[nodiscard]] double PaperCompatSoftmaxPublicShift(
    std::size_t layer,
    std::size_t head);

[[nodiscard]] std::string ComputeCoefficientSha256(
    const std::vector<double>& coefficients);

// Plaintext oracle for the exact OpenFHE Chebyshev coefficient convention:
// c[0] / 2 + sum_{i=1}^d c[i] T_i(x). This never consumes ciphertext data.
[[nodiscard]] double EvaluateChebyshevContractAt(
    const ApproximationContract& contract,
    double input);

}  // namespace moai::openfhe
