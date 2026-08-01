#pragma once

#include "moai/openfhe/approximation_registry.hpp"
#include "moai/openfhe/server_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace moai::openfhe {

using DenseWeights = std::vector<std::vector<double>>;

inline constexpr uint32_t
    kPaperCompatFeaturePackedSoftmaxPostBootstrapDepth = 11;
inline constexpr uint32_t
    kPaperCompatFeaturePackedSoftmaxPreBootstrapDepth = 6;
inline constexpr double kPaperCompatFeaturePackedSoftmaxExpAtZero =
    1.000000000011056;
inline constexpr const char* kPaperCompatFeaturePackedSoftmaxExpCoefficientSha256 =
    "6eda4377151897e8c4ca4d72f2a918db0b888fc6771f8ee5cf1d950b378de76f";

struct GeluResult {
    CipherTensor polynomial_input;
    CipherTensor polynomial_output;
    CipherTensor output;
    bool bootstrapped{false};
};

struct FeedForwardResult {
    CipherTensor pre_activation;
    GeluResult gelu;
    CipherTensor output;
};

struct SoftmaxResult {
    CipherTensor shifted_logits;
    CipherTensor exponentials;
    CipherTensor denominator_before_bootstrap;
    CipherTensor denominator_after_bootstrap;
    CipherTensor reciprocal;
    CipherTensor output;
};

struct LayerNormResult {
    // Ciphertext-only polynomial input after the native bootstrap.  A client
    // may decrypt a copy to verify the frozen inverse-square-root interval;
    // the server never derives plaintext range metadata from activations.
    CipherTensor normalized_variance;
    CipherTensor output;
};

class NonlinearOps {
public:
    explicit NonlinearOps(ServerRuntime& server);

    [[nodiscard]] CipherTensor Gelu(
        const CipherTensor& input,
        const std::vector<double>& active_mask);

    [[nodiscard]] GeluResult GeluWithCheckpoints(
        const CipherTensor& input,
        const std::vector<double>& active_mask);

    [[nodiscard]] SoftmaxResult SoftmaxWithCheckpoints(
        const CipherTensor& logits,
        std::size_t layer,
        std::size_t head,
        const std::vector<double>& active_mask);

    [[nodiscard]] CipherTensor Softmax(
        const CipherTensor& logits,
        std::size_t layer,
        std::size_t head,
        const std::vector<double>& active_mask);

    // Each head span must already contain one score replicated across all
    // head_width slots. The operator normalizes those spans across ciphertexts
    // (keys); it does not derive or inspect this public layout from activations.
    [[nodiscard]] SoftmaxResult MultiHeadSoftmaxWithCheckpoints(
        const CipherTensor& logits_by_key,
        std::size_t layer,
        std::size_t head_width);

    [[nodiscard]] CipherTensor MultiHeadSoftmax(
        const CipherTensor& logits_by_key,
        std::size_t layer,
        std::size_t head_width);

    [[nodiscard]] CipherTensor LayerNorm(
        const CipherTensor& input,
        const std::vector<double>& gamma,
        const std::vector<double>& beta,
        const std::vector<double>& active_mask,
        PaperCompatLayerNormSite site);

    [[nodiscard]] CipherTensor FeaturePackedLayerNorm(
        const CipherTensor& input,
        const std::vector<double>& gamma,
        const std::vector<double>& beta,
        PaperCompatLayerNormSite site,
        std::size_t layer);

    [[nodiscard]] LayerNormResult FeaturePackedLayerNormWithCheckpoints(
        const CipherTensor& input,
        const std::vector<double>& gamma,
        const std::vector<double>& beta,
        PaperCompatLayerNormSite site,
        std::size_t layer);

    [[nodiscard]] FeedForwardResult FeedForward(
        const CipherTensor& input,
        const DenseWeights& input_weights,
        const std::vector<double>& input_bias,
        const DenseWeights& output_weights,
        const std::vector<double>& output_bias,
        const std::vector<double>& active_mask);

private:
    [[nodiscard]] CipherTensor EvaluatePolynomial(
        const CipherTensor& input,
        const ApproximationContract& contract,
        uint32_t reserve_levels = 0);

    [[nodiscard]] GeluResult GeluWithCheckpointsForDownstream(
        const CipherTensor& input,
        const std::vector<double>& active_mask,
        uint32_t downstream_reserve_levels);

    void ValidateMask(
        const CipherTensor& input,
        const std::vector<double>& active_mask) const;

    [[nodiscard]] CipherTensor ApplyMask(
        const CipherTensor& input,
        const std::vector<double>& active_mask);

    [[nodiscard]] CipherTensor MultiplyPostBootstrapSingleScalePlain(
        const CipherTensor& input,
        const std::vector<double>& values);

    [[nodiscard]] CipherTensor Affine(
        const CipherTensor& input,
        const DenseWeights& weights,
        const std::vector<double>& bias,
        const std::vector<double>& active_mask);

    ServerRuntime& server_;
    PaperCompatNonlinearContracts contracts_;
};

}  // namespace moai::openfhe
