#pragma once

#include "moai/openfhe/approximation_registry.hpp"
#include "moai/openfhe/server_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace moai::openfhe {

using DenseWeights = std::vector<std::vector<double>>;

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
    CipherTensor exponentials;
    CipherTensor denominator_before_bootstrap;
    CipherTensor denominator_after_bootstrap;
    CipherTensor reciprocal;
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

    [[nodiscard]] CipherTensor LayerNorm(
        const CipherTensor& input,
        const std::vector<double>& gamma,
        const std::vector<double>& beta,
        const std::vector<double>& active_mask,
        PaperCompatLayerNormSite site);

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

    [[nodiscard]] CipherTensor Affine(
        const CipherTensor& input,
        const DenseWeights& weights,
        const std::vector<double>& bias,
        const std::vector<double>& active_mask);

    ServerRuntime& server_;
    PaperCompatNonlinearContracts contracts_;
};

}  // namespace moai::openfhe
