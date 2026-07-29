#pragma once

#include "moai/openfhe/server_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace moai::openfhe {

// Feature-packed tensors store one token in each ciphertext. The first logical
// dimension is the power-of-two cyclic feature block, and the second dimension
// is the number of token ciphertexts.
struct FeaturePackedAffineSpec {
    std::size_t block_dimension{0};
    std::size_t input_dimension{0};
    std::size_t output_dimension{0};
    std::size_t baby_step{0};
};

using FeaturePackedWeights = std::vector<std::vector<double>>;

// Non-owning input/weight view for a shared-rescale affine sum. The referenced
// objects must remain alive for the synchronous DenseAffineSum call.
struct FeaturePackedAffineTermView {
    const CipherTensor* input{nullptr};
    const FeaturePackedWeights* weights{nullptr};
};

struct FeaturePackedAffineSumResult {
    // Each contribution is the masked BSGS total before the one shared explicit
    // Rescale request. These ciphertext-only values are retained for client-side
    // diagnostics and must not be treated as ordinary DenseAffine outputs.
    std::vector<CipherTensor> raw_contributions;
    CipherTensor output;
};

// Returns the exact rotation indices used by DenseAffine for the declared
// dense rectangular dimensions. Indices are normalized to the cyclic block
// and returned in sorted, duplicate-free order.
[[nodiscard]] std::vector<int32_t> FeaturePackedAffineRotationIndices(
    const FeaturePackedAffineSpec& spec);

class FeaturePackedOps {
public:
    explicit FeaturePackedOps(ServerRuntime& server) : server_(server) {}

    // Computes y = x * weights + bias independently in every token
    // ciphertext. weights is input-major [input_dimension][output_dimension].
    // output_mask is a public block-sized binary vector; it may suppress
    // outputs below output_dimension and must be zero above it. The mask is
    // fused into the public diagonals and bias, so it consumes no extra level.
    [[nodiscard]] CipherTensor DenseAffine(
        const CipherTensor& input,
        const FeaturePackedWeights& weights,
        const std::vector<double>& bias,
        const std::vector<double>& output_mask,
        const FeaturePackedAffineSpec& spec);

    // Computes bias + sum_i(inputs[i] * weights[i]) with one shared explicit
    // Rescale request after all raw BSGS contributions have been accumulated.
    // Every term must have an identical feature-packed level/scale contract.
    // The public output mask remains fused into all diagonals and the one bias.
    [[nodiscard]] FeaturePackedAffineSumResult DenseAffineSum(
        const std::vector<FeaturePackedAffineTermView>& terms,
        const std::vector<double>& bias,
        const std::vector<double>& output_mask,
        const FeaturePackedAffineSpec& spec);

private:
    ServerRuntime& server_;
};

}  // namespace moai::openfhe
