#pragma once

#include "openfhe.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace moai::openfhe {

class ClientRuntime;
class ServerRuntime;

enum class PackingLayout {
    kContiguous,
    kColumn,
    kDiagonal,
    kInterleaved,
};

struct CryptoProfile {
    std::string profile_id;
    std::string security_claim;
    std::string warning;
    std::string parameter_sha256;
    uint32_t effective_profile_schema_version{1};

    uint32_t ring_dimension{0};
    uint32_t slot_count{0};
    uint32_t scaling_modulus_bits{0};
    uint32_t first_modulus_bits{0};
    uint32_t multiplicative_depth{0};
    uint32_t levels_available_after_bootstrap{0};
    uint32_t sparse_secret_hamming_weight{0};
    uint32_t num_large_digits{3};
    uint32_t bootstrap_slots{0};
    uint32_t bootstrap_iterations{1};
    uint32_t bootstrap_precision{0};
    uint32_t bootstrap_correction_factor{0};

    lbcrypto::SecretKeyDist secret_key_distribution{lbcrypto::SPARSE_TERNARY};
    lbcrypto::SecurityLevel security_level{lbcrypto::HEStd_NotSet};
    lbcrypto::KeySwitchTechnique key_switch_technique{lbcrypto::HYBRID};
    lbcrypto::ScalingTechnique scaling_technique{lbcrypto::FLEXIBLEAUTO};

    bool bootstrap_enabled{false};
    bool bootstrap_slots_to_coefficients_first{false};
    std::vector<uint32_t> bootstrap_level_budget;
    std::vector<uint32_t> bootstrap_bsgs_dim;
};

struct PackingSpec {
    PackingLayout layout{PackingLayout::kContiguous};
    std::vector<std::size_t> logical_shape;
    uint32_t batch_lanes{1};
    uint32_t slot_count{0};
    uint32_t active_slots{0};
    uint32_t encoded_slots{0};
    uint32_t level{0};
    uint32_t noise_scale_degree{1};
    double scaling_factor{0.0};
};

struct CipherTensor {
    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ciphertexts;
    PackingSpec packing;

    [[nodiscard]] bool empty() const noexcept {
        return ciphertexts.empty();
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return ciphertexts.size();
    }
};

struct DeclaredRange {
    double minimum{0.0};
    double maximum{0.0};
};

struct ApproximationContract {
    std::string contract_id;
    std::string basis{"chebyshev"};
    std::string coefficient_sha256;
    DeclaredRange interval;
    uint32_t degree{0};
    uint32_t required_depth{0};
    uint64_t estimated_multiplications{0};
    std::vector<double> coefficients;
};

class ServerKeyBundle {
private:
    friend class ClientRuntime;
    friend class ServerRuntime;

    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context;
    lbcrypto::PublicKey<lbcrypto::DCRTPoly> public_key;
    std::string key_tag;
    std::string profile_parameter_sha256;
    std::vector<int32_t> rotation_indices;
    bool has_multiplication_key{false};
    bool has_bootstrap_key{false};
};

struct RunMetrics {
    double relative_l2{0.0};
    double cosine_similarity{0.0};
    double max_absolute_error{0.0};
    double latency_ms{0.0};
    uint64_t peak_rss_bytes{0};

    uint64_t rotations{0};
    uint64_t ct_pt_multiplications{0};
    uint64_t ct_ct_multiplications{0};
    uint64_t rescale_operations{0};
    uint64_t bootstraps{0};
    uint64_t bootstrap_iterations{0};
    uint64_t chebyshev_evaluations{0};
    uint64_t estimated_polynomial_multiplications{0};

    uint32_t multiplicative_depth{0};
    uint32_t max_observed_level{0};
    uint32_t max_polynomial_depth{0};
};

[[nodiscard]] const char* ToString(PackingLayout layout) noexcept;

}  // namespace moai::openfhe
