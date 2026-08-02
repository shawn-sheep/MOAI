#pragma once

#include "moai/openfhe/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace moai::openfhe {

inline constexpr char kOpenFheBinaryArchiveComponentSumV1[] =
    "openfhe_binary_archive_component_sum_v1";

// Each field counts an independent OpenFHE BINARY archive.  The server bundle
// total is the checked sum of the context, public key, multiplication-key, and
// automorphism-key archives.  It is a reproducible component sum, not a wire
// format, and deliberately excludes the client-only private key.
struct SerializedKeySizeMetrics {
    std::string serialization_format{kOpenFheBinaryArchiveComponentSumV1};
    uint64_t context_bytes{0};
    uint64_t public_key_bytes{0};
    uint64_t private_key_bytes{0};
    uint64_t evaluation_multiplication_key_bytes{0};
    uint64_t evaluation_automorphism_key_bytes{0};
    uint64_t server_key_bundle_component_sum_bytes{0};
};

// Ciphertexts are serialized as independent OpenFHE BINARY archives and then
// summed with overflow checking.  Packing metadata is intentionally excluded;
// this component sum is not a CipherTensor wire format.
struct SerializedCipherTensorSizeMetrics {
    std::string serialization_format{kOpenFheBinaryArchiveComponentSumV1};
    uint64_t ciphertext_count{0};
    uint64_t ciphertext_component_sum_bytes{0};
};

class ClientRuntime {
public:
    explicit ClientRuntime(CryptoProfile profile);

    void GenerateEvaluationKeys(
        const std::vector<int32_t>& rotation_indices,
        bool include_bootstrap_keys);

    [[nodiscard]] CipherTensor Encrypt(
        const std::vector<std::vector<double>>& plaintexts,
        const PackingSpec& packing) const;

    [[nodiscard]] std::vector<std::vector<double>> Decrypt(
        const CipherTensor& tensor) const;

    [[nodiscard]] SerializedKeySizeMetrics MeasureSerializedKeySizes() const;

    [[nodiscard]] SerializedCipherTensorSizeMetrics
    MeasureSerializedCipherTensorSizes(const CipherTensor& tensor) const;

    [[nodiscard]] ServerKeyBundle ExportServerKeyBundle() const;

private:
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context_;
    CryptoProfile profile_;
    lbcrypto::PublicKey<lbcrypto::DCRTPoly> public_key_;
    lbcrypto::PrivateKey<lbcrypto::DCRTPoly> private_key_;
    bool keys_generated_{false};
    bool evaluation_keys_generated_{false};
    std::vector<int32_t> rotation_indices_;
    std::vector<lbcrypto::EvalKey<lbcrypto::DCRTPoly>>
        multiplication_eval_keys_;
    std::map<uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>
        automorphism_eval_keys_;
    std::vector<uint32_t> bootstrap_required_indices_;
};

}  // namespace moai::openfhe
