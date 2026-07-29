#pragma once

#include "moai/openfhe/types.hpp"

#include <cstdint>
#include <vector>

namespace moai::openfhe {

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
