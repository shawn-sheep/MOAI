#pragma once

#include "moai/openfhe/types.hpp"

#include <cstdint>
#include <vector>

namespace moai::openfhe {

class ClientRuntime {
public:
    ClientRuntime(
        lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context,
        CryptoProfile profile);

    void GenerateEvaluationKeys(
        const std::vector<int32_t>& rotation_indices,
        bool include_bootstrap_keys);

    [[nodiscard]] CipherTensor Encrypt(
        const std::vector<std::vector<double>>& plaintexts,
        const PackingSpec& packing) const;

    [[nodiscard]] std::vector<std::vector<double>> Decrypt(
        const CipherTensor& tensor) const;

    [[nodiscard]] ServerKeyBundle ExportServerKeyBundle() const;

    [[nodiscard]] const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& public_key() const noexcept {
        return public_key_;
    }

    [[nodiscard]] const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context() const noexcept {
        return context_;
    }

private:
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context_;
    CryptoProfile profile_;
    lbcrypto::PublicKey<lbcrypto::DCRTPoly> public_key_;
    lbcrypto::PrivateKey<lbcrypto::DCRTPoly> private_key_;
    bool keys_generated_{false};
    bool multiplication_key_generated_{false};
    bool bootstrap_key_generated_{false};
    std::vector<int32_t> rotation_indices_;
};

}  // namespace moai::openfhe
