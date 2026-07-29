#pragma once

#include "moai/openfhe/types.hpp"

#include <cstdint>
#include <vector>

namespace moai::openfhe {

class ServerRuntime {
public:
    ServerRuntime(
        ServerKeyBundle key_bundle,
        CryptoProfile profile);

    [[nodiscard]] lbcrypto::Plaintext EncodeModelVector(
        const std::vector<double>& values,
        const PackingSpec& packing) const;

    [[nodiscard]] CipherTensor Rotate(const CipherTensor& input, int32_t index);
    [[nodiscard]] CipherTensor MultiplyPlain(
        const CipherTensor& input,
        const std::vector<lbcrypto::Plaintext>& plaintexts);
    [[nodiscard]] CipherTensor Multiply(
        const CipherTensor& lhs,
        const CipherTensor& rhs);
    [[nodiscard]] CipherTensor Add(
        const CipherTensor& lhs,
        const CipherTensor& rhs);
    [[nodiscard]] CipherTensor Sum(const CipherTensor& input);
    [[nodiscard]] CipherTensor Rescale(const CipherTensor& input);

    void PrepareBootstrap();
    [[nodiscard]] CipherTensor Bootstrap(const CipherTensor& input);

    [[nodiscard]] const RunMetrics& metrics() const noexcept {
        return metrics_;
    }

private:
    ServerKeyBundle key_bundle_;
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context_;
    CryptoProfile profile_;
    RunMetrics metrics_;
    bool bootstrap_prepared_{false};
};

}  // namespace moai::openfhe
