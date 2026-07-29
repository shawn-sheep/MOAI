#pragma once

#include "moai/openfhe/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace moai::openfhe {

class NonlinearOps;

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
    [[nodiscard]] CipherTensor Subtract(
        const CipherTensor& lhs,
        const CipherTensor& rhs);
    [[nodiscard]] CipherTensor AddPlain(
        const CipherTensor& input,
        const std::vector<std::vector<double>>& plaintexts);
    [[nodiscard]] CipherTensor SubtractPlain(
        const CipherTensor& input,
        const std::vector<std::vector<double>>& plaintexts);
    [[nodiscard]] CipherTensor Sum(const CipherTensor& input);
    [[nodiscard]] CipherTensor Rescale(const CipherTensor& input);

    [[nodiscard]] uint32_t RemainingLevels(
        const CipherTensor& input) const;

    void PrepareBootstrap();
    [[nodiscard]] CipherTensor Bootstrap(const CipherTensor& input);

    [[nodiscard]] const RunMetrics& metrics() const noexcept {
        return metrics_;
    }

private:
    friend class NonlinearOps;

    [[nodiscard]] CipherTensor EvaluateChebyshev(
        const CipherTensor& input,
        const ApproximationContract& contract);
    void RequireUsableLevels(
        const CipherTensor& input,
        uint32_t required_levels,
        const std::string& operation) const;

    ServerKeyBundle key_bundle_;
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context_;
    CryptoProfile profile_;
    RunMetrics metrics_;
    bool bootstrap_prepared_{false};
};

}  // namespace moai::openfhe
