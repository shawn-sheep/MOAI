#pragma once

#include "moai/openfhe/types.hpp"

#include <iosfwd>

namespace moai::openfhe {

[[nodiscard]] CryptoProfile MakePaperCompatProfile();
// Feature-packed BERT trace profile. This keeps the M1-M3 46/51-bit profile
// frozen while using the separately hashed M4/M5 modulus sizes and bootstrap
// schedule needed by the 1024-slot encoder packing.
[[nodiscard]] CryptoProfile MakePaperCompatFeaturePackedProfile();
[[nodiscard]] std::string CanonicalCryptoProfileJson(
    const CryptoProfile& profile);
[[nodiscard]] std::string ComputeCryptoProfileParameterSha256(
    const CryptoProfile& profile);
void ConfigureBootstrap(
    CryptoProfile& profile,
    uint32_t bootstrap_slots,
    uint32_t levels_available_after_bootstrap,
    uint32_t bootstrap_iterations = 1,
    uint32_t bootstrap_precision = 0,
    uint32_t bootstrap_correction_factor = 0,
    bool bootstrap_slots_to_coefficients_first = false);
void ValidateCryptoProfile(const CryptoProfile& profile);
void ValidateCryptoContextMatchesProfile(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
    const CryptoProfile& profile);
void PrintSecurityDisclosure(const CryptoProfile& profile, std::ostream& output);

[[nodiscard]] lbcrypto::CryptoContext<lbcrypto::DCRTPoly> MakeCryptoContext(
    const CryptoProfile& profile);

}  // namespace moai::openfhe
