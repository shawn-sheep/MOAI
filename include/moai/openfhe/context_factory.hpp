#pragma once

#include "moai/openfhe/types.hpp"

#include <iosfwd>

namespace moai::openfhe {

[[nodiscard]] CryptoProfile MakePaperCompatProfile();
void ConfigureBootstrap(
    CryptoProfile& profile,
    uint32_t bootstrap_slots,
    uint32_t levels_available_after_bootstrap);
void ValidateCryptoProfile(const CryptoProfile& profile);
void PrintSecurityDisclosure(const CryptoProfile& profile, std::ostream& output);

[[nodiscard]] lbcrypto::CryptoContext<lbcrypto::DCRTPoly> MakeCryptoContext(
    const CryptoProfile& profile);

}  // namespace moai::openfhe
