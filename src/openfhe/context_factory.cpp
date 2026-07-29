#include "moai/openfhe/context_factory.hpp"

#include <ostream>
#include <stdexcept>
#include <string>

namespace moai::openfhe {

const char* ToString(PackingLayout layout) noexcept {
    switch (layout) {
        case PackingLayout::kContiguous:
            return "contiguous";
        case PackingLayout::kColumn:
            return "column";
        case PackingLayout::kDiagonal:
            return "diagonal";
        case PackingLayout::kInterleaved:
            return "interleaved";
    }
    return "unknown";
}

CryptoProfile MakePaperCompatProfile() {
    CryptoProfile profile;
    profile.profile_id = "paper_compat";
    profile.security_claim = "none";
    profile.warning =
        "Research reproduction parameters only. Do not claim 128-bit security.";
    profile.ring_dimension = 65536;
    profile.slot_count = 32768;
    profile.scaling_modulus_bits = 46;
    profile.first_modulus_bits = 51;
    profile.multiplicative_depth = 4;
    profile.levels_available_after_bootstrap = 0;
    profile.sparse_secret_hamming_weight = 192;
    profile.num_large_digits = 3;
    profile.bootstrap_slots = profile.slot_count;
    profile.secret_key_distribution = lbcrypto::SPARSE_TERNARY;
    profile.security_level = lbcrypto::HEStd_NotSet;
    profile.key_switch_technique = lbcrypto::HYBRID;
    profile.scaling_technique = lbcrypto::FLEXIBLEAUTO;
    profile.bootstrap_enabled = false;
    profile.bootstrap_level_budget = {4, 4};
    profile.bootstrap_bsgs_dim = {0, 0};
    return profile;
}

void ConfigureBootstrap(
    CryptoProfile& profile,
    uint32_t bootstrap_slots,
    uint32_t levels_available_after_bootstrap) {
    if (bootstrap_slots == 0 || bootstrap_slots > profile.slot_count) {
        throw std::invalid_argument("bootstrap slots must be in [1, slot_count]");
    }
    profile.bootstrap_enabled = true;
    profile.bootstrap_slots = bootstrap_slots;
    profile.levels_available_after_bootstrap = levels_available_after_bootstrap;
    profile.multiplicative_depth =
        levels_available_after_bootstrap +
        lbcrypto::FHECKKSRNS::GetBootstrapDepth(
            profile.bootstrap_level_budget,
            profile.secret_key_distribution);
}

void ValidateCryptoProfile(const CryptoProfile& profile) {
    if (profile.profile_id != "paper_compat") {
        throw std::invalid_argument("only the paper_compat profile is supported");
    }
    if (profile.security_claim != "none" ||
        profile.security_level != lbcrypto::HEStd_NotSet) {
        throw std::invalid_argument(
            "paper_compat must not carry a standard security claim");
    }
    if (profile.ring_dimension != 65536 || profile.slot_count != 32768) {
        throw std::invalid_argument(
            "paper_compat requires ring_dimension=65536 and slot_count=32768");
    }
    if (profile.scaling_modulus_bits != 46 ||
        profile.first_modulus_bits != 51) {
        throw std::invalid_argument(
            "paper_compat requires the frozen 46/51-bit modulus sizes");
    }
    if (profile.secret_key_distribution != lbcrypto::SPARSE_TERNARY ||
        profile.sparse_secret_hamming_weight != 192) {
        throw std::invalid_argument(
            "paper_compat requires OpenFHE SPARSE_TERNARY (h=192)");
    }
    if (profile.multiplicative_depth == 0) {
        throw std::invalid_argument("multiplicative depth must be positive");
    }
    if (profile.bootstrap_enabled) {
        if (profile.bootstrap_level_budget.size() != 2 ||
            profile.bootstrap_bsgs_dim.size() != 2) {
            throw std::invalid_argument(
                "bootstrap level budget and BSGS dimension must have two entries");
        }
        if (profile.bootstrap_slots == 0 ||
            profile.bootstrap_slots > profile.slot_count) {
            throw std::invalid_argument("invalid bootstrap slot count");
        }
    }
}

void PrintSecurityDisclosure(
    const CryptoProfile& profile,
    std::ostream& output) {
    output << "profile=" << profile.profile_id
           << " security_claim=" << profile.security_claim
           << " warning=\"" << profile.warning << "\"\n";
}

lbcrypto::CryptoContext<lbcrypto::DCRTPoly> MakeCryptoContext(
    const CryptoProfile& profile) {
    ValidateCryptoProfile(profile);

    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(profile.secret_key_distribution);
    parameters.SetSecurityLevel(profile.security_level);
    parameters.SetRingDim(profile.ring_dimension);
    parameters.SetKeySwitchTechnique(profile.key_switch_technique);
    parameters.SetNumLargeDigits(profile.num_large_digits);
    parameters.SetScalingTechnique(profile.scaling_technique);
    parameters.SetScalingModSize(profile.scaling_modulus_bits);
    parameters.SetFirstModSize(profile.first_modulus_bits);
    parameters.SetMultiplicativeDepth(profile.multiplicative_depth);
    parameters.SetBatchSize(profile.slot_count);

    auto context = lbcrypto::GenCryptoContext(parameters);
    context->Enable(lbcrypto::PKE);
    context->Enable(lbcrypto::KEYSWITCH);
    context->Enable(lbcrypto::LEVELEDSHE);
    context->Enable(lbcrypto::ADVANCEDSHE);
    context->Enable(lbcrypto::FHE);

    if (profile.bootstrap_enabled) {
        context->EvalBootstrapSetup(
            profile.bootstrap_level_budget,
            profile.bootstrap_bsgs_dim,
            profile.bootstrap_slots);
    }
    return context;
}

}  // namespace moai::openfhe
