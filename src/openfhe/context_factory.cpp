#include "moai/openfhe/context_factory.hpp"

#include "utils/hashutil.h"

#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace moai::openfhe {
namespace {

std::string UintVectorJson(const std::vector<uint32_t>& values) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << values[index];
    }
    output << ']';
    return output.str();
}

void RefreshParameterHash(CryptoProfile& profile) {
    profile.parameter_sha256 =
        ComputeCryptoProfileParameterSha256(profile);
}

}  // namespace

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
    profile.bootstrap_iterations = 1;
    profile.bootstrap_precision = 0;
    profile.bootstrap_correction_factor = 0;
    profile.secret_key_distribution = lbcrypto::SPARSE_TERNARY;
    profile.security_level = lbcrypto::HEStd_NotSet;
    profile.key_switch_technique = lbcrypto::HYBRID;
    profile.scaling_technique = lbcrypto::FLEXIBLEAUTO;
    profile.bootstrap_enabled = false;
    profile.bootstrap_slots_to_coefficients_first = false;
    profile.bootstrap_level_budget = {4, 4};
    profile.bootstrap_bsgs_dim = {0, 0};
    RefreshParameterHash(profile);
    return profile;
}

std::string CanonicalCryptoProfileJson(const CryptoProfile& profile) {
    std::ostringstream output;
    output
        << "{\"bootstrap_bsgs_dim\":"
        << UintVectorJson(profile.bootstrap_bsgs_dim)
        << ",\"bootstrap_correction_factor\":"
        << profile.bootstrap_correction_factor
        << ",\"bootstrap_enabled\":"
        << (profile.bootstrap_enabled ? "true" : "false")
        << ",\"bootstrap_iterations\":" << profile.bootstrap_iterations
        << ",\"bootstrap_level_budget\":"
        << UintVectorJson(profile.bootstrap_level_budget)
        << ",\"bootstrap_precision\":" << profile.bootstrap_precision
        << ",\"bootstrap_slots\":" << profile.bootstrap_slots
        << ",\"bootstrap_slots_to_coefficients_first\":"
        << (profile.bootstrap_slots_to_coefficients_first ? "true" : "false")
        << ",\"effective_profile_schema_version\":"
        << profile.effective_profile_schema_version
        << ",\"first_modulus_bits\":" << profile.first_modulus_bits
        << ",\"key_switch_technique\":\"HYBRID\""
        << ",\"levels_available_after_bootstrap\":"
        << profile.levels_available_after_bootstrap
        << ",\"multiplicative_depth\":" << profile.multiplicative_depth
        << ",\"num_large_digits\":" << profile.num_large_digits
        << ",\"profile_id\":\"" << profile.profile_id << "\""
        << ",\"ring_dimension\":" << profile.ring_dimension
        << ",\"scaling_modulus_bits\":" << profile.scaling_modulus_bits
        << ",\"scaling_technique\":\"FLEXIBLEAUTO\""
        << ",\"secret_key_distribution\":\"SPARSE_TERNARY\""
        << ",\"security_claim\":\"" << profile.security_claim << "\""
        << ",\"security_level\":\"HEStd_NotSet\""
        << ",\"slot_count\":" << profile.slot_count
        << ",\"sparse_secret_hamming_weight\":"
        << profile.sparse_secret_hamming_weight << '}';
    return output.str();
}

std::string ComputeCryptoProfileParameterSha256(
    const CryptoProfile& profile) {
    return lbcrypto::HashUtil::HashString(
        CanonicalCryptoProfileJson(profile));
}

void ConfigureBootstrap(
    CryptoProfile& profile,
    uint32_t bootstrap_slots,
    uint32_t levels_available_after_bootstrap,
    uint32_t bootstrap_iterations,
    uint32_t bootstrap_precision,
    uint32_t bootstrap_correction_factor,
    bool bootstrap_slots_to_coefficients_first) {
    if (bootstrap_slots == 0 || bootstrap_slots > profile.slot_count) {
        throw std::invalid_argument("bootstrap slots must be in [1, slot_count]");
    }
    if (bootstrap_iterations != 1 && bootstrap_iterations != 2) {
        throw std::invalid_argument("bootstrap iterations must be one or two");
    }
    if ((bootstrap_iterations == 1 && bootstrap_precision != 0) ||
        (bootstrap_iterations == 2 &&
         (bootstrap_precision == 0 || bootstrap_precision >= 31))) {
        throw std::invalid_argument(
            "bootstrap precision must be zero for one iteration and in [1,30] "
            "for two iterations");
    }
    profile.bootstrap_enabled = true;
    profile.bootstrap_slots = bootstrap_slots;
    profile.levels_available_after_bootstrap = levels_available_after_bootstrap;
    profile.bootstrap_iterations = bootstrap_iterations;
    profile.bootstrap_precision = bootstrap_precision;
    profile.bootstrap_correction_factor = bootstrap_correction_factor;
    profile.bootstrap_slots_to_coefficients_first =
        bootstrap_slots_to_coefficients_first;
    const uint64_t multiplicative_depth =
        static_cast<uint64_t>(levels_available_after_bootstrap) +
        lbcrypto::FHECKKSRNS::GetBootstrapDepth(
            profile.bootstrap_level_budget,
            profile.secret_key_distribution) +
        (bootstrap_iterations - 1);
    if (multiplicative_depth > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("bootstrap multiplicative depth overflows uint32");
    }
    profile.multiplicative_depth =
        static_cast<uint32_t>(multiplicative_depth);
    RefreshParameterHash(profile);
}

CryptoProfile MakePaperCompatFeaturePackedProfile() {
    auto profile = MakePaperCompatProfile();
    profile.scaling_modulus_bits = 50;
    profile.first_modulus_bits = 55;
    ConfigureBootstrap(
        profile,
        1024,
        28,
        2,
        14,
        0,
        false);
    return profile;
}

void ValidateCryptoProfile(const CryptoProfile& profile) {
    if (profile.profile_id != "paper_compat") {
        throw std::invalid_argument("only the paper_compat profile is supported");
    }
    if (profile.warning !=
        "Research reproduction parameters only. Do not claim 128-bit security.") {
        throw std::invalid_argument(
            "paper_compat must surface the frozen security warning");
    }
    if (profile.effective_profile_schema_version != 1) {
        throw std::invalid_argument(
            "paper_compat effective profile schema version must be one");
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
    const bool base_modulus_contract =
        profile.scaling_modulus_bits == 46 &&
        profile.first_modulus_bits == 51;
    const bool feature_modulus_contract =
        profile.scaling_modulus_bits == 50 &&
        profile.first_modulus_bits == 55 &&
        profile.bootstrap_enabled &&
        profile.bootstrap_slots == 1024 &&
        profile.levels_available_after_bootstrap == 28 &&
        profile.bootstrap_iterations == 2 &&
        profile.bootstrap_precision == 14 &&
        profile.bootstrap_level_budget == std::vector<uint32_t>{4, 4} &&
        profile.bootstrap_bsgs_dim == std::vector<uint32_t>{0, 0} &&
        profile.bootstrap_correction_factor == 0 &&
        !profile.bootstrap_slots_to_coefficients_first &&
        profile.multiplicative_depth == 47;
    if (!base_modulus_contract && !feature_modulus_contract) {
        throw std::invalid_argument(
            "paper_compat requires the frozen base or feature-packed modulus contract");
    }
    if (profile.secret_key_distribution != lbcrypto::SPARSE_TERNARY ||
        profile.sparse_secret_hamming_weight != 192) {
        throw std::invalid_argument(
            "paper_compat requires OpenFHE SPARSE_TERNARY (h=192)");
    }
    if (profile.multiplicative_depth == 0) {
        throw std::invalid_argument("multiplicative depth must be positive");
    }
    if (profile.key_switch_technique != lbcrypto::HYBRID ||
        profile.scaling_technique != lbcrypto::FLEXIBLEAUTO ||
        profile.num_large_digits != 3) {
        throw std::invalid_argument(
            "paper_compat requires HYBRID/FLEXIBLEAUTO with three large digits");
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
        if (profile.bootstrap_iterations != 1 &&
            profile.bootstrap_iterations != 2) {
            throw std::invalid_argument("invalid bootstrap iteration count");
        }
        if ((profile.bootstrap_iterations == 1 &&
             profile.bootstrap_precision != 0) ||
            (profile.bootstrap_iterations == 2 &&
             (profile.bootstrap_precision == 0 ||
              profile.bootstrap_precision >= 31))) {
            throw std::invalid_argument("invalid bootstrap precision");
        }
    }
    if (profile.parameter_sha256.size() != 64 ||
        profile.parameter_sha256 !=
            ComputeCryptoProfileParameterSha256(profile)) {
        throw std::invalid_argument(
            "CryptoProfile effective parameter SHA-256 is missing or stale");
    }
}

void ValidateCryptoContextMatchesProfile(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
    const CryptoProfile& profile) {
    ValidateCryptoProfile(profile);
    if (!context) {
        throw std::invalid_argument("OpenFHE context must not be null");
    }
    const auto parameters =
        std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(
            context->GetCryptoParameters());
    if (!parameters) {
        throw std::invalid_argument(
            "paper_compat requires an OpenFHE CKKS-RNS context");
    }
    const auto element_parameters = parameters->GetElementParams();
    if (!element_parameters || element_parameters->GetParams().empty()) {
        throw std::invalid_argument(
            "OpenFHE context has no CKKS modulus chain");
    }
    const uint32_t observed_first_modulus_bits =
        element_parameters->GetParams().front()->GetModulus().GetMSB();
    if (context->GetRingDimension() != profile.ring_dimension ||
        parameters->GetBatchSize() != profile.slot_count ||
        parameters->GetPlaintextModulus() != profile.scaling_modulus_bits ||
        observed_first_modulus_bits != profile.first_modulus_bits ||
        parameters->GetMultiplicativeDepth() != profile.multiplicative_depth ||
        parameters->GetSecretKeyDist() != profile.secret_key_distribution ||
        parameters->GetStdLevel() != profile.security_level ||
        parameters->GetKeySwitchTechnique() != profile.key_switch_technique ||
        parameters->GetScalingTechnique() != profile.scaling_technique) {
        throw std::invalid_argument(
            "OpenFHE context parameters do not match the effective CryptoProfile");
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
            profile.bootstrap_slots,
            profile.bootstrap_correction_factor,
            true,
            profile.bootstrap_slots_to_coefficients_first);
    }
    return context;
}

}  // namespace moai::openfhe
