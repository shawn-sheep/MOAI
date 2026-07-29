#include "moai/openfhe/server_runtime.hpp"

#include "moai/openfhe/context_factory.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace moai::openfhe {
namespace {

void RefreshPackingMetadata(CipherTensor& tensor) {
    if (tensor.ciphertexts.empty()) {
        return;
    }
    tensor.packing.level = tensor.ciphertexts.front()->GetLevel();
    tensor.packing.noise_scale_degree =
        tensor.ciphertexts.front()->GetNoiseScaleDeg();
}

std::size_t BroadcastIndex(std::size_t index, std::size_t size) {
    return size == 1 ? 0 : index;
}

}  // namespace

ServerRuntime::ServerRuntime(
    ServerKeyBundle key_bundle,
    CryptoProfile profile)
    : key_bundle_(std::move(key_bundle)),
      context_(key_bundle_.context),
      profile_(std::move(profile)) {
    if (!context_) {
        throw std::invalid_argument("server context must not be null");
    }
    if (!key_bundle_.public_key ||
        key_bundle_.key_tag != key_bundle_.public_key->GetKeyTag() ||
        !key_bundle_.has_multiplication_key) {
        throw std::invalid_argument("server evaluation-key bundle is incomplete");
    }
    ValidateCryptoProfile(profile_);
    metrics_.multiplicative_depth = profile_.multiplicative_depth;
}

lbcrypto::Plaintext ServerRuntime::EncodeModelVector(
    const std::vector<double>& values,
    const PackingSpec& packing) const {
    if (values.size() != packing.active_slots ||
        packing.slot_count != profile_.slot_count ||
        packing.encoded_slots < packing.active_slots ||
        packing.encoded_slots > packing.slot_count) {
        throw std::invalid_argument("model vector does not match PackingSpec");
    }
    return context_->MakeCKKSPackedPlaintext(
        values,
        packing.noise_scale_degree,
        packing.level,
        nullptr,
        packing.encoded_slots);
}

CipherTensor ServerRuntime::Rotate(
    const CipherTensor& input,
    int32_t index) {
    if (input.empty()) {
        throw std::invalid_argument("cannot rotate an empty CipherTensor");
    }
    if (std::find(
            key_bundle_.rotation_indices.begin(),
            key_bundle_.rotation_indices.end(),
            index) == key_bundle_.rotation_indices.end()) {
        throw std::invalid_argument("requested rotation key is not in the server bundle");
    }
    CipherTensor result;
    result.packing = input.packing;
    result.ciphertexts.reserve(input.size());
    for (const auto& ciphertext : input.ciphertexts) {
        result.ciphertexts.push_back(context_->EvalRotate(ciphertext, index));
    }
    metrics_.rotations += input.size();
    RefreshPackingMetadata(result);
    metrics_.max_observed_level =
        std::max(metrics_.max_observed_level, result.packing.level);
    return result;
}

CipherTensor ServerRuntime::MultiplyPlain(
    const CipherTensor& input,
    const std::vector<lbcrypto::Plaintext>& plaintexts) {
    if (input.empty() ||
        (plaintexts.size() != 1 && plaintexts.size() != input.size())) {
        throw std::invalid_argument(
            "plaintext count must be one or match the ciphertext count");
    }
    CipherTensor result;
    result.packing = input.packing;
    result.ciphertexts.reserve(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        result.ciphertexts.push_back(
            context_->EvalMult(
                input.ciphertexts[i],
                plaintexts[BroadcastIndex(i, plaintexts.size())]));
    }
    metrics_.ct_pt_multiplications += input.size();
    RefreshPackingMetadata(result);
    metrics_.max_observed_level =
        std::max(metrics_.max_observed_level, result.packing.level);
    return result;
}

CipherTensor ServerRuntime::Multiply(
    const CipherTensor& lhs,
    const CipherTensor& rhs) {
    if (lhs.empty() || rhs.empty() ||
        (rhs.size() != 1 && rhs.size() != lhs.size())) {
        throw std::invalid_argument(
            "right ciphertext count must be one or match the left count");
    }
    if (!key_bundle_.has_multiplication_key) {
        throw std::logic_error("ciphertext multiplication key is unavailable");
    }
    CipherTensor result;
    result.packing = lhs.packing;
    result.ciphertexts.reserve(lhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result.ciphertexts.push_back(
            context_->EvalMult(
                lhs.ciphertexts[i],
                rhs.ciphertexts[BroadcastIndex(i, rhs.size())]));
    }
    metrics_.ct_ct_multiplications += lhs.size();
    RefreshPackingMetadata(result);
    metrics_.max_observed_level =
        std::max(metrics_.max_observed_level, result.packing.level);
    return result;
}

CipherTensor ServerRuntime::Add(
    const CipherTensor& lhs,
    const CipherTensor& rhs) {
    if (lhs.empty() || rhs.empty() || lhs.size() != rhs.size()) {
        throw std::invalid_argument(
            "ciphertext addition requires non-empty tensors of equal size");
    }
    if (lhs.packing.slot_count != rhs.packing.slot_count ||
        lhs.packing.batch_lanes != rhs.packing.batch_lanes) {
        throw std::invalid_argument("ciphertext packing contracts do not match");
    }
    CipherTensor result;
    result.packing = lhs.packing;
    result.ciphertexts.reserve(lhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result.ciphertexts.push_back(
            context_->EvalAdd(lhs.ciphertexts[i], rhs.ciphertexts[i]));
    }
    RefreshPackingMetadata(result);
    metrics_.max_observed_level =
        std::max(metrics_.max_observed_level, result.packing.level);
    return result;
}

CipherTensor ServerRuntime::Sum(const CipherTensor& input) {
    if (input.empty()) {
        throw std::invalid_argument("cannot sum an empty CipherTensor");
    }
    CipherTensor result;
    result.packing = input.packing;
    result.ciphertexts.push_back(input.ciphertexts.front());
    for (std::size_t i = 1; i < input.size(); ++i) {
        result.ciphertexts.front() = context_->EvalAdd(
            result.ciphertexts.front(), input.ciphertexts[i]);
    }
    RefreshPackingMetadata(result);
    metrics_.max_observed_level =
        std::max(metrics_.max_observed_level, result.packing.level);
    return result;
}

CipherTensor ServerRuntime::Rescale(const CipherTensor& input) {
    if (input.empty()) {
        throw std::invalid_argument("cannot rescale an empty CipherTensor");
    }
    CipherTensor result;
    result.packing = input.packing;
    result.ciphertexts.reserve(input.size());
    for (const auto& ciphertext : input.ciphertexts) {
        result.ciphertexts.push_back(context_->Rescale(ciphertext));
    }
    metrics_.rescale_operations += input.size();
    RefreshPackingMetadata(result);
    metrics_.max_observed_level =
        std::max(metrics_.max_observed_level, result.packing.level);
    return result;
}

void ServerRuntime::PrepareBootstrap() {
    if (!profile_.bootstrap_enabled) {
        throw std::logic_error("bootstrap is not enabled in this profile");
    }
    context_->EvalBootstrapPrecompute(profile_.bootstrap_slots);
    bootstrap_prepared_ = true;
}

CipherTensor ServerRuntime::Bootstrap(const CipherTensor& input) {
    if (!profile_.bootstrap_enabled) {
        throw std::logic_error("bootstrap is not enabled in this profile");
    }
    if (!key_bundle_.has_bootstrap_key) {
        throw std::logic_error("bootstrap key is unavailable");
    }
    if (input.empty()) {
        throw std::invalid_argument("cannot bootstrap an empty CipherTensor");
    }
    if (!bootstrap_prepared_) {
        PrepareBootstrap();
    }
    CipherTensor result;
    result.packing = input.packing;
    result.ciphertexts.reserve(input.size());
    for (const auto& ciphertext : input.ciphertexts) {
        result.ciphertexts.push_back(context_->EvalBootstrap(ciphertext));
    }
    metrics_.bootstraps += input.size();
    RefreshPackingMetadata(result);
    metrics_.max_observed_level =
        std::max(metrics_.max_observed_level, result.packing.level);
    return result;
}

}  // namespace moai::openfhe
