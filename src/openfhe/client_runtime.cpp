#include "moai/openfhe/client_runtime.hpp"

#include "moai/openfhe/context_factory.hpp"
#include "moai/openfhe/evaluation_key_registry.hpp"

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <streambuf>
#include <stdexcept>
#include <string>
#include <utility>

namespace moai::openfhe {
namespace {

using ContextImpl = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>;

class CountingStreamBuffer final : public std::streambuf {
public:
    [[nodiscard]] uint64_t bytes_written() const noexcept {
        return bytes_written_;
    }

protected:
    std::streamsize xsputn(
        const char_type*,
        std::streamsize count) override {
        if (count < 0) {
            throw std::overflow_error(
                "OpenFHE serializer requested a negative byte count");
        }
        AddBytes(static_cast<uint64_t>(count));
        return count;
    }

    int_type overflow(int_type character) override {
        if (traits_type::eq_int_type(character, traits_type::eof())) {
            return traits_type::not_eof(character);
        }
        AddBytes(1);
        return character;
    }

private:
    void AddBytes(uint64_t bytes) {
        if (bytes > std::numeric_limits<uint64_t>::max() - bytes_written_) {
            throw std::overflow_error(
                "OpenFHE binary archive byte count overflowed uint64");
        }
        bytes_written_ += bytes;
    }

    uint64_t bytes_written_{0};
};

template <typename Serializable>
uint64_t CountBinaryArchiveBytes(
    const Serializable& value,
    const char* label) {
    CountingStreamBuffer buffer;
    std::ostream output(&buffer);
    try {
        lbcrypto::Serial::Serialize(
            value,
            output,
            lbcrypto::SerType::BINARY);
    }
    catch (const std::exception& exception) {
        throw std::runtime_error(
            std::string("OpenFHE BINARY serialization failed for ") + label +
            ": " + exception.what());
    }
    const bool serialization_succeeded = output.good();
    const uint64_t bytes = buffer.bytes_written();
    if (!serialization_succeeded || bytes == 0) {
        throw std::runtime_error(
            std::string("OpenFHE BINARY serialization produced no archive for ") +
            label);
    }
    return bytes;
}

uint64_t CheckedAddBytes(
    uint64_t lhs,
    uint64_t rhs,
    const char* label) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        throw std::overflow_error(
            std::string(label) + " byte count overflowed uint64");
    }
    return lhs + rhs;
}

bool ScalesMatch(double lhs, double rhs) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs) || lhs <= 0.0 || rhs <= 0.0) {
        return false;
    }
    const double magnitude =
        std::max(1.0, std::max(std::abs(lhs), std::abs(rhs)));
    return std::abs(lhs - rhs) <= 1e-12 * magnitude;
}

void RefreshPackingMetadata(CipherTensor& tensor) {
    if (tensor.ciphertexts.empty()) {
        return;
    }
    tensor.packing.level = tensor.ciphertexts.front()->GetLevel();
    tensor.packing.noise_scale_degree =
        tensor.ciphertexts.front()->GetNoiseScaleDeg();
    tensor.packing.scaling_factor =
        tensor.ciphertexts.front()->GetScalingFactor();
    for (const auto& ciphertext : tensor.ciphertexts) {
        if (ciphertext->GetLevel() != tensor.packing.level ||
            ciphertext->GetNoiseScaleDeg() != tensor.packing.noise_scale_degree ||
            !ScalesMatch(
                ciphertext->GetScalingFactor(),
                tensor.packing.scaling_factor)) {
            throw std::logic_error(
                "encrypted CipherTensor members do not share scale metadata");
        }
    }
}

void ClearEvaluationKeysForTag(const std::string& key_tag) {
    ContextImpl::ClearEvalMultKeys(key_tag);
    ContextImpl::ClearEvalAutomorphismKeys(key_tag);
}

bool RegistryContainsTag(const std::string& key_tag) {
    return ContextImpl::GetAllEvalMultKeys().count(key_tag) != 0 ||
        ContextImpl::GetAllEvalAutomorphismKeys().count(key_tag) != 0;
}

}  // namespace

ClientRuntime::ClientRuntime(CryptoProfile profile)
    : context_(MakeCryptoContext(profile)), profile_(std::move(profile)) {
    ValidateCryptoProfile(profile_);
    ValidateCryptoContextMatchesProfile(context_, profile_);
    const auto key_pair = context_->KeyGen();
    if (!key_pair.good()) {
        throw std::runtime_error("OpenFHE key generation failed");
    }
    public_key_ = key_pair.publicKey;
    private_key_ = key_pair.secretKey;
    keys_generated_ = true;
}

void ClientRuntime::GenerateEvaluationKeys(
    const std::vector<int32_t>& rotation_indices,
    bool include_bootstrap_keys) {
    if (!keys_generated_) {
        throw std::logic_error("client keys have not been generated");
    }
    if (include_bootstrap_keys) {
        if (!profile_.bootstrap_enabled) {
            throw std::invalid_argument(
                "bootstrap keys requested for a non-bootstrap profile");
        }
    }

    auto canonical_rotations = rotation_indices;
    std::sort(canonical_rotations.begin(), canonical_rotations.end());
    canonical_rotations.erase(
        std::unique(canonical_rotations.begin(), canonical_rotations.end()),
        canonical_rotations.end());

    std::unique_lock registry_lock(EvaluationKeyRegistryMutex());
    const std::string key_tag = public_key_->GetKeyTag();
    if (key_tag.empty() || private_key_->GetKeyTag() != key_tag) {
        throw std::logic_error("client public/private key tags do not match");
    }
    if (RegistryContainsTag(key_tag)) {
        throw std::logic_error(
            "evaluation-key registry already contains the client key tag");
    }

    evaluation_keys_generated_ = false;
    rotation_indices_.clear();
    multiplication_eval_keys_.clear();
    automorphism_eval_keys_.clear();
    bootstrap_required_indices_.clear();
    try {
        context_->EvalMultKeyGen(private_key_);
        multiplication_eval_keys_ =
            ContextImpl::GetEvalMultKeyVector(key_tag);
        if (multiplication_eval_keys_.empty()) {
            throw std::runtime_error(
                "OpenFHE generated an empty multiplication-key vector");
        }

        std::map<uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>
            rotation_eval_keys;
        if (!canonical_rotations.empty()) {
            context_->EvalRotateKeyGen(private_key_, canonical_rotations);
            const auto rotation_map =
                ContextImpl::GetEvalAutomorphismKeyMapPtr(key_tag);
            if (!rotation_map || rotation_map->empty()) {
                throw std::runtime_error(
                    "OpenFHE generated an empty rotation-key map");
            }
            rotation_eval_keys = *rotation_map;
            ContextImpl::ClearEvalAutomorphismKeys(key_tag);
        }

        std::map<uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>
            bootstrap_eval_keys;
        if (include_bootstrap_keys) {
            context_->EvalBootstrapKeyGen(
                private_key_,
                profile_.bootstrap_slots);
            const auto bootstrap_map =
                ContextImpl::GetEvalAutomorphismKeyMapPtr(key_tag);
            if (!bootstrap_map || bootstrap_map->empty()) {
                throw std::runtime_error(
                    "OpenFHE generated an empty bootstrap-key map");
            }
            bootstrap_eval_keys = *bootstrap_map;
            bootstrap_required_indices_.reserve(
                bootstrap_eval_keys.size());
            for (const auto& [index, key] : bootstrap_eval_keys) {
                if (!key) {
                    throw std::runtime_error(
                        "OpenFHE generated a null bootstrap key");
                }
                bootstrap_required_indices_.push_back(index);
            }
            ContextImpl::ClearEvalAutomorphismKeys(key_tag);
        }

        automorphism_eval_keys_ = std::move(rotation_eval_keys);
        for (auto& [index, key] : bootstrap_eval_keys) {
            automorphism_eval_keys_.try_emplace(index, std::move(key));
        }
        rotation_indices_ = std::move(canonical_rotations);
        ClearEvaluationKeysForTag(key_tag);
        evaluation_keys_generated_ = true;
    }
    catch (...) {
        ClearEvaluationKeysForTag(key_tag);
        rotation_indices_.clear();
        multiplication_eval_keys_.clear();
        automorphism_eval_keys_.clear();
        bootstrap_required_indices_.clear();
        throw;
    }
}

CipherTensor ClientRuntime::Encrypt(
    const std::vector<std::vector<double>>& plaintexts,
    const PackingSpec& packing) const {
    if (!keys_generated_) {
        throw std::logic_error("client keys have not been generated");
    }
    if (plaintexts.empty()) {
        throw std::invalid_argument("at least one plaintext vector is required");
    }
    if (packing.slot_count != profile_.slot_count ||
        packing.active_slots == 0 ||
        packing.active_slots > packing.slot_count ||
        packing.batch_lanes == 0 ||
        packing.encoded_slots < packing.active_slots ||
        packing.encoded_slots > packing.slot_count ||
        !std::isfinite(packing.scaling_factor) ||
        packing.scaling_factor <= 0.0) {
        throw std::invalid_argument("packing slot contract does not match the profile");
    }
    if (packing.logical_shape.size() != 2 ||
        packing.logical_shape[0] == 0 ||
        packing.logical_shape[1] == 0 ||
        packing.logical_shape[0] * packing.batch_lanes >
            packing.active_slots ||
        (packing.layout != PackingLayout::kDiagonal &&
         packing.logical_shape[1] != plaintexts.size())) {
        throw std::invalid_argument(
            "logical shape must describe the encrypted row and feature counts");
    }

    CipherTensor result;
    result.packing = packing;
    result.ciphertexts.reserve(plaintexts.size());
    for (const auto& values : plaintexts) {
        if (values.size() != packing.active_slots) {
            throw std::invalid_argument(
                "plaintext length must equal PackingSpec.active_slots");
        }
        auto plaintext = context_->MakeCKKSPackedPlaintext(
            values,
            packing.noise_scale_degree,
            packing.level,
            nullptr,
            packing.encoded_slots);
        result.ciphertexts.push_back(
            context_->Encrypt(public_key_, plaintext));
    }
    RefreshPackingMetadata(result);
    return result;
}

std::vector<std::vector<double>> ClientRuntime::Decrypt(
    const CipherTensor& tensor) const {
    if (tensor.empty()) {
        throw std::invalid_argument("cannot decrypt an empty CipherTensor");
    }
    std::vector<std::vector<double>> decoded;
    decoded.reserve(tensor.ciphertexts.size());
    for (const auto& ciphertext : tensor.ciphertexts) {
        lbcrypto::Plaintext plaintext;
        const auto status = context_->Decrypt(
            private_key_,
            ciphertext,
            &plaintext);
        if (!status.isValid) {
            throw std::runtime_error("OpenFHE decryption reported an invalid result");
        }
        plaintext->SetLength(tensor.packing.active_slots);
        auto values = plaintext->GetRealPackedValue();
        values.resize(tensor.packing.active_slots);
        decoded.push_back(std::move(values));
    }
    return decoded;
}

SerializedKeySizeMetrics ClientRuntime::MeasureSerializedKeySizes() const {
    if (!keys_generated_ || !public_key_ || !private_key_) {
        throw std::logic_error(
            "client keys must be generated before measuring serialized sizes");
    }
    if (!evaluation_keys_generated_ || multiplication_eval_keys_.empty() ||
        automorphism_eval_keys_.empty()) {
        throw std::logic_error(
            "multiplication and automorphism evaluation keys must be generated "
            "before measuring serialized sizes");
    }
    const std::string key_tag = public_key_->GetKeyTag();
    if (key_tag.empty() || private_key_->GetKeyTag() != key_tag) {
        throw std::logic_error(
            "client public/private key tags do not match for serialization");
    }

    // These container shapes match OpenFHE's BINARY evaluation-key archive
    // contract without inserting the client's keys into process-global maps.
    const std::map<
        std::string,
        std::vector<lbcrypto::EvalKey<lbcrypto::DCRTPoly>>>
        multiplication_archive{{key_tag, multiplication_eval_keys_}};
    const std::map<
        std::string,
        std::shared_ptr<std::map<
            uint32_t,
            lbcrypto::EvalKey<lbcrypto::DCRTPoly>>>>
        automorphism_archive{{
            key_tag,
            std::make_shared<std::map<
                uint32_t,
                lbcrypto::EvalKey<lbcrypto::DCRTPoly>>>(
                automorphism_eval_keys_)}};

    SerializedKeySizeMetrics metrics;
    metrics.context_bytes =
        CountBinaryArchiveBytes(context_, "crypto context");
    metrics.public_key_bytes =
        CountBinaryArchiveBytes(public_key_, "public key");
    metrics.private_key_bytes =
        CountBinaryArchiveBytes(private_key_, "private key");
    metrics.evaluation_multiplication_key_bytes =
        CountBinaryArchiveBytes(
            multiplication_archive,
            "multiplication evaluation keys");
    metrics.evaluation_automorphism_key_bytes =
        CountBinaryArchiveBytes(
            automorphism_archive,
            "automorphism evaluation keys");

    uint64_t server_component_sum = metrics.context_bytes;
    server_component_sum = CheckedAddBytes(
        server_component_sum,
        metrics.public_key_bytes,
        "server key bundle component sum");
    server_component_sum = CheckedAddBytes(
        server_component_sum,
        metrics.evaluation_multiplication_key_bytes,
        "server key bundle component sum");
    server_component_sum = CheckedAddBytes(
        server_component_sum,
        metrics.evaluation_automorphism_key_bytes,
        "server key bundle component sum");
    metrics.server_key_bundle_component_sum_bytes = server_component_sum;
    return metrics;
}

SerializedCipherTensorSizeMetrics
ClientRuntime::MeasureSerializedCipherTensorSizes(
    const CipherTensor& tensor) const {
    if (tensor.empty()) {
        throw std::invalid_argument(
            "cannot measure an empty CipherTensor archive");
    }
    if (tensor.ciphertexts.size() >
        std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("CipherTensor count overflowed uint64");
    }

    SerializedCipherTensorSizeMetrics metrics;
    metrics.ciphertext_count =
        static_cast<uint64_t>(tensor.ciphertexts.size());
    for (const auto& ciphertext : tensor.ciphertexts) {
        if (!ciphertext) {
            throw std::invalid_argument(
                "cannot serialize a null CipherTensor ciphertext");
        }
        const uint64_t ciphertext_bytes =
            CountBinaryArchiveBytes(ciphertext, "ciphertext");
        metrics.ciphertext_component_sum_bytes = CheckedAddBytes(
            metrics.ciphertext_component_sum_bytes,
            ciphertext_bytes,
            "CipherTensor component sum");
    }
    return metrics;
}

ServerKeyBundle ClientRuntime::ExportServerKeyBundle() const {
    if (!keys_generated_ || !evaluation_keys_generated_ ||
        multiplication_eval_keys_.empty()) {
        throw std::logic_error(
            "evaluation keys must be generated before exporting the server bundle");
    }
    ServerKeyBundle bundle;
    bundle.context = context_;
    bundle.public_key = public_key_;
    bundle.key_tag = public_key_->GetKeyTag();
    bundle.profile_parameter_sha256 = profile_.parameter_sha256;
    bundle.rotation_indices = rotation_indices_;
    bundle.multiplication_eval_keys = multiplication_eval_keys_;
    bundle.automorphism_eval_keys = automorphism_eval_keys_;
    bundle.bootstrap_required_indices = bootstrap_required_indices_;
    return bundle;
}

}  // namespace moai::openfhe
