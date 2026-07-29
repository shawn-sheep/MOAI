#include "moai/openfhe/client_runtime.hpp"

#include "moai/openfhe/context_factory.hpp"

#include <stdexcept>
#include <utility>

namespace moai::openfhe {

ClientRuntime::ClientRuntime(
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context,
    CryptoProfile profile)
    : context_(std::move(context)), profile_(std::move(profile)) {
    if (!context_) {
        throw std::invalid_argument("client context must not be null");
    }
    ValidateCryptoProfile(profile_);
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
    context_->EvalMultKeyGen(private_key_);
    multiplication_key_generated_ = true;
    if (!rotation_indices.empty()) {
        context_->EvalRotateKeyGen(private_key_, rotation_indices);
        rotation_indices_ = rotation_indices;
    }
    if (include_bootstrap_keys) {
        if (!profile_.bootstrap_enabled) {
            throw std::invalid_argument(
                "bootstrap keys requested for a non-bootstrap profile");
        }
        context_->EvalBootstrapKeyGen(
            private_key_,
            profile_.bootstrap_slots);
        bootstrap_key_generated_ = true;
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
        packing.encoded_slots < packing.active_slots ||
        packing.encoded_slots > packing.slot_count) {
        throw std::invalid_argument("packing slot contract does not match the profile");
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

ServerKeyBundle ClientRuntime::ExportServerKeyBundle() const {
    if (!keys_generated_ || !multiplication_key_generated_) {
        throw std::logic_error(
            "evaluation keys must be generated before exporting the server bundle");
    }
    ServerKeyBundle bundle;
    bundle.context = context_;
    bundle.public_key = public_key_;
    bundle.key_tag = public_key_->GetKeyTag();
    bundle.rotation_indices = rotation_indices_;
    bundle.has_multiplication_key = multiplication_key_generated_;
    bundle.has_bootstrap_key = bootstrap_key_generated_;
    return bundle;
}

}  // namespace moai::openfhe
