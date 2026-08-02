#include "moai/openfhe/client_runtime.hpp"
#include "moai/openfhe/context_factory.hpp"
#include "moai/openfhe/evaluation_key_registry.hpp"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename Callable>
void RequireRejected(Callable&& callable, const char* label) {
    try {
        callable();
    }
    catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(std::string(label) + " did not fail closed");
}

void RequirePositive(uint64_t value, const char* label) {
    if (value == 0) {
        throw std::runtime_error(std::string(label) + " must be positive");
    }
}

std::pair<std::size_t, std::size_t> EvaluationKeyRegistryEntryCounts() {
    std::shared_lock registry_lock(
        moai::openfhe::EvaluationKeyRegistryMutex());
    using ContextImpl =
        lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>;
    return {
        ContextImpl::GetAllEvalMultKeys().size(),
        ContextImpl::GetAllEvalAutomorphismKeys().size()};
}

}  // namespace

int main() {
    try {
        const auto profile = moai::openfhe::MakePaperCompatProfile();
        moai::openfhe::ClientRuntime client(profile);

        RequireRejected(
            [&] {
                static_cast<void>(client.MeasureSerializedKeySizes());
            },
            "key-size measurement without evaluation keys");
        RequireRejected(
            [&] {
                static_cast<void>(client.MeasureSerializedCipherTensorSizes({}));
            },
            "empty CipherTensor size measurement");

        client.GenerateEvaluationKeys({1}, false);
        const auto registry_counts_before =
            EvaluationKeyRegistryEntryCounts();
        const auto key_sizes = client.MeasureSerializedKeySizes();
        const auto repeated_key_sizes = client.MeasureSerializedKeySizes();
        if (EvaluationKeyRegistryEntryCounts() != registry_counts_before) {
            throw std::runtime_error(
                "serialized-size measurement changed the evaluation-key registry");
        }
        if (key_sizes.serialization_format !=
            moai::openfhe::kOpenFheBinaryArchiveComponentSumV1) {
            throw std::runtime_error("serialized key-size format drifted");
        }
        RequirePositive(key_sizes.context_bytes, "context bytes");
        RequirePositive(key_sizes.public_key_bytes, "public-key bytes");
        RequirePositive(key_sizes.private_key_bytes, "private-key bytes");
        RequirePositive(
            key_sizes.evaluation_multiplication_key_bytes,
            "multiplication evaluation-key bytes");
        RequirePositive(
            key_sizes.evaluation_automorphism_key_bytes,
            "automorphism evaluation-key bytes");
        RequirePositive(
            key_sizes.server_key_bundle_component_sum_bytes,
            "server key-bundle component-sum bytes");
        const uint64_t expected_server_sum =
            key_sizes.context_bytes +
            key_sizes.public_key_bytes +
            key_sizes.evaluation_multiplication_key_bytes +
            key_sizes.evaluation_automorphism_key_bytes;
        if (key_sizes.server_key_bundle_component_sum_bytes !=
            expected_server_sum) {
            throw std::runtime_error(
                "server key-bundle component sum includes the wrong fields");
        }
        if (repeated_key_sizes.serialization_format !=
                key_sizes.serialization_format ||
            repeated_key_sizes.context_bytes != key_sizes.context_bytes ||
            repeated_key_sizes.public_key_bytes != key_sizes.public_key_bytes ||
            repeated_key_sizes.private_key_bytes != key_sizes.private_key_bytes ||
            repeated_key_sizes.evaluation_multiplication_key_bytes !=
                key_sizes.evaluation_multiplication_key_bytes ||
            repeated_key_sizes.evaluation_automorphism_key_bytes !=
                key_sizes.evaluation_automorphism_key_bytes ||
            repeated_key_sizes.server_key_bundle_component_sum_bytes !=
                key_sizes.server_key_bundle_component_sum_bytes) {
            throw std::runtime_error(
                "repeated key serialization size changed");
        }

        moai::openfhe::CipherTensor null_ciphertext_tensor;
        null_ciphertext_tensor.ciphertexts.push_back(nullptr);
        RequireRejected(
            [&] {
                static_cast<void>(
                    client.MeasureSerializedCipherTensorSizes(
                        null_ciphertext_tensor));
            },
            "null CipherTensor ciphertext size measurement");

        constexpr uint32_t kActiveSlots = 8;
        moai::openfhe::PackingSpec packing;
        packing.layout = moai::openfhe::PackingLayout::kContiguous;
        packing.logical_shape = {kActiveSlots, 1};
        packing.batch_lanes = 1;
        packing.slot_count = profile.slot_count;
        packing.active_slots = kActiveSlots;
        packing.encoded_slots = kActiveSlots;
        packing.level = 0;
        packing.noise_scale_degree = 1;
        packing.scaling_factor =
            std::ldexp(1.0, static_cast<int>(profile.scaling_modulus_bits));
        const std::vector<double> input{
            -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0};
        const auto encrypted = client.Encrypt({input}, packing);
        const auto ciphertext_sizes =
            client.MeasureSerializedCipherTensorSizes(encrypted);
        if (ciphertext_sizes.serialization_format !=
                moai::openfhe::kOpenFheBinaryArchiveComponentSumV1 ||
            ciphertext_sizes.ciphertext_count != encrypted.size()) {
            throw std::runtime_error(
                "serialized CipherTensor size metadata drifted");
        }
        RequirePositive(
            ciphertext_sizes.ciphertext_component_sum_bytes,
            "ciphertext component-sum bytes");
        const auto repeated_ciphertext_sizes =
            client.MeasureSerializedCipherTensorSizes(encrypted);
        if (repeated_ciphertext_sizes.ciphertext_component_sum_bytes !=
            ciphertext_sizes.ciphertext_component_sum_bytes) {
            throw std::runtime_error(
                "repeated CipherTensor serialization size changed");
        }

        std::cout
            << "{\"test\":\"openfhe_serialized_size_smoke\","
            << "\"serialization_format\":\""
            << key_sizes.serialization_format << "\","
            << "\"context_bytes\":" << key_sizes.context_bytes << ','
            << "\"public_key_bytes\":" << key_sizes.public_key_bytes << ','
            << "\"private_key_bytes\":" << key_sizes.private_key_bytes << ','
            << "\"evaluation_multiplication_key_bytes\":"
            << key_sizes.evaluation_multiplication_key_bytes << ','
            << "\"evaluation_automorphism_key_bytes\":"
            << key_sizes.evaluation_automorphism_key_bytes << ','
            << "\"server_key_bundle_component_sum_bytes\":"
            << key_sizes.server_key_bundle_component_sum_bytes << ','
            << "\"ciphertext_count\":"
            << ciphertext_sizes.ciphertext_count << ','
            << "\"ciphertext_component_sum_bytes\":"
            << ciphertext_sizes.ciphertext_component_sum_bytes << "}\n";
        return 0;
    }
    catch (const std::exception& exception) {
        std::cerr << "openfhe_serialized_size_smoke failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
