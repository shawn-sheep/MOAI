#include "moai/openfhe/client_runtime.hpp"
#include "moai/openfhe/context_factory.hpp"
#include "moai/openfhe/evaluation_key_registry.hpp"
#include "moai/openfhe/server_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace moai::openfhe {

class ServerKeyBundleTestAccess {
public:
    static const std::string& KeyTag(const ServerKeyBundle& bundle) {
        return bundle.key_tag;
    }

    static bool SamePointerContainers(
        const ServerKeyBundle& lhs,
        const ServerKeyBundle& rhs) {
        return lhs.multiplication_eval_keys ==
                rhs.multiplication_eval_keys &&
            lhs.automorphism_eval_keys == rhs.automorphism_eval_keys &&
            lhs.rotation_indices == rhs.rotation_indices &&
            lhs.bootstrap_required_indices ==
                rhs.bootstrap_required_indices;
    }

    static void RemoveMultiplicationKeys(ServerKeyBundle& bundle) {
        bundle.multiplication_eval_keys.clear();
    }

    static uint32_t RotationAutomorphismIndex(
        const ServerKeyBundle& bundle,
        int32_t rotation) {
        return bundle.context->FindAutomorphismIndex(
            static_cast<uint32_t>(rotation));
    }

    static void RemoveAutomorphismKey(
        ServerKeyBundle& bundle,
        uint32_t index) {
        bundle.automorphism_eval_keys.erase(index);
    }

    static uint32_t BootstrapIndexOtherThan(
        const ServerKeyBundle& bundle,
        uint32_t excluded) {
        const auto iterator = std::find_if(
            bundle.bootstrap_required_indices.begin(),
            bundle.bootstrap_required_indices.end(),
            [excluded](uint32_t index) { return index != excluded; });
        if (iterator == bundle.bootstrap_required_indices.end()) {
            throw std::runtime_error(
                "bootstrap test needs an index distinct from rotation");
        }
        return *iterator;
    }

    static void AddUnexpectedAutomorphismEntry(ServerKeyBundle& bundle) {
        if (bundle.automorphism_eval_keys.empty()) {
            throw std::runtime_error("automorphism-key map is unexpectedly empty");
        }
        uint32_t index = 2;
        while (bundle.automorphism_eval_keys.count(index) != 0) {
            if (index == std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error(
                    "could not choose an unused automorphism index");
            }
            ++index;
        }
        bundle.automorphism_eval_keys.emplace(
            index,
            bundle.automorphism_eval_keys.begin()->second);
    }
};

}  // namespace moai::openfhe

namespace {

using ContextImpl = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>;

void RequireRegistryTagAbsent(const std::string& key_tag) {
    std::shared_lock registry_lock(
        moai::openfhe::EvaluationKeyRegistryMutex());
    if (ContextImpl::GetAllEvalMultKeys().count(key_tag) != 0 ||
        ContextImpl::GetAllEvalAutomorphismKeys().count(key_tag) != 0) {
        throw std::runtime_error(
            "client export left evaluation keys in OpenFHE static maps");
    }
}

void RequireRegistryTagPresent(const std::string& key_tag) {
    std::shared_lock registry_lock(
        moai::openfhe::EvaluationKeyRegistryMutex());
    if (ContextImpl::GetAllEvalMultKeys().count(key_tag) != 1 ||
        ContextImpl::GetAllEvalAutomorphismKeys().count(key_tag) != 1) {
        throw std::runtime_error(
            "server evaluation-key installation did not persist");
    }
}

template <typename Callable>
void RequireConstructionRejected(Callable&& callable, const std::string& label) {
    try {
        callable();
    }
    catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(label + " was not rejected before homomorphic work");
}

double MaxAbsoluteError(
    const std::vector<double>& actual,
    const std::vector<double>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("evaluation-key smoke vector size mismatch");
    }
    double maximum = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index])) {
            throw std::runtime_error(
                "evaluation-key smoke decoded a non-finite value");
        }
        maximum = std::max(
            maximum,
            std::abs(actual[index] - expected[index]));
    }
    return maximum;
}

}  // namespace

int main() {
    try {
        auto profile = moai::openfhe::MakePaperCompatProfile();
        profile.bootstrap_level_budget = {2, 2};
        profile.bootstrap_bsgs_dim = {0, 0};
        constexpr uint32_t kSlots = 8;
        moai::openfhe::ConfigureBootstrap(profile, kSlots, 2);

        moai::openfhe::ClientRuntime client(profile);
        client.GenerateEvaluationKeys({1}, true);
        const auto bundle = client.ExportServerKeyBundle();
        const auto repeated_bundle = client.ExportServerKeyBundle();
        const std::string key_tag =
            moai::openfhe::ServerKeyBundleTestAccess::KeyTag(bundle);
        if (!moai::openfhe::ServerKeyBundleTestAccess::SamePointerContainers(
                bundle,
                repeated_bundle)) {
            throw std::runtime_error(
                "repeated export changed evaluation-key pointer containers");
        }
        RequireRegistryTagAbsent(key_tag);

        auto missing_multiplication = bundle;
        moai::openfhe::ServerKeyBundleTestAccess::RemoveMultiplicationKeys(
            missing_multiplication);
        RequireConstructionRejected(
            [&] {
                static_cast<void>(moai::openfhe::ServerRuntime(
                    missing_multiplication,
                    profile));
            },
            "missing multiplication-key bundle");
        RequireRegistryTagAbsent(key_tag);

        const uint32_t rotation_automorphism =
            moai::openfhe::ServerKeyBundleTestAccess::
                RotationAutomorphismIndex(bundle, 1);
        auto missing_rotation = bundle;
        moai::openfhe::ServerKeyBundleTestAccess::RemoveAutomorphismKey(
            missing_rotation,
            rotation_automorphism);
        RequireConstructionRejected(
            [&] {
                static_cast<void>(moai::openfhe::ServerRuntime(
                    missing_rotation,
                    profile));
            },
            "missing rotation automorphism key");
        RequireRegistryTagAbsent(key_tag);

        const uint32_t bootstrap_index =
            moai::openfhe::ServerKeyBundleTestAccess::
                BootstrapIndexOtherThan(bundle, rotation_automorphism);
        auto missing_bootstrap = bundle;
        moai::openfhe::ServerKeyBundleTestAccess::RemoveAutomorphismKey(
            missing_bootstrap,
            bootstrap_index);
        RequireConstructionRejected(
            [&] {
                static_cast<void>(moai::openfhe::ServerRuntime(
                    missing_bootstrap,
                    profile));
            },
            "missing bootstrap automorphism key");
        RequireRegistryTagAbsent(key_tag);

        moai::openfhe::PackingSpec packing;
        packing.layout = moai::openfhe::PackingLayout::kContiguous;
        packing.logical_shape = {kSlots, 1};
        packing.batch_lanes = 1;
        packing.slot_count = profile.slot_count;
        packing.active_slots = kSlots;
        packing.encoded_slots = kSlots;
        packing.level = 0;
        packing.noise_scale_degree = 1;
        packing.scaling_factor =
            std::ldexp(1.0, static_cast<int>(profile.scaling_modulus_bits));

        const std::vector<double> input{
            -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0};
        const std::vector<double> half(kSlots, 0.5);
        const auto encrypted = client.Encrypt({input}, packing);
        const auto encrypted_half = client.Encrypt({half}, packing);

        double multiplication_error = 0.0;
        double rotation_error = 0.0;
        double bootstrap_error = 0.0;
        {
            moai::openfhe::ServerRuntime server(bundle, profile);
            moai::openfhe::ServerRuntime repeated_server(
                repeated_bundle,
                profile);
            static_cast<void>(repeated_server);

            auto different_existing_container = bundle;
            moai::openfhe::ServerKeyBundleTestAccess::
                AddUnexpectedAutomorphismEntry(different_existing_container);
            RequireConstructionRejected(
                [&] {
                    static_cast<void>(moai::openfhe::ServerRuntime(
                        different_existing_container,
                        profile));
                },
                "different existing automorphism-key container");

            const auto multiplied = server.Rescale(
                server.Multiply(encrypted, encrypted_half));
            auto multiplication_expected = input;
            for (double& value : multiplication_expected) {
                value *= 0.5;
            }
            multiplication_error = MaxAbsoluteError(
                client.Decrypt(multiplied).front(),
                multiplication_expected);

            const auto rotated = server.Rotate(encrypted, 1);
            const std::vector<double> rotation_expected{
                -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, -0.75};
            rotation_error = MaxAbsoluteError(
                client.Decrypt(rotated).front(),
                rotation_expected);

            auto depleted_packing = packing;
            depleted_packing.level = profile.multiplicative_depth - 1;
            const auto depleted = client.Encrypt({input}, depleted_packing);
            const auto refreshed = server.Bootstrap(depleted);
            bootstrap_error = MaxAbsoluteError(
                client.Decrypt(refreshed).front(),
                input);
        }

        RequireRegistryTagPresent(key_tag);
        if (multiplication_error > 1e-4 || rotation_error > 1e-6 ||
            bootstrap_error > 5e-3) {
            std::cerr
                << "evaluation-key bundle errors: multiplication="
                << multiplication_error
                << " rotation=" << rotation_error
                << " bootstrap=" << bootstrap_error << '\n';
            throw std::runtime_error(
                "evaluation-key bundle homomorphic accuracy gate failed");
        }

        std::cout
            << "{\"test\":\"openfhe_evaluation_key_bundle_smoke\","
            << "\"profile\":\"paper_compat\","
            << "\"security_claim\":\"none\","
            << "\"static_maps_cleared_before_server\":true,"
            << "\"repeat_export_pointer_equal\":true,"
            << "\"missing_keys_rejected_before_work\":true,"
            << "\"server_destructor_preserved_registry\":true,"
            << "\"multiplication_max_abs\":" << multiplication_error << ','
            << "\"rotation_max_abs\":" << rotation_error << ','
            << "\"bootstrap_max_abs\":" << bootstrap_error << "}\n";
        return 0;
    }
    catch (const std::exception& exception) {
        std::cerr << "openfhe_evaluation_key_bundle_smoke failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
