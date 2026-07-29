#include "moai/openfhe/client_runtime.hpp"
#include "moai/openfhe/context_factory.hpp"
#include "moai/openfhe/server_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double MaxAbsoluteError(
    const std::vector<double>& actual,
    const std::vector<double>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("vector size mismatch");
    }
    double maximum = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) {
            throw std::runtime_error("decoded output is not finite");
        }
        maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
    }
    return maximum;
}

void RequireAtMost(double error, double threshold, const std::string& label) {
    if (error > threshold) {
        throw std::runtime_error(
            label + " error " + std::to_string(error) +
            " exceeds " + std::to_string(threshold));
    }
}

}  // namespace

int main() {
    try {
        auto profile = moai::openfhe::MakePaperCompatProfile();
        moai::openfhe::PrintSecurityDisclosure(profile, std::cout);
        const auto feature_profile =
            moai::openfhe::MakePaperCompatFeaturePackedProfile();
        moai::openfhe::ValidateCryptoProfile(feature_profile);
        if (feature_profile.parameter_sha256 !=
                "94f30e628e21f02146ce7ed9820194eabba3820f6e1e17176a31f8c5acf8b0be" ||
            feature_profile.scaling_modulus_bits != 50 ||
            feature_profile.first_modulus_bits != 55 ||
            feature_profile.bootstrap_slots != 1024 ||
            feature_profile.levels_available_after_bootstrap != 28 ||
            feature_profile.bootstrap_iterations != 2 ||
            feature_profile.bootstrap_precision != 14 ||
            feature_profile.bootstrap_level_budget !=
                std::vector<uint32_t>{4, 4} ||
            feature_profile.bootstrap_bsgs_dim !=
                std::vector<uint32_t>{0, 0} ||
            feature_profile.bootstrap_correction_factor != 0 ||
            feature_profile.bootstrap_slots_to_coefficients_first ||
            feature_profile.multiplicative_depth != 47) {
            throw std::runtime_error(
                "feature-packed paper_compat profile contract drifted");
        }
        const auto require_feature_profile_rejected =
            [](auto mutated, const char* label) {
                mutated.parameter_sha256 =
                    moai::openfhe::ComputeCryptoProfileParameterSha256(mutated);
                bool rejected = false;
                try {
                    moai::openfhe::ValidateCryptoProfile(mutated);
                }
                catch (const std::invalid_argument&) {
                    rejected = true;
                }
                if (!rejected) {
                    throw std::runtime_error(
                        std::string("feature-packed profile accepted mutated ") +
                        label);
                }
            };
        auto mutated_feature_profile = feature_profile;
        mutated_feature_profile.bootstrap_level_budget = {3, 4};
        require_feature_profile_rejected(
            mutated_feature_profile,
            "bootstrap level budget after rehash");
        mutated_feature_profile = feature_profile;
        mutated_feature_profile.bootstrap_bsgs_dim = {1, 0};
        require_feature_profile_rejected(
            mutated_feature_profile,
            "bootstrap BSGS dimension after rehash");
        mutated_feature_profile = feature_profile;
        mutated_feature_profile.bootstrap_correction_factor = 1;
        require_feature_profile_rejected(
            mutated_feature_profile,
            "bootstrap correction factor after rehash");
        mutated_feature_profile = feature_profile;
        mutated_feature_profile.bootstrap_slots_to_coefficients_first = true;
        require_feature_profile_rejected(
            mutated_feature_profile,
            "bootstrap slots-to-coefficients order after rehash");
        auto missing_warning_profile = profile;
        missing_warning_profile.warning.clear();
        bool missing_warning_rejected = false;
        try {
            static_cast<void>(
                moai::openfhe::MakeCryptoContext(missing_warning_profile));
        }
        catch (const std::invalid_argument&) {
            missing_warning_rejected = true;
        }
        if (!missing_warning_rejected) {
            throw std::runtime_error(
                "paper_compat accepted a missing security warning");
        }
        moai::openfhe::PackingSpec packing;
        packing.layout = moai::openfhe::PackingLayout::kContiguous;
        packing.logical_shape = {8, 1};
        packing.batch_lanes = 1;
        packing.slot_count = profile.slot_count;
        packing.encoded_slots = profile.slot_count;
        packing.active_slots = 8;
        packing.level = 0;
        packing.noise_scale_degree = 1;
        packing.scaling_factor =
            std::ldexp(1.0, static_cast<int>(profile.scaling_modulus_bits));

        const std::vector<double> input{
            0.25, -0.5, 0.75, 1.0, -1.25, 1.5, 2.0, -2.5};
        const std::vector<double> half(input.size(), 0.5);
        const std::vector<double> double_weights(input.size(), 2.0);

        double roundtrip_error = 0.0;
        double rotation_error = 0.0;
        double ct_pt_error = 0.0;
        double ct_ct_error = 0.0;
        moai::openfhe::RunMetrics metrics;

        {
            moai::openfhe::ClientRuntime client(profile);
            client.GenerateEvaluationKeys({1}, false);
            auto mismatched_profile = profile;
            moai::openfhe::ConfigureBootstrap(
                mismatched_profile,
                8,
                2);
            bool mismatched_bundle_profile_rejected = false;
            try {
                static_cast<void>(moai::openfhe::ServerRuntime(
                    client.ExportServerKeyBundle(),
                    mismatched_profile));
            }
            catch (const std::invalid_argument&) {
                mismatched_bundle_profile_rejected = true;
            }
            if (!mismatched_bundle_profile_rejected) {
                throw std::runtime_error(
                    "server accepted a key bundle from a different CryptoProfile");
            }
            moai::openfhe::ServerRuntime server(client.ExportServerKeyBundle(), profile);

            auto mismatched_shape = packing;
            mismatched_shape.logical_shape = {8, 2};
            bool mismatched_shape_rejected = false;
            try {
                static_cast<void>(client.Encrypt({input}, mismatched_shape));
            }
            catch (const std::invalid_argument&) {
                mismatched_shape_rejected = true;
            }
            if (!mismatched_shape_rejected) {
                throw std::runtime_error(
                    "client accepted a logical feature count that differs from "
                    "the ciphertext count");
            }

            const auto encrypted = client.Encrypt({input}, packing);
            auto inconsistent_multiply_rhs = encrypted;
            inconsistent_multiply_rhs.packing.logical_shape = {7, 1};
            bool inconsistent_multiply_rejected = false;
            try {
                static_cast<void>(server.Multiply(
                    encrypted,
                    inconsistent_multiply_rhs));
            }
            catch (const std::invalid_argument&) {
                inconsistent_multiply_rejected = true;
            }
            if (!inconsistent_multiply_rejected) {
                throw std::runtime_error(
                    "server accepted inconsistent logical multiplication metadata");
            }
            roundtrip_error =
                MaxAbsoluteError(client.Decrypt(encrypted).front(), input);
            RequireAtMost(roundtrip_error, 1e-6, "roundtrip");

            const auto rotated = server.Rotate(encrypted, 1);
            const std::vector<double> rotation_expected{
                -0.5, 0.75, 1.0, -1.25, 1.5, 2.0, -2.5, 0.0};
            rotation_error = MaxAbsoluteError(
                client.Decrypt(rotated).front(),
                rotation_expected);
            RequireAtMost(rotation_error, 1e-6, "rotation");

            const auto encoded_weights =
                server.EncodeModelVector(double_weights, encrypted.packing);
            const auto ct_pt =
                server.MultiplyPlain(encrypted, {encoded_weights});
            auto ct_pt_expected = input;
            for (double& value : ct_pt_expected) {
                value *= 2.0;
            }
            ct_pt_error =
                MaxAbsoluteError(client.Decrypt(ct_pt).front(), ct_pt_expected);
            RequireAtMost(ct_pt_error, 1e-5, "ciphertext-plaintext multiply");

            const auto encrypted_half = client.Encrypt({half}, packing);
            const auto ct_ct_raw = server.Multiply(encrypted, encrypted_half);
            const auto ct_ct = server.Rescale(ct_ct_raw);
            auto ct_ct_expected = input;
            for (double& value : ct_ct_expected) {
                value *= 0.5;
            }
            ct_ct_error =
                MaxAbsoluteError(client.Decrypt(ct_ct).front(), ct_ct_expected);
            RequireAtMost(ct_ct_error, 1e-4, "ciphertext-ciphertext multiply");

            metrics = server.metrics();
            if (metrics.rotations != 1 ||
                metrics.ct_pt_multiplications != 1 ||
                metrics.ct_ct_multiplications != 1 ||
                metrics.rescale_operations != 1 ||
                metrics.bootstraps != 0) {
                throw std::runtime_error("unexpected server operation counters");
            }
        }

        std::cout << "{\"test\":\"openfhe_fast_smoke\","
                  << "\"profile\":\"paper_compat\","
                  << "\"security_claim\":\"none\","
                  << "\"roundtrip_max_abs\":" << roundtrip_error << ","
                  << "\"rotation_max_abs\":" << rotation_error << ","
                  << "\"ct_pt_max_abs\":" << ct_pt_error << ","
                  << "\"ct_ct_max_abs\":" << ct_ct_error << ","
                  << "\"rotations\":" << metrics.rotations << ","
                  << "\"ct_pt\":" << metrics.ct_pt_multiplications << ","
                  << "\"ct_ct\":" << metrics.ct_ct_multiplications << ","
                  << "\"rescale\":" << metrics.rescale_operations << "}\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "openfhe_fast_smoke failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
