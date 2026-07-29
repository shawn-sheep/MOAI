#include "moai/openfhe/client_runtime.hpp"
#include "moai/openfhe/context_factory.hpp"
#include "moai/openfhe/server_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr uint32_t kFeatureSlots = 1024;
constexpr std::size_t kHiddenSize = 768;
constexpr const char* kExpectedProfileSha256 =
    "94f30e628e21f02146ce7ed9820194eabba3820f6e1e17176a31f8c5acf8b0be";

struct BootstrapError {
    double active_max_abs{0.0};
    double inactive_max_abs{0.0};
};

BootstrapError MeasureError(
    const std::vector<double>& actual,
    const std::vector<double>& expected) {
    if (actual.size() != kFeatureSlots || expected.size() != kFeatureSlots) {
        throw std::runtime_error("feature bootstrap vector size mismatch");
    }
    BootstrapError result;
    for (std::size_t slot = 0; slot < kFeatureSlots; ++slot) {
        if (!std::isfinite(actual[slot])) {
            throw std::runtime_error("feature bootstrap output is not finite");
        }
        const double error = std::abs(actual[slot] - expected[slot]);
        if (slot < kHiddenSize) {
            result.active_max_abs = std::max(result.active_max_abs, error);
        }
        else {
            result.inactive_max_abs = std::max(result.inactive_max_abs, error);
        }
    }
    return result;
}

}  // namespace

int main() {
    try {
        auto profile =
            moai::openfhe::MakePaperCompatFeaturePackedProfile();
        if (profile.parameter_sha256 != kExpectedProfileSha256 ||
            profile.multiplicative_depth != 47) {
            throw std::runtime_error(
                "feature-packed effective profile hash/depth drifted");
        }
        moai::openfhe::PrintSecurityDisclosure(profile, std::cout);

        moai::openfhe::PackingSpec packing;
        packing.layout = moai::openfhe::PackingLayout::kContiguous;
        packing.logical_shape = {kFeatureSlots, 1};
        packing.batch_lanes = 1;
        packing.slot_count = profile.slot_count;
        packing.active_slots = kFeatureSlots;
        packing.encoded_slots = kFeatureSlots;
        packing.level = profile.multiplicative_depth - 1;
        packing.noise_scale_degree = 1;
        packing.scaling_factor =
            std::ldexp(1.0, static_cast<int>(profile.scaling_modulus_bits));

        std::vector<double> input(kFeatureSlots, 0.0);
        for (std::size_t slot = 0; slot < kHiddenSize; ++slot) {
            input[slot] =
                static_cast<double>(static_cast<int>(slot % 29) - 14) / 32.0;
        }

        uint32_t input_level = 0;
        uint32_t output_level = 0;
        uint32_t remaining_levels = 0;
        BootstrapError error;
        moai::openfhe::RunMetrics metrics;
        {
            moai::openfhe::ClientRuntime client(profile);
            client.GenerateEvaluationKeys({}, true);
            moai::openfhe::ServerRuntime server(
                client.ExportServerKeyBundle(),
                profile);
            const auto depleted = client.Encrypt({input}, packing);
            input_level = depleted.packing.level;
            const auto refreshed = server.Bootstrap(depleted);
            output_level = refreshed.packing.level;
            remaining_levels = server.RemainingLevels(refreshed);
            error = MeasureError(client.Decrypt(refreshed).front(), input);
            metrics = server.metrics();
        }

        std::cout
            << "{\"diagnostic\":\"openfhe_feature_bootstrap_pre_gate\","
            << "\"parameter_sha256\":\"" << profile.parameter_sha256 << "\","
            << "\"input_level\":" << input_level << ','
            << "\"output_level\":" << output_level << ','
            << "\"remaining_levels\":" << remaining_levels << ','
            << "\"active_max_abs\":" << error.active_max_abs << ','
            << "\"inactive_max_abs\":" << error.inactive_max_abs << "}\n";
        if (output_level >= input_level || remaining_levels < 28) {
            throw std::runtime_error(
                "1024-slot bootstrap did not restore the graph depth budget");
        }
        if (error.active_max_abs > 1e-6 || error.inactive_max_abs > 1e-6) {
            throw std::runtime_error(
                "1024-slot bootstrap exceeded the fixed 1e-6 error gate");
        }
        if (metrics.bootstraps != 1 || metrics.bootstrap_iterations != 2 ||
            metrics.multiplicative_depth != 47 ||
            metrics.max_observed_level > profile.multiplicative_depth) {
            throw std::runtime_error(
                "1024-slot bootstrap metrics violate the M4 contract");
        }

        std::cout
            << "{\"test\":\"openfhe_feature_bootstrap_smoke\","
            << "\"profile\":\"paper_compat\","
            << "\"security_claim\":\"none\","
            << "\"parameter_sha256\":\"" << profile.parameter_sha256 << "\","
            << "\"encoded_slots\":" << kFeatureSlots << ','
            << "\"hidden_size\":" << kHiddenSize << ','
            << "\"multiplicative_depth\":"
            << profile.multiplicative_depth << ','
            << "\"levels_available_after_bootstrap\":"
            << profile.levels_available_after_bootstrap << ','
            << "\"input_level\":" << input_level << ','
            << "\"output_level\":" << output_level << ','
            << "\"remaining_levels\":" << remaining_levels << ','
            << "\"active_max_abs\":" << error.active_max_abs << ','
            << "\"inactive_max_abs\":" << error.inactive_max_abs << ','
            << "\"bootstraps\":" << metrics.bootstraps << ','
            << "\"bootstrap_iterations\":"
            << metrics.bootstrap_iterations << "}\n";
        return 0;
    }
    catch (const std::exception& exception) {
        std::cerr << "openfhe_feature_bootstrap_smoke failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
