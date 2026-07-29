#include "moai/openfhe/client_runtime.hpp"
#include "moai/openfhe/context_factory.hpp"
#include "moai/openfhe/server_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
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
            throw std::runtime_error("bootstrap output is not finite");
        }
        maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
    }
    return maximum;
}

}  // namespace

int main() {
    try {
        auto profile = moai::openfhe::MakePaperCompatProfile();
        profile.bootstrap_level_budget = {2, 2};
        profile.bootstrap_bsgs_dim = {0, 0};
        constexpr uint32_t kSmokeSlots = 8;
        moai::openfhe::ConfigureBootstrap(profile, kSmokeSlots, 2);
        moai::openfhe::PrintSecurityDisclosure(profile, std::cout);
        std::cout
            << "bootstrap_smoke_scope=test-only sparse 8-slot API validation; "
            << "level_budget=[2,2]; not a 32768-slot workload result\n";

        auto context = moai::openfhe::MakeCryptoContext(profile);
        moai::openfhe::PackingSpec packing;
        packing.layout = moai::openfhe::PackingLayout::kContiguous;
        packing.logical_shape = {kSmokeSlots};
        packing.batch_lanes = 1;
        packing.slot_count = profile.slot_count;
        packing.encoded_slots = kSmokeSlots;
        packing.active_slots = kSmokeSlots;
        packing.level = profile.multiplicative_depth - 1;
        packing.noise_scale_degree = 1;
        packing.scaling_factor =
            std::ldexp(1.0, static_cast<int>(profile.scaling_modulus_bits));

        const std::vector<double> input{
            -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0};
        double maximum_error = 0.0;
        uint32_t input_level = 0;
        uint32_t output_level = 0;
        uint64_t bootstrap_count = 0;

        {
            moai::openfhe::ClientRuntime client(context, profile);
            client.GenerateEvaluationKeys({}, true);
            moai::openfhe::ServerRuntime server(client.ExportServerKeyBundle(), profile);

            const auto depleted = client.Encrypt({input}, packing);
            input_level = depleted.packing.level;
            const auto refreshed = server.Bootstrap(depleted);
            output_level = refreshed.packing.level;
            maximum_error =
                MaxAbsoluteError(client.Decrypt(refreshed).front(), input);
            if (maximum_error > 5e-3) {
                throw std::runtime_error(
                    "bootstrap max absolute error exceeds 5e-3");
            }
            if (output_level >= input_level) {
                throw std::runtime_error(
                    "bootstrap did not replenish usable ciphertext levels");
            }
            bootstrap_count = server.metrics().bootstraps;
            if (bootstrap_count != 1) {
                throw std::runtime_error("unexpected bootstrap operation count");
            }
        }

        context->ClearStaticMapsAndVectors();
        std::cout << "{\"test\":\"openfhe_bootstrap_smoke\","
                  << "\"profile\":\"paper_compat\","
                  << "\"security_claim\":\"none\","
                  << "\"encoded_slots\":" << kSmokeSlots << ","
                  << "\"input_level\":" << input_level << ","
                  << "\"output_level\":" << output_level << ","
                  << "\"max_abs\":" << maximum_error << ","
                  << "\"bootstraps\":" << bootstrap_count << "}\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "openfhe_bootstrap_smoke failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
