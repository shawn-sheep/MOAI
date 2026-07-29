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
        auto context = moai::openfhe::MakeCryptoContext(profile);

        moai::openfhe::PackingSpec packing;
        packing.layout = moai::openfhe::PackingLayout::kContiguous;
        packing.logical_shape = {8};
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
            moai::openfhe::ClientRuntime client(context, profile);
            client.GenerateEvaluationKeys({1}, false);
            moai::openfhe::ServerRuntime server(client.ExportServerKeyBundle(), profile);

            const auto encrypted = client.Encrypt({input}, packing);
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
                server.EncodeModelVector(double_weights, packing);
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

        context->ClearStaticMapsAndVectors();
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
