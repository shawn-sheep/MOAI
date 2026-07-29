#include "support/moai_encoder_fixture.hpp"
#include "support/moai_encoder_plaintext_oracle.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

using moai::openfhe::test::EncoderPlaintextOracleResult;
using moai::openfhe::test::PlainMatrix;

constexpr double kExpectedLayer0RelativeL2 = 0.0015713992547119072;
constexpr double kExpectedLayer0Cosine = 0.999998778895378;
constexpr double kExpectedLayer1RelativeL2 = 0.002747185270206928;
constexpr double kExpectedLayer1Cosine = 0.9999962615533423;
constexpr double kMetricTolerance = 2e-12;

struct QualityMetrics {
    double relative_l2{0.0};
    double cosine{0.0};
    double maximum_absolute{0.0};
};

QualityMetrics MeasureQuality(
    const PlainMatrix& actual,
    const PlainMatrix& expected) {
    if (actual.size() != moai::openfhe::kPaperCompatTraceTokens ||
        actual.size() != expected.size()) {
        throw std::runtime_error("oracle output row count changed");
    }
    long double squared_error = 0.0;
    long double squared_actual = 0.0;
    long double squared_expected = 0.0;
    long double dot = 0.0;
    double maximum_absolute = 0.0;
    for (std::size_t token = 0; token < actual.size(); ++token) {
        if (actual[token].size() != moai::openfhe::kPaperCompatFeatureBlock ||
            expected[token].size() != actual[token].size()) {
            throw std::runtime_error("oracle output feature block changed");
        }
        for (std::size_t feature = 0;
             feature < moai::openfhe::kPaperCompatHiddenSize;
             ++feature) {
            const long double observed = actual[token][feature];
            const long double reference = expected[token][feature];
            if (!std::isfinite(static_cast<double>(observed)) ||
                !std::isfinite(static_cast<double>(reference))) {
                throw std::runtime_error("oracle quality input is non-finite");
            }
            const long double error = observed - reference;
            squared_error += error * error;
            squared_actual += observed * observed;
            squared_expected += reference * reference;
            dot += observed * reference;
            maximum_absolute = std::max(
                maximum_absolute,
                std::abs(static_cast<double>(error)));
        }
        for (std::size_t feature = moai::openfhe::kPaperCompatHiddenSize;
             feature < moai::openfhe::kPaperCompatFeatureBlock;
             ++feature) {
            if (actual[token][feature] != 0.0) {
                throw std::runtime_error("oracle inactive output lane is nonzero");
            }
        }
    }
    if (squared_actual == 0.0 || squared_expected == 0.0) {
        throw std::runtime_error("oracle quality norm is zero");
    }
    return {
        std::sqrt(static_cast<double>(squared_error / squared_expected)),
        static_cast<double>(dot / std::sqrt(squared_actual * squared_expected)),
        maximum_absolute};
}

void RequireFiniteOutput(const EncoderPlaintextOracleResult& result) {
    if (result.output.size() != moai::openfhe::kPaperCompatTraceTokens) {
        throw std::runtime_error("plaintext oracle did not return five rows");
    }
    for (const auto& row : result.output) {
        if (row.size() != moai::openfhe::kPaperCompatFeatureBlock ||
            !std::all_of(row.begin(), row.end(), [](double value) {
                return std::isfinite(value);
            })) {
            throw std::runtime_error(
                "plaintext oracle output is malformed or non-finite");
        }
    }
}

std::filesystem::path ParseDataRoot(int argc, char** argv) {
    std::filesystem::path data_root{"data"};
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--data-root" && index + 1 < argc) {
            data_root = argv[++index];
        }
        else {
            throw std::invalid_argument(
                "unknown or incomplete argument: " + argument);
        }
    }
    return data_root;
}

void PrintRange(
    const char* name,
    const moai::openfhe::test::PlaintextOracleRange& range,
    bool trailing_comma) {
    std::cout << '\"' << name << "\":[" << range.minimum << ','
              << range.maximum << ']';
    if (trailing_comma) {
        std::cout << ',';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::setprecision(17);
        const auto data_root = ParseDataRoot(argc, argv);

        const auto layer0 =
            moai::openfhe::test::LoadEncoderLayerFixture(data_root, 0);
        const auto layer0_result =
            moai::openfhe::test::EvaluateEncoderLayerPlaintextOracle(
                layer0.expected.input,
                layer0.weights);
        RequireFiniteOutput(layer0_result);
        const auto quality = MeasureQuality(
            layer0_result.output,
            layer0.expected.output);
        if (std::abs(
                quality.relative_l2 - kExpectedLayer0RelativeL2) >
                kMetricTolerance ||
            std::abs(quality.cosine - kExpectedLayer0Cosine) >
                kMetricTolerance) {
            throw std::runtime_error(
                "layer-0 frozen-polynomial oracle metric drifted");
        }

        const auto layer1 =
            moai::openfhe::test::LoadEncoderLayerFixture(data_root, 1);
        const auto layer1_result =
            moai::openfhe::test::EvaluateEncoderLayerPlaintextOracle(
                layer0_result.output,
                layer1.weights);
        RequireFiniteOutput(layer1_result);
        const auto layer1_quality = MeasureQuality(
            layer1_result.output,
            layer1.expected.output);
        if (std::abs(
                layer1_quality.relative_l2 - kExpectedLayer1RelativeL2) >
                kMetricTolerance ||
            std::abs(layer1_quality.cosine - kExpectedLayer1Cosine) >
                kMetricTolerance) {
            std::ostringstream message;
            message << std::setprecision(17)
                    << "layer-1 frozen-polynomial oracle metric drifted: rel_l2="
                    << layer1_quality.relative_l2
                    << " cosine=" << layer1_quality.cosine;
            throw std::runtime_error(message.str());
        }

        std::cout
            << "{\"test\":\"openfhe_encoder_plaintext_oracle\","
            << "\"profile\":\"paper_compat\","
            << "\"security_claim\":\"none\","
            << "\"layer0_relative_l2\":" << quality.relative_l2 << ','
            << "\"layer0_cosine\":" << quality.cosine << ','
            << "\"layer0_max_absolute\":" << quality.maximum_absolute << ','
            << "\"layer0_ranges\":{";
        PrintRange(
            "softmax_shifted_logits",
            layer0_result.ranges.softmax_shifted_logits,
            true);
        PrintRange(
            "softmax_denominator",
            layer0_result.ranges.softmax_denominator,
            true);
        PrintRange(
            "ln1_normalized_variance",
            layer0_result.ranges.attention_layernorm_normalized_variance,
            true);
        PrintRange("gelu_input", layer0_result.ranges.gelu_input, true);
        PrintRange(
            "ln2_normalized_variance",
            layer0_result.ranges.output_layernorm_normalized_variance,
            false);
        std::cout
            << "},\"layer1_relative_l2\":"
            << layer1_quality.relative_l2 << ','
            << "\"layer1_cosine\":" << layer1_quality.cosine << ','
            << "\"layer1_max_absolute\":"
            << layer1_quality.maximum_absolute << ','
            << "\"layer1_finite_and_in_range\":true}\n";
        return 0;
    }
    catch (const std::exception& exception) {
        std::cerr << "openfhe_encoder_plaintext_oracle_smoke failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
