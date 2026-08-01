#include "support/moai_encoder_fixture.hpp"
#include "support/moai_encoder_plaintext_oracle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using moai::openfhe::test::EncoderPlaintextOracleRanges;
using moai::openfhe::test::EncoderPlaintextOracleResult;
using moai::openfhe::test::PlainMatrix;
using moai::openfhe::test::PlaintextOracleRange;

constexpr std::size_t kLayerCount = 12;
constexpr double kMetricTolerance = 2e-12;
constexpr double kRangeTolerance = 1e-9;
constexpr double kPerLayerRelativeL2Maximum = 0.012;
constexpr double kPerLayerCosineMinimum = 0.9999;
constexpr double kPerLayerMaximumAbsolute = 0.1;
constexpr double kFinalRelativeL2Maximum = 0.05;
constexpr double kFinalCosineMinimum = 0.99;

struct QualityMetrics {
    double relative_l2{0.0};
    double cosine{0.0};
    double maximum_absolute{0.0};
};

struct FrozenLayerSummary {
    QualityMetrics quality;
    EncoderPlaintextOracleRanges ranges;
};

// Exact chained_oracle_summary values frozen in config/moai_encoder_trace.json.
constexpr std::array<FrozenLayerSummary, kLayerCount> kExpectedLayers{{
    {{0.0015021945973760012, 0.9999989410045913, 0.009993845852454442},
     {{-12.432522113824433, 1.6424796180067638},
      {0.09778020866027208, 10.227016405343853},
      {45.286738442032295, 73.89404010989917},
      {-15.44972744320672, 8.183037801560676},
      {46.13884541869377, 82.18391150782362}}},
    {{0.0027102827822521143, 0.9999965396138284, 0.023135088151543748},
     {{-7.986016926934518, 2.720264982833913},
      {0.06358986225133356, 15.858535106231976},
      {53.679480370319034, 85.36548963834767},
      {-11.822964677756408, 8.439117456735344},
      {51.68335345370109, 88.63799827686323}}},
    {{0.003209968791369369, 0.999995063441511, 0.026402858958057607},
     {{-15.075321045691314, 4.333621756562744},
      {0.013112879348878975, 76.27535888036451},
      {45.41716286168661, 64.96574865378544},
      {-18.6125707519607, 52.15772515852034},
      {46.0515797385602, 74.65883744377427}}},
    {{0.004341529942655493, 0.9999907114744442, 0.03792947464574592},
     {{-13.875028807437811, 3.2530953158306026},
      {0.03681096981729437, 27.13766793510004},
      {45.911472496028175, 82.62656673041245},
      {-60.267448177127235, 10.12047621279129},
      {53.37484996154999, 81.95535344182923}}},
    {{0.00612507042853847, 0.9999814480936299, 0.0570481637659519},
     {{-11.434987479093078, 2.8952812610787735},
      {0.05549652912023434, 18.112067517689685},
      {51.63222525744133, 82.58803369760435},
      {-60.94863917442944, 12.16080737384046},
      {50.571383012039384, 77.75976760356893}}},
    {{0.006895995716331725, 0.9999763353252004, 0.05773242069022633},
     {{-8.523407284612299, 1.681698576147653},
      {0.17789230107714063, 5.389859085212268},
      {45.4114673979457, 84.99135479962678},
      {-48.93856985438508, 11.468367521754935},
      {50.46791876039441, 82.93068412412065}}},
    {{0.007471546935630877, 0.9999723929127488, 0.06322931320438707},
     {{-7.963997321316402, 3.124920375898067},
      {0.0417058666851598, 22.85787003267466},
      {48.63283214684414, 88.91878012356948},
      {-25.52213057785888, 8.434113945472824},
      {45.530491235309945, 80.5532764884045}}},
    {{0.0077610308051309545, 0.9999705805344254, 0.07417271283831872},
     {{-8.921720774188076, 1.7919998706058369},
      {0.16639572244553114, 6.079770899974453},
      {69.47852259864112, 87.02136576532706},
      {-17.81207948620825, 4.950664910757361},
      {50.58095055434945, 77.61961551377833}}},
    {{0.008842568178826627, 0.9999628449103055, 0.0792155914573569},
     {{-8.745449493537345, 2.0514438246548585},
      {0.12339388728253359, 7.947580684033094},
      {63.495126625713745, 74.67765508327844},
      {-14.380093770945972, 4.674517258758698},
      {46.448937246100286, 66.31494431910431}}},
    {{0.009217773300546498, 0.999958489419405, 0.08430604314163981},
     {{-7.788660430576416, 1.930076356076933},
      {0.09134617836936537, 10.958814559588244},
      {61.17993183582617, 85.4113356587975},
      {-25.834861642334758, 35.39362296928234},
      {47.961079559502785, 81.00551446676928}}},
    {{0.009959319663659455, 0.9999525043195464, 0.09286962991808734},
     {{-11.426986652020592, 3.881651282177523},
      {0.01732770392096422, 59.05353591419574},
      {49.04372640751758, 88.65000348739036},
      {-17.21838733993946, 122.77638179497981},
      {49.75842592480486, 89.0434461726423}}},
    {{0.010676974997627438, 0.9999444226229512, 0.07050319720197029},
     {{-7.441972511791998, 0.6531319499383765},
      {0.2577619193501891, 4.0053717958089825},
      {49.92469787664196, 89.40612388178232},
      {-10.881189846419714, 4.5746949001868815},
      {49.29154751331137, 77.292894643863}}},
}};

constexpr QualityMetrics kExpectedGlobalQuality{
    0.010676974997627438,
    0.9999444226229512,
    0.09286962991808734};

constexpr EncoderPlaintextOracleRanges kExpectedGlobalRanges{
    {-15.075321045691314, 4.333621756562744},
    {0.013112879348878975, 76.27535888036451},
    {45.286738442032295, 89.40612388178232},
    {-60.94863917442944, 122.77638179497981},
    {45.530491235309945, 89.0434461726423}};

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

void RequireNear(
    double actual,
    double expected,
    const std::string& label,
    double tolerance = kMetricTolerance) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::ostringstream message;
        message << std::setprecision(17) << label << " drifted: actual=" << actual
                << " expected=" << expected;
        throw std::runtime_error(message.str());
    }
}

void RequireRange(
    const PlaintextOracleRange& actual,
    const PlaintextOracleRange& expected,
    const std::string& label) {
    RequireNear(
        actual.minimum,
        expected.minimum,
        label + ".minimum",
        kRangeTolerance);
    RequireNear(
        actual.maximum,
        expected.maximum,
        label + ".maximum",
        kRangeTolerance);
}

void RequireRanges(
    const EncoderPlaintextOracleRanges& actual,
    const EncoderPlaintextOracleRanges& expected,
    const std::string& label) {
    RequireRange(
        actual.softmax_shifted_logits,
        expected.softmax_shifted_logits,
        label + ".softmax_shifted_logits");
    RequireRange(
        actual.softmax_denominator,
        expected.softmax_denominator,
        label + ".softmax_denominator");
    RequireRange(
        actual.attention_layernorm_normalized_variance,
        expected.attention_layernorm_normalized_variance,
        label + ".ln1_normalized_variance");
    RequireRange(actual.gelu_input, expected.gelu_input, label + ".gelu_input");
    RequireRange(
        actual.output_layernorm_normalized_variance,
        expected.output_layernorm_normalized_variance,
        label + ".ln2_normalized_variance");
}

void RequireLayerGate(const QualityMetrics& quality, std::size_t layer) {
    if (quality.relative_l2 > kPerLayerRelativeL2Maximum ||
        quality.cosine < kPerLayerCosineMinimum ||
        quality.maximum_absolute > kPerLayerMaximumAbsolute) {
        throw std::runtime_error(
            "layer " + std::to_string(layer) + " quality gate failed");
    }
}

void RequireFrozenLayer(
    const QualityMetrics& quality,
    const EncoderPlaintextOracleRanges& ranges,
    std::size_t layer) {
    const auto& expected = kExpectedLayers[layer];
    const std::string label = "layer " + std::to_string(layer);
    RequireNear(
        quality.relative_l2,
        expected.quality.relative_l2,
        label + ".relative_l2");
    RequireNear(quality.cosine, expected.quality.cosine, label + ".cosine");
    RequireNear(
        quality.maximum_absolute,
        expected.quality.maximum_absolute,
        label + ".maximum_absolute");
    RequireRanges(ranges, expected.ranges, label + ".ranges");
}

PlaintextOracleRange MergeRange(
    const PlaintextOracleRange& left,
    const PlaintextOracleRange& right) {
    return {
        std::min(left.minimum, right.minimum),
        std::max(left.maximum, right.maximum)};
}

EncoderPlaintextOracleRanges MergeRanges(
    const EncoderPlaintextOracleRanges& left,
    const EncoderPlaintextOracleRanges& right) {
    return {
        MergeRange(left.softmax_shifted_logits, right.softmax_shifted_logits),
        MergeRange(left.softmax_denominator, right.softmax_denominator),
        MergeRange(
            left.attention_layernorm_normalized_variance,
            right.attention_layernorm_normalized_variance),
        MergeRange(left.gelu_input, right.gelu_input),
        MergeRange(
            left.output_layernorm_normalized_variance,
            right.output_layernorm_normalized_variance)};
}

void PrintRange(const char* name, const PlaintextOracleRange& range) {
    std::cout << '\"' << name << "\":[" << range.minimum << ','
              << range.maximum << ']';
}

void PrintRanges(const EncoderPlaintextOracleRanges& ranges) {
    std::cout << '{';
    PrintRange("softmax_shifted_logits", ranges.softmax_shifted_logits);
    std::cout << ',';
    PrintRange("softmax_denominator", ranges.softmax_denominator);
    std::cout << ',';
    PrintRange(
        "ln1_normalized_variance",
        ranges.attention_layernorm_normalized_variance);
    std::cout << ',';
    PrintRange("gelu_input", ranges.gelu_input);
    std::cout << ',';
    PrintRange(
        "ln2_normalized_variance",
        ranges.output_layernorm_normalized_variance);
    std::cout << '}';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::setprecision(17);
        const auto data_root = ParseDataRoot(argc, argv);
        const auto layer0 =
            moai::openfhe::test::LoadEncoderLayerFixture(data_root, 0);
        PlainMatrix chained_input = layer0.expected.input;
        std::vector<QualityMetrics> qualities;
        std::vector<EncoderPlaintextOracleRanges> ranges;
        qualities.reserve(kLayerCount);
        ranges.reserve(kLayerCount);

        for (std::size_t layer = 0; layer < kLayerCount; ++layer) {
            const auto fixture =
                moai::openfhe::test::LoadEncoderLayerFixture(data_root, layer);
            const auto result =
                moai::openfhe::test::EvaluateEncoderLayerPlaintextOracle(
                    chained_input,
                    fixture.weights);
            RequireFiniteOutput(result);
            const auto quality = MeasureQuality(
                result.output,
                fixture.expected.output);
            RequireLayerGate(quality, layer);
            RequireFrozenLayer(quality, result.ranges, layer);
            qualities.push_back(quality);
            ranges.push_back(result.ranges);
            chained_input = result.output;
        }

        EncoderPlaintextOracleRanges global_ranges = ranges.front();
        QualityMetrics global_quality = qualities.front();
        for (std::size_t layer = 1; layer < kLayerCount; ++layer) {
            global_ranges = MergeRanges(global_ranges, ranges[layer]);
            global_quality.relative_l2 = std::max(
                global_quality.relative_l2,
                qualities[layer].relative_l2);
            global_quality.cosine = std::min(
                global_quality.cosine,
                qualities[layer].cosine);
            global_quality.maximum_absolute = std::max(
                global_quality.maximum_absolute,
                qualities[layer].maximum_absolute);
        }
        RequireNear(
            global_quality.relative_l2,
            kExpectedGlobalQuality.relative_l2,
            "global.worst_relative_l2");
        RequireNear(
            global_quality.cosine,
            kExpectedGlobalQuality.cosine,
            "global.minimum_cosine");
        RequireNear(
            global_quality.maximum_absolute,
            kExpectedGlobalQuality.maximum_absolute,
            "global.maximum_absolute");
        RequireRanges(global_ranges, kExpectedGlobalRanges, "global.ranges");

        const auto& final_quality = qualities.back();
        if (final_quality.relative_l2 > kFinalRelativeL2Maximum ||
            final_quality.cosine < kFinalCosineMinimum ||
            !std::isfinite(final_quality.maximum_absolute)) {
            throw std::runtime_error("12-layer final quality gate failed");
        }

        std::cout
            << "{\"test\":\"openfhe_encoder_plaintext_oracle\","
            << "\"profile\":\"paper_compat\","
            << "\"security_claim\":\"none\","
            << "\"layer_count\":" << kLayerCount << ",\"layers\":[";
        for (std::size_t layer = 0; layer < kLayerCount; ++layer) {
            if (layer != 0) {
                std::cout << ',';
            }
            std::cout << "{\"layer_id\":" << layer
                      << ",\"relative_l2\":" << qualities[layer].relative_l2
                      << ",\"cosine\":" << qualities[layer].cosine
                      << ",\"max_absolute\":"
                      << qualities[layer].maximum_absolute
                      << ",\"finite_and_in_range\":true,\"ranges\":";
            PrintRanges(ranges[layer]);
            std::cout << '}';
        }
        std::cout << "],\"global\":{\"worst_relative_l2\":"
                  << global_quality.relative_l2
                  << ",\"minimum_cosine\":" << global_quality.cosine
                  << ",\"maximum_absolute\":"
                  << global_quality.maximum_absolute << ",\"ranges\":";
        PrintRanges(global_ranges);
        std::cout << "},\"final\":{\"layer_id\":11,\"relative_l2\":"
                  << final_quality.relative_l2
                  << ",\"cosine\":" << final_quality.cosine
                  << ",\"max_absolute\":"
                  << final_quality.maximum_absolute
                  << ",\"finite\":true},\"passed\":true}\n";
        return 0;
    }
    catch (const std::exception& exception) {
        std::cerr << "openfhe_encoder_plaintext_oracle_smoke failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
