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
    {{0.0015713992547119072, 0.999998778895378, 0.012056555059118956},
     {{-12.432522113824433, 1.6424796180067638},
      {0.09778020866027208, 10.227016405343853},
      {17.51842630700885, 27.841426185307984},
      {-15.453325797999769, 8.187285833234172},
      {1.223930981277494, 6.9253009796762415}}},
    {{0.002747185270206928, 0.9999962615533423, 0.025712009375612688},
     {{-7.986447366776763, 2.7119717778763226},
      {0.06364579637891832, 15.843808924754743},
      {13.42185150806978, 35.79859989896021},
      {-11.825415557231615, 8.437722054302307},
      {1.1354547607420578, 7.43332573430004}}},
    {{0.0032669219048856875, 0.9999946813470472, 0.032553716114295916},
     {{-15.067747120327708, 4.339600260745062},
      {0.013164405863243273, 76.73250948682391},
      {13.95060121662069, 45.63883205006145},
      {-18.59976875033425, 52.14020809258102},
      {1.0639819326643813, 64.28660554664827}}},
    {{0.004569260677959954, 0.9999895783334914, 0.04591061634941784},
     {{-13.868456651618557, 3.2375331945167893},
      {0.03681409764452016, 26.732328587358975},
      {12.6675329190837, 41.21021612778427},
      {-60.26836542943887, 10.122530204387617},
      {1.0024529793678243, 53.39072363939901}}},
    {{0.006472339840257425, 0.9999790872858795, 0.06572918731104593},
     {{-11.412655197227338, 2.8953731223051324},
      {0.05545839962958138, 18.11372323943258},
      {9.79797504408733, 41.205698832141266},
      {-60.90997840997095, 12.149560411899845},
      {1.03564465623748, 50.491946891162556}}},
    {{0.007150860864815971, 0.9999744832790439, 0.061381486978444144},
     {{-8.513148865018529, 1.6730266462200847},
      {0.1780803800922579, 5.343450532512705},
      {8.869476286902845, 45.31195604379697},
      {-48.94584888737414, 11.46827442284909},
      {1.0306934621616908, 34.16405384616632}}},
    {{0.007656152266571541, 0.9999707767659259, 0.0733841048839694},
     {{-7.962183403511881, 3.1300383019463913},
      {0.04172275724991148, 22.974816996141424},
      {12.164761720265062, 44.41220464028341},
      {-25.53782371333585, 8.441221771031497},
      {1.0119258508240507, 22.79737077655154}}},
    {{0.008074879419491554, 0.9999688076141282, 0.0836453060087623},
     {{-8.92032500742443, 1.7899726381796448},
      {0.16642613171255416, 6.067707229480634},
      {10.259034970685423, 43.443438918945205},
      {-17.80605742742762, 4.953668822251908},
      {1.2138875254751642, 14.581905019198166}}},
    {{0.009226505147424655, 0.9999614616340301, 0.08651110120415417},
     {{-8.756071640122187, 2.0503536646540756},
      {0.12338106518789421, 7.93921677108049},
      {8.394744064936432, 37.44426836396667},
      {-14.383706608764756, 4.6746562702140135},
      {1.4514937080266404, 16.586084886401245}}},
    {{0.009445907509012629, 0.99995626037816, 0.09210708091288744},
     {{-7.796168138237539, 1.9306575710773464},
      {0.09134889689025238, 10.983516594880971},
      {8.115813905548093, 42.79262775502206},
      {-25.841404602058724, 35.386654625226775},
      {1.1963944019188164, 273.3093936501193}}},
    {{0.00997827288909222, 0.9999520811410911, 0.09976871537536036},
     {{-11.427032878696789, 3.87162756026878},
      {0.01732634499617225, 58.60759280282173},
      {3.559282822276061, 56.631943333644585},
      {-17.219868603401952, 122.80584340973375},
      {1.2934268747660198, 1275.3092456646355}}},
    {{0.010519199488026413, 0.999944942514787, 0.07618263327099317},
     {{-7.441480116416211, 0.6528106275038921},
      {0.2576883762380675, 4.0088864890907505},
      {1.559921924961595, 65.92639994632418},
      {-10.883552280587516, 4.5733827847966015},
      {0.7711727874023071, 1.2082669228528153}}},
}};

constexpr QualityMetrics kExpectedGlobalQuality{
    0.010519199488026413,
    0.999944942514787,
    0.09976871537536036};

constexpr EncoderPlaintextOracleRanges kExpectedGlobalRanges{
    {-15.067747120327708, 4.339600260745062},
    {0.013164405863243273, 76.73250948682391},
    {1.559921924961595, 65.92639994632418},
    {-60.90997840997095, 122.80584340973375},
    {0.7711727874023071, 1275.3092456646355}};

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
