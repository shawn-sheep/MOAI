#include "support/moai_encoder_plaintext_oracle.hpp"

#include "moai/openfhe/approximation_registry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace moai::openfhe::test {
namespace {

constexpr std::size_t kHeadDimension = 64;
constexpr double kAttentionScale = 8.0;
static_assert(
    kPaperCompatAttentionHeads * kHeadDimension == kPaperCompatHiddenSize);
static_assert(
    kPaperCompatIntermediateBlocks * kPaperCompatFeatureBlock ==
    kPaperCompatIntermediateSize);

struct RangeAccumulator {
    double minimum{std::numeric_limits<double>::infinity()};
    double maximum{-std::numeric_limits<double>::infinity()};

    void Observe(
        double value,
        const DeclaredRange& interval,
        const std::string& label) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(label + " contains NaN or Inf");
        }
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        if (value < interval.minimum || value > interval.maximum) {
            throw std::out_of_range(
                label + " is outside its frozen interval; clipping is forbidden");
        }
    }

    [[nodiscard]] PlaintextOracleRange Finalize(
        const std::string& label) const {
        if (!std::isfinite(minimum) || !std::isfinite(maximum)) {
            throw std::logic_error(label + " observed no values");
        }
        return {minimum, maximum};
    }
};

void Require(
    bool condition,
    const std::string& message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

void ValidateVector(
    const std::vector<double>& values,
    std::size_t expected,
    const std::string& label) {
    Require(values.size() == expected, label + " width changed");
    for (const double value : values) {
        Require(std::isfinite(value), label + " contains NaN or Inf");
    }
}

void ValidateMatrix(
    const PlainMatrix& values,
    std::size_t rows,
    std::size_t columns,
    const std::string& label) {
    Require(values.size() == rows, label + " row count changed");
    for (const auto& row : values) {
        ValidateVector(row, columns, label);
    }
}

void ValidateWeights(const EncoderLayerWeights& weights) {
    Require(
        weights.layer_index < kPaperCompatEncoderLayers,
        "encoder layer index is outside paper_compat");
    ValidateMatrix(
        weights.query_weights,
        kPaperCompatHiddenSize,
        kPaperCompatHiddenSize,
        "query weights");
    ValidateVector(weights.query_bias, kPaperCompatHiddenSize, "query bias");
    ValidateMatrix(
        weights.key_weights,
        kPaperCompatHiddenSize,
        kPaperCompatHiddenSize,
        "key weights");
    ValidateVector(weights.key_bias, kPaperCompatHiddenSize, "key bias");
    ValidateMatrix(
        weights.value_weights,
        kPaperCompatHiddenSize,
        kPaperCompatHiddenSize,
        "value weights");
    ValidateVector(weights.value_bias, kPaperCompatHiddenSize, "value bias");
    ValidateMatrix(
        weights.self_output_weights,
        kPaperCompatHiddenSize,
        kPaperCompatHiddenSize,
        "self-output weights");
    ValidateVector(
        weights.self_output_bias,
        kPaperCompatHiddenSize,
        "self-output bias");
    ValidateVector(
        weights.attention_layernorm_gamma,
        kPaperCompatHiddenSize,
        "attention LayerNorm gamma");
    ValidateVector(
        weights.attention_layernorm_beta,
        kPaperCompatHiddenSize,
        "attention LayerNorm beta");
    for (std::size_t block = 0;
         block < kPaperCompatIntermediateBlocks;
         ++block) {
        ValidateMatrix(
            weights.intermediate_weight_blocks[block],
            kPaperCompatHiddenSize,
            kPaperCompatFeatureBlock,
            "intermediate weight block " + std::to_string(block));
        ValidateVector(
            weights.intermediate_bias_blocks[block],
            kPaperCompatFeatureBlock,
            "intermediate bias block " + std::to_string(block));
        ValidateMatrix(
            weights.output_weight_blocks[block],
            kPaperCompatFeatureBlock,
            kPaperCompatHiddenSize,
            "output weight block " + std::to_string(block));
    }
    ValidateVector(weights.output_bias, kPaperCompatHiddenSize, "output bias");
    ValidateVector(
        weights.output_layernorm_gamma,
        kPaperCompatHiddenSize,
        "output LayerNorm gamma");
    ValidateVector(
        weights.output_layernorm_beta,
        kPaperCompatHiddenSize,
        "output LayerNorm beta");
}

void ValidateContract(
    const ApproximationContract& contract,
    const std::string& contract_id,
    double minimum,
    double maximum,
    uint32_t degree,
    const std::string& coefficient_sha256) {
    if (contract.contract_id != contract_id || contract.basis != "chebyshev" ||
        contract.interval.minimum != minimum ||
        contract.interval.maximum != maximum || contract.degree != degree ||
        contract.coefficients.size() != static_cast<std::size_t>(degree) + 1 ||
        contract.coefficient_sha256 != coefficient_sha256 ||
        ComputeCoefficientSha256(contract.coefficients) != coefficient_sha256) {
        throw std::logic_error(
            contract_id + " differs from config/openfhe_approximations.json");
    }
}

void ValidateContracts(const PaperCompatNonlinearContracts& contracts) {
    ValidateContract(
        contracts.gelu,
        "gelu_d319_zero_at_origin",
        -80.0,
        128.0,
        319,
        "35d68b2f56f267f27e8f962c3bd54cddd90892349b77405172f48987864eaaa3");
    ValidateContract(
        contracts.softmax_exponential,
        "softmax_exp_d27",
        -16.0,
        5.0,
        27,
        "6eda4377151897e8c4ca4d72f2a918db0b888fc6771f8ee5cf1d950b378de76f");
    ValidateContract(
        contracts.softmax_reciprocal,
        "softmax_reciprocal_d383",
        0.01,
        80.0,
        383,
        "fa97f298751bca97f40eed3b6de949262d1b3971cda57013b55420d5c9fbbeb3");
    ValidateContract(
        contracts.layernorm_inverse_sqrt,
        "layernorm_inverse_sqrt_d159",
        0.5,
        1536.0,
        159,
        "28d0ffd36436228da4ee6023e0aad7641be995492a4314375b5a1aafe68465be");
    if (contracts.softmax_shifts.contract_id !=
            kPaperCompatSoftmaxShiftContractId ||
        contracts.softmax_shifts.values_sha256 !=
            kPaperCompatSoftmaxShiftSha256) {
        throw std::logic_error(
            "Softmax shifts differ from config/openfhe_approximations.json");
    }
    static_cast<void>(contracts.softmax_shifts.At(0, 0));
}

double EvaluateChebyshev(
    double input,
    const ApproximationContract& contract) {
    if (!std::isfinite(input)) {
        throw std::runtime_error(
            contract.contract_id + " input contains NaN or Inf");
    }
    const double normalized =
        (2.0 * input - contract.interval.minimum - contract.interval.maximum) /
        (contract.interval.maximum - contract.interval.minimum);

    // OpenFHE stores a doubled raw c0. Clenshaw consumes c0/2, matching the
    // frozen NumPy validator while avoiding the instability of power-basis
    // conversion for the degree-383 reciprocal polynomial.
    double next = 0.0;
    double next_next = 0.0;
    for (std::size_t index = contract.coefficients.size(); index-- > 1;) {
        const double current = contract.coefficients[index] +
            2.0 * normalized * next - next_next;
        next_next = next;
        next = current;
    }
    const double result = contract.coefficients.front() / 2.0 +
        normalized * next - next_next;
    if (!std::isfinite(result)) {
        throw std::runtime_error(
            contract.contract_id + " output contains NaN or Inf");
    }
    return result;
}

PlainMatrix DenseAffine(
    const PlainMatrix& input,
    const FeaturePackedWeights& weights,
    const std::vector<double>& bias,
    const std::string& label) {
    Require(!input.empty() && !weights.empty(), label + " is empty");
    ValidateVector(bias, weights.front().size(), label + " bias");
    for (const auto& row : input) {
        Require(
            row.size() >= weights.size(),
            label + " input width is smaller than its weight rows");
    }
    for (const auto& row : weights) {
        ValidateVector(row, bias.size(), label + " weights");
    }

    PlainMatrix output(input.size(), bias);
    for (std::size_t token = 0; token < input.size(); ++token) {
        for (std::size_t in = 0; in < weights.size(); ++in) {
            const double activation = input[token][in];
            Require(std::isfinite(activation), label + " input is non-finite");
            for (std::size_t out = 0; out < bias.size(); ++out) {
                output[token][out] += activation * weights[in][out];
            }
        }
    }
    ValidateMatrix(output, input.size(), bias.size(), label + " output");
    return output;
}

PlainMatrix Add(
    const PlainMatrix& lhs,
    const PlainMatrix& rhs,
    const std::string& label) {
    Require(lhs.size() == rhs.size() && !lhs.empty(), label + " row mismatch");
    PlainMatrix output = lhs;
    for (std::size_t row = 0; row < lhs.size(); ++row) {
        Require(lhs[row].size() == rhs[row].size(), label + " width mismatch");
        for (std::size_t column = 0; column < lhs[row].size(); ++column) {
            output[row][column] += rhs[row][column];
            Require(std::isfinite(output[row][column]), label + " is non-finite");
        }
    }
    return output;
}

PlainMatrix LayerNorm(
    const PlainMatrix& input,
    const std::vector<double>& gamma,
    const std::vector<double>& beta,
    double variance_scale,
    const ApproximationContract& inverse_sqrt,
    RangeAccumulator& observed,
    const std::string& label) {
    ValidateMatrix(
        input,
        kPaperCompatTraceTokens,
        kPaperCompatHiddenSize,
        label + " input");
    ValidateVector(gamma, kPaperCompatHiddenSize, label + " gamma");
    ValidateVector(beta, kPaperCompatHiddenSize, label + " beta");
    Require(
        variance_scale == kPaperCompatLayerNorm1VarianceScale ||
            variance_scale == kPaperCompatLayerNorm2VarianceScale,
        label + " variance scale is not frozen");

    PlainMatrix output(
        kPaperCompatTraceTokens,
        std::vector<double>(kPaperCompatHiddenSize));
    for (std::size_t token = 0; token < kPaperCompatTraceTokens; ++token) {
        double sum = 0.0;
        for (const double value : input[token]) {
            sum += value;
        }
        const double mean = sum / static_cast<double>(kPaperCompatHiddenSize);

        std::vector<double> centered(kPaperCompatHiddenSize);
        double squared_sum = 0.0;
        for (std::size_t feature = 0;
             feature < kPaperCompatHiddenSize;
             ++feature) {
            centered[feature] = input[token][feature] - mean;
            squared_sum += centered[feature] * centered[feature];
        }
        const double variance =
            squared_sum / static_cast<double>(kPaperCompatHiddenSize) +
            kPaperCompatLayerNormEpsilon;
        const double normalized_variance = variance_scale * variance;
        observed.Observe(
            normalized_variance,
            inverse_sqrt.interval,
            label + " normalized variance");
        const double inverse = std::sqrt(variance_scale) *
            EvaluateChebyshev(normalized_variance, inverse_sqrt);
        for (std::size_t feature = 0;
             feature < kPaperCompatHiddenSize;
             ++feature) {
            output[token][feature] = gamma[feature] * centered[feature] * inverse +
                beta[feature];
        }
    }
    ValidateMatrix(
        output,
        kPaperCompatTraceTokens,
        kPaperCompatHiddenSize,
        label + " output");
    return output;
}

PlainMatrix PadHidden(const PlainMatrix& input) {
    ValidateMatrix(
        input,
        kPaperCompatTraceTokens,
        kPaperCompatHiddenSize,
        "oracle final output");
    PlainMatrix output(
        kPaperCompatTraceTokens,
        std::vector<double>(kPaperCompatFeatureBlock, 0.0));
    for (std::size_t token = 0; token < kPaperCompatTraceTokens; ++token) {
        std::copy(input[token].begin(), input[token].end(), output[token].begin());
    }
    return output;
}

}  // namespace

EncoderPlaintextOracleResult EvaluateEncoderLayerPlaintextOracle(
    const PlainMatrix& input,
    const EncoderLayerWeights& weights) {
    ValidateMatrix(
        input,
        kPaperCompatTraceTokens,
        kPaperCompatFeatureBlock,
        "feature-packed oracle input");
    ValidateWeights(weights);

    const auto contracts = MakePaperCompatNonlinearContracts();
    ValidateContracts(contracts);

    const auto query = DenseAffine(
        input,
        weights.query_weights,
        weights.query_bias,
        "query projection");
    const auto key = DenseAffine(
        input,
        weights.key_weights,
        weights.key_bias,
        "key projection");
    const auto value = DenseAffine(
        input,
        weights.value_weights,
        weights.value_bias,
        "value projection");

    RangeAccumulator shifted_range;
    RangeAccumulator denominator_range;
    PlainMatrix attention(
        kPaperCompatTraceTokens,
        std::vector<double>(kPaperCompatHiddenSize, 0.0));
    for (std::size_t query_token = 0;
         query_token < kPaperCompatTraceTokens;
         ++query_token) {
        for (std::size_t head = 0;
             head < kPaperCompatAttentionHeads;
             ++head) {
            std::array<double, kPaperCompatTraceTokens> numerators{};
            for (std::size_t key_token = 0;
                 key_token < kPaperCompatTraceTokens;
                 ++key_token) {
                double dot = 0.0;
                for (std::size_t feature = 0;
                     feature < kHeadDimension;
                     ++feature) {
                    const std::size_t index = head * kHeadDimension + feature;
                    dot += query[query_token][index] * key[key_token][index];
                }
                const double shifted = dot / kAttentionScale -
                    contracts.softmax_shifts.At(weights.layer_index, head);
                shifted_range.Observe(
                    shifted,
                    contracts.softmax_exponential.interval,
                    "Softmax shifted logits");
                numerators[key_token] = EvaluateChebyshev(
                    shifted,
                    contracts.softmax_exponential);
            }

            double denominator = 0.0;
            for (const double numerator : numerators) {
                denominator += numerator;
            }
            denominator_range.Observe(
                denominator,
                contracts.softmax_reciprocal.interval,
                "Softmax denominator");
            const double inverse_denominator = EvaluateChebyshev(
                denominator,
                contracts.softmax_reciprocal);
            for (std::size_t feature = 0;
                 feature < kHeadDimension;
                 ++feature) {
                const std::size_t index = head * kHeadDimension + feature;
                double weighted_value = 0.0;
                for (std::size_t key_token = 0;
                     key_token < kPaperCompatTraceTokens;
                     ++key_token) {
                    const double probability =
                        numerators[key_token] * inverse_denominator;
                    weighted_value += probability * value[key_token][index];
                }
                attention[query_token][index] = weighted_value;
            }
        }
    }
    ValidateMatrix(
        attention,
        kPaperCompatTraceTokens,
        kPaperCompatHiddenSize,
        "attention output");

    const auto self_projection = DenseAffine(
        attention,
        weights.self_output_weights,
        weights.self_output_bias,
        "self-output projection");
    PlainMatrix active_input(
        kPaperCompatTraceTokens,
        std::vector<double>(kPaperCompatHiddenSize));
    for (std::size_t token = 0; token < kPaperCompatTraceTokens; ++token) {
        std::copy_n(
            input[token].begin(),
            kPaperCompatHiddenSize,
            active_input[token].begin());
    }
    const auto self_residual = Add(
        self_projection,
        active_input,
        "attention residual");
    RangeAccumulator attention_variance_range;
    const auto self_output = LayerNorm(
        self_residual,
        weights.attention_layernorm_gamma,
        weights.attention_layernorm_beta,
        kPaperCompatLayerNorm1VarianceScale,
        contracts.layernorm_inverse_sqrt,
        attention_variance_range,
        "attention LayerNorm");

    RangeAccumulator gelu_range;
    std::array<PlainMatrix, kPaperCompatIntermediateBlocks> activations;
    for (std::size_t block = 0;
         block < kPaperCompatIntermediateBlocks;
         ++block) {
        auto intermediate = DenseAffine(
            self_output,
            weights.intermediate_weight_blocks[block],
            weights.intermediate_bias_blocks[block],
            "intermediate projection block " + std::to_string(block));
        for (auto& row : intermediate) {
            for (double& activation : row) {
                gelu_range.Observe(
                    activation,
                    contracts.gelu.interval,
                    "GELU input");
                activation = EvaluateChebyshev(activation, contracts.gelu);
            }
        }
        activations[block] = std::move(intermediate);
    }

    PlainMatrix output_projection(
        kPaperCompatTraceTokens,
        weights.output_bias);
    for (std::size_t block = 0;
         block < kPaperCompatIntermediateBlocks;
         ++block) {
        const std::vector<double> zero_bias(kPaperCompatHiddenSize, 0.0);
        const auto contribution = DenseAffine(
            activations[block],
            weights.output_weight_blocks[block],
            zero_bias,
            "output projection block " + std::to_string(block));
        output_projection = Add(
            output_projection,
            contribution,
            "output projection accumulation");
    }
    const auto output_residual = Add(
        output_projection,
        self_output,
        "output residual");
    RangeAccumulator output_variance_range;
    const auto output = LayerNorm(
        output_residual,
        weights.output_layernorm_gamma,
        weights.output_layernorm_beta,
        kPaperCompatLayerNorm2VarianceScale,
        contracts.layernorm_inverse_sqrt,
        output_variance_range,
        "output LayerNorm");

    EncoderPlaintextOracleResult result;
    result.output = PadHidden(output);
    result.ranges = {
        shifted_range.Finalize("Softmax shifted logits"),
        denominator_range.Finalize("Softmax denominator"),
        attention_variance_range.Finalize("attention LayerNorm variance"),
        gelu_range.Finalize("GELU input"),
        output_variance_range.Finalize("output LayerNorm variance")};
    return result;
}

}  // namespace moai::openfhe::test
