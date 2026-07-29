#include "moai/openfhe/client_runtime.hpp"
#include "moai/openfhe/context_factory.hpp"
#include "moai/openfhe/linear_ops.hpp"
#include "moai/openfhe/packing.hpp"
#include "moai/openfhe/server_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using moai::openfhe::kMoaiBatchLanes;
using moai::openfhe::kMoaiFullSlotCount;
using moai::openfhe::kMoaiRowsPerLane;

struct QualityMetrics {
    double relative_l2{0.0};
    double cosine{0.0};
    double max_absolute{0.0};
};

QualityMetrics MeasureQuality(
    const std::vector<std::vector<double>>& actual,
    const std::vector<std::vector<double>>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("quality tensor count mismatch");
    }
    long double squared_error = 0.0;
    long double squared_actual = 0.0;
    long double squared_expected = 0.0;
    long double dot_product = 0.0;
    double maximum = 0.0;
    for (std::size_t tensor = 0; tensor < actual.size(); ++tensor) {
        if (actual[tensor].size() != expected[tensor].size()) {
            throw std::runtime_error("quality vector size mismatch");
        }
        for (std::size_t slot = 0; slot < actual[tensor].size(); ++slot) {
            if (!std::isfinite(actual[tensor][slot])) {
                throw std::runtime_error("linear kernel output is not finite");
            }
            const long double actual_value = actual[tensor][slot];
            const long double expected_value = expected[tensor][slot];
            const long double error = actual_value - expected_value;
            squared_error += error * error;
            squared_actual += actual_value * actual_value;
            squared_expected += expected_value * expected_value;
            dot_product += actual_value * expected_value;
            maximum = std::max(maximum, std::abs(static_cast<double>(error)));
        }
    }
    if (squared_expected == 0.0 || squared_actual == 0.0) {
        throw std::runtime_error("quality metric received a zero-norm tensor");
    }
    QualityMetrics metrics;
    metrics.relative_l2 =
        std::sqrt(static_cast<double>(squared_error / squared_expected));
    metrics.cosine = static_cast<double>(
        dot_product / std::sqrt(squared_actual * squared_expected));
    metrics.max_absolute = maximum;
    return metrics;
}

void RequireQuality(
    const QualityMetrics& metrics,
    const std::string& label) {
    if (metrics.relative_l2 > 1e-4) {
        throw std::runtime_error(
            label + " rel-L2 exceeds 1e-4: " +
            std::to_string(metrics.relative_l2));
    }
    if (metrics.cosine < 0.99999) {
        throw std::runtime_error(
            label + " cosine is below 0.99999: " +
            std::to_string(metrics.cosine));
    }
}

double XValue(std::size_t lane, std::size_t row, std::size_t column) {
    const double phase =
        0.017 * static_cast<double>((lane + 1) * (column + 2)) +
        0.11 * static_cast<double>(row + 1);
    return 0.12 * std::sin(phase) +
        0.03 * std::cos(0.07 * static_cast<double>(lane + row + column + 3));
}

double YValue(std::size_t lane, std::size_t row, std::size_t column) {
    const double phase =
        0.013 * static_cast<double>((lane + 3) * (column + 1)) -
        0.09 * static_cast<double>(row + 1);
    return 0.10 * std::cos(phase) -
        0.025 * std::sin(0.05 * static_cast<double>(2 * lane + row + column + 1));
}

std::vector<std::vector<double>> MakeColumnSlots(
    const moai::openfhe::PackingSpec& packing,
    bool make_x) {
    const std::size_t columns = packing.logical_shape[1];
    std::vector<std::vector<double>> result(
        columns,
        std::vector<double>(kMoaiFullSlotCount));
    for (std::size_t column = 0; column < columns; ++column) {
        for (std::size_t row = 0; row < kMoaiRowsPerLane; ++row) {
            for (std::size_t lane = 0; lane < kMoaiBatchLanes; ++lane) {
                const std::size_t slot = row * kMoaiBatchLanes + lane;
                result[column][slot] = make_x
                    ? XValue(lane, row, column)
                    : YValue(lane, row, column);
            }
        }
    }
    return result;
}

std::vector<std::vector<double>> CtPtOracle(
    const std::vector<std::vector<double>>& input,
    const moai::openfhe::PlainMatrix& weights) {
    std::vector<std::vector<double>> output(
        weights.front().size(),
        std::vector<double>(kMoaiFullSlotCount));
    for (std::size_t out = 0; out < output.size(); ++out) {
        for (std::size_t in = 0; in < input.size(); ++in) {
            for (std::size_t slot = 0; slot < kMoaiFullSlotCount; ++slot) {
                output[out][slot] += input[in][slot] * weights[in][out];
            }
        }
    }
    return output;
}

std::vector<std::vector<double>> ColumnToDiagonalOracle(
    const std::vector<std::vector<double>>& lhs,
    const std::vector<std::vector<double>>& rhs,
    const std::vector<uint32_t>& offsets) {
    std::vector<std::vector<double>> output(
        offsets.size(),
        std::vector<double>(kMoaiFullSlotCount));
    for (std::size_t diagonal = 0; diagonal < offsets.size(); ++diagonal) {
        const std::size_t offset = offsets[diagonal];
        for (std::size_t row = 0; row < kMoaiRowsPerLane; ++row) {
            const std::size_t rhs_row = (row + offset) % kMoaiRowsPerLane;
            for (std::size_t lane = 0; lane < kMoaiBatchLanes; ++lane) {
                const std::size_t lhs_slot = row * kMoaiBatchLanes + lane;
                const std::size_t rhs_slot = rhs_row * kMoaiBatchLanes + lane;
                for (std::size_t column = 0; column < lhs.size(); ++column) {
                    output[diagonal][lhs_slot] +=
                        lhs[column][lhs_slot] * rhs[column][rhs_slot];
                }
            }
        }
    }
    return output;
}

std::vector<std::vector<double>> DiagonalColumnOracle(
    const std::vector<std::vector<double>>& diagonals,
    const std::vector<uint32_t>& offsets,
    const std::vector<std::vector<double>>& rhs) {
    std::vector<std::vector<double>> output(
        rhs.size(),
        std::vector<double>(kMoaiFullSlotCount));
    for (std::size_t column = 0; column < rhs.size(); ++column) {
        for (std::size_t diagonal = 0; diagonal < offsets.size(); ++diagonal) {
            const std::size_t offset = offsets[diagonal];
            for (std::size_t row = 0; row < kMoaiRowsPerLane; ++row) {
                const std::size_t rhs_row = (row + offset) % kMoaiRowsPerLane;
                for (std::size_t lane = 0; lane < kMoaiBatchLanes; ++lane) {
                    const std::size_t output_slot =
                        row * kMoaiBatchLanes + lane;
                    const std::size_t rhs_slot =
                        rhs_row * kMoaiBatchLanes + lane;
                    output[column][output_slot] +=
                        diagonals[diagonal][output_slot] *
                        rhs[column][rhs_slot];
                }
            }
        }
    }
    return output;
}

}  // namespace

int main() {
    try {
        auto profile = moai::openfhe::MakePaperCompatProfile();
        moai::openfhe::PrintSecurityDisclosure(profile, std::cout);
        const std::vector<int32_t> rotation_indices{
            static_cast<int32_t>(kMoaiBatchLanes),
            static_cast<int32_t>(2 * kMoaiBatchLanes),
            static_cast<int32_t>(3 * kMoaiBatchLanes),
            -static_cast<int32_t>(2 * kMoaiBatchLanes)};

        double sentinel_active_error = 0.0;
        double sentinel_inactive_error = 0.0;
        double sentinel_cross_lane_error = 0.0;
        QualityMetrics ct_pt_quality;
        QualityMetrics column_diagonal_quality;
        QualityMetrics bsgs_quality;
        moai::openfhe::RunMetrics operation_counts;

        {
            moai::openfhe::ClientRuntime client(profile);
            client.GenerateEvaluationKeys(rotation_indices, false);
            moai::openfhe::ServerRuntime server(
                client.ExportServerKeyBundle(),
                profile);
            moai::openfhe::LinearOps linear_ops(server);

            const auto sentinel_packing =
                moai::openfhe::MakeFullSlotInterleavedPacking(
                    profile,
                    moai::openfhe::PackingLayout::kInterleaved,
                    kMoaiRowsPerLane - 1,
                    1);
            std::vector<std::vector<double>> sentinel_lanes(
                kMoaiBatchLanes,
                std::vector<double>(kMoaiRowsPerLane - 1));
            for (std::size_t lane = 0; lane < kMoaiBatchLanes; ++lane) {
                for (std::size_t row = 0;
                     row < kMoaiRowsPerLane - 1;
                     ++row) {
                    sentinel_lanes[lane][row] =
                        (static_cast<double>(lane) - 127.5) / 512.0 +
                        (static_cast<double>(row) - 63.0) / 4096.0;
                }
            }
            const auto sentinel_slots =
                moai::openfhe::PackInterleavedRows(
                    sentinel_lanes,
                    sentinel_packing);
            const auto encrypted_sentinel =
                client.Encrypt({sentinel_slots}, sentinel_packing);
            const auto decoded_sentinel =
                client.Decrypt(encrypted_sentinel).front();
            const auto unpacked_sentinel =
                moai::openfhe::UnpackInterleavedRows(
                    decoded_sentinel,
                    sentinel_packing);
            for (std::size_t lane = 0; lane < kMoaiBatchLanes; ++lane) {
                for (std::size_t row = 0;
                     row < kMoaiRowsPerLane - 1;
                     ++row) {
                    sentinel_active_error = std::max(
                        sentinel_active_error,
                        std::abs(
                            unpacked_sentinel[lane][row] -
                            sentinel_lanes[lane][row]));
                }
                sentinel_inactive_error = std::max(
                    sentinel_inactive_error,
                    std::abs(
                        decoded_sentinel[
                            (kMoaiRowsPerLane - 1) * kMoaiBatchLanes + lane]));
            }

            const auto rotated_sentinel =
                server.Rotate(encrypted_sentinel, kMoaiBatchLanes);
            const auto decoded_rotated =
                client.Decrypt(rotated_sentinel).front();
            for (std::size_t row = 0; row < kMoaiRowsPerLane; ++row) {
                for (std::size_t lane = 0; lane < kMoaiBatchLanes; ++lane) {
                    double expected = 0.0;
                    if (row < kMoaiRowsPerLane - 2) {
                        expected = sentinel_lanes[lane][row + 1];
                    } else if (row == kMoaiRowsPerLane - 1) {
                        expected = sentinel_lanes[lane][0];
                    }
                    sentinel_cross_lane_error = std::max(
                        sentinel_cross_lane_error,
                        std::abs(
                            decoded_rotated[row * kMoaiBatchLanes + lane] -
                            expected));
                }
            }
            if (sentinel_active_error > 1e-6 ||
                sentinel_inactive_error > 1e-6 ||
                sentinel_cross_lane_error > 1e-6) {
                throw std::runtime_error(
                    "32768-slot/256-lane sentinel packing gate exceeds 1e-6");
            }

            const auto column_packing =
                moai::openfhe::MakeFullSlotInterleavedPacking(
                    profile,
                    moai::openfhe::PackingLayout::kColumn,
                    kMoaiRowsPerLane,
                    2);
            const auto x_slots = MakeColumnSlots(column_packing, true);
            const auto y_slots = MakeColumnSlots(column_packing, false);
            const auto encrypted_x = client.Encrypt(x_slots, column_packing);
            const auto encrypted_y = client.Encrypt(y_slots, column_packing);

            const moai::openfhe::PlainMatrix weights{
                {0.75, -0.20},
                {-0.45, 0.55}};
            const auto ct_pt =
                linear_ops.CtPtColumnMatMul(encrypted_x, weights);
            const auto ct_pt_expected = CtPtOracle(x_slots, weights);
            ct_pt_quality = MeasureQuality(
                client.Decrypt(ct_pt),
                ct_pt_expected);
            RequireQuality(ct_pt_quality, "Ct-Pt column matmul");

            const std::vector<uint32_t> diagonal_offsets{0, 1, 2, 3};
            const auto diagonals =
                linear_ops.CtCtColumnMatMulToDiagonal(
                    encrypted_x,
                    encrypted_y,
                    diagonal_offsets);
            const auto diagonal_expected =
                ColumnToDiagonalOracle(
                    x_slots,
                    y_slots,
                    diagonal_offsets);
            column_diagonal_quality = MeasureQuality(
                client.Decrypt(diagonals.values),
                diagonal_expected);
            RequireQuality(
                column_diagonal_quality,
                "Ct-Ct column-to-diagonal matmul");

            const auto bsgs =
                linear_ops.CtCtDiagonalColumnMatMulBsgs(
                    diagonals,
                    encrypted_y,
                    2);
            const auto bsgs_expected =
                DiagonalColumnOracle(
                    diagonal_expected,
                    diagonal_offsets,
                    y_slots);
            bsgs_quality = MeasureQuality(
                client.Decrypt(bsgs),
                bsgs_expected);
            RequireQuality(
                bsgs_quality,
                "Ct-Ct diagonal-column BSGS matmul");

            operation_counts = server.metrics();
            if (operation_counts.ct_pt_multiplications == 0 ||
                operation_counts.ct_ct_multiplications == 0 ||
                operation_counts.rotations == 0 ||
                operation_counts.rescale_operations == 0) {
                throw std::runtime_error(
                    "linear-kernel operation counters were not populated");
            }
        }

        std::cout
            << "{\"test\":\"openfhe_packing_linear_smoke\","
            << "\"profile\":\"paper_compat\","
            << "\"security_claim\":\"none\","
            << "\"slots\":" << kMoaiFullSlotCount << ","
            << "\"batch_lanes\":" << kMoaiBatchLanes << ","
            << "\"sentinel_logical_rows\":" << kMoaiRowsPerLane - 1 << ","
            << "\"sentinel_active_max_abs\":" << sentinel_active_error << ","
            << "\"inactive_max_abs\":" << sentinel_inactive_error << ","
            << "\"cross_lane_max_abs\":" << sentinel_cross_lane_error << ","
            << "\"ct_pt_rel_l2\":" << ct_pt_quality.relative_l2 << ","
            << "\"ct_pt_cosine\":" << ct_pt_quality.cosine << ","
            << "\"ct_pt_max_abs\":" << ct_pt_quality.max_absolute << ","
            << "\"column_diagonal_rel_l2\":"
            << column_diagonal_quality.relative_l2 << ","
            << "\"column_diagonal_cosine\":"
            << column_diagonal_quality.cosine << ","
            << "\"column_diagonal_max_abs\":"
            << column_diagonal_quality.max_absolute << ","
            << "\"bsgs_rel_l2\":" << bsgs_quality.relative_l2 << ","
            << "\"bsgs_cosine\":" << bsgs_quality.cosine << ","
            << "\"bsgs_max_abs\":" << bsgs_quality.max_absolute << ","
            << "\"rotations\":" << operation_counts.rotations << ","
            << "\"ct_pt\":" << operation_counts.ct_pt_multiplications << ","
            << "\"ct_ct\":" << operation_counts.ct_ct_multiplications << ","
            << "\"rescale\":" << operation_counts.rescale_operations
            << "}\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "openfhe_packing_linear_smoke failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
