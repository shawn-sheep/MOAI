#include "moai/openfhe/linear_ops.hpp"

#include "moai/openfhe/packing.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace moai::openfhe {
namespace {

void ValidateColumnTensor(const CipherTensor& tensor) {
    ValidateFullSlotInterleavedPacking(tensor.packing);
    if (tensor.packing.layout != PackingLayout::kColumn ||
        tensor.packing.logical_shape[0] != kMoaiRowsPerLane ||
        tensor.size() != tensor.packing.logical_shape[1]) {
        throw std::invalid_argument(
            "linear kernels require full-row column-packed tensors");
    }
}

CipherTensor SelectCiphertext(const CipherTensor& tensor, std::size_t index) {
    if (index >= tensor.size()) {
        throw std::out_of_range("ciphertext index is out of range");
    }
    CipherTensor selected;
    selected.packing = tensor.packing;
    if (selected.packing.logical_shape.size() == 2) {
        selected.packing.logical_shape[1] = 1;
    }
    selected.ciphertexts.push_back(tensor.ciphertexts[index]);
    return selected;
}

void AppendSingle(CipherTensor& output, CipherTensor&& single) {
    if (single.size() != 1) {
        throw std::logic_error("linear kernel expected one ciphertext");
    }
    if (output.ciphertexts.empty()) {
        output.packing = single.packing;
    }
    output.ciphertexts.push_back(std::move(single.ciphertexts.front()));
}

}  // namespace

CipherTensor LinearOps::CtPtColumnMatMul(
    const CipherTensor& column_input,
    const PlainMatrix& weights) {
    ValidateColumnTensor(column_input);
    if (weights.size() != column_input.size() || weights.empty() ||
        weights.front().empty()) {
        throw std::invalid_argument("Ct-Pt matrix dimensions do not match");
    }
    const std::size_t output_columns = weights.front().size();
    for (const auto& row : weights) {
        if (row.size() != output_columns) {
            throw std::invalid_argument("Ct-Pt weight matrix is ragged");
        }
    }

    CipherTensor output;
    for (std::size_t output_column = 0;
         output_column < output_columns;
         ++output_column) {
        std::vector<lbcrypto::Plaintext> encoded_weights;
        encoded_weights.reserve(weights.size());
        for (std::size_t input_column = 0;
             input_column < weights.size();
             ++input_column) {
            const std::vector<double> repeated_weight(
                column_input.packing.active_slots,
                weights[input_column][output_column]);
            encoded_weights.push_back(
                server_.EncodeModelVector(repeated_weight, column_input.packing));
        }
        auto products = server_.MultiplyPlain(column_input, encoded_weights);
        auto summed = server_.Sum(products);
        AppendSingle(output, server_.Rescale(summed));
    }
    output.packing.layout = PackingLayout::kColumn;
    output.packing.logical_shape = {
        kMoaiRowsPerLane,
        output_columns};
    return output;
}

DiagonalCipherTensor LinearOps::CtCtColumnMatMulToDiagonal(
    const CipherTensor& lhs_columns,
    const CipherTensor& rhs_columns,
    const std::vector<uint32_t>& diagonal_offsets) {
    ValidateColumnTensor(lhs_columns);
    ValidateColumnTensor(rhs_columns);
    if (lhs_columns.size() != rhs_columns.size() || diagonal_offsets.empty()) {
        throw std::invalid_argument("Ct-Ct column matrix dimensions do not match");
    }
    auto sorted_offsets = diagonal_offsets;
    std::sort(sorted_offsets.begin(), sorted_offsets.end());
    if (std::adjacent_find(sorted_offsets.begin(), sorted_offsets.end()) !=
        sorted_offsets.end() || sorted_offsets.back() >= kMoaiRowsPerLane) {
        throw std::invalid_argument("diagonal offsets must be unique and below 128");
    }

    DiagonalCipherTensor output;
    output.offsets = diagonal_offsets;
    for (const uint32_t offset : diagonal_offsets) {
        auto aligned_rhs = offset == 0
            ? rhs_columns
            : server_.Rotate(
                  rhs_columns,
                  static_cast<int32_t>(offset * kMoaiBatchLanes));
        auto products = server_.Multiply(lhs_columns, aligned_rhs);
        auto summed = server_.Sum(products);
        AppendSingle(output.values, server_.Rescale(summed));
    }
    output.values.packing.layout = PackingLayout::kDiagonal;
    output.values.packing.logical_shape = {
        kMoaiRowsPerLane,
        kMoaiRowsPerLane};
    return output;
}

CipherTensor LinearOps::CtCtDiagonalColumnMatMulBsgs(
    const DiagonalCipherTensor& lhs_diagonals,
    const CipherTensor& rhs_columns,
    uint32_t baby_step) {
    ValidateFullSlotInterleavedPacking(lhs_diagonals.values.packing);
    ValidateColumnTensor(rhs_columns);
    if (lhs_diagonals.values.packing.layout != PackingLayout::kDiagonal ||
        lhs_diagonals.values.size() != lhs_diagonals.offsets.size() ||
        lhs_diagonals.offsets.empty() || baby_step == 0 ||
        baby_step > kMoaiRowsPerLane) {
        throw std::invalid_argument("invalid diagonal BSGS input contract");
    }
    auto sorted_offsets = lhs_diagonals.offsets;
    std::sort(sorted_offsets.begin(), sorted_offsets.end());
    if (std::adjacent_find(sorted_offsets.begin(), sorted_offsets.end()) !=
        sorted_offsets.end() || sorted_offsets.back() >= kMoaiRowsPerLane) {
        throw std::invalid_argument("invalid BSGS diagonal offsets");
    }

    const uint32_t giant_count = sorted_offsets.back() / baby_step + 1;
    CipherTensor output;
    for (std::size_t output_column = 0;
         output_column < rhs_columns.size();
         ++output_column) {
        const auto rhs = SelectCiphertext(rhs_columns, output_column);
        std::vector<CipherTensor> baby_rotations;
        baby_rotations.reserve(baby_step);
        baby_rotations.push_back(rhs);
        for (uint32_t baby = 1; baby < baby_step; ++baby) {
            baby_rotations.push_back(server_.Rotate(
                rhs,
                static_cast<int32_t>(baby * kMoaiBatchLanes)));
        }

        CipherTensor total;
        bool has_total = false;
        for (uint32_t giant = 0; giant < giant_count; ++giant) {
            CipherTensor group;
            bool has_group = false;
            const uint32_t giant_offset = giant * baby_step;
            for (std::size_t diagonal_index = 0;
                 diagonal_index < lhs_diagonals.offsets.size();
                 ++diagonal_index) {
                const uint32_t offset = lhs_diagonals.offsets[diagonal_index];
                if (offset / baby_step != giant) {
                    continue;
                }
                auto diagonal =
                    SelectCiphertext(lhs_diagonals.values, diagonal_index);
                if (giant_offset != 0) {
                    diagonal = server_.Rotate(
                        diagonal,
                        -static_cast<int32_t>(
                            giant_offset * kMoaiBatchLanes));
                }
                auto product = server_.Multiply(
                    diagonal,
                    baby_rotations[offset % baby_step]);
                group = has_group ? server_.Add(group, product) : std::move(product);
                has_group = true;
            }
            if (!has_group) {
                continue;
            }
            if (giant_offset != 0) {
                group = server_.Rotate(
                    group,
                    static_cast<int32_t>(
                        giant_offset * kMoaiBatchLanes));
            }
            total = has_total ? server_.Add(total, group) : std::move(group);
            has_total = true;
        }
        if (!has_total) {
            throw std::logic_error("BSGS produced no diagonal groups");
        }
        AppendSingle(output, server_.Rescale(total));
    }
    output.packing.layout = PackingLayout::kColumn;
    output.packing.logical_shape = {
        kMoaiRowsPerLane,
        rhs_columns.size()};
    return output;
}

}  // namespace moai::openfhe
