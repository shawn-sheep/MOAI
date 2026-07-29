#include "moai/openfhe/packing.hpp"

#include "moai/openfhe/context_factory.hpp"

#include <cmath>
#include <stdexcept>

namespace moai::openfhe {

PackingSpec MakeFullSlotInterleavedPacking(
    const CryptoProfile& profile,
    PackingLayout layout,
    std::size_t logical_rows,
    std::size_t logical_columns) {
    ValidateCryptoProfile(profile);
    if (profile.slot_count != kMoaiFullSlotCount) {
        throw std::invalid_argument("MOAI full-slot packing requires 32768 slots");
    }
    if (logical_rows == 0 || logical_rows > kMoaiRowsPerLane ||
        logical_columns == 0) {
        throw std::invalid_argument("invalid logical shape for MOAI packing");
    }
    if (layout != PackingLayout::kColumn &&
        layout != PackingLayout::kDiagonal &&
        layout != PackingLayout::kInterleaved) {
        throw std::invalid_argument(
            "full-slot MOAI packing requires column, diagonal, or interleaved layout");
    }

    PackingSpec packing;
    packing.layout = layout;
    packing.logical_shape = {logical_rows, logical_columns};
    packing.batch_lanes = kMoaiBatchLanes;
    packing.slot_count = kMoaiFullSlotCount;
    packing.active_slots = kMoaiFullSlotCount;
    packing.encoded_slots = kMoaiFullSlotCount;
    packing.level = 0;
    packing.noise_scale_degree = 1;
    packing.scaling_factor =
        std::ldexp(1.0, static_cast<int>(profile.scaling_modulus_bits));
    return packing;
}

void ValidateFullSlotInterleavedPacking(const PackingSpec& packing) {
    if (packing.batch_lanes != kMoaiBatchLanes ||
        packing.slot_count != kMoaiFullSlotCount ||
        packing.active_slots != kMoaiFullSlotCount ||
        packing.encoded_slots != kMoaiFullSlotCount) {
        throw std::invalid_argument(
            "MOAI interleaved packing must occupy all 32768 slots with 256 lanes");
    }
    if (packing.logical_shape.size() != 2 ||
        packing.logical_shape[0] == 0 ||
        packing.logical_shape[0] > kMoaiRowsPerLane ||
        packing.logical_shape[1] == 0) {
        throw std::invalid_argument("invalid MOAI interleaved logical shape");
    }
    if (packing.layout != PackingLayout::kColumn &&
        packing.layout != PackingLayout::kDiagonal &&
        packing.layout != PackingLayout::kInterleaved) {
        throw std::invalid_argument("invalid MOAI interleaved layout");
    }
}

std::size_t InterleavedSlotIndex(
    std::size_t row,
    std::size_t lane,
    const PackingSpec& packing) {
    ValidateFullSlotInterleavedPacking(packing);
    if (row >= kMoaiRowsPerLane || lane >= packing.batch_lanes) {
        throw std::out_of_range("interleaved row or lane is out of range");
    }
    return row * packing.batch_lanes + lane;
}

std::vector<double> PackInterleavedRows(
    const std::vector<std::vector<double>>& lane_rows,
    const PackingSpec& packing) {
    ValidateFullSlotInterleavedPacking(packing);
    const auto logical_rows = packing.logical_shape[0];
    if (lane_rows.size() != packing.batch_lanes) {
        throw std::invalid_argument("lane count does not match PackingSpec.batch_lanes");
    }
    std::vector<double> slots(packing.slot_count, 0.0);
    for (std::size_t lane = 0; lane < lane_rows.size(); ++lane) {
        if (lane_rows[lane].size() != logical_rows) {
            throw std::invalid_argument("lane row count does not match logical shape");
        }
        for (std::size_t row = 0; row < logical_rows; ++row) {
            slots[InterleavedSlotIndex(row, lane, packing)] = lane_rows[lane][row];
        }
    }
    return slots;
}

std::vector<std::vector<double>> UnpackInterleavedRows(
    const std::vector<double>& slots,
    const PackingSpec& packing) {
    ValidateFullSlotInterleavedPacking(packing);
    if (slots.size() < packing.slot_count) {
        throw std::invalid_argument("decoded slot vector is shorter than slot_count");
    }
    const auto logical_rows = packing.logical_shape[0];
    std::vector<std::vector<double>> lane_rows(
        packing.batch_lanes,
        std::vector<double>(logical_rows));
    for (std::size_t lane = 0; lane < lane_rows.size(); ++lane) {
        for (std::size_t row = 0; row < logical_rows; ++row) {
            lane_rows[lane][row] = slots[InterleavedSlotIndex(row, lane, packing)];
        }
    }
    return lane_rows;
}

}  // namespace moai::openfhe
