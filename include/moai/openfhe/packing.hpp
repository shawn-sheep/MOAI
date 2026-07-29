#pragma once

#include "moai/openfhe/types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace moai::openfhe {

inline constexpr uint32_t kMoaiBatchLanes = 256;
inline constexpr uint32_t kMoaiRowsPerLane = 128;
inline constexpr uint32_t kMoaiFullSlotCount =
    kMoaiBatchLanes * kMoaiRowsPerLane;

[[nodiscard]] PackingSpec MakeFullSlotInterleavedPacking(
    const CryptoProfile& profile,
    PackingLayout layout,
    std::size_t logical_rows,
    std::size_t logical_columns);

void ValidateFullSlotInterleavedPacking(const PackingSpec& packing);

[[nodiscard]] std::size_t InterleavedSlotIndex(
    std::size_t row,
    std::size_t lane,
    const PackingSpec& packing);

[[nodiscard]] std::vector<double> PackInterleavedRows(
    const std::vector<std::vector<double>>& lane_rows,
    const PackingSpec& packing);

[[nodiscard]] std::vector<std::vector<double>> UnpackInterleavedRows(
    const std::vector<double>& slots,
    const PackingSpec& packing);

}  // namespace moai::openfhe
