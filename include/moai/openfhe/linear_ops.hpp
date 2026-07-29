#pragma once

#include "moai/openfhe/server_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace moai::openfhe {

using PlainMatrix = std::vector<std::vector<double>>;

struct DiagonalCipherTensor {
    CipherTensor values;
    std::vector<uint32_t> offsets;
};

class LinearOps {
public:
    explicit LinearOps(ServerRuntime& server) : server_(server) {}

    [[nodiscard]] CipherTensor CtPtColumnMatMul(
        const CipherTensor& column_input,
        const PlainMatrix& weights);

    [[nodiscard]] DiagonalCipherTensor CtCtColumnMatMulToDiagonal(
        const CipherTensor& lhs_columns,
        const CipherTensor& rhs_columns,
        const std::vector<uint32_t>& diagonal_offsets);

    [[nodiscard]] CipherTensor CtCtDiagonalColumnMatMulBsgs(
        const DiagonalCipherTensor& lhs_diagonals,
        const CipherTensor& rhs_columns,
        uint32_t baby_step);

private:
    ServerRuntime& server_;
};

}  // namespace moai::openfhe
