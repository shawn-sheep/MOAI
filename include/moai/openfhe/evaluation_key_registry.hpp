#pragma once

#include <shared_mutex>

namespace moai::openfhe {

// OpenFHE 1.5.1 stores evaluation keys in process-global static maps.  All
// MOAI access to those maps must use this application-level mutex.
[[nodiscard]] std::shared_mutex& EvaluationKeyRegistryMutex();

}  // namespace moai::openfhe
