#include "moai/openfhe/evaluation_key_registry.hpp"

namespace moai::openfhe {

std::shared_mutex& EvaluationKeyRegistryMutex() {
    static std::shared_mutex mutex;
    return mutex;
}

}  // namespace moai::openfhe
