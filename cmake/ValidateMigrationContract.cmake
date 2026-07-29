set(required_files
    "${MOAI_SOURCE_DIR}/config/paper_compat.json"
    "${MOAI_SOURCE_DIR}/docs/openfhe-artifact-schema.json"
    "${MOAI_SOURCE_DIR}/docs/openfhe-migration-contract.md"
    "${MOAI_SOURCE_DIR}/docs/openfhe-baseline.json"
    "${MOAI_SOURCE_DIR}/include/moai/openfhe/types.hpp"
    "${MOAI_SOURCE_DIR}/include/moai/openfhe/client_runtime.hpp"
    "${MOAI_SOURCE_DIR}/include/moai/openfhe/server_runtime.hpp"
    "${MOAI_SOURCE_DIR}/src/openfhe/context_factory.cpp"
    "${MOAI_SOURCE_DIR}/src/openfhe/client_runtime.cpp"
    "${MOAI_SOURCE_DIR}/src/openfhe/server_runtime.cpp"
    "${MOAI_SOURCE_DIR}/tests/openfhe_fast_smoke.cpp"
    "${MOAI_SOURCE_DIR}/tests/openfhe_bootstrap_smoke.cpp")

foreach(required_file IN LISTS required_files)
  if(NOT EXISTS "${required_file}")
    message(FATAL_ERROR "Missing migration contract file: ${required_file}")
  endif()
endforeach()

file(READ "${MOAI_SOURCE_DIR}/config/paper_compat.json" paper_profile)
if(NOT paper_profile MATCHES "\"profile_id\"[ \t\r\n]*:[ \t\r\n]*\"paper_compat\"")
  message(FATAL_ERROR "paper_compat profile_id is missing")
endif()
if(NOT paper_profile MATCHES "\"security_claim\"[ \t\r\n]*:[ \t\r\n]*\"none\"")
  message(FATAL_ERROR "paper_compat must declare security_claim=none")
endif()
if(NOT paper_profile MATCHES "\"status\"[ \t\r\n]*:[ \t\r\n]*\"m1_api_validated\"")
  message(FATAL_ERROR "paper_compat OpenFHE translation is not M1-validated")
endif()
if(NOT paper_profile MATCHES "\"secret_key_distribution\"[ \t\r\n]*:[ \t\r\n]*\"SPARSE_TERNARY\"")
  message(FATAL_ERROR "paper_compat must map h=192 to SPARSE_TERNARY")
endif()

file(READ "${MOAI_SOURCE_DIR}/docs/openfhe-migration-contract.md" migration_contract)
foreach(required_term IN ITEMS
    "server-only"
    "PrivateKey"
    "Decryptor"
    "12-layer encoder trace replay")
  if(NOT migration_contract MATCHES "${required_term}")
    message(FATAL_ERROR "Migration contract is missing required term: ${required_term}")
  endif()
endforeach()

message(STATUS "MOAI OpenFHE migration contract is internally consistent")
