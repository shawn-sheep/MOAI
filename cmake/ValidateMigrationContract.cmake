set(required_files
    "${MOAI_SOURCE_DIR}/config/paper_compat.json"
    "${MOAI_SOURCE_DIR}/docs/openfhe-artifact-schema.json"
    "${MOAI_SOURCE_DIR}/docs/openfhe-migration-contract.md"
    "${MOAI_SOURCE_DIR}/docs/openfhe-baseline.json")

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

file(READ "${MOAI_SOURCE_DIR}/docs/openfhe-migration-contract.md" migration_contract)
foreach(required_term IN ITEMS "server-only" "PrivateKey" "Decryptor" "12-layer encoder trace replay")
  if(NOT migration_contract MATCHES "${required_term}")
    message(FATAL_ERROR "Migration contract is missing required term: ${required_term}")
  endif()
endforeach()

message(STATUS "MOAI OpenFHE migration contract is internally consistent")
