set(server_linked_files
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/types.hpp"
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/context_factory.hpp"
  "${MOAI_SOURCE_DIR}/src/openfhe/context_factory.cpp"
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/packing.hpp"
  "${MOAI_SOURCE_DIR}/src/openfhe/packing.cpp"
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/approximation_registry.hpp"
  "${MOAI_SOURCE_DIR}/src/openfhe/approximation_registry.cpp"
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/server_runtime.hpp"
  "${MOAI_SOURCE_DIR}/src/openfhe/server_runtime.cpp"
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/linear_ops.hpp"
  "${MOAI_SOURCE_DIR}/src/openfhe/linear_ops.cpp"
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/nonlinear_ops.hpp"
  "${MOAI_SOURCE_DIR}/src/openfhe/nonlinear_ops.cpp")

foreach(server_file IN LISTS server_linked_files)
  if(NOT EXISTS "${server_file}")
    message(FATAL_ERROR "Missing server runtime file: ${server_file}")
  endif()

  file(READ "${server_file}" server_source)
  foreach(forbidden_term IN ITEMS
      "lbcrypto::PrivateKey"
      "PrivateKey<"
      "private_key_"
      "privateKey"
      "Decryptor"
      "->Decrypt("
      ".Decrypt(")
    string(FIND "${server_source}" "${forbidden_term}" forbidden_position)
    if(NOT forbidden_position EQUAL -1)
      message(FATAL_ERROR
        "Server trust-boundary violation in ${server_file}: ${forbidden_term}")
    endif()
  endforeach()
endforeach()

set(server_operator_files
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/server_runtime.hpp"
  "${MOAI_SOURCE_DIR}/src/openfhe/server_runtime.cpp"
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/linear_ops.hpp"
  "${MOAI_SOURCE_DIR}/src/openfhe/linear_ops.cpp"
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/nonlinear_ops.hpp"
  "${MOAI_SOURCE_DIR}/src/openfhe/nonlinear_ops.cpp")

foreach(server_file IN LISTS server_operator_files)
  file(READ "${server_file}" server_source)
  string(FIND "${server_source}" "DeclaredRange" forbidden_position)
  if(NOT forbidden_position EQUAL -1)
    message(FATAL_ERROR
      "Server operator accepts activation-derived range metadata in ${server_file}")
  endif()
endforeach()

message(STATUS
  "OpenFHE server-linked sources contain no private-key/decryption interface; "
  "operator sources expose no activation-derived range metadata")
