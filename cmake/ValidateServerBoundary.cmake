set(server_files
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/server_runtime.hpp"
  "${MOAI_SOURCE_DIR}/src/openfhe/server_runtime.cpp"
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/linear_ops.hpp"
  "${MOAI_SOURCE_DIR}/src/openfhe/linear_ops.cpp")

foreach(server_file IN LISTS server_files)
  if(NOT EXISTS "${server_file}")
    message(FATAL_ERROR "Missing server runtime file: ${server_file}")
  endif()

  file(READ "${server_file}" server_source)
  foreach(forbidden_term IN ITEMS
      "PrivateKey"
      "SecretKey"
      "secret_key"
      "secretKey"
      "Decryptor"
      "Decrypt(")
    string(FIND "${server_source}" "${forbidden_term}" forbidden_position)
    if(NOT forbidden_position EQUAL -1)
      message(FATAL_ERROR
        "Server trust-boundary violation in ${server_file}: ${forbidden_term}")
    endif()
  endforeach()
endforeach()

message(STATUS
  "OpenFHE server sources contain no private-key or decryption interface")
