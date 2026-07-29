foreach(required_variable IN ITEMS
    MOAI_SOURCE_DIR
    MOAI_SERVER_LINKED_SOURCE_MANIFEST
    MOAI_SERVER_OPERATOR_SOURCE_MANIFEST)
  if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
    message(FATAL_ERROR
      "ValidateServerBoundary requires ${required_variable}")
  endif()
endforeach()

function(read_target_source_manifest manifest_path output_variable)
  if(NOT EXISTS "${manifest_path}")
    message(FATAL_ERROR
      "Missing generated server target source manifest: ${manifest_path}")
  endif()

  file(STRINGS "${manifest_path}" manifest_entries)
  if(NOT manifest_entries)
    message(FATAL_ERROR
      "Generated server target source manifest is empty: ${manifest_path}")
  endif()

  file(REAL_PATH "${MOAI_SOURCE_DIR}" source_root)
  set(resolved_sources)
  foreach(source_entry IN LISTS manifest_entries)
    if(IS_ABSOLUTE "${source_entry}")
      set(source_path "${source_entry}")
    else()
      set(source_path "${MOAI_SOURCE_DIR}/${source_entry}")
    endif()
    if(NOT EXISTS "${source_path}")
      message(FATAL_ERROR
        "Server target source does not exist: ${source_entry}")
    endif()

    file(REAL_PATH "${source_path}" resolved_source)
    file(RELATIVE_PATH relative_source "${source_root}" "${resolved_source}")
    if(relative_source MATCHES "^\\.\\./" OR IS_ABSOLUTE "${relative_source}")
      message(FATAL_ERROR
        "Server target source escapes the MOAI repository: ${resolved_source}")
    endif()
    list(APPEND resolved_sources "${resolved_source}")
  endforeach()

  set(unique_sources ${resolved_sources})
  list(REMOVE_DUPLICATES unique_sources)
  list(LENGTH resolved_sources source_count)
  list(LENGTH unique_sources unique_source_count)
  if(NOT source_count EQUAL unique_source_count)
    message(FATAL_ERROR
      "Generated server target source manifest contains duplicate sources: ${manifest_path}")
  endif()
  set(${output_variable} "${resolved_sources}" PARENT_SCOPE)
endfunction()

read_target_source_manifest(
  "${MOAI_SERVER_LINKED_SOURCE_MANIFEST}"
  server_linked_sources)
read_target_source_manifest(
  "${MOAI_SERVER_OPERATOR_SOURCE_MANIFEST}"
  server_operator_sources)

foreach(operator_source IN LISTS server_operator_sources)
  list(FIND server_linked_sources "${operator_source}" linked_source_index)
  if(linked_source_index EQUAL -1)
    message(FATAL_ERROR
      "Server operator target source is absent from the server-linked source set: "
      "${operator_source}")
  endif()
endforeach()

set(server_common_headers
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/types.hpp"
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/context_factory.hpp"
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/evaluation_key_registry.hpp"
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/packing.hpp"
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/approximation_registry.hpp")

set(server_operator_headers
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/server_runtime.hpp"
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/linear_ops.hpp"
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/feature_packed_ops.hpp"
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/feature_packed_attention.hpp"
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/nonlinear_ops.hpp"
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/encoder_layer.hpp")

set(server_linked_files
  ${server_common_headers}
  ${server_operator_headers}
  ${server_linked_sources})

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
  ${server_operator_headers}
  ${server_operator_sources})

foreach(server_file IN LISTS server_operator_files)
  file(READ "${server_file}" server_source)
  string(FIND "${server_source}" "DeclaredRange" forbidden_position)
  if(NOT forbidden_position EQUAL -1)
    message(FATAL_ERROR
      "Server operator accepts activation-derived range metadata in ${server_file}")
  endif()
endforeach()

message(STATUS
  "Generated OpenFHE server target sources and public headers contain no "
  "private-key/decryption interface; "
  "operator sources expose no activation-derived range metadata")
