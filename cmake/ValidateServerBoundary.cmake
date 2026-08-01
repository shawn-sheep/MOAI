foreach(required_variable IN ITEMS
    MOAI_SOURCE_DIR
    MOAI_SERVER_LINKED_SOURCE_MANIFEST
    MOAI_SERVER_OPERATOR_SOURCE_MANIFEST)
  if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
    message(FATAL_ERROR
      "ValidateServerBoundary requires ${required_variable}")
  endif()
endforeach()

file(REAL_PATH "${MOAI_SOURCE_DIR}" moai_source_root)

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
    if(NOT EXISTS "${source_path}" OR IS_DIRECTORY "${source_path}")
      message(FATAL_ERROR
        "Server target source is not a regular file: ${source_entry}")
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

function(collect_repository_include_closure root_files output_variable)
  file(REAL_PATH "${MOAI_SOURCE_DIR}" source_root)
  set(scan_queue ${root_files})
  set(include_closure)

  while(scan_queue)
    list(POP_FRONT scan_queue source_entry)
    file(REAL_PATH "${source_entry}" resolved_source)
    file(RELATIVE_PATH relative_source "${source_root}" "${resolved_source}")
    if(relative_source MATCHES "^\\.\\./" OR IS_ABSOLUTE "${relative_source}")
      message(FATAL_ERROR
        "Server include closure escapes the MOAI repository: ${resolved_source}")
    endif()

    list(FIND include_closure "${resolved_source}" existing_index)
    if(NOT existing_index EQUAL -1)
      continue()
    endif()
    list(APPEND include_closure "${resolved_source}")

    get_filename_component(source_directory "${resolved_source}" DIRECTORY)
    file(STRINGS "${resolved_source}" include_directives
      REGEX "^[ \t]*#[ \t]*include")
    file(STRINGS "${resolved_source}" include_lines
      REGEX "^[ \t]*#[ \t]*include[ \t]*[<\"][^>\"]+[>\"]")
    list(LENGTH include_directives include_directive_count)
    list(LENGTH include_lines literal_include_count)
    if(NOT include_directive_count EQUAL literal_include_count)
      message(FATAL_ERROR
        "Server source contains a macro, continued, or otherwise unauditable "
        "include directive: ${resolved_source}")
    endif()
    foreach(include_line IN LISTS include_lines)
      string(REGEX REPLACE
        "^[ \t]*#[ \t]*include[ \t]*[<\"]([^>\"]+)[>\"].*$"
        "\\1"
        include_entry
        "${include_line}")
      string(REGEX MATCH "[<\"]" include_delimiter "${include_line}")
      if(include_delimiter STREQUAL "\"")
        set(include_candidates
          "${source_directory}/${include_entry}"
          "${source_root}/include/${include_entry}")
      else()
        set(include_candidates
          "${source_root}/include/${include_entry}")
      endif()
      set(repository_include "")
      set(escaped_repository_candidate "")
      foreach(include_candidate IN LISTS include_candidates)
        if(EXISTS "${include_candidate}" AND NOT IS_DIRECTORY "${include_candidate}")
          file(REAL_PATH "${include_candidate}" resolved_include)
          file(RELATIVE_PATH
            relative_include
            "${source_root}"
            "${resolved_include}")
          if(NOT relative_include MATCHES "^\\.\\./" AND
             NOT IS_ABSOLUTE "${relative_include}")
            set(repository_include "${resolved_include}")
            break()
          else()
            set(escaped_repository_candidate "${resolved_include}")
          endif()
        endif()
      endforeach()
      if(escaped_repository_candidate)
        message(FATAL_ERROR
          "Server include candidate resolves outside the MOAI repository: "
          "${include_entry} -> ${escaped_repository_candidate}")
      elseif(repository_include)
        list(APPEND scan_queue "${repository_include}")
      elseif(include_entry MATCHES "^moai/" OR
             (include_delimiter STREQUAL "\"" AND
              NOT include_entry MATCHES
                "^(openfhe\\.h|utils/hashutil\\.h|math/chebyshev\\.h)$") OR
             (include_delimiter STREQUAL "<" AND include_entry MATCHES "/"))
        message(FATAL_ERROR
          "Unresolved or unaudited server include: ${include_entry} "
          "from ${resolved_source}")
      endif()
    endforeach()
  endwhile()

  set(${output_variable} "${include_closure}" PARENT_SCOPE)
endfunction()

collect_repository_include_closure(
  "${server_linked_files}"
  server_linked_include_closure)

foreach(required_header IN ITEMS
    "${MOAI_SOURCE_DIR}/include/moai/openfhe/types.hpp"
    "${MOAI_SOURCE_DIR}/include/moai/openfhe/server_runtime.hpp"
    "${MOAI_SOURCE_DIR}/include/moai/openfhe/encoder_layer.hpp")
  file(REAL_PATH "${required_header}" resolved_required_header)
  list(FIND
    server_linked_include_closure
    "${resolved_required_header}"
    required_header_index)
  if(required_header_index EQUAL -1)
    message(FATAL_ERROR
      "Server include-closure audit missed required header: ${required_header}")
  endif()
endforeach()

file(REAL_PATH
  "${MOAI_SOURCE_DIR}/include/moai/openfhe/client_runtime.hpp"
  client_runtime_header)
list(FIND
  server_linked_include_closure
  "${client_runtime_header}"
  client_runtime_header_index)
if(NOT client_runtime_header_index EQUAL -1)
  message(FATAL_ERROR
    "Server include closure reaches the client runtime interface")
endif()

foreach(server_file IN LISTS server_linked_include_closure)
  if(NOT EXISTS "${server_file}")
    message(FATAL_ERROR "Missing server runtime file: ${server_file}")
  endif()

  file(RELATIVE_PATH
    repository_server_file
    "${moai_source_root}"
    "${server_file}")
  if(repository_server_file MATCHES
      "^(src/openfhe/client|include/moai/openfhe/client)")
    message(FATAL_ERROR
      "Server source/include closure reaches first-party client code: "
      "${repository_server_file}")
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
  "Generated OpenFHE server target source/include closure contains no "
  "first-party client or private-key/decryption interface; "
  "operator sources expose no activation-derived range metadata")
