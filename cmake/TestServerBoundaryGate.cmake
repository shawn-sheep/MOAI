foreach(required_variable IN ITEMS
    MOAI_SOURCE_DIR
    MOAI_OPENFHE_PREFIX
    MOAI_TEST_BINARY_ROOT)
  if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
    message(FATAL_ERROR
      "TestServerBoundaryGate requires ${required_variable}")
  endif()
endforeach()

function(expect_server_boundary_configure_failure
    case_name
    expected_message
    injection_content)
  set(case_root "${MOAI_TEST_BINARY_ROOT}/${case_name}")
  set(case_build "${case_root}/build")
  set(injection_file "${case_root}/inject.cmake")
  file(REMOVE_RECURSE "${case_root}")
  file(MAKE_DIRECTORY "${case_root}")
  file(WRITE "${injection_file}" "${injection_content}")

  execute_process(
    COMMAND
      "${CMAKE_COMMAND}"
      -S "${MOAI_SOURCE_DIR}"
      -B "${case_build}"
      -DBUILD_TESTING=OFF
      "-DMOAI_OPENFHE_PREFIX=${MOAI_OPENFHE_PREFIX}"
      "-DCMAKE_PROJECT_INCLUDE=${injection_file}"
      ${ARGN}
    RESULT_VARIABLE configure_result
    OUTPUT_VARIABLE configure_stdout
    ERROR_VARIABLE configure_stderr
  )
  set(configure_output "${configure_stdout}\n${configure_stderr}")
  if(configure_result EQUAL 0)
    message(FATAL_ERROR
      "${case_name}: unsafe nested configure unexpectedly succeeded")
  endif()
  string(FIND
    "${configure_output}"
    "${expected_message}"
    expected_message_position)
  if(expected_message_position EQUAL -1)
    message(FATAL_ERROR
      "${case_name}: nested configure failed for the wrong reason; expected "
      "'${expected_message}', observed:\n${configure_output}")
  endif()
  message(STATUS "${case_name}: failed closed as expected")
endfunction()

set(transitive_client_injection [=[
function(moai_test_inject_transitive_client_link)
  target_link_libraries(OPENFHEpke INTERFACE moai_openfhe_client)
endfunction()
cmake_language(
  DEFER
  DIRECTORY "${CMAKE_SOURCE_DIR}"
  CALL moai_test_inject_transitive_client_link
)
]=])
expect_server_boundary_configure_failure(
  transitive-client-target
  "Server link closure reaches unapproved target"
  "${transitive_client_injection}"
)

set(transitive_client_source_injection [=[
function(moai_test_inject_transitive_client_source)
  target_sources(
    OPENFHEpke
    INTERFACE
    "${CMAKE_SOURCE_DIR}/src/openfhe/client_runtime.cpp"
  )
endfunction()
cmake_language(
  DEFER
  DIRECTORY "${CMAKE_SOURCE_DIR}"
  CALL moai_test_inject_transitive_client_source
)
]=])
expect_server_boundary_configure_failure(
  transitive-client-source
  "OPENFHEpke INTERFACE_SOURCES was added"
  "${transitive_client_source_injection}"
)

set(generator_expression_injection [=[
function(moai_test_inject_generator_expression)
  target_link_libraries(
    OPENFHEpke
    INTERFACE
    "$<LINK_ONLY:moai_openfhe_client>"
  )
endfunction()
cmake_language(
  DEFER
  DIRECTORY "${CMAKE_SOURCE_DIR}"
  CALL moai_test_inject_generator_expression
)
]=])
expect_server_boundary_configure_failure(
  generator-expression-link
  "Server link closure contains an unauditable generator expression"
  "${generator_expression_injection}"
)

set(client_pch_injection [=[
function(moai_test_inject_client_pch)
  target_precompile_headers(
    moai_openfhe_server
    PRIVATE
    "${CMAKE_SOURCE_DIR}/include/moai/openfhe/client_runtime.hpp"
  )
endfunction()
cmake_language(
  DEFER
  DIRECTORY "${CMAKE_SOURCE_DIR}"
  CALL moai_test_inject_client_pch
)
]=])
expect_server_boundary_configure_failure(
  client-precompiled-header
  "PRECOMPILE_HEADERS must remain unset"
  "${client_pch_injection}"
)

set(source_forced_include_injection [=[
function(moai_test_inject_source_forced_include)
  set_source_files_properties(
    src/openfhe/server_runtime.cpp
    PROPERTIES
    COMPILE_OPTIONS
    "-include;${CMAKE_SOURCE_DIR}/include/moai/openfhe/client_runtime.hpp"
  )
endfunction()
cmake_language(
  DEFER
  DIRECTORY "${CMAKE_SOURCE_DIR}"
  CALL moai_test_inject_source_forced_include
)
]=])
expect_server_boundary_configure_failure(
  source-forced-include
  "src/openfhe/server_runtime.cpp COMPILE_OPTIONS"
  "${source_forced_include_injection}"
)

expect_server_boundary_configure_failure(
  global-forced-include
  "contains forbidden forced-include input"
  ""
  "-DCMAKE_CXX_FLAGS=-include${MOAI_SOURCE_DIR}/include/moai/openfhe/evaluation_key_registry.hpp"
)

message(STATUS
  "Server trust-boundary negative configure cases all failed closed")
