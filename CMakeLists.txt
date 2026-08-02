cmake_minimum_required(VERSION 3.20)
project(moai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MOAI_ENABLE_SEAL_REFERENCE "Build the legacy SEAL reference executable" OFF)
set(
  MOAI_OPENFHE_PREFIX
  "/home/shawnsheep/opt/openfhe_v1_5_1"
  CACHE PATH
  "OpenFHE v1.5.1 installation prefix"
)
set(
  MOAI_PYTHON_EXECUTABLE
  ""
  CACHE FILEPATH
  "Python 3.10/3.11 interpreter with NumPy for MOAI validators"
)
set(
  MOAI_TRACE_DATA_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/data"
  CACHE PATH
  "Existing MOAI trace data root (read in place)"
)


set(OpenFHE_DIR "${MOAI_OPENFHE_PREFIX}/lib/OpenFHE" CACHE PATH "OpenFHE package directory")
find_package(OpenFHE 1.5.1 EXACT CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# Freeze the imported dependency graph immediately after the trusted package
# loads. The deferred server-boundary gate compares these exact properties
# again after every project target has been declared, so later mutations of an
# imported target cannot smuggle first-party client code into the server.
set(MOAI_SERVER_FROZEN_IMPORTED_TARGETS
  OPENFHEpke
  OPENFHEcore
  OPENFHEbinfhe
  OpenMP::OpenMP_CXX
)
set(MOAI_SERVER_ALLOWED_FIRST_PARTY_TARGETS
  moai_openfhe_server
  moai_openfhe_common
  moai_openfhe_vendor
)
set(MOAI_SERVER_ALLOWED_TARGET_CLOSURE
  ${MOAI_SERVER_ALLOWED_FIRST_PARTY_TARGETS}
  ${MOAI_SERVER_FROZEN_IMPORTED_TARGETS}
)
set(MOAI_SERVER_LINK_PROPERTIES
  LINK_LIBRARIES
  INTERFACE_LINK_LIBRARIES
  INTERFACE_LINK_LIBRARIES_DIRECT
  IMPORTED_LINK_INTERFACE_LIBRARIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELWITHDEBINFO
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELWITHDEBINFO
  IMPORTED_LINK_INTERFACE_LIBRARIES_MINSIZEREL
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL
  IMPORTED_LINK_INTERFACE_LIBRARIES_NOCONFIG
  IMPORTED_LINK_DEPENDENT_LIBRARIES_NOCONFIG
)
set(MOAI_SERVER_FROZEN_IMPORTED_PROPERTIES
  TYPE
  IMPORTED
  IMPORTED_CONFIGURATIONS
  IMPORTED_LOCATION
  IMPORTED_LOCATION_DEBUG
  IMPORTED_LOCATION_RELEASE
  IMPORTED_LOCATION_RELWITHDEBINFO
  IMPORTED_LOCATION_MINSIZEREL
  IMPORTED_LOCATION_NOCONFIG
  ${MOAI_SERVER_LINK_PROPERTIES}
  INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE
  INTERFACE_LINK_OPTIONS
  INTERFACE_LINK_DIRECTORIES
  SOURCES
  INTERFACE_SOURCES
  PRECOMPILE_HEADERS
  PRECOMPILE_HEADERS_REUSE_FROM
  INTERFACE_PRECOMPILE_HEADERS
  COMPILE_OPTIONS
  INTERFACE_COMPILE_OPTIONS
  COMPILE_FLAGS
  COMPILE_DEFINITIONS
  COMPILE_DEFINITIONS_DEBUG
  COMPILE_DEFINITIONS_RELEASE
  COMPILE_DEFINITIONS_RELWITHDEBINFO
  COMPILE_DEFINITIONS_MINSIZEREL
  INTERFACE_COMPILE_DEFINITIONS
  INCLUDE_DIRECTORIES
  INTERFACE_INCLUDE_DIRECTORIES
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
)

function(moai_snapshot_imported_target target snapshot_prefix)
  if(NOT TARGET "${target}")
    message(FATAL_ERROR
      "Required frozen server dependency target does not exist: ${target}")
  endif()
  get_target_property(imported "${target}" IMPORTED)
  if(NOT imported)
    message(FATAL_ERROR
      "Frozen server dependency must be an imported target: ${target}")
  endif()

  foreach(property IN LISTS MOAI_SERVER_FROZEN_IMPORTED_PROPERTIES)
    get_property(property_is_set TARGET "${target}" PROPERTY "${property}" SET)
    set(
      "MOAI_FROZEN_${snapshot_prefix}_${property}_SET"
      "${property_is_set}"
      PARENT_SCOPE
    )
    if(property_is_set)
      get_target_property(property_value "${target}" "${property}")
      set(
        "MOAI_FROZEN_${snapshot_prefix}_${property}_VALUE"
        "${property_value}"
        PARENT_SCOPE
      )
    endif()
  endforeach()

  set(raw_link_items "${MOAI_SERVER_FROZEN_RAW_LINK_ITEMS}")
  foreach(property IN LISTS MOAI_SERVER_LINK_PROPERTIES)
    get_property(property_is_set TARGET "${target}" PROPERTY "${property}" SET)
    if(NOT property_is_set)
      continue()
    endif()
    get_target_property(property_value "${target}" "${property}")
    foreach(link_item IN LISTS property_value)
      if("${link_item}" MATCHES "\\$<")
        message(FATAL_ERROR
          "Frozen imported server graph contains an unauditable generator "
          "expression in ${target} ${property}: ${link_item}")
      elseif(TARGET "${link_item}")
        continue()
      endif()
      list(APPEND raw_link_items "${link_item}")
    endforeach()
  endforeach()
  list(REMOVE_DUPLICATES raw_link_items)
  set(MOAI_SERVER_FROZEN_RAW_LINK_ITEMS "${raw_link_items}" PARENT_SCOPE)
endfunction()

moai_snapshot_imported_target(OPENFHEpke OPENFHE_PKE)
moai_snapshot_imported_target(OPENFHEcore OPENFHE_CORE)
moai_snapshot_imported_target(OPENFHEbinfhe OPENFHE_BINFHE)
moai_snapshot_imported_target(OpenMP::OpenMP_CXX OPENMP_CXX)

set(MOAI_SERVER_FROZEN_IMPORTED_SNAPSHOTS
  "OPENFHEpke|OPENFHE_PKE"
  "OPENFHEcore|OPENFHE_CORE"
  "OPENFHEbinfhe|OPENFHE_BINFHE"
  "OpenMP::OpenMP_CXX|OPENMP_CXX"
)

set(MOAI_SERVER_FROZEN_CXX_FLAG_VARIABLES
  CMAKE_CXX_FLAGS
  CMAKE_CXX_FLAGS_DEBUG
  CMAKE_CXX_FLAGS_RELEASE
  CMAKE_CXX_FLAGS_RELWITHDEBINFO
  CMAKE_CXX_FLAGS_MINSIZEREL
)
if(CMAKE_BUILD_TYPE)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" MOAI_SERVER_BUILD_TYPE_UPPER)
  list(APPEND
    MOAI_SERVER_FROZEN_CXX_FLAG_VARIABLES
    "CMAKE_CXX_FLAGS_${MOAI_SERVER_BUILD_TYPE_UPPER}"
  )
endif()
list(REMOVE_DUPLICATES MOAI_SERVER_FROZEN_CXX_FLAG_VARIABLES)
foreach(flag_variable IN LISTS MOAI_SERVER_FROZEN_CXX_FLAG_VARIABLES)
  set(
    "MOAI_FROZEN_${flag_variable}"
    "${${flag_variable}}"
  )
endforeach()

# OpenFHE 1.5.1 exports un-namespaced targets without include directories.
add_library(moai_openfhe_vendor INTERFACE)
target_include_directories(moai_openfhe_vendor SYSTEM INTERFACE
  "${OpenFHE_INCLUDE}"
  "${OpenFHE_INCLUDE}/core"
  "${OpenFHE_INCLUDE}/pke"
  "${OpenFHE_INCLUDE}/binfhe"
)
target_compile_definitions(moai_openfhe_vendor INTERFACE
  OPENFHE_VERSION=${BASE_OPENFHE_VERSION}
)
target_link_libraries(moai_openfhe_vendor INTERFACE
  OPENFHEpke
  OpenMP::OpenMP_CXX
)

set(MOAI_OPENFHE_COMMON_SOURCES
  src/openfhe/approximation_registry.cpp
  src/openfhe/context_factory.cpp
  src/openfhe/evaluation_key_registry.cpp
  src/openfhe/packing.cpp
)
add_library(moai_openfhe_common STATIC ${MOAI_OPENFHE_COMMON_SOURCES})
target_include_directories(moai_openfhe_common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(moai_openfhe_common PUBLIC moai_openfhe_vendor)
add_library(moai::openfhe_common ALIAS moai_openfhe_common)

set(MOAI_OPENFHE_SERVER_SOURCES
  src/openfhe/server_runtime.cpp
  src/openfhe/linear_ops.cpp
  src/openfhe/feature_packed_ops.cpp
  src/openfhe/feature_packed_attention.cpp
  src/openfhe/nonlinear_ops.cpp
  src/openfhe/encoder_layer.cpp
)
add_library(moai_openfhe_server STATIC ${MOAI_OPENFHE_SERVER_SOURCES})
target_include_directories(moai_openfhe_server PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(moai_openfhe_server PUBLIC moai_openfhe_common)
add_library(moai::openfhe_server ALIAS moai_openfhe_server)

add_library(moai_openfhe_client STATIC src/openfhe/client_runtime.cpp)
target_include_directories(moai_openfhe_client PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(moai_openfhe_client PUBLIC moai_openfhe_common)
add_library(moai::openfhe_client ALIAS moai_openfhe_client)

if(MOAI_ENABLE_SEAL_REFERENCE)
  find_package(SEAL 4.1 REQUIRED)
  add_subdirectory(include/source/bootstrapping/common)
  add_subdirectory(include/source/bootstrapping)

  add_executable(moai_seal_reference
    test.cpp
    ${BOOTSTRAPPING_SOURCE_FILES}
    ${COMMON_SOURCE_FILES}
  )
  target_include_directories(moai_seal_reference PRIVATE
    /usr/local/include
    /usr/local/include/NTL
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/source"
    "${BOOTSTRAPPING_HEADER_DIR}"
    "${COMMON_HEADER_DIR}"
  )
  target_compile_options(moai_seal_reference PRIVATE -O3)
  target_link_libraries(moai_seal_reference PRIVATE
    crypto ntl gmp m pthread SEAL::seal OpenMP::OpenMP_CXX
  )
endif()

set(
  MOAI_SERVER_LINKED_SOURCE_MANIFEST
  "${CMAKE_CURRENT_BINARY_DIR}/moai-server-linked-sources.txt"
)
set(
  MOAI_SERVER_OPERATOR_SOURCE_MANIFEST
  "${CMAKE_CURRENT_BINARY_DIR}/moai-server-operator-sources.txt"
)
set(MOAI_OPENFHE_LINKED_SOURCES
  ${MOAI_OPENFHE_COMMON_SOURCES}
  ${MOAI_OPENFHE_SERVER_SOURCES}
)
string(REPLACE
  ";"
  "\n"
  MOAI_OPENFHE_LINKED_SOURCE_LINES
  "${MOAI_OPENFHE_LINKED_SOURCES}"
)
string(REPLACE
  ";"
  "\n"
  MOAI_OPENFHE_SERVER_SOURCE_LINES
  "${MOAI_OPENFHE_SERVER_SOURCES}"
)
file(WRITE
  "${MOAI_SERVER_LINKED_SOURCE_MANIFEST}"
  "${MOAI_OPENFHE_LINKED_SOURCE_LINES}\n"
)
file(WRITE
  "${MOAI_SERVER_OPERATOR_SOURCE_MANIFEST}"
  "${MOAI_OPENFHE_SERVER_SOURCE_LINES}\n"
)
execute_process(
  COMMAND
    "${CMAKE_COMMAND}"
    "-DMOAI_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
    "-DMOAI_SERVER_LINKED_SOURCE_MANIFEST=${MOAI_SERVER_LINKED_SOURCE_MANIFEST}"
    "-DMOAI_SERVER_OPERATOR_SOURCE_MANIFEST=${MOAI_SERVER_OPERATOR_SOURCE_MANIFEST}"
    -P
    "${CMAKE_SOURCE_DIR}/cmake/ValidateServerBoundary.cmake"
  RESULT_VARIABLE MOAI_SERVER_BOUNDARY_CONFIGURE_RESULT
  OUTPUT_VARIABLE MOAI_SERVER_BOUNDARY_CONFIGURE_OUTPUT
  ERROR_VARIABLE MOAI_SERVER_BOUNDARY_CONFIGURE_ERROR
)
if(NOT MOAI_SERVER_BOUNDARY_CONFIGURE_RESULT EQUAL 0)
  message(FATAL_ERROR
    "OpenFHE server trust-boundary configure audit failed:\n"
    "${MOAI_SERVER_BOUNDARY_CONFIGURE_OUTPUT}"
    "${MOAI_SERVER_BOUNDARY_CONFIGURE_ERROR}"
  )
endif()

include(CTest)
if(BUILD_TESTING)

  if(NOT MOAI_PYTHON_EXECUTABLE)
    set(
      _MOAI_PREFERRED_PYTHON
      "/home/shawnsheep/miniconda3/envs/fhe-inference/bin/python3.10"
    )
    if(EXISTS "${_MOAI_PREFERRED_PYTHON}" AND
       NOT IS_DIRECTORY "${_MOAI_PREFERRED_PYTHON}")
      set(
        MOAI_PYTHON_EXECUTABLE
        "${_MOAI_PREFERRED_PYTHON}"
        CACHE FILEPATH
        "Python 3.10/3.11 interpreter with NumPy for MOAI validators"
        FORCE
      )
    else()
      find_package(Python3 3.10...<3.12 COMPONENTS Interpreter REQUIRED)
      set(
        MOAI_PYTHON_EXECUTABLE
        "${Python3_EXECUTABLE}"
        CACHE FILEPATH
        "Python 3.10/3.11 interpreter with NumPy for MOAI validators"
        FORCE
      )
    endif()
  endif()
  execute_process(
    COMMAND "${MOAI_PYTHON_EXECUTABLE}" -c
            "assert (3, 10) <= __import__('sys').version_info[:2] < (3, 12) and __import__('numpy') is not None"
    RESULT_VARIABLE _MOAI_PYTHON_CHECK_RESULT
    ERROR_VARIABLE _MOAI_PYTHON_CHECK_ERROR
    OUTPUT_QUIET
  )
  if(NOT _MOAI_PYTHON_CHECK_RESULT EQUAL 0)
    message(FATAL_ERROR
      "MOAI validators require Python 3.10/3.11 with NumPy; "
      "set MOAI_PYTHON_EXECUTABLE. ${_MOAI_PYTHON_CHECK_ERROR}"
    )
  endif()
  add_test(
    NAME migration_contract
    COMMAND ${CMAKE_COMMAND}
            -DMOAI_SOURCE_DIR=${CMAKE_SOURCE_DIR}
            -P ${CMAKE_SOURCE_DIR}/cmake/ValidateMigrationContract.cmake
  )

  add_test(
    NAME openfhe_server_trust_boundary
    COMMAND ${CMAKE_COMMAND}
            -DMOAI_SOURCE_DIR=${CMAKE_SOURCE_DIR}
            -DMOAI_SERVER_LINKED_SOURCE_MANIFEST=${MOAI_SERVER_LINKED_SOURCE_MANIFEST}
            -DMOAI_SERVER_OPERATOR_SOURCE_MANIFEST=${MOAI_SERVER_OPERATOR_SOURCE_MANIFEST}
            -P ${CMAKE_SOURCE_DIR}/cmake/ValidateServerBoundary.cmake
  )
  add_test(
    NAME openfhe_server_trust_boundary_negative
    COMMAND
      ${CMAKE_COMMAND}
      -DMOAI_SOURCE_DIR=${CMAKE_SOURCE_DIR}
      -DMOAI_OPENFHE_PREFIX=${MOAI_OPENFHE_PREFIX}
      -DMOAI_TEST_BINARY_ROOT=${CMAKE_CURRENT_BINARY_DIR}/server-boundary-negative
      -P ${CMAKE_SOURCE_DIR}/cmake/TestServerBoundaryGate.cmake
  )
  set_tests_properties(openfhe_server_trust_boundary_negative PROPERTIES
    LABELS "m4;m5;plaintext;trust-boundary;negative"
    TIMEOUT 120
  )
  add_test(
    NAME openfhe_nonlinear_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/scripts/calibrate_openfhe_nonlinear.py"
            --config "${CMAKE_SOURCE_DIR}/config/openfhe_approximations.json"
            --data-root "${MOAI_TRACE_DATA_ROOT}"
  )
  set_tests_properties(openfhe_nonlinear_contract PROPERTIES
    LABELS "m3;plaintext;calibration;slow"
    TIMEOUT 900
  )

  add_test(
    NAME openfhe_profile_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/scripts/validate_openfhe_profile.py"
            --config "${CMAKE_SOURCE_DIR}/config/paper_compat.json"
            --feature-config
            "${CMAKE_SOURCE_DIR}/config/paper_compat_feature_packed.json"
  )
  set_tests_properties(openfhe_profile_contract PROPERTIES
    LABELS "m3;m4;m5;plaintext;profile"
    TIMEOUT 60
  )

  add_test(
    NAME openfhe_profile_validator_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tests/test_validate_openfhe_profile.py"
  )
  set_tests_properties(openfhe_profile_validator_contract PROPERTIES
    LABELS "m4;m5;plaintext;profile"
    TIMEOUT 60
  )

  add_test(
    NAME openfhe_artifact_schema_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/scripts/validate_openfhe_artifact.py"
            --schema "${CMAKE_SOURCE_DIR}/docs/openfhe-artifact-schema.json"
  )
  set_tests_properties(openfhe_artifact_schema_contract PROPERTIES
    LABELS "m3;m4;plaintext;artifact"
    TIMEOUT 60
  )

  add_test(
    NAME openfhe_artifact_validator_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tests/test_validate_openfhe_artifact.py"
  )
  set_tests_properties(openfhe_artifact_validator_contract PROPERTIES
    LABELS "m3;m4;plaintext;artifact"
    TIMEOUT 60
  )

  add_test(
    NAME openfhe_encoder_artifact_runner_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tests/test_run_openfhe_encoder_artifact.py"
  )
  set_tests_properties(openfhe_encoder_artifact_runner_contract PROPERTIES
    LABELS "m4;plaintext;artifact"
    TIMEOUT 60
  )

  add_test(
    NAME openfhe_m4_v5_artifact_schema_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/scripts/validate_openfhe_m4_artifact_v5.py"
            --schema
            "${CMAKE_SOURCE_DIR}/docs/openfhe-m4-artifact-schema-v5.json"
  )
  set_tests_properties(openfhe_m4_v5_artifact_schema_contract PROPERTIES
    LABELS "m4;plaintext;artifact;v5"
    TIMEOUT 60
  )

  add_test(
    NAME openfhe_m4_v5_artifact_validator_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tests/test_validate_openfhe_m4_artifact_v5.py"
  )
  set_tests_properties(openfhe_m4_v5_artifact_validator_contract PROPERTIES
    LABELS "m4;plaintext;artifact;v5"
    TIMEOUT 120
  )

  add_test(
    NAME openfhe_m4_v5_encoder_artifact_runner_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tests/test_run_openfhe_encoder_artifact_v5.py"
  )
  set_tests_properties(openfhe_m4_v5_encoder_artifact_runner_contract PROPERTIES
    LABELS "m4;plaintext;artifact;v5"
    TIMEOUT 60
  )

  add_test(
    NAME openfhe_m5_artifact_schema_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/scripts/validate_openfhe_m5_artifact.py"
            --schema "${CMAKE_SOURCE_DIR}/docs/openfhe-m5-artifact-schema.json"
  )
  set_tests_properties(openfhe_m5_artifact_schema_contract PROPERTIES
    LABELS "m5;plaintext;artifact"
    TIMEOUT 60
  )

  add_test(
    NAME openfhe_m5_artifact_validator_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tests/test_validate_openfhe_m5_artifact.py"
  )
  set_tests_properties(openfhe_m5_artifact_validator_contract PROPERTIES
    LABELS "m5;plaintext;artifact"
    TIMEOUT 60
  )

  add_test(
    NAME openfhe_encoder12_artifact_runner_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tests/test_run_openfhe_encoder12_artifact.py"
  )
  set_tests_properties(openfhe_encoder12_artifact_runner_contract PROPERTIES
    LABELS "m5;plaintext;artifact"
    TIMEOUT 60
  )

  add_test(
    NAME openfhe_m5_v6_artifact_schema_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/scripts/validate_openfhe_m5_artifact_v6.py"
            --schema
            "${CMAKE_SOURCE_DIR}/docs/openfhe-m5-artifact-schema-v6.json"
  )
  set_tests_properties(openfhe_m5_v6_artifact_schema_contract PROPERTIES
    LABELS "m5;plaintext;artifact;v6"
    TIMEOUT 60
  )

  add_test(
    NAME openfhe_m5_v6_artifact_validator_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tests/test_validate_openfhe_m5_artifact_v6.py"
  )
  set_tests_properties(openfhe_m5_v6_artifact_validator_contract PROPERTIES
    LABELS "m5;plaintext;artifact;v6"
    TIMEOUT 600
  )

  add_test(
    NAME openfhe_m5_v6_encoder12_artifact_runner_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tests/test_run_openfhe_encoder12_artifact_v6.py"
  )
  set_tests_properties(openfhe_m5_v6_encoder12_artifact_runner_contract PROPERTIES
    LABELS "m5;plaintext;artifact;v6"
    TIMEOUT 300
  )

  add_test(
    NAME openfhe_m6_benchmark_schema_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/scripts/validate_openfhe_m6_benchmark.py"
            --schema "${CMAKE_SOURCE_DIR}/docs/openfhe-m6-benchmark-schema-v1.json"
  )
  set_tests_properties(openfhe_m6_benchmark_schema_contract PROPERTIES
    LABELS "m6;plaintext;artifact;benchmark;v1"
    TIMEOUT 300
  )
  add_test(
    NAME openfhe_m6_benchmark_validator_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tests/test_validate_openfhe_m6_benchmark.py"
  )
  set_tests_properties(openfhe_m6_benchmark_validator_contract PROPERTIES
    LABELS "m6;plaintext;artifact;benchmark;v1"
    TIMEOUT 300
  )
  add_test(
    NAME openfhe_m6_benchmark_runner_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tests/test_run_openfhe_m6_benchmark.py"
  )
  set_tests_properties(openfhe_m6_benchmark_runner_contract PROPERTIES
    LABELS "m6;plaintext;artifact;benchmark;v1"
    TIMEOUT 300
  )

  add_test(
    NAME moai_trace_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/scripts/validate_moai_trace_contract.py"
            --data-root "${MOAI_TRACE_DATA_ROOT}"
            --manifest "${CMAKE_SOURCE_DIR}/config/moai_trace_channel_scales.json"
  )
  set_tests_properties(moai_trace_contract PROPERTIES
    LABELS "m3;plaintext;trace;slow"
    TIMEOUT 900
  )

  add_test(
    NAME openfhe_encoder_trace_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/scripts/validate_openfhe_encoder_trace.py"
            --data-root "${MOAI_TRACE_DATA_ROOT}"
            --manifest "${CMAKE_SOURCE_DIR}/config/moai_encoder_trace.json"
  )
  set_tests_properties(openfhe_encoder_trace_contract PROPERTIES
    LABELS "m4;m5;prerequisite;plaintext;trace;slow"
    TIMEOUT 900
  )

  add_test(
    NAME openfhe_encoder_trace_validator_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tests/test_validate_openfhe_encoder_trace.py"
  )
  set_tests_properties(openfhe_encoder_trace_validator_contract PROPERTIES
    LABELS "m4;m5;prerequisite;plaintext;trace;slow"
    TIMEOUT 900
  )


  add_executable(openfhe_fast_smoke tests/openfhe_fast_smoke.cpp)
  target_link_libraries(openfhe_fast_smoke PRIVATE
    moai::openfhe_client
    moai::openfhe_server
  )
  set_target_properties(openfhe_fast_smoke PROPERTIES BUILD_RPATH "${OpenFHE_LIBDIR}")
  target_link_options(openfhe_fast_smoke PRIVATE "-Wl,--disable-new-dtags")
  add_test(NAME openfhe_fast_smoke COMMAND openfhe_fast_smoke)
  set_tests_properties(openfhe_fast_smoke PROPERTIES LABELS "openfhe;fast")

  add_executable(
    openfhe_serialized_size_smoke
    tests/openfhe_serialized_size_smoke.cpp
  )
  target_link_libraries(openfhe_serialized_size_smoke PRIVATE
    moai::openfhe_client
  )
  set_target_properties(
    openfhe_serialized_size_smoke
    PROPERTIES BUILD_RPATH "${OpenFHE_LIBDIR}"
  )
  target_link_options(
    openfhe_serialized_size_smoke
    PRIVATE "-Wl,--disable-new-dtags"
  )
  add_test(
    NAME openfhe_serialized_size_smoke
    COMMAND openfhe_serialized_size_smoke
  )
  set_tests_properties(openfhe_serialized_size_smoke PROPERTIES
    LABELS "openfhe;fast;m6;serialization;client-only"
    TIMEOUT 300
  )

  add_executable(
    openfhe_packing_linear_smoke
    tests/openfhe_packing_linear_smoke.cpp
  )
  target_link_libraries(openfhe_packing_linear_smoke PRIVATE
    moai::openfhe_client
    moai::openfhe_server
  )
  set_target_properties(
    openfhe_packing_linear_smoke
    PROPERTIES BUILD_RPATH "${OpenFHE_LIBDIR}"
  )
  target_link_options(
    openfhe_packing_linear_smoke
    PRIVATE "-Wl,--disable-new-dtags"
  )
  add_test(
    NAME openfhe_packing_linear_smoke
    COMMAND openfhe_packing_linear_smoke
  )
  set_tests_properties(openfhe_packing_linear_smoke PROPERTIES
    LABELS "openfhe;fast;m2;packing;linear"
    TIMEOUT 300
  )

  add_executable(
    openfhe_feature_packed_smoke
    tests/openfhe_feature_packed_smoke.cpp
  )
  target_link_libraries(openfhe_feature_packed_smoke PRIVATE
    moai::openfhe_client
    moai::openfhe_server
  )
  set_target_properties(
    openfhe_feature_packed_smoke
    PROPERTIES BUILD_RPATH "${OpenFHE_LIBDIR}"
  )
  target_link_options(
    openfhe_feature_packed_smoke
    PRIVATE "-Wl,--disable-new-dtags"
  )
  add_test(
    NAME openfhe_feature_packed_smoke
    COMMAND openfhe_feature_packed_smoke
  )
  set_tests_properties(openfhe_feature_packed_smoke PROPERTIES
    LABELS "openfhe;fast;m4;feature-packed;linear"
    TIMEOUT 300
  )

  add_executable(
    openfhe_feature_packed_attention_smoke
    tests/openfhe_feature_packed_attention_smoke.cpp
  )
  target_link_libraries(openfhe_feature_packed_attention_smoke PRIVATE
    moai::openfhe_client
    moai::openfhe_server
  )
  set_target_properties(
    openfhe_feature_packed_attention_smoke
    PROPERTIES BUILD_RPATH "${OpenFHE_LIBDIR}"
  )
  target_link_options(
    openfhe_feature_packed_attention_smoke
    PRIVATE "-Wl,--disable-new-dtags"
  )
  add_test(
    NAME openfhe_feature_packed_attention_smoke
    COMMAND openfhe_feature_packed_attention_smoke
  )
  set_tests_properties(openfhe_feature_packed_attention_smoke PROPERTIES
    LABELS "openfhe;slow;m4;feature-packed;attention;bootstrap"
    TIMEOUT 1800
  )

  add_executable(
    openfhe_encoder_layer_smoke
    tests/openfhe_encoder_layer_smoke.cpp
    tests/support/moai_encoder_fixture.cpp
    tests/support/moai_encoder_plaintext_oracle.cpp
  )
  target_include_directories(openfhe_encoder_layer_smoke PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/tests"
  )
  target_link_libraries(openfhe_encoder_layer_smoke PRIVATE
    moai::openfhe_client
    moai::openfhe_server
  )
  set_target_properties(
    openfhe_encoder_layer_smoke
    PROPERTIES BUILD_RPATH "${OpenFHE_LIBDIR}"
  )
  target_link_options(
    openfhe_encoder_layer_smoke
    PRIVATE "-Wl,--disable-new-dtags"
  )
  add_test(
    NAME openfhe_encoder_layer_smoke
    COMMAND openfhe_encoder_layer_smoke
            --data-root "${MOAI_TRACE_DATA_ROOT}"
            --layer 1
            --input-level 29
  )
  set_tests_properties(openfhe_encoder_layer_smoke PROPERTIES
    LABELS "openfhe;slow;m4;encoder;server-only;trace"
    TIMEOUT 7200
  )
  add_test(
    NAME openfhe_encoder_fixture_contract
    COMMAND openfhe_encoder_layer_smoke
            --data-root "${MOAI_TRACE_DATA_ROOT}"
            --preflight-only
  )
  set_tests_properties(openfhe_encoder_fixture_contract PROPERTIES
    LABELS "plaintext;fast;m4;encoder;trace"
    TIMEOUT 300
  )

  add_executable(
    openfhe_encoder_12_layer_smoke
    tests/openfhe_encoder_12_layer_smoke.cpp
    tests/support/moai_encoder_fixture.cpp
    tests/support/moai_encoder_plaintext_oracle.cpp
  )
  target_include_directories(openfhe_encoder_12_layer_smoke PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/tests"
  )
  target_link_libraries(openfhe_encoder_12_layer_smoke PRIVATE
    moai::openfhe_client
    moai::openfhe_server
  )
  set_target_properties(
    openfhe_encoder_12_layer_smoke
    PROPERTIES BUILD_RPATH "${OpenFHE_LIBDIR}"
  )
  target_link_options(
    openfhe_encoder_12_layer_smoke
    PRIVATE "-Wl,--disable-new-dtags"
  )
  add_test(
    NAME openfhe_encoder_12_layer_preflight
    COMMAND openfhe_encoder_12_layer_smoke
            --data-root "${MOAI_TRACE_DATA_ROOT}"
            --preflight-only
  )
  set_tests_properties(openfhe_encoder_12_layer_preflight PROPERTIES
    LABELS "plaintext;fast;m5;encoder;oracle;trace"
    TIMEOUT 300
  )
  add_test(
    NAME openfhe_encoder_12_layer_crypto_preflight
    COMMAND openfhe_encoder_12_layer_smoke
            --data-root "${MOAI_TRACE_DATA_ROOT}"
            --crypto-preflight-only
  )
  set_tests_properties(openfhe_encoder_12_layer_crypto_preflight PROPERTIES
    LABELS "openfhe;slow;m5;encoder;server-only;trace"
    RUN_SERIAL TRUE
    TIMEOUT 300
  )
  add_test(
    NAME openfhe_encoder_3_layer_exact_smoke
    COMMAND openfhe_encoder_12_layer_smoke
            --data-root "${MOAI_TRACE_DATA_ROOT}"
            --diagnostic-layer-count 3
  )
  set_tests_properties(openfhe_encoder_3_layer_exact_smoke PROPERTIES
    LABELS "openfhe;very-slow;m5;encoder;server-only;trace;diagnostic;exact3"
    RUN_SERIAL TRUE
    TIMEOUT 21600
  )
  add_test(
    NAME openfhe_encoder_12_layer_smoke
    COMMAND openfhe_encoder_12_layer_smoke
            --data-root "${MOAI_TRACE_DATA_ROOT}"
  )
  set_tests_properties(openfhe_encoder_12_layer_smoke PROPERTIES
    LABELS "openfhe;very-slow;m5;encoder;server-only;trace"
    RUN_SERIAL TRUE
    TIMEOUT 64800
  )

  add_executable(
    openfhe_encoder_plaintext_oracle_smoke
    tests/openfhe_encoder_plaintext_oracle_smoke.cpp
    tests/support/moai_encoder_fixture.cpp
    tests/support/moai_encoder_plaintext_oracle.cpp
  )
  target_include_directories(openfhe_encoder_plaintext_oracle_smoke PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/tests"
  )
  target_link_libraries(openfhe_encoder_plaintext_oracle_smoke PRIVATE
    moai::openfhe_common
  )
  set_target_properties(
    openfhe_encoder_plaintext_oracle_smoke
    PROPERTIES BUILD_RPATH "${OpenFHE_LIBDIR}"
  )
  target_link_options(
    openfhe_encoder_plaintext_oracle_smoke
    PRIVATE "-Wl,--disable-new-dtags"
  )
  add_test(
    NAME openfhe_encoder_plaintext_oracle_smoke
    COMMAND openfhe_encoder_plaintext_oracle_smoke
            --data-root "${MOAI_TRACE_DATA_ROOT}"
  )
  set_tests_properties(openfhe_encoder_plaintext_oracle_smoke PROPERTIES
    LABELS "plaintext;fast;m4;m5;encoder;oracle;trace"
    TIMEOUT 300
  )

  add_executable(openfhe_bootstrap_smoke tests/openfhe_bootstrap_smoke.cpp)
  target_link_libraries(openfhe_bootstrap_smoke PRIVATE
    moai::openfhe_client
    moai::openfhe_server
  )
  set_target_properties(openfhe_bootstrap_smoke PROPERTIES BUILD_RPATH "${OpenFHE_LIBDIR}")
  target_link_options(openfhe_bootstrap_smoke PRIVATE "-Wl,--disable-new-dtags")
  add_test(NAME openfhe_bootstrap_smoke COMMAND openfhe_bootstrap_smoke)
  set_tests_properties(openfhe_bootstrap_smoke PROPERTIES
    LABELS "openfhe;slow;bootstrap"
    TIMEOUT 900
  )

  add_executable(
    openfhe_evaluation_key_bundle_smoke
    tests/openfhe_evaluation_key_bundle_smoke.cpp
  )
  target_link_libraries(openfhe_evaluation_key_bundle_smoke PRIVATE
    moai::openfhe_client
    moai::openfhe_server
  )
  set_target_properties(
    openfhe_evaluation_key_bundle_smoke
    PROPERTIES BUILD_RPATH "${OpenFHE_LIBDIR}"
  )
  target_link_options(
    openfhe_evaluation_key_bundle_smoke
    PRIVATE "-Wl,--disable-new-dtags"
  )
  add_test(
    NAME openfhe_evaluation_key_bundle_smoke
    COMMAND openfhe_evaluation_key_bundle_smoke
  )
  set_tests_properties(openfhe_evaluation_key_bundle_smoke PROPERTIES
    LABELS "openfhe;slow;bootstrap;m4;evaluation-keys;server-only"
    RUN_SERIAL TRUE
    TIMEOUT 900
  )

  add_executable(openfhe_nonlinear_smoke tests/openfhe_nonlinear_smoke.cpp)
  target_link_libraries(openfhe_nonlinear_smoke PRIVATE
    moai::openfhe_client
    moai::openfhe_server
  )
  set_target_properties(
    openfhe_nonlinear_smoke
    PROPERTIES BUILD_RPATH "${OpenFHE_LIBDIR}"
  )
  target_link_options(
    openfhe_nonlinear_smoke
    PRIVATE "-Wl,--disable-new-dtags"
  )
  add_test(NAME openfhe_nonlinear_smoke COMMAND openfhe_nonlinear_smoke)
  set_tests_properties(openfhe_nonlinear_smoke PROPERTIES
    LABELS "openfhe;slow;bootstrap;m3;nonlinear"
    TIMEOUT 1800
  )

  add_executable(
    openfhe_feature_layernorm_smoke
    tests/openfhe_feature_layernorm_smoke.cpp
  )
  target_link_libraries(openfhe_feature_layernorm_smoke PRIVATE
    moai::openfhe_client
    moai::openfhe_server
  )
  set_target_properties(
    openfhe_feature_layernorm_smoke
    PROPERTIES BUILD_RPATH "${OpenFHE_LIBDIR}"
  )
  target_link_options(
    openfhe_feature_layernorm_smoke
    PRIVATE "-Wl,--disable-new-dtags"
  )
  add_test(
    NAME openfhe_feature_layernorm_smoke
    COMMAND openfhe_feature_layernorm_smoke
  )
  set_tests_properties(openfhe_feature_layernorm_smoke PROPERTIES
    LABELS "m4;m5;openfhe;bootstrap;slow;nonlinear;layernorm"
    RUN_SERIAL TRUE
    TIMEOUT 1800
  )

  add_executable(
    openfhe_feature_bootstrap_smoke
    tests/openfhe_feature_bootstrap_smoke.cpp
  )
  target_link_libraries(openfhe_feature_bootstrap_smoke PRIVATE
    moai::openfhe_client
    moai::openfhe_server
  )
  set_target_properties(
    openfhe_feature_bootstrap_smoke
    PROPERTIES BUILD_RPATH "${OpenFHE_LIBDIR}"
  )
  target_link_options(
    openfhe_feature_bootstrap_smoke
    PRIVATE "-Wl,--disable-new-dtags"
  )
  add_test(
    NAME openfhe_feature_bootstrap_smoke
    COMMAND openfhe_feature_bootstrap_smoke
  )
  set_tests_properties(openfhe_feature_bootstrap_smoke PROPERTIES
    LABELS "m4;openfhe;bootstrap;slow"
    TIMEOUT 1800
  )
endif()

# Keep the server-side trust-boundary audit fail-closed over the final target
# graph, after every target_link_libraries/target_include_directories call in
# this project. Third-party OpenFHE remains reachable only through the audited
# common target; new first-party links or include roots require explicit review.
function(moai_require_exact_target_property target property expected)
  get_target_property(actual "${target}" "${property}")
  if(NOT "${actual}" STREQUAL "${expected}")
    message(FATAL_ERROR
      "${target} ${property} changed from the audited trust-boundary value; "
      "review and extend the server closure before configuring"
    )
  endif()
endfunction()

function(moai_require_unset_target_property target property)
  get_property(is_set TARGET "${target}" PROPERTY "${property}" SET)
  if(is_set)
    get_target_property(actual "${target}" "${property}")
    if(NOT "${actual}" STREQUAL "")
      message(FATAL_ERROR
        "${target} ${property} must remain unset at the server trust boundary"
      )
    endif()
  endif()
endfunction()

function(moai_require_frozen_target_property target snapshot_prefix property)
  set(set_variable "MOAI_FROZEN_${snapshot_prefix}_${property}_SET")
  set(value_variable "MOAI_FROZEN_${snapshot_prefix}_${property}_VALUE")
  if(NOT DEFINED ${set_variable})
    message(FATAL_ERROR
      "Missing frozen trust-boundary snapshot for ${target} ${property}")
  endif()

  set(expected_is_set "${${set_variable}}")
  get_property(actual_is_set TARGET "${target}" PROPERTY "${property}" SET)
  if(expected_is_set AND NOT actual_is_set)
    message(FATAL_ERROR
      "${target} ${property} was removed from the frozen imported graph")
  elseif(NOT expected_is_set AND actual_is_set)
    message(FATAL_ERROR
      "${target} ${property} was added after the imported graph was frozen")
  elseif(expected_is_set)
    get_target_property(actual_value "${target}" "${property}")
    set(expected_value "${${value_variable}}")
    if(NOT "${actual_value}" STREQUAL "${expected_value}")
      message(FATAL_ERROR
        "${target} ${property} changed after the imported graph was frozen")
    endif()
  endif()
endfunction()

function(moai_resolve_target_alias target output_variable)
  if(NOT TARGET "${target}")
    message(FATAL_ERROR
      "Cannot resolve non-target server link item: ${target}")
  endif()
  get_target_property(aliased_target "${target}" ALIASED_TARGET)
  if(aliased_target AND NOT aliased_target MATCHES "-NOTFOUND$")
    set(${output_variable} "${aliased_target}" PARENT_SCOPE)
  else()
    set(${output_variable} "${target}" PARENT_SCOPE)
  endif()
endfunction()

function(moai_collect_target_link_items target output_variable)
  set(link_items)
  foreach(property IN LISTS MOAI_SERVER_LINK_PROPERTIES)
    get_property(property_is_set TARGET "${target}" PROPERTY "${property}" SET)
    if(property_is_set)
      get_target_property(property_value "${target}" "${property}")
      list(APPEND link_items ${property_value})
    endif()
  endforeach()
  set(${output_variable} "${link_items}" PARENT_SCOPE)
endfunction()

function(moai_validate_complete_server_target_link_closure)
  set(scan_queue moai_openfhe_server)
  set(visited_targets)

  while(scan_queue)
    list(POP_FRONT scan_queue queued_target)
    moai_resolve_target_alias("${queued_target}" target)
    if("${target}" IN_LIST visited_targets)
      continue()
    endif()
    if(NOT "${target}" IN_LIST MOAI_SERVER_ALLOWED_TARGET_CLOSURE)
      message(FATAL_ERROR
        "Server link closure reaches unapproved target: ${target}")
    endif()
    list(APPEND visited_targets "${target}")

    moai_collect_target_link_items("${target}" link_items)
    foreach(link_item IN LISTS link_items)
      if("${link_item}" STREQUAL "")
        continue()
      elseif("${link_item}" MATCHES "\\$<")
        message(FATAL_ERROR
          "Server link closure contains an unauditable generator expression "
          "from ${target}: ${link_item}")
      elseif(TARGET "${link_item}")
        moai_resolve_target_alias("${link_item}" linked_target)
        if(NOT "${linked_target}" IN_LIST MOAI_SERVER_ALLOWED_TARGET_CLOSURE)
          message(FATAL_ERROR
            "Server link closure reaches unapproved target: "
            "${target} -> ${linked_target}")
        endif()
        list(APPEND scan_queue "${linked_target}")
      elseif("${link_item}" IN_LIST MOAI_SERVER_FROZEN_RAW_LINK_ITEMS)
        continue()
      else()
        message(FATAL_ERROR
          "Server link closure contains an unresolved or unfrozen link item "
          "from ${target}: ${link_item}")
      endif()
    endforeach()
  endwhile()

  set(expected_targets ${MOAI_SERVER_ALLOWED_TARGET_CLOSURE})
  list(SORT expected_targets)
  list(SORT visited_targets)
  if(NOT "${visited_targets}" STREQUAL "${expected_targets}")
    message(FATAL_ERROR
      "Server link closure no longer matches the frozen OpenFHE/OpenMP graph; "
      "expected ${expected_targets}, observed ${visited_targets}")
  endif()
endfunction()

function(moai_require_unset_source_property source property)
  get_source_file_property(actual_value "${source}" "${property}")
  if(NOT "${actual_value}" STREQUAL "NOTFOUND" AND
     NOT "${actual_value}" STREQUAL "")
    message(FATAL_ERROR
      "Server-linked source ${source} ${property} must remain unset")
  endif()
endfunction()

function(moai_reject_forced_include_flags variable_name flags)
  foreach(forbidden_fragment IN ITEMS
      "-include"
      "-imacros"
      "/FI")
    string(FIND "${flags}" "${forbidden_fragment}" forbidden_position)
    if(NOT forbidden_position EQUAL -1)
      message(FATAL_ERROR
        "${variable_name} contains forbidden forced-include input: "
        "${forbidden_fragment}")
    endif()
  endforeach()
endfunction()

function(moai_validate_final_server_target_graph)
  moai_validate_complete_server_target_link_closure()

  moai_require_exact_target_property(
    moai_openfhe_server
    LINK_LIBRARIES
    moai_openfhe_common
  )
  moai_require_exact_target_property(
    moai_openfhe_common
    LINK_LIBRARIES
    moai_openfhe_vendor
  )
  moai_require_exact_target_property(
    moai_openfhe_server
    INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )
  moai_require_exact_target_property(
    moai_openfhe_server
    INTERFACE_LINK_LIBRARIES
    moai_openfhe_common
  )
  moai_require_exact_target_property(
    moai_openfhe_common
    INTERFACE_LINK_LIBRARIES
    moai_openfhe_vendor
  )
  moai_require_exact_target_property(
    moai_openfhe_server
    INTERFACE_INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )
  moai_require_exact_target_property(
    moai_openfhe_common
    INTERFACE_INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )
  moai_require_exact_target_property(
    moai_openfhe_vendor
    INTERFACE_LINK_LIBRARIES
    "OPENFHEpke;OpenMP::OpenMP_CXX"
  )
  moai_require_exact_target_property(
    moai_openfhe_vendor
    INTERFACE_INCLUDE_DIRECTORIES
    "${OpenFHE_INCLUDE};${OpenFHE_INCLUDE}/core;${OpenFHE_INCLUDE}/pke;${OpenFHE_INCLUDE}/binfhe"
  )
  moai_require_exact_target_property(
    moai_openfhe_vendor
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
    "${OpenFHE_INCLUDE};${OpenFHE_INCLUDE}/core;${OpenFHE_INCLUDE}/pke;${OpenFHE_INCLUDE}/binfhe"
  )
  moai_require_exact_target_property(
    moai_openfhe_vendor
    INTERFACE_COMPILE_DEFINITIONS
    "OPENFHE_VERSION=${BASE_OPENFHE_VERSION}"
  )

  foreach(audited_target IN ITEMS
      moai_openfhe_server
      moai_openfhe_common
      moai_openfhe_vendor)
    foreach(unset_property IN ITEMS
        INTERFACE_LINK_LIBRARIES_DIRECT
        INTERFACE_LINK_LIBRARIES_DIRECT_EXCLUDE
        INTERFACE_LINK_OPTIONS
        INTERFACE_LINK_DIRECTORIES
        INTERFACE_SOURCES
        PRECOMPILE_HEADERS
        PRECOMPILE_HEADERS_REUSE_FROM
        INTERFACE_PRECOMPILE_HEADERS
        COMPILE_OPTIONS
        INTERFACE_COMPILE_OPTIONS
        COMPILE_FLAGS
        COMPILE_DEFINITIONS
        COMPILE_DEFINITIONS_DEBUG
        COMPILE_DEFINITIONS_RELEASE
        COMPILE_DEFINITIONS_RELWITHDEBINFO
        COMPILE_DEFINITIONS_MINSIZEREL
        UNITY_BUILD_CODE_BEFORE_INCLUDE
        UNITY_BUILD_CODE_AFTER_INCLUDE)
      moai_require_unset_target_property(
        "${audited_target}"
        "${unset_property}"
      )
    endforeach()
  endforeach()

  foreach(audited_target IN ITEMS
      moai_openfhe_server
      moai_openfhe_common)
    foreach(unset_property IN ITEMS
        INTERFACE_COMPILE_DEFINITIONS
        LINK_OPTIONS
        LINK_DIRECTORIES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
      moai_require_unset_target_property(
        "${audited_target}"
        "${unset_property}"
      )
    endforeach()
  endforeach()

  foreach(snapshot IN LISTS MOAI_SERVER_FROZEN_IMPORTED_SNAPSHOTS)
    string(REPLACE "|" ";" snapshot_fields "${snapshot}")
    list(GET snapshot_fields 0 imported_target)
    list(GET snapshot_fields 1 snapshot_prefix)
    if(NOT TARGET "${imported_target}")
      message(FATAL_ERROR
        "Frozen imported server target disappeared: ${imported_target}")
    endif()
    foreach(property IN LISTS MOAI_SERVER_FROZEN_IMPORTED_PROPERTIES)
      moai_require_frozen_target_property(
        "${imported_target}"
        "${snapshot_prefix}"
        "${property}"
      )
    endforeach()
  endforeach()

  moai_require_exact_target_property(
    moai_openfhe_common
    SOURCES
    "${MOAI_OPENFHE_COMMON_SOURCES}"
  )
  moai_require_exact_target_property(
    moai_openfhe_server
    SOURCES
    "${MOAI_OPENFHE_SERVER_SOURCES}"
  )
  moai_require_exact_target_property(
    moai_openfhe_common
    INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )

  foreach(source IN LISTS MOAI_OPENFHE_LINKED_SOURCES)
    foreach(property IN ITEMS
        COMPILE_OPTIONS
        COMPILE_FLAGS
        COMPILE_DEFINITIONS
        COMPILE_DEFINITIONS_DEBUG
        COMPILE_DEFINITIONS_RELEASE
        COMPILE_DEFINITIONS_RELWITHDEBINFO
        COMPILE_DEFINITIONS_MINSIZEREL
        INCLUDE_DIRECTORIES)
      moai_require_unset_source_property("${source}" "${property}")
    endforeach()
  endforeach()

  foreach(flag_variable IN LISTS MOAI_SERVER_FROZEN_CXX_FLAG_VARIABLES)
    if(NOT "${${flag_variable}}" STREQUAL "${MOAI_FROZEN_${flag_variable}}")
      message(FATAL_ERROR
        "${flag_variable} changed after the server compile inputs were frozen")
    endif()
    moai_reject_forced_include_flags(
      "${flag_variable}"
      "${${flag_variable}}"
    )
  endforeach()
endfunction()

cmake_language(DEFER CALL moai_validate_final_server_target_graph)
