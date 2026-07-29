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

add_library(moai_openfhe_common STATIC
  src/openfhe/approximation_registry.cpp
  src/openfhe/context_factory.cpp
  src/openfhe/packing.cpp
)
target_include_directories(moai_openfhe_common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(moai_openfhe_common PUBLIC moai_openfhe_vendor)
add_library(moai::openfhe_common ALIAS moai_openfhe_common)

add_library(moai_openfhe_server STATIC
  src/openfhe/server_runtime.cpp
  src/openfhe/linear_ops.cpp
  src/openfhe/nonlinear_ops.cpp
)
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
            -P ${CMAKE_SOURCE_DIR}/cmake/ValidateServerBoundary.cmake
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
  )
  set_tests_properties(openfhe_profile_contract PROPERTIES
    LABELS "m3;plaintext;profile"
    TIMEOUT 60
  )

  add_test(
    NAME openfhe_artifact_schema_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/scripts/validate_openfhe_artifact.py"
            --schema "${CMAKE_SOURCE_DIR}/docs/openfhe-artifact-schema.json"
  )
  set_tests_properties(openfhe_artifact_schema_contract PROPERTIES
    LABELS "m3;plaintext;artifact"
    TIMEOUT 60
  )

  add_test(
    NAME openfhe_artifact_validator_contract
    COMMAND "${MOAI_PYTHON_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tests/test_validate_openfhe_artifact.py"
  )
  set_tests_properties(openfhe_artifact_validator_contract PROPERTIES
    LABELS "m3;plaintext;artifact"
    TIMEOUT 60
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
endif()
