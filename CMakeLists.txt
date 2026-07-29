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

add_library(moai_openfhe_common STATIC src/openfhe/context_factory.cpp)
target_include_directories(moai_openfhe_common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(moai_openfhe_common PUBLIC moai_openfhe_vendor)
add_library(moai::openfhe_common ALIAS moai_openfhe_common)

add_library(moai_openfhe_server STATIC src/openfhe/server_runtime.cpp)
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

  add_executable(openfhe_fast_smoke tests/openfhe_fast_smoke.cpp)
  target_link_libraries(openfhe_fast_smoke PRIVATE
    moai::openfhe_client
    moai::openfhe_server
  )
  set_target_properties(openfhe_fast_smoke PROPERTIES BUILD_RPATH "${OpenFHE_LIBDIR}")
  target_link_options(openfhe_fast_smoke PRIVATE "-Wl,--disable-new-dtags")
  add_test(NAME openfhe_fast_smoke COMMAND openfhe_fast_smoke)
  set_tests_properties(openfhe_fast_smoke PROPERTIES LABELS "openfhe;fast")

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
endif()
