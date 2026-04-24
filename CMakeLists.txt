cmake_minimum_required(VERSION 3.16)
project(hcp_ringattn CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(hcp_ringattn_core STATIC
    src/ringattn_runtime.cc
)
target_include_directories(hcp_ringattn_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(ringattn_coordinator_smoke
    src/ringattn_coordinator_smoke_main.cc
)
target_link_libraries(ringattn_coordinator_smoke
    PRIVATE
        hcp_ringattn_core
)

