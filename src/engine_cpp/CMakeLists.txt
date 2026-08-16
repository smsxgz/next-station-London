cmake_minimum_required(VERSION 3.15)
project(next_station_native_engine LANGUAGES CXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(next_station_engine STATIC
    src/engine.cpp
)
target_include_directories(next_station_engine
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (MSVC)
    target_compile_options(next_station_engine PRIVATE /W4 /permissive-)
else()
    target_compile_options(next_station_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(next_station_engine_check
    src/smoke.cpp
)
target_link_libraries(next_station_engine_check PRIVATE next_station_engine)

add_library(next_station_engine_capi SHARED
    src/c_api.cpp
)
target_link_libraries(next_station_engine_capi PRIVATE
    next_station_engine
    Threads::Threads
)
target_include_directories(next_station_engine_capi
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if (MINGW)
    target_link_libraries(next_station_engine_capi PRIVATE
        -static-libgcc -static-libstdc++
    )
endif()
