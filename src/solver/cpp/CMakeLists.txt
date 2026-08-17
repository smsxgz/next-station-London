add_library(next_station_solver STATIC
    search.cpp
)
target_link_libraries(next_station_solver PUBLIC next_station_engine)
target_include_directories(next_station_solver
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
if (MSVC)
    target_compile_options(next_station_solver PRIVATE /W4 /permissive-)
else()
    target_compile_options(next_station_solver PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_sources(next_station_engine_capi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/c_api.cpp
)
target_link_libraries(next_station_engine_capi PRIVATE next_station_solver)
target_include_directories(next_station_engine_capi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../engine_cpp/src
)

add_executable(next_station_solver_check
    smoke.cpp
)
target_link_libraries(next_station_solver_check PRIVATE next_station_solver)
