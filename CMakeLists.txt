cmake_minimum_required(VERSION 3.20)
project(hexa-color-solver)

# Find FTXUI (for terminal UI layout)
find_package(ftxui REQUIRED)

# Options
option(USE_ROCM "Build for ROCm/HIP" OFF)

if(USE_ROCM)
    message(STATUS "Building for ROCm (HIP)...")
    enable_language(CXX)

    # Find hipify-perl
    find_program(HIPIFY_PERL hipify-perl)
    if(NOT HIPIFY_PERL)
        message(FATAL_ERROR "hipify-perl not found! Required for ROCm build.")
    endif()

    set(CUDA_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/hexa-color-solver.cu")
    set(HIP_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/hexa-color-solver.hip.cpp")

    # Generate HIP source
    add_custom_command(
        OUTPUT ${HIP_SOURCE}
        COMMAND ${HIPIFY_PERL} ${CUDA_SOURCE} > ${HIP_SOURCE}
        DEPENDS ${CUDA_SOURCE}
        COMMENT "Hipifying hexa-color-solver.cu"
        VERBATIM
    )

    add_executable(hexa-color-solver ${HIP_SOURCE})
    
    # If CMAKE_CXX_COMPILER is hipcc, it handles most things.
    # But we need to link libraries explicitly if not added by the wrapper.
    target_link_libraries(hexa-color-solver PRIVATE rocrand hiprand ftxui::screen ftxui::dom)

    # Optimization
    target_compile_options(hexa-color-solver PRIVATE -O3)

else()
    message(STATUS "Building for CUDA...")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_executable(hexa-color-solver hexa-color-solver.cu)

    # Use C++17
    set_target_properties(hexa-color-solver PROPERTIES CUDA_STANDARD 17)

    # Architectures: 7.5 (Turing), 8.9 (Ada), 9.0 (Hopper/Blackwell)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 89 90)
    endif()

    target_link_libraries(hexa-color-solver PRIVATE CUDA::curand ftxui::screen ftxui::dom)
    target_compile_options(hexa-color-solver PRIVATE -O3)
endif()

install(TARGETS hexa-color-solver DESTINATION bin)

# Tests (host-only, no CUDA required)
enable_testing()
add_executable(color_test color_test.cpp)
target_compile_features(color_test PRIVATE cxx_std_17)
add_test(NAME color_test COMMAND color_test)
