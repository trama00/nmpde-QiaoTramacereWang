cmake_minimum_required(VERSION 3.12.0)

# Force the compiler to match deal.II before project()
set(CMAKE_CXX_COMPILER "/u/sw/toolchains/gcc-glibc/11.2.0/prefix/bin/g++")

# Set build type to DebugRelease to match deal.II installation
set(CMAKE_BUILD_TYPE "DebugRelease" CACHE STRING "Choose the type of build")

project(WaveEquationTest LANGUAGES CXX)

# Locate deal.II and initialize its variables.
set(CMAKE_PREFIX_PATH "/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.5.1/lib/cmake/deal.II" ${CMAKE_PREFIX_PATH})
find_package(deal.II 9.5.1 REQUIRED
        HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{mkDealiiPrefix})
deal_ii_initialize_cached_variables()

# Add executable for main CG wave solver
add_executable(test_wave_1d
    src/apps/main_cg.cpp
    src/cg/CGWaveSolver.cpp
    src/core/Problems/GaussianPulse.cpp
    src/core/Utilities/VTKOutput.cpp
    src/core/Utilities/ErrorCalculator.cpp
)

# Add include directories
target_include_directories(test_wave_1d PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/wave_solver
)

# Setup deal.II for the target with RELEASE flags (optimized)
deal_ii_setup_target(test_wave_1d RELEASE)

# Add manufactured solution test executable
add_executable(test_mms
    src/apps/test_mms.cpp
    src/cg/CGWaveSolver.cpp
    src/core/Problems/ManufacturedSolution.cpp
    src/core/Utilities/VTKOutput.cpp
    src/core/Utilities/ErrorCalculator.cpp
)

target_include_directories(test_mms PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/wave_solver
)

deal_ii_setup_target(test_mms RELEASE)

# Add spatial convergence test executable
add_executable(test_convergence_space
    src/apps/test_convergence_space.cpp
    src/cg/CGWaveSolver.cpp
    src/core/Problems/ManufacturedSolution.cpp
    src/core/Utilities/VTKOutput.cpp
    src/core/Utilities/ErrorCalculator.cpp
)

target_include_directories(test_convergence_space PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/wave_solver
)

deal_ii_setup_target(test_convergence_space RELEASE)

# Add temporal convergence test executable
add_executable(test_convergence_time
    src/apps/test_convergence_time.cpp
    src/cg/CGWaveSolver.cpp
    src/core/Problems/ManufacturedSolution.cpp
    src/core/Utilities/VTKOutput.cpp
    src/core/Utilities/ErrorCalculator.cpp
)

target_include_directories(test_convergence_time PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/wave_solver
)

deal_ii_setup_target(test_convergence_time RELEASE)

# Add space-time convergence test executable
add_executable(test_convergence_spacetime
    src/apps/test_convergence_spacetime.cpp
    src/cg/CGWaveSolver.cpp
    src/core/Problems/ManufacturedSolution.cpp
    src/core/Utilities/VTKOutput.cpp
    src/core/Utilities/ErrorCalculator.cpp
)

target_include_directories(test_convergence_spacetime PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/wave_solver
)

deal_ii_setup_target(test_convergence_spacetime RELEASE)