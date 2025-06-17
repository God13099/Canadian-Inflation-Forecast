include(FetchContent)
FetchContent_Declare(
        matplotplusplus
        GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus.git
        GIT_TAG        master
)

FetchContent_MakeAvailable(matplotplusplus)

add_executable(untitled3 main.cpp)
target_link_libraries(untitled3 matplot)

target_include_directories(untitled3 PRIVATE /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3)
