set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR mips64el)

set(CMAKE_C_COMPILER "mips64el-linux-gnuabi64-gcc")
set(CMAKE_CXX_COMPILER "mips64el-linux-gnuabi64-g++")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_C_FLAGS "-mabi=64 -march=mips3 -mtune=loongson2f -mloongson-mmi")
set(CMAKE_CXX_FLAGS "-mabi=64 -march=mips3 -mtune=loongson2f -mloongson-mmi")

# cache flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")
