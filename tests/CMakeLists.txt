
if(MSVC)
    # warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.
    add_definitions(/wd4996)
endif()

macro(ncnn_add_test name)
    add_executable(test_${name} test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE ncnn)

    add_test(NAME test_${name} COMMAND ${CMAKE_COMMAND} -DTEST_EXECUTABLE=$<TARGET_FILE:test_${name}> -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/run_test.cmake)

    # add test to a virtual project group
    set_property(TARGET test_${name} PROPERTY FOLDER "tests")
endmacro()

macro(ncnn_add_layer_test class)
    string(TOLOWER ${class} name)

    # enable if WITH_LAYER_xxx option ON
    if(${WITH_LAYER_${name}})
        ncnn_add_test(${name})
    endif()
endmacro()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/layer)

if(NCNN_PIXEL_AFFINE)
    ncnn_add_test(mat_pixel_affine)
endif()

if(NCNN_PIXEL_DRAWING)
    ncnn_add_test(mat_pixel_drawing)
endif()

if(NCNN_PIXEL_ROTATE)
    ncnn_add_test(mat_pixel_rotate)
endif()

if(NCNN_PIXEL)
    ncnn_add_test(mat_pixel_resize)
    ncnn_add_test(mat_pixel)
    ncnn_add_test(squeezenet)
endif()

ncnn_add_test(c_api)
ncnn_add_test(cpu)

if(NCNN_VULKAN)
    ncnn_add_test(command)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    target_link_libraries(test_squeezenet PRIVATE nodefs.js)
endif()

ncnn_add_layer_test(AbsVal)
ncnn_add_layer_test(DeformableConv2D)
