include_directories(${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_CXX_STANDARD 14)

file(GLOB_RECURSE all_tests RELATIVE ${CMAKE_CURRENT_LIST_DIR} *.cc)

foreach(test ${all_tests})
  get_filename_component(test_name ${test} NAME_WE)
  get_filename_component(test_dir ${test} DIRECTORY)
  add_executable(${test_name} ${test})
  set_property(TARGET ${test_name} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
endforeach()
