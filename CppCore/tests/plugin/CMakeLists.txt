add_library(rgpot_null SHARED ../../examples/plugins/null_plugin.c)
add_library(rgpot_morse SHARED ../../examples/plugins/morse_plugin.c)
add_library(rgpot_coulomb SHARED ../../examples/plugins/coulomb_plugin.f90
                                    ../../examples/plugins/coulomb_plugin_wrapper.c)
add_library(rgpot_plugin_lifetime SHARED ../plugin_lifetime_fixture.c)
foreach(_plugin rgpot_null rgpot_morse rgpot_coulomb rgpot_plugin_lifetime)
  target_include_directories(${_plugin} PRIVATE "${PROJECT_SOURCE_DIR}/include")
  set_target_properties(${_plugin} PROPERTIES PREFIX "lib")
endforeach()
if(UNIX)
  target_link_libraries(rgpot_morse PRIVATE m)
endif()
if(APPLE)
  # Link with the C driver so the wrapper's libSystem references resolve at
  # the C toolchain's deployment target, with the Fortran runtime added
  # explicitly.
  set_target_properties(rgpot_coulomb PROPERTIES LINKER_LANGUAGE C)
  target_link_directories(rgpot_coulomb PRIVATE
                          ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES})
  target_link_libraries(rgpot_coulomb PRIVATE
                        ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
endif()
foreach(_test PluginTest PluginContractTest ExternalPluginTest ExternalFortranPluginTest)
  add_pot_test(${_test} ../${_test}.cc)
  set_tests_properties(${_test} PROPERTIES
    ENVIRONMENT "RGPOT_TEST_PLUGIN_DIR=$<TARGET_FILE_DIR:rgpot_null>")
endforeach()
add_executable(PluginLifetimeTest ../PluginLifetimeTest.cc)
target_link_libraries(PluginLifetimeTest PRIVATE rgpot::rgpot)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME PluginLifetimeTest
         COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/../test_plugin_lifetime.py"
                 $<TARGET_FILE:PluginLifetimeTest> $<TARGET_FILE:rgpot_plugin_lifetime>)
set_tests_properties(PluginLifetimeTest PROPERTIES TIMEOUT 30)
