# Vendored eOn water kernels. Short includes (ccl.hpp, zhu_philpott.hpp)
# resolve in this directory. The public wrappers are WaterPots.
water_pots = static_library(
    'water_pots',
    [
        'potential_base.cpp',
        'ccl.cpp',
        'tip4p_ccl.cpp',
        'spce_ccl.cpp',
        'zhu_philpott_parameters.cpp',
        'zhu_philpott.cpp',
        'WaterPots.cc',
    ],
    dependencies: _deps,
    cpp_args: _args,
    include_directories: ['.', '../../'],
    pic: true,
    install: false,
)
