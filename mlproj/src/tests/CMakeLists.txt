
add_executable(mlight_tests
mlight_tests.cpp
linearregression_test.cpp
naivebayesclassifier_test.cpp
logisticregression_test.cpp
neuralnetwork_test.cpp
util_test.cpp
tibia_random_test.cpp
)

find_package(Boost 1.58 COMPONENTS unit_test_framework REQUIRED)

target_link_libraries(mlight_tests cblas tcmalloc_minimal ${BOOST_LIBRARIES})

set(CMAKE_CXX_LINK_FLAGS "-L/opt/cuda/lib64 -L/opt/magma/lib -L/usr/lib/acml/ifort -Wl,-rpath /opt/magma/lib -Wl,-rpath /usr/lib/acml/ifort")

# OpenBLAS
#target_link_libraries(mlproj cblas tcmalloc_minimal)

# ACML
#target_link_libraries(mlproj acml tcmalloc_minimal)

# ACML MP
#target_link_libraries(mlproj acml_mp iomp5 tcmalloc_minimal)

# MAGMA
#target_link_libraries(mlproj magma cblas tcmalloc_minimal)

# Cula
#target_link_libraries(mlproj cula_lapack_basic_link tcmalloc_minimal)
