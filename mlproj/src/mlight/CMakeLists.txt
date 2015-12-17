add_library(mlight STATIC
linearregression.hpp
logisticregression.hpp
naivebayesclassifier.hpp
neuralnetwork.hpp
optimizers.hpp

util/dumper.hpp
util/format.hpp
util/mat.hpp
prereqs.hpp
)

set_target_properties(mlight PROPERTIES LINKER_LANGUAGE CXX)
