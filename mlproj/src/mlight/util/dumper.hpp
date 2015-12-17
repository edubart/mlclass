#ifndef DUMPER_HPP
#define DUMPER_HPP

#include <cstddef>
#include <iostream>

namespace mlight {

struct dumper {
    dumper() { }
    struct dumper_dummy {
        ~dumper_dummy() { std::cout << std::endl; }
        template<class T> dumper_dummy& operator<<(const T& v) { std::cout << v << " "; return *this; }
    };
    template<class T> dumper_dummy operator<<(const T& v) const { dumper_dummy d; d << v; return d; }
};
const static dumper dump;

}

#endif
