#ifndef FORMAT_HPP
#define FORMAT_HPP

#include <cstring>
#include <cstdio>
#include <cassert>
#include <tuple>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace mlight {

// Traits utils
template<class T> struct replace_extent { typedef T type; };
template<class T> struct replace_extent<T[]> { typedef const T* type; };
template<class T, unsigned long N> struct replace_extent<T[N]> { typedef const T* type;};

// Print to ostream
template<class T> void print_ostream(std::ostringstream& stream, const T& last) { stream << last; }
template<class T, class... Args>
void print_ostream(std::ostringstream& stream, const T& first, const Args&... rest) { stream << "\t" << first; print_ostream(stream, rest...); }
template<class... T>

// Utility for printing variables
void print(const T&... args) { std::ostringstream buf; print_ostream(buf, args...); std::cout << buf.str() << std::endl; }

// Improved snprintf that accepts std::string and other types
template<typename T>
typename std::enable_if<std::is_integral<T>::value ||
                        std::is_pointer<T>::value ||
                        std::is_floating_point<T>::value ||
                        std::is_enum<T>::value, T>::type sprintf_cast(const T& t) { return t; }
inline const char *sprintf_cast(const char *s) { return s; }
inline const char *sprintf_cast(const std::string& s) { return s.c_str(); }

template<int N> struct expand_snprintf {
    template<typename Tuple, typename... Args> static int call(char *s, size_t maxlen, const char *format, const Tuple& tuple, const Args&... args) {
        return expand_snprintf<N-1>::call(s, maxlen, format, tuple, sprintf_cast(std::get<N-1>(tuple)), args...); }};
template<> struct expand_snprintf<0> {
    template<typename Tuple, typename... Args> static int call(char *s, size_t maxlen, const char *format, const Tuple& tuple, const Args&... args) {
        return snprintf(s, maxlen, format, args...); }};

template<typename... Args>
int snprintf(char *s, size_t maxlen, const char *format, const Args&... args) {
    std::tuple<typename replace_extent<Args>::type...> tuple(args...);
    return expand_snprintf<std::tuple_size<decltype(tuple)>::value>::call(s, maxlen, format, tuple);
}

template<typename... Args>
inline int snprintf(char *s, size_t maxlen, const char *format) {
    std::strncpy(s, format, maxlen);
    s[maxlen-1] = 0;
    return strlen(s);
}

// Format strings with the sprintf style, accepting std::string and string convertible types for %s
template<typename... Args>
inline std::string format() { return std::string(); }

template<typename... Args>
inline std::string format(const std::string& f) { return f; }

template<typename... Args>
std::string format(const std::string& f, const Args&... args) {
    int n, size = 1024;
    std::string str;
    while(true) {
        str.resize(size);
        n = snprintf(&str[0], size, f.c_str(), args...);
        assert(n != -1);
        if(n < size) {
            str.resize(n);
            return str;
        }
        size *= 2;
    }
}

// Print format to stdout
template<typename... Args>
inline void pformat() { }

template<typename... Args>
inline void pformat(const std::string& f) {
    std::cout << f << std::endl;
}

template<typename... Args>
void pformat(const std::string& f, const Args&... args) {
    std::cout << format(f, args...) << std::endl;
}

}

#endif
