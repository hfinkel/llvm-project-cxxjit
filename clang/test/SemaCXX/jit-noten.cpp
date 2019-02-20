// RUN: %clang_cc1 -triple x86_64-linux-gnu -verify -fsyntax-only -std=c++11 %s

#if __has_feature(clang_cxx_jit)
#error "Unexpected __has_feature"
#endif

template <typename T, int I>
[[clang::jit]] void bar(T x) { } // expected-warning{{'jit' attribute ignored; use -fjit to enable the attribute}}

