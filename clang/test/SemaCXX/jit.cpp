// RUN: %clang_cc1 -triple x86_64-linux-gnu -verify -fsyntax-only -std=c++11 -fjit %s

#if !__has_feature(clang_cxx_jit)
#error "Missing __has_feature"
#endif

[[clang::jit]] void foo() { } // expected-warning{{'jit' attribute only applies to function templates}}

template <typename T, int I>
[[clang::jit]] void bar(T x) { }


