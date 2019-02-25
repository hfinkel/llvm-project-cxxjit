// RUN: %clang_cc1 -triple x86_64-linux-gnu -verify -fsyntax-only -std=c++11 -fjit %s

#if !__has_feature(clang_cxx_jit)
#error "Missing __has_feature"
#endif

[[clang::jit]] void foo() { } // expected-warning{{'jit' attribute only applies to function templates}}

template <typename T, int I>
[[clang::jit]] void bar(T x) { }

template <>
void bar<int, 10>(int x) { }

template <typename T>
void u1() { }

template <void (*Q)(int)>
void u2() { // expected-note{{candidate template ignored: invalid explicitly-specified argument for template parameter 'Q'}}
  Q(5);
}

void test1(int argc) {
  u1<decltype(bar<int, argc+2>)>();

  u2<bar<int, argc+2>>(); // expected-error{{no matching function for call to 'u2'}}
  u2<bar<int, 10>>();
}

