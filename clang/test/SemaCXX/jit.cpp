// RUN: %clang_cc1 -triple x86_64-linux-gnu -verify -fsyntax-only -std=c++11 -fjit %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -verify -fsyntax-only -std=c++14 -fjit %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -verify -fsyntax-only -std=c++17 -fjit %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -verify -fsyntax-only -std=c++2a -fjit %s

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

#ifdef __cpp_decltype_auto

template <typename T, int I>
[[clang::jit]] decltype(auto) bara() { return I; } // expected-note{{'bara<int, argc + 2>' declared here}}

template <>
decltype(auto) bara<int, 10>() { return 5; }

void test2(int argc) {
  auto x = bara<int, argc+2>(); // expected-error{{function 'bara<int, argc + 2>' with deduced return type cannot be used before it is defined}}

  auto y = bara<int, 10>();
}

#endif // __cpp_decltype_auto
