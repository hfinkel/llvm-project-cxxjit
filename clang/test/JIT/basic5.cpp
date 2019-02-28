// RUN: %clangxx -o %t %s -fjit -std=c++11 -O0
// RUN: %t | FileCheck %s

#include <cstdio>

namespace tns {
class F {
public:
  static void p() {
    printf("type: F\n");
  }
};

class G {
public:
  static void p() {
    printf("type: G\n");
  }
};

template<int X>
class G2 {
public:
  static void p() {
    printf("type: G2: %d\n", X);
  }
};

template<typename T, int X>
class G3 {
public:
  static void p() {
    T::p();
    printf("type: G3: %d\n", X);
  }
};

template <typename T>
[[clang::jit]] void foo() {
  std::printf("here\n");
  T::p();
}

int do_main() {
  std::printf("%s\n", "main");
// CHECK: main

  foo<"F">();
// CHECK-NEXT: here
// CHECK-NEXT: type: F

  const char *s = "G";
  foo<s>();
// CHECK-NEXT: here
// CHECK-NEXT: type: G

  const char *s2 = "tns::G";
  foo<s2>();
// CHECK-NEXT: here
// CHECK-NEXT: type: G

  const char *s3 = "G2<5>";
  foo<s3>();
// CHECK-NEXT: here
// CHECK-NEXT: type: G2: 5

  const char *s4 = "G3<F, 8>";
  foo<s4>();
// CHECK-NEXT: here
// CHECK-NEXT: type: F
// CHECK-NEXT: type: G3: 8

  const char *s5 = "G3<G2<16>, 8>";
  foo<s5>();
// CHECK-NEXT: here
// CHECK-NEXT: type: G2: 16
// CHECK-NEXT: type: G3: 8

  return 0;
}
} // namespace tns

int main() {
  return tns::do_main();
}
