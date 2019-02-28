// RUN: %clangxx -o %t %s -fjit -std=c++11 -O0
// RUN: %t | FileCheck %s

#include <cstdio>
#include <string>

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

template <typename T, int s>
[[clang::jit]] void foo() {
  std::printf("here\n");
  T::p();
}

int do_main() {
  std::printf("%s\n", "main");
// CHECK: main

  foo<"F", 0>();
// CHECK-NEXT: here
// CHECK-NEXT: type: F

  std::string s = "G";
  foo<s, 0>();
// CHECK-NEXT: here
// CHECK-NEXT: type: G

  std::string s2 = "tns::G";
  foo<s2, 0>();
// CHECK-NEXT: here
// CHECK-NEXT: type: G

  std::string s3 = "G2<5>";
  foo<s3, 0>();
// CHECK-NEXT: here
// CHECK-NEXT: type: G2: 5

  std::string s4 = "G3<F, 8>";
  foo<s4, 0>();
// CHECK-NEXT: here
// CHECK-NEXT: type: F
// CHECK-NEXT: type: G3: 8

  std::string s5 = "G3<G2<16>, 8>";
  foo<s5, 0>();
// CHECK-NEXT: here
// CHECK-NEXT: type: G2: 16
// CHECK-NEXT: type: G3: 8

  return 0;
}
} // namespace tns

int main() {
  return tns::do_main();
}
