// RUN: %clangxx -o %t %s -fjit -std=c++11 -O0
// RUN: %t | FileCheck %s

#include <cstdio>

class F {
public:
  static void p() {
    std::printf("type: F\n");
  }
};

class G {
public:
  static void p() {
    std::printf("type: G\n");
  }
};

template <typename T>
[[clang::jit]] void foo() {
  std::printf("here\n");
  T::p();
}

int main() {
  std::printf("%s\n", "main");
// CHECK: main

  foo<"F">();
// CHECK-NEXT: here
// CHECK-NEXT: type: F

  const char *s = "G";
  foo<s>();
// CHECK-NEXT: here
// CHECK-NEXT: type: G

  const char *s2 = "::G";
  foo<s2>();
// CHECK-NEXT: here
// CHECK-NEXT: type: G

  return 0;
}

