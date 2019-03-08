// RUN: %clangxx -o %t %s -fjit -std=c++11 -O0
// RUN: %t | FileCheck %s

// RUN: %clangxx -o %t %s -fjit -std=c++11 -O0 -g
// RUN: %t | FileCheck %s

#include <iostream>

static int gi = 4;

template <int *X>
[[clang::jit]] void bar() {
  std::cout << "a: " << *X << "\n";
}

static void ps() {
  std::cout << "f: ok\n";
}

template <void (*X)()>
[[clang::jit]] void fbar() {
  X();
}

int main(int argc, char *argv[]) {
  std::cout << "main\n";
// CHECK: main

  void (*f)() = bar<&argc>;
  f();
// CHECK-NEXT: a: 1
  bar<&gi>();
// CHECK-NEXT: a: 4
  fbar<ps>();
// CHECK-NEXT: f: ok
}

