// RUN: %clangxx -o %t %s -fjit -std=c++11 -O0
// RUN: %t | FileCheck %s

#include <cstdio>

class e { };

static void foos() { }

template <int x>
[[clang::jit]] void foo() noexcept {
  foos();
}

// Force Clang to generate __clang_call_terminate in the base object file (and
// we'll need it as well in the JIT-compiled code).
void hrmm() noexcept {
  foos();
}

int main(int argc, char *argv[]) {
  std::printf("%s\n", "main");
// CHECK: main

  hrmm();
  foo<argc>();
  return 0;
}

