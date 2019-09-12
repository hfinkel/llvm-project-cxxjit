// RUN: %clangxx -o %t %s -fjit -std=c++11 -O0
// RUN: %t a | FileCheck %s

#include <stdio.h>

template <unsigned N, const char (&a)[N]>
void inner() {
  printf("r: %s %d\n", a, N);
}

template <unsigned N, const char (&a)[N]>
[[clang::jit]] void testa() {
  inner<N, a>();
}

int main(int argc, char *argv[]) {
  printf("main\n");
// CHECK: main

  testa<argc, argv[1]>();
// CHECK-NEXT: r: a 2
}

