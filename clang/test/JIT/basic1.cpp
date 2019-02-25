// RUN: %clangxx -o %t %s -fjit -std=c++11 -O0
// RUN: %t | FileCheck %s

#include <iostream>

template <typename T, int x>
[[clang::jit]] int bar() {
  int y = x;
  return y;
}

template <>
int bar<int, 10>() {
  int q = 5;
  return q;
}

int main(int argc, char *argv[]) {
  std::cout << "main\n";
// CHECK: main

  int (*f)() = bar<int, argc+2>;
  std::cout << "r: " << f() << "\n";
// CHECK-NEXT: r: 3
  std::cout << "r: " << bar<int, argc+1>() << "\n";
// CHECK-NEXT: r: 2
  std::cout << "r: " << bar<int, argc+8>() << "\n";
// CHECK-NEXT: r: 9

  std::cout << "r: " << bar<int, 5>() << "\n";
// CHECK-NEXT: r: 5

  std::cout << "r: " << bar<int, argc+9>() << "\n";
// CHECK-NEXT: r: 5
  std::cout << "r: " << bar<int, 10>() << "\n";
// CHECK-NEXT: r: 5
}

