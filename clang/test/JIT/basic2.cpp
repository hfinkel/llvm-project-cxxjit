// RUN: %clangxx -o %t %s -fjit -std=c++11 -O0
// RUN: %t | FileCheck %s

#include <iostream>

int g = 0;
int go = 0;

void pnt(int x) {
  std::cout << "e: " << x << "\n";
}

static void pnts(int x) {
  std::cout << "s: " << x << "\n";
}

static void pntso(int x) {
  std::cout << "so: " << x << "\n";
}

template <typename T, int x>
[[clang::jit]] void bar() {
  int y = x;
  std::cout << "r: " << y << "\n";
  pnt(y);

  pntso(y);

  ++g;
  pnts(g);

  ++go;
  std::cout << "go: " << go << "\n";
}

template <>
void bar<int, 10>() {
  int q = 5;
  std::cout << "r: " << q << "\n";
  pnt(q);

  g += 2;
  pnts(g);
}

int main(int argc, char *argv[]) {
  std::cout << "main\n";
// CHECK: main

  void (*f)() = bar<int, argc+2>;
  f();
// CHECK-NEXT: r: 3
// CHECK-NEXT: e: 3
// CHECK-NEXT: so: 3
// CHECK-NEXT: s: 1
// CHECK-NEXT: go: 1
  bar<int, argc+1>();
// CHECK-NEXT: r: 2
// CHECK-NEXT: e: 2
// CHECK-NEXT: so: 2
// CHECK-NEXT: s: 2
// CHECK-NEXT: go: 2
  bar<int, argc+8>();
// CHECK-NEXT: r: 9
// CHECK-NEXT: e: 9
// CHECK-NEXT: so: 9
// CHECK-NEXT: s: 3
// CHECK-NEXT: go: 3

  bar<int, 5>();
// CHECK-NEXT: r: 5
// CHECK-NEXT: e: 5
// CHECK-NEXT: so: 5
// CHECK-NEXT: s: 4
// CHECK-NEXT: go: 4

  bar<int, argc+9>();
// CHECK-NEXT: r: 5
// CHECK-NEXT: e: 5
// CHECK-NEXT: s: 6
  bar<int, 10>();
// CHECK-NEXT: r: 5
// CHECK-NEXT: e: 5
// CHECK-NEXT: s: 8
}

