// RUN: %clangxx -o %t %s -fjit -std=c++11 -O0
// RUN: %t | FileCheck %s

// RUN: %clangxx -o %t %s -fjit -std=c++11 -O3
// RUN: %t | FileCheck %s

#include <cstdio>

template<int... Jitted, typename LB>
[[clang::jit]] void jittable_func(LB body){
  body(Jitted...);
}

template<typename LB, typename... Args>
void jit_region(LB body, Args... args){
  jittable_func<args...>(body);
}

struct test{};
int main(){
  std::printf("%s\n", "main");
// CHECK: main

  jit_region(
      []() {
         std::printf("%s\n", "here");
      }
  );
// CHECK-NEXT: here

  jit_region(
      [](const int p) {
         std::printf("%d\n", p);
      }, 2
  );
// CHECK-NEXT: 2

  jit_region(
      [](const int p, const int q) {
         std::printf("%d %d\n", p, q);
      }, 2, 4
  );
// CHECK-NEXT: 2 4

  jit_region(
      [](const int p, const int q, const int r) {
         std::printf("%d %d %d\n", p, q, r);
      }, 2, 4, 6
  );
// CHECK-NEXT: 2 4 6
}

