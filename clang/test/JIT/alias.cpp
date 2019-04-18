// RUN: %clangxx -o %t %s -fjit -std=c++14 -O3
// RUN: %t int short | FileCheck %s

#include <memory>
#include <cstdio>

class some_base_class
{
public:
  virtual ~some_base_class();
};

some_base_class::~some_base_class() = default;

template<typename T>
class some_foo : public some_base_class {
public:
  some_foo() {
    std::printf("H1: %d\n", (int) sizeof(T));
  }
};

template<typename T>
[[clang::jit]] std::unique_ptr<some_base_class> factory() {
  return std::make_unique<some_foo<T>>();
}

int main(int argc, char ** argv) {
  std::printf("%s\n", "main");
// CHECK: main

  factory<argv[1]>();
// CHECK-NEXT: H1: 4
  factory<argv[2]>();
// CHECK-NEXT: H1: 2
}

