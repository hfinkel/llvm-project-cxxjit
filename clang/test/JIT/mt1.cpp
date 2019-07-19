// RUN: %clangxx -o %t %s -fjit -std=c++11 -O0
// RUN: %t | FileCheck %s

#include <iostream>
 
class Test
{
private:
  int x;

public:
  Test();

  template<int val>
  int getResult();
};

Test::Test() : x(2) {}

template <int val>
[[clang::jit]] int Test::getResult() {
  return val * x;
}

int main() {
  std::cout << "main\n";
// CHECK: main

  Test obj;

  std::cout << "result = " << obj.getResult<5>() << "\n";
// CHECK-NEXT: result = 10
  std::cout << "result = " << obj.getResult<6>() << "\n";
// CHECK-NEXT: result = 12
}

