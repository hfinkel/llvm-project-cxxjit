/*===----------------- __clang_jit_types.h - C++ JIT support ---------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

/*
 * WARNING: This header is intended to be directly -include'd by
 * the compiler and is not supposed to be included by users.
 */

#ifndef __CLANG_JIT_TYPES_H__
#define __CLANG_JIT_TYPES_H__

#if __has_include(<source_location>)
#include <source_location>
#define __SL_NS std
#elif __has_include(<experimental/source_location>)
#include <experimental/source_location>
#define __SL_NS std::experimental
#else
#error Cannot find source_location
#endif

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace __clang_jit {

struct diagnostic {
  using source_location = __SL_NS::source_location;

  const std::string &message() const {
    return m;
  }

  const source_location &loc() const {
    return s;
  }

  diagnostic(const std::string &m, const source_location &s) :
    m(m), s(s) {}

private:
  std::string m;
  source_location s;
};

struct dynamic_function_template_instantiation_base {
  const std::vector<diagnostic> &warnings() const {
    return w;
  }

  const std::vector<diagnostic> &errors() const {
    return e;
  }

  operator bool() const {
    return p != nullptr;
  }

  dynamic_function_template_instantiation_base(
    const std::vector<diagnostic> &w,
    const std::vector<diagnostic> &e,
    void  *p) : w(w), e(e), p(p) {}

private:
  std::vector<diagnostic> w, e;

protected:
  void *p;
};

template <typename Fn>
struct dynamic_function_template_instantiation :
  public dynamic_function_template_instantiation_base {
  using dynamic_function_template_instantiation_base::dynamic_function_template_instantiation_base;

  template <typename... Args>
  typename std::result_of<Fn(Args...)>::type
  operator () (Args&&... args) {
    return reinterpret_cast<Fn>(p)(std::forward<Args>(args)...);
  }
};

} // namespace __clang_jit

#undef __SL_NS

#endif // __CLANG_JIT_TYPES_H__

