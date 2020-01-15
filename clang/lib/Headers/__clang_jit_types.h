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

#if __has_include(<source_location>) && __cplusplus > 201103L
#include <source_location>
#define __SL_NS std
#elif __has_include(<experimental/source_location>) && __cplusplus > 201103L
#include <experimental/source_location>
#define __SL_NS std::experimental
#else
#define __SL_NS __clang_jit
#include <cstdint>
namespace __clang_jit {
// Include our own implementation here, with a current() constructor compatible
// with libstdcxx.
struct source_location {
  static constexpr source_location
  current(const char* file = __builtin_FILE(),
          const char* func = __builtin_FUNCTION(),
          int line = __builtin_LINE(),
          int col = 0) noexcept {
    return source_location(file, func, line, col);
  }

  constexpr source_location() noexcept
    : file("unknown"), func("unknown"), lne(0), col(0) { }

  constexpr uint_least32_t line() const noexcept { return lne; }
  constexpr uint_least32_t column() const noexcept { return col; }
  constexpr const char* file_name() const noexcept { return file; }
  constexpr const char* function_name() const noexcept { return func; }

private:
  constexpr source_location(const char* file, const char* func,
                           int line, int col) noexcept
    : file(file), func(func), lne(line), col(col) { }

  const char* file;
  const char* func;
  uint_least32_t lne;
  uint_least32_t col;
};
} // namespace __clang_jit
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

extern "C" void __clang_jit_release(void *);
extern "C" void *__clang_jit_error_vector(void *);
extern "C" void *__clang_jit_warning_vector(void *);

struct dynamic_function_template_instantiation_base {
  ~dynamic_function_template_instantiation_base() {
    __clang_jit_release(d);
  }

  dynamic_function_template_instantiation_base(
    const dynamic_function_template_instantiation_base &) = delete;

  dynamic_function_template_instantiation_base(
    dynamic_function_template_instantiation_base &&) = default;

  const std::vector<diagnostic> &warnings() const {
    return *reinterpret_cast<std::vector<diagnostic> *>(__clang_jit_warning_vector(d));
  }

  const std::vector<diagnostic> &errors() const {
    return *reinterpret_cast<std::vector<diagnostic> *>(__clang_jit_error_vector(d));
  }

  operator bool() const {
    return p != nullptr;
  }

  explicit dynamic_function_template_instantiation_base(
    void *p, void *d) : d(d), p(p) {}

private:
  void *d;

protected:
  void *p;
};

template <typename Fn>
struct dynamic_function_template_instantiation :
  public dynamic_function_template_instantiation_base {
  using dynamic_function_template_instantiation_base::dynamic_function_template_instantiation_base;

  template <typename... Args>
  typename std::result_of<Fn&(Args...)>::type
  operator () (Args&&... args) {
    return reinterpret_cast<Fn&>(p)(std::forward<Args>(args)...);
  }
};

extern "C" void __clang_jit_dd_release(void *);
extern "C" void __clang_jit_dd_reference(void *);
extern "C" void *__clang_jit_dd_compose(void *, std::size_t, ...);

struct dynamic_template_argument {
  ~dynamic_template_argument() {
    __clang_jit_dd_release(d);
  }

  dynamic_template_argument(const dynamic_template_argument &dd) {
    d = dd.d;
    __clang_jit_dd_reference(d);
  }

  dynamic_template_argument(dynamic_template_argument &&) = default;

  explicit dynamic_template_argument(void *d) : d(d) {}

private:
  void *d;
};

#if __has_feature(clang_cxx_jit)
struct dynamic_template_template_argument : public dynamic_template_argument {
  using dynamic_template_argument::dynamic_template_argument;

  template <typename... Args>
  dynamic_template_argument compose(Args args...) const {
    auto gd = [](const dynamic_template_argument &dd) {
      return dd->d;
    };

    return dynamic_template_argument(
      __clang_jit_dd_compose(d, sizeof...(args), gd(make(args))...));
  }

protected:

  // If you compose with anything other than another descriptor, convert it first.
  template <typename Arg>
  dynamic_template_argument make(const Arg &arg) {
    return __clang_dynamic_template_argument<arg>;
  }

  template <>
  dynamic_template_argument make<dynamic_template_argument>(
                              const dynamic_template_argument &dd) {
    return dd;
  }
};
#endif // __has_feature(clang_cxx_jit)

} // namespace __clang_jit

#undef __SL_NS

#endif // __CLANG_JIT_TYPES_H__

