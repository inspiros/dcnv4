#pragma once

#if defined(_WIN32) && !defined(DCNV4_BUILD_STATIC_LIBS)
#if defined(dcnv4_EXPORTS)
#define DCNV4_API __declspec(dllexport)
#else
#define DCNV4_API __declspec(dllimport)
#endif
#else
#define DCNV4_API
#endif

#if (defined __cpp_inline_variables) || __cplusplus >= 201703L
#define DCNV4_INLINE_VARIABLE inline
#else
#ifdef _MSC_VER
#define DCNV4_INLINE_VARIABLE __declspec(selectany)
#define HINT_MSVC_LINKER_INCLUDE_SYMBOL
#else
#define DCNV4_INLINE_VARIABLE __attribute__((weak))
#endif
#endif
