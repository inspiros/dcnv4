#pragma once

#include <ATen/Dispatch.h>

#define AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES_AND(SCALARTYPE, ...) \
  AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES(__VA_ARGS__)               \
  AT_DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

#define AT_DISPATCH_REDUCED_FLOATING_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                                       \
      TYPE,                                                                 \
      NAME,                                                                 \
      AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES_AND(SCALARTYPE, __VA_ARGS__))

#define AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, ...) \
  AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES(__VA_ARGS__)                              \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                                        \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)

#define AT_DISPATCH_REDUCED_FLOATING_TYPES_AND2(       \
    SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)         \
  AT_DISPATCH_SWITCH(                                  \
      TYPE,                                            \
      NAME,                                            \
      AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES_AND2(    \
          SCALARTYPE1, SCALARTYPE2, __VA_ARGS__))
