// MinGW/UCRT's <cstdlib> declares neither C11 aligned_alloc nor std::aligned_alloc, which
// kickmix uses to get 32-byte-aligned gate arrays. The build script rewrites those call
// sites (and their matching releases) to UCRT's _aligned_malloc/_aligned_free pair; this
// header only makes sure that pair is declared.
#pragma once
#include <cstdlib>
