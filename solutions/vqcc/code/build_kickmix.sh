#!/usr/bin/env bash
# Build the official Qualtran kickmix Python extension for Windows/MSYS2-UCRT.
# Source: github.com/quantumlib/Qualtran @ c58703d0 (Apache-2.0), the same commit the
# competition dataset's kickmix.so was built from.
#
# Build-only adaptations for MinGW-w64, none touching kickmix logic:
#   * three call sites use C11 std::aligned_alloc, which MinGW's UCRT headers don't declare.
#     They are rewritten to _aligned_malloc, and the releases that undo them (and nothing
#     else) to _aligned_free, in a private copy of the tree. Files that allocate with plain
#     malloc and free normally are left alone.
#   * the cmd/ CLI tools and their arg_parse helper are excluded: the Python extension never
#     uses them, and arg_parse.h does not compile without a <cstdint> include.
set -euo pipefail
SRC=${1:?usage: build_kickmix.sh <Qualtran-checkout>/kickmix <out-dir>}
OUT=${2:?usage: build_kickmix.sh <Qualtran-checkout>/kickmix <out-dir>}
PY=${PY:-python}
COMPAT=$(cd "$(dirname "$0")" && pwd)/compat
PYINC=$($PY -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
PBLIB=$($PY -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))')
PYVER=$($PY -c 'import sysconfig;print("python"+sysconfig.get_config_var("py_version_nodot"))')
PBINC=$($PY -c 'import pybind11;print(pybind11.get_include())')

rm -rf "$OUT/src" "$OUT/obj"
mkdir -p "$OUT/src" "$OUT/obj"
cp -r "$SRC/src" "$OUT/src/"
T=$OUT/src/src/kickmix
sed -i -e 's|std::aligned_alloc(32, bytes)|_aligned_malloc(bytes, 32)|' \
    "$T/circuit/circuit.cc" "$T/circuit/mutable_circuit.cc" "$T/py/val/converted_array.pybind.cc"
sed -i -e 's|(char \*)aligned_alloc(alloc_alignment, new_size)|(char *)_aligned_malloc(new_size, alloc_alignment)|' \
    "$T/mem/monotonic_arena.cc"
sed -i -e 's|\bfree(|_aligned_free(|g' \
    "$T/circuit/circuit.cc" "$T/circuit/mutable_circuit.cc" \
    "$T/mem/monotonic_arena.cc" "$T/py/val/converted_array.pybind.cc" "$T/py/val/converted_array.pybind.h"

BASE="g++ -std=c++20 -O3 -march=native -fPIC -fvisibility=hidden -fno-strict-aliasing"
INCS="-I$OUT/src/src -I$PBINC -I$PYINC -include $COMPAT/msys_aligned_alloc.h"
OBJS=()
while read -r f; do
  o="$OUT/obj/$(printf '%s' "$f" | tr '/' '_').o"
  $BASE $INCS -c "$OUT/src/$f" -o "$o"
  OBJS+=("$o")
done < <(cd "$OUT/src" && find src -name '*.cc' \
           -not -name '*.test.cc' -not -name '*.perf.cc' -not -name 'main.cc' \
           -not -path '*/cmd/*' -not -name 'arg_parse.cc' | sort)
$BASE -shared -o "$OUT/kickmix.pyd" "${OBJS[@]}" \
  -L"$PBLIB" -l"$PYVER" -static-libgcc -static-libstdc++
echo "built $OUT/kickmix.pyd"
