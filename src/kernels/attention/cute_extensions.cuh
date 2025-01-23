#pragma once

#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>

#include "cute/config.hpp"
#include "cute/layout.hpp"
#include "cute/layout_composed.hpp"
#include "cute/tensor_impl.hpp"

namespace cute {

template <class NewT, typename ThrMMA, class BTensor>
CUTE_HOST_DEVICE constexpr auto make_fragment_B(const ThrMMA& thr_mma,
                                                BTensor const& btensor) {
  return make_fragment_like<NewT>(thr_mma.partition_B(btensor));
}

template <size_t I, class IntTupleA, class IntTupleB>
CUTE_HOST_DEVICE constexpr auto elem_less(IntTupleA const& a,
                                          IntTupleB const& b) {
  return elem_less(get<I>(a), get<I>(b));
}

template <class Copy_Atom, class TensorS, class TensorD>
CUTE_HOST_DEVICE void zfill(const Copy_Atom& copy_atom,
                            const TensorS& src,
                            TensorD&& dst) {
  CUTE_STATIC_ASSERT(TensorS::rank == TensorD::rank, "rank-mismatch.");

  auto has_with_bool = cute::is_valid(
      [](auto t) -> void_t<decltype(declval<typename decltype(t)::Traits>()
                                        .with(true))> {},
      copy_atom);
  if constexpr (has_with_bool) {
    constexpr int R = TensorD::rank;
    if constexpr (R == 1) {  // Dispatch the copy
      copy_atom.with(false).call(src, dst);
    } else {  // Loop over all but the first mode
      Tensor src_v = group_modes<1, R>(src);
      Tensor dst_v = group_modes<1, R>(dst);
      CUTE_UNROLL
      for (int i = 0; i < size<1>(dst_v); ++i) {
        copy_atom.with(false).call(src_v(_, i), dst_v(_, i));
      }
    }
  } else {
    // just call clear if no with method
    clear(dst);
  }
}

template <class... CopyArgs, class TensorS, class TensorD>
CUTE_HOST_DEVICE void zfill(const Copy_Atom<CopyArgs...>& copy_atom,
                            const TensorS& src,
                            TensorD& dst) {
  zfill(copy_atom, src, dst);
}

template <bool EVEN_MN,
          bool EVEN_K,
          bool ZFILL_MN,
          bool ZFILL_K,
          class CopyAtom,
          class TV,
          class Tiler,
          class TensorS,
          class TensorD,
          class TensorC,
          class Coord>
CUTE_HOST_DEVICE void safe_copy(
    const TiledCopy<CopyAtom, TV, Tiler>& tiled_copy,
    const TensorS& src,       // (CPY, CPY_M/N, CPY_K)
    TensorD& dst,             // (CPY, CPY_M/N, CPY_K)
    const TensorC& identity,  // (CPY, CPY_M/N, CPY_K) -> (blk_m/n, blk_k)
    const Coord& max_coord    // max_coord(blk_m/n, blk_k)
) {
  CUTE_STATIC_ASSERT(TensorS::rank == TensorD::rank, "rank-mismatch.");
  auto copy_atom = static_cast<const CopyAtom&>(tiled_copy);

  if constexpr (!EVEN_MN && !EVEN_K) {
    // handle both m/n and k oob
    CUTE_UNROLL
    for (int mi = 0; mi < size<1>(src); ++mi) {
      if (elem_less<0>(identity(_0{}, mi, _0{}), max_coord)) {
        CUTE_UNROLL
        for (int ki = 0; ki < size<2>(src); ++ki) {
          if (elem_less<1>(identity(_0{}, _0{}, ki), max_coord)) {
            copy(copy_atom, src(_, mi, ki), dst(_, mi, ki));
          } else {
            if constexpr (ZFILL_K) {
              zfill(copy_atom, src(_, mi, ki), dst(_, mi, ki));
            }
          }
        }
      } else {
        if constexpr (ZFILL_MN) {
          zfill(copy_atom, src(_, mi, _), dst(_, mi, _));
        }
      }
    }
  } else if constexpr (!EVEN_MN && EVEN_K) {
    // only handle m/n oob
    CUTE_UNROLL
    for (int mi = 0; mi < size<1>(src); ++mi) {
      if (elem_less<0>(identity(_0{}, mi, _0{}), max_coord)) {
        copy(copy_atom, src(_, mi, _), dst(_, mi, _));
      } else {
        if constexpr (ZFILL_MN) {
          zfill(copy_atom, src(_, mi, _), dst(_, mi, _));
        }
      }
    }
  } else if constexpr (EVEN_MN && !EVEN_K) {
    // only handle k oob
    CUTE_UNROLL
    for (int ki = 0; ki < size<2>(src); ++ki) {
      if (elem_less<1>(identity(_0{}, _0{}, ki), max_coord)) {
        copy(copy_atom, src(_, _, ki), dst(_, _, ki));
      } else {
        if constexpr (ZFILL_K) {
          zfill(copy_atom, src(_, _, ki), dst(_, _, ki));
        }
      }
    }
  } else {
    // no oob, just copy
    copy(copy_atom, src, dst);
  }
}

// support mixed precision mma
// Dispatch [4]: (V,M) x (V,N) => (V,M,N)
template <class MMA, class FragmentA, class FragmentB, class FragmentC>
CUTE_HOST_DEVICE void mixed_gemm(MMA_Atom<MMA> const& mma,
                                 const FragmentA& A,  // (V,M) Logical data
                                 const FragmentB& B,  // (V,N) Logical data
                                 FragmentC& C)        // (V,M,N) Logical data
{
  using AType = typename FragmentA::value_type;
  using BType = typename FragmentB::value_type;

  if constexpr (std::is_same_v<AType, BType>) {
    // same type, call gemm
    gemm(mma, A, B, C);
  } else {
    // handle mixed precision
    auto M = size<1>(A);
    auto N = size<1>(B);

    // Col-major serpentine iteration
    CUTE_UNROLL
    for (int n = 0; n < N; ++n) {
      // Covnert B to same type as A before gemm
      auto B_ = make_fragment_like<AType>(B(_, n));
      fast_cast(B(_, n), B_);

      CUTE_UNROLL
      for (int m = 0; m < M; ++m) {
        int ms = (n & 1) ? M - 1 - m : m;  // Serpentine coordinate
        gemm(mma, A(_, ms), B_, C(_, ms, n));
      }
    }
  }
}

template <int N, int I, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto upcast(Shape const& shape,
                                       Stride const& stride) {
  if constexpr (is_tuple<Shape>::value) {
    return transform_layout(shape, stride, [](auto const& s, auto const& d) {
      return upcast<N, I>(s, d);
    });
  } else if constexpr (is_scaled_basis<Stride>::value) {
    if constexpr (Stride::mode() == I) {
      return make_layout(shape_div(shape, Int<N>{}),
                         shape_div(stride, Int<N>{}));
    } else {
      return make_layout(shape, stride);
    }
  } else {
    return upcast<N>(shape, stride);
  }

  CUTE_GCC_UNREACHABLE;
}

template <int N,
          class OuterShape,
          class OuterStride,
          class Offset,
          class Shape,
          class Stride>
CUTE_HOST_DEVICE constexpr auto upcast(
    ComposedLayout<Layout<OuterShape, OuterStride>,
                   Offset,
                   Layout<Shape, Stride>> const& layout) {
  // Find index of the stride-1 mode - that is the only one that requires
  // updating inner shape and offset
  auto idx = find_if(layout.layout_a().stride(),
                     [](auto x) { return is_constant<1, decltype(x)>{}; });
  constexpr int I = decltype(idx)::value;

  // Upcast the outer layout (works as expected)
  auto outer = upcast<N>(layout.layout_a());

  // Upcast the accumulated offset along stride-1 mode
  auto offset = as_arithmetic_tuple(
      replace<I>(layout.offset(), upcast<N>(get<I>(layout.offset()))));

  // Upcast the inner layout's shape along stride-1 mode
  auto inner =
      upcast<N, I>(layout.layout_b().shape(), layout.layout_b().stride());

  return composition(outer, offset, inner);
}

}  // namespace cute