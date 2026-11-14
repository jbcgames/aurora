#include "gx.hpp"
#include "__gx.h"

#include "../../gx/fifo.hpp"

#include <cstring>

static __GXData_struct sSavedGXData;
static u32 sDlCacheGen = 1;

namespace {
struct DlPureCacheEntry {
  const void* ptr = nullptr;
  u32 nbytes = 0;
  u32 dlGen = 0;
  u64 layoutHash = 0;
  bool pure = false;
};
DlPureCacheEntry s_dlPureCache[2048][2]{};

struct DlBatchPureCacheEntry {
  const void* batchKey = nullptr;
  const void* firstPtr = nullptr;
  const void* lastPtr = nullptr;
  const void* minPtr = nullptr;
  const void* maxEnd = nullptr;
  u32 count = 0;
  u32 dlGen = 0;
  u64 layoutHash = 0;
  bool pure = false;
};
DlBatchPureCacheEntry s_dlBatchPureCache[1024][2]{};

u64 current_dl_layout_hash() noexcept {
  static u32 s_cachedVatGen = 0;
  static u64 s_cachedVatHash = 0;
  if (s_cachedVatGen != __gxShadowExt.vcdVatGen) {
    u64 h = 0x243F6A8885A308D3ULL;
    for (u32 f = 0; f < GX_MAX_VTXFMT; ++f) {
      h ^= (static_cast<u64>(__gx->vatA[f]) * 0x9E3779B97F4A7C15ULL) ^ (static_cast<u64>(__gx->vatB[f]) << 17) ^
           (static_cast<u64>(__gx->vatC[f]) << 37);
      h *= 0xBF58476D1CE4E5B9ULL;
    }
    s_cachedVatHash = h;
    s_cachedVatGen = __gxShadowExt.vcdVatGen;
  }
  return s_cachedVatHash ^ (static_cast<u64>(__gx->vcdLo) * 0x9E3779B97F4A7C15ULL) ^
         (static_cast<u64>(__gx->vcdHi) * 0xBF58476D1CE4E5B9ULL);
}
inline u32 __gx_dl_comp_type_bytes(u32 attr, u32 compType) noexcept {
  if (attr == GX_VA_CLR0 || attr == GX_VA_CLR1) {
    switch (compType) {
    case GX_RGB565:
    case GX_RGBA4:
      return 2;
    case GX_RGB8:
    case GX_RGBA6:
      return 3;
    case GX_RGBX8:
    case GX_RGBA8:
      return 4;
    default:
      return 0;
    }
  }
  switch (compType) {
  case GX_U8:
  case GX_S8:
    return 1;
  case GX_U16:
  case GX_S16:
    return 2;
  case GX_F32:
    return 4;
  default:
    return 0;
  }
}

inline u32 __gx_dl_vtx_stride(u32 fmt) noexcept {
  const u32 lo = __gx->vcdLo;
  const u32 hi = __gx->vcdHi;
  const u32 va = __gx->vatA[fmt];
  const u32 vb = __gx->vatB[fmt];
  const u32 vc = __gx->vatC[fmt];

  u32 stride = __builtin_popcount(lo & 0x1FFu);
  auto addAttr = [&](GXAttr attr, u32 vcdType, u32 cntBit, u32 compType, bool isNrm = false) noexcept -> bool {
    if (vcdType == GX_NONE) {
      return true;
    }
    if (vcdType == GX_INDEX8) {
      stride += (isNrm && (va & (1u << 9)) && (va & (1u << 31))) ? 3u : 1u;
      return true;
    }
    if (vcdType == GX_INDEX16) {
      stride += (isNrm && (va & (1u << 9)) && (va & (1u << 31))) ? 6u : 2u;
      return true;
    }
    if (vcdType != GX_DIRECT) {
      return false;
    }
    const u32 elemBytes = __gx_dl_comp_type_bytes(attr, compType);
    if (elemBytes == 0) {
      return false;
    }
    u32 compCount = 0;
    switch (attr) {
    case GX_VA_POS:
      compCount = (cntBit != 0) ? 3u : 2u;
      break;
    case GX_VA_NRM:
      compCount = (cntBit != 0) ? 9u : 3u;
      break;
    case GX_VA_CLR0:
    case GX_VA_CLR1:
      compCount = 1u;
      break;
    default:
      compCount = (cntBit != 0) ? 2u : 1u;
      break;
    }
    stride += elemBytes * compCount;
    return true;
  };

  if (!addAttr(GX_VA_POS, (lo >> 9) & 3u, va & 1u, (va >> 1) & 7u)) return 0;
  if (!addAttr(GX_VA_NRM, (lo >> 11) & 3u, (va >> 9) & 1u, (va >> 10) & 7u, true)) return 0;
  if (!addAttr(GX_VA_CLR0, (lo >> 13) & 3u, (va >> 13) & 1u, (va >> 14) & 7u)) return 0;
  if (!addAttr(GX_VA_CLR1, (lo >> 15) & 3u, (va >> 17) & 1u, (va >> 18) & 7u)) return 0;

  if (!addAttr(GX_VA_TEX0, (hi >> 0) & 3u, (va >> 21) & 1u, (va >> 22) & 7u)) return 0;
  if (!addAttr(GX_VA_TEX1, (hi >> 2) & 3u, vb & 1u, (vb >> 1) & 7u)) return 0;
  if (!addAttr(GX_VA_TEX2, (hi >> 4) & 3u, (vb >> 9) & 1u, (vb >> 10) & 7u)) return 0;
  if (!addAttr(GX_VA_TEX3, (hi >> 6) & 3u, (vb >> 18) & 1u, (vb >> 19) & 7u)) return 0;
  if (!addAttr(GX_VA_TEX4, (hi >> 8) & 3u, (vb >> 27) & 1u, (vb >> 28) & 7u)) return 0;
  if (!addAttr(GX_VA_TEX5, (hi >> 10) & 3u, (vc >> 5) & 1u, (vc >> 6) & 7u)) return 0;
  if (!addAttr(GX_VA_TEX6, (hi >> 12) & 3u, (vc >> 14) & 1u, (vc >> 15) & 7u)) return 0;
  if (!addAttr(GX_VA_TEX7, (hi >> 14) & 3u, (vc >> 23) & 1u, (vc >> 24) & 7u)) return 0;
  return stride;
}

bool __gx_dl_is_pure_prim(const void* data, u32 nbytes) noexcept {
  if (data == nullptr || nbytes == 0) {
    return true;
  }
  const u8* bytes = static_cast<const u8*>(data);
  const u8 first = bytes[0];
  // Quick reject for material / register display lists.
  if (first != 0x00 && first < 0x80 && first != GX_AURORA) {
    return false;
  }

  const u64 layoutHash = current_dl_layout_hash();
  const u64 ptrBits = reinterpret_cast<uintptr_t>(data);
  const u32 setIdx = static_cast<u32>((ptrBits ^ (ptrBits >> 11) ^ nbytes ^ (layoutHash >> 20)) & 2047u);
  DlPureCacheEntry* ways = s_dlPureCache[setIdx];
  for (int w = 0; w < 2; ++w) {
    if (ways[w].ptr == data && ways[w].nbytes == nbytes && ways[w].dlGen == sDlCacheGen &&
        ways[w].layoutHash == layoutHash) {
      return ways[w].pure;
    }
  }

  u32 strides[GX_MAX_VTXFMT]{};
  u8 strideReady = 0;
  u32 pos = 0;
  bool pure = true;
  while (pos < nbytes) {
    const u8 cmd = bytes[pos];
    if (cmd == 0x00) {
      ++pos;
      continue;
    }
    if (cmd == GX_AURORA) {
      if (pos + 10 > nbytes) {
        pure = false;
        break;
      }
      const u16 subCmd = (static_cast<u16>(bytes[pos + 1]) << 8) | bytes[pos + 2];
      if (subCmd != GX_AURORA_DRAW_INDEXED) {
        pure = false;
        break;
      }
      const u8 fmt = bytes[pos + 3] & GX_VAT_MASK;
      if ((strideReady & (1u << fmt)) == 0) {
        strides[fmt] = __gx_dl_vtx_stride(fmt);
        strideReady |= static_cast<u8>(1u << fmt);
      }
      const u32 stride = strides[fmt];
      if (stride == 0) {
        pure = false;
        break;
      }
      const u32 vtxCount = (static_cast<u32>(bytes[pos + 4]) << 8) | bytes[pos + 5];
      const u32 indexCount = (static_cast<u32>(bytes[pos + 6]) << 24) | (static_cast<u32>(bytes[pos + 7]) << 16) |
                             (static_cast<u32>(bytes[pos + 8]) << 8) | bytes[pos + 9];
      const u64 cmdSize = 10ULL + static_cast<u64>(indexCount) * 2ULL + static_cast<u64>(vtxCount) * stride;
      if (pos + cmdSize > nbytes) {
        pure = false;
        break;
      }
      pos += static_cast<u32>(cmdSize);
      continue;
    }
    const u8 op = cmd & GX_OPCODE_MASK;
    if (op != GX_QUADS && op != GX_TRIANGLES && op != GX_TRIANGLESTRIP && op != GX_TRIANGLEFAN && op != GX_LINES &&
        op != GX_LINESTRIP && op != GX_POINTS) {
      pure = false;
      break;
    }
    if (pos + 3 > nbytes) {
      pure = false;
      break;
    }
    const u8 fmt = cmd & GX_VAT_MASK;
    if ((strideReady & (1u << fmt)) == 0) {
      strides[fmt] = __gx_dl_vtx_stride(fmt);
      strideReady |= static_cast<u8>(1u << fmt);
    }
    const u32 stride = strides[fmt];
    if (stride == 0) {
      pure = false;
      break;
    }
    const u32 vtxCount = (static_cast<u32>(bytes[pos + 1]) << 8) | bytes[pos + 2];
    const u64 cmdSize = 3ULL + static_cast<u64>(vtxCount) * stride;
    if (pos + cmdSize > nbytes) {
      pure = false;
      break;
    }
    pos += static_cast<u32>(cmdSize);
  }

  ways[1] = ways[0];
  ways[0] = {data, nbytes, sDlCacheGen, layoutHash, pure};
  return pure;
}
} // namespace

void __gx_dl_invalidate_cache() noexcept {
  if (++sDlCacheGen == 0) {
    sDlCacheGen = 1;
  }
}

void __gx_dl_forget_memory(const void* lo, const void* hi) noexcept {
  const uintptr_t loAddr = reinterpret_cast<uintptr_t>(lo);
  const uintptr_t hiAddr = reinterpret_cast<uintptr_t>(hi);
  for (auto& ways : s_dlPureCache) {
    for (auto& entry : ways) {
      if (entry.ptr != nullptr) {
        const uintptr_t p = reinterpret_cast<uintptr_t>(entry.ptr);
        if (p >= loAddr && p < hiAddr) {
          entry.ptr = nullptr;
          entry.dlGen = 0;
        }
      }
    }
  }
  for (auto& ways : s_dlBatchPureCache) {
    for (auto& entry : ways) {
      if (entry.batchKey != nullptr) {
        const uintptr_t bk = reinterpret_cast<uintptr_t>(entry.batchKey);
        const uintptr_t mn = reinterpret_cast<uintptr_t>(entry.minPtr);
        const uintptr_t mx = reinterpret_cast<uintptr_t>(entry.maxEnd);
        if ((bk >= loAddr && bk < hiAddr) || (entry.minPtr != nullptr && mn < hiAddr && mx > loAddr)) {
          entry.batchKey = nullptr;
          entry.dlGen = 0;
        }
      }
    }
  }
}

extern "C" {
void GXBeginDisplayList(void* list, u32 size) {
  CHECK(!aurora::gx::fifo::in_display_list(), "Display list began twice!");

  // Flush any pending dirty state before recording
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }

  // Save current shadow register state if requested
  if (__gx->dlSaveContext != 0) {
    std::memcpy(&sSavedGXData, __gx, sizeof(sSavedGXData));
  }

  __gx->inDispList = 1;

  // Redirect FIFO writes to the user-provided buffer
  aurora::gx::fifo::begin_display_list(static_cast<u8*>(list), size);
}

u32 GXEndDisplayList() {
  // Flush any pending dirty state into the display list
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }

  // End FIFO redirection and get the byte count (ROUNDUP32)
  u32 bytesWritten = aurora::gx::fifo::end_display_list();

  // Restore saved shadow register state
  if (__gx->dlSaveContext != 0) {
    std::memcpy(__gx, &sSavedGXData, sizeof(*__gx));
  }

  __gx->inDispList = 0;
  if (++sDlCacheGen == 0) {
    sDlCacheGen = 1;
  }
  __gx_shadow_invalidate();

  return bytesWritten;
}

void GXCallDisplayList(const void* data, u32 nbytes) {
  // Flush any pending dirty state before calling
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }

  // Flush pending primitives
  if (*reinterpret_cast<u32*>(&__gx->vNum) != 0) {
    __GXSendFlushPrim();
  }

  // Write display list contents to the FIFO
  aurora::gx::fifo::write_data(data, nbytes);
  if (!aurora::gx::fifo::in_display_list() && !__gx_dl_is_pure_prim(data, nbytes)) {
    __gx_shadow_invalidate();
  }

  // TEMP: debugging aid
  // aurora::gx::fifo::drain();
  // aurora::gx::fifo::process(static_cast<const u8*>(data), nbytes, true);
}

void GXCallStaticDisplayList(const void* data, u32 nbytes) {
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }
  if (*reinterpret_cast<u32*>(&__gx->vNum) != 0) {
    __GXSendFlushPrim();
  }
  if (!aurora::gx::fifo::in_display_list() && data != nullptr && nbytes >= 32 &&
      __gx_dl_is_pure_prim(data, nbytes)) {
    GX_WRITE_AURORA(GX_AURORA_CALL_STATIC_DL);
    GX_WRITE_U64(reinterpret_cast<u64>(reinterpret_cast<uintptr_t>(data)));
    GX_WRITE_U32(nbytes);
    GX_WRITE_U32(sDlCacheGen);
    return;
  }
  aurora::gx::fifo::write_data(data, nbytes);
  if (!aurora::gx::fifo::in_display_list() && !__gx_dl_is_pure_prim(data, nbytes)) {
    __gx_shadow_invalidate();
  }
}

void GXCallStaticDisplayListBatch(const void* batchKey, const AuroraStaticDlDesc* lists, u32 count) {
  if (lists == nullptr || count == 0) {
    return;
  }
  if (count == 1 || batchKey == nullptr || aurora::gx::fifo::in_display_list()) {
    for (u32 i = 0; i < count; ++i) {
      GXCallStaticDisplayList(lists[i].data, lists[i].nbytes);
    }
    return;
  }
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }
  if (*reinterpret_cast<u32*>(&__gx->vNum) != 0) {
    __GXSendFlushPrim();
  }
  const u64 layoutHash = current_dl_layout_hash();
  const uintptr_t bk = reinterpret_cast<uintptr_t>(batchKey);
  const u32 setIdx = static_cast<u32>((bk ^ (bk >> 11) ^ count ^ (layoutHash >> 20)) & 1023u);
  DlBatchPureCacheEntry* ways = s_dlBatchPureCache[setIdx];
  const void* const firstPtr = lists[0].data;
  const void* const lastPtr = lists[count - 1].data;
  bool found = false;
  bool pure = false;
  for (int w = 0; w < 2; ++w) {
    if (ways[w].batchKey == batchKey && ways[w].count == count && ways[w].dlGen == sDlCacheGen &&
        ways[w].layoutHash == layoutHash && ways[w].firstPtr == firstPtr && ways[w].lastPtr == lastPtr) {
      pure = ways[w].pure;
      found = true;
      break;
    }
  }
  if (!found) {
    pure = true;
    uintptr_t minAddr = UINTPTR_MAX;
    uintptr_t maxEndAddr = 0;
    for (u32 i = 0; i < count; ++i) {
      const void* d = lists[i].data;
      const u32 sz = lists[i].nbytes;
      if (sz > 0) {
        if (d == nullptr || !__gx_dl_is_pure_prim(d, sz)) {
          pure = false;
          break;
        }
        const uintptr_t p = reinterpret_cast<uintptr_t>(d);
        if (p < minAddr) minAddr = p;
        if (p + sz > maxEndAddr) maxEndAddr = p + sz;
      }
    }
    ways[1] = ways[0];
    ways[0] = {batchKey,
               firstPtr,
               lastPtr,
               (minAddr != UINTPTR_MAX) ? reinterpret_cast<const void*>(minAddr) : nullptr,
               (maxEndAddr != 0) ? reinterpret_cast<const void*>(maxEndAddr) : nullptr,
               count,
               sDlCacheGen,
               layoutHash,
               pure};
  }
  if (!pure) {
    for (u32 i = 0; i < count; ++i) {
      GXCallStaticDisplayList(lists[i].data, lists[i].nbytes);
    }
    return;
  }
  GX_WRITE_AURORA(GX_AURORA_CALL_STATIC_DL_BATCH);
  GX_WRITE_U64(static_cast<u64>(bk));
  GX_WRITE_U32(count);
  GX_WRITE_U32(sDlCacheGen);
  for (u32 i = 0; i < count; ++i) {
    GX_WRITE_U64(static_cast<u64>(reinterpret_cast<uintptr_t>(lists[i].data)));
    GX_WRITE_U32(lists[i].nbytes);
  }
}

}
