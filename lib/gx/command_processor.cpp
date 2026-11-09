#include "command_processor.hpp"

#include "../gfx/common.hpp"
#include "../gfx/depth_peek.hpp"
#include "dolphin/gx/GXAurora.h"
#include "gx.hpp"
#include "gx_fmt.hpp"
#include "pipeline.hpp"
#include "shader_info.hpp"
#include "../internal.hpp"
#include <aurora/paper.h>

#include <absl/container/flat_hash_map.h>
#include <limits>
#include <tracy/Tracy.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <set>
#include <vector>

namespace aurora::gx::fifo {
static Module Log("aurora::gx::fifo");

// Bumped by GX_CMD_INVL_VC (GXInvalidateVtxCache); invalidates the memoized attribute
// array content hashes the native geometry cache keys indexed draws by.
static u32 s_gxArrayInvalidateGen = 0;

static u16 prepare_idx_buffer(ByteBuffer& buf, GXPrimitive prim, u16 vtxStart, u16 vtxCount) {
  u16 numIndices = 0;
  if (prim == GX_QUADS) {
    buf.reserve_extra((vtxCount / 4) * 6 * sizeof(u16));

    for (u16 v = 0; v < vtxCount; v += 4) {
      u16 idx0 = vtxStart + v;
      u16 idx1 = vtxStart + v + 1;
      u16 idx2 = vtxStart + v + 2;
      u16 idx3 = vtxStart + v + 3;

      buf.append(idx0);
      buf.append(idx1);
      buf.append(idx2);
      numIndices += 3;

      buf.append(idx2);
      buf.append(idx3);
      buf.append(idx0);
      numIndices += 3;
    }
  } else if (prim == GX_TRIANGLES) {
    buf.reserve_extra(vtxCount * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      buf.append(idx);
      ++numIndices;
    }
  } else if (prim == GX_TRIANGLEFAN) {
    buf.reserve_extra(((u32(vtxCount) - 3) * 3 + 3) * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      if (v < 3) {
        buf.append(idx);
        ++numIndices;
        continue;
      }
      buf.append(std::array{vtxStart, static_cast<u16>(idx - 1), idx});
      numIndices += 3;
    }
  } else if (prim == GX_TRIANGLESTRIP) {
    buf.reserve_extra(((static_cast<u32>(vtxCount) - 3) * 3 + 3) * sizeof(u16));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      if (v < 3) {
        buf.append(idx);
        ++numIndices;
        continue;
      }
      if ((v & 1) == 0) {
        buf.append(std::array{static_cast<u16>(idx - 2), static_cast<u16>(idx - 1), idx});
      } else {
        buf.append(std::array{static_cast<u16>(idx - 1), static_cast<u16>(idx - 2), idx});
      }
      numIndices += 3;
    }
  } else if (prim == GX_LINES || prim == GX_LINESTRIP || prim == GX_POINTS) {
    buf.reserve_extra(6 * sizeof(u16));
    buf.append<u16>(0);
    buf.append<u16>(1);
    buf.append<u16>(3);
    buf.append<u16>(3);
    buf.append<u16>(2);
    buf.append<u16>(0);
    numIndices = 6;
  } else
    UNLIKELY FATAL("unsupported primitive type {}", static_cast<u32>(prim));
  return numIndices;
}

// Emit index data that keeps a triangle strip AS a strip on the GPU (TriangleStrip
// topology), rather than unrolling it into an independent triangle list. `restart`
// prepends a 0xffff primitive-restart index so an earlier strip in the same index
// buffer ends before this one begins. Returns the index count written (incl. restart).
static u16 build_triangle_strip_topology_indices(ByteBuffer& buf, u16 vtxStart, u16 vtxCount, bool restart) {
  const u16 numIndices = static_cast<u16>(vtxCount + (restart ? 1 : 0));
  buf.reserve_extra(static_cast<size_t>(numIndices) * sizeof(u16));
  if (restart) {
    buf.append<u16>(0xffff);
  }
  for (u16 v = 0; v < vtxCount; ++v) {
    buf.append(static_cast<u16>(vtxStart + v));
  }
  return numIndices;
}

// Concatenate several strips into one index buffer with 0xffff restart markers between
// them, so a whole run of adjacent strip draws collapses into a single DrawIndexed.
static u32 build_triangle_strip_batch_topology_indices(ByteBuffer& buf, const std::vector<u16>& stripCounts) {
  u32 indexCount = 0;
  u16 vtxStart = 0;
  for (size_t i = 0; i < stripCounts.size(); ++i) {
    const bool restart = i != 0;
    indexCount += build_triangle_strip_topology_indices(buf, vtxStart, stripCounts[i], restart);
    vtxStart = static_cast<u16>(vtxStart + stripCounts[i]);
  }
  return indexCount;
}

// GX FIFO opcodes - use CP_ prefix to avoid clashing with GXCommandList.h macros
static constexpr u8 CP_CMD_NOP = GX_NOP;
static constexpr u8 CP_CMD_LOAD_CP_REG = GX_LOAD_CP_REG;
static constexpr u8 CP_CMD_LOAD_XF_REG = GX_LOAD_XF_REG;
static constexpr u8 CP_CMD_LOAD_INDX_A = GX_LOAD_INDX_A;
static constexpr u8 CP_CMD_LOAD_INDX_B = GX_LOAD_INDX_B;
static constexpr u8 CP_CMD_LOAD_INDX_C = GX_LOAD_INDX_C;
static constexpr u8 CP_CMD_LOAD_INDX_D = GX_LOAD_INDX_D;
static constexpr u8 CP_CMD_CALL_DL = GX_CMD_CALL_DL;
static constexpr u8 CP_CMD_INVAL_VTX = GX_CMD_INVL_VC;
static constexpr u8 CP_CMD_LOAD_BP_REG = GX_LOAD_BP_REG & GX_OPCODE_MASK;

// Primitive type mask
static constexpr u8 CP_OPCODE_MASK = GX_OPCODE_MASK;
static constexpr u8 CP_VAT_MASK = GX_VAT_MASK;

// Read helpers for big/little endian
#if _MSC_VER
template <typename T>
__forceinline // Yes, this was necessary.
    inline T unaligned_load(const T* ptr) {
  return *static_cast<const __unaligned T*>(ptr);
}
#else
template <typename T>
inline T unaligned_load(const T* ptr) {
  T copy;
  memcpy(&copy, ptr, sizeof(T));
  return copy;
}
#endif

static inline u16 read_u16(const u8* ptr, bool bigEndian) {
  const u16 val = unaligned_load(reinterpret_cast<const u16*>(ptr));
  if (bigEndian) {
    return bswap(val);
  }
  return val;
}

static inline u32 read_u32(const u8* ptr, bool bigEndian) {
  const u32 val = unaligned_load(reinterpret_cast<const u32*>(ptr));
  if (bigEndian) {
    return bswap(val);
  }
  return val;
}

static u32 bp_get(u32 reg, u32 size, u32 shift);

static GXPixelFmt decode_pixel_fmt(u32 peCtrl, u32 cmode1) {
  switch (bp_get(peCtrl, 3, 0)) {
  case 0:
    return GX_PF_RGB8_Z24;
  case 1:
    return GX_PF_RGBA6_Z24;
  case 2:
    return GX_PF_RGB565_Z16;
  case 3:
    return GX_PF_Z24;
  case 4:
    switch (bp_get(cmode1, 2, 9)) {
    case 0:
      return GX_PF_Y8;
    case 1:
      return GX_PF_U8;
    case 2:
      return GX_PF_V8;
    default:
      Log.warn("command_processor: unsupported cmode1 pixel subtype {}", bp_get(cmode1, 2, 9));
      return GX_PF_Y8;
    }
  case 5:
    return GX_PF_YUV420;
  default:
    Log.warn("command_processor: unsupported PE pixel format {}", bp_get(peCtrl, 3, 0));
    return GX_PF_RGB8_Z24;
  }
}

static inline u64 read_u64(const u8* ptr, bool bigEndian) {
  u64 loaded;
  // Unaligned-safe load
  memcpy(&loaded, ptr, sizeof(u64));

  if (bigEndian) {
    return bswap(loaded);
  }

  return loaded;
}

struct TexBpRegMapping {
  u8 texMapId;
  enum class Kind : uint8_t { Mode0, Mode1, Image0, Image1, Image2, Image3, Tlut } kind;
};

static std::optional<TexBpRegMapping> decode_tex_bp_reg(u32 regId) {
  constexpr std::array mode0Ids{0x80u, 0x81u, 0x82u, 0x83u, 0xA0u, 0xA1u, 0xA2u, 0xA3u};
  constexpr std::array mode1Ids{0x84u, 0x85u, 0x86u, 0x87u, 0xA4u, 0xA5u, 0xA6u, 0xA7u};
  constexpr std::array image0Ids{0x88u, 0x89u, 0x8Au, 0x8Bu, 0xA8u, 0xA9u, 0xAAu, 0xABu};
  constexpr std::array image1Ids{0x8Cu, 0x8Du, 0x8Eu, 0x8Fu, 0xACu, 0xADu, 0xAEu, 0xAFu};
  constexpr std::array image2Ids{0x90u, 0x91u, 0x92u, 0x93u, 0xB0u, 0xB1u, 0xB2u, 0xB3u};
  constexpr std::array image3Ids{0x94u, 0x95u, 0x96u, 0x97u, 0xB4u, 0xB5u, 0xB6u, 0xB7u};
  constexpr std::array tlutIds{0x98u, 0x99u, 0x9Au, 0x9Bu, 0xB8u, 0xB9u, 0xBAu, 0xBBu};

  for (u8 i = 0; i < MaxTextures; ++i) {
    if (regId == mode0Ids[i]) {
      return TexBpRegMapping{.texMapId = i, .kind = TexBpRegMapping::Kind::Mode0};
    }
    if (regId == mode1Ids[i]) {
      return TexBpRegMapping{.texMapId = i, .kind = TexBpRegMapping::Kind::Mode1};
    }
    if (regId == image0Ids[i]) {
      return TexBpRegMapping{.texMapId = i, .kind = TexBpRegMapping::Kind::Image0};
    }
    if (regId == image1Ids[i]) {
      return TexBpRegMapping{.texMapId = i, .kind = TexBpRegMapping::Kind::Image1};
    }
    if (regId == image2Ids[i]) {
      return TexBpRegMapping{.texMapId = i, .kind = TexBpRegMapping::Kind::Image2};
    }
    if (regId == image3Ids[i]) {
      return TexBpRegMapping{.texMapId = i, .kind = TexBpRegMapping::Kind::Image3};
    }
    if (regId == tlutIds[i]) {
      return TexBpRegMapping{.texMapId = i, .kind = TexBpRegMapping::Kind::Tlut};
    }
  }
  return std::nullopt;
}

// Helper to convert packed RGBA8 to Vec4<float>
static Vec4<float> unpack_color(u32 packed) {
  return {
      static_cast<float>(packed >> 24 & 0xFF) / 255.f,
      static_cast<float>(packed >> 16 & 0xFF) / 255.f,
      static_cast<float>(packed >> 8 & 0xFF) / 255.f,
      static_cast<float>(packed & 0xFF) / 255.f,
  };
}

static inline f32 read_f32(const u8* ptr, bool bigEndian) {
  u32 bits = read_u32(ptr, bigEndian);
  f32 val;
  std::memcpy(&val, &bits, sizeof(val));
  return val;
}

static bool copy_xf_data(u32 addr, const u8* data, u32 len, bool bigEndian) {
  if (addr < 0x78) {
    // Position matrices (0x0000 - 0x0077)
    u32 mtxIdx = addr / 12;
    u32 startOffset = addr % 12;
    // We only support full writes to matrices
    CHECK(mtxIdx < MaxPnMtx, "XF: PosMtx copy oob? Should never happen; mtxIdx={}", mtxIdx);
    CHECK(startOffset == 0 && len == 12, "XF: PosMtx sub-copy unsupported: offs={}, len={}", startOffset, len);
    auto& mtx = g_gxState.pnMtx[mtxIdx].pos;
    f32* flat = reinterpret_cast<f32*>(&mtx);
    bool changed = false;
    for (u32 i = 0; i < len; i++) {
      const f32 v = read_f32(data + i * 4, bigEndian);
      if (flat[i] != v) {
        flat[i] = v;
        changed = true;
      }
    }
    if (changed) {
      g_gxState.stateDirty = true;
    }
    return true;
  } else if (addr < 0x0F0) {
    // Texture matrices (0x078-0x0EF)
    u32 texBase = addr - 0x078;
    u32 mtxIdx = texBase / 12;
    u32 startOffset = texBase % 12;
    CHECK(mtxIdx < MaxTexMtx, "XF TexMtx copy oob? Should never happen; mtxIdx={}", mtxIdx);
    CHECK(startOffset == 0 && (len == 8 || len == 12), "XF TexMtx sub-copy unsupported: offs={}, len={}", startOffset,
          len);

    // Determine if 2x4 or 3x4 from count
    auto& mtx = g_gxState.texMtxs[mtxIdx];
    f32* flat = reinterpret_cast<f32*>(&mtx);
    bool changed = false;
    for (u32 i = 0; i < len; i++) {
      const f32 v = read_f32(data + i * 4, bigEndian);
      if (flat[i] != v) {
        flat[i] = v;
        changed = true;
      }
    }
    if (changed) {
      g_gxState.stateDirty = true;
    }
    return true;
  } else if (addr >= 0x400 && addr < 0x45A) {
    // Normal matrices (0x400-0x459)
    u32 nrmBase = addr - 0x400;
    u32 mtxIdx = nrmBase / 9;
    u32 startOffset = nrmBase % 9;
    // We only support full writes to matrices
    CHECK(mtxIdx < MaxPnMtx, "XF: NrmMtx copy oob? Should never happen; mtxIdx={}", mtxIdx);
    CHECK(startOffset == 0 && len == 9, "XF: NrmMtx sub-copy unsupported: offs={}, len={}", startOffset, len);
    auto& mtx = g_gxState.pnMtx[mtxIdx].nrm;
    f32* flat = reinterpret_cast<f32*>(&mtx);
    bool changed = false;
    static constexpr uint8_t s_nrmIdx[9] = {0, 1, 2, 4, 5, 6, 8, 9, 10};
    for (u32 i = 0; i < len; i++) {
      const f32 v = read_f32(data + i * 4, bigEndian);
      const u32 dstIdx = s_nrmIdx[i];
      if (flat[dstIdx] != v) {
        flat[dstIdx] = v;
        changed = true;
      }
    }
    if (changed) {
      g_gxState.stateDirty = true;
    }
    return true;
  } else if (addr >= 0x500 && addr < 0x5F0) {
    // Post-transform texture matrices (0x500-0x5EF)
    u32 ptBase = addr - 0x500;
    u32 mtxIdx = ptBase / 12;
    u32 startOffset = ptBase % 12;
    CHECK(mtxIdx < MaxPTTexMtx, "XF: PTTexMtx copy oob? Should never happen; mtxIdx={}", mtxIdx);
    CHECK(startOffset == 0 && len == 12, "XF: PTTexMtx sub-copy unsupported: offs={}, len={}", startOffset, len);
    auto& mtx = g_gxState.ptTexMtxs[mtxIdx];
    f32* flat = reinterpret_cast<f32*>(&mtx);
    for (u32 i = 0; i < len; i++) {
      flat[startOffset + i] = read_f32(data + i * 4, bigEndian);
    }
    g_gxState.stateDirty = true;
    return true;
  } else if (addr >= 0x600 && addr < 0x680) {
    // Lights (0x600-0x67F) - 8 lights, 16 values each
    u32 lightBase = addr - 0x600;
    u32 lightIdx = lightBase / 0x10;
    u32 startOffset = lightBase % 0x10;
    CHECK(lightIdx < 8, "XF: Light copy oob? Should never happen; lightIdx={}", lightIdx);
    CHECK(startOffset + len <= 0x10, "XF: Light copy that crosses across light boundaries unsupported: offs={}, len={}",
          startOffset, len);
    auto& light = g_gxState.lights[lightIdx];
    for (u32 i = 0; i < len; i++) {
      u32 field = startOffset + i;
      f32 val = read_f32(data + i * 4, bigEndian);
      u32 ival = read_u32(data + i * 4, bigEndian);
      switch (field) {
      case 3: // Color (packed u32)
        light.color = unpack_color(ival);
        break;
      case 4:
        light.cosAtt[0] = val;
        break; // a0
      case 5:
        light.cosAtt[1] = val;
        break; // a1
      case 6:
        light.cosAtt[2] = val;
        break; // a2
      case 7:
        light.distAtt[0] = val;
        break; // k0
      case 8:
        light.distAtt[1] = val;
        break; // k1
      case 9:
        light.distAtt[2] = val;
        break; // k2
      case 10:
        light.pos[0] = val;
        break; // px
      case 11:
        light.pos[1] = val;
        break; // py
      case 12:
        light.pos[2] = val;
        break; // pz
      case 13:
        light.dir[0] = val;
        break; // nx
      case 14:
        light.dir[1] = val;
        break; // ny
      case 15:
        light.dir[2] = val;
        break; // nz
      default:
        break; // padding (0-2)
      }
    }
    g_gxState.stateDirty = true;
    return true;
  }
  return false;
}

// Forward declarations for register handlers
static void handle_bp(u32 value, bool bigEndian);
static void handle_cp(u8 addr, u32 value, bool bigEndian);
static void handle_xf(const u8* data, u32& pos, u32 size, bool bigEndian, bool& pipelineChanged);
static void handle_draw(u8 cmd, const u8* data, u32& pos, u32 size, bool bigEndian);
static void handle_aurora(const u8* data, u32& pos, u32 size, bool bigEndian);

namespace {
static bool s_hasLastDrawState = false;
static bool s_hasLastUniformRange = false;
static PipelineConfig s_lastConfig{};
static ShaderConfig s_lastShaderConfig{};
static ShaderInfo s_lastInfo{};
static gfx::PipelineRef s_lastPipeline{};
static GXBindGroups s_lastBindGroups{};
static uint32_t s_lastTextureDirty = 0;
static uint32_t s_lastUniformDirty = 0;
static gfx::Range s_lastUniformRange{};

static u64 s_uniformCacheHits = 0;
static u64 s_uniformCacheMisses = 0;

struct DrainTiming {
  double ms_bp = 0.0;
  double ms_cp = 0.0;
  double ms_xf = 0.0;
  double ms_indx = 0.0;
  double ms_draw_total = 0.0;
  double ms_geom_hash = 0.0;
  double ms_geom_expand = 0.0;
  double ms_pipe_cfg = 0.0;
  double ms_pipe_ref = 0.0;
  double ms_uni_build = 0.0;
  double ms_push_cmd = 0.0;
  double ms_scan = 0.0;
  double ms_tex_resolve = 0.0;
  double ms_bg_build = 0.0;
  double ms_geom_miss = 0.0;
  double ms_total = 0.0;
  u32 frames = 0;
  u32 draw_count = 0;
  u32 bg_calls = 0;
};
static DrainTiming s_drainTiming;

struct CachedShaderDraw {
  uint64_t shaderHash = 0;
  ShaderInfo info{};
  GXBindGroups bindGroups{};
  gfx::Range uniformRange{};
  uint32_t uniformDirty = UINT32_MAX;
  uint32_t textureDirty = UINT32_MAX;
  uint32_t frame = UINT32_MAX;
};

static constexpr size_t kShaderDrawCacheSize = 512;
static std::array<CachedShaderDraw, kShaderDrawCacheSize> s_shaderDrawCache{};
static u32 s_vatGen = 1;
} // namespace

static void reset_frame_draw_cache() {
  s_hasLastDrawState = false;
  s_hasLastUniformRange = false;
  s_lastUniformRange = {};
  g_gxState.uniformDirty++;
  g_gxState.pipelineDirty++;
}

static inline bool bp_affects_pipeline(u32 regId) noexcept {
  if (regId == 0x00) return true; // genMode (cullMode, numTexGens, numChans, numTevStages, numIndStages)
  if (regId >= 0x10 && regId <= 0x1F) return true; // TEV indirect stages
  if (regId == 0x25 || regId == 0x26) return true; // Ind tex scale
  if (regId >= 0x40 && regId <= 0x43) return true; // Z mode, alpha compare, blend mode, clamp
  if (regId >= 0xC0 && regId <= 0xDF) return true; // TEV stages color/alpha combiners
  if (regId >= 0xF6 && regId <= 0xFD) return true; // TEV swap table
  return false;
}

static inline bool bp_affects_uniforms(u32 regId) noexcept {
  // Indirect texture matrices (0x06..0x0E)
  if (regId >= 0x06 && regId <= 0x0E) return true;
  // Line / point size (0x22)
  if (regId == 0x22) return true;
  // SU texture coordinate scale (0x30..0x3F)
  if (regId >= 0x30 && regId <= 0x3F) return true;
  // TEV color / K color registers (0xE0..0xE7)
  if (regId >= 0xE0 && regId <= 0xE7) return true;
  // Fog parameters (0xEE..0xF2)
  if (regId >= 0xEE && regId <= 0xF2) return true;
  return false;
}

void process(const u8* data, u32 size, bool bigEndian) {
  ZoneScoped;
  const auto t_proc_start = std::chrono::steady_clock::now();
  reset_frame_draw_cache();
  u32 pos = 0;

  while (pos < size) {
    u8 cmd = data[pos++];
    u8 opcode = cmd & CP_OPCODE_MASK;
    // Log.warn("Processing opcode {:02x} at pos {} (size {})", opcode, pos - 1, size);

    switch (opcode) {
    case CP_CMD_NOP:
      continue;

    case CP_CMD_LOAD_BP_REG: {
      CHECK(pos + 4 <= size, "BP reg read overrun");
      u32 value = read_u32(data + pos, bigEndian);
      pos += 4;
      const u32 regId = (value >> 24) & 0xFF;
      if (regId != 0xFE && regId != 0x52 && g_gxState.bpRegCache[0xFE] == 0x00FFFFFF && g_gxState.bpRegCache[regId] == value) {
        continue;
      }
      const auto t0 = std::chrono::steady_clock::now();
      handle_bp(value, bigEndian);
      const auto t1 = std::chrono::steady_clock::now();
      s_drainTiming.ms_bp += std::chrono::duration<double, std::milli>(t1 - t0).count();
      if (bp_affects_uniforms(regId)) {
        g_gxState.uniformDirty++;
      }
      if (bp_affects_pipeline(regId)) {
        g_gxState.pipelineDirty++;
      }
      break;
    }

    case CP_CMD_LOAD_CP_REG: {
      CHECK(pos + 5 <= size, "CP reg read overrun");
      u8 addr = data[pos++];
      u32 value = read_u32(data + pos, bigEndian);
      pos += 4;
      const auto t0 = std::chrono::steady_clock::now();
      handle_cp(addr, value, bigEndian);
      const auto t1 = std::chrono::steady_clock::now();
      s_drainTiming.ms_cp += std::chrono::duration<double, std::milli>(t1 - t0).count();
      g_gxState.pipelineDirty++;
      s_vatGen++;
      // CP registers set vertex attribute tables (VCD/VAT) and array bases/strides.
      // These affect PipelineConfig vertex attributes, NOT uniform buffer contents.
      break;
    }

    case CP_CMD_LOAD_XF_REG: {
      const bool prevDirty = g_gxState.stateDirty;
      g_gxState.stateDirty = false;
      const auto t0 = std::chrono::steady_clock::now();
      bool pipelineChanged = false;
      handle_xf(data, pos, size, bigEndian, pipelineChanged);
      const auto t1 = std::chrono::steady_clock::now();
      s_drainTiming.ms_xf += std::chrono::duration<double, std::milli>(t1 - t0).count();
      if (g_gxState.stateDirty) {
        g_gxState.uniformDirty++;
      }
      if (pipelineChanged) {
        g_gxState.pipelineDirty++;
      }
      g_gxState.stateDirty |= prevDirty;
      break;
    }

    case CP_CMD_LOAD_INDX_A:
    case CP_CMD_LOAD_INDX_B:
    case CP_CMD_LOAD_INDX_C:
    case CP_CMD_LOAD_INDX_D: {
      ZoneScopedN("LOAD_INDX");
      const auto t0 = std::chrono::steady_clock::now();
      // Indexed XF load: 4 bytes of data
      CHECK(pos + 4 <= size, "indexed XF read overrun");
      u32 arrayType = GX_POS_MTX_ARRAY + (opcode - (CP_CMD_LOAD_INDX_A / 0x08));
      u8 srcArrayIdx = data[pos++];
      auto const& array = g_gxState.arrays[arrayType];
      u8* srcData = ((u8*)array.data) + srcArrayIdx * array.stride;
      u16 addrLen = read_u16(data + pos, bigEndian);
      u16 len = (addrLen >> 12) + 1;
      u16 dstAddr = addrLen & 0x0FFF;
      const bool prevDirty = g_gxState.stateDirty;
      g_gxState.stateDirty = false;
      if (!copy_xf_data(dstAddr, srcData, len, bigEndian)) {
#ifndef NDEBUG
        Log.debug("Unimplemented indexed XF load (opcode 0x{:02X}, dstAddr=%04x)", opcode, dstAddr);
#endif
      }
      if (g_gxState.stateDirty) {
        g_gxState.uniformDirty++;
      }
      g_gxState.stateDirty |= prevDirty;
      pos += 4;
      const auto t1 = std::chrono::steady_clock::now();
      s_drainTiming.ms_indx += std::chrono::duration<double, std::milli>(t1 - t0).count();
      break;
    }

    case CP_CMD_CALL_DL: {
      // Call display list: 8 bytes (address + size)
      CHECK(pos + 8 <= size, "call DL read overrun");
      Log.warn("Ignoring nested GX_CMD_CALL_DL");
      pos += 8;
      break;
    }

    case CP_CMD_INVAL_VTX: {
      // Invalidate vertex cache: the game may have rewritten attribute array contents in
      // place (e.g. particle systems), so drop memoized array content hashes.
      ++s_gxArrayInvalidateGen;
      break;
    }

    case GX_AURORA: {
      handle_aurora(data, pos, size, bigEndian);
      g_gxState.uniformDirty++;
      break;
    }

    // Draw commands: 0x80-0xBF
    case GX_DRAW_QUADS:
    case GX_DRAW_TRIANGLES:
    case GX_DRAW_TRIANGLE_STRIP:
    case GX_DRAW_TRIANGLE_FAN:
    case GX_DRAW_LINES:
    case GX_DRAW_LINE_STRIP:
    case GX_DRAW_POINTS: {
      const auto td0 = std::chrono::steady_clock::now();
      ++s_drainTiming.draw_count;
      handle_draw(cmd, data, pos, size, bigEndian);
      const auto td1 = std::chrono::steady_clock::now();
      s_drainTiming.ms_draw_total += std::chrono::duration<double, std::milli>(td1 - td0).count();
      break;
    }

    default:
      // Check if it's a draw command (0x80-0xBF range)
      if (cmd >= 0x80) {
        const auto td0 = std::chrono::steady_clock::now();
        ++s_drainTiming.draw_count;
        handle_draw(cmd, data, pos, size, bigEndian);
        const auto td1 = std::chrono::steady_clock::now();
        s_drainTiming.ms_draw_total += std::chrono::duration<double, std::milli>(td1 - td0).count();
      } else {
        // Hex dump surrounding bytes for debugging
        {
          u32 dumpStart = (pos > 17) ? pos - 17 : 0;
          u32 dumpEnd = (pos + 16 < size) ? pos + 16 : size;
          std::string hex;
          for (u32 i = dumpStart; i < dumpEnd; i++) {
            if (i == pos - 1)
              hex += fmt::format("[{:02x}]", data[i]);
            else
              hex += fmt::format(" {:02x}", data[i]);
          }
          Log.error("  hex dump (pos {}-{}):{}", dumpStart, dumpEnd - 1, hex);
        }
        FATAL("command_processor: unknown opcode 0x{:02X} at pos {}", cmd, pos - 1);
      }
      break;
    }
  }

  const auto t_proc_end = std::chrono::steady_clock::now();
  s_drainTiming.ms_total += std::chrono::duration<double, std::milli>(t_proc_end - t_proc_start).count();
  if (++s_drainTiming.frames >= 30) {
    const double f = 1.0 / s_drainTiming.frames;
    std::printf("[DRAIN-TIMING-30f] avg: total %.1fms | draw_tot %.1fms (draws:%.0f hash:%.1f exp:%.1f g_miss:%.1f scan:%.1f pipe_cfg:%.1f pipe_ref:%.1f res:%.1f bg:%.1f[calls:%.0f] uni:%.1f push:%.1f) | xf %.1fms | indx %.1fms | bp %.1fms | cp %.1fms\n",
                s_drainTiming.ms_total * f,
                s_drainTiming.ms_draw_total * f,
                s_drainTiming.draw_count * f,
                s_drainTiming.ms_geom_hash * f,
                s_drainTiming.ms_geom_expand * f,
                s_drainTiming.ms_geom_miss * f,
                s_drainTiming.ms_scan * f,
                s_drainTiming.ms_pipe_cfg * f,
                s_drainTiming.ms_pipe_ref * f,
                s_drainTiming.ms_tex_resolve * f,
                s_drainTiming.ms_bg_build * f,
                s_drainTiming.bg_calls * f,
                s_drainTiming.ms_uni_build * f,
                s_drainTiming.ms_push_cmd * f,
                s_drainTiming.ms_xf * f,
                s_drainTiming.ms_indx * f,
                s_drainTiming.ms_bp * f,
                s_drainTiming.ms_cp * f);
    std::fflush(stdout);
    s_drainTiming = DrainTiming{};
  }
}

// Helper to extract bit fields from a 32-bit register
inline static u32 bp_get(u32 reg, u32 size, u32 shift) { return reg >> shift & (1u << size) - 1; }

// BP register handler - decodes BP (RAS/pixel engine) register writes and updates g_gxState
static void handle_bp(u32 value, bool bigEndian) {
  u32 regId = (value >> 24) & 0xFF;
  // Mask off the register ID from the value for field extraction
  // (the regId is stored in bits 24-31, data is in bits 0-23)

  if (regId == 0xFE) {
    g_gxState.bpRegCache[regId] = value & 0x00FFFFFF;
    return;
  } else {
    u32 ssMask = g_gxState.bpRegCache[0xFE];
    g_gxState.bpRegCache[0xFE] = 0x00FFFFFF;
    const u32 merged = (g_gxState.bpRegCache[regId] & ~ssMask) | (value & ssMask);
    value = (regId << 24) | (merged & 0x00FFFFFF);
    if (g_gxState.bpRegCache[regId] == value && regId != 0x52)
      return;
    g_gxState.bpRegCache[regId] = value;
  }

  // TEV color combiner stages (0xC0, 0xC2, 0xC4, ... 0xDE)
  if (regId >= 0xC0 && regId <= 0xDE && (regId & 1) == 0) {
    u32 stage = (regId - 0xC0) / 2;
    if (stage < MaxTevStages) {
      auto& s = g_gxState.tevStages[stage];
      s.colorPass.d = static_cast<GXTevColorArg>(bp_get(value, 4, 0));
      s.colorPass.c = static_cast<GXTevColorArg>(bp_get(value, 4, 4));
      s.colorPass.b = static_cast<GXTevColorArg>(bp_get(value, 4, 8));
      s.colorPass.a = static_cast<GXTevColorArg>(bp_get(value, 4, 12));
      s.colorOp.clamp = bp_get(value, 1, 19) != 0;
      s.colorOp.outReg = static_cast<GXTevRegID>(bp_get(value, 2, 22));
      if (bp_get(value, 2, 16) == 3) {
        // Bias==3 means compare mode: reconstruct GXTevOp enum (8 + 3-bit hw value)
        u32 hwOp = bp_get(value, 1, 18) | (bp_get(value, 2, 20) << 1);
        s.colorOp.op = static_cast<GXTevOp>(hwOp + 8);
        s.colorOp.bias = GX_TB_ZERO;
        s.colorOp.scale = GX_CS_SCALE_1;
      } else {
        // Normal mode: bit18 is op (0=ADD, 1=SUB), bits16-17 is bias, bits20-21 is scale
        s.colorOp.op = static_cast<GXTevOp>(bp_get(value, 1, 18));
        s.colorOp.bias = static_cast<GXTevBias>(bp_get(value, 2, 16));
        s.colorOp.scale = static_cast<GXTevScale>(bp_get(value, 2, 20));
      }
      g_gxState.stateDirty = true;
    }
    return;
  }

  // TEV alpha combiner stages (0xC1, 0xC3, 0xC5, ... 0xDF)
  if (regId >= 0xC1 && regId <= 0xDF && (regId & 1) == 1) {
    u32 stage = (regId - 0xC1) / 2;
    if (stage < MaxTevStages) {
      auto& s = g_gxState.tevStages[stage];
      s.tevSwapRas = static_cast<GXTevSwapSel>(bp_get(value, 2, 0));
      s.tevSwapTex = static_cast<GXTevSwapSel>(bp_get(value, 2, 2));
      s.alphaPass.d = static_cast<GXTevAlphaArg>(bp_get(value, 3, 4));
      s.alphaPass.c = static_cast<GXTevAlphaArg>(bp_get(value, 3, 7));
      s.alphaPass.b = static_cast<GXTevAlphaArg>(bp_get(value, 3, 10));
      s.alphaPass.a = static_cast<GXTevAlphaArg>(bp_get(value, 3, 13));
      s.alphaOp.clamp = bp_get(value, 1, 19) != 0;
      s.alphaOp.outReg = static_cast<GXTevRegID>(bp_get(value, 2, 22));
      if (bp_get(value, 2, 16) == 3) {
        u32 hwOp = bp_get(value, 1, 18) | (bp_get(value, 2, 20) << 1);
        s.alphaOp.op = static_cast<GXTevOp>(hwOp + 8);
        s.alphaOp.bias = GX_TB_ZERO;
        s.alphaOp.scale = GX_CS_SCALE_1;
      } else {
        s.alphaOp.op = static_cast<GXTevOp>(bp_get(value, 1, 18));
        s.alphaOp.bias = static_cast<GXTevBias>(bp_get(value, 2, 16));
        s.alphaOp.scale = static_cast<GXTevScale>(bp_get(value, 2, 20));
      }
      g_gxState.stateDirty = true;
    }
    return;
  }

  switch (regId) {
  // genMode (0x00)
  case 0x00: {
    g_gxState.numTexGens = bp_get(value, 4, 0);
    g_gxState.numChans = bp_get(value, 3, 4);
    g_gxState.numTevStages = bp_get(value, 4, 10) + 1;
    u32 hwCull = bp_get(value, 2, 14);
    // Swap front/back to match GX convention
    switch (hwCull) {
    case GX_CULL_FRONT:
      g_gxState.cullMode = GX_CULL_BACK;
      break;
    case GX_CULL_BACK:
      g_gxState.cullMode = GX_CULL_FRONT;
      break;
    default:
      g_gxState.cullMode = static_cast<GXCullMode>(hwCull);
      break;
    }
    g_gxState.numIndStages = bp_get(value, 3, 16);
    g_gxState.stateDirty = true;
    break;
  }

  // BP mask (0x0F) - internal, applies to next BP write
  case 0x0F:
#ifndef NDEBUG
    Log.debug("BP mask set to {:06x}, but selective updates are not implemented", value & 0xFFFFFF);
#endif
    break;

  // TEV indirect stages (0x10-0x1F)
  case 0x10:
  case 0x11:
  case 0x12:
  case 0x13:
  case 0x14:
  case 0x15:
  case 0x16:
  case 0x17:
  case 0x18:
  case 0x19:
  case 0x1A:
  case 0x1B:
  case 0x1C:
  case 0x1D:
  case 0x1E:
  case 0x1F: {
    u32 stage = regId - 0x10;
    if (stage < MaxTevStages) {
      auto& s = g_gxState.tevStages[stage];
      s.indTexStage = static_cast<GXIndTexStageID>(bp_get(value, 2, 0));
      s.indTexFormat = static_cast<GXIndTexFormat>(bp_get(value, 2, 2));
      s.indTexBiasSel = static_cast<GXIndTexBiasSel>(bp_get(value, 3, 4));
      s.indTexAlphaSel = static_cast<GXIndTexAlphaSel>(bp_get(value, 2, 7));
      s.indTexMtxId = static_cast<GXIndTexMtxID>(bp_get(value, 4, 9));
      s.indTexWrapS = static_cast<GXIndTexWrap>(bp_get(value, 3, 13));
      s.indTexWrapT = static_cast<GXIndTexWrap>(bp_get(value, 3, 16));
      s.indTexUseOrigLOD = bp_get(value, 1, 19) != 0;
      s.indTexAddPrev = bp_get(value, 1, 20) != 0;
      g_gxState.stateDirty = true;
    }
    break;
  }

  // Scissor registers (0x20, 0x21)
  case 0x20:
  case 0x21: {
    const u32 scis0 = g_gxState.bpRegCache[0x20];
    const u32 scis1 = g_gxState.bpRegCache[0x21];
    const int32_t tp = static_cast<int32_t>(bp_get(scis0, 11, 0)) - 342;
    const int32_t lf = static_cast<int32_t>(bp_get(scis0, 11, 12)) - 342;
    const int32_t bm = static_cast<int32_t>(bp_get(scis1, 11, 0)) - 342;
    const int32_t rt = static_cast<int32_t>(bp_get(scis1, 11, 12)) - 342;
    const int32_t wd = std::max(rt - lf + 1, 0);
    const int32_t ht = std::max(bm - tp + 1, 0);
    set_logical_scissor({lf, tp, wd, ht});
    break;
  }

  // Line/point size (0x22)
  case 0x22: {
    g_gxState.lineWidth = static_cast<u8>(bp_get(value, 8, 0));
    g_gxState.pointSize = static_cast<u8>(bp_get(value, 8, 8));
    g_gxState.lineTexOffset = static_cast<GXTexOffset>(bp_get(value, 3, 16));
    g_gxState.pointTexOffset = static_cast<GXTexOffset>(bp_get(value, 3, 19));
    g_gxState.lineHalfAspect = bp_get(value, 1, 22) != 0;
    g_gxState.stateDirty = true;
    break;
  }

  // Indirect texture scale (0x25, 0x26)
  case 0x25: {
    if (MaxIndStages > 0) {
      g_gxState.indStages[0].scaleS = static_cast<GXIndTexScale>(bp_get(value, 4, 0));
      g_gxState.indStages[0].scaleT = static_cast<GXIndTexScale>(bp_get(value, 4, 4));
    }
    if (MaxIndStages > 1) {
      g_gxState.indStages[1].scaleS = static_cast<GXIndTexScale>(bp_get(value, 4, 8));
      g_gxState.indStages[1].scaleT = static_cast<GXIndTexScale>(bp_get(value, 4, 12));
    }
    g_gxState.stateDirty = true;
    break;
  }
  case 0x26: {
    if (MaxIndStages > 2) {
      g_gxState.indStages[2].scaleS = static_cast<GXIndTexScale>(bp_get(value, 4, 0));
      g_gxState.indStages[2].scaleT = static_cast<GXIndTexScale>(bp_get(value, 4, 4));
    }
    if (MaxIndStages > 3) {
      g_gxState.indStages[3].scaleS = static_cast<GXIndTexScale>(bp_get(value, 4, 8));
      g_gxState.indStages[3].scaleT = static_cast<GXIndTexScale>(bp_get(value, 4, 12));
    }
    g_gxState.stateDirty = true;
    break;
  }

  // Indirect texture reference (0x27)
  case 0x27: {
    for (u32 i = 0; i < MaxIndStages; i++) {
      g_gxState.indStages[i].texMapId = static_cast<GXTexMapID>(bp_get(value, 3, i * 6));
      g_gxState.indStages[i].texCoordId = static_cast<GXTexCoordID>(bp_get(value, 3, i * 6 + 3));
    }
    g_gxState.stateDirty = true;
    break;
  }

  // TEV order / tref (0x28-0x2F) - 2 stages per register
  case 0x28:
  case 0x29:
  case 0x2A:
  case 0x2B:
  case 0x2C:
  case 0x2D:
  case 0x2E:
  case 0x2F: {
    u32 idx = regId - 0x28;
    u32 stage0 = idx * 2;
    u32 stage1 = idx * 2 + 1;

    // Channel ID reverse mapping from hardware to GX
    static const GXChannelID r2c[] = {GX_COLOR0A0, GX_COLOR1A1,   GX_COLOR0A0,    GX_COLOR1A1,
                                      GX_COLOR0A0, GX_ALPHA_BUMP, GX_ALPHA_BUMPN, GX_COLOR_ZERO};

    if (stage0 < MaxTevStages) {
      auto& s = g_gxState.tevStages[stage0];
      s.texMapId = static_cast<GXTexMapID>(bp_get(value, 3, 0));
      s.texCoordId = static_cast<GXTexCoordID>(bp_get(value, 3, 3));
      // bit 6 = tex enable
      if (!bp_get(value, 1, 6)) {
        s.texMapId = GX_TEXMAP_NULL;
      }
      u32 chanHw = bp_get(value, 3, 7);
      s.channelId = (chanHw < 8) ? r2c[chanHw] : GX_COLOR_NULL;
    }
    if (stage1 < MaxTevStages) {
      auto& s = g_gxState.tevStages[stage1];
      s.texMapId = static_cast<GXTexMapID>(bp_get(value, 3, 12));
      s.texCoordId = static_cast<GXTexCoordID>(bp_get(value, 3, 15));
      if (!bp_get(value, 1, 18)) {
        s.texMapId = GX_TEXMAP_NULL;
      }
      u32 chanHw = bp_get(value, 3, 19);
      s.channelId = (chanHw < 8) ? r2c[chanHw] : GX_COLOR_NULL;
    }
    g_gxState.stateDirty = true;
    break;
  }

  // Z mode (0x40)
  case 0x40: {
    g_gxState.depthCompare = bp_get(value, 1, 0) != 0;
    g_gxState.depthFunc = static_cast<GXCompare>(bp_get(value, 3, 1));
    g_gxState.depthUpdate = bp_get(value, 1, 4) != 0;
    g_gxState.stateDirty = true;
    break;
  }

  // Blend mode / cmode0 (0x41)
  case 0x41: {
    bool blendEn = bp_get(value, 1, 0) != 0;
    bool logicEn = bp_get(value, 1, 1) != 0;
    bool dither = bp_get(value, 1, 2) != 0;
    g_gxState.colorUpdate = bp_get(value, 1, 3) != 0;
    g_gxState.alphaUpdate = bp_get(value, 1, 4) != 0;
    g_gxState.blendFacDst = static_cast<GXBlendFactor>(bp_get(value, 3, 5));
    g_gxState.blendFacSrc = static_cast<GXBlendFactor>(bp_get(value, 3, 8));
    bool subtract = bp_get(value, 1, 11) != 0;
    g_gxState.blendOp = static_cast<GXLogicOp>(bp_get(value, 4, 12));

    if (subtract) {
      g_gxState.blendMode = GX_BM_SUBTRACT;
    } else if (blendEn) {
      g_gxState.blendMode = GX_BM_BLEND;
    } else if (logicEn) {
      g_gxState.blendMode = GX_BM_LOGIC;
    } else {
      g_gxState.blendMode = GX_BM_NONE;
    }
    g_gxState.stateDirty = true;
    break;
  }

  // Dst alpha / cmode1 (0x42)
  case 0x42: {
    u8 alpha = bp_get(value, 8, 0);
    bool enabled = bp_get(value, 1, 8) != 0;
    g_gxState.dstAlpha = enabled ? alpha : UINT32_MAX;
    g_gxState.pixelFmt = decode_pixel_fmt(g_gxState.bpRegCache[0x43], value);
    g_gxState.stateDirty = true;
    break;
  }

  // PE control (0x43) - pixel format, z format, zcomp location
  case 0x43: {
    g_gxState.pixelFmt = decode_pixel_fmt(value, g_gxState.bpRegCache[0x42]);
    g_gxState.zFmt = static_cast<GXZFmt16>(bp_get(value, 3, 3));
    g_gxState.zCompLocBeforeTex = bp_get(value, 1, 6) != 0;
    g_gxState.stateDirty = true;
    break;
  }

  // Texture copy
  case 0x52: {
    const bool clear = bp_get(value, 1, 11) != 0;
    if (bp_get(value, 1, 14) != 0) {
      Log.warn("STUB: display copy is not implemented");
    } else {
      copy_tex(g_gxState.texCopyDest, clear);
      g_gxState.textureDirty++;
      g_gxState.stateDirty = true;
    }
    break;
  }

  // TLUT load address / execute (0x64, 0x65)
  case 0x64:
    break;
  case 0x65: {
    const auto idx = bp_get(value, 10, 0);
    if (idx < MaxTluts) {
      auto& slot = g_gxState.loadedTluts[idx];
      slot.loadTlut0 = g_gxState.bpRegCache[0x64];
      slot.numEntries = static_cast<u16>(bp_get(value, 10, 10) + 1);
    }
    break;
  }

  // Alpha compare (0xF3)
  case 0xF3: {
    g_gxState.alphaCompare.ref0 = bp_get(value, 8, 0);
    g_gxState.alphaCompare.ref1 = bp_get(value, 8, 8);
    g_gxState.alphaCompare.comp0 = static_cast<GXCompare>(bp_get(value, 3, 16));
    g_gxState.alphaCompare.comp1 = static_cast<GXCompare>(bp_get(value, 3, 19));
    g_gxState.alphaCompare.op = static_cast<GXAlphaOp>(bp_get(value, 2, 22));
    g_gxState.stateDirty = true;
    break;
  }

  // TEV K color/alpha select (0xF6-0xFD)
  case 0xF6:
  case 0xF7:
  case 0xF8:
  case 0xF9:
  case 0xFA:
  case 0xFB:
  case 0xFC:
  case 0xFD: {
    u32 kselIdx = regId - 0xF6;
    // Swap table entries (packed into pairs of ksel registers)
    if (kselIdx < MaxTevSwap * 2) {
      u32 swapIdx = kselIdx / 2;
      if (kselIdx & 1) {
        g_gxState.tevSwapTable[swapIdx].blue = static_cast<GXTevColorChan>(bp_get(value, 2, 0));
        g_gxState.tevSwapTable[swapIdx].alpha = static_cast<GXTevColorChan>(bp_get(value, 2, 2));
      } else {
        g_gxState.tevSwapTable[swapIdx].red = static_cast<GXTevColorChan>(bp_get(value, 2, 0));
        g_gxState.tevSwapTable[swapIdx].green = static_cast<GXTevColorChan>(bp_get(value, 2, 2));
      }
    }
    // K color/alpha selection for 2 stages per register
    u32 stage0 = kselIdx * 2;
    u32 stage1 = kselIdx * 2 + 1;
    if (stage0 < MaxTevStages) {
      g_gxState.tevStages[stage0].kcSel = static_cast<GXTevKColorSel>(bp_get(value, 5, 4));
      g_gxState.tevStages[stage0].kaSel = static_cast<GXTevKAlphaSel>(bp_get(value, 5, 9));
    }
    if (stage1 < MaxTevStages) {
      g_gxState.tevStages[stage1].kcSel = static_cast<GXTevKColorSel>(bp_get(value, 5, 14));
      g_gxState.tevStages[stage1].kaSel = static_cast<GXTevKAlphaSel>(bp_get(value, 5, 19));
    }
    g_gxState.stateDirty = true;
    break;
  }

  // Fog A/B parameters (0xEE-0xF0)
  // FOG0 (0xEE): A parameter - sign(1)|exp(8)|mantissa(11) partial IEEE 754 float
  case 0xEE: {
    g_gxState.fog.fog0Raw = value;
    // Reconstruct A = a_encoded * 2^b_s
    u32 a_mant = bp_get(value, 11, 0);
    u32 a_exp = bp_get(value, 8, 11);
    u32 a_sign = bp_get(value, 1, 19);
    u32 a_bits = (a_sign << 31) | (a_exp << 23) | (a_mant << 12);
    float a_encoded;
    std::memcpy(&a_encoded, &a_bits, sizeof(a_encoded));
    u32 b_s = g_gxState.fog.fog2Raw & 0x1F;
    g_gxState.fog.a = std::ldexp(a_encoded, static_cast<int>(b_s));
    g_gxState.stateDirty = true;
    break;
  }
  // FOG1 (0xEF): B mantissa (24-bit)
  case 0xEF: {
    g_gxState.fog.fog1Raw = value;
    u32 b_m = bp_get(value, 24, 0);
    u32 b_s = g_gxState.fog.fog2Raw & 0x1F;
    float B_mant = static_cast<float>(b_m) / 8388638.0f;
    g_gxState.fog.b = std::ldexp(B_mant, static_cast<int>(b_s) - 1);
    g_gxState.stateDirty = true;
    break;
  }
  // FOG2 (0xF0): B shift/exponent (5-bit)
  case 0xF0: {
    g_gxState.fog.fog2Raw = value;
    u32 b_s = bp_get(value, 5, 0);
    // Recompute A with updated b_s
    u32 a_mant = bp_get(g_gxState.fog.fog0Raw, 11, 0);
    u32 a_exp = bp_get(g_gxState.fog.fog0Raw, 8, 11);
    u32 a_sign = bp_get(g_gxState.fog.fog0Raw, 1, 19);
    u32 a_bits = (a_sign << 31) | (a_exp << 23) | (a_mant << 12);
    float a_encoded;
    std::memcpy(&a_encoded, &a_bits, sizeof(a_encoded));
    g_gxState.fog.a = std::ldexp(a_encoded, static_cast<int>(b_s));
    // Recompute B with updated b_s
    u32 b_m = bp_get(g_gxState.fog.fog1Raw, 24, 0);
    float B_mant = static_cast<float>(b_m) / 8388638.0f;
    g_gxState.fog.b = std::ldexp(B_mant, static_cast<int>(b_s) - 1);
    g_gxState.stateDirty = true;
    break;
  }

  // Fog type + C parameter from FOG3 (0xF1)
  case 0xF1: {
    GXFogType fogType = static_cast<GXFogType>(bp_get(value, 3, 21));
    g_gxState.fog.type = fogType;
    // Decode C parameter (same partial float encoding as A)
    u32 c_mant = bp_get(value, 11, 0);
    u32 c_exp = bp_get(value, 8, 11);
    u32 c_sign = bp_get(value, 1, 19);
    u32 c_bits = (c_sign << 31) | (c_exp << 23) | (c_mant << 12);
    std::memcpy(&g_gxState.fog.c, &c_bits, sizeof(g_gxState.fog.c));
    g_gxState.stateDirty = true;
    break;
  }

  // Fog color from FOGCLR (0xF2)
  case 0xF2: {
    u8 b = bp_get(value, 8, 0);
    u8 g = bp_get(value, 8, 8);
    u8 r = bp_get(value, 8, 16);
    g_gxState.fog.color = {
        static_cast<float>(r) / 255.f,
        static_cast<float>(g) / 255.f,
        static_cast<float>(b) / 255.f,
        1.f,
    };
    g_gxState.stateDirty = true;
    break;
  }

  // TEV color registers / K color registers (0xE0-0xE7)
  // RA registers: 0xE0, 0xE2, 0xE4, 0xE6 (even)
  // BG registers: 0xE1, 0xE3, 0xE5, 0xE7 (odd)
  // Bit 23 distinguishes: 0 = TEV color register, 1 = K color register
  case 0xE0:
  case 0xE1:
  case 0xE2:
  case 0xE3:
  case 0xE4:
  case 0xE5:
  case 0xE6:
  case 0xE7: {
    u32 idx = (regId - 0xE0) / 2;
    bool isRA = (regId & 1) == 0;
    bool isKColor = bp_get(value, 1, 23) != 0;

    if (isKColor) {
      // K color register (8-bit components)
      if (idx < GX_MAX_KCOLOR) {
        auto& kc = g_gxState.kcolors[idx];
        if (isRA) {
          kc[0] = static_cast<float>(bp_get(value, 8, 0)) / 255.f;  // R
          kc[3] = static_cast<float>(bp_get(value, 8, 12)) / 255.f; // A
        } else {
          kc[2] = static_cast<float>(bp_get(value, 8, 0)) / 255.f;  // B
          kc[1] = static_cast<float>(bp_get(value, 8, 12)) / 255.f; // G
        }
        g_gxState.stateDirty = true;
      }
    } else {
      // TEV color register (11-bit signed components)
      if (idx < MaxTevRegs) {
        auto& cr = g_gxState.colorRegs[idx];
        if (isRA) {
          // 11-bit signed: sign-extend from 11 bits
          s32 r = bp_get(value, 11, 0);
          if (r & 0x400)
            r |= ~0x7FF; // sign extend
          s32 a = bp_get(value, 11, 12);
          if (a & 0x400)
            a |= ~0x7FF;
          cr[0] = static_cast<float>(r) / 255.f;
          cr[3] = static_cast<float>(a) / 255.f;
        } else {
          s32 b = bp_get(value, 11, 0);
          if (b & 0x400)
            b |= ~0x7FF;
          s32 g = bp_get(value, 11, 12);
          if (g & 0x400)
            g |= ~0x7FF;
          cr[2] = static_cast<float>(b) / 255.f;
          cr[1] = static_cast<float>(g) / 255.f;
        }
        g_gxState.stateDirty = true;
      }
    }
    break;
  }

  // Indirect texture matrices (0x06-0x0E)
  // Each matrix uses 3 consecutive registers (one per row of the 3x2 matrix).
  // Matrix 0: 0x06-0x08, Matrix 1: 0x09-0x0B, Matrix 2: 0x0C-0x0E
  case 0x06:
  case 0x07:
  case 0x08:
  case 0x09:
  case 0x0A:
  case 0x0B:
  case 0x0C:
  case 0x0D:
  case 0x0E: {
    u32 idx = (regId - 0x06) / 3;    // matrix index (0-2)
    u32 column = (regId - 0x06) % 3; // column index (0-2)
    auto& info = g_gxState.indTexMtxs[idx];

    // Decode one packed matrix column: [m[0][column], m[1][column]].
    s32 col0 = bp_get(value, 11, 0);
    if (col0 & 0x400)
      col0 |= ~0x7FF; // sign-extend from 11 bits
    s32 col1 = bp_get(value, 11, 11);
    if (col1 & 0x400)
      col1 |= ~0x7FF;

    auto& packedColumn = column == 0 ? info.mtx.m0 : (column == 1 ? info.mtx.m1 : info.mtx.m2);
    packedColumn.x = static_cast<float>(col0) / 1024.0f;
    packedColumn.y = static_cast<float>(col1) / 1024.0f;

    // Accumulate the indirect matrix scale exponent. The SDK writes two bits per column, but
    // the hardware appears to ignore the top bit from the third column, leaving an effective
    // 5-bit value for adjScale = scaleExp + 17.
    u32 scaleBits = bp_get(value, 2, 22);
    u32 shift = column * 2;
    if (column == 2) {
      info.adjScaleRaw = (info.adjScaleRaw & ~(1u << shift)) | ((scaleBits & 1u) << shift);
    } else {
      info.adjScaleRaw = (info.adjScaleRaw & ~(3u << shift)) | (scaleBits << shift);
    }
    info.scaleExp = static_cast<s8>(info.adjScaleRaw) - 17;

    g_gxState.stateDirty = true;
    break;
  }

  // SU texture coordinate scale registers (0x30-0x3F)
  // Even registers (suTs0): S-axis scale, bias, cyl wrap, line/point offset
  // Odd registers (suTs1): T-axis scale, bias, cyl wrap
  case 0x30:
  case 0x31:
  case 0x32:
  case 0x33:
  case 0x34:
  case 0x35:
  case 0x36:
  case 0x37:
  case 0x38:
  case 0x39:
  case 0x3A:
  case 0x3B:
  case 0x3C:
  case 0x3D:
  case 0x3E:
  case 0x3F: {
    u32 coordIdx = (regId - 0x30) / 2;
    bool isT = (regId & 1) != 0;
    auto& tcs = g_gxState.texCoordScales[coordIdx];
    if (isT) {
      tcs.scaleT = static_cast<u16>(bp_get(value, 16, 0));
      tcs.biasT = bp_get(value, 1, 16) != 0;
      tcs.cylWrapT = bp_get(value, 1, 17) != 0;
    } else {
      tcs.scaleS = static_cast<u16>(bp_get(value, 16, 0));
      tcs.biasS = bp_get(value, 1, 16) != 0;
      tcs.cylWrapS = bp_get(value, 1, 17) != 0;
      tcs.lineOffset = bp_get(value, 1, 18) != 0;
      tcs.pointOffset = bp_get(value, 1, 19) != 0;
    }
    g_gxState.stateDirty = true;
    break;
  }

  // Copy clear color (0x4F-0x50) and depth (0x51)
  case 0x4F: {
    u8 r = bp_get(value, 8, 0);
    u8 a = bp_get(value, 8, 8);
    g_gxState.clearColor[0] = static_cast<float>(r) / 255.f;
    g_gxState.clearColor[3] = static_cast<float>(a) / 255.f;
    g_gxState.stateDirty = true;
    break;
  }
  case 0x50: {
    u8 b = bp_get(value, 8, 0);
    u8 g = bp_get(value, 8, 8);
    g_gxState.clearColor[2] = static_cast<float>(b) / 255.f;
    g_gxState.clearColor[1] = static_cast<float>(g) / 255.f;
    g_gxState.stateDirty = true;
    break;
  }
  case 0x51: {
    g_gxState.clearDepth = bp_get(value, 24, 0);
    g_gxState.stateDirty = true;
    break;
  }

  default:
    if (const auto mapping = decode_tex_bp_reg(regId); mapping.has_value()) {
      g_gxState.textureDirty++;
      auto& slot = g_gxState.loadedTextures[mapping->texMapId];
      switch (mapping->kind) {
      case TexBpRegMapping::Kind::Mode0:
        slot.mode0 = value;
        break;
      case TexBpRegMapping::Kind::Mode1:
        slot.mode1 = value;
        break;
      case TexBpRegMapping::Kind::Image0:
        slot.image0 = value;
        slot.mWidth = 0;
        slot.mHeight = 0;
        slot.mFormat = gfx::InvalidTextureFormat;
        break;
      case TexBpRegMapping::Kind::Image3:
        slot.image3 = value;
        break;
      case TexBpRegMapping::Kind::Tlut:
        // TLUT region's TMEM offset
        break;
      case TexBpRegMapping::Kind::Image1:
      case TexBpRegMapping::Kind::Image2:
        // GXTexRegion regs
        break;
      }
      g_gxState.stateDirty = true;
    } else {
#ifndef NDEBUG
      Log.debug("Unhandled BP register 0x{:02X} (value 0x{:06X})", regId, value & 0xFFFFFF);
#endif
    }
    break;
  }
}

// CP register handler - decodes CP register writes and updates g_gxState
static void handle_cp(u8 addr, u32 value, bool bigEndian) {
  switch (addr) {
  // VCD low (0x50)
  case 0x50: {
    auto& vd = g_gxState.vtxDesc;
    vd[GX_VA_PNMTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 0));
    vd[GX_VA_TEX0MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 1));
    vd[GX_VA_TEX1MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 2));
    vd[GX_VA_TEX2MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 3));
    vd[GX_VA_TEX3MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 4));
    vd[GX_VA_TEX4MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 5));
    vd[GX_VA_TEX5MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 6));
    vd[GX_VA_TEX6MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 7));
    vd[GX_VA_TEX7MTXIDX] = static_cast<GXAttrType>(bp_get(value, 1, 8));
    vd[GX_VA_POS] = static_cast<GXAttrType>(bp_get(value, 2, 9));
    vd[GX_VA_NRM] = static_cast<GXAttrType>(bp_get(value, 2, 11));
    vd[GX_VA_CLR0] = static_cast<GXAttrType>(bp_get(value, 2, 13));
    vd[GX_VA_CLR1] = static_cast<GXAttrType>(bp_get(value, 2, 15));
    g_gxState.stateDirty = true;
    g_gxState.clearVtxSizeCache();
    break;
  }

  // VCD high (0x60)
  case 0x60: {
    auto& vd = g_gxState.vtxDesc;
    vd[GX_VA_TEX0] = static_cast<GXAttrType>(bp_get(value, 2, 0));
    vd[GX_VA_TEX1] = static_cast<GXAttrType>(bp_get(value, 2, 2));
    vd[GX_VA_TEX2] = static_cast<GXAttrType>(bp_get(value, 2, 4));
    vd[GX_VA_TEX3] = static_cast<GXAttrType>(bp_get(value, 2, 6));
    vd[GX_VA_TEX4] = static_cast<GXAttrType>(bp_get(value, 2, 8));
    vd[GX_VA_TEX5] = static_cast<GXAttrType>(bp_get(value, 2, 10));
    vd[GX_VA_TEX6] = static_cast<GXAttrType>(bp_get(value, 2, 12));
    vd[GX_VA_TEX7] = static_cast<GXAttrType>(bp_get(value, 2, 14));
    g_gxState.stateDirty = true;
    g_gxState.clearVtxSizeCache();
    break;
  }

  // Matrix index A (0x30)
  case 0x30: {
    const u32 newPnMtx = bp_get(value, 6, 0) / 3;
    if (g_gxState.currentPnMtx != newPnMtx) {
      g_gxState.currentPnMtx = newPnMtx;
      g_gxState.stateDirty = true;
      g_gxState.uniformDirty++;
    }
    break;
  }

  // Matrix index B (0x40)
  case 0x40:
    // Texture matrix indices - used for multi-matrix texgen
    break;

  default:
    // VAT A registers (0x70-0x77)
    if (addr >= 0x70 && addr <= 0x77) {
      u32 fmt = addr - 0x70;
      auto& vf = g_gxState.vtxFmts[fmt];
      vf.attrs[GX_VA_POS].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 0));
      vf.attrs[GX_VA_POS].type = static_cast<GXCompType>(bp_get(value, 3, 1));
      vf.attrs[GX_VA_POS].frac = static_cast<u8>(bp_get(value, 5, 4));
      const auto nrm_cnt = bp_get(value, 1, 9);
      const auto nrm_nbt3 = bp_get(value, 1, 31);
      vf.attrs[GX_VA_NRM].cnt = static_cast<GXCompCnt>(nrm_nbt3 ? GX_NRM_NBT3 : (nrm_cnt ? GX_NRM_NBT : GX_NRM_XYZ));
      vf.attrs[GX_VA_NRM].type = static_cast<GXCompType>(bp_get(value, 3, 10));
      if (vf.attrs[GX_VA_NRM].type == GX_U8 || vf.attrs[GX_VA_NRM].type == GX_S8) {
        vf.attrs[GX_VA_NRM].frac = 6;
      } else if (vf.attrs[GX_VA_NRM].type == GX_U16 || vf.attrs[GX_VA_NRM].type == GX_S16) {
        vf.attrs[GX_VA_NRM].frac = 14;
      } else {
        vf.attrs[GX_VA_NRM].frac = 0;
      }
      vf.attrs[GX_VA_CLR0].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 13));
      vf.attrs[GX_VA_CLR0].type = static_cast<GXCompType>(bp_get(value, 3, 14));
      vf.attrs[GX_VA_CLR0].frac = 0;
      vf.attrs[GX_VA_CLR1].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 17));
      vf.attrs[GX_VA_CLR1].type = static_cast<GXCompType>(bp_get(value, 3, 18));
      vf.attrs[GX_VA_CLR1].frac = 0;
      vf.attrs[GX_VA_TEX0].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 21));
      vf.attrs[GX_VA_TEX0].type = static_cast<GXCompType>(bp_get(value, 3, 22));
      vf.attrs[GX_VA_TEX0].frac = static_cast<u8>(bp_get(value, 5, 25));
      g_gxState.stateDirty = true;
      g_gxState.clearVtxSizeCache();
    }
    // VAT B registers (0x80-0x87)
    else if (addr >= 0x80 && addr <= 0x87) {
      u32 fmt = addr - 0x80;
      auto& vf = g_gxState.vtxFmts[fmt];
      vf.attrs[GX_VA_TEX1].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 0));
      vf.attrs[GX_VA_TEX1].type = static_cast<GXCompType>(bp_get(value, 3, 1));
      vf.attrs[GX_VA_TEX1].frac = static_cast<u8>(bp_get(value, 5, 4));
      vf.attrs[GX_VA_TEX2].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 9));
      vf.attrs[GX_VA_TEX2].type = static_cast<GXCompType>(bp_get(value, 3, 10));
      vf.attrs[GX_VA_TEX2].frac = static_cast<u8>(bp_get(value, 5, 13));
      vf.attrs[GX_VA_TEX3].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 18));
      vf.attrs[GX_VA_TEX3].type = static_cast<GXCompType>(bp_get(value, 3, 19));
      vf.attrs[GX_VA_TEX3].frac = static_cast<u8>(bp_get(value, 5, 22));
      vf.attrs[GX_VA_TEX4].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 27));
      vf.attrs[GX_VA_TEX4].type = static_cast<GXCompType>(bp_get(value, 3, 28));
      // TEX4 frac is in VAT C
      g_gxState.stateDirty = true;
      g_gxState.clearVtxSizeCache();
    }
    // VAT C registers (0x90-0x97)
    else if (addr >= 0x90 && addr <= 0x97) {
      u32 fmt = addr - 0x90;
      auto& vf = g_gxState.vtxFmts[fmt];
      vf.attrs[GX_VA_TEX4].frac = static_cast<u8>(bp_get(value, 5, 0));
      vf.attrs[GX_VA_TEX5].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 5));
      vf.attrs[GX_VA_TEX5].type = static_cast<GXCompType>(bp_get(value, 3, 6));
      vf.attrs[GX_VA_TEX5].frac = static_cast<u8>(bp_get(value, 5, 9));
      vf.attrs[GX_VA_TEX6].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 14));
      vf.attrs[GX_VA_TEX6].type = static_cast<GXCompType>(bp_get(value, 3, 15));
      vf.attrs[GX_VA_TEX6].frac = static_cast<u8>(bp_get(value, 5, 18));
      vf.attrs[GX_VA_TEX7].cnt = static_cast<GXCompCnt>(bp_get(value, 1, 23));
      vf.attrs[GX_VA_TEX7].type = static_cast<GXCompType>(bp_get(value, 3, 24));
      vf.attrs[GX_VA_TEX7].frac = static_cast<u8>(bp_get(value, 5, 27));
      g_gxState.stateDirty = true;
      g_gxState.clearVtxSizeCache();
    }
    // Array base addresses (0xA0-0xAF)
    else if (addr >= 0xA0 && addr <= 0xAF) {
      Log.error("CP_REG_ARRAYBASE_ID is not supported on Aurora. Use GX_AURORA_LOAD_ARRAYBASE instead.");
    }
    // Array strides (0xB0-0xBF)
    else if (addr >= 0xB0 && addr <= 0xBF) {
      u32 attrIdx = addr - 0xB0 + GX_VA_POS;
      if (attrIdx < GX_VA_MAX_ATTR) {
        auto& array = g_gxState.arrays[attrIdx];
        const auto newStride = static_cast<u8>(value);
        if (array.stride != newStride) {
          array.stride = newStride;
          g_gxState.stateDirty = true;
        }
      }
    }
    break;
  }
}

// XF register handler - decodes XF (transform unit) register writes and updates g_gxState
static void handle_xf(const u8* data, u32& pos, u32 size, bool bigEndian, bool& pipelineChanged) {
  CHECK(pos + 4 <= size, "XF header read overrun");
  u32 header = read_u32(data + pos, bigEndian);
  pos += 4;

  u32 count = ((header >> 16) & 0xFFFF) + 1;
  u32 addr = header & 0xFFFF;
  u32 dataBytes = count * 4;
  // Log.warn("  xf: addr {:04x} count {} dataBytes {} pos {} -> {}", addr, count, dataBytes, pos, pos + dataBytes);
  CHECK(pos + dataBytes <= size, "XF data read overrun: need {} bytes at pos {}", dataBytes, pos);

  const u8* xfData = data + pos;

  if (copy_xf_data(addr, xfData, count, bigEndian)) {
    // copy_xf_data handled everything (matrices and lights, uniforms only).
  } else if (addr >= 0x1000) {
    // XF registers (0x1000+)
    u32 xfAddr = addr - 0x1000;
    for (u32 i = 0; i < count; i++) {
      u32 reg = xfAddr + i;
      u32 val = read_u32(xfData + i * 4, bigEndian);

      // Skip scalar register writes that haven't changed (viewport/projection handled below)
      if (reg <= 0x19 && val == g_gxState.xfRegCache[reg])
        continue;
      if (reg <= 0x19)
        g_gxState.xfRegCache[reg] = val;

      switch (reg) {
      case 0x08:
        // XF vertex specs (numColors, numNormals, numTexCoords) - informational
        break;
      case 0x09:
        // numChans
        g_gxState.numChans = val;
        g_gxState.stateDirty = true;
        pipelineChanged = true;
        break;
      case 0x0A:
        // Ambient color 0
        g_gxState.colorChannelState[GX_COLOR0].ambColor = unpack_color(val);
        g_gxState.colorChannelState[GX_ALPHA0].ambColor = unpack_color(val);
        g_gxState.stateDirty = true;
        break;
      case 0x0B:
        // Ambient color 1
        g_gxState.colorChannelState[GX_COLOR1].ambColor = unpack_color(val);
        g_gxState.colorChannelState[GX_ALPHA1].ambColor = unpack_color(val);
        g_gxState.stateDirty = true;
        break;
      case 0x0C:
        // Material color 0
        g_gxState.colorChannelState[GX_COLOR0].matColor = unpack_color(val);
        g_gxState.colorChannelState[GX_ALPHA0].matColor = unpack_color(val);
        g_gxState.stateDirty = true;
        break;
      case 0x0D:
        // Material color 1
        g_gxState.colorChannelState[GX_COLOR1].matColor = unpack_color(val);
        g_gxState.colorChannelState[GX_ALPHA1].matColor = unpack_color(val);
        g_gxState.stateDirty = true;
        break;
      case 0x0E:
      case 0x0F:
      case 0x10:
      case 0x11: {
        // Channel control registers
        u32 chanId = reg - 0x0E;
        if (chanId < MaxColorChannels) {
          auto& chan = g_gxState.colorChannelConfig[chanId];
          chan.matSrc = static_cast<GXColorSrc>(bp_get(val, 1, 0));
          chan.lightingEnabled = bp_get(val, 1, 1) != 0;
          u32 lightsLo = bp_get(val, 4, 2);
          chan.ambSrc = static_cast<GXColorSrc>(bp_get(val, 1, 6));
          chan.diffFn = static_cast<GXDiffuseFn>(bp_get(val, 2, 7));
          // Encoding: bit 9 = (attnFn != GX_AF_SPEC), bit 10 = (attnFn != GX_AF_NONE)
          bool bit9 = bp_get(val, 1, 9) != 0;
          bool bit10 = bp_get(val, 1, 10) != 0;
          u32 lightsHi = bp_get(val, 4, 11);
          if (!bit10) {
            chan.attnFn = GX_AF_NONE;
          } else if (!bit9) {
            chan.attnFn = GX_AF_SPEC;
          } else {
            chan.attnFn = GX_AF_SPOT;
          }
          u32 lightMask = lightsLo | (lightsHi << 4);
          g_gxState.colorChannelState[chanId].lightMask = GX::LightMask{lightMask};
          g_gxState.stateDirty = true;
          pipelineChanged = true;
        }
        break;
      }
      case 0x18: {
        // Matrix index A: PnMtx + TexCoord0-3 matrix indices
        g_gxState.currentPnMtx = bp_get(val, 6, 0) / 3;
        for (u32 i = 0; i < 4 && i < MaxTexCoord; i++) {
          auto texMtx = static_cast<GXTexMtx>(bp_get(val, 6, 6 + i * 6));
          assert(texMtx >= 0 && texMtx <= GXTexMtx::GX_IDENTITY);
          if (g_gxState.tcgs[i].mtx != texMtx) {
            g_gxState.tcgs[i].mtx = texMtx;
            pipelineChanged = true;
          }
        }
        g_gxState.stateDirty = true;
        break;
      }
      case 0x19: {
        // Matrix index B: TexCoord4-7 matrix indices
        for (u32 i = 0; i < 4 && (i + 4) < MaxTexCoord; i++) {
          auto texMtx = static_cast<GXTexMtx>(bp_get(val, 6, i * 6));
          if (g_gxState.tcgs[i + 4].mtx != texMtx) {
            g_gxState.tcgs[i + 4].mtx = texMtx;
            pipelineChanged = true;
          }
        }
        g_gxState.stateDirty = true;
        break;
      }
      case 0x1A:
      case 0x1B:
      case 0x1C:
      case 0x1D:
      case 0x1E:
      case 0x1F: {
        // Viewport: sx, sy, sz, ox, oy, oz at XF 0x101A-0x101F
        u32 vpOff = reg - 0x1A;
        if (vpOff == 0 && count >= 6) {
          f32 sx = read_f32(xfData + 0, bigEndian);
          f32 sy = read_f32(xfData + 4, bigEndian);
          f32 sz = read_f32(xfData + 8, bigEndian);
          f32 ox = read_f32(xfData + 12, bigEndian);
          f32 oy = read_f32(xfData + 16, bigEndian);
          f32 oz = read_f32(xfData + 20, bigEndian);
          f32 width = sx * 2.0f;
          f32 height = -sy * 2.0f;
          set_logical_viewport({
              .left = ox - 340.0f - width / 2.0f,
              .top = oy - 340.0f - height / 2.0f,
              .width = width,
              .height = height,
              .znear = (oz - sz) / 1.6777215e7f,
              .zfar = oz / 1.6777215e7f,
          });
        }
        break;
      }
      case 0x20:
      case 0x21:
      case 0x22:
      case 0x23:
      case 0x24:
      case 0x25:
      case 0x26: {
        // Projection: 6 params + type at XF 0x1020-0x1026
        u32 projOff = reg - 0x20;
        if (projOff == 0 && count >= 7) {
          f32 p0 = read_f32(xfData + 0, bigEndian);
          f32 p1 = read_f32(xfData + 4, bigEndian);
          f32 p2 = read_f32(xfData + 8, bigEndian);
          f32 p3 = read_f32(xfData + 12, bigEndian);
          f32 p4 = read_f32(xfData + 16, bigEndian);
          f32 p5 = read_f32(xfData + 20, bigEndian);
          u32 projType = read_u32(xfData + 24, bigEndian);
          g_gxState.projType = static_cast<GXProjectionType>(projType);
          // Reconstruct 4x4 projection matrix from 6 params
          auto& proj = g_gxState.proj;
          proj = {};
          proj.m0[0] = p0;
          proj.m1[1] = p2;
          proj.m2[2] = p4;
          proj.m2[3] = p5;
          if (projType == GX_ORTHOGRAPHIC) {
            proj.m0[3] = p1;
            proj.m1[3] = p3;
            proj.m3[3] = 1.0f;
          } else {
            proj.m0[2] = p1;
            proj.m1[2] = p3;
            proj.m3[2] = -1.0f;
          }
          g_gxState.stateDirty = true;
        }
        break;
      }
      case 0x3F:
        // numTexGens
        g_gxState.numTexGens = val;
        g_gxState.stateDirty = true;
        pipelineChanged = true;
        break;
      default:
        // TexGen config (0x40-0x4F) and post-transform (0x50-0x5F)
        if (reg >= 0x40 && reg <= 0x4F) {
          u32 tcIdx = reg - 0x40;
          if (tcIdx < MaxTexCoord) {
            auto& tcg = g_gxState.tcgs[tcIdx];
            bool proj = bp_get(val, 1, 1) != 0;
            u32 form = bp_get(val, 1, 2);
            u32 tgType = bp_get(val, 3, 4);
            u32 srcRow = bp_get(val, 5, 7);

            if (tgType == 0) {
              tcg.type = proj ? GX_TG_MTX3x4 : GX_TG_MTX2x4;
            } else if (tgType == 1) {
              // Bump mapping: type encodes emboss light
              tcg.type = static_cast<GXTexGenType>(bp_get(val, 3, 15) + 2);
            } else if (tgType == 2 || tgType == 3) {
              tcg.type = GX_TG_SRTG;
            }
            // Emboss source texcoord (bits 12-14); 0 for non-bump types
            tcg.embossSrc = bp_get(val, 3, 12);

            // Decode source from row
            static const GXTexGenSrc rowToSrc[] = {GX_TG_POS,  GX_TG_NRM,  GX_TG_COLOR0, GX_TG_BINRM, GX_TG_TANGENT,
                                                   GX_TG_TEX0, GX_TG_TEX1, GX_TG_TEX2,   GX_TG_TEX3,  GX_TG_TEX4,
                                                   GX_TG_TEX5, GX_TG_TEX6, GX_TG_TEX7};
            if (srcRow < 13) {
              tcg.src = rowToSrc[srcRow];
            }
            g_gxState.stateDirty = true;
            pipelineChanged = true;
          }
        } else if (reg >= 0x50 && reg <= 0x5F) {
          u32 tcIdx = reg - 0x50;
          if (tcIdx < MaxTexCoord) {
            g_gxState.tcgs[tcIdx].postMtx = static_cast<GXPTTexMtx>(bp_get(val, 6, 0) + 64);
            g_gxState.tcgs[tcIdx].normalize = bp_get(val, 1, 8) != 0;
            g_gxState.stateDirty = true;
            pipelineChanged = true;
          }
        } else {
#ifndef NDEBUG
          Log.debug("Unhandled XF register 0x{:04X} (value 0x{:08X})", reg, val);
#endif
        }
        break;
      }
    }
  }

  pos += dataBytes;
}

static void handle_draw_overrun [[noreturn]] (u32 totalVtxBytes, const u8* data, const u32& pos, u32 size) {
  // Hex dump around the draw command for debugging
  u32 cmdPos = pos - 2 - 1; // opcode byte position (before vtxCount and pos++)
  u32 dumpStart = (cmdPos > 16) ? cmdPos - 16 : 0;
  u32 dumpEnd = (cmdPos + 32 < size) ? cmdPos + 32 : size;
  std::string hex;
  for (u32 i = dumpStart; i < dumpEnd; i++) {
    if (i == cmdPos)
      hex += fmt::format("[{:02x}]", data[i]);
    else
      hex += fmt::format(" {:02x}", data[i]);
  }
  Log.error("  hex dump around draw cmd (pos {}-{}):{}", dumpStart, dumpEnd - 1, hex);
  FATAL("draw vertex data overrun: need {} bytes at pos {}, have {}", totalVtxBytes, pos, size);
}

// Draw command handler - parses vertices inline and caches results

struct BoundingBox {
  bool armed = false;
  bool unknown = false;
  float left = 0.f, top = 0.f, right = -1.f, bottom = -1.f;
  u16 result[4] = {0x3FF, 0x3FF, 0, 0};
};
BoundingBox sBbox;

bool bbox_read_position(const u8* vtx, GXVtxFmt fmt, float out[3], u32& pnMtx) noexcept {
  const auto& vf = g_gxState.vtxFmts[fmt];
  u32 off = 0;
  pnMtx = g_gxState.currentPnMtx;
  for (int i = GX_VA_PNMTXIDX; i < GX_VA_POS; ++i) {
    switch (g_gxState.vtxDesc[i]) {
    case GX_NONE:
      break;
    case GX_DIRECT:
      if (i == GX_VA_PNMTXIDX) {
        pnMtx = vtx[off] / 3;
      }
      off += comp_type_size(static_cast<GXAttr>(i), vf.attrs[i].type) *
             comp_cnt_count(static_cast<GXAttr>(i), vf.attrs[i].cnt);
      break;
    case GX_INDEX8:
      off += 1;
      break;
    case GX_INDEX16:
      off += 2;
      break;
    }
  }
  const auto& af = vf.attrs[GX_VA_POS];
  const u32 count = af.cnt == GX_POS_XY ? 2 : 3;
  const u8* src = nullptr;
  bool le = false;
  switch (g_gxState.vtxDesc[GX_VA_POS]) {
  case GX_DIRECT:
    src = vtx + off;
    break;
  case GX_INDEX8:
  case GX_INDEX16: {
    const u32 idx = g_gxState.vtxDesc[GX_VA_POS] == GX_INDEX8 ? vtx[off] : (u32(vtx[off]) << 8 | vtx[off + 1]);
    const auto& arr = g_gxState.arrays[GX_VA_POS];
    if (arr.data == nullptr || arr.stride == 0) {
      return false;
    }
    if (arr.size != 0 && u64(idx) * arr.stride + u64(count) * 4 > arr.size + 4) {
      return false;
    }
    src = static_cast<const u8*>(arr.data) + u64(idx) * arr.stride;
    le = arr.le;
    break;
  }
  default:
    return false;
  }
  const float scale = 1.f / static_cast<float>(1u << af.frac);
  for (u32 c = 0; c < count; ++c) {
    switch (af.type) {
    case GX_U8:
      out[c] = static_cast<float>(src[c]) * scale;
      break;
    case GX_S8:
      out[c] = static_cast<float>(static_cast<s8>(src[c])) * scale;
      break;
    case GX_U16:
    case GX_S16: {
      u16 v;
      memcpy(&v, src + 2 * c, 2);
      if (!le) {
        v = bswap(v);
      }
      out[c] = (af.type == GX_U16 ? static_cast<float>(v) : static_cast<float>(static_cast<s16>(v))) * scale;
      break;
    }
    case GX_F32: {
      u32 v;
      memcpy(&v, src + 4 * c, 4);
      if (!le) {
        v = bswap(v);
      }
      out[c] = std::bit_cast<float>(v);
      break;
    }
    default:
      return false;
    }
  }
  if (count == 2) {
    out[2] = 0.f;
  }
  return true;
}

void bbox_add_vertices(const u8* data, u32 vtxCount, u32 vtxSize, GXVtxFmt fmt) noexcept {
  if (vtxSize == 0) {
    return;
  }
  const auto& proj = g_gxState.proj;
  const auto& vp = g_gxState.logicalViewport;
  for (u32 i = 0; i < vtxCount && !sBbox.unknown; ++i) {
    float p[3];
    u32 mtx;
    if (!bbox_read_position(data + i * vtxSize, fmt, p, mtx) || mtx >= MaxPnMtx) {
      sBbox.unknown = true;
      break;
    }
    const auto& m = g_gxState.pnMtx[mtx].pos;
    const float v0 = m.m0[0] * p[0] + m.m0[1] * p[1] + m.m0[2] * p[2] + m.m0[3];
    const float v1 = m.m1[0] * p[0] + m.m1[1] * p[1] + m.m1[2] * p[2] + m.m1[3];
    const float v2 = m.m2[0] * p[0] + m.m2[1] * p[1] + m.m2[2] * p[2] + m.m2[3];
    const float cx = proj.m0[0] * v0 + proj.m0[1] * v1 + proj.m0[2] * v2 + proj.m0[3];
    const float cy = proj.m1[0] * v0 + proj.m1[1] * v1 + proj.m1[2] * v2 + proj.m1[3];
    const float cw = proj.m3[0] * v0 + proj.m3[1] * v1 + proj.m3[2] * v2 + proj.m3[3];
    if (!(cw > 1e-6f)) {
      sBbox.unknown = true;
      break;
    }
    const float sx = vp.left + (cx / cw + 1.f) * 0.5f * vp.width;
    const float sy = vp.top + (1.f - cy / cw) * 0.5f * vp.height;
    sBbox.left = std::min(sBbox.left, sx);
    sBbox.right = std::max(sBbox.right, sx);
    sBbox.top = std::min(sBbox.top, sy);
    sBbox.bottom = std::max(sBbox.bottom, sy);
  }
}

static u32 calculate_last_vtx_size(GXVtxFmt fmt) {
  u32 vtxSize = 0;
  const auto& vtxFmt = g_gxState.vtxFmts[fmt];
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    switch (g_gxState.vtxDesc[i]) {
    case GX_NONE:
      break;
    case GX_DIRECT: {
      const auto attr = static_cast<GXAttr>(i);
      const auto& attrFmt = vtxFmt.attrs[i];
      vtxSize += comp_type_size(attr, attrFmt.type) * comp_cnt_count(attr, attrFmt.cnt);
      break;
    }
    case GX_INDEX8:
      vtxSize += (i == GX_VA_NRM && vtxFmt.attrs[i].cnt == GX_NRM_NBT3) ? 3 : 1;
      break;
    case GX_INDEX16:
      vtxSize += (i == GX_VA_NRM && vtxFmt.attrs[i].cnt == GX_NRM_NBT3) ? 6 : 2;
      break;
    }
  }

  g_gxState.lastVtxFmt = fmt;
  g_gxState.lastVtxSize = vtxSize;

  return vtxSize;
}

static void handle_draw_unmerged(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange);
static void push_gx_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange, gfx::Range idxRange,
                         u32 numIndices);
// Mali fork: storage vertex fetch is impossible on the target
// (GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS=0), so a draw that native fetch declines —
// instanced lines/points (e.g. GX_LINESTRIP minimap draws), NBT normals, exotic
// component types — is dropped instead of building a storage pipeline that can't
// link and aborts the device. If any dropped geometry turns out to matter, extend
// native fetch to cover it (e.g. instanced lines); texture-vertex fetch is NOT an
// option — it's unusably slow on Mali. Flip to false only to exercise storage on desktop.
static constexpr bool kSkipStorageVertexFetch = true;

static bool try_native_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, const u8* data, u32& pos, u32 vtxSize);
static bool try_native_draw_run(u8 cmd, GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, u32 vtxSize, const u8* data,
                                u32& pos, u32 size, bool bigEndian);
static bool try_native_draw_indexed(GXVtxFmt fmt, u16 vtxCount, const u8* vtxData, u32 vtxSize, const u8* idxData,
                                    u32 idxBytes, u32 indexCount);

// Draw command handler - parses vertices inline and caches results
static ByteBuffer handle_draw_idx_buf;

static void draw_prim(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, const u8* data, u32& pos, u32 size) {
  ZoneScoped;
  u32 vtxSize;
  if (g_gxState.lastVtxFmt == fmt)
    LIKELY { vtxSize = g_gxState.lastVtxSize; }
  else
    UNLIKELY { vtxSize = calculate_last_vtx_size(fmt); }

  u32 totalVtxBytes = vtxCount * vtxSize;
  if (pos + totalVtxBytes > size)
    UNLIKELY { handle_draw_overrun(totalVtxBytes, data, pos, size); }

  if (sBbox.armed)
    UNLIKELY { bbox_add_vertices(data + pos, vtxCount, vtxSize, fmt); }

  // Native vertex fetch expands supported layouts to hardware attributes on the CPU
  // (advancing pos itself). Unsupported layouts fall through to the storage path.
  if (try_native_draw(prim, fmt, vtxCount, data, pos, vtxSize)) {
    return;
  }

  auto* lastDraw = !g_gxState.stateDirty ? gfx::get_last_draw_command<DrawData>() : nullptr;
  // A storage draw must never merge into a native draw's range (incompatible buffer layout).
  const bool canMerge = lastDraw != nullptr && prim != GX_LINES && prim != GX_LINESTRIP && prim != GX_POINTS &&
                        lastDraw->instanceCount == 1 && !lastDraw->nativeVertexFetch;

  // Push raw vertex data to buffer. Merged draws must remain byte-contiguous with the previous range.
  gfx::Range vertRange = gfx::push_verts(data + pos, totalVtxBytes, canMerge ? 0 : 4);
  pos += totalVtxBytes;

  // Try to merge with previous draw call
  if (canMerge) {
    // Only if the previous draw call was a single instance draw (no lines/points handling)
    u32 numIndices = 0;
    gfx::Range idxRange;
    const bool hadIndexRange = lastDraw->idxRange.size != 0;
    if (lastDraw->indexCount == 0 && prim != GX_TRIANGLES) {
      // Generate triangle index buffer for previous draw
      lastDraw->indexCount = prepare_idx_buffer(handle_draw_idx_buf, GX_TRIANGLES, 0, lastDraw->vtxCount);
    }
    if (lastDraw->indexCount != 0) {
      numIndices += prepare_idx_buffer(handle_draw_idx_buf, prim, lastDraw->vtxCount, vtxCount);
      idxRange = gfx::push_indices(handle_draw_idx_buf.data(), handle_draw_idx_buf.size(), hadIndexRange ? 0 : 4);
      handle_draw_idx_buf.clear();
    }
    CHECK(lastDraw->vertRange.offset + lastDraw->vertRange.size == vertRange.offset,
          "Non-consecutive vertex ranges ({} < {})", lastDraw->vertRange.offset + lastDraw->vertRange.size,
          vertRange.offset);
    if (hadIndexRange) {
      CHECK(lastDraw->idxRange.offset + lastDraw->idxRange.size == idxRange.offset,
            "Non-consecutive index ranges ({} < {})", lastDraw->idxRange.offset + lastDraw->idxRange.size,
            idxRange.offset);
    }
    lastDraw->vertRange.size += vertRange.size;
    if (lastDraw->idxRange.size == 0) {
      lastDraw->idxRange = idxRange;
    } else {
      lastDraw->idxRange.size += idxRange.size;
    }
    lastDraw->vtxCount += vtxCount;
    lastDraw->indexCount += numIndices;
    ++gfx::g_mergedDrawCallCount;
    return;
  }

  handle_draw_unmerged(prim, fmt, vtxCount, vertRange);
}

static void handle_draw(u8 cmd, const u8* data, u32& pos, u32 size, bool bigEndian) {
  GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
  GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);

  CHECK(pos + 2 <= size, "draw vtxCount read overrun");
  u16 vtxCount = read_u16(data + pos, bigEndian);
  pos += 2;

  u32 vtxSize;
  if (g_gxState.lastVtxFmt == fmt)
    LIKELY { vtxSize = g_gxState.lastVtxSize; }
  else
    UNLIKELY { vtxSize = calculate_last_vtx_size(fmt); }

  // Fold this draw plus any immediately-following identical draw commands into one native
  // hardware draw (strip topology + primitive restart). Unsupported primitives/layouts fall
  // through to draw_prim (single native, else the storage/software path).
  if (!sBbox.armed && try_native_draw_run(cmd, prim, fmt, vtxCount, vtxSize, data, pos, size, bigEndian)) {
    return;
  }

  draw_prim(prim, fmt, vtxCount, data, pos, size);
}

static ByteBuffer handle_draw_unmerged_idxBuf;

static void handle_draw_unmerged(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange) {
  ZoneScoped;
  u32 numIndices = 0;
  gfx::Range idxRange;

  if (prim != GX_TRIANGLES) {
    ZoneScopedN("build idx buffer");
    auto& realBuf = handle_draw_unmerged_idxBuf;
    numIndices = prepare_idx_buffer(realBuf, prim, 0, vtxCount);
    idxRange = gfx::push_indices(realBuf.data(), realBuf.size(), 4);
    realBuf.clear();
  }

  push_gx_draw(prim, fmt, vtxCount, vertRange, idxRange, numIndices);
}

static void push_gx_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange, gfx::Range idxRange,
                         u32 numIndices) {
  if constexpr (kSkipStorageVertexFetch) {
    // Native fetch declined this layout; the storage fallback would bind vertex SSBOs
    // and can't link on Mali. The caller already consumed the FIFO vertices, so just
    // drop the draw. Warn a few times so we know what geometry is being skipped.
    static Module Log("aurora::gx");
    static int warned = 0;
    if (warned++ < 8) {
      Log.warn("Dropping non-native draw (prim={} fmt={} verts={}) — storage vertex fetch unavailable on Mali",
               static_cast<int>(prim), static_cast<int>(fmt), vtxCount);
    }
    return;
  }
  // Build pipeline, bind groups, and push draw command
  BindGroupRanges ranges{};
  for (int i = GX_VA_POS; i <= GX_VA_TEX7; ++i) {
    if (g_gxState.vtxDesc[i] != GX_INDEX8 && g_gxState.vtxDesc[i] != GX_INDEX16) {
      continue;
    }
    auto& array = g_gxState.arrays[i];
    if (array.cachedRange.size > 0) {
      ranges.vaRanges[i - GX_VA_POS] = array.cachedRange;
    } else {
      const auto range = gfx::push_storage(static_cast<const uint8_t*>(array.data), array.size);
      ranges.vaRanges[i - GX_VA_POS] = range;
      array.cachedRange = range;
    }
  }

  PipelineConfig config{};
  populate_pipeline_config(config, prim, fmt);
  struct PipelineMemoEntry {
    PipelineConfig config{};
    ShaderInfo info{};
    gfx::PipelineRef pipeline{};
    bool used = false;
  };
  static PipelineMemoEntry s_pipeMemo[64]{};
  const uint64_t cfgHash = xxh3_hash(config);
  const uint32_t slot = cfgHash & 63;

  ShaderInfo info;
  gfx::PipelineRef pipeline;
  if (s_pipeMemo[slot].used && std::memcmp(&config, &s_pipeMemo[slot].config, sizeof(PipelineConfig)) == 0) {
    info = s_pipeMemo[slot].info;
    pipeline = s_pipeMemo[slot].pipeline;
  } else {
    info = build_shader_info(config.shaderConfig);
    pipeline = gfx::pipeline_ref(config);
    s_pipeMemo[slot].config = config;
    s_pipeMemo[slot].info = info;
    s_pipeMemo[slot].pipeline = pipeline;
    s_pipeMemo[slot].used = true;
  }

  resolve_sampled_textures(info);
  const auto bindGroups = build_bind_groups(info);

  uint32_t instanceCount = 1;
  if (prim == GX_LINES) {
    instanceCount = vtxCount / 2;
  } else if (prim == GX_LINESTRIP) {
    instanceCount = vtxCount - 1;
  } else if (prim == GX_POINTS) {
    instanceCount = vtxCount;
  }
  gfx::push_draw_command(DrawData{
      .pipeline = pipeline,
      .vertRange = vertRange,
      .idxRange = idxRange,
      .uniformRange = build_uniform(info, vertRange.offset, ranges),
      .vtxCount = vtxCount,
      .indexCount = numIndices,
      .instanceCount = instanceCount,
      .bindGroups = bindGroups,
      .dstAlpha = g_gxState.dstAlpha,
  });
}

// ---------------------------------------------------------------------------
// Native vertex fetch.
//
// The storage path software-fetches GX vertices from SSBOs, which (a) miscompiles
// the per-vertex matrix-palette index into slot 0 on AGX/Mali (the "porcupine")
// and (b) cannot link on Mali (0 vertex-stage SSBOs). Here we expand each GX vertex
// on the CPU into real hardware vertex attributes — most importantly pnmtxidx
// becomes a plain Uint32 attribute, so it never flows through the miscompiled
// byte-load — and let the fixed-function vertex input path feed the shader.
// Skinning/projection stay on the GPU. Layouts native fetch can't express fall
// through to the storage path per-draw (try_native_draw* return false).
// ---------------------------------------------------------------------------

// Canonical hardware-attribute byte width per GX attr (must match native_vertex_layout).
static u8 canonical_attr_size(GXAttr attr) noexcept {
  if (attr == GX_VA_PNMTXIDX || (attr >= GX_VA_TEX0MTXIDX && attr <= GX_VA_TEX7MTXIDX)) {
    return 4; // Uint32
  }
  if (attr == GX_VA_POS || attr == GX_VA_NRM) {
    return 12; // Float32x3
  }
  if (attr == GX_VA_CLR0 || attr == GX_VA_CLR1) {
    return 4; // Unorm8x4
  }
  return 8; // TEXn: Float32x2
}

static bool native_component_decodable(GXCompType type) noexcept {
  switch (type) {
  case GX_U8:
  case GX_S8:
  case GX_U16:
  case GX_S16:
  case GX_F32:
    return true;
  default:
    return false;
  }
}

static bool native_color_decodable(GXCompType type) noexcept {
  switch (type) {
  case GX_RGB565:
  case GX_RGB8:
  case GX_RGBX8:
  case GX_RGBA4:
  case GX_RGBA6:
  case GX_RGBA8:
    return true;
  default:
    return false;
  }
}

static f32 read_component_as_float(const u8* p, GXCompType type, u8 frac, bool bigEndian) noexcept {
  const f32 scale = 1.0f / static_cast<f32>(1u << frac);
  switch (type) {
  case GX_U8:
    return static_cast<f32>(*p) * scale;
  case GX_S8:
    return static_cast<f32>(static_cast<int8_t>(*p)) * scale;
  case GX_U16:
    return static_cast<f32>(read_u16(p, bigEndian)) * scale;
  case GX_S16:
    return static_cast<f32>(static_cast<int16_t>(read_u16(p, bigEndian))) * scale;
  case GX_F32:
    return std::bit_cast<f32>(read_u32(p, bigEndian));
  default:
    return 0.0f;
  }
}

static u32 expand_bits_to_8(u32 v, u32 bits) noexcept {
  const u32 shifted = v << (8u - bits);
  return shifted | (shifted >> bits);
}

// Decode a GX color to a packed u32 (byte 0 = R, byte 3 = A) for a Unorm8x4 attribute.
static u32 read_color_as_rgba8(const u8* p, GXCompType type, bool bigEndian) noexcept {
  const auto pack = [](u32 r, u32 g, u32 b, u32 a) { return r | (g << 8) | (b << 16) | (a << 24); };
  switch (type) {
  case GX_RGB8:
  case GX_RGBX8:
    return pack(p[0], p[1], p[2], 0xFFu);
  case GX_RGBA8:
    return pack(p[0], p[1], p[2], p[3]);
  case GX_RGB565: {
    const u16 v = read_u16(p, bigEndian);
    return pack(expand_bits_to_8((v >> 11) & 0x1Fu, 5), expand_bits_to_8((v >> 5) & 0x3Fu, 6),
                expand_bits_to_8(v & 0x1Fu, 5), 0xFFu);
  }
  case GX_RGBA4: {
    const u16 v = read_u16(p, bigEndian);
    return pack(expand_bits_to_8((v >> 12) & 0xFu, 4), expand_bits_to_8((v >> 8) & 0xFu, 4),
                expand_bits_to_8((v >> 4) & 0xFu, 4), expand_bits_to_8(v & 0xFu, 4));
  }
  case GX_RGBA6: {
    const u32 v = bigEndian ? (u32(p[0]) << 16) | (u32(p[1]) << 8) | u32(p[2])
                            : (u32(p[2]) << 16) | (u32(p[1]) << 8) | u32(p[0]);
    return pack(expand_bits_to_8((v >> 18) & 0x3Fu, 6), expand_bits_to_8((v >> 12) & 0x3Fu, 6),
                expand_bits_to_8((v >> 6) & 0x3Fu, 6), expand_bits_to_8(v & 0x3Fu, 6));
  }
  default:
    return 0xFFFFFFFFu;
  }
}

// One present GX attr in a native-expanded vertex: how to read it from the FIFO
// record / indexed array, and (via canonical order) where it lands in the output.
struct NativeAttrDesc {
  GXAttr attr;
  u8 attrType;  // GX_DIRECT / GX_INDEX8 / GX_INDEX16
  u8 srcOffset; // byte offset of this attr (direct) or its index (indexed) in the FIFO record
  u8 compType;  // GXCompType of the source component(s)
  u8 cnt;       // source component count
  u8 compSize;  // bytes per source component
  u8 frac;      // fixed-point fraction bits
  u8 arrayStride;
  bool le; // source endianness (direct FIFO = false = big-endian)
};

// Build the native attribute layout for the current vertex format. Returns false
// (declines to the storage path) for anything native cannot faithfully express:
// NBT normals, exotic component types, missing position, >16 attributes, or an
// indexed attr whose source array is not resident.
// Diagnostic: flip kLogNativeDecline to true to log (rate-limited) why a layout
// declined native fetch and fell back to storage. Storage is impossible on Mali
// (GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS=0), so every decline logged here is a draw
// that gets dropped on Mali (see kSkipStorageVertexFetch) — the fix, if it matters,
// is to widen native fetch, not texture fetch. Compile-time so there's no env cruft.
static constexpr bool kLogNativeDecline = true;
static bool native_decline(int reason, GXVtxFmt fmt, int attr) {
  if constexpr (kLogNativeDecline) {
    static Module Log("aurora::gx");
    static const char* const reasons[8] = {"NBT-normal",      "undecodable-type", "nonresident-index-array",
                                            "non-direct-type", ">16-attrs",        "no-position",
                                            "?",               "?"};
    // Dedup by unique declined layout so the boot flood can't exhaust a fixed budget
    // before an in-game dialog/HUD layout first appears — each distinct layout logs once.
    static std::set<std::string> seen;
    const auto& vf = g_gxState.vtxFmts[fmt];
    std::string layout;
    for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
      const auto t = g_gxState.vtxDesc[i];
      if (t == GX_NONE) {
        continue;
      }
      const auto& af = vf.attrs[i];
      layout += fmt::format(" a{}:desc{}/type{}/cnt{}/frac{}", i, static_cast<int>(t), static_cast<int>(af.type),
                            static_cast<int>(af.cnt), static_cast<int>(af.frac));
      if (t == GX_INDEX8 || t == GX_INDEX16) {
        const auto& arr = g_gxState.arrays[i];
        layout += fmt::format("[arrData={} sz={} str={}]", arr.data != nullptr, arr.size, arr.stride);
      }
    }
    const std::string key = fmt::format("r{}f{}{}", reason, static_cast<int>(fmt), layout);
    if (seen.size() < 64 && seen.insert(key).second) {
      Log.warn("[decline-diag] reason={} fmt={} badAttr={} |{}", reasons[(reason < 0 || reason > 7) ? 6 : reason],
               static_cast<int>(fmt), attr, layout);
    }
  }
  return false;
}

static bool build_native_layout(GXVtxFmt fmt, std::vector<NativeAttrDesc>& descs, u8& nativeStride,
                                std::array<AttrConfig, MaxVtxAttr>& nativeAttrs) {
  const auto& vtxFmt = g_gxState.vtxFmts[fmt];
  descs.clear();
  nativeAttrs = {};
  u8 srcOffset = 0;
  u8 outOffset = 0;
  bool hasPos = false;
  int count = 0;
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    const auto attr = static_cast<GXAttr>(i);
    const auto type = g_gxState.vtxDesc[i];
    if (type == GX_NONE) {
      continue;
    }
    const auto& attrFmt = vtxFmt.attrs[i];
    const auto cnt = comp_cnt_count(attr, attrFmt.cnt);
    const bool isMtxIdx = attr == GX_VA_PNMTXIDX || (attr >= GX_VA_TEX0MTXIDX && attr <= GX_VA_TEX7MTXIDX);
    const bool isColor = attr == GX_VA_CLR0 || attr == GX_VA_CLR1;
    // Native fetch supports standard 3-component normals (cnt 3) and NBT/NBT3 (cnt 9).
    // The shader samples the primary normal (first 3 components) for lighting and reflection.
    if (!isMtxIdx) {
      const auto compType = static_cast<GXCompType>(attrFmt.type);
      if (isColor ? !native_color_decodable(compType) : !native_component_decodable(compType)) {
        return native_decline(1, fmt, i);
      }
    }

    NativeAttrDesc d{};
    d.attr = attr;
    d.attrType = static_cast<u8>(type);
    d.srcOffset = srcOffset;
    d.compType = static_cast<u8>(attrFmt.type);
    d.cnt = cnt;
    d.frac = attrFmt.frac;
    if (type == GX_DIRECT) {
      d.le = false;
      d.compSize = isMtxIdx ? 1 : comp_type_size(attr, attrFmt.type);
      srcOffset += isMtxIdx ? 1 : static_cast<u8>(comp_type_size(attr, attrFmt.type) * cnt);
    } else if (type == GX_INDEX8 || type == GX_INDEX16) {
      const auto& array = g_gxState.arrays[i];
      if (array.data == nullptr || array.stride == 0) {
        return native_decline(2, fmt, i);
      }
      d.le = array.le;
      d.arrayStride = array.stride;
      d.compSize = comp_type_size(attr, attrFmt.type);
      if (attr == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3) {
        srcOffset += (type == GX_INDEX8) ? 3 : 6;
      } else {
        srcOffset += (type == GX_INDEX8) ? 1 : 2;
      }
    } else {
      return native_decline(3, fmt, i);
    }

    auto& na = nativeAttrs[i];
    na.attrType = GX_DIRECT;
    na.offset = outOffset;
    na.cnt = isMtxIdx ? 1 : (isColor ? 1 : (attr == GX_VA_POS || attr == GX_VA_NRM ? 3 : 2));
    na.compType = static_cast<u8>(attrFmt.type);
    outOffset += canonical_attr_size(attr);
    descs.push_back(d);
    if (attr == GX_VA_POS) {
      hasPos = true;
    }
    if (++count > 16) {
      return native_decline(4, fmt, i);
    }
  }
  if (!hasPos) {
    return native_decline(5, fmt, GX_VA_POS);
  }
  nativeStride = outOffset;
  return true;
}

struct CachedNativeLayout {
  u32 vatGen = UINT32_MAX;
  bool success = false;
  u8 nativeStride = 0;
  std::array<AttrConfig, MaxVtxAttr> nativeAttrs{};
  std::vector<NativeAttrDesc> descs;
};
static std::array<CachedNativeLayout, 8> s_cachedNativeLayouts{};

static bool get_native_layout(GXVtxFmt fmt, const std::vector<NativeAttrDesc>*& outDescs, u8& nativeStride,
                              std::array<AttrConfig, MaxVtxAttr>& nativeAttrs) {
  auto& c = s_cachedNativeLayouts[static_cast<size_t>(fmt)];
  if (c.vatGen != s_vatGen) {
    c.descs.clear();
    c.success = build_native_layout(fmt, c.descs, c.nativeStride, c.nativeAttrs);
    c.vatGen = s_vatGen;
  }
  if (!c.success) return false;
  outDescs = &c.descs;
  nativeStride = c.nativeStride;
  nativeAttrs = c.nativeAttrs;
  return true;
}

// Expand one GX vertex (FIFO record at `rec`) into hardware attributes.
static void expand_native_vertex(const std::vector<NativeAttrDesc>& descs, const u8* rec, ByteBuffer& out) {
  for (const auto& d : descs) {
    const u8* p;
    if (d.attrType == GX_DIRECT) {
      p = rec + d.srcOffset;
    } else {
      u32 index = d.attrType == GX_INDEX8 ? u32(*(rec + d.srcOffset)) : u32(read_u16(rec + d.srcOffset, true));
      const auto& array = g_gxState.arrays[d.attr];
      if (array.size != 0 && d.arrayStride != 0) {
        const u32 maxIndex = static_cast<u32>(array.size) / d.arrayStride;
        if (index >= maxIndex) {
          index = 0; // guard against malformed indices; valid content never hits this
        }
      }
      p = static_cast<const u8*>(array.data) + static_cast<size_t>(index) * d.arrayStride;
    }
    const bool be = !d.le;
    const auto attr = d.attr;
    const auto compType = static_cast<GXCompType>(d.compType);
    if (attr == GX_VA_PNMTXIDX) {
      out.append<u32>(u32(*p) / 3u); // /3: GX counts matrix rows; shader indexes postex_mtx[in_pnmtxidx] directly
    } else if (attr >= GX_VA_TEX0MTXIDX && attr <= GX_VA_TEX7MTXIDX) {
      out.append<u32>(u32(*p)); // raw; shader divides by 3
    } else if (attr == GX_VA_POS || attr == GX_VA_NRM) {
      for (int c = 0; c < 3; ++c) {
        out.append<f32>(c < d.cnt ? read_component_as_float(p + c * d.compSize, compType, d.frac, be) : 0.0f);
      }
    } else if (attr == GX_VA_CLR0 || attr == GX_VA_CLR1) {
      out.append<u32>(read_color_as_rgba8(p, compType, be));
    } else { // TEXn
      for (int c = 0; c < 2; ++c) {
        out.append<f32>(c < d.cnt ? read_component_as_float(p + c * d.compSize, compType, d.frac, be) : 0.0f);
      }
    }
  }
}

static ByteBuffer s_nativeVtxBuf;
static ByteBuffer s_nativeIdxBuf;
static std::vector<NativeAttrDesc> s_nativeDescs;

// ---------------------------------------------------------------------------
// Native geometry cache (M2)
//
// CPU-expanded native vertex/index data is uploaded once into persistent GPU buffers and
// replayed on later frames by content hash. Because skinning stays on the GPU (bone
// matrices are per-draw uniforms, not baked into the verts), a mesh's expanded bytes are
// byte-identical every frame — so steady-state world and character geometry expands +
// uploads exactly once, and every subsequent frame is just a hash lookup plus a draw that
// references the cached range. This is the fix for the "M1: no cache" per-frame re-expand
// + re-upload cost. Keyed by the source FIFO bytes plus the decode layout (and, for
// indexed attributes, the referenced arrays' contents) so two draws collide in the cache
// only when they truly expand to identical geometry.
static constexpr bool kNativeGeomCacheEnabled = true;
// Byte floor below which caching isn't worth the hash/bookkeeping. NOTE: the 2-frame
// recurrence gate in native_geom_resolve_ranges (not this size gate) is what actually keeps
// transient one-shot draws out of the persistent cache, so this only needs to skip trivially
// tiny draws. Lowered 1024 -> 64 so the shadow pre-pass (16B/vert x 4-21 verts = 64-336B) and
// small static props become cacheable and stop re-uploading into the per-frame vertex ring
// every frame -- that ring was ~440KB/frame, ~38% of the per-frame libmali upload volume that
// is the device CPU bottleneck.
static constexpr size_t kNativeGeomCacheMinBytes = 64;

// Keep triangle strips as strips on the GPU (TriangleStrip topology + primitive restart)
// so a run of adjacent strip draws collapses to one DrawIndexed, instead of unrolling
// each into an independent triangle list. This is the top CPU-side draw-submission lever:
// far fewer draw commands and index bytes in strip-heavy world scenes.
static constexpr bool kStripTopology = true;

// Emit a periodic one-line cache/batching summary to confirm the geometry cache is
// actually hitting on-device (and how effective run-batching is). Perf-tuning scaffolding;
// off by default in shipped builds.
static constexpr bool kLogNativeGeomCacheStats = true;
static constexpr u32 kNativeGeomStatsLogInterval = 30; // frames between summaries
// Per-window counters (reset each log interval).
static u64 s_geomCacheHits = 0;       // draws served from the persistent cache (no expand/upload)
static u64 s_geomCacheMisses = 0;     // draws that had to expand this frame
static u64 s_geomCachePromotes = 0;   // expansions promoted into the persistent cache
static u64 s_geomCacheRingPushes = 0; // expansions that went to the per-frame ring
static u64 s_geomCacheFull = 0;       // promotions that failed (cache exhausted -> reset)
static u64 s_geomRunsBatched = 0;     // native strip/prim runs emitted (raw-FIFO handle_draw)
static u64 s_geomSegmentsBatched = 0; // draw commands folded into those runs
static u64 s_geomDrawsIndexed = 0;    // draws via GX_AURORA_DRAW_INDEXED (dl::optimize output)
static u64 s_geomDrawsSingle = 0;     // draws via GX_AURORA_DRAW_SIZED (single native)
static u32 s_geomStatsLastFrame = 0;

struct NativeGeomKey {
  u64 fifoHash = 0;
  u64 metaHash = 0;
  u32 srcBytes = 0;
  bool operator==(const NativeGeomKey& o) const noexcept {
    return fifoHash == o.fifoHash && metaHash == o.metaHash && srcBytes == o.srcBytes;
  }
};
struct NativeGeomKeyHash {
  size_t operator()(const NativeGeomKey& k) const noexcept {
    return static_cast<size_t>(k.fifoHash ^ (k.metaHash * 0x9e3779b97f4a7c15ull) ^ k.srcBytes);
  }
};
struct NativeGeomEntry {
  gfx::Range vertRange;
  gfx::Range idxRange;
  u32 indexCount = 0;
};
static absl::flat_hash_map<NativeGeomKey, NativeGeomEntry, NativeGeomKeyHash> s_nativeGeomCache;
// Keys seen but not yet promoted, mapped to the frame they were last seen in. Content is
// promoted to the persistent cache only once it recurs across frames, which keeps
// single-shot and per-frame-animated geometry from ever entering the cache.
static absl::flat_hash_map<NativeGeomKey, u32, NativeGeomKeyHash> s_nativeGeomSeen;
static u32 s_nativeGeomCacheGen = 0;

// Emit a cache/batching summary every kNativeGeomStatsLogInterval frames. Called once
// per native draw attempt; the frame-delta guard keeps it to one line per interval.
static void native_geom_cache_maybe_log_stats() {
  if constexpr (!kLogNativeGeomCacheStats) {
    return;
  }
  const u32 frame = gfx::current_frame();
  if (frame == UINT32_MAX || frame - s_geomStatsLastFrame < kNativeGeomStatsLogInterval) {
    return;
  }
  const u32 frames = frame - s_geomStatsLastFrame;
  s_geomStatsLastFrame = frame;
  const u64 draws = s_geomCacheHits + s_geomCacheMisses;
  Log.info(
      "[geom-cache] {}f: {} draws, {}/{} hit ({:.1f}%), {} promote, {} ring, {} full; "
      "src {} run/{} indexed/{} sized, batched {}<-{} segs ({:.2f}/run); {} entries",
      frames, draws, s_geomCacheHits, draws, draws != 0 ? 100.0 * static_cast<double>(s_geomCacheHits) / draws : 0.0,
      s_geomCachePromotes, s_geomCacheRingPushes, s_geomCacheFull, s_geomRunsBatched, s_geomDrawsIndexed,
      s_geomDrawsSingle, s_geomRunsBatched, s_geomSegmentsBatched,
      s_geomRunsBatched != 0 ? static_cast<double>(s_geomSegmentsBatched) / s_geomRunsBatched : 0.0,
      s_nativeGeomCache.size());
  const u64 totalUni = s_uniformCacheHits + s_uniformCacheMisses;
  Log.info(
      "[uniform-cache] {} draws, {}/{} hit ({:.1f}%)",
      totalUni, s_uniformCacheHits, totalUni,
      totalUni != 0 ? 100.0 * static_cast<double>(s_uniformCacheHits) / totalUni : 0.0);
  s_uniformCacheHits = 0;
  s_uniformCacheMisses = 0;
  // Per-frame staging->GPU upload volume (last completed frame). This is what feeds the libmali
  // copy worker: ring verts/indices (cache misses), uniforms (every draw, no dedup), and texture
  // uploads. Cached geometry lives in a persistent buffer and does NOT appear in lastVertSize.
  const double kib = 1.0 / 1024.0;
  const u64 uploadTotal = static_cast<u64>(gfx::g_stats.lastVertSize) + gfx::g_stats.lastIndexSize +
                          gfx::g_stats.lastUniformSize + gfx::g_stats.lastTextureUploadSize +
                          gfx::g_stats.lastStorageSize;
  Log.info("[upload] last-frame: vert {:.1f}K, idx {:.1f}K, uni {:.1f}K, tex {:.1f}K, storage {:.1f}K -> total {:.1f}K",
           gfx::g_stats.lastVertSize * kib, gfx::g_stats.lastIndexSize * kib,
           gfx::g_stats.lastUniformSize * kib, gfx::g_stats.lastTextureUploadSize * kib,
           gfx::g_stats.lastStorageSize * kib, static_cast<double>(uploadTotal) * kib);
  s_geomCacheHits = 0;
  s_geomCacheMisses = 0;
  s_geomCachePromotes = 0;
  s_geomCacheRingPushes = 0;
  s_geomCacheFull = 0;
  s_geomRunsBatched = 0;
  s_geomSegmentsBatched = 0;
  s_geomDrawsIndexed = 0;
  s_geomDrawsSingle = 0;
}

// The persistent cache byte regions are recycled on reset (generation bump), which
// invalidates every cached (key -> range) mapping. Drop both maps when that happens.
static void native_geom_sync_generation() {
  native_geom_cache_maybe_log_stats();
  const u32 gen = gfx::native_geom_cache_generation();
  if (gen != s_nativeGeomCacheGen) {
    s_nativeGeomCacheGen = gen;
    s_nativeGeomCache.clear();
    s_nativeGeomSeen.clear();
  }
}

// Content hash of an indexed attribute's source array, memoized per (pointer, frame,
// invalidate gen). Frame-scoped so arrays the game rewrites in place between frames (CPU
// envelope skinning deforms positions into the same buffers) are re-hashed each frame;
// GXInvalidateVtxCache handles rewrites within a frame.
struct ArrayHashMemo {
  u32 frame = UINT32_MAX;
  u32 invalGen = UINT32_MAX;
  u32 size = 0;
  u64 hash = 0;
};
static absl::flat_hash_map<const void*, ArrayHashMemo> s_arrayHashMemo;

static u64 native_array_content_hash(const AttrArray& array) {
  if (s_arrayHashMemo.size() > 1024) {
    s_arrayHashMemo.clear();
  }
  const u32 frame = gfx::current_frame();
  auto [it, inserted] = s_arrayHashMemo.try_emplace(array.data);
  auto& memo = it->second;
  if (inserted || memo.frame != frame || memo.size != array.size) {
    memo.frame = frame;
    memo.invalGen = s_gxArrayInvalidateGen;
    memo.size = array.size;
    if (array.size == 0) {
      memo.hash = xxh3_hash(reinterpret_cast<uintptr_t>(array.data));
    } else {
      memo.hash = xxh3_hash_s(array.data, array.size);
    }
  }
  return memo.hash;
}

// Fast hash for geometry meta (layout + counts) avoiding Hasher heap/state overhead
static inline u64 hash_native_geom_meta(GXPrimitive prim, GXVtxFmt fmt, u32 vtxSize, u32 vtxCount,
                                        const std::vector<NativeAttrDesc>& descs,
                                        const void* extraData = nullptr, size_t extraBytes = 0) {
  struct MetaHdr {
    u32 primFmt;
    u32 vtxSize;
    u32 vtxCount;
  } hdr{
    .primFmt = (static_cast<u32>(prim) << 16) | static_cast<u32>(fmt),
    .vtxSize = vtxSize,
    .vtxCount = vtxCount,
  };
  u64 meta = XXH3_64bits(&hdr, sizeof(hdr));
  if (!descs.empty()) {
    meta = XXH3_64bits_withSeed(descs.data(), descs.size() * sizeof(NativeAttrDesc), meta);
    for (const auto& d : descs) {
      if (d.attrType == GX_INDEX8 || d.attrType == GX_INDEX16) {
        const u64 arrHash = native_array_content_hash(g_gxState.arrays[d.attr]);
        meta = xxh3_hash_s(&arrHash, sizeof(arrHash), meta);
      }
    }
  }
  if (extraData != nullptr && extraBytes > 0) {
    meta = XXH3_64bits_withSeed(extraData, extraBytes, meta);
  }
  return meta;
}

// Fold the decode-relevant layout into the meta hasher: which attrs are present, their
// formats, and for indexed attrs the referenced array's per-frame content hash. Two draws
// with equal FIFO bytes and equal meta hash expand to identical native vertices.
static void native_geom_hash_layout(Hasher& h, const std::vector<NativeAttrDesc>& descs) {
  for (const auto& d : descs) {
    h.update(static_cast<u8>(d.attr));
    h.update(d.attrType);
    h.update(d.compType);
    h.update(d.cnt);
    h.update(d.compSize);
    h.update(d.frac);
    h.update(d.arrayStride);
    h.update(static_cast<u8>(d.le ? 1 : 0));
    if (d.attrType == GX_INDEX8 || d.attrType == GX_INDEX16) {
      h.update(native_array_content_hash(g_gxState.arrays[d.attr]));
    }
  }
}

// Resolve the GPU ranges for a freshly expanded batch. On the first cross-frame recurrence
// of identical content the batch is promoted into the persistent cache; otherwise it goes
// to the per-frame ring. `cached` reports which buffers the returned ranges address.
// `idxData`/`idxBytes` may be null/0 for non-indexed (triangle-list) draws.
static void native_geom_resolve_ranges(const NativeGeomKey& key, const ByteBuffer& vtxBytes, const u8* idxData,
                                       size_t idxBytes, u32 numIndices, gfx::Range& vertRange, gfx::Range& idxRange,
                                       bool& cached) {
  cached = false;
  if (kNativeGeomCacheEnabled && vtxBytes.size() >= kNativeGeomCacheMinBytes) {
    const u32 frame = gfx::current_frame();
    auto [seenIt, inserted] = s_nativeGeomSeen.emplace(key, frame);
    if (!inserted && seenIt->second != frame) {
      // Content recurred across frames — promote into the persistent cache.
      auto [cv, vok] = gfx::push_native_cached_verts(vtxBytes.data(), vtxBytes.size());
      gfx::Range ci{};
      bool iok = true;
      if (idxData != nullptr && idxBytes > 0) {
        const auto pushed = gfx::push_native_cached_indices(idxData, idxBytes);
        ci = pushed.first;
        iok = pushed.second;
      }
      if (vok && iok) {
        vertRange = cv;
        idxRange = ci;
        cached = true;
        s_nativeGeomCache.emplace(key, NativeGeomEntry{.vertRange = cv, .idxRange = ci, .indexCount = numIndices});
        s_nativeGeomSeen.erase(seenIt);
        ++s_geomCachePromotes;
        return;
      }
      // Cache exhausted — recycle it at the next frame boundary and use the ring now.
      ++s_geomCacheFull;
      gfx::request_native_geometry_cache_reset();
    } else if (!inserted) {
      seenIt->second = frame;
    }
    if (s_nativeGeomSeen.size() > 65536) {
      s_nativeGeomSeen.clear();
    }
  }
  // Per-frame ring fallback.
  ++s_geomCacheRingPushes;
  vertRange = gfx::push_verts(vtxBytes.data(), vtxBytes.size(), 4);
  if (idxData != nullptr && idxBytes > 0) {
    idxRange = gfx::push_indices(idxData, idxBytes, 4);
  }
}

static void push_native_gx_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount,
                                 const std::array<AttrConfig, MaxVtxAttr>& nativeAttrs, u8 nativeStride,
                                 gfx::Range vertRange, gfx::Range idxRange, u32 numIndices, bool cachedGeometry = false,
                                 bool stripTopology = false) {
  static uint32_t s_lastPipelineDirty = UINT32_MAX;
  static GXPrimitive s_lastPrim = static_cast<GXPrimitive>(UINT32_MAX);
  static GXVtxFmt s_lastFmt = static_cast<GXVtxFmt>(UINT32_MAX);
  static bool s_lastStripTopology = false;
  static uint8_t s_lastNativeStride = 0;
  static u32 s_lastUniformFrame = UINT32_MAX;

  const bool samePipeline = s_hasLastDrawState &&
                            (s_lastPipelineDirty == g_gxState.pipelineDirty) &&
                            (s_lastPrim == prim) &&
                            (s_lastFmt == fmt) &&
                            (s_lastStripTopology == stripTopology) &&
                            (s_lastNativeStride == nativeStride);

  static PipelineConfig s_currentConfig{};
  const PipelineConfig* configPtr = nullptr;
  if (samePipeline) {
    configPtr = &s_lastConfig;
  } else {
    const auto tp0 = std::chrono::steady_clock::now();
    populate_pipeline_config(s_currentConfig, prim, fmt); // TEV/color/blend/etc; also fills storage attrs (overwritten below)
    s_currentConfig.shaderConfig.attrs = nativeAttrs;
    s_currentConfig.shaderConfig.vtxStride = nativeStride;
    s_currentConfig.shaderConfig.nativeVertexFetch = 1;
    s_currentConfig.triangleStripTopology = stripTopology ? 1u : 0u;
    const auto tp1 = std::chrono::steady_clock::now();
    s_drainTiming.ms_pipe_cfg += std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    s_lastPipelineDirty = g_gxState.pipelineDirty;
    s_lastPrim = prim;
    s_lastFmt = fmt;
    s_lastStripTopology = stripTopology;
    s_lastNativeStride = nativeStride;
    configPtr = &s_currentConfig;
  }

  ShaderInfo info;
  gfx::PipelineRef pipeline;
  GXBindGroups bindGroups;
  gfx::Range uniformRange;
  const u32 curFrame = gfx::current_frame();

  const bool sameConfig = samePipeline || (s_hasLastDrawState && std::memcmp(configPtr, &s_lastConfig, sizeof(PipelineConfig)) == 0);
  if (sameConfig) {
    if (!samePipeline) {
      s_lastPipelineDirty = g_gxState.pipelineDirty;
      s_lastPrim = prim;
      s_lastFmt = fmt;
      s_lastStripTopology = stripTopology;
      s_lastNativeStride = nativeStride;
    }
    info = s_lastInfo;
    pipeline = s_lastPipeline;
    if (s_lastTextureDirty == g_gxState.textureDirty) {
      bindGroups = s_lastBindGroups;
    } else {
      const auto t_res0 = std::chrono::steady_clock::now();
      resolve_sampled_textures(info);
      const auto t_res1 = std::chrono::steady_clock::now();
      bindGroups = build_bind_groups(info);
      const auto t_bg1 = std::chrono::steady_clock::now();
      s_drainTiming.ms_tex_resolve += std::chrono::duration<double, std::milli>(t_res1 - t_res0).count();
      s_drainTiming.ms_bg_build += std::chrono::duration<double, std::milli>(t_bg1 - t_res1).count();
      ++s_drainTiming.bg_calls;
      s_lastBindGroups = bindGroups;
      s_lastTextureDirty = g_gxState.textureDirty;
    }
    if (s_lastUniformFrame == curFrame && s_lastUniformDirty == g_gxState.uniformDirty && s_lastTextureDirty == g_gxState.textureDirty && s_lastUniformRange.size > 0) {
      uniformRange = s_lastUniformRange;
      ++s_uniformCacheHits;
    } else {
      BindGroupRanges ranges{};
      const auto tu0 = std::chrono::steady_clock::now();
      uniformRange = build_uniform(info, vertRange.offset, ranges);
      const auto tu1 = std::chrono::steady_clock::now();
      s_drainTiming.ms_uni_build += std::chrono::duration<double, std::milli>(tu1 - tu0).count();
      s_lastUniformRange = uniformRange;
      s_lastUniformDirty = g_gxState.uniformDirty;
      s_lastUniformFrame = curFrame;
      ++s_uniformCacheMisses;
    }
  } else {
    const auto tr0 = std::chrono::steady_clock::now();
    pipeline = gfx::pipeline_ref(*configPtr);
    const auto tr1 = std::chrono::steady_clock::now();
    s_drainTiming.ms_pipe_ref += std::chrono::duration<double, std::milli>(tr1 - tr0).count();
    s_lastPipeline = pipeline;
    s_lastConfig = *configPtr;
    s_hasLastDrawState = true;

    const bool sameShader = std::memcmp(&configPtr->shaderConfig, &s_lastShaderConfig, sizeof(ShaderConfig)) == 0;
    if (sameShader) {
      info = s_lastInfo;
      if (s_lastTextureDirty == g_gxState.textureDirty) {
        bindGroups = s_lastBindGroups;
      } else {
        const auto t_res0 = std::chrono::steady_clock::now();
        resolve_sampled_textures(info);
        const auto t_res1 = std::chrono::steady_clock::now();
        bindGroups = build_bind_groups(info);
        const auto t_bg1 = std::chrono::steady_clock::now();
        s_drainTiming.ms_tex_resolve += std::chrono::duration<double, std::milli>(t_res1 - t_res0).count();
        s_drainTiming.ms_bg_build += std::chrono::duration<double, std::milli>(t_bg1 - t_res1).count();
        ++s_drainTiming.bg_calls;
        s_lastBindGroups = bindGroups;
        s_lastTextureDirty = g_gxState.textureDirty;
      }
      if (s_lastUniformFrame == curFrame && s_lastUniformDirty == g_gxState.uniformDirty && s_lastTextureDirty == g_gxState.textureDirty && s_lastUniformRange.size > 0) {
        uniformRange = s_lastUniformRange;
        ++s_uniformCacheHits;
      } else {
        BindGroupRanges ranges{};
        const auto tu0 = std::chrono::steady_clock::now();
        uniformRange = build_uniform(info, vertRange.offset, ranges);
        const auto tu1 = std::chrono::steady_clock::now();
        s_drainTiming.ms_uni_build += std::chrono::duration<double, std::milli>(tu1 - tu0).count();
        s_lastUniformRange = uniformRange;
        s_lastUniformDirty = g_gxState.uniformDirty;
        s_lastUniformFrame = curFrame;
        ++s_uniformCacheMisses;
      }
    } else {
      const uint64_t shaderHash = XXH3_64bits(&configPtr->shaderConfig, sizeof(ShaderConfig));
      const size_t slot = (shaderHash ^ (shaderHash >> 16)) & (kShaderDrawCacheSize - 1);
      auto& entry = s_shaderDrawCache[slot];

      if (entry.shaderHash == shaderHash) {
        info = entry.info;
        if (entry.textureDirty == g_gxState.textureDirty) {
          bindGroups = entry.bindGroups;
        } else {
          const auto t_res0 = std::chrono::steady_clock::now();
          resolve_sampled_textures(info);
          const auto t_res1 = std::chrono::steady_clock::now();
          bindGroups = build_bind_groups(info);
          const auto t_bg1 = std::chrono::steady_clock::now();
          s_drainTiming.ms_tex_resolve += std::chrono::duration<double, std::milli>(t_res1 - t_res0).count();
          s_drainTiming.ms_bg_build += std::chrono::duration<double, std::milli>(t_bg1 - t_res1).count();
          ++s_drainTiming.bg_calls;
          entry.bindGroups = bindGroups;
          entry.textureDirty = g_gxState.textureDirty;
        }
        if (entry.frame == curFrame && entry.uniformDirty == g_gxState.uniformDirty && entry.textureDirty == g_gxState.textureDirty && entry.uniformRange.size > 0) {
          uniformRange = entry.uniformRange;
          ++s_uniformCacheHits;
        } else {
          BindGroupRanges ranges{};
          const auto tu0 = std::chrono::steady_clock::now();
          uniformRange = build_uniform(info, vertRange.offset, ranges);
          const auto tu1 = std::chrono::steady_clock::now();
          s_drainTiming.ms_uni_build += std::chrono::duration<double, std::milli>(tu1 - tu0).count();
          entry.uniformRange = uniformRange;
          entry.uniformDirty = g_gxState.uniformDirty;
          entry.textureDirty = g_gxState.textureDirty;
          entry.frame = curFrame;
          ++s_uniformCacheMisses;
        }
      } else {
        info = build_shader_info(configPtr->shaderConfig);
        const auto t_res0 = std::chrono::steady_clock::now();
        resolve_sampled_textures(info);
        const auto t_res1 = std::chrono::steady_clock::now();
        bindGroups = build_bind_groups(info);
        const auto t_bg1 = std::chrono::steady_clock::now();
        s_drainTiming.ms_tex_resolve += std::chrono::duration<double, std::milli>(t_res1 - t_res0).count();
        s_drainTiming.ms_bg_build += std::chrono::duration<double, std::milli>(t_bg1 - t_res1).count();
        ++s_drainTiming.bg_calls;
        BindGroupRanges ranges{};
        const auto tu0 = std::chrono::steady_clock::now();
        uniformRange = build_uniform(info, vertRange.offset, ranges);
        const auto tu1 = std::chrono::steady_clock::now();
        s_drainTiming.ms_uni_build += std::chrono::duration<double, std::milli>(tu1 - tu0).count();

        entry.shaderHash = shaderHash;
        entry.info = info;
        entry.bindGroups = bindGroups;
        entry.uniformRange = uniformRange;
        entry.textureDirty = g_gxState.textureDirty;
        entry.uniformDirty = g_gxState.uniformDirty;
        entry.frame = curFrame;
        ++s_uniformCacheMisses;
      }

      s_lastShaderConfig = configPtr->shaderConfig;
      s_lastInfo = info;
      s_lastBindGroups = bindGroups;
      s_lastUniformRange = uniformRange;
      s_lastTextureDirty = g_gxState.textureDirty;
      s_lastUniformDirty = g_gxState.uniformDirty;
      s_lastUniformFrame = curFrame;
    }
  }

  const auto tc0 = std::chrono::steady_clock::now();
  gfx::push_draw_command(DrawData{
      .pipeline = pipeline,
      .vertRange = vertRange,
      .idxRange = idxRange,
      .uniformRange = uniformRange,
      .vtxCount = vtxCount,
      .indexCount = numIndices,
      .instanceCount = 1,
      .bindGroups = bindGroups,
      .dstAlpha = g_gxState.dstAlpha,
      .nativeVertexFetch = true,
      .cachedGeometry = cachedGeometry,
  });
  const auto tc1 = std::chrono::steady_clock::now();
  s_drainTiming.ms_push_cmd += std::chrono::duration<double, std::milli>(tc1 - tc0).count();
}

// Expand `vtxCount` raw FIFO vertices and push a native draw, serving steady-state
// geometry from the persistent content-hash cache (no re-expand, no re-upload).
static bool try_native_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, const u8* data, u32& pos, u32 vtxSize) {
  if (prim != GX_TRIANGLES && prim != GX_TRIANGLESTRIP && prim != GX_TRIANGLEFAN && prim != GX_QUADS) {
    return false;
  }
  u8 nativeStride = 0;
  std::array<AttrConfig, MaxVtxAttr> nativeAttrs{};
  const std::vector<NativeAttrDesc>* descsPtr = nullptr;
  if (!get_native_layout(fmt, descsPtr, nativeStride, nativeAttrs)) {
    return false;
  }
  const auto& descs = *descsPtr;
  ++s_geomDrawsSingle;
  const u32 srcBytes = static_cast<u32>(vtxCount) * vtxSize;

  // Content-hash key: source FIFO bytes + decode layout (+ indexed array contents). A hit
  // replays the cached GPU ranges without touching the CPU-side vertex data at all.
  NativeGeomKey key{};
  if constexpr (kNativeGeomCacheEnabled) {
    const auto th0 = std::chrono::steady_clock::now();
    const u64 fifoHash = XXH3_64bits(data + pos, srcBytes);
    const u64 metaHash = hash_native_geom_meta(prim, fmt, vtxSize, vtxCount, descs);
    key = {fifoHash, metaHash, srcBytes};
    native_geom_sync_generation();
    const auto th1 = std::chrono::steady_clock::now();
    s_drainTiming.ms_geom_hash += std::chrono::duration<double, std::milli>(th1 - th0).count();
    if (const auto it = s_nativeGeomCache.find(key); it != s_nativeGeomCache.end()) {
      ++s_geomCacheHits;
      pos += srcBytes;
      push_native_gx_draw(prim, fmt, vtxCount, nativeAttrs, nativeStride, it->second.vertRange, it->second.idxRange,
                          it->second.indexCount, /*cachedGeometry=*/true);
      return true;
    }
    ++s_geomCacheMisses;
  }

  // Miss — expand the run into flat native attributes.
  s_nativeVtxBuf.clear();
  s_nativeVtxBuf.reserve_extra(static_cast<u32>(vtxCount) * nativeStride);
  for (u32 v = 0; v < vtxCount; ++v) {
    expand_native_vertex(descs, data + pos + static_cast<size_t>(v) * vtxSize, s_nativeVtxBuf);
  }
  pos += srcBytes;

  // Strips/fans/quads get a generated index buffer; plain triangle lists draw non-indexed.
  const bool indexed = prim != GX_TRIANGLES;
  u32 numIndices = 0;
  s_nativeIdxBuf.clear();
  if (indexed) {
    numIndices = prepare_idx_buffer(s_nativeIdxBuf, prim, 0, vtxCount);
  }

  gfx::Range vertRange{};
  gfx::Range idxRange{};
  bool cached = false;
  native_geom_resolve_ranges(key, s_nativeVtxBuf, indexed ? s_nativeIdxBuf.data() : nullptr,
                             indexed ? s_nativeIdxBuf.size() : 0, numIndices, vertRange, idxRange, cached);
  s_nativeIdxBuf.clear();

  push_native_gx_draw(prim, fmt, vtxCount, nativeAttrs, nativeStride, vertRange, idxRange, numIndices, cached);
  return true;
}

// Batched native path for raw-FIFO draw commands. Display lists emit runs of back-to-back
// identical draw commands (same VAT byte ⇒ same primitive + layout; no state change can
// sit between adjacent draws), so this collapses a whole run into ONE hardware draw — the
// biggest CPU-side win, since per-draw pipeline/bind-group/uniform submission dominates
// strip-heavy scenes. Triangle strips keep strip topology with 0xffff primitive-restart
// separators rather than being unrolled to a triangle list. `pos` enters at the first
// draw's vertex data (cmd + vtxCount already consumed) and is advanced past the whole
// consumed run on success. Serves steady-state geometry from the persistent content-hash
// cache (no re-expand, no re-upload).
static bool try_native_draw_run(u8 cmd, GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, u32 vtxSize, const u8* data,
                                u32& pos, u32 size, bool bigEndian) {
  if (prim != GX_TRIANGLES && prim != GX_TRIANGLESTRIP && prim != GX_TRIANGLEFAN && prim != GX_QUADS) {
    return false;
  }
  if (vtxSize == 0 || vtxCount == 0) {
    return false;
  }
  u8 nativeStride = 0;
  std::array<AttrConfig, MaxVtxAttr> nativeAttrs{};
  const std::vector<NativeAttrDesc>* descsPtr = nullptr;
  if (!get_native_layout(fmt, descsPtr, nativeStride, nativeAttrs)) {
    return false;
  }
  const auto& descs = *descsPtr;

  // Scan forward across back-to-back identical draw commands. FIFO layout per draw is
  // [cmd][vtxCount:u16][vtxData]; require the same cmd byte and a whole vertex block.
  static std::vector<u16> segCounts;
  if (segCounts.capacity() < 2048) segCounts.reserve(2048);
  segCounts.clear();
  segCounts.push_back(vtxCount);
  u32 batchVtxCount = vtxCount;
  u32 scan = pos + static_cast<u32>(vtxCount) * vtxSize;
  const u16 minVtx = prim == GX_QUADS ? 4 : 3;
  const auto ts0 = std::chrono::steady_clock::now();
  while (scan + 3 <= size && data[scan] == cmd) {
    const u16 nextVtxCount = read_u16(data + scan + 1, bigEndian);
    // Cap the batch below 0xffff verts so no index collides with the 0xffff restart marker.
    if (nextVtxCount < minVtx || batchVtxCount + nextVtxCount >= 0xffffu) {
      break;
    }
    const u32 nextVtxBytes = static_cast<u32>(nextVtxCount) * vtxSize;
    if (scan + 3 + nextVtxBytes > size) {
      break;
    }
    segCounts.push_back(nextVtxCount);
    batchVtxCount += nextVtxCount;
    scan += 3 + nextVtxBytes;
  }
  const auto ts1 = std::chrono::steady_clock::now();
  s_drainTiming.ms_scan += std::chrono::duration<double, std::milli>(ts1 - ts0).count();

  ++s_geomRunsBatched;
  s_geomSegmentsBatched += segCounts.size();
  const bool stripTopology = prim == GX_TRIANGLESTRIP && kStripTopology;
  const u32 srcBytes = batchVtxCount * vtxSize;

  // Content-hash key: FIFO bytes of every segment + decode layout + the run's segment
  // structure (+ indexed array contents). A hit replays cached GPU ranges with no CPU work.
  NativeGeomKey key{};
  if constexpr (kNativeGeomCacheEnabled) {
    const auto th0 = std::chrono::steady_clock::now();
    const u64 fifoHash = XXH3_64bits(data + pos, scan - pos);
    const u64 metaHash = hash_native_geom_meta(prim, fmt, vtxSize, batchVtxCount, descs,
                                               segCounts.data(), segCounts.size() * sizeof(u16));
    key = {fifoHash, metaHash, srcBytes};
    native_geom_sync_generation();
    const auto th1 = std::chrono::steady_clock::now();
    s_drainTiming.ms_geom_hash += std::chrono::duration<double, std::milli>(th1 - th0).count();
    if (const auto it = s_nativeGeomCache.find(key); it != s_nativeGeomCache.end()) {
      ++s_geomCacheHits;
      pos = scan;
      push_native_gx_draw(prim, fmt, static_cast<u16>(batchVtxCount), nativeAttrs, nativeStride, it->second.vertRange,
                          it->second.idxRange, it->second.indexCount, /*cachedGeometry=*/true, stripTopology);
      return true;
    }
    ++s_geomCacheMisses;
  }

  // Miss — expand every segment into flat native attributes (contiguous in one buffer).
  const auto t_miss0 = std::chrono::steady_clock::now();
  const auto te0 = std::chrono::steady_clock::now();
  s_nativeVtxBuf.clear();
  s_nativeVtxBuf.reserve_extra(batchVtxCount * nativeStride);
  const u8* segData = data + pos;
  for (size_t s = 0; s < segCounts.size(); ++s) {
    const u16 count = segCounts[s];
    for (u16 v = 0; v < count; ++v) {
      expand_native_vertex(descs, segData + static_cast<size_t>(v) * vtxSize, s_nativeVtxBuf);
    }
    segData += static_cast<size_t>(count) * vtxSize + 3;
  }
  const auto te1 = std::chrono::steady_clock::now();
  s_drainTiming.ms_geom_expand += std::chrono::duration<double, std::milli>(te1 - te0).count();
  pos = scan;

  // Index topology for the run: strips stay strips (restart-separated); a plain triangle
  // list draws non-indexed; fans/quads (and strips when strip topology is off) unroll to a
  // per-segment triangle list.
  s_nativeIdxBuf.clear();
  u32 numIndices = 0;
  bool indexed = true;
  if (stripTopology) {
    numIndices = build_triangle_strip_batch_topology_indices(s_nativeIdxBuf, segCounts);
  } else if (prim == GX_TRIANGLES) {
    indexed = false; // numIndices stays 0 -> non-indexed Draw(batchVtxCount)
  } else {
    u16 segStart = 0;
    for (const u16 count : segCounts) {
      numIndices += prepare_idx_buffer(s_nativeIdxBuf, prim, segStart, count);
      segStart = static_cast<u16>(segStart + count);
    }
  }

  gfx::Range vertRange{};
  gfx::Range idxRange{};
  bool cached = false;
  native_geom_resolve_ranges(key, s_nativeVtxBuf, indexed ? s_nativeIdxBuf.data() : nullptr,
                             indexed ? s_nativeIdxBuf.size() : 0, numIndices, vertRange, idxRange, cached);
  s_nativeIdxBuf.clear();
  const auto t_miss1 = std::chrono::steady_clock::now();
  s_drainTiming.ms_geom_miss += std::chrono::duration<double, std::milli>(t_miss1 - t_miss0).count();

  push_native_gx_draw(prim, fmt, static_cast<u16>(batchVtxCount), nativeAttrs, nativeStride, vertRange, idxRange,
                      numIndices, cached, stripTopology);
  return true;
}

// Emit ONE GX_AURORA_DRAW_INDEXED normally (no merge): expand the (already deduped) vertices and use
// the supplied triangle-list indices, both served from the content-hash cache on recurrence.
static bool native_emit_indexed(GXVtxFmt fmt, u16 vtxCount, const u8* vtxData, u32 vtxSize, const u8* idxData,
                                u32 idxBytes, u32 indexCount) {
  u8 nativeStride = 0;
  std::array<AttrConfig, MaxVtxAttr> nativeAttrs{};
  const std::vector<NativeAttrDesc>* descsPtr = nullptr;
  if (!get_native_layout(fmt, descsPtr, nativeStride, nativeAttrs)) {
    return false;
  }
  const auto& descs = *descsPtr;
  const u32 srcBytes = static_cast<u32>(vtxCount) * vtxSize;

  NativeGeomKey key{};
  if constexpr (kNativeGeomCacheEnabled) {
    const auto th0 = std::chrono::steady_clock::now();
    const u64 fifoHash = XXH3_64bits(vtxData, srcBytes);
    const u64 metaHash = hash_native_geom_meta(GX_TRIANGLES, fmt, vtxSize, vtxCount, descs, idxData, idxBytes);
    key = {fifoHash, metaHash, srcBytes};
    native_geom_sync_generation();
    const auto th1 = std::chrono::steady_clock::now();
    s_drainTiming.ms_geom_hash += std::chrono::duration<double, std::milli>(th1 - th0).count();
    if (const auto it = s_nativeGeomCache.find(key); it != s_nativeGeomCache.end()) {
      ++s_geomCacheHits;
      push_native_gx_draw(GX_TRIANGLES, fmt, vtxCount, nativeAttrs, nativeStride, it->second.vertRange,
                          it->second.idxRange, it->second.indexCount, /*cachedGeometry=*/true);
      return true;
    }
    ++s_geomCacheMisses;
  }

  s_nativeVtxBuf.clear();
  s_nativeVtxBuf.reserve_extra(static_cast<u32>(vtxCount) * nativeStride);
  for (u32 v = 0; v < vtxCount; ++v) {
    expand_native_vertex(descs, vtxData + static_cast<size_t>(v) * vtxSize, s_nativeVtxBuf);
  }

  gfx::Range vertRange{};
  gfx::Range idxRange{};
  bool cached = false;
  native_geom_resolve_ranges(key, s_nativeVtxBuf, idxData, idxBytes, indexCount, vertRange, idxRange, cached);

  push_native_gx_draw(GX_TRIANGLES, fmt, vtxCount, nativeAttrs, nativeStride, vertRange, idxRange, indexCount, cached);
  return true;
}

// Native path for GX_AURORA_DRAW_INDEXED: expand the marker's vertices into hardware attributes and
// emit one native draw (served from the content-hash cache on recurrence).
static bool try_native_draw_indexed(GXVtxFmt fmt, u16 vtxCount, const u8* vtxData, u32 vtxSize, const u8* idxData,
                                    u32 idxBytes, u32 indexCount) {
  ++s_geomDrawsIndexed;
  return native_emit_indexed(fmt, vtxCount, vtxData, vtxSize, idxData, idxBytes, indexCount);
}

std::string read_string(const u8* data, u32& pos, u32 size, bool bigEndian) {
  CHECK(pos + 2 <= size, "Aurora string length read overrun");
  const u16 length = read_u16(data + pos, bigEndian);
  pos += 2;

  CHECK(pos + length <= size, "Aurora string read overrun");
  std::string str(reinterpret_cast<const char*>(data) + pos, length);
  pos += length;
  return str;
}

void handle_aurora(const u8* data, u32& pos, u32 size, bool bigEndian) {
  ZoneScoped;
  CHECK(pos + 2 <= size, "Aurora cmd read overrun");
  u16 subCmd = read_u16(data + pos, bigEndian);
  pos += 2;

  // Setting of vertex array bases.
  if (subCmd == GX_AURORA_LOAD_VIEWPORT_RENDER) {
    CHECK(pos + 24 <= size, "GX_AURORA_LOAD_VIEWPORT_RENDER read overrun");
    const f32 left = read_f32(data + pos, bigEndian);
    pos += 4;
    const f32 top = read_f32(data + pos, bigEndian);
    pos += 4;
    const f32 width = read_f32(data + pos, bigEndian);
    pos += 4;
    const f32 height = read_f32(data + pos, bigEndian);
    pos += 4;
    const f32 nearZ = read_f32(data + pos, bigEndian);
    pos += 4;
    const f32 farZ = read_f32(data + pos, bigEndian);
    pos += 4;
    set_render_viewport({
        .left = left,
        .top = top,
        .width = width,
        .height = height,
        .znear = nearZ,
        .zfar = farZ,
    });
  } else if (subCmd == GX_AURORA_LOAD_SCISSOR_RENDER) {
    CHECK(pos + 16 <= size, "GX_AURORA_LOAD_SCISSOR_RENDER read overrun");
    const int32_t left = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    const int32_t top = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    const int32_t width = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    const int32_t height = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    set_render_scissor({left, top, width, height});
  } else if (subCmd == GX_AURORA_SET_PROJECTION_SCALE_X) {
    CHECK(pos + 4 <= size, "GX_AURORA_SET_PROJECTION_SCALE_X read overrun");
    const f32 scale = read_f32(data + pos, bigEndian);
    pos += 4;
    if (g_gxState.projScaleX != scale) {
      g_gxState.projScaleX = scale;
      g_gxState.uniformDirty++;
      g_gxState.stateDirty = true;
    }
  } else if (subCmd == GX_AURORA_SET_PAPER) {
    CHECK(pos + 10 <= size, "GX_AURORA_SET_PAPER read overrun");
    const u8 edges = data[pos];
    pos += 1;
    const bool shadow = data[pos] != 0;
    pos += 1;
    const auto sdf = reinterpret_cast<const void*>(read_u64(data + pos, bigEndian));
    pos += 8;
    if (edges != g_gxState.paperEdges || shadow != g_gxState.paperShadow || sdf != g_gxState.paperSdf) {
      g_gxState.paperEdges = edges;
      g_gxState.paperShadow = shadow;
      g_gxState.paperSdf = sdf;
      u8 sdfMaps = 0;
      for (u32 i = 0; i < MaxTextures; ++i) {
        if (sdf != nullptr && g_gxState.loadedTextures[i].data == sdf) {
          sdfMaps |= static_cast<u8>(1u << i);
        }
      }
      g_gxState.paperSdfMaps = sdfMaps;
      g_gxState.pipelineDirty++;
      g_gxState.stateDirty = true;
    }
  } else if (subCmd == GX_AURORA_LOAD_PROJECTION_FULL) {
    CHECK(pos + 64 <= size, "GX_AURORA_LOAD_PROJECTION_FULL read overrun");
    auto& proj = g_gxState.proj;
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        proj[r][c] = read_f32(data + pos, bigEndian);
        pos += 4;
      }
    }
    g_gxState.stateDirty = true;
  } else if (subCmd >= GX_AURORA_LOAD_ARRAYBASE && subCmd <= (GX_AURORA_LOAD_ARRAYBASE | 0x0f)) {
    CHECK(pos + 13 <= size, "GX_AURORA_LOAD_ARRAYBASE read overrun");
    u32 attrIdx = subCmd - GX_AURORA_LOAD_ARRAYBASE + GX_VA_POS;

    u64 arrayAddr = read_u64(data + pos, bigEndian);
    pos += 8;
    u32 arraySize = read_u32(data + pos, bigEndian);
    pos += 4;
    bool le = data[pos] == 1;
    pos += 1;

    auto& array = g_gxState.arrays[attrIdx];
    const auto newData = reinterpret_cast<void*>(arrayAddr);
    if (array.data != newData || array.size != arraySize || array.le != le) {
      array.data = newData;
      array.size = arraySize;
      array.le = le;
      // Only drop the cached upload when the backing array actually changes.
      array.cachedRange = {};
      g_gxState.stateDirty = true;
    }
  } else if (subCmd == GX_AURORA_LOAD_TEXOBJ) {
    CHECK(pos + 34 <= size, "GX_AURORA_LOAD_TEXOBJ read overrun");
    const auto texMapId = data[pos];
    pos += 1;
    CHECK(texMapId < MaxTextures, "invalid texture map id {}", texMapId);
    g_gxState.textureDirty++;
    auto& slot = g_gxState.loadedTextures[texMapId];
    slot.data = reinterpret_cast<const void*>(read_u64(data + pos, bigEndian));
    pos += 8;
    slot.mWidth = read_u32(data + pos, bigEndian);
    pos += 4;
    slot.mHeight = read_u32(data + pos, bigEndian);
    pos += 4;
    slot.mFormat = static_cast<GXTexFmt>(read_u32(data + pos, bigEndian));
    pos += 4;
    slot.tlut = static_cast<GXTlut>(read_u32(data + pos, bigEndian));
    pos += 4;
    if (data[pos] != 0) {
      slot.flags |= 1u;
    } else {
      slot.flags &= ~1u;
    }
    pos += 1;
    slot.texObjId = read_u32(data + pos, bigEndian);
    pos += 4;
    slot.texDataVersion = read_u32(data + pos, bigEndian);
    pos += 4;
    slot.set_no_cache(false); // Reset no-cache flag
    const u8 bit = static_cast<u8>(1u << texMapId);
    const u8 sdfMaps = g_gxState.paperSdf != nullptr && slot.data == g_gxState.paperSdf
                           ? static_cast<u8>(g_gxState.paperSdfMaps | bit)
                           : static_cast<u8>(g_gxState.paperSdfMaps & ~bit);
    if (sdfMaps != g_gxState.paperSdfMaps) {
      g_gxState.paperSdfMaps = sdfMaps;
      g_gxState.pipelineDirty++;
    }
    g_gxState.stateDirty = true;
  } else if (subCmd == GX_AURORA_LOAD_TLUT) {
    CHECK(pos + 23 <= size, "GX_AURORA_LOAD_TLUT read overrun");
    const auto idx = data[pos];
    pos += 1;
    CHECK(idx < MaxTluts, "invalid tlut slot {}", idx);
    auto& slot = g_gxState.loadedTluts[idx];
    slot.data = reinterpret_cast<const void*>(read_u64(data + pos, bigEndian));
    pos += 8;
    slot.format = static_cast<GXTlutFmt>(read_u32(data + pos, bigEndian));
    pos += 4;
    slot.numEntries = read_u16(data + pos, bigEndian);
    pos += 2;
    slot.tlutObjId = read_u32(data + pos, bigEndian);
    pos += 4;
    slot.tlutDataVersion = read_u32(data + pos, bigEndian);
    pos += 4;
    slot.set_no_cache(false); // Reset no-cache flag
    g_gxState.stateDirty = true;
  } else if (subCmd == GX2_SET_POLYGON_OFFSET) {
    CHECK(pos + 20 <= size, "GX2_SET_POLYGON_OFFSET read overrun");
    g_gxState.frontOffset = read_f32(data + pos, bigEndian);
    pos += 4;
    g_gxState.frontScale = read_f32(data + pos, bigEndian);
    pos += 4;
    g_gxState.backOffset = read_f32(data + pos, bigEndian);
    pos += 4;
    g_gxState.backScale = read_f32(data + pos, bigEndian);
    pos += 4;
    g_gxState.clamp = read_f32(data + pos, bigEndian);
    pos += 4;
    g_gxState.stateDirty = true;
  } else if (subCmd == GX_AURORA_LOAD_COPY_SRC) {
    CHECK(pos + 16 <= size, "GX_AURORA_LOAD_COPY_SRC read overrun");
    const int32_t left = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    const int32_t top = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    const int32_t width = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    const int32_t height = static_cast<int32_t>(read_u32(data + pos, bigEndian));
    pos += 4;
    g_gxState.texCopySrc = {left, top, width, height};
  } else if (subCmd == GX_AURORA_LOAD_COPY_DST) {
    CHECK(pos + 13 <= size, "GX_AURORA_LOAD_COPY_DST read overrun");
    g_gxState.texCopyDstWidth = read_u32(data + pos, bigEndian);
    pos += 4;
    g_gxState.texCopyDstHeight = read_u32(data + pos, bigEndian);
    pos += 4;
    g_gxState.texCopyFmt = static_cast<GXTexFmt>(read_u32(data + pos, bigEndian));
    pos += 4;
    g_gxState.texCopyDstWide = true;
  } else if (subCmd == GX_AURORA_LOAD_COPY_DEST) {
    CHECK(pos + 8 <= size, "GX_AURORA_LOAD_COPY_DEST read overrun");
    g_gxState.texCopyDest = reinterpret_cast<const void*>(read_u64(data + pos, bigEndian));
    pos += 8;
  } else if (subCmd == GX_AURORA_REQUEST_DEPTH_SNAPSHOT) {
    gfx::depth_peek::request_snapshot();
  } else if (subCmd == GX_AURORA_BEGIN_OFFSCREEN) {
    CHECK(pos + 8 <= size, "GX_AURORA_BEGIN_OFFSCREEN read overrun");
    const u32 width = read_u32(data + pos, bigEndian);
    pos += 4;
    const u32 height = read_u32(data + pos, bigEndian);
    pos += 4;
    gfx::begin_offscreen(width, height);
  } else if (subCmd == GX_AURORA_END_OFFSCREEN) {
    gfx::end_offscreen();
  } else if (subCmd == GX_AURORA_DESTROY_TEXOBJ) {
    CHECK(pos + 4 <= size, "GX_AURORA_DESTROY_TEXOBJ read overrun");
    evict_texture_object(read_u32(data + pos, bigEndian));
    pos += 4;
  } else if (subCmd == GX_AURORA_DESTROY_TLUT) {
    CHECK(pos + 4 <= size, "GX_AURORA_DESTROY_TLUT read overrun");
    evict_tlut_object(read_u32(data + pos, bigEndian));
    pos += 4;
  } else if (subCmd == GX_AURORA_DESTROY_COPY_TEX) {
    CHECK(pos + 8 <= size, "GX_AURORA_DESTROY_COPY_TEX read overrun");
    evict_copy_texture(reinterpret_cast<const void*>(read_u64(data + pos, bigEndian)));
    pos += 8;
  } else if (subCmd == GX_AURORA_ALIAS_COPY_TEX) {
    CHECK(pos + 16 <= size, "GX_AURORA_ALIAS_COPY_TEX read overrun");
    const auto* src = reinterpret_cast<const void*>(read_u64(data + pos, bigEndian));
    const auto* dest = reinterpret_cast<const void*>(read_u64(data + pos + 8, bigEndian));
    alias_copy_texture(src, dest);
    pos += 16;
  } else if (subCmd == GX_AURORA_FORGET_MEMORY) {
    CHECK(pos + 16 <= size, "GX_AURORA_FORGET_MEMORY read overrun");
    const auto* lo = reinterpret_cast<const void*>(read_u64(data + pos, bigEndian));
    const auto* hi = reinterpret_cast<const void*>(read_u64(data + pos + 8, bigEndian));
    forget_memory(lo, hi);
    pos += 16;
  } else if (subCmd == GX_AURORA_MOVE_MEMORY) {
    CHECK(pos + 24 <= size, "GX_AURORA_MOVE_MEMORY read overrun");
    const auto* from = reinterpret_cast<const void*>(read_u64(data + pos, bigEndian));
    const auto* to = reinterpret_cast<const void*>(read_u64(data + pos + 8, bigEndian));
    const u64 sz = read_u64(data + pos + 16, bigEndian);
    move_memory(from, to, static_cast<size_t>(sz));
    pos += 24;
  } else if (subCmd == GX_AURORA_DRAW_SIZED) {
    CHECK(pos + 5 <= size, "GX_AURORA_DRAW_SIZED read overrun");
    u8 cmd = data[pos];
    pos += 1;
    u32 byteLen = read_u32(data + pos, bigEndian);
    pos += 4;
    GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
    GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
    if (byteLen != 0) {
      u32 vtxSize;
      if (g_gxState.lastVtxFmt == fmt) {
        vtxSize = g_gxState.lastVtxSize;
      } else {
        vtxSize = calculate_last_vtx_size(fmt);
      }
      ASSERT(vtxSize != 0 && byteLen % vtxSize == 0,
             "GX_AURORA_DRAW_SIZED: {} bytes is not a whole number of size-{} vertices", byteLen, vtxSize);
      u32 vtxCount = byteLen / vtxSize;
      ASSERT(vtxCount <= 0xFFFF, "GX_AURORA_DRAW_SIZED: too many vertices ({})", vtxCount);
      draw_prim(prim, fmt, static_cast<u16>(vtxCount), data, pos, size);
    }
  } else if (subCmd == GX_AURORA_DRAW_INDEXED) {
    ZoneScopedN("DRAW_INDEXED");
    CHECK(pos + 7 <= size, "GX_AURORA_DRAW_INDEXED read overrun");
    const u8 cmd = data[pos];
    pos += 1;
    const u16 vtxCount = read_u16(data + pos, bigEndian);
    pos += 2;
    const u32 indexCount = read_u32(data + pos, bigEndian);
    pos += 4;
    const GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
    const GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
    ASSERT(prim == GX_TRIANGLES, "GX_AURORA_DRAW_INDEXED: primitive must be GX_TRIANGLES, got {}",
           static_cast<u32>(prim));
    const u32 idxBytes = indexCount * static_cast<u32>(sizeof(u16));
    CHECK(pos + idxBytes <= size, "GX_AURORA_DRAW_INDEXED index data overrun");
    // Index data is always host-endian. The native path pushes it (to the geometry cache
    // or the per-frame ring) itself; the storage fallback pushes it to the ring below.
    const u8* idxData = data + pos;
    pos += idxBytes;
    u32 vtxSize;
    if (g_gxState.lastVtxFmt == fmt) {
      vtxSize = g_gxState.lastVtxSize;
    } else {
      vtxSize = calculate_last_vtx_size(fmt);
    }
    const u32 totalVtxBytes = vtxCount * vtxSize;
    CHECK(pos + totalVtxBytes <= size, "GX_AURORA_DRAW_INDEXED vertex data overrun");
    const u8* vtxData = data + pos;
    pos += totalVtxBytes;
    if (sBbox.armed && indexCount != 0) {
      bbox_add_vertices(vtxData, vtxCount, vtxSize, fmt);
    }
    if (indexCount != 0) {
      if (!try_native_draw_indexed(fmt, vtxCount, vtxData, vtxSize, idxData, idxBytes, indexCount)) {
        const gfx::Range idxRange = gfx::push_indices(idxData, idxBytes, 4);
        const gfx::Range vertRange = gfx::push_verts(vtxData, totalVtxBytes, 4);
        push_gx_draw(prim, fmt, vtxCount, vertRange, idxRange, indexCount);
      }
    }
  } else if (subCmd == GX_AURORA_DEBUG_GROUP_PUSH) {
    auto label = read_string(data, pos, size, bigEndian);
    gfx::push_debug_group(std::move(label));
  } else if (subCmd == GX_AURORA_DEBUG_GROUP_POP) {
    pop_debug_group();
  } else if (subCmd == GX_AURORA_DEBUG_MARKER_INSERT) {
    auto label = read_string(data, pos, size, bigEndian);
    gfx::insert_debug_marker(std::move(label));
  }

  else {
    Log.error("Unknown Aurora subcommand: {:04X}", subCmd);
  }
}

} // namespace aurora::gx::fifo

namespace aurora::gx {
void bbox_reset() noexcept {
  auto& b = fifo::sBbox;
  b.armed = true;
  b.unknown = false;
  b.left = b.top = std::numeric_limits<float>::max();
  b.right = b.bottom = -std::numeric_limits<float>::max();
}

void bbox_read(u16* left, u16* top, u16* right, u16* bottom) noexcept {
  auto& b = fifo::sBbox;
  if (b.armed) {
    const auto& sc = g_gxState.logicalScissor;
    const float x0 = static_cast<float>(sc.x);
    const float y0 = static_cast<float>(sc.y);
    const float x1 = static_cast<float>(sc.x + sc.width - 1);
    const float y1 = static_cast<float>(sc.y + sc.height - 1);
    if (b.unknown) {
      b.result[0] = static_cast<u16>(std::max(x0, 0.f));
      b.result[1] = static_cast<u16>(std::max(y0, 0.f));
      b.result[2] = static_cast<u16>(std::max(x1, 0.f));
      b.result[3] = static_cast<u16>(std::max(y1, 0.f));
    } else if (b.right < b.left || b.right < x0 || b.left > x1 || b.bottom < y0 || b.top > y1) {
      b.result[0] = 0x3FF;
      b.result[1] = 0x3FF;
      b.result[2] = 0;
      b.result[3] = 0;
    } else {
      b.result[0] = static_cast<u16>(std::floor(std::clamp(b.left, x0, x1)));
      b.result[1] = static_cast<u16>(std::floor(std::clamp(b.top, y0, y1)));
      b.result[2] = static_cast<u16>(std::ceil(std::clamp(b.right, x0, x1)));
      b.result[3] = static_cast<u16>(std::ceil(std::clamp(b.bottom, y0, y1)));
    }
    b.armed = false;
  }
  *left = b.result[0];
  *top = b.result[1];
  *right = b.result[2];
  *bottom = b.result[3];
}
} // namespace aurora::gx
