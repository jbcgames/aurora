#pragma once

#include <dolphin/gx.h>

#include "../../gx/fifo.hpp"

#define GX_WRITE_U8(ub) aurora::gx::fifo::write_u8(static_cast<uint8_t>(ub))
#define GX_WRITE_DATA(data, length) aurora::gx::fifo::write_data(data, length)
#define GX_WRITE_U16(us) aurora::gx::fifo::write_u16(static_cast<uint16_t>(us))
#define GX_WRITE_U32(ui) aurora::gx::fifo::write_u32(static_cast<uint32_t>(ui))
#define GX_WRITE_U64(ui) aurora::gx::fifo::write_u64(static_cast<uint64_t>(ui))
#define GX_WRITE_F32(f) aurora::gx::fifo::write_f32(static_cast<float>(f))

// XF register write: opcode 0x10, address in XF space (0x1000+addr), then value
#define GX_WRITE_XF_REG(addr, value)                                                                                   \
  do {                                                                                                                 \
    GX_WRITE_U8(0x10);                                                                                                 \
    GX_WRITE_U32(0x1000 + (addr));                                                                                     \
    GX_WRITE_U32(value);                                                                                               \
  } while (0)

// XF register write for u32 data (no address header - used in bulk writes)
#define GX_WRITE_XF_REG_2(addr, value)                                                                                 \
  do {                                                                                                                 \
    GX_WRITE_U32(value);                                                                                               \
  } while (0)

// XF register write for f32 data (no address header - used in bulk writes)
#define GX_WRITE_XF_REG_F(addr, value)                                                                                 \
  do {                                                                                                                 \
    GX_WRITE_F32(value);                                                                                               \
  } while (0)

// BP (RAS) register write: opcode 0x61, then 32-bit value (upper 8 bits = reg ID)
#define GX_WRITE_RAS_REG(value)                                                                                        \
  do {                                                                                                                 \
    GX_WRITE_U8(0x61);                                                                                                 \
    GX_WRITE_U32(value);                                                                                               \
  } while (0)

// CP register write with base/stride tracking for indexed arrays
// GX_WRITE_SOME_REG2: CP array base address
#define GX_WRITE_SOME_REG2(a, b, c, addr)                                                                              \
  do {                                                                                                                 \
    GX_WRITE_U8(a);                                                                                                    \
    GX_WRITE_U8(b);                                                                                                    \
    GX_WRITE_U32(c);                                                                                                   \
  } while (0)

// GX_WRITE_SOME_REG3: CP array stride
#define GX_WRITE_SOME_REG3(a, b, c, addr)                                                                              \
  do {                                                                                                                 \
    GX_WRITE_U8(a);                                                                                                    \
    GX_WRITE_U8(b);                                                                                                    \
    GX_WRITE_U32(c);                                                                                                   \
  } while (0)

// GX_WRITE_SOME_REG4: generic CP register write (VCD, VAT, matrix index)
#define GX_WRITE_SOME_REG4(a, b, c, addr)                                                                              \
  do {                                                                                                                 \
    GX_WRITE_U8(a);                                                                                                    \
    GX_WRITE_U8(b);                                                                                                    \
    GX_WRITE_U32(c);                                                                                                   \
  } while (0)

#define GET_REG_FIELD(reg, size, shift) (static_cast<int>(((reg) >> (shift)) & ((1 << (size)) - 1)))

#define SET_REG_FIELD(line, reg, size, shift, val)                                                                     \
  do {                                                                                                                 \
    (reg) = ((u32)(reg) & ~(((1u << (size)) - 1) << (shift))) | ((u32)(val) << (shift));                               \
  } while (0)

#define GX_WRITE_AURORA(cmd)     \
  do {                           \
    GX_WRITE_U8(GX_AURORA);      \
    GX_WRITE_U16(cmd);           \
  } while (0)

#define GX_WRITE_CP_REG(addr, value)  \
  do {                                \
  GX_WRITE_U8(GX_LOAD_CP_REG);        \
  GX_WRITE_U8(addr);                  \
  GX_WRITE_U32(value);                \
  } while (0)

static inline void __gx_put_be16(u8* dst, u16 val) {
  const u16 be = bswap(val);
  std::memcpy(dst, &be, sizeof(be));
}

static inline void __gx_put_be32(u8* dst, u32 val) {
  const u32 be = bswap(val);
  std::memcpy(dst, &be, sizeof(be));
}

static inline void __gx_put_be64(u8* dst, u64 val) {
  const u64 be = bswap(val);
  std::memcpy(dst, &be, sizeof(be));
}

static inline void __gx_put_bef32(u8* dst, f32 val) {
  const f32 be = bswap(val);
  std::memcpy(dst, &be, sizeof(be));
}

template <size_t N>
static inline void __gx_write_ras_regs(const u32 (&regs)[N]) {
  u8 buf[N * 5];
  for (size_t i = 0; i < N; ++i) {
    buf[i * 5] = 0x61;
    __gx_put_be32(buf + i * 5 + 1, regs[i]);
  }
  GX_WRITE_DATA(buf, sizeof(buf));
}

// Shadow register struct
struct __GXData_struct {
  u16 vNum;         // vertex count for flush prim
  u16 bpSent;       // BP register was sent (need flush prim before next draw)
  u32 vLim;         // vertex data size limit (bytes per vertex for flush prim)

  u32 vcdLo;        // vertex component descriptor low
  u32 vcdHi;        // vertex component descriptor high

  u32 vatA[8];      // vertex attribute table A (per vtxfmt)
  u32 vatB[8];      // vertex attribute table B
  u32 vatC[8];      // vertex attribute table C

  u32 lpSize;       // line/point size register

  u32 matIdxA;      // matrix index A (pnmtx + texmtx 0-3)
  u32 matIdxB;      // matrix index B (texmtx 4-7)

  u32 ambColor[2];  // ambient colors (XF registers)
  u32 matColor[2];  // material colors (XF registers)

  u32 suTs0[8];     // SU texture S0 registers
  u32 suTs1[8];     // SU texture S1 registers
  u8 tcsManEnab;    // bitmask: manual tex coord scale enabled per coord
  u32 suScis0;      // scissor top-left
  u32 suScis1;      // scissor bottom-right

  u32 tref[8];      // TEV texture/color reference (2 stages per register)
  u32 iref;         // indirect texture reference
  u32 bpMask;       // BP mask register

  u32 IndTexScale0; // indirect texture scale 0 (stages 0-1)
  u32 IndTexScale1; // indirect texture scale 1 (stages 2-3)

  u32 tevc[16];     // TEV color combiner registers
  u32 teva[16];     // TEV alpha combiner registers
  u32 tevKsel[8];   // TEV K color/alpha selection

  u32 cmode0;       // blend mode register
  u32 cmode1;       // destination alpha register
  u32 zmode;        // Z-buffer mode register
  u32 peCtrl;       // pixel engine control
  u32 cpTex;        // copy texture trigger register

  u32 genMode;      // general mode (numTexGens, numChans, numTevStages, cullMode, numIndStages)

  u32 tImage0[8];   // texture image 0 registers
  u32 tMode0[8];    // texture mode 0 registers
  u32 texmapId[16]; // texture map ID tracking

  GXAttrType nrmType; // normal attribute type
  u8 hasNrms;       // has normal vectors
  u8 hasBiNrms;     // has binormal vectors

  u32 projType;     // projection type (perspective/orthographic)
  f32 projMtx[6];   // projection matrix (6 params)

  f32 vpLeft;       // viewport left
  f32 vpTop;        // viewport top
  f32 vpWd;         // viewport width
  f32 vpHt;         // viewport height
  f32 vpNearz;      // viewport near Z
  f32 vpFarz;       // viewport far Z

  u8 fgRange;       // fog range adjustment enabled
  f32 fgSideX;      // fog range side X

  u8 inDispList;    // currently recording a display list
  u8 dlSaveContext; // save/restore context around display lists
  u8 dirtyVAT;      // bitmask of dirty VAT entries

  u32 dirtyState;   // dirty state flags:
                    // bit 0: SU tex regs
                    // bit 1: BP mask
                    // bit 2: gen mode
                    // bit 3: VCD
                    // bit 4: VAT
};

extern __GXData_struct* __gx;

extern "C" {

// Dirty state flush functions
void __GXSetVCD();
void __GXSetVAT();
void __GXUpdateBPMask();
void __GXSetSUTexRegs();
void __GXSetDirtyState();
void __GXSetGenMode();
void __GXSetMatrixIndex(GXAttr matIdxAttr);
void __GXSendFlushPrim();
void __GXFlushTextureState();
};

struct __GXShadowCache {
  struct LightEntry {
    u32 epoch = 0;
    u32 words[13]{}; // packed color + a0..a2 + k0..k2 + px..pz + nx..nz
  };

  u32 epoch = 1;
  u32 cmode0Epoch = 0, cmode0 = 0;
  u32 cmode1Epoch = 0, cmode1 = 0;
  u32 zmodeEpoch = 0, zmode = 0;
  u32 peCtrlEpoch = 0, peCtrl = 0;
  u32 genModeEpoch = 0, genMode = 0;
  u32 alphaCompEpoch = 0, alphaComp = 0;
  u32 suScisEpoch = 0, suScis0 = 0, suScis1 = 0;
  u32 fogParamEpoch = 0;
  GXFogType fogType = GX_FOG_NONE;
  f32 fogStartZ = 0.0f, fogEndZ = 0.0f, fogNearZ = 0.0f, fogFarZ = 0.0f;
  u32 fogRegs[4]{};
  u32 fogClrEpoch = 0, fogClr = 0;
  u32 matIdxAEpoch = 0, matIdxA = 0;
  u32 matIdxBEpoch = 0, matIdxB = 0;
  u32 xfNumChansEpoch = 0, xfNumChans = 0;
  u32 xfNumTexGensEpoch = 0, xfNumTexGens = 0;
  u32 xfClipModeEpoch = 0, xfClipMode = 0;
  u32 xfAmbEpoch[2]{}, xfAmb[2]{};
  u32 xfMatEpoch[2]{}, xfMat[2]{};
  u32 xfChanCtrlEpoch[4]{}, xfChanCtrl[4]{};
  u32 xfTexGenEpoch[8]{}, xfTexGenReg[8]{}, xfTexGenPost[8]{};
  u32 tevCEpoch[16]{}, tevC[16]{};
  u32 tevAEpoch[16]{}, tevA[16]{};
  u32 trefEpoch[8]{}, tref[8]{};
  u32 tevKselEpoch[8]{}, tevKsel[8]{};
  u32 tevColorEpoch[4]{}, tevColorRA[4]{}, tevColorBG[4]{};
  u32 tevKColorEpoch[4]{}, tevKColorRA[4]{}, tevKColorBG[4]{};
  u32 suTs0Epoch[8]{}, suTs0[8]{};
  u32 suTs1Epoch[8]{}, suTs1[8]{};
  u32 vatEpoch[8]{}, vatA[8]{}, vatB[8]{}, vatC[8]{};
  u32 vcdEpoch = 0, vcdLo = 0, vcdHi = 0;
  u8 vcdHasNrms = 0, vcdHasBiNrms = 0;
  u32 projEpoch = 0;
  u32 projType = 0;
  f32 projMtx[6]{};
  u32 vpEpoch = 0;
  f32 vpParams[6]{};
  LightEntry lights[8]{};
};

extern __GXShadowCache __gxShadow;

struct __GXShadowCacheExt {
  struct TexSlot {
    u32 epoch;
    u32 texEpoch;
    u32 mode0;
    u32 mode1;
    u32 image0;
    u32 image3;
    const void* data;
    u32 mWidth;
    u32 mHeight;
    u32 mFormat;
    u32 tlut;
    u32 texObjId;
    u32 texDataVersion;
    u8 flags;
  } texObjs[8];
  struct ArraySlot {
    u32 epoch;
    u32 arrayEpoch;
    const void* data;
    u32 size;
    u8 stride;
    u8 le;
  } arrays[16];
  u32 texEpoch;
  u32 arrayEpoch;
  u32 vcdVatGen;
  u32 xfRowEpoch[64];
  u32 xfRowWords[64][4];
  u32 nrmRowEpoch[40];
  u32 nrmRowWords[40][3];
  u32 ptRowEpoch[64];
  u32 ptRowWords[64][4];
};
extern __GXShadowCacheExt __gxShadowExt;

static inline void __gx_shadow_reset() noexcept {
  std::memset(&__gxShadow, 0, sizeof(__gxShadow));
  std::memset(&__gxShadowExt, 0, sizeof(__gxShadowExt));
  __gxShadow.epoch = 1;
  __gxShadowExt.texEpoch = 1;
  __gxShadowExt.arrayEpoch = 1;
  __gxShadowExt.vcdVatGen = 1;
}

static inline void __gx_shadow_invalidate_tex() noexcept {
  if (++__gxShadowExt.texEpoch == 0) {
    __gxShadowExt.texEpoch = 1;
    std::memset(__gxShadowExt.texObjs, 0, sizeof(__gxShadowExt.texObjs));
  }
}

static inline void __gx_shadow_invalidate_arrays() noexcept {
  if (++__gxShadowExt.arrayEpoch == 0) {
    __gxShadowExt.arrayEpoch = 1;
    std::memset(__gxShadowExt.arrays, 0, sizeof(__gxShadowExt.arrays));
  }
}

static inline void __gx_shadow_invalidate() noexcept {
  if (++__gxShadow.epoch == 0) {
    __gx_shadow_reset();
  }
}

void __gx_dl_invalidate_cache() noexcept;
void __gx_dl_forget_memory(const void* lo, const void* hi) noexcept;

static inline void __gx_write_shadow_ras(u32 reg, u32& cachedReg, u32& cachedEpoch) noexcept {
  if (!aurora::gx::fifo::in_display_list()) {
    const u32 ep = __gxShadow.epoch;
    if (cachedEpoch == ep && cachedReg == reg) {
      return;
    }
    cachedEpoch = ep;
    cachedReg = reg;
  }
  GX_WRITE_RAS_REG(reg);
  __gx->bpSent = 1;
}

static inline void __gx_write_shadow_ras_pair(u32 reg0, u32 reg1, u32& cached0, u32& cached1, u32& cachedEpoch) noexcept {
  if (!aurora::gx::fifo::in_display_list()) {
    const u32 ep = __gxShadow.epoch;
    if (cachedEpoch == ep && cached0 == reg0 && cached1 == reg1) {
      return;
    }
    cachedEpoch = ep;
    cached0 = reg0;
    cached1 = reg1;
  }
  const u32 regs[2] = {reg0, reg1};
  __gx_write_ras_regs(regs);
  __gx->bpSent = 1;
}

static inline void __gx_write_shadow_xf(u32 addr, u32 val, u32& cachedVal, u32& cachedEpoch) noexcept {
  if (!aurora::gx::fifo::in_display_list()) {
    const u32 ep = __gxShadow.epoch;
    if (cachedEpoch == ep && cachedVal == val) {
      return;
    }
    cachedEpoch = ep;
    cachedVal = val;
  }
  GX_WRITE_XF_REG(addr, val);
}
