#include <cstring>

#include "gx.hpp"
#include "__gx.h"
#include "dolphin/mtx/GeoTypes.h"

static inline void CacheProjectionVector(const f32* ptr, GXProjectionType type) {
  __gx->projType = type;
  for (int i = 0; i < 6; ++i) {
    __gx->projMtx[i] = ptr[i + 1];
  }
}

static inline void WriteProjection() {
  if (!aurora::gx::fifo::in_display_list()) {
    const u32 ep = __gxShadow.epoch;
    if (__gxShadow.projEpoch == ep && __gxShadow.projType == __gx->projType &&
        std::memcmp(__gxShadow.projMtx, __gx->projMtx, sizeof(__gx->projMtx)) == 0) {
      return;
    }
    __gxShadow.projEpoch = ep;
    __gxShadow.projType = __gx->projType;
    std::memcpy(__gxShadow.projMtx, __gx->projMtx, sizeof(__gx->projMtx));
  }

  // XF bulk write: 6 params + projection type at 0x1020-0x1026
  GX_WRITE_U8(0x10);
  GX_WRITE_U32(0x00061020);
  for (int i = 0; i < 6; ++i) {
    GX_WRITE_F32(__gx->projMtx[i]);
  }
  GX_WRITE_U32(__gx->projType);
  __gx->bpSent = 0;
}

extern "C" {

void GXSetProjection(const void* mtx_, GXProjectionType type) {
  const auto& mtx = *reinterpret_cast<const aurora::Mat4x4<float>*>(mtx_);
  const f32 projVec[] = {
      static_cast<f32>(type == GX_ORTHOGRAPHIC),
      mtx[0][0],
      type == GX_ORTHOGRAPHIC ? mtx[0][3] : mtx[0][2],
      mtx[1][1],
      type == GX_ORTHOGRAPHIC ? mtx[1][3] : mtx[1][2],
      mtx[2][2],
      mtx[2][3],
  };
  CacheProjectionVector(projVec, type);
  WriteProjection();
}

void GXSetProjectionv(const f32* ptr) {
  CHECK(ptr != nullptr, "null projection vector");

  const GXProjectionType type = ptr[0] == 0.0f ? GX_PERSPECTIVE : GX_ORTHOGRAPHIC;
  CacheProjectionVector(ptr, type);
  WriteProjection();
}

void GXLoadPosMtxImm(const void* mtx_, u32 id) {
  // smstrikers-port: skinning may address any row of matrix memory, not just the
  // ten named pairs. See MaxPnMtx in gx.hpp.
  CHECK(id <= (aurora::gx::MaxPnMtx - 1) * 3, "invalid pn mtx {}", static_cast<int>(id));
  const auto* mtx = reinterpret_cast<const f32*>(mtx_);

  u32 words[12];
  std::memcpy(words, mtx, sizeof(words));
  if (!aurora::gx::fifo::in_display_list() && id + 3 <= 64) {
    const u32 ep = __gxShadow.epoch;
    if (__gxShadowExt.xfRowEpoch[id] == ep && __gxShadowExt.xfRowEpoch[id + 1] == ep &&
        __gxShadowExt.xfRowEpoch[id + 2] == ep &&
        std::memcmp(__gxShadowExt.xfRowWords[id], words, sizeof(words)) == 0) {
      return;
    }
    __gxShadowExt.xfRowEpoch[id] = ep;
    __gxShadowExt.xfRowEpoch[id + 1] = ep;
    __gxShadowExt.xfRowEpoch[id + 2] = ep;
    std::memcpy(__gxShadowExt.xfRowWords[id], words, sizeof(words));
  }

  u8 pkt[1 + 13 * 4];
  pkt[0] = 0x10;
  __gx_put_be32(pkt + 1, (id * 4) | 0xB0000);
  for (int i = 0; i < 12; i++) {
    __gx_put_be32(pkt + 5 + i * 4, words[i]);
  }
  GX_WRITE_DATA(pkt, sizeof(pkt));
}

void GXLoadNrmMtxImm(const void* mtx_, u32 id) {
  // smstrikers-port: see GXLoadPosMtxImm above.
  CHECK(id <= (aurora::gx::MaxPnMtx - 1) * 3, "invalid pn mtx {}", static_cast<int>(id));
  const auto* mtx = reinterpret_cast<const f32*>(mtx_);

  u32 words[9];
  std::memcpy(&words[0], &mtx[0], 12);
  std::memcpy(&words[3], &mtx[4], 12);
  std::memcpy(&words[6], &mtx[8], 12);
  if (!aurora::gx::fifo::in_display_list() && id + 3 <= 40) {
    const u32 ep = __gxShadow.epoch;
    if (__gxShadowExt.nrmRowEpoch[id] == ep && __gxShadowExt.nrmRowEpoch[id + 1] == ep &&
        __gxShadowExt.nrmRowEpoch[id + 2] == ep &&
        std::memcmp(__gxShadowExt.nrmRowWords[id], words, sizeof(words)) == 0) {
      return;
    }
    __gxShadowExt.nrmRowEpoch[id] = ep;
    __gxShadowExt.nrmRowEpoch[id + 1] = ep;
    __gxShadowExt.nrmRowEpoch[id + 2] = ep;
    std::memcpy(__gxShadowExt.nrmRowWords[id], words, sizeof(words));
  }

  u8 pkt[1 + 10 * 4];
  pkt[0] = 0x10;
  __gx_put_be32(pkt + 1, (id * 3 + 0x400) | 0x80000);
  // Write 3x3 from 3x4 matrix (skip translation column)
  for (int i = 0; i < 9; i++) {
    __gx_put_be32(pkt + 5 + i * 4, words[i]);
  }
  GX_WRITE_DATA(pkt, sizeof(pkt));
}

void GXSetCurrentMtx(u32 id) {
  CHECK(id >= GX_PNMTX0 && id <= GX_PNMTX9, "invalid pn mtx {}", id);
  SET_REG_FIELD(0, __gx->matIdxA, 6, 0, id);
  __GXSetMatrixIndex(GX_VA_PNMTXIDX);
}

void GXLoadTexMtxImm(const void* mtx_, u32 id, GXTexMtxType type) {
  CHECK((id >= GX_TEXMTX0 && id <= GX_IDENTITY) || (id >= GX_PTTEXMTX0 && id <= GX_PTIDENTITY), "invalid tex mtx {}",
        id);

  const auto* mtx = reinterpret_cast<const f32*>(mtx_);
  u32 count = (type == GX_MTX2x4) ? 8 : 12;
  u32 rows = (count == 8) ? 2 : 3;
  u32 words[12];
  std::memcpy(words, mtx, count * sizeof(u32));

  u32 addr;
  if (id >= GX_PTTEXMTX0) {
    const u32 ptRow = id - GX_PTTEXMTX0;
    addr = ptRow * 4 + 0x500;
    CHECK(type == GX_MTX3x4, "invalid pt mtx type {}", underlying(type));
    if (!aurora::gx::fifo::in_display_list() && ptRow + 3 <= 64) {
      const u32 ep = __gxShadow.epoch;
      if (__gxShadowExt.ptRowEpoch[ptRow] == ep && __gxShadowExt.ptRowEpoch[ptRow + 1] == ep &&
          __gxShadowExt.ptRowEpoch[ptRow + 2] == ep &&
          std::memcmp(__gxShadowExt.ptRowWords[ptRow], words, 12 * sizeof(u32)) == 0) {
        return;
      }
      __gxShadowExt.ptRowEpoch[ptRow] = ep;
      __gxShadowExt.ptRowEpoch[ptRow + 1] = ep;
      __gxShadowExt.ptRowEpoch[ptRow + 2] = ep;
      std::memcpy(__gxShadowExt.ptRowWords[ptRow], words, 12 * sizeof(u32));
    }
  } else {
    addr = id * 4;
    if (!aurora::gx::fifo::in_display_list() && id + rows <= 64) {
      const u32 ep = __gxShadow.epoch;
      bool match = (__gxShadowExt.xfRowEpoch[id] == ep && __gxShadowExt.xfRowEpoch[id + 1] == ep &&
                    (rows == 2 || __gxShadowExt.xfRowEpoch[id + 2] == ep));
      if (match && std::memcmp(__gxShadowExt.xfRowWords[id], words, count * sizeof(u32)) == 0) {
        return;
      }
      __gxShadowExt.xfRowEpoch[id] = ep;
      __gxShadowExt.xfRowEpoch[id + 1] = ep;
      if (rows == 3) {
        __gxShadowExt.xfRowEpoch[id + 2] = ep;
      }
      std::memcpy(__gxShadowExt.xfRowWords[id], words, count * sizeof(u32));
    }
  }

  u32 reg = addr | ((count - 1) << 16);
  u8 pkt[1 + 13 * 4];
  pkt[0] = 0x10;
  __gx_put_be32(pkt + 1, reg);
  for (u32 i = 0; i < count; i++) {
    __gx_put_be32(pkt + 5 + i * 4, words[i]);
  }
  GX_WRITE_DATA(pkt, 5 + count * 4);
}

void GXSetViewport(float left, float top, float width, float height, float nearZ, float farZ) {
  GXSetViewportJitter(left, top, width, height, nearZ, farZ, 1);
}

void GXSetViewportJitter(float left, float top, float width, float height, float nearZ, float farZ, u32 field) {
  float sx;
  float sy;
  float sz;
  float ox;
  float oy;
  float oz;
  float zmin;
  float zmax;

  if (field == 0) {
    top -= 0.5f;
  }

  sx = width / 2.0f;
  sy = -height / 2.0f;
  ox = 340.0f + (left + width / 2.0f);
  oy = 340.0f + (top + height / 2.0f);
  zmin = 1.6777215e7f * nearZ;
  zmax = 1.6777215e7f * farZ;
  sz = zmax - zmin;
  oz = zmax;

  __gx->vpLeft = left;
  __gx->vpTop = top;
  __gx->vpWd = width;
  __gx->vpHt = height;
  __gx->vpNearz = nearZ;
  __gx->vpFarz = farZ;

  const f32 vp[6] = {sx, sy, sz, ox, oy, oz};
  if (!aurora::gx::fifo::in_display_list()) {
    const u32 ep = __gxShadow.epoch;
    if (__gxShadow.vpEpoch == ep && std::memcmp(__gxShadow.vpParams, vp, sizeof(vp)) == 0) {
      return;
    }
    __gxShadow.vpEpoch = ep;
    std::memcpy(__gxShadow.vpParams, vp, sizeof(vp));
  }

  GX_WRITE_U8(0x10);
  GX_WRITE_U32(0x0005101A);
  for (int i = 0; i < 6; ++i) {
    GX_WRITE_F32(vp[i]);
  }
  __gx->bpSent = 0;
}

void GXProject(f32 x, f32 y, f32 z, const f32 mtx[3][4], const f32* pm, const f32* vp, f32* sx,
               f32* sy, f32* sz) {
  Vec peye;
  f32 xc;
  f32 yc;
  f32 zc;
  f32 wc;

  peye.x = mtx[0][3] + ((mtx[0][2] * z) + ((mtx[0][0] * x) + (mtx[0][1] * y)));
  peye.y = mtx[1][3] + ((mtx[1][2] * z) + ((mtx[1][0] * x) + (mtx[1][1] * y)));
  peye.z = mtx[2][3] + ((mtx[2][2] * z) + ((mtx[2][0] * x) + (mtx[2][1] * y)));
  if (pm[0] == 0.0f) {
    xc = (peye.x * pm[1]) + (peye.z * pm[2]);
    yc = (peye.y * pm[3]) + (peye.z * pm[4]);
    zc = pm[6] + (peye.z * pm[5]);
    wc = 1.0f / -peye.z;
  } else {
    xc = pm[2] + (peye.x * pm[1]);
    yc = pm[4] + (peye.y * pm[3]);
    zc = pm[6] + (peye.z * pm[5]);
    wc = 1.0f;
  }
  *sx = (vp[2] / 2.0f) + (vp[0] + (wc * (xc * vp[2] / 2.0f)));
  *sy = (vp[3] / 2.0f) + (vp[1] + (wc * (-yc * vp[3] / 2.0f)));
  *sz = vp[5] + (wc * (zc * (vp[5] - vp[4])));
}

// TODO GXLoadPosMtxIndx
// TODO GXLoadNrmMtxImm3x3
// TODO GXLoadNrmMtxIndx3x3
// TODO GXLoadTexMtxIndx
// TODO GXSetZScaleOffset
}
