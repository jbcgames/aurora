#include "gx.hpp"
#include "__gx.h"

#include <cmath>

extern "C" {
void GXSetFog(GXFogType type, float startZ, float endZ, float nearZ, float farZ, GXColor color) {
  // BP FOGCLR (0xF2) - color
  u32 fogclr = 0;
  SET_REG_FIELD(0, fogclr, 8, 0, color.b);
  SET_REG_FIELD(0, fogclr, 8, 8, color.g);
  SET_REG_FIELD(0, fogclr, 8, 16, color.r);
  SET_REG_FIELD(0, fogclr, 8, 24, 0xF2);

  const bool inDL = aurora::gx::fifo::in_display_list();
  const u32 ep = __gxShadow.epoch;
  if (!inDL && __gxShadow.fogParamEpoch == ep && __gxShadow.fogType == type && __gxShadow.fogStartZ == startZ &&
      __gxShadow.fogEndZ == endZ && __gxShadow.fogNearZ == nearZ && __gxShadow.fogFarZ == farZ) {
    __gx_write_shadow_ras(fogclr, __gxShadow.fogClr, __gxShadow.fogClrEpoch);
    return;
  }

  // Compute fog coefficients matching SDK
  f32 A, B, C;
  if (farZ == nearZ || endZ == startZ) {
    A = 0.0f;
    B = 0.5f;
    C = 0.0f;
  } else {
    A = (farZ * nearZ) / ((farZ - nearZ) * (endZ - startZ));
    B = farZ / (farZ - nearZ);
    C = startZ / (endZ - startZ);
  }

  // Normalize B mantissa
  f32 B_mant = B;
  u32 B_expn = 0;
  while (B_mant > 1.0f) {
    B_mant *= 0.5f;
    B_expn++;
  }
  while (B_mant > 0.0f && B_mant < 0.5f) {
    B_mant *= 2.0f;
    B_expn--;
  }

  f32 a = A / static_cast<f32>(1 << (B_expn + 1));
  u32 b_m = static_cast<u32>(8.388638e6f * B_mant);
  u32 b_s = B_expn + 1;
  f32 c = C;

  u32 a_hex, c_hex;
  std::memcpy(&a_hex, &a, sizeof(a_hex));
  std::memcpy(&c_hex, &c, sizeof(c_hex));

  // BP FOG0 (0xEE) - A parameter
  u32 fog0 = 0;
  SET_REG_FIELD(0, fog0, 11, 0, (a_hex >> 12) & 0x7FF);
  SET_REG_FIELD(0, fog0, 8, 11, (a_hex >> 23) & 0xFF);
  SET_REG_FIELD(0, fog0, 1, 19, (a_hex >> 31));
  SET_REG_FIELD(0, fog0, 8, 24, 0xEE);

  // BP FOG1 (0xEF) - B mantissa
  u32 fog1 = 0;
  SET_REG_FIELD(0, fog1, 24, 0, b_m);
  SET_REG_FIELD(0, fog1, 8, 24, 0xEF);

  // BP FOG2 (0xF0) - B scale
  u32 fog2 = 0;
  SET_REG_FIELD(0, fog2, 5, 0, b_s);
  SET_REG_FIELD(0, fog2, 8, 24, 0xF0);

  // BP FOG3 (0xF1) - C parameter + type
  u32 fog3 = 0;
  SET_REG_FIELD(0, fog3, 11, 0, (c_hex >> 12) & 0x7FF);
  SET_REG_FIELD(0, fog3, 8, 11, (c_hex >> 23) & 0xFF);
  SET_REG_FIELD(0, fog3, 1, 19, (c_hex >> 31));
  SET_REG_FIELD(0, fog3, 3, 21, type);
  SET_REG_FIELD(0, fog3, 8, 24, 0xF1);

  if (!inDL) {
    __gxShadow.fogParamEpoch = ep;
    __gxShadow.fogType = type;
    __gxShadow.fogStartZ = startZ;
    __gxShadow.fogEndZ = endZ;
    __gxShadow.fogNearZ = nearZ;
    __gxShadow.fogFarZ = farZ;
    __gxShadow.fogRegs[0] = fog0;
    __gxShadow.fogRegs[1] = fog1;
    __gxShadow.fogRegs[2] = fog2;
    __gxShadow.fogRegs[3] = fog3;
  }

  const bool writeClr = inDL || __gxShadow.fogClrEpoch != ep || __gxShadow.fogClr != fogclr;
  if (writeClr) {
    if (!inDL) {
      __gxShadow.fogClrEpoch = ep;
      __gxShadow.fogClr = fogclr;
    }
    const u32 regs[5] = {fog0, fog1, fog2, fog3, fogclr};
    __gx_write_ras_regs(regs);
  } else {
    const u32 regs[4] = {fog0, fog1, fog2, fog3};
    __gx_write_ras_regs(regs);
  }
  __gx->bpSent = 1;
}

void GXSetFogColor(GXColor color) {
  // BP FOGCLR (0xF2)
  u32 fogclr = 0;
  SET_REG_FIELD(0, fogclr, 8, 0, color.b);
  SET_REG_FIELD(0, fogclr, 8, 8, color.g);
  SET_REG_FIELD(0, fogclr, 8, 16, color.r);
  SET_REG_FIELD(0, fogclr, 8, 24, 0xF2);
  __gx_write_shadow_ras(fogclr, __gxShadow.fogClr, __gxShadow.fogClrEpoch);
}

void GXSetFogRangeAdj(GXBool enable, u16 center, GXFogAdjTable* table) {
  u32 i;
  u32 range_adj;
  u32 range_c;

  if (enable) {
    for (i = 0; i < 10; i += 2) {
      range_adj = 0;
      SET_REG_FIELD(0x10D, range_adj, 12, 0, table->r[i]);
      SET_REG_FIELD(0x10E, range_adj, 12, 12, table->r[i + 1]);
      SET_REG_FIELD(0x10F, range_adj, 8, 24, (i >> 1) + 0xE9);
      GX_WRITE_RAS_REG(range_adj);
    }
  }
  range_c = 0;
  SET_REG_FIELD(0x115, range_c, 10, 0, center + 342);
  SET_REG_FIELD(0x116, range_c, 1, 10, enable);
  SET_REG_FIELD(0x117, range_c, 8, 24, 0xE8);
  GX_WRITE_RAS_REG(range_c);
  __gx->bpSent = 1;
}

void GXSetBlendMode(GXBlendMode mode, GXBlendFactor src, GXBlendFactor dst, GXLogicOp op) {
  SET_REG_FIELD(0, __gx->cmode0, 1, 0, (mode == GX_BM_BLEND || mode == GX_BM_SUBTRACT));
  SET_REG_FIELD(0, __gx->cmode0, 1, 11, (mode == GX_BM_SUBTRACT));
  SET_REG_FIELD(0, __gx->cmode0, 1, 1, (mode == GX_BM_LOGIC));
  SET_REG_FIELD(0, __gx->cmode0, 4, 12, op);
  SET_REG_FIELD(0, __gx->cmode0, 3, 8, src);
  SET_REG_FIELD(0, __gx->cmode0, 3, 5, dst);
  __gx_write_shadow_ras(__gx->cmode0, __gxShadow.cmode0, __gxShadow.cmode0Epoch);
}

void GXSetColorUpdate(GXBool enabled) {
  SET_REG_FIELD(0, __gx->cmode0, 1, 3, enabled);
  __gx_write_shadow_ras(__gx->cmode0, __gxShadow.cmode0, __gxShadow.cmode0Epoch);
}

void GXSetAlphaUpdate(bool enabled) {
  SET_REG_FIELD(0, __gx->cmode0, 1, 4, enabled);
  __gx_write_shadow_ras(__gx->cmode0, __gxShadow.cmode0, __gxShadow.cmode0Epoch);
}

void GXSetZMode(bool compare_enable, GXCompare func, bool update_enable) {
  SET_REG_FIELD(0, __gx->zmode, 1, 0, compare_enable);
  SET_REG_FIELD(0, __gx->zmode, 3, 1, func);
  SET_REG_FIELD(0, __gx->zmode, 1, 4, update_enable);
  __gx_write_shadow_ras(__gx->zmode, __gxShadow.zmode, __gxShadow.zmodeEpoch);
}

void GXSetZCompLoc(GXBool before_tex) {
  SET_REG_FIELD(0, __gx->peCtrl, 1, 6, before_tex);
  __gx_write_shadow_ras(__gx->peCtrl, __gxShadow.peCtrl, __gxShadow.peCtrlEpoch);
}

void GXSetPixelFmt(GXPixelFmt pix_fmt, GXZFmt16 z_fmt) {
  u32 oldPeCtrl = __gx->peCtrl;
  u8 hwPixelFmt = 0;

  switch (pix_fmt) {
  case GX_PF_RGB8_Z24:
    hwPixelFmt = 0;
    break;
  case GX_PF_RGBA6_Z24:
    hwPixelFmt = 1;
    break;
  case GX_PF_RGB565_Z16:
    hwPixelFmt = 2;
    break;
  case GX_PF_Z24:
    hwPixelFmt = 3;
    break;
  case GX_PF_Y8:
  case GX_PF_U8:
  case GX_PF_V8:
    hwPixelFmt = 4;
    break;
  case GX_PF_YUV420:
    hwPixelFmt = 5;
    break;
  default:
    UNLIKELY FATAL("GXSetPixelFmt: unsupported pixel format {}", static_cast<u32>(pix_fmt));
  }

  SET_REG_FIELD(0, __gx->peCtrl, 3, 0, hwPixelFmt);
  SET_REG_FIELD(0, __gx->peCtrl, 3, 3, z_fmt);
  if (oldPeCtrl != __gx->peCtrl) {
    __gx_write_shadow_ras(__gx->peCtrl, __gxShadow.peCtrl, __gxShadow.peCtrlEpoch);
    SET_REG_FIELD(0, __gx->genMode, 1, 9, pix_fmt == GX_PF_RGB565_Z16);
    __gx->dirtyState |= 4;
  }

  if (hwPixelFmt == 4) {
    SET_REG_FIELD(0, __gx->cmode1, 2, 9, (static_cast<u32>(pix_fmt) - GX_PF_Y8) & 0x3);
    __gx_write_shadow_ras(__gx->cmode1, __gxShadow.cmode1, __gxShadow.cmode1Epoch);
  }

  __gx->bpSent = 1;
}

void GXSetDither(GXBool dither) {
  SET_REG_FIELD(0, __gx->cmode0, 1, 2, dither);
  __gx_write_shadow_ras(__gx->cmode0, __gxShadow.cmode0, __gxShadow.cmode0Epoch);
}

void GXSetDstAlpha(bool enabled, u8 value) {
  SET_REG_FIELD(0, __gx->cmode1, 8, 0, value);
  SET_REG_FIELD(0, __gx->cmode1, 1, 8, enabled);
  __gx_write_shadow_ras(__gx->cmode1, __gxShadow.cmode1, __gxShadow.cmode1Epoch);
}

void GXSetFieldMask(GXBool odd_mask, GXBool even_mask) {
  u32 reg;

  reg = 0;
  SET_REG_FIELD(0x1FB, reg, 1, 0, even_mask);
  SET_REG_FIELD(0x1FC, reg, 1, 1, odd_mask);
  SET_REG_FIELD(0x1FD, reg, 8, 24, 0x44);
  GX_WRITE_RAS_REG(reg);
  __gx->bpSent = 1;
}

void GXSetFieldMode(GXBool field_mode, GXBool half_aspect_ratio) {
  u32 reg;

  SET_REG_FIELD(0x21A, __gx->lpSize, 1, 22, half_aspect_ratio);
  GX_WRITE_RAS_REG(__gx->lpSize);
  __GXFlushTextureState();

  SET_REG_FIELD(0, reg, 8, 24, 0x68);
  SET_REG_FIELD(0, reg, 1, 0, field_mode);
  GX_WRITE_RAS_REG(reg);
  __GXFlushTextureState();
}
}
