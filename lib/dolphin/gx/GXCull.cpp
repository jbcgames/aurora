#include "gx.hpp"
#include "__gx.h"

extern "C" {

void GXSetScissor(u32 left, u32 top, u32 width, u32 height) {
  const u32 tp = top + 342;
  const u32 lf = left + 342;
  const u32 bm = tp + height - 1;
  const u32 rt = lf + width - 1;

  SET_REG_FIELD(0, __gx->suScis0, 11, 0, tp);
  SET_REG_FIELD(0, __gx->suScis0, 11, 12, lf);
  SET_REG_FIELD(0, __gx->suScis1, 11, 0, bm);
  SET_REG_FIELD(0, __gx->suScis1, 11, 12, rt);

  __gx_write_shadow_ras_pair(__gx->suScis0, __gx->suScis1, __gxShadow.suScis0, __gxShadow.suScis1,
                             __gxShadow.suScisEpoch);
}

void GXSetScissorBoxOffset(s32 x_off, s32 y_off) {
  (void)x_off;
  (void)y_off;
}

void GXSetCullMode(GXCullMode mode) {
  // Swap front/back to match hardware convention
  GXCullMode hwMode;
  switch (mode) {
  case GX_CULL_FRONT:
    hwMode = GX_CULL_BACK;
    break;
  case GX_CULL_BACK:
    hwMode = GX_CULL_FRONT;
    break;
  default:
    hwMode = mode;
    break;
  }
  SET_REG_FIELD(0, __gx->genMode, 2, 14, hwMode);
  __gx->dirtyState |= 4; // gen mode dirty
}

void GXSetClipMode(GXClipMode mode) {
  __gx_write_shadow_xf(5, static_cast<u32>(mode), __gxShadow.xfClipMode, __gxShadow.xfClipModeEpoch);
  __gx->bpSent = 1;
}

void GXSetCoPlanar(GXBool enable) {
  u32 reg;
  SET_REG_FIELD(0, __gx->genMode, 1, 19, enable);

  // This sets a mask that causes only one bit to be updated in genMode on the next write.
  reg = 0x0008000;
  SET_REG_FIELD(0, reg, 8, 24, 0xFE);
  SET_REG_FIELD(0, reg, 24, 0, 1 << 19);
  GX_WRITE_RAS_REG(reg);

  // immediately flush
  GX_WRITE_RAS_REG(__gx->genMode);
  if (!aurora::gx::fifo::in_display_list()) {
    __gxShadow.genModeEpoch = __gxShadow.epoch;
    __gxShadow.genMode = __gx->genMode;
  }
}
}
