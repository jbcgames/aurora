#ifndef AURORA_DISPLAY_H
#define AURORA_DISPLAY_H

/* Display options of the application's own (Paper Cut's port/patches/aurora): a HUD transform, texture
 * filtering, supersampling, a filter over the presented picture and variable refresh rate. */

#include <dolphin/types.h>

#ifdef __cplusplus
#include <cstdbool>
extern "C" {
#else
#include <stdbool.h>
#endif

/**
 * Scales and offsets clip-space x and y of every following draw to the main framebuffer (not offscreen
 * targets), before GX_AURORA_SET_PROJECTION_SCALE_X, and its scissor boxes with them: a HUD drawn larger or
 * smaller about a point. Must be followed by four f32s (scale x, scale y, offset x, offset y).
 */
#define GX_AURORA_SET_PROJECTION_TRANSFORM 0x0005

/**
 * x' = x * scaleX + offsetX, y' = y * scaleY + offsetY in clip space (offsets in NDC units) for the following
 * draws to the main framebuffer, and their scissor boxes alike. (1, 1, 0, 0) = off.
 */
void GXSetProjectionTransform(f32 scaleX, f32 scaleY, f32 offsetX, f32 offsetY);

/**
 * Anisotropic filtering of every mipmapped texture the game samples: at least maxAniso (1, 2, 4, 8 or 16;
 * 1 = the game's own). Not with sharp textures (AuroraSetSharpTextures). Applies from the next draw.
 */
void AuroraSetTextureAnisotropy(u8 maxAniso);

/**
 * Added to every texture's LOD bias (negative: sharper, from finer mip levels). 0 = the game's own.
 */
void AuroraSetTextureLodBias(f32 bias);

/** Supersampling: the framebuffer factor (1-4) times its size, area-downsampled to the window. Default 1. */
void aurora_set_supersample(int factor);

/** A filter over the presented picture (not ImGui): 0 none, 1 CRT, 2 paper grain, 3 sharpen. Default 0. */
void aurora_set_post_filter(int mode);

/**
 * Variable refresh rate: present without waiting for vblank (Immediate) whatever vsync says, so a VRR display
 * shows each frame when it comes; the application paces. Default false.
 */
void aurora_set_vrr(bool enabled);

#ifdef __cplusplus
}
#endif

#endif
