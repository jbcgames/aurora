#ifndef AURORA_UPSCALE_H
#define AURORA_UPSCALE_H

/* Texture upscaling by the application's own filter (Paper Cut's port/patches/aurora). */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Each texture the game loads from its data (not a copy of the framebuffer, not a replacement, not one
 * uploaded as it is, such as a block-compressed one), up to 512 texels on a side, is uploaded factor (2 or 4)
 * times larger: its first level decoded to RGBA8 and given to upscaler (out: factor*width x factor*height
 * RGBA8), the smaller levels averaged down from the result. factor 1 or upscaler NULL: off. Applies to
 * textures loaded after the call.
 */
typedef void (*AuroraTextureUpscaler)(const unsigned char* rgba, unsigned width, unsigned height, unsigned factor,
                                      unsigned char* out);
void AuroraSetTextureUpscale(unsigned factor, AuroraTextureUpscaler upscaler);

#ifdef __cplusplus
}
#endif

#endif
