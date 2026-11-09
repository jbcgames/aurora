#ifndef AURORA_COLOR_H
#define AURORA_COLOR_H

/* A colour matrix over the presented picture, the application's own (Paper Cut's port/patches/aurora): for
 * colour-blind corrections (daltonization). */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Each presented pixel's gamma-encoded r, g, b (0-1) multiplied by the row-major 3x3 matrix (out = matrix * in),
 * clamped to 0-1, in a pass of its own after the present resample and its post filter (aurora_set_post_filter);
 * not ImGui. NULL: none (the default). Applies from the next frame.
 */
void aurora_set_color_matrix(const float* matrix);

#ifdef __cplusplus
}
#endif

#endif
