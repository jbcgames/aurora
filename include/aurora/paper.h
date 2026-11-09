#ifndef AURORA_PAPER_H
#define AURORA_PAPER_H

/* Draw-only paper effects (Paper Cut's port/patches/aurora): smooth cut-out edges, a distance-field font, and
 * shadow draws. Each goes through the GX command stream, so it applies from the next draw, in order with the
 * draws around it. None of them changes anything the game reads back. */

#ifdef __cplusplus
extern "C" {
#endif

/* GX command: u8 edges, u8 shadow, u64 distance-field texture data address */
#define GX_AURORA_SET_PAPER 0x0060

/**
 * Smooth paper edges: an alpha-tested draw (a cut-out: GX_GREATER / GX_GEQUAL against a real threshold) gets an
 * edge one pixel wide instead of a hard step: as coverage of the samples with MSAA when it doesn't blend, else
 * faded into its alpha when it blends with source alpha. Other draws are as before. 0: off (the default).
 */
void aurora_set_paper_edges(int on);

/**
 * The texture whose data is at texels (as given to GXInitTexObj; usually served by a replacement) holds a
 * distance field in every channel: 0.5 is the edge, more inside. Each draw sampling it thresholds the distance by
 * its own pixel size, so its shapes stay sharp at any resolution. NULL: none (the default).
 */
void aurora_set_paper_distance_field(const void* texels);

/**
 * Shadow draws: while on, every draw is a translucent dark shadow of itself (its alpha, cut-outs and all, at
 * about half strength), alpha blended, depth tested and written with GX_LESS (so overlapping parts darken once),
 * culling nothing. The game's own state is left as it was; off restores its draws.
 */
void aurora_set_paper_shadow(int on);

#ifdef __cplusplus
}
#endif

#endif
