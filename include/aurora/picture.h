#ifndef AURORA_PICTURE_H
#define AURORA_PICTURE_H

/* Picture effects over the presented picture, and a capture of it: the application's own (Paper Cut's
 * port/patches/aurora), for its paper filters and photo mode. */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AuroraPictureEffects {
  /* 0 none, 1 a cut-paper outline (ink round the shapes, a pale cut edge beside it), 2 folding light (the
   * picture on paper folded in panels, a slow light across them) */
  int paper;
  float paperStrength; /* 0-1 */
  /* a colour look: out = clamp(look * (r, g, b, 1)), rows r, g, b of four (three weights and an offset), on
   * gamma-encoded colour (0-1); lookOn 0: none */
  int lookOn;
  float look[12];
  float vignette; /* 0-1: the corners darker */
  /* depth of field: sharp round (focusX, focusY) (0-1 from the picture's top left) out to focusRadius (in
   * picture heights), blurred past it up to blur (in pixels of a 480-line picture); blur 0: none */
  float focusX, focusY, focusRadius, blur;
  /* 0 none, 1 a paper border, 2 a stamp's perforated edge, 3 a film strip, 4 a storybook's */
  int frame;
} AuroraPictureEffects;

/**
 * The effects for the frames from the next one (a pass of its own after the present resample, its post filter and
 * the colour matrix; not ImGui). NULL or all off: none (the default).
 */
void aurora_set_picture_effects(const AuroraPictureEffects* effects);

/**
 * Called once with the next presented picture as it is shown, effects and all (not ImGui): rgba, 4 bytes a pixel,
 * rows top to bottom, width x height at the presented size; or with rgba NULL (0 x 0) when it cannot be read
 * (an unusual surface format). On whatever thread the GPU's map callback comes on; rgba is valid only during the
 * call. Returns 0 (and never calls) if a capture is already waiting.
 */
typedef void (*AuroraPictureCallback)(void* user, const unsigned char* rgba, unsigned int width, unsigned int height);
int aurora_capture_picture(AuroraPictureCallback callback, void* user);

#ifdef __cplusplus
}
#endif

#endif
