#ifndef AURORA_HDR_H
#define AURORA_HDR_H

/* HDR output (Paper Cut's port/patches/aurora): the SDR picture presented through a float16 extended-sRGB
 * swapchain at a chosen paper-white brightness, where the display and backend can show it. */

#ifdef __cplusplus
#include <cstdbool>
extern "C" {
#else
#include <stdbool.h>
#endif

/**
 * HDR output can be used now: the window is on a display with HDR on (SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN) and the
 * backend presents a float16 swapchain in extended sRGB (D3D12/D3D11: scRGB; Metal: an EDR layer). Vulkan and
 * OpenGL swapchains are SDR only. Changes with the display the window is on (SDL_EVENT_WINDOW_HDR_STATE_CHANGED).
 */
bool aurora_hdr_supported(void);

/**
 * HDR output: when supported, the swapchain becomes float16 extended sRGB and the SDR picture (game and UI) is
 * linearised and scaled so its white is paperWhiteNits (80-1000; scRGB 1.0 = 80 nits; on Metal the system's white
 * is taken as 100 nits), then what is near white is raised by up to highlightBoost (0-1, a fraction of paper
 * white), never past the display's headroom. Otherwise, or when disabled, SDR output as before. Default off.
 */
void aurora_set_hdr(bool enabled, float paperWhiteNits, float highlightBoost);

/** The swapchain is presenting in HDR now (aurora_set_hdr on and supported). */
bool aurora_hdr_active(void);

#ifdef __cplusplus
}
#endif

#endif
