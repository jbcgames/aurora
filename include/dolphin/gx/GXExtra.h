#ifndef DOLPHIN_GXEXTRA_H
#define DOLPHIN_GXEXTRA_H
// Extra types for PC
#ifdef TARGET_PC
#include <dolphin/gx/GXStruct.h>
#include <dolphin/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  float r;
  float g;
  float b;
  float a;
} GXColorF32;

void GXDestroyTexObj(GXTexObj* obj);
void GXDestroyTlutObj(GXTlutObj* obj);
void GXDestroyCopyTex(void* dest);
/* The copy texture at src also stands for dest (the game copied src's memory there). */
void GXAliasCopyTex(const void* src, const void* dest);

void GXColor4f32(float r, float g, float b, float a);

#ifdef __cplusplus
}
#endif
#endif

#endif
