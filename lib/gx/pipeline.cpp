#include "pipeline.hpp"

#include "../webgpu/gpu.hpp"
#include "gx_fmt.hpp"
#include "shader_info.hpp"
#include "tracy/Tracy.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace aurora::gx {
static Module Log("aurora::gx");

// The hardware vertex-attribute layout (native fetch) is built in Phase 3, where
// the draw path creates a VAO per layout hash and issues glVertexAttrib(I)Pointer
// in canonical GX attr order:
//   PNMTXIDX Uint32 -> TEX0-7MTXIDX Uint32 -> POS F32x3 -> NRM F32x3 ->
//   CLR0/1 Unorm8x4 -> TEX0-7 F32x2
// That order must stay byte-identical to shader.cpp's native inputs and the CPU
// expansion in command_processor. The old wgpu::VertexBufferLayout builder is gone;
// gl::Pipeline::vertexLayout carries the layout hash instead.

gl::Pipeline create_pipeline(const PipelineConfig& config) {
  ZoneScoped;
  const uint32_t program = build_shader(config.shaderConfig);
  const auto label = fmt::format("GX Pipeline {:x} shader {:x}",
                                 xxh3_hash(config, static_cast<HashType>(gfx::ShaderType::GX)),
                                 xxh3_hash(config.shaderConfig));
  return build_pipeline(config, program, label.c_str());
}

void render(const DrawData& data, gl::PassEncoder& pass) {
  static const bool s_detail = std::getenv("TTYD_TIMING_DETAIL") != nullptr;
  using Clock = std::chrono::steady_clock;
  static double s_msPipe = 0, s_msUni = 0, s_msTex = 0, s_msBuf = 0, s_msDraw = 0;
  static uint32_t s_draws = 0;
  const auto t0 = s_detail ? Clock::now() : Clock::time_point{};
  if (!gfx::bind_pipeline(data.pipeline, pass)) {
    return;
  }
  const auto t1 = s_detail ? Clock::now() : Clock::time_point{};

  // Group 1: the GX uniform block, bound directly with its per-draw range.
  pass.SetUniformBuffer0(gfx::g_uniformBuffer.id, data.uniformRange.offset,
                         static_cast<uint32_t>(data.uniformRange.size));
  const auto t2 = s_detail ? Clock::now() : Clock::time_point{};
  static gfx::BindGroupRef s_lastTextureBindGroup{};
  static uint64_t s_lastTexturePassId = 0;
  if (data.bindGroups.textureBindGroup &&
      (data.bindGroups.textureBindGroup != s_lastTextureBindGroup || pass.id() != s_lastTexturePassId)) {
    pass.SetBindGroup(2, gfx::find_bind_group(data.bindGroups.textureBindGroup));
    s_lastTextureBindGroup = data.bindGroups.textureBindGroup;
    s_lastTexturePassId = pass.id();
  }
  const auto t3 = s_detail ? Clock::now() : Clock::time_point{};
  if (data.nativeVertexFetch) {
    // Native draws read from a real vertex buffer instead of the group-0 SSBOs. Cached
    // geometry lives in the persistent cross-frame cache buffer; everything else in the
    // shared per-frame vertex ring.
    const gl::Buffer& vtxBuffer = data.cachedGeometry ? gfx::g_nativeVertexCacheBuffer : gfx::g_vertexBuffer;
    pass.SetVertexBuffer(0, vtxBuffer, data.vertRange.offset, data.vertRange.size);
  }
  const gl::Buffer& idxBuffer = data.cachedGeometry ? gfx::g_nativeIndexCacheBuffer : gfx::g_indexBuffer;
  pass.SetIndexBuffer(idxBuffer, gl::IndexFormat::Uint16, data.idxRange.offset, data.idxRange.size);
  if (data.dstAlpha != UINT32_MAX) {
    const gl::Color color{0.f, 0.f, 0.f, data.dstAlpha / 255.f};
    pass.SetBlendConstant(&color);
  }
  const auto t4 = s_detail ? Clock::now() : Clock::time_point{};
  if (data.indexCount == 0) {
    pass.Draw(data.vtxCount, data.instanceCount);
  } else {
    pass.DrawIndexed(data.indexCount, data.instanceCount);
  }
  if (s_detail) {
    const auto t5 = Clock::now();
    s_msPipe += std::chrono::duration<double, std::milli>(t1 - t0).count();
    s_msUni += std::chrono::duration<double, std::milli>(t2 - t1).count();
    s_msTex += std::chrono::duration<double, std::milli>(t3 - t2).count();
    s_msBuf += std::chrono::duration<double, std::milli>(t4 - t3).count();
    s_msDraw += std::chrono::duration<double, std::milli>(t5 - t4).count();
    if (++s_draws >= 4700) {
      const double f = 10.0 / s_draws; // per 10-frame scale ~470 draws -> per-frame ms normalized to 470 draws
      const double perFrame = 470.0 / s_draws;
      std::fprintf(stdout,
                   "[REN-DETAIL/470d] pipe=%.2fms uni=%.2fms tex=%.2fms buf=%.2fms draw=%.2fms tot=%.2fms\n",
                   s_msPipe * perFrame, s_msUni * perFrame, s_msTex * perFrame, s_msBuf * perFrame, s_msDraw * perFrame,
                   (s_msPipe + s_msUni + s_msTex + s_msBuf + s_msDraw) * perFrame);
      (void)f;
      s_msPipe = s_msUni = s_msTex = s_msBuf = s_msDraw = 0.0;
      s_draws = 0;
    }
  }
}
} // namespace aurora::gx
