#pragma once

#include "gx.hpp"

namespace aurora::gx {
ShaderInfo build_shader_info(const ShaderConfig& config) noexcept;
float current_tex_lod_bias(u32 texMapId) noexcept;
gfx::Range build_uniform(const ShaderInfo& info, uint32_t vtxStart, const BindGroupRanges& ranges) noexcept;
gfx::Range build_merged_uniform(const ShaderInfo& baseInfo, uint32_t baseUniformOffset, uint32_t baseUniformSize,
                                const Mat3x4<float>* posMtxs, const Mat3x4<float>* nrmMtxs,
                                uint32_t mtxCount, const float* texBatchBiases = nullptr,
                                uint32_t texSlotCount = 0) noexcept;
u8 color_channel(GXChannelID id) noexcept;
}; // namespace aurora::gx
