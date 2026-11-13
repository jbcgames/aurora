#include "gx.hpp"

#include "pipeline.hpp"
#include "../dolphin/vi/vi_internal.hpp"
#include "../webgpu/gpu.hpp"
#include "../internal.hpp"
#include "../gfx/common.hpp"
#include "../gfx/hash.hpp"
#include "../gfx/tex_palette_conv.hpp"
#include "../gfx/texture.hpp"
#include "../gfx/texture_convert.hpp"
#include "../gfx/texture_replacement.hpp"
#include "gx_fmt.hpp"

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <tracy/Tracy.hpp>

#include <atomic>
#include <bit>
#include <cfloat>
#include <cmath>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

static aurora::Module Log("aurora::gx");

namespace aurora::gx {
using webgpu::g_graphicsConfig;

GXState g_gxState{};

// The 1x1 fills bound into unused GX texture slots. GL has no separate
// bind-group-layout / pipeline-layout objects (Dawn concepts), so those are gone;
// binding is expressed directly as a gl::BindingSet. Real GL objects for the
// empties are created on the render worker in Phase 2.
static gl::Texture sEmptyTexture;
static gl::Sampler sEmptySampler;
gl::BindingSet g_emptyTextureBindGroup;

namespace {
struct DynamicPaletteKey {
  const void* sourceIdentity = nullptr;
  u32 width = 0;
  u32 height = 0;
  u32 format = 0;

  bool operator==(const DynamicPaletteKey& rhs) const = default;
  template <typename H>
  friend H AbslHashValue(H h, const DynamicPaletteKey& key) {
    return H::combine(std::move(h), key.sourceIdentity, key.width, key.height, key.format);
  }
};

struct DynamicPaletteEntry {
  gfx::TextureHandle handle;
  u32 sourceRevision = 0;
  u32 tlutDataVersion = 0;
  u32 lastUsedFrame = 0;
};

struct CachedTextureEntry {
  gfx::TextureHandle handle;
  gfx::TextureBind bind;
  const void* data = nullptr;
  u32 width = 0;
  u32 height = 0;
  u32 format = 0;
  u32 mipCount = 0;
  u32 texDataVersion = 0;
  u32 tlutObjId = 0;
  u32 tlutDataVersion = 0;
  u32 mode0 = 0;
  u32 mode1 = 0;
  u32 lastUsedFrame = 0;

  bool matches(const GXTexObj_& obj, u32 expectedTlutObjId, u32 expectedTlutDataVersion) const noexcept {
    return handle && data == obj.data && width == obj.width() && height == obj.height() &&
           format == static_cast<u32>(obj.format()) && mipCount == obj.mip_count() &&
           texDataVersion == obj.texDataVersion && tlutObjId == expectedTlutObjId &&
           tlutDataVersion == expectedTlutDataVersion;
  }
};

struct CachedTlutTextureEntry {
  gfx::TextureHandle handle;
  u32 tlutDataVersion = 0;
  u32 lastUsedFrame = 0;
};

struct TlutObjectCache {
  CachedTlutTextureEntry tlutTexture;
  absl::flat_hash_map<DynamicPaletteKey, DynamicPaletteEntry> dynamicPaletteTextures;
  absl::flat_hash_set<u32> staticTextureUsers;
};

struct StaticContentKey {
  u32 width = 0;
  u32 height = 0;
  u32 format = 0;
  u32 mips = 0;
  u32 tlutFormat = 0;
  u32 tlutEntries = 0;
  u64 dataHash = 0;
  u64 tlutHash = 0;

  bool operator==(const StaticContentKey& o) const = default;

  template <typename H>
  friend H AbslHashValue(H h, const StaticContentKey& k) {
    return H::combine(std::move(h), k.width, k.height, k.format, k.mips, k.tlutFormat, k.tlutEntries, k.dataHash, k.tlutHash);
  }
};

static absl::flat_hash_map<StaticContentKey, gfx::TextureHandle> s_staticContentCache;

struct PointerTextureKey {
  const void* data = nullptr;
  const void* tlutData = nullptr;
  u32 image0 = 0;
  u32 format = 0;
  u32 mipCount = 0;
  u32 texDataVersion = 0;
  u32 tlutObjId = 0;
  u32 tlutDataVersion = 0;

  bool operator==(const PointerTextureKey& o) const = default;

  template <typename H>
  friend H AbslHashValue(H h, const PointerTextureKey& k) {
    return H::combine(std::move(h), k.data, k.tlutData, k.image0, k.format, k.mipCount, k.texDataVersion, k.tlutObjId,
                      k.tlutDataVersion);
  }
};

struct PointerTextureEntry {
  gfx::TextureHandle handle;
  u32 lastUsedFrame = 0;
};

static absl::flat_hash_map<PointerTextureKey, PointerTextureEntry> s_pointerTextureCache;

absl::flat_hash_map<u32, CachedTextureEntry> s_textureObjectCaches;
absl::flat_hash_map<u32, TlutObjectCache> s_tlutObjectCaches;
struct FastResolvedTexture {
  u32 texObjId = 0;
  u32 texDataVersion = 0;
  const void* data = nullptr;
  u32 image0 = 0;
  u32 format = 0;
  u32 mipCount = 0;
  u32 mode0 = 0;
  u32 mode1 = 0;
  u32 tlutObjId = 0;
  u32 tlutDataVersion = 0;
  const void* tlutData = nullptr;
  gfx::TextureBind bind{};
};
static constexpr size_t kFastResolvedBits = 11; // 2048 entries
static constexpr size_t kFastResolvedSize = 1 << kFastResolvedBits;
static std::array<FastResolvedTexture, kFastResolvedSize> s_fastResolvedTextures{};

static inline size_t fast_tex_slot(u32 id) noexcept {
  return ((static_cast<uint32_t>(id) * 2654435769u) >> (32 - kFastResolvedBits)) & (kFastResolvedSize - 1);
}

struct FastResolvedPointerTexture {
  const void* data = nullptr;
  const void* tlutData = nullptr;
  u32 image0 = 0;
  u32 format = 0;
  u32 mipCount = 0;
  u32 texDataVersion = 0;
  u32 tlutObjId = 0;
  u32 tlutDataVersion = 0;
  u32 mode0 = 0;
  u32 mode1 = 0;
  gfx::TextureBind bind{};
};
static constexpr size_t kFastResolvedPtrBits = 10; // 1024 entries
static constexpr size_t kFastResolvedPtrSize = 1 << kFastResolvedPtrBits;
static std::array<FastResolvedPointerTexture, kFastResolvedPtrSize> s_fastResolvedPtrTextures{};

static inline size_t fast_ptr_slot(const void* data, u32 image0) noexcept {
  const uint32_t pval = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(data) >> 4) ^ image0;
  return ((pval * 2654435769u) >> (32 - kFastResolvedPtrBits)) & (kFastResolvedPtrSize - 1);
}

struct LastResolvedMeta {
  bool valid = false;
  bool isCopy = false;
  u32 texObjId = 0;
  u32 texDataVersion = 0;
  u32 tlutObjId = 0;
  u32 tlutDataVersion = 0;
  const void* tlutData = nullptr;
  const void* data = nullptr;
  u32 image0 = 0;
  u32 format = 0;
  u32 mipCount = 0;
  u32 mode0 = 0;
  u32 mode1 = 0;
};
static std::array<LastResolvedMeta, MaxTextures> s_lastResolvedMeta{};

std::atomic_bool s_staticTextureCacheClearPending = false;

void clear_texture_dependency(u32 texObjId, u32 tlutObjId);

void invalidate_fast_tex_id(u32 texObjId) noexcept {
  if (texObjId == 0) {
    return;
  }
  const size_t slot = fast_tex_slot(texObjId);
  if (s_fastResolvedTextures[slot].texObjId == texObjId) {
    s_fastResolvedTextures[slot] = {};
  }
  for (auto& meta : s_lastResolvedMeta) {
    if (meta.valid && meta.texObjId == texObjId) {
      meta = {};
    }
  }
}

void invalidate_data_caches(const void* data) noexcept {
  if (data == nullptr) {
    return;
  }
  for (auto it = s_textureObjectCaches.begin(); it != s_textureObjectCaches.end();) {
    if (it->second.data == data) {
      clear_texture_dependency(it->first, it->second.tlutObjId);
      s_textureObjectCaches.erase(it++);
    } else {
      ++it;
    }
  }
  absl::erase_if(s_pointerTextureCache, [&](const auto& kv) { return kv.first.data == data; });
  for (auto& pfast : s_fastResolvedPtrTextures) {
    if (pfast.data == data) {
      pfast = {};
    }
  }
  for (auto& fast : s_fastResolvedTextures) {
    if (fast.data == data) {
      fast = {};
    }
  }
  for (auto& meta : s_lastResolvedMeta) {
    if (meta.valid && meta.data == data) {
      meta = {};
    }
  }
}

void invalidate_tlut_dependents(u32 tlutObjId, const void* tlutData = nullptr) noexcept {
  for (auto& fast : s_fastResolvedTextures) {
    if ((tlutObjId != 0 && fast.tlutObjId == tlutObjId) ||
        (tlutData != nullptr && fast.tlutData == tlutData)) {
      fast = {};
    }
  }
  for (auto& meta : s_lastResolvedMeta) {
    if (meta.valid &&
        ((tlutObjId != 0 && meta.tlutObjId == tlutObjId) ||
         (tlutData != nullptr && meta.tlutData == tlutData))) {
      meta = {};
    }
  }
  absl::erase_if(s_pointerTextureCache, [&](const auto& kv) {
    return (tlutObjId != 0 && kv.first.tlutObjId == tlutObjId) ||
           (tlutData != nullptr && kv.first.tlutData == tlutData);
  });
  for (auto& pfast : s_fastResolvedPtrTextures) {
    if ((tlutObjId != 0 && pfast.tlutObjId == tlutObjId) ||
        (tlutData != nullptr && pfast.tlutData == tlutData)) {
      pfast = {};
    }
  }
}

template <typename Pred>
void invalidate_caches_if(const Pred& in) noexcept {
  for (auto it = s_textureObjectCaches.begin(); it != s_textureObjectCaches.end();) {
    if (in(it->second.data)) {
      clear_texture_dependency(it->first, it->second.tlutObjId);
      s_textureObjectCaches.erase(it++);
    } else {
      ++it;
    }
  }
  absl::erase_if(s_pointerTextureCache, [&](const auto& kv) { return in(kv.first.data) || in(kv.first.tlutData); });
  for (auto& pfast : s_fastResolvedPtrTextures) {
    if (in(pfast.data) || in(pfast.tlutData)) {
      pfast = {};
    }
  }
  for (auto& fast : s_fastResolvedTextures) {
    if (in(fast.data) || in(fast.tlutData)) {
      fast = {};
    }
  }
  for (auto& meta : s_lastResolvedMeta) {
    if (meta.valid && (in(meta.data) || in(meta.tlutData))) {
      meta = {};
    }
  }
}

void do_clear_static_texture_cache() noexcept {
  s_lastResolvedMeta.fill({});
  s_fastResolvedTextures.fill({});
  s_fastResolvedPtrTextures.fill({});
  s_textureObjectCaches.clear();
  s_staticContentCache.clear();
  s_pointerTextureCache.clear();
  for (auto& [_, cache] : s_tlutObjectCaches) {
    cache.staticTextureUsers.clear();
  }
}

DynamicPaletteKey make_dynamic_palette_key(const GXTexObj_& obj, const GXState::CopyTextureRef& source) {
  return {
      .sourceIdentity = source.handle.get(),
      .width = obj.width(),
      .height = obj.height(),
      .format = obj.format(),
  };
}

void clear_texture_dependency(u32 texObjId, u32 tlutObjId) {
  if (texObjId == 0 || tlutObjId == 0) {
    return;
  }
  if (auto it = s_tlutObjectCaches.find(tlutObjId); it != s_tlutObjectCaches.end()) {
    it->second.staticTextureUsers.erase(texObjId);
    if (!it->second.tlutTexture.handle && it->second.dynamicPaletteTextures.empty() &&
        it->second.staticTextureUsers.empty()) {
      s_tlutObjectCaches.erase(it);
    }
  }
}

void store_cached_texture(const GXTexObj_& obj, gfx::TextureHandle handle, u32 tlutObjId = 0, u32 tlutDataVersion = 0) {
  if (obj.texObjId == 0) {
    return;
  }

  auto& entry = s_textureObjectCaches[obj.texObjId];
  if (entry.tlutObjId != tlutObjId) {
    clear_texture_dependency(obj.texObjId, entry.tlutObjId);
  }

  entry.handle = std::move(handle);
  entry.bind = {};
  entry.data = obj.data;
  entry.width = obj.width();
  entry.height = obj.height();
  entry.format = static_cast<u32>(obj.format());
  entry.mipCount = obj.mip_count();
  entry.texDataVersion = obj.texDataVersion;
  entry.tlutObjId = tlutObjId;
  entry.tlutDataVersion = tlutDataVersion;
  entry.mode0 = 0;
  entry.mode1 = 0;
  entry.lastUsedFrame = gfx::current_frame();

  if (tlutObjId != 0) {
    s_tlutObjectCaches[tlutObjId].staticTextureUsers.insert(obj.texObjId);
  }
}

gfx::TextureHandle get_tlut_texture(const GXTlutObj_& tlut) {
  if (tlut.tlutObjId != 0) {
    auto& cache = s_tlutObjectCaches[tlut.tlutObjId];
    if (cache.tlutTexture.handle && cache.tlutTexture.tlutDataVersion == tlut.tlutDataVersion) {
      cache.tlutTexture.lastUsedFrame = gfx::current_frame();
      return cache.tlutTexture.handle;
    }
    cache.dynamicPaletteTextures.clear();
    for (const u32 texObjId : cache.staticTextureUsers) {
      s_textureObjectCaches.erase(texObjId);
      invalidate_fast_tex_id(texObjId);
    }
    cache.staticTextureUsers.clear();
    invalidate_tlut_dependents(tlut.tlutObjId, tlut.data);
  }

  const auto handle = gfx::new_static_texture_2d(
      tlut.numEntries, 1, 1, gfx::tlut_texture_format(tlut.format),
      {static_cast<const u8*>(tlut.data), static_cast<size_t>(tlut.numEntries) * sizeof(u16)}, true, "Loaded TLUT");
  if (tlut.tlutObjId != 0) {
    auto& cache = s_tlutObjectCaches[tlut.tlutObjId];
    cache.tlutTexture.handle = handle;
    cache.tlutTexture.tlutDataVersion = tlut.tlutDataVersion;
    cache.tlutTexture.lastUsedFrame = gfx::current_frame();
  }
  return handle;
}

gfx::TextureHandle resolve_static_texture(const GXTexObj_& obj) {
  ZoneScoped;
  if (__builtin_expect(s_staticTextureCacheClearPending.load(std::memory_order_relaxed), 0)) {
    if (s_staticTextureCacheClearPending.exchange(false, std::memory_order_acq_rel)) {
      do_clear_static_texture_cache();
    }
  }

  if (obj.texObjId != 0) {
    if (const auto it = s_textureObjectCaches.find(obj.texObjId); it != s_textureObjectCaches.end()) {
      auto& entry = it->second;
      if (entry.matches(obj, 0, 0)) {
        entry.lastUsedFrame = gfx::current_frame();
        return entry.handle;
      }
    }
  }

  const PointerTextureKey pkey{obj.data, nullptr, obj.image0, static_cast<u32>(obj.format()), obj.mip_count(),
                               obj.texDataVersion, 0, 0};
  if (obj.data != nullptr) {
    if (const auto it = s_pointerTextureCache.find(pkey); it != s_pointerTextureCache.end() && it->second.handle) {
      it->second.lastUsedFrame = gfx::current_frame();
      return it->second.handle;
    }
  }

  const u32 baseBytes = GXGetTexBufferSize(obj.width(), obj.height(), obj.format(), false, 0);
  const u64 dataHash = (obj.data && baseBytes > 0) ? xxh3_hash_s(obj.data, baseBytes) : 0;
  const StaticContentKey contentKey{
      .width = obj.width(),
      .height = obj.height(),
      .format = obj.format(),
      .mips = obj.mip_count(),
      .dataHash = dataHash,
  };

  if (const auto it = s_staticContentCache.find(contentKey); it != s_staticContentCache.end()) {
    if (it->second) {
      if (!obj.no_cache()) {
        if (obj.data != nullptr) {
          s_pointerTextureCache[pkey] = {it->second, gfx::current_frame()};
        }
        store_cached_texture(obj, it->second);
      }
      return it->second;
    }
  }

  gfx::TextureHandle handle;
  if (const auto replacement = gfx::texture_replacement::find_replacement(obj); replacement.has_value()) {
    handle = *replacement;
  } else {
#if DEBUG
    const auto name = gfx::texture_replacement::build_texture_replacement_name(obj);
    const auto nameStr = name.c_str();
#else
    const auto nameStr = "GX Static Texture";
#endif
    handle = gfx::new_static_texture_2d(obj.width(), obj.height(), obj.mip_count(), obj.format(),
                                        {static_cast<const uint8_t*>(obj.data), UINT32_MAX}, false, nameStr);
  }
  if (handle) {
    s_staticContentCache[contentKey] = handle;
    if (!obj.no_cache()) {
      if (obj.data != nullptr) {
        s_pointerTextureCache[pkey] = {handle, gfx::current_frame()};
      }
      store_cached_texture(obj, handle);
    }
  }
  return handle;
}

gfx::TextureHandle resolve_static_palette_texture(const GXTexObj_& obj, const GXTlutObj_& tlut) {
  ZoneScoped;
  if (__builtin_expect(s_staticTextureCacheClearPending.load(std::memory_order_relaxed), 0)) {
    if (s_staticTextureCacheClearPending.exchange(false, std::memory_order_acq_rel)) {
      do_clear_static_texture_cache();
    }
  }

  if (obj.texObjId != 0) {
    if (const auto it = s_textureObjectCaches.find(obj.texObjId); it != s_textureObjectCaches.end()) {
      auto& entry = it->second;
      if (entry.matches(obj, tlut.tlutObjId, tlut.tlutDataVersion)) {
        entry.lastUsedFrame = gfx::current_frame();
        return entry.handle;
      }
    }
  }

  const PointerTextureKey pkey{obj.data, tlut.data, obj.image0, static_cast<u32>(obj.format()), obj.mip_count(),
                               obj.texDataVersion, tlut.tlutObjId, tlut.tlutDataVersion};
  if (obj.data != nullptr) {
    if (const auto it = s_pointerTextureCache.find(pkey); it != s_pointerTextureCache.end() && it->second.handle) {
      it->second.lastUsedFrame = gfx::current_frame();
      return it->second.handle;
    }
  }

  const u32 baseBytes = GXGetTexBufferSize(obj.width(), obj.height(), obj.format(), false, 0);
  const u64 dataHash = (obj.data && baseBytes > 0) ? xxh3_hash_s(obj.data, baseBytes) : 0;
  const u64 tlutHash = (tlut.data && tlut.numEntries > 0) ? xxh3_hash_s(tlut.data, static_cast<size_t>(tlut.numEntries) * 2) : 0;
  const StaticContentKey contentKey{
      .width = obj.width(),
      .height = obj.height(),
      .format = obj.format(),
      .mips = obj.mip_count(),
      .tlutFormat = static_cast<u32>(tlut.format),
      .tlutEntries = tlut.numEntries,
      .dataHash = dataHash,
      .tlutHash = tlutHash,
  };

  if (const auto it = s_staticContentCache.find(contentKey); it != s_staticContentCache.end()) {
    if (it->second) {
      if (!obj.no_cache() && !tlut.no_cache()) {
        if (obj.data != nullptr) {
          s_pointerTextureCache[pkey] = {it->second, gfx::current_frame()};
        }
        store_cached_texture(obj, it->second, tlut.tlutObjId, tlut.tlutDataVersion);
      }
      return it->second;
    }
  }

  gfx::TextureHandle handle;
  if (const auto replacement = gfx::texture_replacement::find_replacement(obj, tlut); replacement.has_value()) {
    handle = *replacement;
  } else {
    auto converted = gfx::convert_texture_palette(
        obj.format(), obj.width(), obj.height(), obj.mip_count(), {static_cast<const u8*>(obj.data), UINT32_MAX},
        tlut.format, tlut.numEntries, {static_cast<const u8*>(tlut.data), static_cast<size_t>(tlut.numEntries) * 2});
    if (converted.data.empty()) {
      return {};
    }
    handle =
        gfx::new_static_texture_2d(obj.width(), obj.height(), obj.mip_count(), GX_TF_RGBA8_PC,
                                   {converted.data.data(), converted.data.size()}, false, "GX Static Palette Texture");
    handle->hasArbitraryMips = converted.hasArbitraryMips;
  }
  if (handle) {
    s_staticContentCache[contentKey] = handle;
    if (!obj.no_cache() && !tlut.no_cache()) {
      if (obj.data != nullptr) {
        s_pointerTextureCache[pkey] = {handle, gfx::current_frame()};
      }
      store_cached_texture(obj, handle, tlut.tlutObjId, tlut.tlutDataVersion);
    }
  }
  return handle;
}

gfx::TextureHandle resolve_dynamic_palette_texture(const GXTexObj_& obj, const GXState::CopyTextureRef& source,
                                                   const GXTlutObj_& tlut) {
  ZoneScoped;

  const auto tlutHandle = get_tlut_texture(tlut);
  auto& tlutCache = s_tlutObjectCaches[tlut.tlutObjId];
  auto& entry = tlutCache.dynamicPaletteTextures[make_dynamic_palette_key(obj, source)];
  entry.lastUsedFrame = gfx::current_frame();
  if (!entry.handle) {
    // Use source size instead of target (logical) size
    entry.handle = gfx::new_conv_texture(source.handle->size.width, source.handle->size.height, GX_TF_RGBA8,
                                         "GX Dynamic Palette Texture");
  }
  if (entry.sourceRevision != source.revision || entry.tlutDataVersion != tlut.tlutDataVersion) {
    gfx::queue_palette_conv({
        .variant = obj.format() == GX_TF_C4 ? gfx::tex_palette_conv::Variant::FromFloat4
                                            : gfx::tex_palette_conv::Variant::FromFloat8,
        .src = source.handle,
        .dst = entry.handle,
        .tlut = tlutHandle,
    });
    entry.sourceRevision = source.revision;
    entry.tlutDataVersion = tlut.tlutDataVersion;
  }
  return entry.handle;
}

u32 resolved_format_for_handle(const gfx::TextureHandle& handle) {
  if (!handle) {
    return GX_TF_RGBA8;
  }
  if (handle->gxFormat != gfx::InvalidTextureFormat) {
    return handle->gxFormat;
  }
  return GX_TF_RGBA8_PC;
}

template <typename T>
T round_away_from_zero(float value) noexcept {
  return static_cast<T>(value < 0.0f ? std::floor(value) : std::ceil(value));
}

std::pair<f32, f32> polygon_offset_for_cull_mode(GXCullMode cullMode) noexcept {
  if (cullMode == GX_CULL_FRONT) {
    return {g_gxState.backOffset, g_gxState.backScale};
  }
  return {g_gxState.frontOffset, g_gxState.frontScale};
}
} // namespace

Vec2<uint32_t> logical_fb_size() noexcept {
  return gfx::is_offscreen() ? gfx::get_render_target_size() : vi::configured_fb_size();
}

gfx::Viewport map_logical_viewport(const gfx::Viewport& logicalViewport) noexcept {
  if (g_gxState.viewportPolicy == AURORA_VIEWPORT_NATIVE) {
    return logicalViewport;
  }

  const auto [logicalFbWidth, logicalFbHeight] = logical_fb_size();
  const auto [targetWidth, targetHeight] = gfx::get_render_target_size();
  if (logicalFbWidth == 0 || logicalFbHeight == 0 || targetWidth == 0 || targetHeight == 0) {
    return logicalViewport;
  }

  const float scaleX = static_cast<float>(targetWidth) / static_cast<float>(logicalFbWidth);
  const float scaleY = static_cast<float>(targetHeight) / static_cast<float>(logicalFbHeight);
  return {
      .left = logicalViewport.left * scaleX,
      .top = logicalViewport.top * scaleY,
      .width = logicalViewport.width * scaleX,
      .height = logicalViewport.height * scaleY,
      .znear = logicalViewport.znear,
      .zfar = logicalViewport.zfar,
  };
}

gfx::ClipRect map_logical_scissor(const gfx::ClipRect& logicalScissor) noexcept {
  if (g_gxState.viewportPolicy == AURORA_VIEWPORT_NATIVE) {
    return logicalScissor;
  }

  const auto [logicalFbWidth, logicalFbHeight] = logical_fb_size();
  const auto [targetWidth, targetHeight] = gfx::get_render_target_size();
  if (logicalFbWidth == 0 || logicalFbHeight == 0 || targetWidth == 0 || targetHeight == 0) {
    return logicalScissor;
  }

  const float scaleX = static_cast<float>(targetWidth) / static_cast<float>(logicalFbWidth);
  const float scaleY = static_cast<float>(targetHeight) / static_cast<float>(logicalFbHeight);

  const float left = static_cast<float>(logicalScissor.x) * scaleX;
  const float top = static_cast<float>(logicalScissor.y) * scaleY;
  const float right = static_cast<float>(logicalScissor.x + logicalScissor.width) * scaleX;
  const float bottom = static_cast<float>(logicalScissor.y + logicalScissor.height) * scaleY;

  const auto mappedLeft = std::clamp(static_cast<int32_t>(std::floor(left)), 0, static_cast<int32_t>(targetWidth));
  const auto mappedTop = std::clamp(static_cast<int32_t>(std::floor(top)), 0, static_cast<int32_t>(targetHeight));
  const auto mappedRight =
      std::clamp(static_cast<int32_t>(std::ceil(right)), mappedLeft, static_cast<int32_t>(targetWidth));
  const auto mappedBottom =
      std::clamp(static_cast<int32_t>(std::ceil(bottom)), mappedTop, static_cast<int32_t>(targetHeight));

  return {
      .x = mappedLeft,
      .y = mappedTop,
      .width = mappedRight - mappedLeft,
      .height = mappedBottom - mappedTop,
  };
}

void set_logical_viewport(const gfx::Viewport& viewport) noexcept {
  g_gxState.logicalViewport = viewport;
  set_render_viewport(map_logical_viewport(viewport));
}

void set_render_viewport(const gfx::Viewport& viewport) noexcept {
  g_gxState.renderViewport = viewport;
  gfx::set_viewport(viewport);
}

void set_logical_scissor(const gfx::ClipRect& scissor) noexcept {
  g_gxState.logicalScissor = scissor;
  set_render_scissor(map_logical_scissor(g_gxState.logicalScissor));
}

void set_render_scissor(const gfx::ClipRect& scissor) noexcept {
  g_gxState.renderScissor = scissor;
  gfx::set_scissor(scissor);
}

const gfx::TextureBind& get_texture(GXTexMapID id) noexcept { return g_gxState.textures[static_cast<size_t>(id)]; }

void evict_texture_object(u32 texObjId) noexcept {
  const void* evictedData = nullptr;
  if (const auto it = s_textureObjectCaches.find(texObjId); it != s_textureObjectCaches.end()) {
    evictedData = it->second.data;
    clear_texture_dependency(texObjId, it->second.tlutObjId);
    s_textureObjectCaches.erase(it);
  }
  invalidate_fast_tex_id(texObjId);
  if (evictedData != nullptr) {
    invalidate_data_caches(evictedData);
  }
  // If there is a loaded slot with this ID, mark it as no_cache to avoid inserting it when it's resolved.
  // This also handles the case where the texture was created, loaded, and immediately destroyed before we resolved it.
  for (auto& obj : g_gxState.loadedTextures) {
    if (obj.texObjId == texObjId) {
      obj.set_no_cache(true);
      invalidate_data_caches(obj.data);
    }
  }
}

void evict_tlut_object(u32 tlutObjId) noexcept {
  if (const auto it = s_tlutObjectCaches.find(tlutObjId); it != s_tlutObjectCaches.end()) {
    for (const u32 texObjId : it->second.staticTextureUsers) {
      s_textureObjectCaches.erase(texObjId);
      invalidate_fast_tex_id(texObjId);
    }
    s_tlutObjectCaches.erase(it);
  }
  const void* evictedTlutData = nullptr;
  // If there is a loaded slot with this ID, mark it as no_cache to avoid inserting it when it's resolved.
  // This also handles the case where the texture was created, loaded, and immediately destroyed before we resolved it.
  for (auto& obj : g_gxState.loadedTluts) {
    if (obj.tlutObjId == tlutObjId) {
      obj.set_no_cache(true);
      if (obj.data != nullptr) {
        evictedTlutData = obj.data;
      }
    }
  }
  invalidate_tlut_dependents(tlutObjId, evictedTlutData);
}

// GXAliasCopyTex destinations whose source was not copied yet (dest -> src)
static absl::flat_hash_map<const void*, const void*> s_copyAliases;

void clear_copy_texture_cache() noexcept {
  g_gxState.copyTextures.clear();
  s_copyAliases.clear();
  g_gxState.copyTextureCache.clear();
  for (auto& [_, cache] : s_tlutObjectCaches) {
    cache.dynamicPaletteTextures.clear();
  }
  for (auto& meta : s_lastResolvedMeta) {
    if (meta.isCopy) {
      meta = {};
    }
  }
}

void clear_static_texture_cache() noexcept { s_staticTextureCacheClearPending.store(true, std::memory_order_release); }

void invalidate_copy_caches(const void* dest) noexcept { invalidate_data_caches(dest); }

void alias_copy_texture(const void* src, const void* dest) noexcept {
  invalidate_data_caches(dest);
  const auto it = g_gxState.copyTextures.find(src);
  if (it == g_gxState.copyTextures.end() || !it->second.handle) {
    s_copyAliases[dest] = src;
    return;
  }
  g_gxState.copyTextures[dest] = it->second;
  s_copyAliases.erase(dest);
}

void forget_memory(const void* lo_, const void* hi_) noexcept {
  const auto lo = reinterpret_cast<uintptr_t>(lo_);
  const auto hi = reinterpret_cast<uintptr_t>(hi_);
  if (hi <= lo) {
    return;
  }
  const auto in = [lo, hi](const void* p) {
    return reinterpret_cast<uintptr_t>(p) >= lo && reinterpret_cast<uintptr_t>(p) < hi;
  };
  absl::erase_if(g_gxState.copyTextures, [&](const auto& kv) { return in(kv.first); });
  absl::erase_if(s_copyAliases, [&](const auto& kv) { return in(kv.first) || in(kv.second); });
  for (auto it = g_gxState.copyTextureCache.begin(); it != g_gxState.copyTextureCache.end();) {
    if (in(it->first.dest)) {
      g_gxState.copyTextureCache.erase(it++);
    } else {
      ++it;
    }
  }
  invalidate_caches_if(in);
}

void move_memory(const void* from_, const void* to_, size_t size) noexcept {
  const auto from = reinterpret_cast<uintptr_t>(from_);
  const auto to = reinterpret_cast<uintptr_t>(to_);
  if (from == to || size == 0) {
    return;
  }
  const auto delta = static_cast<intptr_t>(to - from);
  const auto in = [from, size](const void* p) {
    const auto u = reinterpret_cast<uintptr_t>(p);
    return u >= from && u < from + size;
  };
  const auto shift = [delta](const void* p) {
    return reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(p) + delta);
  };
  std::vector<std::pair<const void*, GXState::CopyTextureRef>> copies;
  for (auto& [dest, ref] : g_gxState.copyTextures) {
    if (in(dest)) {
      copies.emplace_back(shift(dest), std::move(ref));
    }
  }
  absl::erase_if(g_gxState.copyTextures, [&](const auto& kv) { return in(kv.first); });

  std::vector<std::pair<const void*, const void*>> aliases;
  for (const auto& [dest, src] : s_copyAliases) {
    if (in(dest) || in(src)) {
      const auto d = in(dest) ? shift(dest) : dest;
      const auto s = in(src) ? shift(src) : src;
      aliases.emplace_back(d, s);
    }
  }
  absl::erase_if(s_copyAliases, [&](const auto& kv) { return in(kv.first) || in(kv.second); });

  for (auto& [dest, ref] : copies) {
    g_gxState.copyTextures[dest] = std::move(ref);
  }
  for (const auto& [dest, src] : aliases) {
    s_copyAliases[dest] = src;
  }
  invalidate_caches_if(in);
}

void expire_texture_object_caches() noexcept {
  // Ages out cache entries whose game-side GXTexObj is gone. GXInitTexObj mints a fresh
  // texObjId every call and almost nothing calls GXDestroyTexObj (the game frees textures
  // by wholesale JKR heap teardown on area/actor unload), so without this sweep every
  // orphaned entry pins its GL texture forever -- the ~1 texture/s live-count climb that
  // OOMs the 1 GB G31. Eviction is safe: a later use is a cache miss that re-decodes from
  // obj.data, which must be alive for the texobj to be drawn again.
  constexpr u32 SweepPeriod = 30;
  constexpr u32 RetainFrames = 300; // ~10 s at 30 Hz

  const u32 frame = gfx::current_frame();
  if (frame == UINT32_MAX || frame % SweepPeriod != 0) {
    return;
  }
  ZoneScoped;

  // Anything currently referenced by a texmap/tlut slot must survive regardless of age: a
  // texture can stay bound (and drawn) indefinitely without re-resolving, because
  // resolve_sampled_textures short-circuits on an unchanged bind.
  absl::flat_hash_set<u32> boundTexObjs;
  absl::flat_hash_set<u32> boundTluts;
  absl::flat_hash_set<const void*> boundDataAddrs;
  for (const auto& obj : g_gxState.loadedTextures) {
    if (obj.texObjId != 0) {
      boundTexObjs.insert(obj.texObjId);
    }
    if (obj.data != nullptr) {
      boundDataAddrs.insert(obj.data);
    }
  }
  for (const auto& bind : g_gxState.textures) {
    if (bind.texObj.texObjId != 0) {
      boundTexObjs.insert(bind.texObj.texObjId);
    }
    if (bind.texObj.data != nullptr) {
      boundDataAddrs.insert(bind.texObj.data);
    }
  }
  for (const auto& tlut : g_gxState.loadedTluts) {
    if (tlut.tlutObjId != 0) {
      boundTluts.insert(tlut.tlutObjId);
    }
  }

  for (auto it = s_textureObjectCaches.begin(); it != s_textureObjectCaches.end();) {
    if (!boundTexObjs.contains(it->first) && frame - it->second.lastUsedFrame > RetainFrames) {
      invalidate_fast_tex_id(it->first);
      clear_texture_dependency(it->first, it->second.tlutObjId);
      s_textureObjectCaches.erase(it++);
    } else {
      ++it;
    }
  }
  for (auto it = s_pointerTextureCache.begin(); it != s_pointerTextureCache.end();) {
    if (!boundDataAddrs.contains(it->first.data) && frame - it->second.lastUsedFrame > RetainFrames) {
      const size_t pslot = fast_ptr_slot(it->first.data, it->first.image0);
      auto& pfast = s_fastResolvedPtrTextures[pslot];
      if (pfast.data == it->first.data && pfast.tlutData == it->first.tlutData &&
          pfast.image0 == it->first.image0 && pfast.format == it->first.format &&
          pfast.mipCount == it->first.mipCount && pfast.texDataVersion == it->first.texDataVersion &&
          pfast.tlutObjId == it->first.tlutObjId && pfast.tlutDataVersion == it->first.tlutDataVersion) {
        pfast = {};
      }
      s_pointerTextureCache.erase(it++);
    } else {
      ++it;
    }
  }
  for (auto it = s_tlutObjectCaches.begin(); it != s_tlutObjectCaches.end();) {
    auto& cache = it->second;
    for (auto dp = cache.dynamicPaletteTextures.begin(); dp != cache.dynamicPaletteTextures.end();) {
      if (frame - dp->second.lastUsedFrame > RetainFrames) {
        cache.dynamicPaletteTextures.erase(dp++);
      } else {
        ++dp;
      }
    }
    if (cache.tlutTexture.handle && !boundTluts.contains(it->first) &&
        frame - cache.tlutTexture.lastUsedFrame > RetainFrames) {
      cache.tlutTexture = {};
    }
    if (!cache.tlutTexture.handle && cache.dynamicPaletteTextures.empty() && cache.staticTextureUsers.empty()) {
      s_tlutObjectCaches.erase(it++);
    } else {
      ++it;
    }
  }

  // Evict stale EFB copy textures that haven't been written or sampled recently.
  std::vector<const void*> staleCopyDests;
  for (const auto& [dest, ref] : g_gxState.copyTextures) {
    if (!boundDataAddrs.contains(dest) && frame - ref.lastUsedFrame > 180) {
      staleCopyDests.push_back(dest);
    }
  }
  for (const void* dest : staleCopyDests) {
    evict_copy_texture(dest);
  }
  for (auto it = g_gxState.copyTextureCache.begin(); it != g_gxState.copyTextureCache.end();) {
    if (!boundDataAddrs.contains(it->first.dest) && frame - it->second.lastUsedFrame > 180) {
      g_gxState.copyTextureCache.erase(it++);
    } else {
      ++it;
    }
  }

  // If cached textures exceed budget on 1 GB RAM devices, drop stale non-bound texture objects first.
  if (s_staticContentCache.size() > 4096) {
    for (auto it = s_textureObjectCaches.begin(); it != s_textureObjectCaches.end();) {
      if (!boundTexObjs.contains(it->first) && frame - it->second.lastUsedFrame > 300) {
        invalidate_fast_tex_id(it->first);
        clear_texture_dependency(it->first, it->second.tlutObjId);
        s_textureObjectCaches.erase(it++);
      } else {
        ++it;
      }
    }
  }

  // Prune unreferenced static content cache entries only when over safe budget (2048 entries).
  if (s_staticContentCache.size() > 2048) {
    for (auto it = s_staticContentCache.begin(); it != s_staticContentCache.end();) {
      if (!it->second || it->second.use_count() <= 1) {
        s_staticContentCache.erase(it++);
      } else {
        ++it;
      }
    }
  }
}

void evict_copy_texture(const void* dest) noexcept {
  s_copyAliases.erase(dest);
  invalidate_data_caches(dest);
  absl::flat_hash_set<const void*> sourceIdentities;
  if (const auto it = g_gxState.copyTextures.find(dest); it != g_gxState.copyTextures.end()) {
    if (it->second.handle) {
      sourceIdentities.insert(it->second.handle.get());
    }
    g_gxState.copyTextures.erase(it);
  }

  for (auto it = g_gxState.copyTextureCache.begin(); it != g_gxState.copyTextureCache.end();) {
    if (it->first.dest == dest) {
      if (it->second.handle) {
        sourceIdentities.insert(it->second.handle.get());
      }
      g_gxState.copyTextureCache.erase(it++);
    } else {
      ++it;
    }
  }

  if (sourceIdentities.empty()) {
    return;
  }

  for (auto& [_, cache] : s_tlutObjectCaches) {
    for (auto it = cache.dynamicPaletteTextures.begin(); it != cache.dynamicPaletteTextures.end();) {
      if (sourceIdentities.contains(it->first.sourceIdentity)) {
        cache.dynamicPaletteTextures.erase(it++);
      } else {
        ++it;
      }
    }
  }
}

static uint32_t s_dbg_calls = 0;
static uint32_t s_dbg_sameTex = 0;
static uint32_t s_dbg_fastHit = 0;
static uint32_t s_dbg_fastMiss = 0;
static uint32_t s_dbg_copy = 0;
static uint32_t s_dbg_zeroId = 0;
static uint32_t s_dbg_nullData = 0;
static uint32_t s_m_id = 0;
static uint32_t s_m_ver = 0;
static uint32_t s_m_mode0 = 0;
static uint32_t s_m_mode1 = 0;
static uint32_t s_m_tlut = 0;
static uint32_t s_m_ref = 0;

void resolve_sampled_textures(const ShaderInfo& info) noexcept {
  ZoneScoped;

  ++s_dbg_calls;

  if (__builtin_expect(s_staticTextureCacheClearPending.load(std::memory_order_relaxed), 0)) {
    if (s_staticTextureCacheClearPending.exchange(false, std::memory_order_acq_rel)) {
      do_clear_static_texture_cache();
    }
  }

  for (u32 i = 0; i < MaxTextures; ++i) {
    if (!info.sampledTextures.test(i)) {
      continue;
    }

    GXTexObj_ obj = g_gxState.loadedTextures[i];
    const u32 rawFormat = static_cast<u32>(obj.format());
    const u32 mipCount = obj.mip_count();
    u32 tlutObjId = 0;
    u32 tlutDataVersion = 0;
    const void* tlutData = nullptr;
    bool tlutNoCache = false;
    if (is_palette_format(obj.format())) {
      const auto tlutIdx = static_cast<size_t>(obj.tlut);
      if (tlutIdx < g_gxState.loadedTluts.size()) {
        const auto& tlut = g_gxState.loadedTluts[tlutIdx];
        tlutObjId = tlut.tlutObjId;
        tlutDataVersion = tlut.tlutDataVersion;
        tlutData = tlut.data;
        tlutNoCache = tlut.no_cache();
      }
    }

    auto& textureBind = g_gxState.textures[i];
    auto& lastMeta = s_lastResolvedMeta[i];
    const auto updateLastMeta = [&](bool isCopy) noexcept {
      if (textureBind.ref) {
        lastMeta = {
            .valid = true,
            .isCopy = isCopy,
            .texObjId = obj.texObjId,
            .texDataVersion = obj.texDataVersion,
            .tlutObjId = tlutObjId,
            .tlutDataVersion = tlutDataVersion,
            .tlutData = tlutData,
            .data = obj.data,
            .image0 = obj.image0,
            .format = rawFormat,
            .mipCount = mipCount,
            .mode0 = obj.mode0,
            .mode1 = obj.mode1,
        };
      } else {
        lastMeta = {};
      }
    };
    const auto makeFastTex = [&]() noexcept -> FastResolvedTexture {
      return {
          .texObjId = obj.texObjId,
          .texDataVersion = obj.texDataVersion,
          .data = obj.data,
          .image0 = obj.image0,
          .format = rawFormat,
          .mipCount = mipCount,
          .mode0 = obj.mode0,
          .mode1 = obj.mode1,
          .tlutObjId = tlutObjId,
          .tlutDataVersion = tlutDataVersion,
          .tlutData = tlutData,
          .bind = textureBind,
      };
    };

    const bool sameTex =
        lastMeta.valid && textureBind.ref &&
        lastMeta.data == obj.data && lastMeta.image0 == obj.image0 &&
        lastMeta.mode0 == obj.mode0 && lastMeta.mode1 == obj.mode1 &&
        lastMeta.format == rawFormat && lastMeta.mipCount == mipCount &&
        lastMeta.tlutObjId == tlutObjId && lastMeta.tlutDataVersion == tlutDataVersion &&
        lastMeta.tlutData == tlutData &&
        (obj.texObjId != 0
             ? (lastMeta.texObjId == obj.texObjId && lastMeta.texDataVersion == obj.texDataVersion)
             : (lastMeta.texObjId == 0 && obj.data != nullptr));
    if (sameTex) {
      // Texture bind unchanged
      ++s_dbg_sameTex;
      continue;
    }

    // Step 2: Instant fast cache lookup (3 nanoseconds, 0 hash map lookups)
    if (obj.texObjId != 0) {
      const size_t slot = fast_tex_slot(obj.texObjId);
      auto& fast = s_fastResolvedTextures[slot];
      if (fast.texObjId == obj.texObjId && fast.texDataVersion == obj.texDataVersion &&
          fast.data == obj.data && fast.image0 == obj.image0 &&
          fast.format == rawFormat && fast.mipCount == mipCount &&
          fast.mode0 == obj.mode0 && fast.mode1 == obj.mode1 &&
          fast.tlutObjId == tlutObjId && fast.tlutDataVersion == tlutDataVersion &&
          fast.tlutData == tlutData && fast.bind.ref) {
        textureBind = fast.bind;
        updateLastMeta(false);
        ++s_dbg_fastHit;
        continue;
      }
    }

    if (obj.texObjId == 0) ++s_dbg_zeroId;
    if (obj.data == nullptr) ++s_dbg_nullData;

    GXState::CopyTextureRef* copyRef = nullptr;
    if (!g_gxState.copyTextures.empty() && obj.data != nullptr) {
      auto copyIt = g_gxState.copyTextures.find(obj.data);
      if (copyIt == g_gxState.copyTextures.end() && !s_copyAliases.empty()) {
        if (const auto alias = s_copyAliases.find(obj.data); alias != s_copyAliases.end()) {
          if (const auto src = g_gxState.copyTextures.find(alias->second);
              src != g_gxState.copyTextures.end() && src->second.handle) {
            copyIt = g_gxState.copyTextures.insert_or_assign(obj.data, src->second).first;
            s_copyAliases.erase(alias);
          }
        }
      }
      if (copyIt != g_gxState.copyTextures.end()) {
        copyRef = &copyIt->second;
        copyRef->lastUsedFrame = gfx::current_frame();
      }
    }

    if (copyRef != nullptr) {
      ++s_dbg_copy;
      if (!is_palette_format(obj.format())) {
        if (copyRef->bind.ref && copyRef->mode0 == obj.mode0 && copyRef->mode1 == obj.mode1) {
          textureBind = copyRef->bind;
          updateLastMeta(true);
          ++s_dbg_fastHit;
          continue;
        }
        obj.mFormat = resolved_format_for_handle(copyRef->handle);
        textureBind = gfx::TextureBind{obj, copyRef->handle};
        if (textureBind.ref) {
          textureBind.sampler = gfx::sampler_ref(textureBind.get_descriptor());
        }
        copyRef->bind = textureBind;
        copyRef->mode0 = obj.mode0;
        copyRef->mode1 = obj.mode1;
        updateLastMeta(true);
        ++s_dbg_fastHit;
        continue;
      }
    } else {
      const bool allowCache = !obj.no_cache() && !tlutNoCache;
      if (obj.texObjId != 0) {
        const size_t slot = fast_tex_slot(obj.texObjId);
        auto& fast = s_fastResolvedTextures[slot];
        if (fast.texObjId == obj.texObjId) {
          if (fast.texDataVersion != obj.texDataVersion) ++s_m_ver;
          else if (fast.mode0 != obj.mode0) ++s_m_mode0;
          else if (fast.mode1 != obj.mode1) ++s_m_mode1;
          else if (fast.tlutObjId != tlutObjId || fast.tlutDataVersion != tlutDataVersion) ++s_m_tlut;
          else if (!fast.bind.ref) ++s_m_ref;
        } else {
          ++s_m_id;
        }

        // Secondary check: look up in s_textureObjectCaches directly before heavy resolve
        if (const auto it = s_textureObjectCaches.find(obj.texObjId); it != s_textureObjectCaches.end()) {
          auto& entry = it->second;
          if (entry.matches(obj, tlutObjId, tlutDataVersion)) {
            entry.lastUsedFrame = gfx::current_frame();
            if (entry.bind.ref && entry.mode0 == obj.mode0 && entry.mode1 == obj.mode1) {
              textureBind = entry.bind;
              if (allowCache) {
                fast = makeFastTex();
              }
              updateLastMeta(false);
              ++s_dbg_fastHit;
              continue;
            }
            // Handle is cached, just rebuild TextureBind
            obj.mFormat = resolved_format_for_handle(entry.handle);
            textureBind = gfx::TextureBind{obj, entry.handle};
            if (textureBind.ref) {
              textureBind.sampler = gfx::sampler_ref(textureBind.get_descriptor());
            }
            entry.bind = textureBind;
            entry.mode0 = obj.mode0;
            entry.mode1 = obj.mode1;
            if (allowCache) {
              fast = makeFastTex();
            }
            updateLastMeta(false);
            ++s_dbg_fastHit;
            continue;
          }
        }
      }

      if (obj.data != nullptr) {
        const size_t pslot = fast_ptr_slot(obj.data, obj.image0);
        auto& pfast = s_fastResolvedPtrTextures[pslot];
        if (pfast.data == obj.data && pfast.tlutData == tlutData && pfast.image0 == obj.image0 &&
            pfast.format == rawFormat && pfast.mipCount == mipCount &&
            pfast.texDataVersion == obj.texDataVersion && pfast.tlutObjId == tlutObjId &&
            pfast.tlutDataVersion == tlutDataVersion && pfast.mode0 == obj.mode0 &&
            pfast.mode1 == obj.mode1 && pfast.bind.ref) {
          textureBind = pfast.bind;
          if (obj.texObjId != 0 && allowCache) {
            const size_t slot = fast_tex_slot(obj.texObjId);
            s_fastResolvedTextures[slot] = makeFastTex();
          }
          updateLastMeta(false);
          ++s_dbg_fastHit;
          continue;
        }

        // Pointer texture cache secondary lookup (avoids XXH3 content hashing!)
        const PointerTextureKey pkey{obj.data, tlutData, obj.image0, rawFormat, mipCount,
                                     obj.texDataVersion, tlutObjId, tlutDataVersion};
        if (const auto pit = s_pointerTextureCache.find(pkey); pit != s_pointerTextureCache.end() && pit->second.handle) {
          pit->second.lastUsedFrame = gfx::current_frame();
          obj.mFormat = resolved_format_for_handle(pit->second.handle);
          textureBind = gfx::TextureBind{obj, pit->second.handle};
          if (textureBind.ref) {
            textureBind.sampler = gfx::sampler_ref(textureBind.get_descriptor());
          }
          if (allowCache) {
            pfast = {
                .data = obj.data,
                .tlutData = tlutData,
                .image0 = obj.image0,
                .format = rawFormat,
                .mipCount = mipCount,
                .texDataVersion = obj.texDataVersion,
                .tlutObjId = tlutObjId,
                .tlutDataVersion = tlutDataVersion,
                .mode0 = obj.mode0,
                .mode1 = obj.mode1,
                .bind = textureBind,
            };
            if (obj.texObjId != 0) {
              store_cached_texture(obj, pit->second.handle, tlutObjId, tlutDataVersion);
              s_textureObjectCaches[obj.texObjId].bind = textureBind;
              s_textureObjectCaches[obj.texObjId].mode0 = obj.mode0;
              s_textureObjectCaches[obj.texObjId].mode1 = obj.mode1;
              const size_t slot = fast_tex_slot(obj.texObjId);
              s_fastResolvedTextures[slot] = makeFastTex();
            }
          }
          updateLastMeta(false);
          ++s_dbg_fastHit;
          continue;
        }
      }
      ++s_dbg_fastMiss;
    }

    gfx::TextureHandle handle;
    if (is_palette_format(obj.format())) {
      const auto tlutIdx = static_cast<size_t>(obj.tlut);
      if (tlutIdx < g_gxState.loadedTluts.size()) {
        const auto& tlut = g_gxState.loadedTluts[tlutIdx];
        if (tlut.data != nullptr) {
          if (copyRef != nullptr) {
            handle = resolve_dynamic_palette_texture(obj, *copyRef, tlut);
          } else if (obj.has_data()) {
            handle = resolve_static_palette_texture(obj, tlut);
          }
        }
      }
    } else if (copyRef != nullptr) {
      handle = copyRef->handle;
    } else if (obj.has_data()) {
      handle = resolve_static_texture(obj);
    }

    obj.mFormat = resolved_format_for_handle(handle);
    textureBind = gfx::TextureBind{obj, std::move(handle)};
    if (textureBind.ref) {
      textureBind.sampler = gfx::sampler_ref(textureBind.get_descriptor());
    }

    if (copyRef != nullptr && textureBind.ref) {
      copyRef->bind = textureBind;
      copyRef->mode0 = obj.mode0;
      copyRef->mode1 = obj.mode1;
    } else if (copyRef == nullptr && textureBind.ref && !obj.no_cache() && !tlutNoCache) {
      if (obj.texObjId != 0) {
        if (const auto it = s_textureObjectCaches.find(obj.texObjId); it != s_textureObjectCaches.end()) {
          it->second.bind = textureBind;
          it->second.mode0 = obj.mode0;
          it->second.mode1 = obj.mode1;
        }
        const size_t slot = fast_tex_slot(obj.texObjId);
        s_fastResolvedTextures[slot] = makeFastTex();
      } else if (obj.data != nullptr) {
        const size_t pslot = fast_ptr_slot(obj.data, obj.image0);
        s_fastResolvedPtrTextures[pslot] = FastResolvedPointerTexture{
            .data = obj.data,
            .tlutData = tlutData,
            .image0 = obj.image0,
            .format = rawFormat,
            .mipCount = mipCount,
            .texDataVersion = obj.texDataVersion,
            .tlutObjId = tlutObjId,
            .tlutDataVersion = tlutDataVersion,
            .mode0 = obj.mode0,
            .mode1 = obj.mode1,
            .bind = textureBind,
        };
      }
    }
    updateLastMeta(copyRef != nullptr);
  }
}

static inline gfx::BlendFactor to_blend_factor(GXBlendFactor fac, bool isDst) {
  switch (fac) {
    DEFAULT_FATAL("invalid blend factor {}", underlying(fac));
  case GX_BL_ZERO:
    return gfx::BlendFactor::Zero;
  case GX_BL_ONE:
    return gfx::BlendFactor::One;
  case GX_BL_SRCCLR: // + GX_BL_DSTCLR
    if (isDst) {
      return gfx::BlendFactor::Src;
    } else {
      return gfx::BlendFactor::Dst;
    }
  case GX_BL_INVSRCCLR: // + GX_BL_INVDSTCLR
    if (isDst) {
      return gfx::BlendFactor::OneMinusSrc;
    } else {
      return gfx::BlendFactor::OneMinusDst;
    }
  case GX_BL_SRCALPHA:
    return gfx::BlendFactor::SrcAlpha;
  case GX_BL_INVSRCALPHA:
    return gfx::BlendFactor::OneMinusSrcAlpha;
  case GX_BL_DSTALPHA:
    return gfx::BlendFactor::DstAlpha;
  case GX_BL_INVDSTALPHA:
    return gfx::BlendFactor::OneMinusDstAlpha;
  }
}

static inline gfx::CompareFunction to_compare_function(GXCompare func) {
  switch (func) {
    DEFAULT_FATAL("invalid depth fn {}", underlying(func));
  case GX_NEVER:
    return gfx::CompareFunction::Never;
  case GX_LESS:
    return UseReversedZ ? gfx::CompareFunction::Greater : gfx::CompareFunction::Less;
  case GX_EQUAL:
    return gfx::CompareFunction::Equal;
  case GX_LEQUAL:
    return UseReversedZ ? gfx::CompareFunction::GreaterEqual : gfx::CompareFunction::LessEqual;
  case GX_GREATER:
    return UseReversedZ ? gfx::CompareFunction::Less : gfx::CompareFunction::Greater;
  case GX_NEQUAL:
    return gfx::CompareFunction::NotEqual;
  case GX_GEQUAL:
    return UseReversedZ ? gfx::CompareFunction::LessEqual : gfx::CompareFunction::GreaterEqual;
  case GX_ALWAYS:
    return gfx::CompareFunction::Always;
  }
}

// The color+alpha blend equation, resolved from the GX blend/logic-op state. GL
// has no BlendState object; these six fields feed glBlendEquationSeparate /
// glBlendFuncSeparate when the pipeline's state is applied (Phase 3).
struct GLBlendComponents {
  gfx::BlendOperation colorOp = gfx::BlendOperation::Add;
  gfx::BlendFactor colorSrc = gfx::BlendFactor::One;
  gfx::BlendFactor colorDst = gfx::BlendFactor::Zero;
  gfx::BlendOperation alphaOp = gfx::BlendOperation::Add;
  gfx::BlendFactor alphaSrc = gfx::BlendFactor::One;
  gfx::BlendFactor alphaDst = gfx::BlendFactor::Zero;
};

static inline GLBlendComponents to_blend_state(GXBlendMode mode, GXBlendFactor srcFac, GXBlendFactor dstFac,
                                               GXLogicOp op, u32 dstAlpha) {
  GLBlendComponents blend;
  switch (mode) {
    DEFAULT_FATAL("unsupported blend mode {}", underlying(mode));
  case GX_BM_NONE:
    blend.colorOp = gfx::BlendOperation::Add;
    blend.colorSrc = gfx::BlendFactor::One;
    blend.colorDst = gfx::BlendFactor::Zero;
    break;
  case GX_BM_BLEND:
    blend.colorOp = gfx::BlendOperation::Add;
    blend.colorSrc = to_blend_factor(srcFac, false);
    blend.colorDst = to_blend_factor(dstFac, true);
    break;
  case GX_BM_SUBTRACT:
    blend.colorOp = gfx::BlendOperation::ReverseSubtract;
    blend.colorSrc = gfx::BlendFactor::One;
    blend.colorDst = gfx::BlendFactor::One;
    break;
  case GX_BM_LOGIC:
    switch (op) {
      DEFAULT_FATAL("unsupported logic op {}", underlying(op));
    case GX_LO_CLEAR:
      blend.colorOp = gfx::BlendOperation::Add;
      blend.colorSrc = gfx::BlendFactor::Zero;
      blend.colorDst = gfx::BlendFactor::Zero;
      break;
    case GX_LO_COPY:
      blend.colorOp = gfx::BlendOperation::Add;
      blend.colorSrc = gfx::BlendFactor::One;
      blend.colorDst = gfx::BlendFactor::Zero;
      break;
    case GX_LO_NOOP:
      blend.colorOp = gfx::BlendOperation::Add;
      blend.colorSrc = gfx::BlendFactor::Zero;
      blend.colorDst = gfx::BlendFactor::One;
      break;
    }
    break;
  }
  if (dstAlpha != UINT32_MAX) {
    blend.alphaOp = gfx::BlendOperation::Add;
    blend.alphaSrc = gfx::BlendFactor::Constant;
    blend.alphaDst = gfx::BlendFactor::Zero;
  } else {
    blend.alphaOp = blend.colorOp;
    blend.alphaSrc = blend.colorSrc;
    blend.alphaDst = blend.colorDst;
  }
  return blend;
}

static inline gfx::ColorWriteMask to_write_mask(bool colorUpdate, bool alphaUpdate) {
  gfx::ColorWriteMask writeMask = gfx::ColorWriteMask::None;
  if (colorUpdate) {
    writeMask = writeMask | gfx::ColorWriteMask::Red | gfx::ColorWriteMask::Green | gfx::ColorWriteMask::Blue;
  }
  if (alphaUpdate) {
    writeMask = writeMask | gfx::ColorWriteMask::Alpha;
  }
  return writeMask;
}

static inline gfx::CullMode to_cull_mode(GXCullMode gx_cullMode) {
  switch (gx_cullMode) {
  case GX_CULL_FRONT:
    return gfx::CullMode::Front;
  case GX_CULL_BACK:
  case GX_CULL_ALL:
    return gfx::CullMode::Back;
  case GX_CULL_NONE:
    return gfx::CullMode::None;
  default:
    return gfx::CullMode::None;
  }
}

// Bake the GX pipeline state into the backend-neutral BakedState. `program` is the
// already-linked GLSL program (0 during Phase 1, where draws don't execute yet).
// The fixed-function state here is applied by lib/gl/state.cpp at draw time; note
// FrontFace stays in WebGPU terms (CW) and the S1a winding flip happens once there.
gl::Pipeline build_pipeline(const PipelineConfig& config, uint32_t program, const char* label) noexcept {
  ZoneScoped;
  const float depthBias = (UseReversedZ ? -1.0f : 1.0f) * std::bit_cast<float>(config.polygonOffsetBits);
  const float depthBiasSlopeScale = (UseReversedZ ? -1.0f : 1.0f) * std::bit_cast<float>(config.polygonOffsetScaleBits);

  const auto blend = to_blend_state(config.blendMode, config.blendFacSrc, config.blendFacDst, config.blendOp,
                                    config.dstAlpha);

  gl::BakedState state{};
  // Blend is always attached (matches Dawn always supplying a blend state); an
  // One/Zero/Add equation is an identity passthrough, so this is a no-op there.
  state.blendEnabled = true;
  state.colorOp = blend.colorOp;
  state.colorSrc = blend.colorSrc;
  state.colorDst = blend.colorDst;
  state.alphaOp = blend.alphaOp;
  state.alphaSrc = blend.alphaSrc;
  state.alphaDst = blend.alphaDst;
  state.writeMask = config.cullMode == GX_CULL_ALL ? gfx::ColorWriteMask::None
                                                   : to_write_mask(config.colorUpdate, config.alphaUpdate);

  state.depthTest = config.depthCompare;
  state.depthWrite = config.depthCompare && config.depthUpdate;
  state.depthCompare = config.depthCompare ? to_compare_function(config.depthFunc) : gfx::CompareFunction::Always;

  state.cull = to_cull_mode(config.cullMode);
  state.frontFace = gfx::FrontFace::CW;
  state.polygonOffset = depthBias != 0.0f || depthBiasSlopeScale != 0.0f;
  state.depthBiasSlope = depthBiasSlopeScale;
  state.depthBiasUnits = depthBias;

  // TriangleStrip batches separate strips with the max Uint16 index (0xffff) as an
  // implicit primitive restart; a triangle *list* must NOT enable restart (S6).
  state.topology =
      config.triangleStripTopology != 0 ? gfx::PrimitiveTopology::TriangleStrip : gfx::PrimitiveTopology::TriangleList;
  state.primitiveRestart = config.triangleStripTopology != 0;
  state.alphaToCoverage = (config.shaderConfig.edgeAA == 1 && config.msaaSamples > 1);

  // Native vertex layout as a present-attr bitmask over canonical GX attr order
  // (GX_VA_PNMTXIDX..GX_VA_TEX7). The draw path (pass.cpp) decodes it into the VAO;
  // bit positions == GXAttr values == the GLSL emitter's packed `layout(location)`.
  uint32_t vertexLayout = 0;
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    if (config.shaderConfig.attrs[i].attrType != GX_NONE) {
      vertexLayout |= (1u << i);
    }
  }

  return gl::Pipeline{
      .program = program,
      .state = state,
      .vertexLayout = vertexLayout,
  };
}

void populate_pipeline_config(PipelineConfig& config, GXPrimitive primitive, GXVtxFmt fmt, bool fillAttrs) noexcept {
  ZoneScoped;

  config.shaderConfig.fogType = g_gxState.fog.type;
  if (fillAttrs) {
    const auto& vtxFmt = g_gxState.vtxFmts[fmt];
    u8 vtxOffset = 0;
    for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
      const auto attr = static_cast<GXAttr>(i);
      const auto type = g_gxState.vtxDesc[i];
      auto& mapping = config.shaderConfig.attrs[i];
      if (type == GX_NONE) {
        mapping = {};
        continue;
      }
      const auto& attrFmt = vtxFmt.attrs[i];
      const auto cnt = comp_cnt_count(attr, attrFmt.cnt);
      const bool nbt3 = attr == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3;
      mapping = AttrConfig{
          .attrType = static_cast<u8>(type),
          .cnt = cnt,
          .compType = static_cast<u8>(attrFmt.type),
          .offset = vtxOffset,
          .stride = 0,
          .frac = attrFmt.frac,
          .le = false,
          .nbt3 = nbt3,
      };
      switch (type) {
      case GX_DIRECT: {
        vtxOffset += comp_type_size(attr, attrFmt.type) * cnt;
        break;
      }
      case GX_INDEX8:
        mapping.stride = g_gxState.arrays[i].stride;
        mapping.le = g_gxState.arrays[i].le;
        vtxOffset += nbt3 ? 3 : 1;
        break;
      case GX_INDEX16:
        mapping.stride = g_gxState.arrays[i].stride;
        mapping.le = g_gxState.arrays[i].le;
        vtxOffset += nbt3 ? 6 : 2;
        break;
      default:
        Log.fatal("populate_pipeline_config: Invalid vertex type {}", type);
      }
    }
    config.shaderConfig.vtxStride = vtxOffset;
  }
  if (primitive == GX_LINES) {
    config.shaderConfig.lineMode = 1;
  } else if (primitive == GX_LINESTRIP) {
    config.shaderConfig.lineMode = 2;
  } else if (primitive == GX_POINTS) {
    config.shaderConfig.lineMode = 3;
  } else {
    config.shaderConfig.lineMode = 0;
  }
  config.shaderConfig.tevSwapTable = g_gxState.tevSwapTable;
  for (u8 i = 0; i < g_gxState.numTevStages; ++i) {
    config.shaderConfig.tevStages[i] = g_gxState.tevStages[i];
  }
  config.shaderConfig.tevStageCount = g_gxState.numTevStages;
  for (u8 i = 0; i < g_gxState.numIndStages; ++i) {
    config.shaderConfig.indStages[i] = g_gxState.indStages[i];
  }
  config.shaderConfig.numIndStages = g_gxState.numIndStages;
  for (u8 i = 0; i < MaxColorChannels; ++i) {
    const auto& cc = g_gxState.colorChannelConfig[i];
    if (cc.lightingEnabled) {
      config.shaderConfig.colorChannels[i] = cc;
    } else {
      // Only matSrc matters when lighting disabled
      config.shaderConfig.colorChannels[i] = {
          .matSrc = cc.matSrc,
      };
    }
  }
  for (u8 i = 0; i < g_gxState.numTexGens; ++i) {
    config.shaderConfig.tcgs[i] = g_gxState.tcgs[i];
  }
  if (g_gxState.alphaCompare) {
    config.shaderConfig.alphaCompare = g_gxState.alphaCompare;
  }
  auto cullMode = config.shaderConfig.lineMode == 0 ? g_gxState.cullMode : GX_CULL_NONE;
  if (g_gxState.projScaleX < 0.f && !gfx::is_offscreen()) {
    if (cullMode == GX_CULL_FRONT) {
      cullMode = GX_CULL_BACK;
    } else if (cullMode == GX_CULL_BACK) {
      cullMode = GX_CULL_FRONT;
    }
  }
  const auto [polygonOffset, polygonOffsetScale] = polygon_offset_for_cull_mode(cullMode);

  // aurora/paper.h: distance-field maps the stages sample, shadow draws, smooth cut-out edges
  if (g_gxState.paperSdfMaps != 0) {
    u8 sampled = 0;
    for (u8 i = 0; i < g_gxState.numTevStages; ++i) {
      const auto map = g_gxState.tevStages[i].texMapId;
      if (map != GX_TEXMAP_NULL && map < MaxTextures) {
        sampled |= static_cast<u8>(1u << map);
      }
    }
    config.shaderConfig.sdfMaps = sampled & g_gxState.paperSdfMaps;
  }
  const bool paperShadow = g_gxState.paperShadow && config.shaderConfig.lineMode == 0 && !gfx::is_offscreen();
  config.shaderConfig.shadowFill = paperShadow ? 1 : 0;
  if (g_gxState.paperEdges != 0 && !paperShadow && config.shaderConfig.lineMode == 0 &&
      paper_cutout_threshold(config.shaderConfig.alphaCompare) >= 0.f) {
    const bool blends = g_gxState.blendMode == GX_BM_BLEND && g_gxState.blendFacSrc == GX_BL_SRCALPHA &&
                        g_gxState.blendFacDst == GX_BL_INVSRCALPHA;
    if (blends) {
      config.shaderConfig.edgeAA = 2;
    } else if (g_gxState.blendMode == GX_BM_NONE && !g_gxState.alphaUpdate && gfx::get_sample_count() > 1 &&
               !gfx::is_offscreen()) {
      config.shaderConfig.edgeAA = 1;
    }
  }

  config = {
      .msaaSamples = gfx::get_sample_count(),
      .shaderConfig = config.shaderConfig,
      .depthFunc = g_gxState.depthFunc,
      .cullMode = cullMode,
      .blendMode = g_gxState.blendMode,
      .blendFacSrc = g_gxState.blendFacSrc,
      .blendFacDst = g_gxState.blendFacDst,
      .blendOp = g_gxState.blendOp,
      .dstAlpha = g_gxState.dstAlpha,
      .polygonOffsetBits = std::bit_cast<uint32_t>(polygonOffset),
      .polygonOffsetScaleBits = std::bit_cast<uint32_t>(polygonOffsetScale),
      .polygonOffsetClampBits = std::bit_cast<uint32_t>(g_gxState.clamp),
      .depthCompare = g_gxState.depthCompare,
      .depthUpdate = cullMode == GX_CULL_ALL ? false : g_gxState.depthUpdate,
      .alphaUpdate = cullMode == GX_CULL_ALL ? false : g_gxState.alphaUpdate,
      .colorUpdate = cullMode == GX_CULL_ALL ? false : g_gxState.colorUpdate,
  };
  if (paperShadow) {
    config.depthFunc = GX_LESS;
    config.cullMode = GX_CULL_NONE;
    config.blendMode = GX_BM_BLEND;
    config.blendFacSrc = GX_BL_SRCALPHA;
    config.blendFacDst = GX_BL_INVSRCALPHA;
    config.blendOp = GX_LO_CLEAR;
    config.dstAlpha = UINT32_MAX;
    config.depthCompare = true;
    config.depthUpdate = true;
    config.alphaUpdate = false;
    config.colorUpdate = true;
  }
}

GXBindGroups build_bind_groups(const ShaderInfo& info) noexcept {
  ZoneScoped;

  if (!info.sampledTextures.any() && !info.sampledIndTextures.any()) {
    // Don't bother re-binding anything
    return {};
  }

  // Up to 8 texture+sampler pairs; unused slots get the 1x1 empty fill. Texture
  // unit i == GX slot i; the GLSL sampler uniforms are bound to those units at link.
  gl::BindingSet set{};
  for (u32 i = 0; i < MaxTextures; ++i) {
    const auto& tex = g_gxState.textures[i];
    auto& binding = set.textures[i];
    if (tex && (info.sampledTextures[i] || info.sampledIndTextures[i])) {
      binding.texture = tex.ref->sampleTextureView.id;
      binding.sampler = tex.sampler.id != 0 ? tex.sampler.id : gfx::sampler_ref(tex.get_descriptor()).id;
    } else {
      binding.texture = sEmptyTexture.id;
      binding.sampler = sEmptySampler.id;
    }
  }
  struct FastBindEntry {
    std::array<gl::BindingSet::TextureBinding, MaxTextures> textures;
    gfx::BindGroupRef ref = 0;
    u32 frame = UINT32_MAX;
  };
  static constexpr size_t kFastBindCacheSize = 16;
  static FastBindEntry s_fastBindCache[kFastBindCacheSize]{};
  static size_t s_fastBindHead = 0;
  const u32 currentFrame = gfx::current_frame();

  for (size_t k = 0; k < kFastBindCacheSize; ++k) {
    const size_t idx = (s_fastBindHead + kFastBindCacheSize - 1 - k) % kFastBindCacheSize;
    if (s_fastBindCache[idx].ref != 0 && s_fastBindCache[idx].frame == currentFrame &&
        std::memcmp(s_fastBindCache[idx].textures.data(), set.textures.data(), sizeof(set.textures)) == 0) {
      return {
          .textureBindGroup = s_fastBindCache[idx].ref,
      };
    }
  }

  const auto ref = gfx::bind_group_ref(set);
  s_fastBindCache[s_fastBindHead].textures = set.textures;
  s_fastBindCache[s_fastBindHead].ref = ref;
  s_fastBindCache[s_fastBindHead].frame = currentFrame;
  s_fastBindHead = (s_fastBindHead + 1) % kFastBindCacheSize;
  return {
      .textureBindGroup = ref,
  };
}

void initialize() noexcept {
  // GL has no bind-group-layout / pipeline-layout objects to build. But unused GX
  // texture slots still need a *real* GL texture + sampler bound: a slot left at id 0
  // binds the incomplete default texture, which shows up as glBindTexture(No Resource)
  // in captures and which strict drivers (Mali/PowerVR) can reject on a draw whose
  // GLSL declares that sampler. Build a 1x1 fill here. These are GL objects, so they
  // must be created on the render worker (initialize() runs on the main thread, which
  // does not own the render context); create_gl_texture/create_gl_sampler marshal the
  // work to the worker and block. `renderable=true` makes create_texture zero-fill
  // level 0, so the fill reads opaque black rather than sampling uninitialized GPU
  // memory (Normalcy Doctrine) — the flag is otherwise inert for a 1x1 that is only
  // ever bound as a sampler, never an FBO attachment.
  sEmptyTexture = gfx::create_gl_texture(webgpu::g_graphicsConfig.surfaceConfiguration.format, gl::Extent3D{1, 1, 1}, 1,
                                         /*renderable=*/true);
  sEmptySampler = gfx::create_gl_sampler(gl::SamplerDescriptor{});
  g_emptyTextureBindGroup = gl::BindingSet{};
}

void shutdown() noexcept {
  sEmptySampler = {};
  sEmptyTexture = {};
  g_emptyTextureBindGroup = {};
  for (auto& item : g_gxState.textures) {
    item.ref.reset();
  }
  s_textureObjectCaches.clear();
  s_staticContentCache.clear();
  s_pointerTextureCache.clear();
  s_tlutObjectCaches.clear();
  g_gxState.loadedTextures.fill({});
  g_gxState.loadedTluts.fill({});
  clear_copy_texture_cache();
  clear_shader_program_cache();
}
} // namespace aurora::gx

namespace aurora {
static gl::AddressMode gl_address_mode(GXTexWrapMode mode) {
  switch (mode) {
    DEFAULT_FATAL("invalid wrap mode {}", underlying(mode));
  case GX_CLAMP:
    return gl::AddressMode::ClampToEdge;
  case GX_REPEAT:
    return gl::AddressMode::Repeat;
  case GX_MIRROR:
    return gl::AddressMode::MirrorRepeat;
  }
}

static std::pair<gl::FilterMode, gl::MipmapFilterMode> gl_filter_mode(GXTexFilter filter) {
  switch (filter) {
    DEFAULT_FATAL("invalid filter mode {}", static_cast<int>(filter));
  case GX_NEAR:
    return {gl::FilterMode::Nearest, gl::MipmapFilterMode::Undefined};
  case GX_LINEAR:
    return {gl::FilterMode::Linear, gl::MipmapFilterMode::Undefined};
  case GX_NEAR_MIP_NEAR:
    return {gl::FilterMode::Nearest, gl::MipmapFilterMode::Nearest};
  case GX_LIN_MIP_NEAR:
    return {gl::FilterMode::Linear, gl::MipmapFilterMode::Nearest};
  case GX_NEAR_MIP_LIN:
    return {gl::FilterMode::Nearest, gl::MipmapFilterMode::Linear};
  case GX_LIN_MIP_LIN:
    return {gl::FilterMode::Linear, gl::MipmapFilterMode::Linear};
  }
}

static u16 gl_aniso(GXAnisotropy aniso) {
  switch (aniso) {
    DEFAULT_FATAL("invalid aniso {}", static_cast<int>(aniso));
  case GX_ANISO_1:
  case GX_MAX_ANISOTROPY:
    return 1;
  case GX_ANISO_2:
    return std::max<u16>(aurora::webgpu::g_graphicsConfig.textureAnisotropy / 2, 1);
  case GX_ANISO_4:
    return std::max<u16>(aurora::webgpu::g_graphicsConfig.textureAnisotropy, 1);
  }
}

gl::SamplerDescriptor aurora::gfx::TextureBind::get_descriptor() const noexcept {
  auto [minFilter, mipFilter] = gl_filter_mode(texObj.min_filter());
  auto [magFilter, _] = gl_filter_mode(texObj.mag_filter());
  const bool mipsEnabled = mipFilter != gl::MipmapFilterMode::Undefined;
  float minLod = texObj.min_lod();
  float maxLod = texObj.max_lod();
  u16 maxAnisotropy = gl_aniso(texObj.max_aniso());
  if (ref && ref->isReplacement) {
    minLod = 0.f;
    maxLod = 1000.f;
    if (!mipsEnabled) {
      mipFilter = gl::MipmapFilterMode::Nearest;
    }
  } else if (mipFilter == gl::MipmapFilterMode::Undefined) {
    minLod = 0.f;
    maxLod = 0.f;
  }
  if ((ref && ref->hasArbitraryMips) || !mipsEnabled) {
    maxAnisotropy = 1;
  } else if (maxAnisotropy > 1) {
    magFilter = gl::FilterMode::Linear;
    minFilter = gl::FilterMode::Linear;
    mipFilter = gl::MipmapFilterMode::Linear;
  }
  return {
      .addressU = gl_address_mode(texObj.wrap_s()),
      .addressV = gl_address_mode(texObj.wrap_t()),
      .addressW = gl::AddressMode::Repeat,
      .magFilter = magFilter,
      .minFilter = minFilter,
      .mipmapFilter = mipFilter,
      .lodMinClamp = minLod,
      .lodMaxClamp = maxLod,
      .maxAnisotropy = maxAnisotropy,
  };
}
} // namespace aurora
