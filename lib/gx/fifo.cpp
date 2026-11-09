#include "fifo.hpp"
#include "command_processor.hpp"
#include "../internal.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace aurora::gx::fifo {
static Module Log("aurora::gx::fifo");

namespace detail {
uint8_t* sBufferData = nullptr;
uint32_t sBufferSize = 0;
uint32_t sBufferCapacity = 0;
bool sInDisplayList = false;
uint8_t* sDlBuffer = nullptr;
uint32_t sDlSize = 0;
uint32_t sDlWritePos = 0;
} // namespace detail

struct FifoSlot {
  uint8_t* data = nullptr;
  uint32_t size = 0;
  uint32_t capacity = 0;
};

static FifoSlot sSlots[2]{};
static uint32_t sProducerSlot = 0;
static uint32_t sConsumerSlot = 0;
static std::atomic<uint64_t> sFramesProduced{0};
static std::atomic<uint64_t> sFramesConsumed{0};
static std::mutex sFifoMutex;
static std::condition_variable sProducerCv;
static std::condition_variable sConsumerCv;
static std::atomic<bool> sDoubleBuffering{false};
static std::atomic<bool> sStopping{false};

void init() {
  constexpr uint32_t initialCapacity = 512 * 1024;
  std::lock_guard<std::mutex> lock(sFifoMutex);
  sStopping.store(false, std::memory_order_relaxed);
  for (int i = 0; i < 2; ++i) {
    std::free(sSlots[i].data);
    sSlots[i].data = static_cast<uint8_t*>(std::malloc(initialCapacity));
    sSlots[i].size = 0;
    sSlots[i].capacity = initialCapacity;
  }
  sProducerSlot = 0;
  sConsumerSlot = 0;
  sFramesProduced.store(0, std::memory_order_relaxed);
  sFramesConsumed.store(0, std::memory_order_relaxed);

  detail::sBufferData = sSlots[0].data;
  detail::sBufferSize = 0;
  detail::sBufferCapacity = initialCapacity;
  detail::sInDisplayList = false;
  detail::sDlBuffer = nullptr;
  detail::sDlSize = 0;
  detail::sDlWritePos = 0;
}

void shutdown() {
  sStopping.store(true, std::memory_order_release);
  sProducerCv.notify_all();
  sConsumerCv.notify_all();
  std::lock_guard<std::mutex> lock(sFifoMutex);
  for (int i = 0; i < 2; ++i) {
    std::free(sSlots[i].data);
    sSlots[i].data = nullptr;
    sSlots[i].size = 0;
    sSlots[i].capacity = 0;
  }
  detail::sBufferData = nullptr;
  detail::sBufferSize = 0;
  detail::sBufferCapacity = 0;
}

void set_double_buffering(bool enable) {
  if (sDoubleBuffering.load(std::memory_order_acquire) == enable) return;
  if (enable) {
    std::unique_lock<std::mutex> lock(sFifoMutex);
    sDoubleBuffering.store(true, std::memory_order_release);
    Log.info("Double-buffered GX FIFO enabled");
  } else {
    std::unique_lock<std::mutex> lock(sFifoMutex);
    while (sFramesProduced.load(std::memory_order_acquire) > sFramesConsumed.load(std::memory_order_acquire) &&
           !sStopping.load(std::memory_order_relaxed)) {
      sProducerCv.wait(lock);
    }
    sDoubleBuffering.store(false, std::memory_order_release);
    Log.info("Double-buffered GX FIFO disabled");
  }
}

bool double_buffering_enabled() {
  return sDoubleBuffering.load(std::memory_order_relaxed);
}

void finish_frame_producer() {
  if (!sDoubleBuffering.load(std::memory_order_acquire)) {
    return;
  }
  sSlots[sProducerSlot].size = detail::sBufferSize;
  sSlots[sProducerSlot].data = detail::sBufferData;
  sSlots[sProducerSlot].capacity = detail::sBufferCapacity;

  std::unique_lock<std::mutex> lock(sFifoMutex);
  const uint64_t produced = ++sFramesProduced;
  sConsumerCv.notify_all();

  while (produced - sFramesConsumed.load(std::memory_order_acquire) > 1 &&
         !sStopping.load(std::memory_order_relaxed)) {
    sProducerCv.wait(lock);
  }
  if (sStopping.load(std::memory_order_relaxed)) return;

  sProducerSlot = 1 - sProducerSlot;
  detail::sBufferData = sSlots[sProducerSlot].data;
  detail::sBufferSize = 0;
  detail::sBufferCapacity = sSlots[sProducerSlot].capacity;
}

void write_data_grow(const void* data, uint32_t length) {
  uint32_t needed = detail::sBufferSize + length;
  uint32_t newCap = std::max(detail::sBufferCapacity * 2, needed);
  detail::sBufferData = static_cast<uint8_t*>(std::realloc(detail::sBufferData, newCap));
  std::memcpy(detail::sBufferData + detail::sBufferSize, data, length);
  detail::sBufferSize = needed;
  detail::sBufferCapacity = newCap;
  sSlots[sProducerSlot].data = detail::sBufferData;
  sSlots[sProducerSlot].capacity = newCap;
}

void patch_u32(const uint32_t offset, const uint32_t val) {
  ASSERT(!detail::sInDisplayList && offset + sizeof(uint32_t) <= detail::sBufferSize,
         "fifo::patch_u32: invalid patch offset {} (buffer size {})", offset, detail::sBufferSize);
  const auto out = bswap(val);
  std::memcpy(detail::sBufferData + offset, &out, sizeof(out));
}

void begin_display_list(uint8_t* buf, uint32_t size) {
  detail::sInDisplayList = true;
  detail::sDlBuffer = buf;
  detail::sDlSize = size;
  detail::sDlWritePos = 0;
}

uint32_t end_display_list() {
  detail::sInDisplayList = false;
  uint32_t bytesWritten = detail::sDlWritePos;
  uint32_t padded = (bytesWritten + 31) & ~31u;
  while (detail::sDlWritePos < padded && detail::sDlWritePos < detail::sDlSize) {
    detail::sDlBuffer[detail::sDlWritePos++] = 0;
  }
  detail::sDlBuffer = nullptr;
  detail::sDlSize = 0;
  detail::sDlWritePos = 0;
  return padded;
}

bool in_display_list() { return detail::sInDisplayList; }

void drain() {
  if (!sDoubleBuffering.load(std::memory_order_acquire)) {
    if (detail::sBufferSize == 0) {
      return;
    }
    process(detail::sBufferData, detail::sBufferSize, true);
    detail::sBufferSize = 0;
    return;
  }

  std::unique_lock<std::mutex> lock(sFifoMutex);
  while (sFramesProduced.load(std::memory_order_acquire) <= sFramesConsumed.load(std::memory_order_acquire) &&
         !sStopping.load(std::memory_order_relaxed)) {
    sConsumerCv.wait(lock);
  }
  if (sStopping.load(std::memory_order_relaxed)) return;

  const uint32_t consumeSlot = sConsumerSlot;
  const uint8_t* consumeData = sSlots[consumeSlot].data;
  const uint32_t consumeSize = sSlots[consumeSlot].size;
  lock.unlock();

  if (consumeSize > 0) {
    process(consumeData, consumeSize, true);
  }

  lock.lock();
  sSlots[consumeSlot].size = 0;
  sConsumerSlot = 1 - sConsumerSlot;
  ++sFramesConsumed;
  sProducerCv.notify_all();
}

const uint8_t* get_buffer_data() { return detail::sBufferData; }
uint32_t get_buffer_size() { return detail::sBufferSize; }
void clear_buffer() { detail::sBufferSize = 0; }

} // namespace aurora::gx::fifo
