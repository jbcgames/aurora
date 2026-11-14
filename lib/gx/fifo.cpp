#include "fifo.hpp"
#include "command_processor.hpp"
#include "../internal.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

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

// Async end-of-frame drain thread state: drains the previous producer slot while the main thread
// concurrently runs flush_present().
static std::thread sAsyncDrainThread;
static std::mutex sAsyncDrainMutex;
static std::condition_variable sAsyncDrainCv;
static bool sAsyncDrainBusy = false;
static bool sAsyncDrainStop = false;
static uint32_t sAsyncConsumeSlot = 0;
static double sAsyncDrainLastMs = 0.0;

static void async_drain_worker() {
  for (;;) {
    uint32_t slot = 0;
    {
      std::unique_lock<std::mutex> lock(sAsyncDrainMutex);
      sAsyncDrainCv.wait(lock, [] { return sAsyncDrainBusy || sAsyncDrainStop; });
      if (sAsyncDrainStop && !sAsyncDrainBusy) {
        return;
      }
      slot = sAsyncConsumeSlot;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t sz = sSlots[slot].size;
    if (sz > 0) {
      process(sSlots[slot].data, sz, true);
      sSlots[slot].size = 0;
    }
    const auto t1 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(sAsyncDrainMutex);
      sAsyncDrainLastMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
      sAsyncDrainBusy = false;
    }
    sAsyncDrainCv.notify_all();
  }
}

void init() {
  constexpr uint32_t initialCapacity = 512 * 1024;
  {
    std::lock_guard<std::mutex> alock(sAsyncDrainMutex);
    sAsyncDrainStop = false;
    sAsyncDrainBusy = false;
    sAsyncDrainLastMs = 0.0;
  }
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
  std::thread threadToJoin;
  {
    std::lock_guard<std::mutex> lock(sAsyncDrainMutex);
    sAsyncDrainStop = true;
    threadToJoin = std::move(sAsyncDrainThread);
  }
  sAsyncDrainCv.notify_all();
  if (threadToJoin.joinable()) {
    threadToJoin.join();
  }

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
    drain_async_wait();
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

void drain_async_start() {
  if (sDoubleBuffering.load(std::memory_order_acquire) || detail::sInDisplayList) {
    return;
  }
  {
    std::unique_lock<std::mutex> lock(sAsyncDrainMutex);
    sAsyncDrainCv.wait(lock, [] { return !sAsyncDrainBusy; });
    if (detail::sBufferSize == 0) {
      sAsyncDrainLastMs = 0.0;
      return;
    }
    if (!sAsyncDrainThread.joinable() && !sAsyncDrainStop) {
      sAsyncDrainThread = std::thread(async_drain_worker);
    }
  }
  const uint32_t consumeSlot = sProducerSlot;
  const uint32_t nextSlot = 1u - sProducerSlot;
  if (sSlots[nextSlot].data == nullptr) {
    drain();
    return;
  }
  sSlots[consumeSlot].data = detail::sBufferData;
  sSlots[consumeSlot].size = detail::sBufferSize;
  sSlots[consumeSlot].capacity = detail::sBufferCapacity;

  sProducerSlot = nextSlot;
  detail::sBufferData = sSlots[nextSlot].data;
  detail::sBufferSize = 0;
  detail::sBufferCapacity = sSlots[nextSlot].capacity;

  {
    std::lock_guard<std::mutex> lock(sAsyncDrainMutex);
    sAsyncConsumeSlot = consumeSlot;
    sAsyncDrainLastMs = 0.0;
    sAsyncDrainBusy = true;
  }
  sAsyncDrainCv.notify_one();
}

double drain_async_wait() {
  std::unique_lock<std::mutex> lock(sAsyncDrainMutex);
  sAsyncDrainCv.wait(lock, [] { return !sAsyncDrainBusy; });
  return sAsyncDrainLastMs;
}

const uint8_t* get_buffer_data() { return detail::sBufferData; }
uint32_t get_buffer_size() { return detail::sBufferSize; }
void clear_buffer() { detail::sBufferSize = 0; }

} // namespace aurora::gx::fifo
