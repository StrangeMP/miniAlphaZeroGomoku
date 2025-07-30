#pragma once

#include "concurrentqueue.h"
#include <atomic>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <format>

class AsyncLogger {
public:
  static AsyncLogger &getInstance();

  // Deleted copy and move constructors and assignment operators
  AsyncLogger(const AsyncLogger &) = delete;
  void operator=(const AsyncLogger &) = delete;

  template <typename... Args> void log(std::format_string<Args...> fmt, Args &&...args) {
    if (running_) {
      log_queue_.enqueue(std::format(fmt, std::forward<Args>(args)...));
    }
  }

  void stop();

private:
  AsyncLogger(const std::string &filename = "dispatcher.log");
  ~AsyncLogger();

  void writerLoop();

  std::atomic<bool> running_{false};
  moodycamel::ConcurrentQueue<std::string> log_queue_;
  std::unique_ptr<std::thread> writer_thread_;
  std::ofstream log_file_;
};
