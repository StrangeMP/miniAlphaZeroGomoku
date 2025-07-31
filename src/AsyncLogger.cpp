#include "AsyncLogger.hpp"
#include <chrono>
#include <string_view>

AsyncLogger &AsyncLogger::getInstance(std::string_view filename) {
  static AsyncLogger instance(filename);
  return instance;
}

AsyncLogger::AsyncLogger(std::string_view filename) {
  log_file_.open(filename.data(), std::ios::out);
  if (!log_file_.is_open()) {
    throw std::runtime_error("Failed to open log file.");
  }
  running_.store(true);
  writer_thread_ = std::make_unique<std::thread>(&AsyncLogger::writerLoop, this);
}

AsyncLogger::~AsyncLogger() {
  if (running_.load()) {
    stop();
  }
}

void AsyncLogger::stop() {
  if (!running_.exchange(false)) {
    return; // Already stopped
  }

  // Wait for the writer thread to finish processing remaining messages
  if (writer_thread_ && writer_thread_->joinable()) {
    writer_thread_->join();
  }

  // Final flush of any remaining items after the thread has stopped
    std::string message;
    while (log_queue_.try_dequeue(message)) {
        if (log_file_.is_open()) {
            log_file_ << message << std::endl;
        }
    }


  if (log_file_.is_open()) {
    log_file_.close();
  }
}

void AsyncLogger::writerLoop() {
  std::string message;
  while (running_.load()) {
    // Dequeue and write messages
    if (log_queue_.try_dequeue(message)) {
      if (log_file_.is_open()) {
        log_file_ << message << std::endl;
      }
    }else{
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}
