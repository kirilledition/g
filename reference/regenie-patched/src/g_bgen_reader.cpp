#include "g_bgen_reader.hpp"

#ifdef USE_G_BGEN_READER

GBgenReader::GBgenReader() : reader_(nullptr), last_error_("") {}

GBgenReader::~GBgenReader() {
  if (reader_ != nullptr) {
    g_bgen_reader_close(reader_);
    reader_ = nullptr;
  }
}

bool GBgenReader::open(const std::string& bgen_file, bool trusted_no_missing_diploid) {
  if (reader_ != nullptr) {
    return true;
  }

  g_bgen_reader* opened_reader = nullptr;
  g_bgen_status const status =
      g_bgen_reader_open(bgen_file.c_str(), trusted_no_missing_diploid, &opened_reader);
  if (status != G_BGEN_STATUS_OK) {
    last_error_ = "Rust BGEN reader failed to open file.";
    return false;
  }
  reader_ = opened_reader;
  last_error_.clear();
  return true;
}

bool GBgenReader::prepare_samples(const std::vector<int64_t>& sample_indices) {
  if (reader_ == nullptr) {
    last_error_ = "Rust BGEN reader is not open.";
    return false;
  }
  g_bgen_status const status =
      g_bgen_reader_prepare_samples(reader_, sample_indices.data(), sample_indices.size());
  if (status != G_BGEN_STATUS_OK) {
    capture_reader_error();
    return false;
  }
  last_error_.clear();
  return true;
}

bool GBgenReader::read_variant_major_dosage_by_offsets(
    const std::vector<uint64_t>& variant_offsets,
    std::vector<float>& output_values) {
  if (reader_ == nullptr) {
    last_error_ = "Rust BGEN reader is not open.";
    return false;
  }
  g_bgen_status const status = g_bgen_reader_read_variant_major_dosage_by_offsets(
      reader_,
      variant_offsets.data(),
      variant_offsets.size(),
      output_values.data(),
      output_values.size());
  if (status != G_BGEN_STATUS_OK) {
    capture_reader_error();
    return false;
  }
  last_error_.clear();
  return true;
}

bool GBgenReader::is_open() const {
  return reader_ != nullptr;
}

std::string GBgenReader::last_error() const {
  return last_error_;
}

void GBgenReader::capture_reader_error() {
  if (reader_ == nullptr) {
    last_error_ = "Rust BGEN reader is not open.";
    return;
  }

  std::vector<char> message_buffer(1024, '\0');
  size_t const required_size =
      g_bgen_reader_last_error(reader_, message_buffer.data(), message_buffer.size());
  if (required_size > message_buffer.size()) {
    message_buffer.assign(required_size, '\0');
    g_bgen_reader_last_error(reader_, message_buffer.data(), message_buffer.size());
  }
  last_error_ = std::string(message_buffer.data());
}

#endif
