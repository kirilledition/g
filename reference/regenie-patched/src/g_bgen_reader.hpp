#ifndef G_BGEN_READER_HPP
#define G_BGEN_READER_HPP

#ifdef USE_G_BGEN_READER

#include <cstdint>
#include <string>
#include <vector>

#include "g_bgen_capi.h"

class GBgenReader {
 public:
  GBgenReader();
  ~GBgenReader();

  GBgenReader(const GBgenReader&) = delete;
  GBgenReader& operator=(const GBgenReader&) = delete;

  bool open(const std::string& bgen_file, bool trusted_no_missing_diploid);
  bool prepare_samples(const std::vector<int64_t>& sample_indices);
  bool read_variant_major_dosage_by_offsets(
      const std::vector<uint64_t>& variant_offsets,
      std::vector<float>& output_values);
  bool is_open() const;
  std::string last_error() const;

 private:
  g_bgen_reader* reader_;
  std::string last_error_;

  void capture_reader_error();
};

#endif

#endif
