#ifndef G_BGEN_CAPI_H
#define G_BGEN_CAPI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct g_bgen_reader g_bgen_reader;

typedef enum g_bgen_status {
  G_BGEN_STATUS_OK = 0,
  G_BGEN_STATUS_NULL_POINTER = 1,
  G_BGEN_STATUS_INVALID_ARGUMENT = 2,
  G_BGEN_STATUS_READER_ERROR = 3,
  G_BGEN_STATUS_PANIC = 4
} g_bgen_status;

g_bgen_status g_bgen_reader_open(
    const char *bgen_path,
    bool trusted_no_missing_diploid,
    g_bgen_reader **reader_out);

void g_bgen_reader_close(g_bgen_reader *reader);

size_t g_bgen_reader_last_error(
    const g_bgen_reader *reader,
    char *message_buffer,
    size_t message_buffer_length);

g_bgen_status g_bgen_reader_sample_count(
    const g_bgen_reader *reader,
    size_t *sample_count_out);

g_bgen_status g_bgen_reader_variant_count(
    const g_bgen_reader *reader,
    size_t *variant_count_out);

g_bgen_status g_bgen_reader_variant_offset(
    const g_bgen_reader *reader,
    size_t variant_index,
    uint64_t *variant_offset_out);

g_bgen_status g_bgen_reader_prepare_samples(
    g_bgen_reader *reader,
    const int64_t *sample_indices,
    size_t sample_count);

g_bgen_status g_bgen_reader_read_variant_major_dosage_by_indices(
    g_bgen_reader *reader,
    const size_t *variant_indices,
    size_t variant_count,
    float *output_values,
    size_t output_value_count);

g_bgen_status g_bgen_reader_read_variant_major_dosage_by_offsets(
    g_bgen_reader *reader,
    const uint64_t *variant_offsets,
    size_t variant_count,
    float *output_values,
    size_t output_value_count);

#ifdef __cplusplus
}
#endif

#endif
