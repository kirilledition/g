/*

   This file is part of the regenie software package.

   Copyright (c) 2020-2024 Joelle Mbatchou, Andrey Ziyatdinov & Jonathan Marchini

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

*/

#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

#include "Regenie.hpp"
#include "RegenieProfile.hpp"

namespace {

struct StageAggregate {
  double seconds = 0.0;
  uint64_t count = 0;
};

bool profile_enabled = false;
bool profile_written = false;
std::string profile_path;
std::vector<std::string> command_arguments;
std::chrono::high_resolution_clock::time_point program_start_time;
std::map<std::string, StageAggregate> stage_aggregates;
std::map<std::string, uint64_t> counters;
std::map<std::string, std::string> metadata;
std::mutex profile_mutex;

std::string json_escape(std::string const& input) {
  std::ostringstream output;
  for(char const character : input) {
    switch(character) {
      case '\\':
        output << "\\\\";
        break;
      case '"':
        output << "\\\"";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        unsigned char const byte = static_cast<unsigned char>(character);
        if(byte < 0x20) {
          output << "\\u00";
          char const* digits = "0123456789abcdef";
          output << digits[(byte >> 4) & 0x0f] << digits[byte & 0x0f];
        } else {
          output << character;
        }
    }
  }
  return output.str();
}

template <typename Value>
void write_numeric_map(std::ofstream& output, std::map<std::string, Value> const& values, int const indent) {
  bool first = true;
  std::string const prefix(indent, ' ');
  for(auto const& item : values) {
    if(!first) output << ",\n";
    output << prefix << "\"" << json_escape(item.first) << "\": " << item.second;
    first = false;
  }
  if(!values.empty()) output << "\n";
}

}  // namespace

namespace regenie_profile {

bool enabled(void) {
  return profile_enabled;
}

void initialize(int argc, char** argv) {
  char const* environment_path = std::getenv("REGENIE_PROFILE_JSON");
  if(environment_path == nullptr || std::string(environment_path).empty()) return;

  std::lock_guard<std::mutex> lock(profile_mutex);
  profile_enabled = true;
  profile_written = false;
  profile_path = environment_path;
  command_arguments.clear();
  for(int index = 0; index < argc; ++index) command_arguments.push_back(argv[index]);
  program_start_time = std::chrono::high_resolution_clock::now();
  stage_aggregates.clear();
  counters.clear();
  metadata.clear();
}

void finalize(std::string const& status, std::string const& error_message) {
  if(!profile_enabled) return;

  std::lock_guard<std::mutex> lock(profile_mutex);
  if(profile_written) return;
  profile_written = true;

  auto const stop_time = std::chrono::high_resolution_clock::now();
  double const total_seconds = std::chrono::duration<double>(stop_time - program_start_time).count();

  std::ofstream output(profile_path.c_str(), std::ios::out);
  if(!output.good()) return;

  output << "{\n";
  output << "  \"schema_version\": 1,\n";
  output << "  \"status\": \"" << json_escape(status) << "\",\n";
  output << "  \"error\": \"" << json_escape(error_message) << "\",\n";
  output << "  \"total_wall_time_seconds\": " << total_seconds << ",\n";
  output << "  \"argv\": [";
  for(size_t index = 0; index < command_arguments.size(); ++index) {
    if(index > 0) output << ", ";
    output << "\"" << json_escape(command_arguments[index]) << "\"";
  }
  output << "],\n";

  output << "  \"metadata\": {";
  if(!metadata.empty()) output << "\n";
  bool first_metadata = true;
  for(auto const& item : metadata) {
    if(!first_metadata) output << ",\n";
    output << "    \"" << json_escape(item.first) << "\": \"" << json_escape(item.second) << "\"";
    first_metadata = false;
  }
  if(!metadata.empty()) output << "\n  ";
  output << "},\n";

  output << "  \"stage_totals_seconds\": {";
  if(!stage_aggregates.empty()) output << "\n";
  bool first_stage = true;
  for(auto const& item : stage_aggregates) {
    if(!first_stage) output << ",\n";
    output << "    \"" << json_escape(item.first) << "\": " << item.second.seconds;
    first_stage = false;
  }
  if(!stage_aggregates.empty()) output << "\n  ";
  output << "},\n";

  output << "  \"stage_counts\": {";
  if(!stage_aggregates.empty()) output << "\n";
  first_stage = true;
  for(auto const& item : stage_aggregates) {
    if(!first_stage) output << ",\n";
    output << "    \"" << json_escape(item.first) << "\": " << item.second.count;
    first_stage = false;
  }
  if(!stage_aggregates.empty()) output << "\n  ";
  output << "},\n";

  output << "  \"counters\": {";
  if(!counters.empty()) output << "\n";
  write_numeric_map(output, counters, 4);
  output << "  }\n";
  output << "}\n";
}

void set_metadata(std::string const& name, std::string const& value) {
  if(!profile_enabled) return;
  std::lock_guard<std::mutex> lock(profile_mutex);
  metadata[name] = value;
}

void set_metadata(std::string const& name, int value) {
  set_metadata(name, std::to_string(value));
}

void set_metadata(std::string const& name, uint64_t value) {
  set_metadata(name, std::to_string(value));
}

void increment_counter(std::string const& name, uint64_t count) {
  if(!profile_enabled) return;
  std::lock_guard<std::mutex> lock(profile_mutex);
  counters[name] += count;
}

void record_stage_seconds(std::string const& name, double seconds) {
  if(!profile_enabled) return;
  std::lock_guard<std::mutex> lock(profile_mutex);
  stage_aggregates[name].seconds += seconds;
  stage_aggregates[name].count += 1;
}

ScopedStage::ScopedStage(std::string const& name) : name_(name), start_time_(std::chrono::high_resolution_clock::now()), active_(profile_enabled) {}

ScopedStage::~ScopedStage(void) {
  if(!active_) return;
  auto const stop_time = std::chrono::high_resolution_clock::now();
  record_stage_seconds(name_, std::chrono::duration<double>(stop_time - start_time_).count());
}

}  // namespace regenie_profile
