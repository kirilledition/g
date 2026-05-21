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

#ifndef REGENIE_PROFILE_H
#define REGENIE_PROFILE_H

#include <chrono>
#include <cstdint>
#include <string>

namespace regenie_profile {

bool enabled(void);
void initialize(int argc, char** argv);
void finalize(std::string const& status, std::string const& error_message = "");
void set_metadata(std::string const& name, std::string const& value);
void set_metadata(std::string const& name, int value);
void set_metadata(std::string const& name, uint64_t value);
void increment_counter(std::string const& name, uint64_t count = 1);
void record_stage_seconds(std::string const& name, double seconds);

class ScopedStage {
  public:
    explicit ScopedStage(std::string const& name);
    ~ScopedStage(void);

    ScopedStage(ScopedStage const&) = delete;
    ScopedStage& operator=(ScopedStage const&) = delete;

  private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_time_;
    bool active_;
};

}  // namespace regenie_profile

#endif
