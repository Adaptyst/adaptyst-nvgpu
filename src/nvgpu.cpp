// SPDX-FileCopyrightText: 2025 CERN
// SPDX-License-Identifier: GPL-3.0-or-later

#define ADAPTYST_MODULE_ENTRYPOINT
#include <adaptyst/hw.h>

#include <string>
#include <unordered_map>
#include <vector>
#include <regex>
#include <mutex>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>

volatile const char *name = "nvgpu";
volatile const char *version = "0.1.0-dev.2025.11a";
volatile const int version_nums[] = {0, 1, 0, 1, -1};
volatile const char *options[] = { "cuda_api_type", NULL };
volatile const char *tags[] = { NULL };
volatile const char *log_types[] = { NULL };
volatile const unsigned int max_count_per_entity = 1;

volatile const char *cuda_api_type_help = "CUDA API calls to trace "
  "(\"runtime\", \"driver\", or \"both\", default: \"both\")";
volatile const option_type cuda_api_type_type = STRING;
volatile const char *cuda_api_type_default = "both";

namespace fs = std::filesystem;

class NvgpuModule {
private:
  typedef struct Region {
    bool start_defined;
    unsigned long long start;
    bool end_defined;
    unsigned long long end;
  } Region;

  std::string cuda_api_type;
  amod_t module_id;
  std::unordered_map<std::string, std::unordered_map<std::string,
                                                     Region> > regions;
  std::mutex region_lock;
  nlohmann::json data;

public:
  static NvgpuModule *instance;

  NvgpuModule(amod_t module_id,
              std::string cuda_api_type) {
    this->module_id = module_id;
    this->cuda_api_type = cuda_api_type;
    this->data = nlohmann::json::object();
  }

  bool process() {
    adaptyst_profile_notify(this->module_id);

    const char *msg;

    if (!adaptyst_receive_string(this->module_id, &msg)) {
      adaptyst_set_error(this->module_id, "Error when receiving injection "
                         "data from the workflow");
      return false;
    }

    std::string request(msg);

    if (request == "cuda_api_type") {
      if (!adaptyst_send_string(this->module_id, this->cuda_api_type.c_str())) {
        adaptyst_set_error(this->module_id, "Could not send injection reply "
                           "to the workflow");
        return false;
      }
    } else {
      adaptyst_set_error(this->module_id, "Invalid injection request "
                         "received from the workflow");
      return false;
    }

    std::unordered_map<std::string, std::vector<std::pair<std::string, unsigned long long> > > stacks;

    do {
      if (!adaptyst_receive_string_timeout(this->module_id, &msg, 1)) {
        if (adaptyst_get_internal_error_code(this->module_id) == ADAPTYST_ERR_TIMEOUT) {
          if (!adaptyst_is_workflow_running(this->module_id)) {
            break;
          }
        } else {
          adaptyst_set_error(this->module_id, "Error when receiving injection "
                             "data from the workflow");
          return false;
        }
      }

      if (!msg) {
        continue;
      }

      std::smatch match;
      std::string msg_str(msg);

      if (!std::regex_match(msg_str,
                            match,
                            std::regex("^(-?\\d+) (.+) (enter|exit) (.+)$"))) {
        adaptyst_print(this->module_id,
                       ("Invalid message from the injection part, ignoring: " +
                        msg_str).c_str(), true, false, "General");
        continue;
      }

      adaptyst_log(this->module_id, msg, "General");

      std::string timestamp_str = match[1].str();

      if (timestamp_str == "-1") {
        adaptyst_print(this->module_id,
                       ("Unknown timestamp received from the injection part, ignoring: " +
                        msg_str).c_str(), true, false, "General");
        continue;
      }

      unsigned long long timestamp = std::stoull(timestamp_str);

      std::string part_id = match[2].str();
      std::vector<std::string> applicable_regions;

      {
        std::unique_lock lock(this->region_lock);
        if (this->regions.find(part_id) == this->regions.end()) {
          adaptyst_print(this->module_id,
                         (part_id + " doesn't seem to have any active regions, ignoring: " +
                          msg_str).c_str(), true, false, "General");
          continue;
        }

        for (auto &region : this->regions[part_id]) {
          Region &data = region.second;

          if ((data.start_defined && data.end_defined &&
               timestamp >= data.start && timestamp <= data.end) ||
              (data.start_defined && !data.end_defined &&
               timestamp >= data.start) ||
              (!data.start_defined && data.end_defined &&
               timestamp <= data.end) ||
              (!data.start_defined && !data.end_defined)) {
            applicable_regions.push_back(region.first);
          }
        }
      }

      std::string state = match[3].str();
      std::string func_name = match[4].str();

      for (auto &region_name : applicable_regions) {
        if (state == "enter") {
          if (stacks.find(region_name) == stacks.end()) {
            stacks[region_name] = std::vector<std::pair<std::string, unsigned long long> >();
          }

          stacks[region_name].push_back(std::make_pair(func_name, timestamp));
        } else if (state == "exit") {
          if (stacks.find(region_name) == stacks.end() ||
              stacks[region_name].empty() ||
              stacks[region_name][stacks[region_name].size() - 1].first !=
                  func_name) {
            adaptyst_print(this->module_id,
                           ("Received message from the injection part doesn't correspond to "
                           "the current stack of region "
                            "\"" + region_name + "\", ignoring: " + msg_str).c_str(),
                           true, false, "General");
            continue;
          }

          auto &cur_stack = stacks[region_name];

          unsigned long long length =
            timestamp - cur_stack[cur_stack.size() - 1].second;

          if (!this->data.contains(region_name)) {
            this->data[region_name] = nlohmann::json::object();
          }

          if (!this->data[region_name].contains("data")) {
            this->data[region_name]["data"] = nlohmann::json::object();
          }

          if (!this->data[region_name]["data"].contains(cur_stack[0].first)) {
            this->data[region_name]["data"][cur_stack[0].first] = nlohmann::json::object();
            this->data[region_name]["data"][cur_stack[0].first]["length"] = 0ULL;
            this->data[region_name]["data"][cur_stack[0].first]["children"] =
              nlohmann::json::object();
          }

          this->data[region_name]["data"][cur_stack[0].first]["length"] =
            (unsigned long long)this->data[region_name]["data"][cur_stack[0].first]["length"] + length;

          nlohmann::json *cur_object = &this->data[region_name]["data"][cur_stack[0].first]["children"];

          for (int i = 1; i < cur_stack.size(); i++) {
            if (!cur_object->contains(cur_stack[i].first)) {
              (*cur_object)[cur_stack[i].first] = nlohmann::json::object();
              (*cur_object)[cur_stack[i].first]["length"] = 0ULL;
              (*cur_object)[cur_stack[i].first]["children"] = nlohmann::json::object();
            }

            (*cur_object)[cur_stack[i].first]["length"] =
              (unsigned long long)(*cur_object)[cur_stack[i].first]["length"] + length;

            cur_object = &(*cur_object)[cur_stack[i].first]["children"];
          }

          cur_stack.pop_back();
        }
      }
    } while (msg);

    adaptyst_profile_wait(this->module_id);

    {
      std::unique_lock lock(this->region_lock);
      for (auto &part_id : this->regions) {
        for (auto &region : part_id.second) {
          std::string name = region.first;
          Region &data = region.second;

          if (!this->data.contains(name)) {
            this->data[name] = nlohmann::json::object();
            this->data[name]["data"] = nlohmann::json::object();
          }

          unsigned long long start, end;
          unsigned long long workflow_start_time = adaptyst_get_workflow_start_time(this->module_id);

          if (adaptyst_get_internal_error_code(this->module_id) !=
              ADAPTYST_OK) {
            return false;
          }

          if (!data.start_defined) {
            start = 0;
          } else {
            start = data.start - workflow_start_time;
          }

          if (!data.end_defined) {
            end = adaptyst_get_workflow_end_time(this->module_id);
            if (adaptyst_get_internal_error_code(this->module_id) !=
                ADAPTYST_OK) {
              return false;
            }
          } else {
            end = data.end - workflow_start_time;
          }

          this->data[name]["length"] = (unsigned long long)(end - start);
          this->data[name]["start"] = start;
        }
      }
    }

    const char *dir = adaptyst_get_module_dir(this->module_id);

    if (!dir) {
      adaptyst_set_error(this->module_id, "adaptyst_get_module_dir() returned null");
      return false;
    }

    fs::path path = fs::path(adaptyst_get_module_dir(this->module_id)) / "regions.json";
    std::ofstream stream(path);

    if (!stream) {
      adaptyst_set_error(this->module_id,
                         ("Could not open " + path.string()).c_str());
      return false;
    }

    if (!(stream << this->data.dump() << std::endl)) {
      adaptyst_set_error(this->module_id,
                         ("Could not write data to " + path.string()).c_str());
      return false;
    }

    return true;
  }

  void region_start(std::string name, std::string part_id,
                    std::string timestamp_str) {
    Region region;

    if (timestamp_str == "-1") {
      region.start_defined = false;
      region.start = 0;
    } else {
      region.start_defined = true;
      region.start = std::stoull(timestamp_str);
    }

    region.end_defined = false;
    region.end = 0;

    {
      std::unique_lock lock(this->region_lock);
      if (this->regions.find(part_id) == this->regions.end()) {
        this->regions[part_id] = std::unordered_map<std::string, Region>();
      }

      this->regions[part_id][name] = region;
    }
  }

  void region_end(std::string name, std::string part_id,
                  std::string timestamp_str) {
    if (timestamp_str == "-1") {
      return;
    }

    {
      std::unique_lock lock(this->region_lock);
      Region &region = this->regions[part_id][name];
      region.end_defined = true;
      region.end = std::stoull(timestamp_str);
    }
  }
};

NvgpuModule *NvgpuModule::instance = nullptr;

extern "C" {
  bool adaptyst_module_init(amod_t module_id) {
    option *cuda_api_type_opt = adaptyst_get_option(module_id, "cuda_api_type");
    std::string cuda_api_type(*(const char **)cuda_api_type_opt->data);

    if (cuda_api_type != "runtime" && cuda_api_type != "driver" &&
        cuda_api_type != "both") {
      adaptyst_set_error(module_id, "cuda_api_type must be one of: "
                         "\"runtime\", \"driver\", or \"both\"");
      return false;
    }

    try {
      NvgpuModule::instance = new NvgpuModule(module_id, cuda_api_type);
    } catch (std::exception &e) {
      adaptyst_set_error(module_id, e.what());
      return false;
    }

    bool success = adaptyst_set_will_profile(module_id, true);

    if (!success) {
      adaptyst_set_error(module_id, "adaptyst_set_will_profile() returned false");
      return false;
    }

    return true;
  }

  bool adaptyst_module_process(amod_t module_id, const char *sdfg) {
    try {
      return NvgpuModule::instance->process();
    } catch (std::exception &e) {
      adaptyst_set_error(module_id, e.what());
      return false;
    }
  }

  void adaptyst_module_close(amod_t module_id) {
    delete NvgpuModule::instance;
  }

  bool adaptyst_region_start(amod_t module_id, const char *name,
                             const char *part_id, const char *timestamp_str) {
    try {
      NvgpuModule::instance->region_start(
          std::string(name), std::string(part_id), std::string(timestamp_str));
      return true;
    } catch (std::exception &e) {
      adaptyst_set_error(module_id, e.what());
      return false;
    }
  }

  bool adaptyst_region_end(amod_t module_id, const char *name,
                           const char *part_id, const char *timestamp_str) {
    try {
      NvgpuModule::instance->region_end(std::string(name), std::string(part_id),
                                        std::string(timestamp_str));
      return true;
    } catch (std::exception &e) {
      adaptyst_set_error(module_id, e.what());
      return false;
    }
  }
}
