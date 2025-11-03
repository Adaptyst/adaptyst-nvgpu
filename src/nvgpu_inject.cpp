// SPDX-FileCopyrightText: 2025 CERN
// SPDX-License-Identifier: GPL-3.0-or-later

// This code contains commented sections using the concurrent
// queue library from https://github.com/cameron314/concurrentqueue.
// These are commented out because there is a bug causing a segmentation
// fault. TODO: investigate this

#include <adaptyst/hw_inject.h>
#include <cupti.h>
#include <string>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <future>
#include <sstream>
//#include <blockingconcurrentqueue.h>

class NvgpuInjection {
public:
  typedef enum ApiType {
    RUNTIME,
    DRIVER,
    BOTH
  } ApiType;

  NvgpuInjection(amod_t module_id, ApiType cuda_api_type) {
    this->status = ADAPTYST_MODULE_OK;
    this->module_id = module_id;
    this->cuda_api_type = cuda_api_type;

    CUptiResult result = cuptiSubscribe(&this->handle, NvgpuInjection::callback,
                                        this);

    if (result != CUPTI_SUCCESS) {
      if (result == CUPTI_ERROR_NOT_INITIALIZED) {
        adaptyst_set_error("Could not initialise CUPTI");
        this->status = ADAPTYST_MODULE_ERR;
        return;
      } else if (result == CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED) {
        adaptyst_set_error("Another NVIDIA GPU tool is already "
                           "attached");
        this->status = ADAPTYST_MODULE_ERR;
        return;
      } else if (result == CUPTI_ERROR_INVALID_PARAMETER) {
        adaptyst_set_error("Invalid parameters have been passed to "
                           "cuptiSubscribe() (this is a bug, report it)");
        this->status = ADAPTYST_MODULE_ERR;
        return;
      } else {
        adaptyst_set_error("Unknown error when calling "
                           "cuptiSubscribe()");
        this->status = ADAPTYST_MODULE_ERR;
        return;
      }
    }

    // this->messenger = std::async([this]() {
    //   std::string msg;
    //   this->messages.wait_dequeue(msg);

    //   while (msg != "stop") {
    //     adaptyst_send_string(this->module_id, msg.c_str());
    //     this->messages.wait_dequeue(msg);
    //   }
    // });

    this->subscribed = true;
  }

  ~NvgpuInjection() {
    if (this->subscribed) {
      cuptiUnsubscribe(this->handle);
      cuptiFinalize();
    }

    // this->messages.enqueue("stop");
  }

  int get_status() {
    return this->status;
  }

  int start(std::string part_id) {
    if (this->active_threads.empty()) {
      CUptiResult result;

      if (this->cuda_api_type == RUNTIME || this->cuda_api_type == BOTH) {
        result = cuptiEnableDomain(1, this->handle,
                                   CUPTI_CB_DOMAIN_RUNTIME_API);

        if (result != CUPTI_SUCCESS) {
          adaptyst_set_error(("cuptiEnableDomain() returned " +
                              std::to_string(result) + " for " +
                              "runtime API").c_str());
          return ADAPTYST_MODULE_ERR;
        }
      }

      if (this->cuda_api_type == DRIVER || this->cuda_api_type == BOTH) {
        result = cuptiEnableDomain(1, this->handle,
                                   CUPTI_CB_DOMAIN_DRIVER_API);

        if (result != CUPTI_SUCCESS) {
          cuptiEnableDomain(0, this->handle, CUPTI_CB_DOMAIN_RUNTIME_API);
          adaptyst_set_error(("cuptiEnableDomain() returned " +
                              std::to_string(result) + " for " + "driver API")
                             .c_str());
          return ADAPTYST_MODULE_ERR;
        }
      }
    }

    if (this->active_threads.find(part_id) == this->active_threads.end()) {
      this->active_threads[part_id] = 1;
    } else {
      this->active_threads[part_id]++;
    }

    return ADAPTYST_MODULE_OK;
  }

  int stop(std::string part_id) {
    this->active_threads[part_id]--;

    if (this->active_threads[part_id] == 0) {
      this->active_threads.erase(part_id);
    }

    if (this->active_threads.empty()) {
      cuptiEnableDomain(0, this->handle, CUPTI_CB_DOMAIN_RUNTIME_API);
      cuptiEnableDomain(0, this->handle, CUPTI_CB_DOMAIN_DRIVER_API);
    }

    return ADAPTYST_MODULE_OK;
  }

private:
  static void callback(void *userdata, CUpti_CallbackDomain domain,
                       CUpti_CallbackId cbid, const void *cbdata) {
    NvgpuInjection *obj = (NvgpuInjection *)userdata;

    std::string part_id = std::to_string(getpid()) + "_" +
      std::to_string(gettid());

    if (obj->active_threads.find(part_id) == obj->active_threads.end()) {
      return;
    }

    const CUpti_CallbackData *data = (const CUpti_CallbackData *)cbdata;

    int error;
    std::stringstream stream;
    stream << adaptyst_get_timestamp(&error) << " ";

    stream << part_id << " ";

    if (data->callbackSite == CUPTI_API_ENTER) {
      stream << "enter ";
    } else if (data->callbackSite == CUPTI_API_EXIT) {
      stream << "exit ";
    }

    stream << data->functionName;

    if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
        cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060 ||
        cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000 ||
        cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060 ||
        cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000 ||
        cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000 ||
        cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000 ||
        cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
        cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx ||
        cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz ||
        cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz ||
        cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel ||
        cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz ||
        cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice) {
      if (data->symbolName) {
        stream << " " << data->symbolName;
      }
    }

    adaptyst_send_string(obj->module_id, stream.str().c_str());
    // obj->messages.enqueue(stream.str());
  }

  CUpti_SubscriberHandle handle;
  bool subscribed;
  int status;
  std::future<void> messenger;
  // moodycamel::BlockingConcurrentQueue<std::string> messages;
  amod_t module_id;
  ApiType cuda_api_type;
  std::unordered_map<std::string, unsigned int> active_threads;
};

static std::unordered_map<amod_t, std::unique_ptr<NvgpuInjection> > injections;

extern "C" {
  int adaptyst_init(amod_t module_id) {
    std::string request = "cuda_api_type";
    if (adaptyst_send_string_nl(module_id, request.c_str()) != 0) {
      adaptyst_set_error_nl("Could not send \"cuda_api_type\" injection request "
                            "to Adaptyst");
      return ADAPTYST_MODULE_ERR;
    }

    const char *msg;

    if (adaptyst_receive_string_nl(module_id, &msg) != 0) {
      adaptyst_set_error_nl("Could not receive injection data from Adaptyst");
      return ADAPTYST_MODULE_ERR;
    }

    std::string reply(msg);

    NvgpuInjection::ApiType type;

    if (reply == "runtime") {
      type = NvgpuInjection::RUNTIME;
    } else if (reply == "driver") {
      type = NvgpuInjection::DRIVER;
    } else if (reply == "both") {
      type = NvgpuInjection::BOTH;
    } else {
      adaptyst_set_error_nl(("Invalid reply to \"cuda_api_type\" received "
                             "from Adaptyst: " + reply).c_str());
      return ADAPTYST_MODULE_ERR;
    }

    try {
      injections[module_id] = std::make_unique<NvgpuInjection>(module_id, type);
      return injections[module_id]->get_status();
    } catch (std::exception &e) {
      adaptyst_set_error_nl(e.what());
      return ADAPTYST_MODULE_ERR;
    }
  }

  int adaptyst_region_start(amod_t module_id, const char *part_id,
                            const char *name, const char *timestamp_str) {
    try {
      return injections[module_id]->start(std::string(part_id));
    } catch (std::exception &e) {
      adaptyst_set_error_nl(e.what());
      return ADAPTYST_MODULE_ERR;
    }
  }

  int adaptyst_region_end(amod_t module_id, const char *part_id,
                          const char *name, const char *timestamp_str) {
    try {
      return injections[module_id]->stop(std::string(part_id));
    } catch (std::exception &e) {
      adaptyst_set_error_nl(e.what());
      return ADAPTYST_MODULE_ERR;
    }
  }

  void adaptyst_close(amod_t module_id) {

  }
}
