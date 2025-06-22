#include <adaptyst/output.hpp>

#define ADAPTYST_MODULE_ENTRYPOINT
#include <adaptyst/hw.h>

volatile const char *options[] = { NULL };
volatile const char *tags[] = { NULL };
volatile const char *log_types[] = { "General", NULL };

using namespace adaptyst;

class NVGPUModule {
public:
  static NVGPUModule *instance;

  bool init() {

  }

  bool process(const char *sdfg) {

  }
};

amod_t module_id = 0;
NVGPUModule *NVGPUModule::instance = nullptr;

extern "C" {
  bool adaptyst_module_init() {
    try {
      NVGPUModule::instance = new NVGPUModule();
    } catch (std::bad_alloc &e) {
      adaptyst_set_error(module_id, "Could not allocate memory for NVGPUModule");
      return false;
    } catch (std::exception &e) {
      adaptyst_set_error(module_id, ("An exception has occurred: " +
                                     std::string(e.what())).c_str());
      return false;
    }

    return NVGPUModule::instance->init();
  }

  bool adaptyst_module_process(const char *sdfg) {
    return NVGPUModule::instance->process(sdfg);
  }

  void adaptyst_module_close() {
    delete NVGPUModule::instance;
  }
}
