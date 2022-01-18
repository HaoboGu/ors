#include "onnxruntime_c_api.h"
#include "cpu_provider_factory.h"
#include "onnxruntime_run_options_config_keys.h"
#include "onnxruntime_session_options_config_keys.h"
#include "tensorrt_provider_factory.h"
// Provider options header seems a c++ header
// #include "provider_options.h"