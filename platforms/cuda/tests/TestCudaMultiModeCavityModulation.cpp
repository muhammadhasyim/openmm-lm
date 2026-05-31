/* Tests for per-mode adaptive modulation — CUDA platform. */

#include "CudaPlatform.h"
#include <string>

using namespace OpenMM;

CudaPlatform cudaPlatform;
Platform& platform = cudaPlatform;

void initializeTests(int argc, char* argv[]) {
    if (argc > 1)
        cudaPlatform.setPropertyDefaultValue("Precision", std::string(argv[1]));
    if (argc > 2)
        cudaPlatform.setPropertyDefaultValue("DeviceIndex", std::string(argv[2]));
}

#include "TestMultiModeCavityModulation.h"

void runPlatformTests() {
}
