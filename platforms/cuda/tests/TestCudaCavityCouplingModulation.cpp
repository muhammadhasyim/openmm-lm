/* -------------------------------------------------------------------------- *
 * Tests for GPU-side coupling modulation — CUDA platform.                    *
 * -------------------------------------------------------------------------- */

#include "openmm/CudaPlatform.h"
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

#include "TestCavityCouplingModulation.h"

void runPlatformTests() {
}
