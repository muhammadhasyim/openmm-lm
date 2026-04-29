/* -------------------------------------------------------------------------- *
 * Tests for GPU-side coupling modulation — OpenCL platform.                  *
 * -------------------------------------------------------------------------- */

#include "openmm/OpenCLPlatform.h"
#include <string>

using namespace OpenMM;

OpenCLPlatform openclPlatform;
Platform& platform = openclPlatform;

void initializeTests(int argc, char* argv[]) {
    if (argc > 1)
        openclPlatform.setPropertyDefaultValue("Precision", std::string(argv[1]));
    if (argc > 2)
        openclPlatform.setPropertyDefaultValue("DeviceIndex", std::string(argv[2]));
}

#include "TestCavityCouplingModulation.h"

void runPlatformTests() {
}
