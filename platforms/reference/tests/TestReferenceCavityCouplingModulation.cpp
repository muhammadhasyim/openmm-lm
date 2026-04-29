/* -------------------------------------------------------------------------- *
 * Tests for GPU-side coupling modulation — Reference platform.               *
 * -------------------------------------------------------------------------- */

#include "openmm/ReferencePlatform.h"

using namespace OpenMM;

ReferencePlatform referencePlatform;
Platform& platform = referencePlatform;

void initializeTests(int argc, char* argv[]) {
}

#include "TestCavityCouplingModulation.h"

void runPlatformTests() {
}
