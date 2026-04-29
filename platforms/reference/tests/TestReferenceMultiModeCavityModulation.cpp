/* Tests for per-mode adaptive modulation — Reference platform. */

#include "openmm/ReferencePlatform.h"

using namespace OpenMM;

ReferencePlatform referencePlatform;
Platform& platform = referencePlatform;

void initializeTests(int argc, char* argv[]) {
}

#include "TestMultiModeCavityModulation.h"

void runPlatformTests() {
}
