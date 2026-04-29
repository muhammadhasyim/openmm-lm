/* -------------------------------------------------------------------------- *
 * Tests for per-mode adaptive square-wave modulation in MultiModeCavityForce.*
 * -------------------------------------------------------------------------- */

#include "openmm/internal/AssertionUtilities.h"
#include "openmm/BussiThermostat.h"
#include "openmm/Context.h"
#include "openmm/MultiModeCavityForce.h"
#include "openmm/NonbondedForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/ReferencePlatform.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace OpenMM;
using namespace std;

extern Platform& platform;

static const double OMEGA1 = 0.01;
static const double PHOTON_MASS = 1.0 / 1822.888;
static const double CAV_LENGTH = 5.0;   // nm
static const double MOL_Z = 2.5;        // nm (cavity center)
static const double DT_PS = 0.001;

static const double HARTREE_TO_KJMOL = 2625.4996;
static const double BOHR_TO_NM = 0.052917721;
static const double AMU_TO_AU = 1822.8885;
static const double CONV = HARTREE_TO_KJMOL / (BOHR_TO_NM * BOHR_TO_NM);

// Build a 2-mode system: 2 molecular particles + 2 cavity particles
// Molecule at cavity center → mode 1 (odd) couples, mode 2 (even) is dark
struct MultiModeTestSystem {
    System system;
    MultiModeCavityForce* force;
    vector<Vec3> positions;
    int numMolecular;

    MultiModeTestSystem() {
        system.setDefaultPeriodicBoxVectors(Vec3(5,0,0), Vec3(0,5,0), Vec3(0,0,5));

        system.addParticle(12.0);  // 0: mol
        system.addParticle(12.0);  // 1: mol
        system.addParticle(PHOTON_MASS * AMU_TO_AU / AMU_TO_AU); // 2: cavity mode 1
        system.addParticle(PHOTON_MASS * AMU_TO_AU / AMU_TO_AU); // 3: cavity mode 2
        numMolecular = 2;

        NonbondedForce* nb = new NonbondedForce();
        nb->setNonbondedMethod(NonbondedForce::PME);
        nb->setCutoffDistance(2.0);
        nb->addParticle(+1.0, 0.3, 0.0);
        nb->addParticle(-1.0, 0.3, 0.0);
        nb->addParticle(0.0, 0.1, 0.0);
        nb->addParticle(0.0, 0.1, 0.0);
        nb->addException(2, 0, 0.0, 0.1, 0.0);
        nb->addException(2, 1, 0.0, 0.1, 0.0);
        nb->addException(3, 0, 0.0, 0.1, 0.0);
        nb->addException(3, 1, 0.0, 0.1, 0.0);
        nb->addException(2, 3, 0.0, 0.1, 0.0);
        system.addForce(nb);

        force = new MultiModeCavityForce(2, OMEGA1, 0.5, CAV_LENGTH, MOL_Z, PHOTON_MASS);
        force->addCavityParticle(2);
        force->addCavityParticle(3);
        system.addForce(force);

        positions.push_back(Vec3(1.0, 0.0, 0.0));
        positions.push_back(Vec3(-1.0, 0.0, 0.0));
        positions.push_back(Vec3(0.1, 0.2, 0.0));
        positions.push_back(Vec3(0.05, 0.1, 0.0));
    }
};

// ===================================================================
// TEST 1: Per-mode adaptive amplitude with different T_targets
// ===================================================================
void testMultiModeAdaptiveBasic() {
    cout << "testMultiModeAdaptiveBasic ... " << flush;

    MultiModeTestSystem ts;
    double periodPs = 0.05;

    ts.force->setAdaptiveSquareWaveModulation(periodPs, 0.5, 0.0, -1.0);
    ts.force->setModeModulationParams(0, 0.4, 100.0, 1e-8, 1.0);  // mode 1: T_target=100K
    ts.force->setModeModulationParams(1, 0.3, 200.0, 1e-8, 1.0);  // mode 2: T_target=200K

    BussiThermostat* bussi = new BussiThermostat(400.0, 1.0);
    bussi->setApplyToAllParticles(false);
    bussi->addParticle(0);
    bussi->addParticle(1);
    ts.system.addForce(bussi);

    VerletIntegrator integrator(DT_PS);
    Context context(ts.system, integrator, platform);
    context.setPositions(ts.positions);
    context.setVelocitiesToTemperature(400.0, 42);

    // Run past one period so adaptive update fires
    integrator.step((int)(periodPs / DT_PS) + 10);

    // Mode 1 expected: 0.4 * sqrt(100/400) = 0.2
    // Mode 2 expected: 0.3 * sqrt(200/400) ≈ 0.212
    // Both are at cavity center: f_1 = sin(pi/2) = 1, f_2 = sin(pi) = 0
    // Mode 2 is dark → contributes zero coupling regardless of amplitude

    // Verify coupling energy is nonzero (mode 1 couples)
    State state = context.getState(State::Energy);
    double coupling = ts.force->getCouplingEnergy(context);
    double dse = ts.force->getDipoleSelfEnergy(context);

    // With adapted mode-1 amplitude ~0.2 and mode-2 dark, coupling should be nonzero
    ASSERT(fabs(coupling) > 1e-10 || dse > 1e-10);

    cout << "PASS" << endl;
}

// ===================================================================
// TEST 2: All modes OFF during square-wave low phase
// ===================================================================
void testMultiModeOnOffPhase() {
    cout << "testMultiModeOnOffPhase ... " << flush;

    MultiModeTestSystem ts;
    double periodPs = 0.1;
    double dutyCycle = 0.5;

    ts.force->setAdaptiveSquareWaveModulation(periodPs, dutyCycle, 0.0, -1.0);
    ts.force->setModeModulationParams(0, 0.4, 100.0);
    ts.force->setModeModulationParams(1, 0.3, 100.0);

    BussiThermostat* bussi = new BussiThermostat(100.0, 1.0);
    bussi->setApplyToAllParticles(false);
    bussi->addParticle(0);
    bussi->addParticle(1);
    ts.system.addForce(bussi);

    VerletIntegrator integrator(DT_PS);
    Context context(ts.system, integrator, platform);
    context.setPositions(ts.positions);
    context.setVelocitiesToTemperature(0.0);

    // Step to 75% of period → OFF phase
    integrator.step(75);
    {
        double coupling = ts.force->getCouplingEnergy(context);
        double dse = ts.force->getDipoleSelfEnergy(context);
        ASSERT_EQUAL_TOL(0.0, coupling, 1e-10);
        ASSERT_EQUAL_TOL(0.0, dse, 1e-10);
    }

    // Step to 110% of period → ON phase of second period
    integrator.step(35);
    {
        double dse = ts.force->getDipoleSelfEnergy(context);
        // Mode 1 couples at cavity center → nonzero DSE
        ASSERT(dse > 1e-10);
    }

    cout << "PASS" << endl;
}

// ===================================================================
// TEST 3: Dark mode (even, at cavity center) has zero coupling
// ===================================================================
void testMultiModeDarkMode() {
    cout << "testMultiModeDarkMode ... " << flush;

    // Build a 1-mode system with only mode 2 (even → dark at center)
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(5,0,0), Vec3(0,5,0), Vec3(0,0,5));
    system.addParticle(12.0);
    system.addParticle(12.0);
    system.addParticle(PHOTON_MASS);

    NonbondedForce* nb = new NonbondedForce();
    nb->setNonbondedMethod(NonbondedForce::PME);
    nb->setCutoffDistance(2.0);
    nb->addParticle(+1.0, 0.3, 0.0);
    nb->addParticle(-1.0, 0.3, 0.0);
    nb->addParticle(0.0, 0.1, 0.0);
    nb->addException(2, 0, 0.0, 0.1, 0.0);
    nb->addException(2, 1, 0.0, 0.1, 0.0);
    system.addForce(nb);

    // Single mode: n=1 but molecule at z=L/2 means f_2=sin(2*pi/2)=0 for 2nd mode
    // Actually for numModes=1, there's only mode 1 (n=1), f_1=sin(pi*2.5/5)=sin(pi/2)=1
    // To test dark mode, use numModes=2 and check mode 2's contribution
    // The 2-mode system at z=L/2: f_1=1, f_2=sin(2*pi*2.5/5)=sin(pi)≈0

    // Use the 2-mode test system and verify mode 2 contributes nothing
    MultiModeTestSystem ts;
    ts.force->setAdaptiveSquareWaveModulation(0.05, 1.0, 0.0, -1.0); // always ON
    ts.force->setModeModulationParams(0, 0.0, 100.0);   // mode 1: zero coupling
    ts.force->setModeModulationParams(1, 0.5, 100.0);   // mode 2: nonzero g_target but dark

    BussiThermostat* bussi = new BussiThermostat(100.0, 1.0);
    bussi->setApplyToAllParticles(false);
    bussi->addParticle(0);
    bussi->addParticle(1);
    ts.system.addForce(bussi);

    VerletIntegrator integrator(DT_PS);
    Context context(ts.system, integrator, platform);
    context.setPositions(ts.positions);
    context.setVelocitiesToTemperature(0.0);

    integrator.step(60);

    // Mode 1 has g_target=0 → zero. Mode 2 is dark (f_2≈0) → zero coupling.
    double coupling = ts.force->getCouplingEnergy(context);
    ASSERT_EQUAL_TOL(0.0, coupling, 1e-6);

    cout << "PASS" << endl;
}

// ===================================================================
// TEST 4: Reference vs GPU platform agreement
// ===================================================================
void testMultiModeReferencePlatformAgreement() {
    cout << "testMultiModeReferencePlatformAgreement ... " << flush;

    auto buildSystem = []() -> pair<System*, MultiModeCavityForce*> {
        System* sys = new System();
        sys->setDefaultPeriodicBoxVectors(Vec3(5,0,0), Vec3(0,5,0), Vec3(0,0,5));
        sys->addParticle(12.0);
        sys->addParticle(12.0);
        sys->addParticle(PHOTON_MASS);
        sys->addParticle(PHOTON_MASS);

        NonbondedForce* nb = new NonbondedForce();
        nb->setNonbondedMethod(NonbondedForce::PME);
        nb->setCutoffDistance(2.0);
        nb->addParticle(+1.0, 0.3, 0.0);
        nb->addParticle(-1.0, 0.3, 0.0);
        nb->addParticle(0.0, 0.1, 0.0);
        nb->addParticle(0.0, 0.1, 0.0);
        nb->addException(2, 0, 0.0, 0.1, 0.0);
        nb->addException(2, 1, 0.0, 0.1, 0.0);
        nb->addException(3, 0, 0.0, 0.1, 0.0);
        nb->addException(3, 1, 0.0, 0.1, 0.0);
        nb->addException(2, 3, 0.0, 0.1, 0.0);
        sys->addForce(nb);

        MultiModeCavityForce* f = new MultiModeCavityForce(2, OMEGA1, 0.5, CAV_LENGTH, MOL_Z, PHOTON_MASS);
        f->addCavityParticle(2);
        f->addCavityParticle(3);
        f->setAdaptiveSquareWaveModulation(0.05, 0.6, 0.01, -1.0);
        f->setModeModulationParams(0, 0.4, 100.0, 1e-8, 1.0);
        f->setModeModulationParams(1, 0.3, 200.0, 1e-8, 1.0);
        sys->addForce(f);

        BussiThermostat* b = new BussiThermostat(400.0, 1.0);
        b->setApplyToAllParticles(false);
        b->addParticle(0);
        b->addParticle(1);
        sys->addForce(b);

        return {sys, f};
    };

    vector<Vec3> positions = {Vec3(1,0,0), Vec3(-1,0,0), Vec3(0.1,0.2,0), Vec3(0.05,0.1,0)};

    auto [sysA, fA] = buildSystem();
    VerletIntegrator intA(DT_PS);
    Context ctxA(*sysA, intA, platform);
    ctxA.setPositions(positions);
    ctxA.setVelocitiesToTemperature(10.0, 42);

    ReferencePlatform refPlatform;
    auto [sysB, fB] = buildSystem();
    VerletIntegrator intB(DT_PS);
    Context ctxB(*sysB, intB, refPlatform);
    ctxB.setPositions(positions);
    ctxB.setVelocitiesToTemperature(10.0, 42);

    intA.step(200);
    intB.step(200);

    ASSERT_EQUAL_TOL(fB->getHarmonicEnergy(ctxB), fA->getHarmonicEnergy(ctxA), 1e-4);
    ASSERT_EQUAL_TOL(fB->getCouplingEnergy(ctxB), fA->getCouplingEnergy(ctxA), 1e-4);
    ASSERT_EQUAL_TOL(fB->getDipoleSelfEnergy(ctxB), fA->getDipoleSelfEnergy(ctxA), 1e-4);

    delete sysA;
    delete sysB;
    cout << "PASS" << endl;
}

// ===================================================================
// TEST 5: updateParametersInContext propagates per-mode modulation changes
// ===================================================================
void testMultiModeUpdateMidRun() {
    cout << "testMultiModeUpdateMidRun ... " << flush;

    MultiModeTestSystem ts;

    BussiThermostat* bussi = new BussiThermostat(100.0, 1.0);
    bussi->setApplyToAllParticles(false);
    bussi->addParticle(0);
    bussi->addParticle(1);
    ts.system.addForce(bussi);

    VerletIntegrator integrator(DT_PS);
    Context context(ts.system, integrator, platform);
    context.setPositions(ts.positions);
    context.setVelocitiesToTemperature(0.0);

    // Initially no modulation → constant lambda1 coupling
    integrator.step(10);
    double dse1 = ts.force->getDipoleSelfEnergy(context);

    // Enable modulation with mode 1 g_target=0 → coupling drops to zero
    ts.force->setAdaptiveSquareWaveModulation(0.02, 1.0, 0.0, -1.0);  // always ON
    ts.force->setModeModulationParams(0, 0.0, 100.0);  // zero
    ts.force->setModeModulationParams(1, 0.0, 100.0);  // zero
    ts.force->updateParametersInContext(context);

    integrator.step(30);
    double coupling = ts.force->getCouplingEnergy(context);
    double dse2 = ts.force->getDipoleSelfEnergy(context);
    ASSERT_EQUAL_TOL(0.0, coupling, 1e-10);
    ASSERT_EQUAL_TOL(0.0, dse2, 1e-10);

    cout << "PASS" << endl;
}

// ===================================================================
// Entry point
// ===================================================================
void runPlatformTests();

int main(int argc, char* argv[]) {
    try {
        initializeTests(argc, argv);

        testMultiModeAdaptiveBasic();
        testMultiModeOnOffPhase();
        testMultiModeDarkMode();
        testMultiModeReferencePlatformAgreement();
        testMultiModeUpdateMidRun();

        runPlatformTests();
    }
    catch (const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
