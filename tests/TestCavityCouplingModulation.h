/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * Tests for GPU-side coupling modulation in CavityForce.                     *
 *                                                                            *
 * Verifies that ModulationNone, ModulationStep, ModulationSquareWave, and    *
 * ModulationDecayingStep produce correct effectiveLambda values at known     *
 * time points, that Reference and GPU platforms agree numerically, and that  *
 * updateParametersInContext propagates new modulation parameters mid-run.    *
 * -------------------------------------------------------------------------- */

#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/CavityForce.h"
#include "openmm/NonbondedForce.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/BussiThermostat.h"
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

// ---------------------------------------------------------------------------
// Constants (must match cavityForce.cc and ReferenceKernels.cpp exactly)
// ---------------------------------------------------------------------------
static const double HARTREE_TO_KJMOL = 2625.4996;
static const double BOHR_TO_NM      = 0.052917721;
static const double AMU_TO_AU       = 1822.8885;
static const double CONV = HARTREE_TO_KJMOL / (BOHR_TO_NM * BOHR_TO_NM);

// Cavity parameters used by every test
static const double OMEGAC       = 0.01;   // a.u.
static const double PHOTON_MASS  = 1.0;    // amu
static const double DT_PS        = 0.001;  // 1 fs integrator timestep

// ---------------------------------------------------------------------------
// Helper: build a minimal 3-particle system (2 molecular + 1 cavity)
//
//   particle 0: charge +1, at ( 1, 0, 0)
//   particle 1: charge -1, at (-1, 0, 0)
//   particle 2: cavity,    at ( 0.1, 0.2, 0)   (off-origin so energies ≠ 0)
//
//   dipole = (+1)(1,0,0) + (-1)(-1,0,0) = (2, 0, 0)
// ---------------------------------------------------------------------------
struct TestSystem {
    System system;
    CavityForce* cavityForce;
    vector<Vec3> positions;
    int numMolecular;
    int cavityIndex;

    // Expected dipole (x,y) and derived constants
    double dipoleX, dipoleY;
    double K, epsilonFor; // spring constant, epsilon for a given lambda

    TestSystem(double lambda = 0.5) {
        system.setDefaultPeriodicBoxVectors(Vec3(5, 0, 0), Vec3(0, 5, 0), Vec3(0, 0, 5));

        // Molecular particles
        system.addParticle(12.0);
        system.addParticle(12.0);
        // Cavity particle
        cavityIndex = system.addParticle(PHOTON_MASS);
        numMolecular = 2;

        // NonbondedForce with charges
        NonbondedForce* nb = new NonbondedForce();
        nb->setNonbondedMethod(NonbondedForce::PME);
        nb->setCutoffDistance(2.0);
        nb->addParticle(+1.0, 0.3, 0.0);  // particle 0
        nb->addParticle(-1.0, 0.3, 0.0);  // particle 1
        nb->addParticle( 0.0, 0.1, 0.0);  // cavity: no charge, no LJ
        // Exclude cavity from all nonbonded
        nb->addException(2, 0, 0.0, 0.1, 0.0);
        nb->addException(2, 1, 0.0, 0.1, 0.0);
        system.addForce(nb);

        // CavityForce (starts with given lambda)
        cavityForce = new CavityForce(cavityIndex, OMEGAC, lambda, PHOTON_MASS);
        system.addForce(cavityForce);

        // Positions
        positions.push_back(Vec3( 1.0, 0.0, 0.0));
        positions.push_back(Vec3(-1.0, 0.0, 0.0));
        positions.push_back(Vec3( 0.1, 0.2, 0.0));  // cavity off-origin

        // Precompute expected values
        dipoleX = (+1.0) * 1.0 + (-1.0) * (-1.0);  // = 2.0
        dipoleY = 0.0;

        double mass_au = PHOTON_MASS * AMU_TO_AU;
        K = mass_au * OMEGAC * OMEGAC * CONV;
        epsilonFor = lambda * OMEGAC * CONV;
    }
};

// ---------------------------------------------------------------------------
// Compute expected total cavity energy for a given lambda and cavity position
// ---------------------------------------------------------------------------
static double expectedCavityEnergy(double lambda, double qx, double qy, double qz,
                                   double dx, double dy) {
    double mass_au = PHOTON_MASS * AMU_TO_AU;
    double K = mass_au * OMEGAC * OMEGAC * CONV;
    double eps = lambda * OMEGAC * CONV;

    double harmonic  = 0.5 * K * (qx*qx + qy*qy + qz*qz);
    double coupling  = eps * (qx*dx + qy*dy);
    double dse       = 0.5 * eps * eps / K * (dx*dx + dy*dy);
    return harmonic + coupling + dse;
}

// ===================================================================
//  TEST 1: ModulationNone — coupling is the constant lambdaCoupling
// ===================================================================
void testModulationNone() {
    cout << "testModulationNone ... " << flush;
    double lambda = 0.3;
    TestSystem ts(lambda);

    // Explicitly set ModulationNone (default)
    ts.cavityForce->setCouplingModulation(CavityForce::ModulationNone,
                                          0.0, 0.0, 0.5, 0.0, -1.0, 1.0);

    VerletIntegrator integrator(DT_PS);
    Context context(ts.system, integrator, platform);
    context.setPositions(ts.positions);

    // Energy should match constant lambda = 0.3
    State state = context.getState(State::Energy);
    double harmonic = ts.cavityForce->getHarmonicEnergy(context);
    double coupling = ts.cavityForce->getCouplingEnergy(context);
    double dse      = ts.cavityForce->getDipoleSelfEnergy(context);

    double expected = expectedCavityEnergy(lambda, 0.1, 0.2, 0.0, ts.dipoleX, ts.dipoleY);
    double actual   = harmonic + coupling + dse;

    ASSERT_EQUAL_TOL(expected, actual, 1e-5);
    cout << "PASS" << endl;
}

// ===================================================================
//  TEST 2: ModulationStep — zero before startTime, amplitude after
// ===================================================================
void testModulationStep() {
    cout << "testModulationStep ... " << flush;
    double amplitude = 0.4;
    double startTime = 0.05;  // 50 fs

    TestSystem ts(0.0);  // base lambda = 0 (modulation overrides)
    ts.cavityForce->setCouplingModulation(CavityForce::ModulationStep,
                                          amplitude, 0.0, 0.5,
                                          startTime, -1.0, 1.0);

    VerletIntegrator integrator(DT_PS);
    Context context(ts.system, integrator, platform);
    context.setPositions(ts.positions);
    context.setVelocitiesToTemperature(0.0);  // freeze particles

    // At t=0 (before startTime): coupling should be off
    {
        State state = context.getState(State::Energy | State::Positions);
        double coupling = ts.cavityForce->getCouplingEnergy(context);
        double dse      = ts.cavityForce->getDipoleSelfEnergy(context);
        // With lambda=0: coupling and DSE should be zero
        ASSERT_EQUAL_TOL(0.0, coupling, 1e-10);
        ASSERT_EQUAL_TOL(0.0, dse, 1e-10);
    }

    // Step past startTime
    int stepsToPass = (int)(startTime / DT_PS) + 10;
    integrator.step(stepsToPass);

    // After startTime: coupling should use amplitude
    {
        State state = context.getState(State::Energy | State::Positions);
        Vec3 cavPos = state.getPositions()[ts.cavityIndex];
        double coupling = ts.cavityForce->getCouplingEnergy(context);

        // Coupling energy = epsilon * (qx*dx + qy*dy)
        // epsilon = amplitude * omegac * CONV
        double eps = amplitude * OMEGAC * CONV;
        double expectedCoupling = eps * (cavPos[0] * ts.dipoleX + cavPos[1] * ts.dipoleY);
        // Particles may have moved slightly, so use moderate tolerance
        ASSERT_EQUAL_TOL(expectedCoupling, coupling, 0.05);
    }

    cout << "PASS" << endl;
}

// ===================================================================
//  TEST 3: ModulationSquareWave — alternates between amplitude and 0
// ===================================================================
void testModulationSquareWave() {
    cout << "testModulationSquareWave ... " << flush;
    double amplitude = 0.5;
    double period    = 0.1;    // 100 fs period
    double dutyCycle = 0.5;    // 50% on
    double startTime = 0.0;

    TestSystem ts(0.0);
    ts.cavityForce->setCouplingModulation(CavityForce::ModulationSquareWave,
                                          amplitude, period, dutyCycle,
                                          startTime, -1.0, 1.0);

    VerletIntegrator integrator(DT_PS);
    Context context(ts.system, integrator, platform);
    context.setPositions(ts.positions);
    context.setVelocitiesToTemperature(0.0);

    // Step to t = 0.01 ps (10% into first period, within ON phase)
    integrator.step(10);
    {
        State state = context.getState(State::Energy | State::Positions);
        double coupling = ts.cavityForce->getCouplingEnergy(context);
        double dse      = ts.cavityForce->getDipoleSelfEnergy(context);
        // ON phase: coupling and DSE should be nonzero
        ASSERT(fabs(coupling) > 1e-10 || fabs(dse) > 1e-10);
    }

    // Step to t = 0.075 ps (75% into period, within OFF phase for 50% duty)
    integrator.step(65);
    {
        State state = context.getState(State::Energy | State::Positions);
        double coupling = ts.cavityForce->getCouplingEnergy(context);
        double dse      = ts.cavityForce->getDipoleSelfEnergy(context);
        // OFF phase: coupling and DSE should be zero (lambda=0)
        ASSERT_EQUAL_TOL(0.0, coupling, 1e-10);
        ASSERT_EQUAL_TOL(0.0, dse, 1e-10);
    }

    // Step to t = 0.11 ps (10% into second period, ON phase again)
    integrator.step(35);
    {
        State state = context.getState(State::Energy | State::Positions);
        double coupling = ts.cavityForce->getCouplingEnergy(context);
        double dse      = ts.cavityForce->getDipoleSelfEnergy(context);
        // ON phase again
        ASSERT(fabs(coupling) > 1e-10 || fabs(dse) > 1e-10);
    }

    cout << "PASS" << endl;
}

// ===================================================================
//  TEST 4: ModulationSquareWave duty cycle — verify ON duration
// ===================================================================
void testSquareWaveDutyCycle() {
    cout << "testSquareWaveDutyCycle ... " << flush;
    double amplitude = 0.5;
    double period    = 0.1;
    double dutyCycle = 0.25;  // 25% ON
    double startTime = 0.0;

    TestSystem ts(0.0);
    ts.cavityForce->setCouplingModulation(CavityForce::ModulationSquareWave,
                                          amplitude, period, dutyCycle,
                                          startTime, -1.0, 1.0);

    VerletIntegrator integrator(DT_PS);
    Context context(ts.system, integrator, platform);
    context.setPositions(ts.positions);
    context.setVelocitiesToTemperature(0.0);

    // t = 0.01 ps: 10% into period → ON (< 25%)
    integrator.step(10);
    {
        double dse = ts.cavityForce->getDipoleSelfEnergy(context);
        ASSERT(dse > 1e-10);  // ON
    }

    // t = 0.04 ps: 40% into period → OFF (> 25%)
    integrator.step(30);
    {
        double coupling = ts.cavityForce->getCouplingEnergy(context);
        double dse      = ts.cavityForce->getDipoleSelfEnergy(context);
        ASSERT_EQUAL_TOL(0.0, coupling, 1e-10);
        ASSERT_EQUAL_TOL(0.0, dse, 1e-10);
    }

    cout << "PASS" << endl;
}

// ===================================================================
//  TEST 5: ModulationDecayingStep — exponential decay after start
// ===================================================================
void testModulationDecayingStep() {
    cout << "testModulationDecayingStep ... " << flush;
    double amplitude  = 0.5;
    double startTime  = 0.0;
    double decayTau   = 0.1;  // 100 fs decay time

    TestSystem ts(0.0);
    ts.cavityForce->setCouplingModulation(CavityForce::ModulationDecayingStep,
                                          amplitude, 0.0, 0.5,
                                          startTime, -1.0, decayTau);

    VerletIntegrator integrator(DT_PS);
    Context context(ts.system, integrator, platform);
    context.setPositions(ts.positions);
    context.setVelocitiesToTemperature(0.0);

    // At t=0: lambda = amplitude (decay factor = 1)
    {
        State state = context.getState(State::Energy);
        double dse0 = ts.cavityForce->getDipoleSelfEnergy(context);
        // DSE ∝ lambda^2.  At t=0, lambda = amplitude.
        double eps0 = amplitude * OMEGAC * CONV;
        double expectedDSE0 = 0.5 * eps0 * eps0 / ts.K * (ts.dipoleX * ts.dipoleX);
        ASSERT_EQUAL_TOL(expectedDSE0, dse0, 1e-4);
    }

    // At t = 2*tau = 0.2 ps: lambda = amplitude * exp(-2) ≈ 0.0677
    integrator.step(200);
    {
        State state = context.getState(State::Energy | State::Positions);
        double dse200 = ts.cavityForce->getDipoleSelfEnergy(context);

        // Particles have moved, so get current dipole & cavity pos
        Vec3 cavPos = state.getPositions()[ts.cavityIndex];
        // Recompute dipole from current positions
        vector<Vec3> pos = state.getPositions();
        double dx = (+1.0) * pos[0][0] + (-1.0) * pos[1][0];
        double dy = (+1.0) * pos[0][1] + (-1.0) * pos[1][1];

        double time_ps = 0.2;
        double decayedLambda = amplitude * exp(-time_ps / decayTau);
        double eps = decayedLambda * OMEGAC * CONV;
        double expectedDSE = 0.5 * eps * eps / ts.K * (dx * dx + dy * dy);
        ASSERT_EQUAL_TOL(expectedDSE, dse200, 0.1);  // particles moved → looser tol
    }

    cout << "PASS" << endl;
}

// ===================================================================
//  TEST 6: ModulationSquareWave stopTime — coupling off after stop
// ===================================================================
void testSquareWaveStopTime() {
    cout << "testSquareWaveStopTime ... " << flush;
    double amplitude = 0.5;
    double period    = 0.05;
    double dutyCycle = 0.5;
    double startTime = 0.0;
    double stopTime  = 0.1;  // 100 fs

    TestSystem ts(0.0);
    ts.cavityForce->setCouplingModulation(CavityForce::ModulationSquareWave,
                                          amplitude, period, dutyCycle,
                                          startTime, stopTime, 1.0);

    VerletIntegrator integrator(DT_PS);
    Context context(ts.system, integrator, platform);
    context.setPositions(ts.positions);
    context.setVelocitiesToTemperature(0.0);

    // Step well past stopTime
    integrator.step(200);  // t = 0.2 ps > stopTime = 0.1 ps

    double coupling = ts.cavityForce->getCouplingEnergy(context);
    double dse      = ts.cavityForce->getDipoleSelfEnergy(context);
    ASSERT_EQUAL_TOL(0.0, coupling, 1e-10);
    ASSERT_EQUAL_TOL(0.0, dse, 1e-10);

    cout << "PASS" << endl;
}

// ===================================================================
//  TEST 7: ModulationStep with startTime > 0 — verify late activation
// ===================================================================
void testStepLateActivation() {
    cout << "testStepLateActivation ... " << flush;
    double amplitude = 0.4;
    double startTime = 0.1;  // 100 fs

    TestSystem ts(0.0);
    ts.cavityForce->setCouplingModulation(CavityForce::ModulationStep,
                                          amplitude, 0.0, 0.5,
                                          startTime, -1.0, 1.0);

    VerletIntegrator integrator(DT_PS);
    Context context(ts.system, integrator, platform);
    context.setPositions(ts.positions);
    context.setVelocitiesToTemperature(0.0);

    // Before startTime: all coupling energy should be zero
    integrator.step(50);  // t = 0.05 ps < 0.1 ps
    {
        double coupling = ts.cavityForce->getCouplingEnergy(context);
        double dse      = ts.cavityForce->getDipoleSelfEnergy(context);
        ASSERT_EQUAL_TOL(0.0, coupling, 1e-10);
        ASSERT_EQUAL_TOL(0.0, dse, 1e-10);
    }

    // After startTime: energy should be nonzero
    integrator.step(60);  // t = 0.11 ps > 0.1 ps
    {
        double dse = ts.cavityForce->getDipoleSelfEnergy(context);
        ASSERT(dse > 1e-10);
    }

    cout << "PASS" << endl;
}

// ===================================================================
//  TEST 8: updateParametersInContext propagates modulation changes
// ===================================================================
void testUpdateModulationMidRun() {
    cout << "testUpdateModulationMidRun ... " << flush;

    // Start with no modulation, constant lambda = 0.3
    TestSystem ts(0.3);

    VerletIntegrator integrator(DT_PS);
    Context context(ts.system, integrator, platform);
    context.setPositions(ts.positions);
    context.setVelocitiesToTemperature(0.0);

    // Step 1: constant coupling, verify nonzero energy
    {
        State state = context.getState(State::Energy);
        double dse = ts.cavityForce->getDipoleSelfEnergy(context);
        ASSERT(dse > 1e-10);
    }

    // Step 2: switch to square-wave modulation, run to OFF phase
    ts.cavityForce->setCouplingModulation(CavityForce::ModulationSquareWave,
                                          0.3, 0.02, 0.5,  // period=20fs, 50% duty
                                          0.0, -1.0, 1.0);
    ts.cavityForce->updateParametersInContext(context);

    // Run to 75% of the first period (OFF phase)
    // Current time after previous step might not be zero, so step enough
    // to land in a definite OFF window
    integrator.step(15);  // should be ~15 fs into 20 fs period = 75% → OFF

    {
        double coupling = ts.cavityForce->getCouplingEnergy(context);
        double dse      = ts.cavityForce->getDipoleSelfEnergy(context);
        ASSERT_EQUAL_TOL(0.0, coupling, 1e-10);
        ASSERT_EQUAL_TOL(0.0, dse, 1e-10);
    }

    // Step 3: switch back to ModulationNone, constant lambda = 0.3
    ts.cavityForce->setCouplingModulation(CavityForce::ModulationNone,
                                          0.0, 0.0, 0.5, 0.0, -1.0, 1.0);
    ts.cavityForce->updateParametersInContext(context);

    {
        State state = context.getState(State::Energy);
        double dse = ts.cavityForce->getDipoleSelfEnergy(context);
        ASSERT(dse > 1e-10);
    }

    cout << "PASS" << endl;
}

// ===================================================================
//  TEST 9: Reference vs GPU platform agreement
//
//  This test creates the same system twice (once on each platform),
//  steps both for the same number of steps with identical initial
//  conditions, and asserts that the energy components match.
//
//  Note: this test only runs if a second platform is available.
//  When run from TestReferenceCavityModulation.cpp, it uses Reference
//  for both (self-consistency check).  From TestCudaCavityModulation.cpp
//  it compares CUDA against Reference.
// ===================================================================
void testReferencePlatformAgreement() {
    cout << "testReferencePlatformAgreement ... " << flush;

    double amplitude = 0.4;
    double period    = 0.05;
    double dutyCycle = 0.6;
    double startTime = 0.01;

    // Build two identical systems
    auto buildSystem = [&]() -> pair<System*, CavityForce*> {
        System* sys = new System();
        sys->setDefaultPeriodicBoxVectors(Vec3(5,0,0), Vec3(0,5,0), Vec3(0,0,5));
        sys->addParticle(12.0);
        sys->addParticle(12.0);
        sys->addParticle(PHOTON_MASS);

        NonbondedForce* nb = new NonbondedForce();
        nb->setNonbondedMethod(NonbondedForce::PME);
        nb->setCutoffDistance(2.0);
        nb->addParticle(+1.0, 0.3, 0.0);
        nb->addParticle(-1.0, 0.3, 0.0);
        nb->addParticle( 0.0, 0.1, 0.0);
        nb->addException(2, 0, 0.0, 0.1, 0.0);
        nb->addException(2, 1, 0.0, 0.1, 0.0);
        sys->addForce(nb);

        CavityForce* cf = new CavityForce(2, OMEGAC, 0.0, PHOTON_MASS);
        cf->setCouplingModulation(CavityForce::ModulationSquareWave,
                                  amplitude, period, dutyCycle,
                                  startTime, -1.0, 1.0);
        sys->addForce(cf);
        return {sys, cf};
    };

    vector<Vec3> positions = {Vec3(1, 0, 0), Vec3(-1, 0, 0), Vec3(0.1, 0.2, 0)};

    // System A: on the test platform (CUDA or Reference depending on test binary)
    auto [sysA, cfA] = buildSystem();
    VerletIntegrator intA(DT_PS);
    Context ctxA(*sysA, intA, platform);
    ctxA.setPositions(positions);
    ctxA.setVelocitiesToTemperature(10.0, 42);

    // System B: on Reference platform
    ReferencePlatform refPlatform;
    auto [sysB, cfB] = buildSystem();
    VerletIntegrator intB(DT_PS);
    Context ctxB(*sysB, intB, refPlatform);
    ctxB.setPositions(positions);
    ctxB.setVelocitiesToTemperature(10.0, 42);

    // Run both for 150 steps (through multiple square-wave cycles)
    int nSteps = 150;
    intA.step(nSteps);
    intB.step(nSteps);

    // Compare energy components
    double harmA = cfA->getHarmonicEnergy(ctxA);
    double coupA = cfA->getCouplingEnergy(ctxA);
    double dseA  = cfA->getDipoleSelfEnergy(ctxA);

    double harmB = cfB->getHarmonicEnergy(ctxB);
    double coupB = cfB->getCouplingEnergy(ctxB);
    double dseB  = cfB->getDipoleSelfEnergy(ctxB);

    // Tolerance: 1e-4 for mixed precision GPU vs double-precision Reference
    ASSERT_EQUAL_TOL(harmB, harmA, 1e-4);
    ASSERT_EQUAL_TOL(coupB, coupA, 1e-4);
    ASSERT_EQUAL_TOL(dseB,  dseA,  1e-4);

    // Also compare positions to verify dynamics agreement
    State stA = ctxA.getState(State::Positions);
    State stB = ctxB.getState(State::Positions);
    for (int i = 0; i < 3; i++) {
        ASSERT_EQUAL_VEC(stB.getPositions()[i], stA.getPositions()[i], 1e-3);
    }

    delete sysA;
    delete sysB;
    cout << "PASS" << endl;
}

// ===================================================================
//  TEST 10: Modulation overrides the old step-function schedule
// ===================================================================
void testModulationOverridesSchedule() {
    cout << "testModulationOverridesSchedule ... " << flush;

    // Set a legacy coupling schedule that would turn on at step 0
    TestSystem ts(0.5);

    // Also set a modulation that forces lambda = 0 for all time
    ts.cavityForce->setCouplingModulation(CavityForce::ModulationStep,
                                          0.0,   // amplitude = 0
                                          0.0, 0.5, 0.0, -1.0, 1.0);

    VerletIntegrator integrator(DT_PS);
    Context context(ts.system, integrator, platform);
    context.setPositions(ts.positions);
    context.setVelocitiesToTemperature(0.0);

    // With modulation active (amplitude=0), coupling energy should be zero
    // even though base lambdaCoupling is 0.5
    integrator.step(10);
    double coupling = ts.cavityForce->getCouplingEnergy(context);
    double dse      = ts.cavityForce->getDipoleSelfEnergy(context);
    ASSERT_EQUAL_TOL(0.0, coupling, 1e-10);
    ASSERT_EQUAL_TOL(0.0, dse, 1e-10);

    cout << "PASS" << endl;
}

// ===================================================================
//  TEST 11: Energy conservation with square-wave modulation
//
//  During constant-lambda phases (ON or OFF), total energy should be
//  conserved by VerletIntegrator.  We check within a single ON phase
//  (no switching) that KE + PE drift is small.
// ===================================================================
void testEnergyConservationDuringOnPhase() {
    cout << "testEnergyConservationDuringOnPhase ... " << flush;
    double amplitude = 0.3;
    double period    = 1.0;     // long period so we stay in one phase
    double dutyCycle = 1.0;     // always ON
    double startTime = 0.0;

    TestSystem ts(0.0);
    ts.cavityForce->setCouplingModulation(CavityForce::ModulationSquareWave,
                                          amplitude, period, dutyCycle,
                                          startTime, -1.0, 1.0);

    VerletIntegrator integrator(DT_PS);
    Context context(ts.system, integrator, platform);
    context.setPositions(ts.positions);
    context.setVelocitiesToTemperature(100.0, 42);

    // Get initial total energy (C++ API returns double in kJ/mol)
    integrator.step(1);
    State s0 = context.getState(State::Energy);
    double E0 = s0.getPotentialEnergy() + s0.getKineticEnergy();

    // Run 1000 steps (1 ps) and check drift
    integrator.step(1000);
    State s1 = context.getState(State::Energy);
    double E1 = s1.getPotentialEnergy() + s1.getKineticEnergy();

    double drift = fabs(E1 - E0) / max(fabs(E0), 1.0);
    ASSERT(drift < 0.01);  // < 1% energy drift over 1 ps

    cout << "PASS (drift = " << drift << ")" << endl;
}

// ===================================================================
//  TEST 12: Validation — invalid parameters throw exceptions
// ===================================================================
void testInvalidModulationParameters() {
    cout << "testInvalidModulationParameters ... " << flush;

    TestSystem ts(0.0);

    // Square wave with period <= 0 should throw
    bool threw = false;
    try {
        ts.cavityForce->setCouplingModulation(CavityForce::ModulationSquareWave,
                                              0.5, 0.0, 0.5, 0.0, -1.0, 1.0);
    } catch (const OpenMMException&) {
        threw = true;
    }
    ASSERT(threw);

    // Square wave with duty cycle > 1 should throw
    threw = false;
    try {
        ts.cavityForce->setCouplingModulation(CavityForce::ModulationSquareWave,
                                              0.5, 1.0, 1.5, 0.0, -1.0, 1.0);
    } catch (const OpenMMException&) {
        threw = true;
    }
    ASSERT(threw);

    // Decaying step with tau <= 0 should throw
    threw = false;
    try {
        ts.cavityForce->setCouplingModulation(CavityForce::ModulationDecayingStep,
                                              0.5, 0.0, 0.5, 0.0, -1.0, 0.0);
    } catch (const OpenMMException&) {
        threw = true;
    }
    ASSERT(threw);

    cout << "PASS" << endl;
}

// ===================================================================
//  TEST 13: Adaptive square wave — amplitude adapts to T_bath
// ===================================================================
void testAdaptiveSquareWave() {
    cout << "testAdaptiveSquareWave ... " << flush;

    // Build system with Bussi thermostat so "BussiTemperature" parameter exists
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
    nb->addParticle( 0.0, 0.1, 0.0);
    nb->addException(2, 0, 0.0, 0.1, 0.0);
    nb->addException(2, 1, 0.0, 0.1, 0.0);
    system.addForce(nb);

    double targetCoupling = 0.5;
    double targetTempK = 100.0;
    double periodPs = 0.05;  // 50 fs period
    double dutyCycle = 0.5;

    CavityForce* cf = new CavityForce(2, OMEGAC, 0.0, PHOTON_MASS);
    cf->setAdaptiveSquareWaveModulation(targetCoupling, targetTempK,
                                         periodPs, dutyCycle, 0.0, -1.0,
                                         1e-8, 1.0);
    system.addForce(cf);

    // Add Bussi thermostat at T_bath = 400 K (different from T_target = 100 K)
    BussiThermostat* bussi = new BussiThermostat(400.0, 1.0);
    bussi->setApplyToAllParticles(false);
    bussi->addParticle(0);
    bussi->addParticle(1);
    system.addForce(bussi);

    vector<Vec3> positions = {Vec3(1,0,0), Vec3(-1,0,0), Vec3(0.1, 0.2, 0)};
    VerletIntegrator integrator(DT_PS);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setVelocitiesToTemperature(400.0, 42);

    // Run past one full period so the adaptive update triggers
    int stepsPerPeriod = (int)(periodPs / DT_PS);
    integrator.step(stepsPerPeriod + 10);

    // Expected adaptive amplitude: g_target * sqrt(T_target / T_bath)
    // = 0.5 * sqrt(100 / 400) = 0.5 * 0.5 = 0.25
    // The DSE ∝ lambda^2.  Check it's consistent with adapted amplitude, not original.
    // At this point we're in the ON phase of second period.
    // Step to be safely in ON phase of second period.
    double time_now = context.getState(State::Positions).getTime();

    // Get DSE and verify it reflects adapted amplitude, not original
    double dse = cf->getDipoleSelfEnergy(context);

    // DSE with original amplitude (0.5) would be much larger than with adapted (0.25)
    // Compute expected DSE with adapted amplitude
    State st = context.getState(State::Positions);
    vector<Vec3> pos = st.getPositions();
    double dx = (+1.0)*pos[0][0] + (-1.0)*pos[1][0];
    double dy = (+1.0)*pos[0][1] + (-1.0)*pos[1][1];

    double expectedLambda = targetCoupling * sqrt(targetTempK / 400.0);  // 0.25
    double eps = expectedLambda * OMEGAC * CONV;
    double mass_au = PHOTON_MASS * AMU_TO_AU;
    double K = mass_au * OMEGAC * OMEGAC * CONV;
    double expectedDSE = 0.5 * eps * eps / K * (dx*dx + dy*dy);

    // Tolerance is moderate because particles have moved under forces
    ASSERT_EQUAL_TOL(expectedDSE, dse, 0.2);

    cout << "PASS" << endl;
}

// ===================================================================
//  TEST 14: Adaptive amplitude respects min/max clamps
// ===================================================================
void testAdaptiveAmplitudeClamp() {
    cout << "testAdaptiveAmplitudeClamp ... " << flush;

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
    nb->addParticle( 0.0, 0.1, 0.0);
    nb->addException(2, 0, 0.0, 0.1, 0.0);
    nb->addException(2, 1, 0.0, 0.1, 0.0);
    system.addForce(nb);

    // T_target >> T_bath → amplitude wants to be very large → should clamp at maxAmplitude
    double maxAmp = 0.05;
    CavityForce* cf = new CavityForce(2, OMEGAC, 0.0, PHOTON_MASS);
    cf->setAdaptiveSquareWaveModulation(0.5, 10000.0,  // T_target = 10000 K
                                         0.05, 0.5, 0.0, -1.0,
                                         1e-8, maxAmp);
    system.addForce(cf);

    BussiThermostat* bussi = new BussiThermostat(100.0, 1.0);
    bussi->setApplyToAllParticles(false);
    bussi->addParticle(0);
    bussi->addParticle(1);
    system.addForce(bussi);

    vector<Vec3> positions = {Vec3(1,0,0), Vec3(-1,0,0), Vec3(0.1, 0.2, 0)};
    VerletIntegrator integrator(DT_PS);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setVelocitiesToTemperature(100.0, 42);

    // Run past two periods
    integrator.step(120);

    // DSE should reflect maxAmplitude, not the unclamped value
    State st = context.getState(State::Positions);
    vector<Vec3> pos = st.getPositions();
    double dx = (+1.0)*pos[0][0] + (-1.0)*pos[1][0];
    double dy = (+1.0)*pos[0][1] + (-1.0)*pos[1][1];

    double eps = maxAmp * OMEGAC * CONV;
    double mass_au = PHOTON_MASS * AMU_TO_AU;
    double K = mass_au * OMEGAC * OMEGAC * CONV;
    double expectedDSE = 0.5 * eps * eps / K * (dx*dx + dy*dy);

    double dse = cf->getDipoleSelfEnergy(context);
    ASSERT_EQUAL_TOL(expectedDSE, dse, 0.2);

    cout << "PASS" << endl;
}

// ===================================================================
//  TEST 15: Adaptive responds to mid-run BussiTemperature change
// ===================================================================
void testAdaptiveWithTemperatureChange() {
    cout << "testAdaptiveWithTemperatureChange ... " << flush;

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
    nb->addParticle( 0.0, 0.1, 0.0);
    nb->addException(2, 0, 0.0, 0.1, 0.0);
    nb->addException(2, 1, 0.0, 0.1, 0.0);
    system.addForce(nb);

    double targetCoupling = 0.4;
    double targetTempK = 100.0;
    double periodPs = 0.05;

    CavityForce* cf = new CavityForce(2, OMEGAC, 0.0, PHOTON_MASS);
    cf->setAdaptiveSquareWaveModulation(targetCoupling, targetTempK,
                                         periodPs, 0.5, 0.0, -1.0,
                                         1e-8, 1.0);
    system.addForce(cf);

    BussiThermostat* bussi = new BussiThermostat(100.0, 1.0);
    bussi->setApplyToAllParticles(false);
    bussi->addParticle(0);
    bussi->addParticle(1);
    system.addForce(bussi);

    vector<Vec3> positions = {Vec3(1,0,0), Vec3(-1,0,0), Vec3(0.1, 0.2, 0)};
    VerletIntegrator integrator(DT_PS);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setVelocitiesToTemperature(100.0, 42);

    // Run 2 periods at T_bath = 100 K → adapted amplitude = 0.4 * sqrt(100/100) = 0.4
    integrator.step(110);
    double dse1 = cf->getDipoleSelfEnergy(context);

    // Change T_bath to 400 K → next period should adapt to 0.4 * sqrt(100/400) = 0.2
    context.setParameter(BussiThermostat::Temperature(), 400.0);

    // Run 2 more periods
    integrator.step(110);
    double dse2 = cf->getDipoleSelfEnergy(context);

    // DSE ∝ lambda^2.  After temperature change, amplitude halved → DSE should be ~4x smaller.
    // Due to particle motion, use a rough check: dse2 should be significantly less than dse1.
    // (Both should be > 0 since we're checking during ON phase.)
    // The ratio of adapted amplitudes is 0.2/0.4 = 0.5, so DSE ratio ~ 0.25
    // Allow generous tolerance due to dynamics
    if (dse1 > 1e-10) {
        double ratio = dse2 / dse1;
        ASSERT(ratio < 0.8);  // adapted amplitude decreased substantially
    }

    cout << "PASS" << endl;
}

// ===================================================================
//  Entry point
// ===================================================================
void runPlatformTests();

int main(int argc, char* argv[]) {
    try {
        initializeTests(argc, argv);

        testModulationNone();
        testModulationStep();
        testModulationSquareWave();
        testSquareWaveDutyCycle();
        testModulationDecayingStep();
        testSquareWaveStopTime();
        testStepLateActivation();
        testUpdateModulationMidRun();
        testReferencePlatformAgreement();
        testModulationOverridesSchedule();
        testEnergyConservationDuringOnPhase();
        testInvalidModulationParameters();
        testAdaptiveSquareWave();
        testAdaptiveAmplitudeClamp();
        testAdaptiveWithTemperatureChange();

        runPlatformTests();
    }
    catch (const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
