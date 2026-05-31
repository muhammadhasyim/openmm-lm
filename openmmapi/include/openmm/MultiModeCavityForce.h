#ifndef OPENMM_MULTIMODECAVITYFORCE_H_
#define OPENMM_MULTIMODECAVITYFORCE_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org.                                        *
 *                                                                            *
 * Portions copyright (c) 2025 Stanford University and the Authors.           *
 * Authors: Muhammad Hasyim                                                   *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "Force.h"
#include "Vec3.h"
#include "internal/MathConstants.h"
#include "internal/windowsExport.h"
#include <string>
#include <vector>
#include <cmath>

namespace OpenMM {

/**
 * This class implements multi-mode Fabry-Perot cavity-molecule interaction for
 * cavity molecular dynamics simulations. The cavity supports N longitudinal modes,
 * each modeled as a fictitious particle (photon) that couples to the molecular
 * dipole moment.
 *
 * The multi-mode cavity Hamiltonian is:
 *
 *   H = sum_n [ p_n^2/(2*m) + (1/2)*K_n*q_n^2 + eps_n*f_n(z0)*q_n.d ]
 *       + (1/2) * ( sum_n eps_n^2/K_n * f_n(z0)^2 ) * d^2
 *
 * where for mode n (1-indexed):
 *   - omega_n = n * omega_1           (harmonically spaced frequencies)
 *   - lambda_n = lambda_1             (constant coupling; effective coupling eps_n = n * eps_1)
 *   - eps_n = lambda_n * omega_n      (effective coupling)
 *   - K_n = m * omega_n^2             (spring constant, same mass for all modes)
 *   - f_n(z0) = sin(n*pi*z0/L)       (spatial profile at molecule position z0)
 *   - d is the molecular dipole moment (x,y components only)
 *   - q_n is the position of the cavity particle for mode n
 *
 * The dipole self-energy (DSE) sums contributions from ALL modes and is applied
 * as a single term to avoid double-counting.
 *
 * IMPORTANT: The spatial profile f_n(z0) = sin(n*pi*z0/L) means that for
 * molecules at the cavity center (z0 = L/2), even modes (n=2,4,...) have
 * f_n = 0 and are completely dark (no coupling). Only odd modes couple.
 *
 * All position calculations use UNWRAPPED coordinates for correct dipole
 * moment calculations across periodic boundaries.
 *
 * Units:
 *   - omega_1: atomic units (Hartree)
 *   - lambda_1: dimensionless
 *   - photonMass: amu
 *   - cavityLength: nanometers
 *   - moleculeZ: nanometers
 */
class OPENMM_EXPORT MultiModeCavityForce : public Force {
public:
    /**
     * Create a MultiModeCavityForce.
     *
     * @param numModes       the number of cavity modes (N >= 1)
     * @param omega1         the fundamental cavity frequency in atomic units (Hartree)
     * @param lambda1        the dimensionless coupling for the fundamental mode
     * @param cavityLength   the cavity length in nanometers
     * @param moleculeZ      the z-position of molecules in the cavity (nm)
     * @param photonMass     the effective photon mass in amu (default: 1/1822.888)
     */
    MultiModeCavityForce(int numModes, double omega1, double lambda1,
                         double cavityLength, double moleculeZ,
                         double photonMass = 1.0/1822.888);
    /**
     * Get the number of cavity modes.
     */
    int getNumModes() const {
        return numModes;
    }
    /**
     * Get the fundamental cavity frequency omega_1 (in atomic units).
     */
    double getOmega1() const {
        return omega1;
    }
    /**
     * Set the fundamental cavity frequency omega_1.
     *
     * @param omega1  the fundamental frequency in atomic units (Hartree)
     */
    void setOmega1(double omega1);
    /**
     * Get the dimensionless coupling parameter lambda_1 for the fundamental mode.
     */
    double getLambda1() const {
        return lambda1;
    }
    /**
     * Set the dimensionless coupling parameter lambda_1.
     *
     * @param lambda1  the dimensionless coupling for the fundamental mode
     */
    void setLambda1(double lambda1);
    /**
     * Get the photon mass (in amu). Same for all modes.
     */
    double getPhotonMass() const {
        return photonMass;
    }
    /**
     * Set the photon mass.
     *
     * @param mass  the photon mass in amu
     */
    void setPhotonMass(double mass);
    /**
     * Set whether the dipole self-energy (self-polarization) term is included.
     * When false, the summed DSE energy and its force contribution via the
     * displaced coordinate are omitted for all modes.
     *
     * @param include  true to include DSE (default), false to omit it
     */
    void setIncludeDipoleSelfEnergy(bool include);
    /**
     * Get whether the dipole self-energy term is included.
     */
    bool getIncludeDipoleSelfEnergy() const;
    /**
     * Get the cavity length (in nm).
     */
    double getCavityLength() const {
        return cavityLength;
    }
    /**
     * Get the molecule z-position (in nm).
     */
    double getMoleculeZ() const {
        return moleculeZ;
    }
    /**
     * Add a cavity particle index for a specific mode. Call this N times,
     * once for each mode, in order (mode 1, mode 2, ..., mode N).
     *
     * @param particleIndex  the system particle index for this mode's photon
     * @return the mode index (0-based) that was added
     */
    int addCavityParticle(int particleIndex);
    /**
     * Get the particle index for a specific mode.
     *
     * @param modeIndex  the 0-based mode index
     * @return the system particle index
     */
    int getCavityParticleIndex(int modeIndex) const;
    /**
     * Get all cavity particle indices.
     */
    const std::vector<int>& getCavityParticleIndices() const {
        return cavityParticleIndices;
    }
    /**
     * Get the frequency of mode n (1-indexed): omega_n = n * omega_1.
     *
     * @param n  the mode number (1-indexed)
     * @return omega_n in atomic units
     */
    double getOmegaN(int n) const {
        return n * omega1;
    }
    /**
     * Get the coupling parameter of mode n (1-indexed): lambda_n = lambda_1 (constant).
     *
     * @param n  the mode number (1-indexed)
     * @return lambda_n (dimensionless)
     */
    double getLambdaN(int n) const {
        return lambda1;
    }
    /**
     * Get the effective coupling of mode n: eps_n = lambda_n * omega_n.
     *
     * @param n  the mode number (1-indexed)
     * @return eps_n in atomic units
     */
    double getEffectiveCouplingN(int n) const {
        return getLambdaN(n) * getOmegaN(n);
    }
    /**
     * Get the spring constant of mode n: K_n = photonMass * omega_n^2.
     *
     * @param n  the mode number (1-indexed)
     * @return K_n in atomic units
     */
    double getSpringConstantN(int n) const {
        double omega_n = getOmegaN(n);
        return photonMass * omega_n * omega_n;
    }
    /**
     * Get the spatial profile for mode n: f_n(z0) = sin(n * pi * z0 / L).
     *
     * @param n  the mode number (1-indexed)
     * @return f_n (dimensionless, between -1 and 1)
     */
    double getSpatialProfile(int n) const {
        return std::sin(n * OpenMM_Pi * moleculeZ / cavityLength);
    }
    /**
     * Get the precomputed spatial profiles for all modes.
     */
    const std::vector<double>& getSpatialProfiles() const {
        return spatialProfiles;
    }
    /**
     * Get the precomputed DSE prefactor: (1/2) * sum_n(eps_n^2 / K_n * f_n^2).
     * This is in atomic units and must be converted to OpenMM units by the kernel.
     */
    double getDSEPrefactor() const {
        return dsePrefactor;
    }
    /**
     * Get the total harmonic energy summed over all modes.
     * Must be called after forces have been computed.
     *
     * @param context  the Context to query
     * @return the total harmonic energy in kJ/mol
     */
    double getHarmonicEnergy(const Context& context) const;
    /**
     * Get the total coupling energy summed over all modes.
     * Must be called after forces have been computed.
     *
     * @param context  the Context to query
     * @return the total coupling energy in kJ/mol
     */
    double getCouplingEnergy(const Context& context) const;
    /**
     * Get the dipole self-energy (single term summed over all modes).
     * Must be called after forces have been computed.
     *
     * @param context  the Context to query
     * @return the dipole self-energy in kJ/mol
     */
    double getDipoleSelfEnergy(const Context& context) const;
    /**
     * Get the total cavity energy (sum of harmonic, coupling, and DSE).
     *
     * @param context  the Context to query
     * @return the total cavity energy in kJ/mol
     */
    double getTotalCavityEnergy(const Context& context) const;
    /**
     * Enable adaptive square-wave modulation with shared timing for all modes.
     * Each mode adapts its own amplitude via g_next = g_target_n * sqrt(T_target_n / T_bath).
     * Per-mode coupling parameters must be set via setModeModulationParams() for each mode.
     *
     * @param periodPs     square-wave period in ps (shared by all modes)
     * @param dutyCycle    fraction of period that is ON [0,1] (shared)
     * @param startTimePs  activation time in ps (shared)
     * @param stopTimePs   deactivation time in ps; -1 = never (shared)
     */
    void setAdaptiveSquareWaveModulation(double periodPs, double dutyCycle = 0.5,
                                         double startTimePs = 0.0, double stopTimePs = -1.0);
    /**
     * Set per-mode adaptive coupling parameters. Must be called after
     * setAdaptiveSquareWaveModulation() and for each mode index 0..numModes-1.
     *
     * @param modeIndex    0-based mode index
     * @param gTarget      target coupling g_target_n (dimensionless)
     * @param tTargetK     target temperature T_target_n in Kelvin
     * @param minAmplitude lower amplitude clamp
     * @param maxAmplitude upper amplitude clamp
     */
    void setModeModulationParams(int modeIndex, double gTarget, double tTargetK,
                                 double minAmplitude = 1e-8, double maxAmplitude = 0.1);
    bool isModulationEnabled() const { return modEnabled; }
    double getModulationPeriodPs() const { return modPeriodPs; }
    double getModulationDutyCycle() const { return modDutyCycle; }
    double getModulationStartTimePs() const { return modStartTimePs; }
    double getModulationStopTimePs() const { return modStopTimePs; }
    double getModeGTarget(int modeIndex) const { return modeGTargets[modeIndex]; }
    double getModeTTargetK(int modeIndex) const { return modeTTargets[modeIndex]; }
    double getModeMinAmplitude(int modeIndex) const { return modeMinAmps[modeIndex]; }
    double getModeMaxAmplitude(int modeIndex) const { return modeMaxAmps[modeIndex]; }
    /**
     * Update the parameters in a Context to match those stored in this Force object.
     *
     * @param context  the Context to update
     */
    void updateParametersInContext(Context& context);
    /**
     * Returns whether this force uses periodic boundary conditions.
     *
     * @return true (multi-mode cavity force uses PBC for unwrapped position calculations)
     */
    bool usesPeriodicBoundaryConditions() const {
        return true;
    }
protected:
    ForceImpl* createImpl() const;
private:
    int numModes;
    double omega1;
    double lambda1;
    double photonMass;
    double cavityLength;
    double moleculeZ;
    bool includeDipoleSelfEnergy;
    std::vector<int> cavityParticleIndices;
    std::vector<double> spatialProfiles;
    double dsePrefactor;
    // Per-mode adaptive square-wave modulation
    bool modEnabled;
    double modPeriodPs;
    double modDutyCycle;
    double modStartTimePs;
    double modStopTimePs;
    std::vector<double> modeGTargets;
    std::vector<double> modeTTargets;
    std::vector<double> modeMinAmps;
    std::vector<double> modeMaxAmps;
    /**
     * Recompute spatial profiles and DSE prefactor from current parameters.
     */
    void recomputeDerivedQuantities();
};

} // namespace OpenMM

#endif /*OPENMM_MULTIMODECAVITYFORCE_H_*/
