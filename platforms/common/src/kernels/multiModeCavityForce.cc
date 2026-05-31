/**
 * Multi-Mode Fabry-Perot Cavity Force Kernel
 *
 * Implements the Hamiltonian:
 *   H = sum_n [ (1/2)*K_n*q_n^2 + eps_n*f_n(z0)*q_n.d ]
 *       + (1/2) * ( sum_n eps_n^2/K_n * f_n(z0)^2 ) * d^2
 *
 * where for mode n (1-indexed):
 *   omega_n = n * omega_1
 *   lambda_n = lambda_1 (constant)
 *   eps_n = lambda_n * omega_n = n * eps_1
 *   K_n = m * omega_n^2 = n^2 * K_1
 *   f_n(z0) = sin(n * pi * z0 / L)
 *
 * The molecular dipole is computed ONCE and reused for all N modes.
 * The DSE prefactor is precomputed on the host.
 *
 * Per-mode parameters are packed into a float4 array (modeParams):
 *   modeParams[n].x = K_n       (spring constant in OpenMM units)
 *   modeParams[n].y = eps_n     (effective coupling in OpenMM units)
 *   modeParams[n].z = f_n       (spatial profile, dimensionless)
 *   modeParams[n].w = (float) reordered cavity particle index for mode n
 */

/**
 * Clear the dipole and energy buffers.
 */
KERNEL void clearMultiModeBuffers(GLOBAL float* RESTRICT dipole,
        GLOBAL float* RESTRICT energyBuffer) {
    if (GLOBAL_ID == 0) {
        dipole[0] = 0.0f;
        dipole[1] = 0.0f;
        dipole[2] = 0.0f;
        dipole[3] = 0.0f;  // padding
        energyBuffer[0] = 0.0f;  // harmonic total
        energyBuffer[1] = 0.0f;  // coupling total
        energyBuffer[2] = 0.0f;  // DSE total
    }
}

/**
 * Update per-mode adaptive amplitudes on the GPU.
 * Runs as 1 thread before the force kernel. Checks whether a new square-wave
 * period started; if so, updates each mode's amplitude independently:
 *   g_next_n = g_target_n * sqrt(T_target_n / max(T_bath, 1))
 *
 * adaptiveState: float2[NUM_MODES], .x = currentAmplitude, .y = lastUpdatedPeriod
 * modParams:     float4[NUM_MODES], .x = g_target, .y = T_target, .z = minAmp, .w = maxAmp
 */
KERNEL void updateMultiModeAdaptiveAmplitudes(
        GLOBAL float2* RESTRICT adaptiveState,
        GLOBAL const float4* RESTRICT modModParams,
        float time_ps, float periodPs, float startTimePs, float stopTimePs,
        float bathTemperatureK) {
    if (GLOBAL_ID != 0) return;

    if (time_ps < startTimePs) {
        for (int n = 0; n < NUM_MODES; n++)
            adaptiveState[n].x = modModParams[n].x;
        return;
    }
    if (stopTimePs >= 0.0f && time_ps >= stopTimePs) {
        for (int n = 0; n < NUM_MODES; n++)
            adaptiveState[n].x = 0.0f;
        return;
    }

    float dt = time_ps - startTimePs;
    int currentPeriod = (int)(dt / periodPs);

    for (int n = 0; n < NUM_MODES; n++) {
        int lastPeriod = (int)adaptiveState[n].y;
        if (currentPeriod > lastPeriod) {
            float4 mp = modModParams[n];
            float g_target = mp.x;
            float T_target = mp.y;
            float minAmp = mp.z;
            float maxAmp = mp.w;
            float T_bath = fmaxf(bathTemperatureK, 1.0f);
            float newAmp = g_target * sqrtf(T_target / T_bath);
            newAmp = fmaxf(minAmp, fminf(maxAmp, newAmp));
            adaptiveState[n].x = newAmp;
            adaptiveState[n].y = (float)currentPeriod;
        }
    }
}

/**
 * Compute the molecular dipole moment, excluding ALL cavity particles.
 * Uses UNWRAPPED positions (posq corrected by posCellOffsets) so that the
 * dipole is continuous across periodic boundaries -- matching cav-hoomd.
 */
KERNEL void computeMultiModeDipole(GLOBAL const real4* RESTRICT posq,
        GLOBAL const float* RESTRICT charges,
        GLOBAL float* RESTRICT dipole,
        GLOBAL const int* RESTRICT cavityIndices,
        GLOBAL const int4* RESTRICT posCellOffsets,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    LOCAL float localDipoleX[WORK_GROUP_SIZE];
    LOCAL float localDipoleY[WORK_GROUP_SIZE];
    int localId = LOCAL_ID;

    float dx = 0.0f, dy = 0.0f;
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        bool isCavity = false;
        for (int m = 0; m < NUM_MODES; m++) {
            if (i == cavityIndices[m]) {
                isCavity = true;
                break;
            }
        }
        if (!isCavity) {
            real4 pos = posq[i];
            int4 offset = posCellOffsets[i];
            float ux = pos.x - offset.x * periodicBoxVecX.x - offset.y * periodicBoxVecY.x - offset.z * periodicBoxVecZ.x;
            float uy = pos.y - offset.x * periodicBoxVecX.y - offset.y * periodicBoxVecY.y - offset.z * periodicBoxVecZ.y;
            float q = charges[i];
            dx += q * ux;
            dy += q * uy;
        }
    }

    localDipoleX[localId] = dx;
    localDipoleY[localId] = dy;
    SYNC_THREADS;

    // Tree reduction
    for (int stride = WORK_GROUP_SIZE/2; stride > 0; stride >>= 1) {
        if (localId < stride) {
            localDipoleX[localId] += localDipoleX[localId + stride];
            localDipoleY[localId] += localDipoleY[localId + stride];
        }
        SYNC_THREADS;
    }

    if (localId == 0) {
        ATOMIC_ADD(&dipole[0], localDipoleX[0]);
        ATOMIC_ADD(&dipole[1], localDipoleY[0]);
    }
}

/**
 * Compute multi-mode cavity forces and energies.
 *
 * Each molecular particle thread loops over all N modes to accumulate forces.
 * Each cavity particle gets its own force from its respective mode.
 *
 * modeParams[n]: (K_n, eps_n, f_n, reorderedCavIdx_n) as float4
 * dsePrefactor: precomputed (1/2) * sum_n(eps_n^2/K_n * f_n^2) in OpenMM units
 */
KERNEL void computeMultiModeForces(GLOBAL const real4* RESTRICT posq,
        GLOBAL const float* RESTRICT charges,
        GLOBAL mm_ulong* RESTRICT forceBuffers,
        GLOBAL const float* RESTRICT dipole,
        GLOBAL float* RESTRICT energyBuffer,
        GLOBAL const int4* RESTRICT posCellOffsets,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL const float4* RESTRICT modeParams,
        GLOBAL const int* RESTRICT cavityIndices,
        float dsePrefactor,
        int paddedNumAtoms,
        GLOBAL const float2* RESTRICT adaptiveState,
        int modulationEnabled,
        float modPeriodPs, float modDutyCycle,
        float modStartTimePs, float modStopTimePs, float time_ps,
        float omega1, float conversionFactor, float photonMassAu, int includeDSE) {

    const float HARTREE_TO_KJMOL = 2625.4996f;
    const float BOHR_TO_NM = 0.052917721f;
    const float AMU_TO_AU = 1822.8885f;
    const float CONV = HARTREE_TO_KJMOL / (BOHR_TO_NM * BOHR_TO_NM);

    float dipoleX = dipole[0];
    float dipoleY = dipole[1];

    // Determine per-mode effective lambda when modulation is active
    // For non-modulated path, eps_n comes from modeParams[n].y (precomputed)
    float effectiveLambda[NUM_MODES];
    float effectiveEps[NUM_MODES];
    float dynamicDSEPrefactor = dsePrefactor;

    if (modulationEnabled) {
        float squareWaveOn = 0.0f;
        if (time_ps >= modStartTimePs && (modStopTimePs < 0.0f || time_ps < modStopTimePs)) {
            float dt = time_ps - modStartTimePs;
            float phase = dt / modPeriodPs;
            phase = phase - floorf(phase);
            squareWaveOn = (phase < modDutyCycle) ? 1.0f : 0.0f;
        }

        dynamicDSEPrefactor = 0.0f;
        for (int n = 0; n < NUM_MODES; n++) {
            float lambda_n = squareWaveOn * adaptiveState[n].x;
            effectiveLambda[n] = lambda_n;
            float omega_n = (float)(n + 1) * omega1;
            float eps_n = lambda_n * omega_n * CONV;
            effectiveEps[n] = eps_n;
            float f_n = modeParams[n].z;
            float K_n = modeParams[n].x;
            if (K_n > 0.0f)
                dynamicDSEPrefactor += eps_n * eps_n / K_n * f_n * f_n;
        }
        dynamicDSEPrefactor *= 0.5f;
    } else {
        for (int n = 0; n < NUM_MODES; n++) {
            effectiveEps[n] = modeParams[n].y;
        }
    }

    // Energy (thread 0 only)
    if (GLOBAL_ID == 0) {
        float harmonicTotal = 0.0f;
        float couplingTotal = 0.0f;

        for (int n = 0; n < NUM_MODES; n++) {
            float K_n = modeParams[n].x;
            float eps_n = effectiveEps[n];
            float f_n = modeParams[n].z;
            int cavIdx = (int) modeParams[n].w;

            real4 posWrapped = posq[cavIdx];
            int4 offset = posCellOffsets[cavIdx];
            float qx = posWrapped.x - offset.x * periodicBoxVecX.x - offset.y * periodicBoxVecY.x - offset.z * periodicBoxVecZ.x;
            float qy = posWrapped.y - offset.x * periodicBoxVecX.y - offset.y * periodicBoxVecY.y - offset.z * periodicBoxVecZ.y;
            float qz = posWrapped.z - offset.x * periodicBoxVecX.z - offset.y * periodicBoxVecY.z - offset.z * periodicBoxVecZ.z;

            harmonicTotal += 0.5f * K_n * (qx*qx + qy*qy + qz*qz);
            couplingTotal += eps_n * f_n * (qx*dipoleX + qy*dipoleY);
        }

        float useDSE = modulationEnabled ? dynamicDSEPrefactor : dsePrefactor;
        float dipoleSelfTotal = includeDSE ? useDSE * (dipoleX*dipoleX + dipoleY*dipoleY) : 0.0f;

        energyBuffer[0] = harmonicTotal;
        energyBuffer[1] = couplingTotal;
        energyBuffer[2] = dipoleSelfTotal;
    }

    // Molecular particle forces
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        bool isCavity = false;
        for (int m = 0; m < NUM_MODES; m++) {
            if (i == cavityIndices[m]) {
                isCavity = true;
                break;
            }
        }

        if (!isCavity) {
            float q_i = charges[i];
            float fx = 0.0f;
            float fy = 0.0f;

            for (int n = 0; n < NUM_MODES; n++) {
                float K_n = modeParams[n].x;
                float eps_n = effectiveEps[n];
                float f_n = modeParams[n].z;
                int cavIdx = (int) modeParams[n].w;
                float epsf_n = eps_n * f_n;

                real4 posWrapped = posq[cavIdx];
                int4 offset = posCellOffsets[cavIdx];
                float qx = posWrapped.x - offset.x * periodicBoxVecX.x - offset.y * periodicBoxVecY.x - offset.z * periodicBoxVecZ.x;
                float qy = posWrapped.y - offset.x * periodicBoxVecX.y - offset.y * periodicBoxVecY.y - offset.z * periodicBoxVecZ.y;

                float epsfOverK_n = (includeDSE && K_n > 0.0f) ? (epsf_n / K_n) : 0.0f;
                float DqX = qx + epsfOverK_n * dipoleX;
                float DqY = qy + epsfOverK_n * dipoleY;

                fx += -epsf_n * q_i * DqX;
                fy += -epsf_n * q_i * DqY;
            }

            ATOMIC_ADD(&forceBuffers[i], (mm_ulong) realToFixedPoint((real)fx));
            ATOMIC_ADD(&forceBuffers[i+paddedNumAtoms], (mm_ulong) realToFixedPoint((real)fy));
        }
    }

    // Cavity particle forces
    if (GLOBAL_ID == 0) {
        for (int n = 0; n < NUM_MODES; n++) {
            float K_n = modeParams[n].x;
            float eps_n = effectiveEps[n];
            float f_n = modeParams[n].z;
            int cavIdx = (int) modeParams[n].w;
            float epsf_n = eps_n * f_n;

            real4 posWrapped = posq[cavIdx];
            int4 offset = posCellOffsets[cavIdx];
            float qx = posWrapped.x - offset.x * periodicBoxVecX.x - offset.y * periodicBoxVecY.x - offset.z * periodicBoxVecZ.x;
            float qy = posWrapped.y - offset.x * periodicBoxVecX.y - offset.y * periodicBoxVecY.y - offset.z * periodicBoxVecZ.y;
            float qz = posWrapped.z - offset.x * periodicBoxVecX.z - offset.y * periodicBoxVecY.z - offset.z * periodicBoxVecZ.z;

            real fxCav = -K_n * qx - epsf_n * dipoleX;
            real fyCav = -K_n * qy - epsf_n * dipoleY;
            real fzCav = -K_n * qz;

            ATOMIC_ADD(&forceBuffers[cavIdx], (mm_ulong) realToFixedPoint(fxCav));
            ATOMIC_ADD(&forceBuffers[cavIdx+paddedNumAtoms], (mm_ulong) realToFixedPoint(fyCav));
            ATOMIC_ADD(&forceBuffers[cavIdx+2*paddedNumAtoms], (mm_ulong) realToFixedPoint(fzCav));
        }
    }
}
