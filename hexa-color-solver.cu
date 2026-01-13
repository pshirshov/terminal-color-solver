/**
 * Hexa Color Solver v3.0 - OKLCH + APCA
 *
 * GPU-accelerated genetic algorithm for optimal 16-color terminal palettes.
 *
 * Color Space: OKLCH (perceptually uniform)
 *   - L: Lightness (0-1)
 *   - C: Chroma/saturation (0-0.4)
 *   - H: Hue angle (0-360°)
 *
 * Contrast Metric: APCA (WCAG 3.0)
 *   - Better dark mode support than WCAG 2.1
 *   - Lc values: |Lc| >= 40 for readable text
 *
 * Fixed colors:
 *   - Black: #000000
 *   - Br.White: #ffffff
 *
 * Build: cmake .. && cmake --build .
 * Run:   ./hexa-color-solver -g 5000 -p 200000
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <cerrno>
#include <climits>
#include <algorithm>
#include <vector>
#include <random>
#include <sys/stat.h>

#include "color.cuh"
#include "output.hpp"

// =============================================================================
// Color indices
// =============================================================================
enum ColorIndex {
    BLACK = 0, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE,
    BR_BLACK, BR_RED, BR_GREEN, BR_YELLOW, BR_BLUE, BR_MAGENTA, BR_CYAN, BR_WHITE
};

// =============================================================================
// OKLCH Constraint System
// =============================================================================

// Slot constraint in OKLCH space
struct OklchSlotConstraint {
    double target_hue;      // Target hue in degrees (0-360)
    double hue_tolerance;   // Allowed hue deviation (+/- degrees)
    double min_L, max_L;    // Lightness range (0-1)
    double min_C, max_C;    // Chroma range (0-~0.4)
    bool fixed;            // Is this a fixed RGB color?
    double fixed_r, fixed_g, fixed_b;  // Fixed RGB values (0-255)
    int8_t base_slot;      // For bright colors: base slot index (-1 if none)
    double max_hue_drift;   // Max hue deviation from base color (degrees, 0 = unlimited)
};

// APCA pair constraint
struct ApcaPairConstraint {
    int8_t fg_index;       // Foreground color index (0-15)
    int8_t bg_index;       // Background color index (0-15)
    double min_apca;       // Minimum APCA contrast (absolute value)
    double target_apca;    // Target APCA for uniformity (0 = no target, just meet minimum)
};

// OKLCH Hue Reference Values (degrees):
// Red:     ~29°
// Yellow:  ~110°
// Green:   ~142°
// Cyan:    ~195°
// Blue:    ~264°
// Magenta: ~328°

// Slot constraints with reasonable OKLCH defaults
// Format: target_hue, hue_tolerance, min_L, max_L, min_C, max_C, fixed, fixed_r, fixed_g, fixed_b, base_slot, max_hue_drift
const OklchSlotConstraint oklch_slot_constraints[16] = {
    // Base colors (0-7)
    {   0,   0, 0.00, 0.00, 0.00, 0.00, true,    0,   0,   0, -1,  0},  // 0: BLACK (fixed)
    {  29,  25, 0.40, 0.90, 0.08, 0.30, false,   0,   0,   0, -1,  0},  // 1: RED (L max for APCA≥60)
    { 142,  25, 0.40, 0.90, 0.08, 0.30, false,   0,   0,   0, -1,  0},  // 2: GREEN
    { 110,  25, 0.40, 0.90, 0.08, 0.30, false,   0,   0,   0, -1,  0},  // 3: YELLOW
    { 264,  25, 0.40, 0.90, 0.08, 0.30, false,   0,   0,   0, -1,  0},  // 4: BLUE (L max for APCA≥60)
    { 328,  25, 0.40, 0.90, 0.08, 0.30, false,   0,   0,   0, -1,  0},  // 5: MAGENTA (L max for APCA)
    { 195,  25, 0.40, 0.90, 0.08, 0.30, false,   0,   0,   0, -1,  0},  // 6: CYAN
    {   0,   0, 0.75, 0.92, 0.00, 0.03, false,   0,   0,   0, -1,  0},  // 7: WHITE (L max for APCA≥85)

    // Bright colors (8-15)
    {   0,   0, 0.50, 0.60, 0.00, 0.03, false,   0,   0,   0, -1,  0},  // 8: BR_BLACK (L≥0.50 for APCA≥40 on black)
    {  29,  25, 0.65, 0.98, 0.12, 0.30, false,   0,   0,   0,  1, 20},  // 9: BR_RED (L max for APCA≥80)
    { 142,  25, 0.65, 0.98, 0.12, 0.30, false,   0,   0,   0,  2, 20},  // 10: BR_GREEN (base=GREEN)
    { 110,  25, 0.65, 0.98, 0.12, 0.30, false,   0,   0,   0,  3, 20},  // 11: BR_YELLOW (base=YELLOW)
    { 264,  25, 0.65, 0.98, 0.12, 0.30, false,   0,   0,   0,  4, 20},  // 12: BR_BLUE (L max for APCA≥80)
    { 328,  25, 0.65, 0.98, 0.12, 0.30, false,   0,   0,   0,  5, 20},  // 13: BR_MAGENTA (L max for APCA≥80)
    { 195,  25, 0.65, 0.98, 0.12, 0.30, false,   0,   0,   0,  6, 20},  // 14: BR_CYAN (base=CYAN)
    {   0,   0, 1.00, 1.00, 0.00, 0.00, true,  255, 255, 255, -1,  0},  // 15: BR_WHITE (fixed)
};

// APCA pair constraints: {fg_index, bg_index, min_apca, target_apca}
// target_apca > 0 enables uniformity optimization within groups
const ApcaPairConstraint apca_pair_constraints[] = {
    // Base colors on black - target 50 for uniformity (all should cluster around this value)
    {RED,        BLACK, 60.0, 65.0},  // red on black
    {YELLOW,     BLACK, 60.0, 65.0},  // yellow on black
    {MAGENTA,    BLACK, 60.0, 65.0},  // magenta on black

    {CYAN,       BLACK, 60.0, 60.0},  // cyan on black
    {GREEN,      BLACK, 50.0, 50.0},  // green on black
    {BLUE,       BLACK, 30.0, 30.0},  // blue on black
    {WHITE,      BLACK, 85.0, 85.0},  // white on black

    // Bright colors on black - target 80 for uniformity
    {BR_RED,     BLACK, 85.0, 85.0},  // br.red on black
    {BR_YELLOW,  BLACK, 85.0, 85.0},  // br.yellow on black
    {BR_MAGENTA, BLACK, 85.0, 85.0},  // br.magenta on black

    // might want to lower
    {BR_GREEN,   BLACK, 85.0, 85.0},  // br.green on black
    {BR_BLUE,    BLACK, 85.0, 85.0},  // br.blue on black
    {BR_CYAN,    BLACK, 85.0, 85.0},  // br.cyan on black

    {BR_BLACK,   BLACK, 40.0, 40.0},  // br.black on black (no uniformity target - standalone)
};

constexpr int APCA_CONSTRAINT_COUNT = sizeof(apca_pair_constraints) / sizeof(apca_pair_constraints[0]);

// Device constant memory for OKLCH constraints
__constant__ OklchSlotConstraint d_oklch_slots[16];
__constant__ ApcaPairConstraint d_apca_pairs[64];  // Max 64 pairs
__constant__ int d_apca_pair_count;

// =============================================================================
// CUDA Kernels
// =============================================================================

__global__ void init_curand(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

/**
 * Initialize population in OKLCH space.
 * Generates random L, C, H values within slot constraints.
 * Clamps chroma to stay in sRGB gamut.
 */
__global__ void init_population(double* palettes, curandState* states, int n_palettes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_palettes) return;

    curandState localState = states[idx];

    for (int slot = 0; slot < 16; slot++) {
        OklchSlotConstraint c = d_oklch_slots[slot];
        int base = idx * 16 * 3 + slot * 3;

        if (c.fixed) {
            // Fixed RGB color - convert to OKLCH for storage
            double L, C, H;
            color::rgb_to_oklch(c.fixed_r, c.fixed_g, c.fixed_b, &L, &C, &H);
            palettes[base + 0] = L;
            palettes[base + 1] = C;
            palettes[base + 2] = H;
        } else {
            // Random OKLCH within constraints
            double L = c.min_L + (double)curand_uniform(&localState) * (c.max_L - c.min_L);
            double H = c.target_hue + ((double)curand_uniform(&localState) - 0.5) * 2.0 * c.hue_tolerance;
            H = color::oklch::normalize_hue(H);

            // Determine chroma range, clamped to gamut
            double max_C = color::oklch_max_chroma(L, H);
            double C_min = fmin(c.min_C, max_C);
            double C_max = fmin(c.max_C, max_C);

            double C = C_min + (double)curand_uniform(&localState) * (C_max - C_min);

            palettes[base + 0] = L;
            palettes[base + 1] = C;
            palettes[base + 2] = H;
        }
    }

    states[idx] = localState;
}

/**
 * Evaluate fitness using APCA constraints and OKLCH perceptual metrics.
 * Palettes are in OKLCH space, converted to RGB for APCA evaluation.
 */
__global__ void evaluate_fitness(double* palettes, double* fitness, int n_palettes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_palettes) return;

    double score = 0.0;
    int base = idx * 16 * 3;

    // Convert OKLCH palette to RGB (cache for reuse)
    double rgb[16][3];
    for (int i = 0; i < 16; i++) {
        double L = palettes[base + i * 3 + 0];
        double C = palettes[base + i * 3 + 1];
        double H = palettes[base + i * 3 + 2];
        color::oklch_to_srgb(L, C, H, &rgb[i][0], &rgb[i][1], &rgb[i][2]);
    }

    // =========================================================================
    // CONSTRAINT 1: APCA pair constraints (hard requirements + uniformity)
    // =========================================================================
    for (int i = 0; i < d_apca_pair_count; i++) {
        ApcaPairConstraint p = d_apca_pairs[i];
        int fg = p.fg_index;
        int bg = p.bg_index;

        double apca = color::apca::contrast_abs(
            rgb[fg][0], rgb[fg][1], rgb[fg][2],
            rgb[bg][0], rgb[bg][1], rgb[bg][2]
        );

        if (apca >= p.min_apca) {
            // Constraint met - base reward
            score += 100.0;

            if (p.target_apca > 0.0) {
                // Uniformity mode: asymmetric penalty (heavier below, lighter above)
                if (apca < p.target_apca) {
                    double shortfall = p.target_apca - apca;
                    score -= shortfall * 3.0;  // Heavy penalty for below target
                } else {
                    double excess = apca - p.target_apca;
                    score -= excess * 1.0;  // Light penalty for exceeding (uniformity)
                }
            } else {
                // No target: reward exceeding minimum (old behavior)
                score += (apca - p.min_apca) * 5.0;
            }
        } else {
            // Constraint violated - heavy penalty proportional to shortfall
            score -= (p.min_apca - apca) * 50.0;
        }
    }

    // =========================================================================
    // CONSTRAINT 2: Hue drift for bright colors (must match base)
    // =========================================================================
    for (int slot = 8; slot <= 14; slot++) {
        OklchSlotConstraint c = d_oklch_slots[slot];
        if (c.base_slot < 0 || c.max_hue_drift <= 0.0) continue;

        double H_bright = palettes[base + slot * 3 + 2];
        double H_base = palettes[base + c.base_slot * 3 + 2];

        double hdist = color::hue_distance(H_bright, H_base);
        if (hdist <= c.max_hue_drift) {
            score += 30.0;  // Bonus for matching hue
        } else {
            score -= (hdist - c.max_hue_drift) * 5.0;  // Penalty for drift
        }
    }

    // =========================================================================
    // CONSTRAINT 3: Gamut validity
    // =========================================================================
    for (int i = 0; i < 16; i++) {
        double L = palettes[base + i * 3 + 0];
        double C = palettes[base + i * 3 + 1];
        double H = palettes[base + i * 3 + 2];

        if (!color::oklch_in_gamut(L, C, H)) {
            score -= 500.0;  // Heavy penalty for out-of-gamut
        }
    }

    // =========================================================================
    // BONUS 1: Hue spacing for base colors (1-6)
    // Reward evenly spaced hues on the color wheel
    // =========================================================================
    {
        double hues[6];
        for (int i = 0; i < 6; i++) {
            hues[i] = palettes[base + (i + 1) * 3 + 2];
        }

        double min_hue_dist = 360.0;
        for (int i = 0; i < 6; i++) {
            for (int j = i + 1; j < 6; j++) {
                double hdist = color::hue_distance(hues[i], hues[j]);
                if (hdist < min_hue_dist) {
                    min_hue_dist = hdist;
                }
            }
        }

        // Ideal minimum spacing for 6 colors is 60° but we accept 40°
        if (min_hue_dist >= 40.0) {
            score += min_hue_dist * 2.0;
        } else {
            score -= (40.0 - min_hue_dist) * 5.0;
        }
    }

    // =========================================================================
    // BONUS 2: Chroma (prefer more saturated colors)
    // =========================================================================
    for (int i = 1; i <= 6; i++) {  // Base colors only
        double C = palettes[base + i * 3 + 1];
        score += C * 50.0;  // Small bonus for saturation
    }

    // =========================================================================
    // BONUS 3: Perceptual distance (Oklab) between base colors
    // =========================================================================
    {
        double min_dist = 1000.0;
        for (int i = 1; i <= 7; i++) {
            for (int j = i + 1; j <= 7; j++) {
                double dist = color::oklab_distance(
                    rgb[i][0], rgb[i][1], rgb[i][2],
                    rgb[j][0], rgb[j][1], rgb[j][2]
                );
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
        }

        // Reward good separation (0.15 is noticeable difference)
        if (min_dist >= 0.15) {
            score += min_dist * 100.0;
        } else {
            score -= (0.15 - min_dist) * 300.0;
        }
    }

    // =========================================================================
    // BONUS 4: All other APCA pairs (soft bonus for general readability)
    // =========================================================================
    // Small bonus for any pair with good APCA (not covered by constraints)
    for (int bg = 0; bg < 8; bg++) {
        for (int fg = 0; fg < 16; fg++) {
            if (fg == bg) continue;
            double apca = color::apca::contrast_abs(
                rgb[fg][0], rgb[fg][1], rgb[fg][2],
                rgb[bg][0], rgb[bg][1], rgb[bg][2]
            );
            if (apca >= 40.0) {
                score += 1.0;  // Small bonus for readable pairs
            }
        }
    }

    fitness[idx] = score;
}

/**
 * Crossover and mutation in OKLCH space.
 * Uses circular interpolation for hue.
 */
__global__ void crossover_and_mutate(
    double* old_pop, double* new_pop, double* fitness,
    int* elite_indices, int elite_count,
    curandState* states, double mutation_rate, int n_palettes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_palettes) return;

    curandState localState = states[idx];
    int new_base = idx * 16 * 3;

    // Elite: copy directly
    if (idx < elite_count) {
        int old_idx = elite_indices[idx];
        int old_base = old_idx * 16 * 3;
        for (int i = 0; i < 48; i++) {
            new_pop[new_base + i] = old_pop[old_base + i];
        }
        states[idx] = localState;
        return;
    }

    // Tournament selection for parents
    int p1_idx = elite_indices[(int)((double)curand_uniform(&localState) * elite_count)];
    int p2_idx = elite_indices[(int)((double)curand_uniform(&localState) * elite_count)];

    int p1_base = p1_idx * 16 * 3;
    int p2_base = p2_idx * 16 * 3;

    // Crossover and mutate each color slot
    for (int slot = 0; slot < 16; slot++) {
        OklchSlotConstraint c = d_oklch_slots[slot];
        int offset = slot * 3;

        if (c.fixed) {
            // Fixed color - convert from RGB
            double L, C, H;
            color::rgb_to_oklch(c.fixed_r, c.fixed_g, c.fixed_b, &L, &C, &H);
            new_pop[new_base + offset + 0] = L;
            new_pop[new_base + offset + 1] = C;
            new_pop[new_base + offset + 2] = H;
        } else {
            // Crossover: blend or select
            double L1 = old_pop[p1_base + offset + 0];
            double C1 = old_pop[p1_base + offset + 1];
            double H1 = old_pop[p1_base + offset + 2];
            double L2 = old_pop[p2_base + offset + 0];
            double C2 = old_pop[p2_base + offset + 1];
            double H2 = old_pop[p2_base + offset + 2];

            double t = (double)curand_uniform(&localState);
            double L = L1 + t * (L2 - L1);
            double C = C1 + t * (C2 - C1);
            double H = color::oklch::lerp_hue(H1, H2, t);

            // Mutation
            if ((double)curand_uniform(&localState) < mutation_rate) {
                // Mutate L
                double L_range = c.max_L - c.min_L;
                L += (double)curand_normal(&localState) * L_range * 0.1;
                if (L < c.min_L) L = c.min_L;
                if (L > c.max_L) L = c.max_L;
            }

            if ((double)curand_uniform(&localState) < mutation_rate) {
                // Mutate H (circular)
                H += (double)curand_normal(&localState) * c.hue_tolerance * 0.3;
                H = color::oklch::normalize_hue(H);

                // Clamp to constraint range
                double target = c.target_hue;
                double hdist = color::hue_distance(H, target);
                if (hdist > c.hue_tolerance) {
                    // Push back toward valid range
                    double t_factor = c.hue_tolerance / hdist;
                    H = color::oklch::lerp_hue(target, H, t_factor);
                }
            }

            if ((double)curand_uniform(&localState) < mutation_rate) {
                // Mutate C
                double C_range = c.max_C - c.min_C;
                C += (double)curand_normal(&localState) * C_range * 0.15;

                // Clamp to constraint range and gamut
                double max_C = color::oklch_max_chroma(L, H);
                if (C < c.min_C) C = c.min_C;
                if (C > c.max_C) C = c.max_C;
                if (C > max_C) C = max_C;
            }

            // Ensure gamut validity
            double max_C = color::oklch_max_chroma(L, H);
            if (C > max_C) C = max_C;

            new_pop[new_base + offset + 0] = L;
            new_pop[new_base + offset + 1] = C;
            new_pop[new_base + offset + 2] = H;
        }
    }

    states[idx] = localState;
}

/**
 * Convert OKLCH palette to RGB (for output/display).
 * Single palette conversion on host side.
 */
void oklch_palette_to_rgb(double* oklch_palette, double* rgb_palette) {
    for (int i = 0; i < 16; i++) {
        const OklchSlotConstraint& c = oklch_slot_constraints[i];
        if (c.fixed) {
            // Use exact fixed RGB values to avoid OKLCH round-trip precision loss
            rgb_palette[i * 3 + 0] = c.fixed_r;
            rgb_palette[i * 3 + 1] = c.fixed_g;
            rgb_palette[i * 3 + 2] = c.fixed_b;
        } else {
            double L = oklch_palette[i * 3 + 0];
            double C = oklch_palette[i * 3 + 1];
            double H = oklch_palette[i * 3 + 2];
            color::oklch_to_srgb(L, C, H, &rgb_palette[i * 3 + 0], &rgb_palette[i * 3 + 1], &rgb_palette[i * 3 + 2]);
        }
    }
}

// =============================================================================
// Host Functions (Reporting)
// =============================================================================

const char* apca_ansi_color(double lc) {
    double abs_lc = fabs(lc);
    if (abs_lc >= 75.0) return "\033[32m";  // Green - body text
    if (abs_lc >= 60.0) return "\033[33m";  // Yellow - large text
    if (abs_lc >= 45.0) return "\033[38;2;255;165;0m";  // Orange - bold only
    return "\033[31m";                        // Red - insufficient
}

void print_color_demo(double* palette) {
    const char* names[] = {
        "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white",
        "br.black", "br.red", "br.green", "br.yellow", "br.blue", "br.magenta", "br.cyan", "br.white"
    };

    printf("\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("                              COLOR PALETTE RESULTS\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n\n");

    // Palette table and contrast matrix (using FTXUI)
    output::print_palette_and_matrix(palette, names);

    printf("\n");

    // FM pairs tables (bright on regular, colors on blue/green/cyan)
    output::print_fm_pairs_tables(palette, names);

    printf("\n");

    // APCA contrast check against black background
    printf("APCA Contrast (colors on black background):\n");
    printf("────────────────────────────────────────────────────────────────\n");

    double black_r = palette[BLACK * 3 + 0];
    double black_g = palette[BLACK * 3 + 1];
    double black_b = palette[BLACK * 3 + 2];

    for (int i = 1; i <= 14; i++) {
        if (i == 8) continue;  // Skip br.black
        double r = palette[i * 3 + 0];
        double g = palette[i * 3 + 1];
        double b = palette[i * 3 + 2];
        double lc = color::apca::contrast(r, g, b, black_r, black_g, black_b);
        double abs_lc = fabs(lc);

        double required = 40.0;
        const char* status = abs_lc >= required ? "\033[32m✓\033[0m" : "\033[31m✗\033[0m";
        printf("  %-12s Lc=%6.1f (min 40)  %s\n", names[i], abs_lc, status);
    }

    // Br.black on black (special case - lower requirement)
    {
        double r = palette[BR_BLACK * 3 + 0];
        double g = palette[BR_BLACK * 3 + 1];
        double b = palette[BR_BLACK * 3 + 2];
        double lc = color::apca::contrast(r, g, b, black_r, black_g, black_b);
        double abs_lc = fabs(lc);
        const char* status = abs_lc >= 15.0 ? "\033[32m✓\033[0m" : "\033[31m✗\033[0m";
        printf("  %-12s Lc=%6.1f (min 15)  %s\n", names[BR_BLACK], abs_lc, status);
    }

    // OKLCH Analysis
    printf("\nOKLCH Analysis (Perceptually Uniform Color Space):\n");
    printf("────────────────────────────────────────────────────────────────\n");
    printf("%-12s  L (light)  C (chroma)  H (hue°)\n", "Color");
    printf("────────────────────────────────────────────────────────────────\n");

    double base_hues[6];
    for (int i = 1; i <= 6; i++) {
        double r = palette[i * 3 + 0];
        double g = palette[i * 3 + 1];
        double b = palette[i * 3 + 2];
        double L, C, H;
        color::rgb_to_oklch(r, g, b, &L, &C, &H);
        base_hues[i-1] = H;
        printf("%-12s  %5.3f      %5.3f       %6.1f°\n", names[i], L, C, H);
    }

    // Hue spacing analysis
    printf("\nHue Spacing (ideal: 60° between colors):\n");
    printf("────────────────────────────────────────────────────────────────\n");
    double min_hue_dist = 360.0;
    const char* min_pair_a = "";
    const char* min_pair_b = "";
    for (int i = 0; i < 6; i++) {
        for (int j = i + 1; j < 6; j++) {
            double hdist = color::hue_distance(base_hues[i], base_hues[j]);
            if (hdist < min_hue_dist) {
                min_hue_dist = hdist;
                min_pair_a = names[i + 1];
                min_pair_b = names[j + 1];
            }
        }
    }
    printf("  Minimum hue distance: %.1f° (between %s and %s)\n", min_hue_dist, min_pair_a, min_pair_b);
    if (min_hue_dist >= 50.0) {
        printf("  Status: \033[32m✓ Good spacing\033[0m\n");
    } else if (min_hue_dist >= 30.0) {
        printf("  Status: \033[33m~ Acceptable spacing\033[0m\n");
    } else {
        printf("  Status: \033[31m✗ Colors too close in hue\033[0m\n");
    }

    // Bright color hue matching
    printf("\nBright/Base Hue Matching (bright should match base hue):\n");
    printf("────────────────────────────────────────────────────────────────\n");
    for (int i = 1; i <= 6; i++) {
        double L1, C1, H1, L2, C2, H2;
        color::rgb_to_oklch(palette[i * 3 + 0], palette[i * 3 + 1], palette[i * 3 + 2], &L1, &C1, &H1);
        color::rgb_to_oklch(palette[(i + 8) * 3 + 0], palette[(i + 8) * 3 + 1], palette[(i + 8) * 3 + 2], &L2, &C2, &H2);
        double hdist = color::hue_distance(H1, H2);
        printf("  %-10s → br.%-7s: ΔH=%5.1f°  %s\n",
               names[i], names[i], hdist,
               hdist <= 30.0 ? "\033[32m✓\033[0m" : "\033[31m✗ hue drift\033[0m");
    }
}

// =============================================================================
// Main
// =============================================================================

// Write theme to file
void write_theme_file(double* palette, const char* filepath) {
    FILE* f = fopen(filepath, "w");
    if (!f) {
        printf("Error: Could not open %s for writing: %s\n", filepath, strerror(errno));
        return;
    }

    fprintf(f, "# OKLCH + APCA Optimized Terminal Theme\n");
    fprintf(f, "# Generated by Hexa Color Solver v3.0\n");
    fprintf(f, "# APCA contrast: Lc>=40 on black, perceptually uniform hues\n");
    fprintf(f, "#\n\n");

    // Palette
    for (int i = 0; i < 16; i++) {
        int r = (int)palette[i * 3 + 0];
        int g = (int)palette[i * 3 + 1];
        int b = (int)palette[i * 3 + 2];
        fprintf(f, "palette = %d=#%02x%02x%02x\n", i, r, g, b);
    }

    // Background/foreground
    int bg_r = (int)palette[BLACK * 3 + 0];
    int bg_g = (int)palette[BLACK * 3 + 1];
    int bg_b = (int)palette[BLACK * 3 + 2];
    int fg_r = (int)palette[WHITE * 3 + 0];
    int fg_g = (int)palette[WHITE * 3 + 1];
    int fg_b = (int)palette[WHITE * 3 + 2];

    fprintf(f, "\nbackground = #%02x%02x%02x\n", bg_r, bg_g, bg_b);
    fprintf(f, "foreground = #%02x%02x%02x\n", fg_r, fg_g, fg_b);

    // Cursor (bright yellow)
    int cursor_r = (int)palette[BR_YELLOW * 3 + 0];
    int cursor_g = (int)palette[BR_YELLOW * 3 + 1];
    int cursor_b = (int)palette[BR_YELLOW * 3 + 2];
    fprintf(f, "\ncursor-color = #%02x%02x%02x\n", cursor_r, cursor_g, cursor_b);
    fprintf(f, "cursor-text = #%02x%02x%02x\n", bg_r, bg_g, bg_b);

    // Selection (blue background)
    int sel_r = (int)palette[BLUE * 3 + 0];
    int sel_g = (int)palette[BLUE * 3 + 1];
    int sel_b = (int)palette[BLUE * 3 + 2];
    fprintf(f, "\nselection-background = #%02x%02x%02x\n", sel_r, sel_g, sel_b);
    fprintf(f, "selection-foreground = #ffffff\n");

    if (fclose(f) != 0) {
        printf("Error: Failed to write %s: %s\n", filepath, strerror(errno));
        return;
    }

    // Verify file was written and show absolute path
    char abspath[PATH_MAX];
    if (realpath(filepath, abspath)) {
        printf("\n✓ Theme written to: %s\n", abspath);
    } else {
        printf("Warning: Theme file may not have been written correctly to %s\n", filepath);
    }
}

int main(int argc, char** argv) {
    int population_size = 200000;
    int generations = 5000;
    double mutation_rate = 0.15;
    double elite_ratio = 0.1;
    const char* output_file = NULL;
    char default_output[256];

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--population") == 0 || strcmp(argv[i], "-p") == 0) {
            population_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--generations") == 0 || strcmp(argv[i], "-g") == 0) {
            generations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--mutation") == 0 || strcmp(argv[i], "-m") == 0) {
            mutation_rate = atof(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 || strcmp(argv[i], "-o") == 0) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Hexa Color Solver v3.0 - OKLCH + APCA Terminal Palette Optimizer\n\n");
            printf("Usage: %s [options]\n\n", argv[0]);
            printf("Options:\n");
            printf("  -p, --population N     Population size (default: 200000)\n");
            printf("  -g, --generations N    Number of generations (default: 5000)\n");
            printf("  -m, --mutation F       Mutation rate (default: 0.15)\n");
            printf("  -o, --output FILE      Output theme file (default: ./themes/theme-YYMMDD-HHMMSS)\n");
            printf("  -h, --help             Show this help\n");
            return 0;
        }
    }

    // Generate default output filename if not specified
    if (!output_file) {
        mkdir("./themes", 0755);
        time_t now = time(NULL);
        struct tm* t = localtime(&now);
        snprintf(default_output, sizeof(default_output), "./themes/theme-%02d%02d%02d-%02d%02d%02d",
                 (t->tm_year % 100), t->tm_mon + 1, t->tm_mday,
                 t->tm_hour, t->tm_min, t->tm_sec);
        output_file = default_output;
    }

    // Validate parameters
    if (population_size < 100) {
        printf("Error: population size must be >= 100 (got %d)\n", population_size);
        return 1;
    }
    if (generations < 1) {
        printf("Error: generations must be >= 1 (got %d)\n", generations);
        return 1;
    }

    int elite_count = (int)(population_size * elite_ratio);

    const char* names[] = {
        "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white",
        "br.black", "br.red", "br.green", "br.yellow", "br.blue", "br.magenta", "br.cyan", "br.white"
    };

    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║              Hexa Color Solver v3.0 (OKLCH + APCA)                            ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n\n");

    printf("Parameters:\n");
    printf("  Population: %d\n", population_size);
    printf("  Generations: %d\n", generations);
    printf("  Mutation rate: %.2f (adaptive)\n", mutation_rate);
    printf("  Elite ratio: %.2f\n", elite_ratio);
    printf("  Output: %s\n\n", output_file);

    printf("Color Space: OKLCH (perceptually uniform)\n");
    printf("Contrast Metric: APCA (WCAG 3.0)\n\n");

    printf("Fixed colors:\n");
    printf("  Black:    #000000\n");
    printf("  Br.White: #ffffff\n\n");

    printf("APCA Contrast Requirements:\n");
    printf("  On black (bg=0): colors 1-7, 9-14 >= 40 Lc\n");
    printf("  br.black on black: >= 15 Lc\n");
    printf("  Bright on regular (br.X on X): >= 30 Lc\n");
    printf("  On cyan/green/blue: various >= 20-30 Lc\n\n");

    printf("OKLCH Slot Constraints:\n");
    printf("  %-12s %6s %6s %11s %11s\n", "Color", "Hue°", "±Tol", "L range", "C range");
    printf("  ────────────────────────────────────────────────────────────\n");
    for (int i = 0; i < 16; i++) {
        const OklchSlotConstraint& c = oklch_slot_constraints[i];
        if (c.fixed) {
            printf("  %-12s (fixed #%02x%02x%02x)\n", names[i],
                   (int)c.fixed_r, (int)c.fixed_g, (int)c.fixed_b);
        } else {
            printf("  %-12s %5.0f° %5.0f° %4.2f-%-4.2f %4.2f-%-4.2f\n",
                   names[i], c.target_hue, c.hue_tolerance,
                   c.min_L, c.max_L, c.min_C, c.max_C);
        }
    }
    printf("\n");

    // Check CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using GPU: %s\n\n", prop.name);

    // Copy constraints to device
    cudaMemcpyToSymbol(d_oklch_slots, oklch_slot_constraints, sizeof(oklch_slot_constraints));
    cudaMemcpyToSymbol(d_apca_pairs, apca_pair_constraints, sizeof(apca_pair_constraints));
    int pair_count = APCA_CONSTRAINT_COUNT;
    cudaMemcpyToSymbol(d_apca_pair_count, &pair_count, sizeof(int));

    // Allocate memory
    size_t palette_size = population_size * 16 * 3 * sizeof(double);
    double *d_pop1, *d_pop2, *d_fitness;
    curandState* d_states;
    int* d_elite_indices;

    cudaMalloc(&d_pop1, palette_size);
    cudaMalloc(&d_pop2, palette_size);
    cudaMalloc(&d_fitness, population_size * sizeof(double));
    cudaMalloc(&d_states, population_size * sizeof(curandState));
    cudaMalloc(&d_elite_indices, elite_count * sizeof(int));

    // Initialize
    int blockSize = 256;
    int numBlocks = (population_size + blockSize - 1) / blockSize;

    printf("Initializing population...\n");
    init_curand<<<numBlocks, blockSize>>>(d_states, time(NULL), population_size);
    init_population<<<numBlocks, blockSize>>>(d_pop1, d_states, population_size);
    cudaDeviceSynchronize();

    // Host arrays for elite selection
    std::vector<double> h_fitness(population_size);
    std::vector<int> h_elite_indices(elite_count);

    // Best-ever tracking (don't rely solely on elitism)
    double best_ever_fitness = -1e9;
    std::vector<double> best_ever_palette(16 * 3);
    int best_ever_generation = 0;

    int stagnant_generations = 0;
    double current_mutation = mutation_rate;

    printf("Starting evolution...\n\n");

    for (int gen = 0; gen < generations; gen++) {
        // Evaluate fitness
        evaluate_fitness<<<numBlocks, blockSize>>>(d_pop1, d_fitness, population_size);
        cudaDeviceSynchronize();

        // Copy fitness to host
        cudaMemcpy(h_fitness.data(), d_fitness, population_size * sizeof(double), cudaMemcpyDeviceToHost);

        // Find elite indices
        std::vector<int> indices(population_size);
        for (int i = 0; i < population_size; i++) indices[i] = i;

        std::partial_sort(indices.begin(), indices.begin() + elite_count, indices.end(),
            [&h_fitness](int a, int b) { return h_fitness[a] > h_fitness[b]; });

        for (int i = 0; i < elite_count; i++) {
            h_elite_indices[i] = indices[i];
        }

        double gen_best = h_fitness[indices[0]];

        // Track best-ever palette
        if (gen_best > best_ever_fitness) {
            best_ever_fitness = gen_best;
            best_ever_generation = gen;
            // Save the best palette from device
            int best_idx = indices[0];
            cudaMemcpy(best_ever_palette.data(), d_pop1 + best_idx * 16 * 3,
                       16 * 3 * sizeof(double), cudaMemcpyDeviceToHost);
            stagnant_generations = 0;
            current_mutation = mutation_rate;
        } else {
            stagnant_generations++;
            if (stagnant_generations > 100) {
                current_mutation = fmin(0.5, current_mutation * 1.01);
            }
        }

        // Progress output
        if (gen % 500 == 0 || gen == generations - 1) {
            printf("Gen %5d: best=%.2f, avg=%.2f, mutation=%.3f\n",
                   gen, gen_best,
                   std::accumulate(h_fitness.begin(), h_fitness.end(), 0.0) / population_size,
                   current_mutation);
        }

        if (gen < generations - 1) {
            // Copy elite indices to device
            cudaMemcpy(d_elite_indices, h_elite_indices.data(), elite_count * sizeof(int), cudaMemcpyHostToDevice);

            // Crossover and mutation
            crossover_and_mutate<<<numBlocks, blockSize>>>(
                d_pop1, d_pop2, d_fitness, d_elite_indices, elite_count,
                d_states, current_mutation, population_size
            );
            cudaDeviceSynchronize();

            // Swap populations
            std::swap(d_pop1, d_pop2);
        }
    }

    // Use best-ever palette (not just final generation)
    printf("\nBest solution found at generation %d (fitness=%.2f)\n",
           best_ever_generation, best_ever_fitness);

    // Convert OKLCH palette to RGB for display and output
    std::vector<double> rgb_palette(16 * 3);
    oklch_palette_to_rgb(best_ever_palette.data(), rgb_palette.data());

    // Print results
    print_color_demo(rgb_palette.data());

    // Write theme file (always, defaults to ./themes/theme-YYMMDD-HHMMSS)
    write_theme_file(rgb_palette.data(), output_file);

    // Cleanup
    cudaFree(d_pop1);
    cudaFree(d_pop2);
    cudaFree(d_fitness);
    cudaFree(d_states);
    cudaFree(d_elite_indices);

    return 0;
}
