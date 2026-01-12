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
    float target_hue;      // Target hue in degrees (0-360)
    float hue_tolerance;   // Allowed hue deviation (+/- degrees)
    float min_L, max_L;    // Lightness range (0-1)
    float min_C, max_C;    // Chroma range (0-~0.4)
    bool fixed;            // Is this a fixed RGB color?
    float fixed_r, fixed_g, fixed_b;  // Fixed RGB values (0-255)
    int8_t base_slot;      // For bright colors: base slot index (-1 if none)
    float max_hue_drift;   // Max hue deviation from base color (degrees, 0 = unlimited)
};

// APCA pair constraint
struct ApcaPairConstraint {
    int8_t fg_index;       // Foreground color index (0-15)
    int8_t bg_index;       // Background color index (0-15)
    float min_apca;        // Minimum APCA contrast (absolute value)
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
    {  29,  20, 0.40, 0.70, 0.10, 0.30, false,   0,   0,   0, -1,  0},  // 1: RED
    { 142,  25, 0.45, 0.75, 0.10, 0.25, false,   0,   0,   0, -1,  0},  // 2: GREEN
    { 110,  20, 0.70, 0.90, 0.12, 0.25, false,   0,   0,   0, -1,  0},  // 3: YELLOW
    { 264,  25, 0.45, 0.70, 0.10, 0.25, false,   0,   0,   0, -1,  0},  // 4: BLUE
    { 328,  25, 0.50, 0.75, 0.12, 0.28, false,   0,   0,   0, -1,  0},  // 5: MAGENTA
    { 195,  25, 0.55, 0.80, 0.08, 0.20, false,   0,   0,   0, -1,  0},  // 6: CYAN
    {   0,   0, 0.75, 0.85, 0.00, 0.03, false,   0,   0,   0, -1,  0},  // 7: WHITE (neutral)

    // Bright colors (8-15)
    {   0,   0, 0.35, 0.45, 0.00, 0.03, false,   0,   0,   0, -1,  0},  // 8: BR_BLACK (neutral)
    {  29,  25, 0.60, 0.85, 0.12, 0.30, false,   0,   0,   0,  1, 20},  // 9: BR_RED (base=RED)
    { 142,  30, 0.65, 0.90, 0.12, 0.28, false,   0,   0,   0,  2, 20},  // 10: BR_GREEN (base=GREEN)
    { 110,  25, 0.85, 0.98, 0.12, 0.22, false,   0,   0,   0,  3, 15},  // 11: BR_YELLOW (base=YELLOW)
    { 264,  30, 0.60, 0.85, 0.10, 0.25, false,   0,   0,   0,  4, 25},  // 12: BR_BLUE (base=BLUE)
    { 328,  30, 0.65, 0.88, 0.12, 0.26, false,   0,   0,   0,  5, 20},  // 13: BR_MAGENTA (base=MAGENTA)
    { 195,  30, 0.70, 0.92, 0.08, 0.20, false,   0,   0,   0,  6, 20},  // 14: BR_CYAN (base=CYAN)
    {   0,   0, 1.00, 1.00, 0.00, 0.00, true,  255, 255, 255, -1,  0},  // 15: BR_WHITE (fixed)
};

// APCA pair constraints: {fg_index, bg_index, min_apca}
const ApcaPairConstraint apca_pair_constraints[] = {
    // Base colors on black (need APCA >= 40 for readable text)
    {RED,        BLACK, 40.0f},  // red on black
    {GREEN,      BLACK, 40.0f},  // green on black
    {YELLOW,     BLACK, 40.0f},  // yellow on black
    {BLUE,       BLACK, 40.0f},  // blue on black
    {MAGENTA,    BLACK, 40.0f},  // magenta on black
    {CYAN,       BLACK, 40.0f},  // cyan on black
    {WHITE,      BLACK, 40.0f},  // white on black

    // Bright colors on black (need APCA >= 40)
    {BR_RED,     BLACK, 40.0f},  // br.red on black
    {BR_GREEN,   BLACK, 40.0f},  // br.green on black
    {BR_YELLOW,  BLACK, 40.0f},  // br.yellow on black
    {BR_BLUE,    BLACK, 40.0f},  // br.blue on black
    {BR_MAGENTA, BLACK, 40.0f},  // br.magenta on black
    {BR_CYAN,    BLACK, 40.0f},  // br.cyan on black

    // br.black on black (lower threshold - subtle visibility)
    {BR_BLACK,   BLACK, 15.0f},  // br.black on black

    // Bright on corresponding base color (need APCA >= 30)
    {BR_RED,     RED,     30.0f},  // br.red on red
    {BR_GREEN,   GREEN,   30.0f},  // br.green on green
    {BR_YELLOW,  YELLOW,  30.0f},  // br.yellow on yellow
    {BR_BLUE,    BLUE,    30.0f},  // br.blue on blue
    {BR_MAGENTA, MAGENTA, 30.0f},  // br.magenta on magenta
    {BR_CYAN,    CYAN,    30.0f},  // br.cyan on cyan
    {BR_WHITE,   WHITE,   30.0f},  // br.white on white

    // Colors on cyan background (need APCA >= 20)
    {BLACK,   CYAN, 20.0f},  // black on cyan
    {RED,     CYAN, 20.0f},  // red on cyan
    {GREEN,   CYAN, 20.0f},  // green on cyan
    {YELLOW,  CYAN, 20.0f},  // yellow on cyan
    {BLUE,    CYAN, 20.0f},  // blue on cyan
    {MAGENTA, CYAN, 20.0f},  // magenta on cyan
    {WHITE,   CYAN, 20.0f},  // white on cyan

    // Colors on green background (need APCA >= 30)
    {BLACK,   GREEN, 30.0f},  // black on green
    {RED,     GREEN, 30.0f},  // red on green
    {YELLOW,  GREEN, 30.0f},  // yellow on green
    {BLUE,    GREEN, 30.0f},  // blue on green
    {MAGENTA, GREEN, 30.0f},  // magenta on green
    {CYAN,    GREEN, 30.0f},  // cyan on green
    {WHITE,   GREEN, 30.0f},  // white on green

    // Colors on blue background (need APCA >= 30)
    {BLACK,   BLUE, 30.0f},  // black on blue
    {RED,     BLUE, 30.0f},  // red on blue
    {GREEN,   BLUE, 30.0f},  // green on blue
    {YELLOW,  BLUE, 30.0f},  // yellow on blue
    {MAGENTA, BLUE, 30.0f},  // magenta on blue
    {CYAN,    BLUE, 30.0f},  // cyan on blue
    {WHITE,   BLUE, 30.0f},  // white on blue
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
__global__ void init_population(float* palettes, curandState* states, int n_palettes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_palettes) return;

    curandState localState = states[idx];

    for (int slot = 0; slot < 16; slot++) {
        OklchSlotConstraint c = d_oklch_slots[slot];
        int base = idx * 16 * 3 + slot * 3;

        if (c.fixed) {
            // Fixed RGB color - convert to OKLCH for storage
            float L, C, H;
            color::rgb_to_oklch(c.fixed_r, c.fixed_g, c.fixed_b, &L, &C, &H);
            palettes[base + 0] = L;
            palettes[base + 1] = C;
            palettes[base + 2] = H;
        } else {
            // Random OKLCH within constraints
            float L = c.min_L + curand_uniform(&localState) * (c.max_L - c.min_L);
            float H = c.target_hue + (curand_uniform(&localState) - 0.5f) * 2.0f * c.hue_tolerance;
            H = color::oklch::normalize_hue(H);

            // Determine chroma range, clamped to gamut
            float max_C = color::oklch_max_chroma(L, H);
            float C_min = fminf(c.min_C, max_C);
            float C_max = fminf(c.max_C, max_C);

            float C = C_min + curand_uniform(&localState) * (C_max - C_min);

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
__global__ void evaluate_fitness(float* palettes, float* fitness, int n_palettes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_palettes) return;

    float score = 0.0f;
    int base = idx * 16 * 3;

    // Convert OKLCH palette to RGB (cache for reuse)
    float rgb[16][3];
    for (int i = 0; i < 16; i++) {
        float L = palettes[base + i * 3 + 0];
        float C = palettes[base + i * 3 + 1];
        float H = palettes[base + i * 3 + 2];
        color::oklch_to_srgb(L, C, H, &rgb[i][0], &rgb[i][1], &rgb[i][2]);
    }

    // =========================================================================
    // CONSTRAINT 1: APCA pair constraints (hard requirements)
    // =========================================================================
    for (int i = 0; i < d_apca_pair_count; i++) {
        ApcaPairConstraint p = d_apca_pairs[i];
        int fg = p.fg_index;
        int bg = p.bg_index;

        float apca = color::apca::contrast_abs(
            rgb[fg][0], rgb[fg][1], rgb[fg][2],
            rgb[bg][0], rgb[bg][1], rgb[bg][2]
        );

        if (apca >= p.min_apca) {
            // Constraint met - reward with bonus for exceeding minimum
            score += 100.0f + (apca - p.min_apca) * 5.0f;
        } else {
            // Constraint violated - heavy penalty proportional to shortfall
            score -= (p.min_apca - apca) * 50.0f;
        }
    }

    // =========================================================================
    // CONSTRAINT 2: Hue drift for bright colors (must match base)
    // =========================================================================
    for (int slot = 8; slot <= 14; slot++) {
        OklchSlotConstraint c = d_oklch_slots[slot];
        if (c.base_slot < 0 || c.max_hue_drift <= 0.0f) continue;

        float H_bright = palettes[base + slot * 3 + 2];
        float H_base = palettes[base + c.base_slot * 3 + 2];

        float hdist = color::hue_distance(H_bright, H_base);
        if (hdist <= c.max_hue_drift) {
            score += 30.0f;  // Bonus for matching hue
        } else {
            score -= (hdist - c.max_hue_drift) * 5.0f;  // Penalty for drift
        }
    }

    // =========================================================================
    // CONSTRAINT 3: Gamut validity
    // =========================================================================
    for (int i = 0; i < 16; i++) {
        float L = palettes[base + i * 3 + 0];
        float C = palettes[base + i * 3 + 1];
        float H = palettes[base + i * 3 + 2];

        if (!color::oklch_in_gamut(L, C, H)) {
            score -= 500.0f;  // Heavy penalty for out-of-gamut
        }
    }

    // =========================================================================
    // BONUS 1: Hue spacing for base colors (1-6)
    // Reward evenly spaced hues on the color wheel
    // =========================================================================
    {
        float hues[6];
        for (int i = 0; i < 6; i++) {
            hues[i] = palettes[base + (i + 1) * 3 + 2];
        }

        float min_hue_dist = 360.0f;
        for (int i = 0; i < 6; i++) {
            for (int j = i + 1; j < 6; j++) {
                float hdist = color::hue_distance(hues[i], hues[j]);
                if (hdist < min_hue_dist) {
                    min_hue_dist = hdist;
                }
            }
        }

        // Ideal minimum spacing for 6 colors is 60° but we accept 40°
        if (min_hue_dist >= 40.0f) {
            score += min_hue_dist * 2.0f;
        } else {
            score -= (40.0f - min_hue_dist) * 5.0f;
        }
    }

    // =========================================================================
    // BONUS 2: Chroma (prefer more saturated colors)
    // =========================================================================
    for (int i = 1; i <= 6; i++) {  // Base colors only
        float C = palettes[base + i * 3 + 1];
        score += C * 50.0f;  // Small bonus for saturation
    }

    // =========================================================================
    // BONUS 3: Perceptual distance (Oklab) between base colors
    // =========================================================================
    {
        float min_dist = 1000.0f;
        for (int i = 1; i <= 7; i++) {
            for (int j = i + 1; j <= 7; j++) {
                float dist = color::oklab_distance(
                    rgb[i][0], rgb[i][1], rgb[i][2],
                    rgb[j][0], rgb[j][1], rgb[j][2]
                );
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
        }

        // Reward good separation (0.15 is noticeable difference)
        if (min_dist >= 0.15f) {
            score += min_dist * 100.0f;
        } else {
            score -= (0.15f - min_dist) * 300.0f;
        }
    }

    // =========================================================================
    // BONUS 4: All other APCA pairs (soft bonus for general readability)
    // =========================================================================
    // Small bonus for any pair with good APCA (not covered by constraints)
    for (int bg = 0; bg < 8; bg++) {
        for (int fg = 0; fg < 16; fg++) {
            if (fg == bg) continue;
            float apca = color::apca::contrast_abs(
                rgb[fg][0], rgb[fg][1], rgb[fg][2],
                rgb[bg][0], rgb[bg][1], rgb[bg][2]
            );
            if (apca >= 40.0f) {
                score += 1.0f;  // Small bonus for readable pairs
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
    float* old_pop, float* new_pop, float* fitness,
    int* elite_indices, int elite_count,
    curandState* states, float mutation_rate, int n_palettes
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
    int p1_idx = elite_indices[(int)(curand_uniform(&localState) * elite_count)];
    int p2_idx = elite_indices[(int)(curand_uniform(&localState) * elite_count)];

    int p1_base = p1_idx * 16 * 3;
    int p2_base = p2_idx * 16 * 3;

    // Crossover and mutate each color slot
    for (int slot = 0; slot < 16; slot++) {
        OklchSlotConstraint c = d_oklch_slots[slot];
        int offset = slot * 3;

        if (c.fixed) {
            // Fixed color - convert from RGB
            float L, C, H;
            color::rgb_to_oklch(c.fixed_r, c.fixed_g, c.fixed_b, &L, &C, &H);
            new_pop[new_base + offset + 0] = L;
            new_pop[new_base + offset + 1] = C;
            new_pop[new_base + offset + 2] = H;
        } else {
            // Crossover: blend or select
            float L1 = old_pop[p1_base + offset + 0];
            float C1 = old_pop[p1_base + offset + 1];
            float H1 = old_pop[p1_base + offset + 2];
            float L2 = old_pop[p2_base + offset + 0];
            float C2 = old_pop[p2_base + offset + 1];
            float H2 = old_pop[p2_base + offset + 2];

            float t = curand_uniform(&localState);
            float L = L1 + t * (L2 - L1);
            float C = C1 + t * (C2 - C1);
            float H = color::oklch::lerp_hue(H1, H2, t);

            // Mutation
            if (curand_uniform(&localState) < mutation_rate) {
                // Mutate L
                float L_range = c.max_L - c.min_L;
                L += curand_normal(&localState) * L_range * 0.1f;
                if (L < c.min_L) L = c.min_L;
                if (L > c.max_L) L = c.max_L;
            }

            if (curand_uniform(&localState) < mutation_rate) {
                // Mutate H (circular)
                H += curand_normal(&localState) * c.hue_tolerance * 0.3f;
                H = color::oklch::normalize_hue(H);

                // Clamp to constraint range
                float target = c.target_hue;
                float hdist = color::hue_distance(H, target);
                if (hdist > c.hue_tolerance) {
                    // Push back toward valid range
                    float t_factor = c.hue_tolerance / hdist;
                    H = color::oklch::lerp_hue(target, H, t_factor);
                }
            }

            if (curand_uniform(&localState) < mutation_rate) {
                // Mutate C
                float C_range = c.max_C - c.min_C;
                C += curand_normal(&localState) * C_range * 0.15f;

                // Clamp to constraint range and gamut
                float max_C = color::oklch_max_chroma(L, H);
                if (C < c.min_C) C = c.min_C;
                if (C > c.max_C) C = c.max_C;
                if (C > max_C) C = max_C;
            }

            // Ensure gamut validity
            float max_C = color::oklch_max_chroma(L, H);
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
void oklch_palette_to_rgb(float* oklch_palette, float* rgb_palette) {
    for (int i = 0; i < 16; i++) {
        float L = oklch_palette[i * 3 + 0];
        float C = oklch_palette[i * 3 + 1];
        float H = oklch_palette[i * 3 + 2];
        color::oklch_to_srgb(L, C, H, &rgb_palette[i * 3 + 0], &rgb_palette[i * 3 + 1], &rgb_palette[i * 3 + 2]);
    }
}

// =============================================================================
// Host Functions (Reporting)
// =============================================================================

const char* apca_ansi_color(float lc) {
    float abs_lc = fabsf(lc);
    if (abs_lc >= 75.0f) return "\033[32m";  // Green - body text
    if (abs_lc >= 60.0f) return "\033[33m";  // Yellow - large text
    if (abs_lc >= 45.0f) return "\033[38;2;255;165;0m";  // Orange - bold only
    return "\033[31m";                        // Red - insufficient
}

void print_color_demo(float* palette) {
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

    printf("APCA Legend: \033[32m≥75 body\033[0m  \033[33m≥60 large\033[0m  \033[38;2;255;165;0m≥45 bold\033[0m  \033[31m<45 insufficient\033[0m\n\n");

    // APCA contrast check against black background
    printf("APCA Contrast (colors on black background):\n");
    printf("────────────────────────────────────────────────────────────────\n");

    float black_r = palette[BLACK * 3 + 0];
    float black_g = palette[BLACK * 3 + 1];
    float black_b = palette[BLACK * 3 + 2];

    for (int i = 1; i <= 14; i++) {
        if (i == 8) continue;  // Skip br.black
        float r = palette[i * 3 + 0];
        float g = palette[i * 3 + 1];
        float b = palette[i * 3 + 2];
        float lc = color::apca::contrast(r, g, b, black_r, black_g, black_b);
        float abs_lc = fabsf(lc);

        float required = 40.0f;
        const char* status = abs_lc >= required ? "\033[32m✓\033[0m" : "\033[31m✗\033[0m";
        printf("  %-12s Lc=%6.1f (min 40)  %s\n", names[i], abs_lc, status);
    }

    // Br.black on black (special case - lower requirement)
    {
        float r = palette[BR_BLACK * 3 + 0];
        float g = palette[BR_BLACK * 3 + 1];
        float b = palette[BR_BLACK * 3 + 2];
        float lc = color::apca::contrast(r, g, b, black_r, black_g, black_b);
        float abs_lc = fabsf(lc);
        const char* status = abs_lc >= 15.0f ? "\033[32m✓\033[0m" : "\033[31m✗\033[0m";
        printf("  %-12s Lc=%6.1f (min 15)  %s\n", names[BR_BLACK], abs_lc, status);
    }

    // OKLCH Analysis
    printf("\nOKLCH Analysis (Perceptually Uniform Color Space):\n");
    printf("────────────────────────────────────────────────────────────────\n");
    printf("%-12s  L (light)  C (chroma)  H (hue°)\n", "Color");
    printf("────────────────────────────────────────────────────────────────\n");

    float base_hues[6];
    for (int i = 1; i <= 6; i++) {
        float r = palette[i * 3 + 0];
        float g = palette[i * 3 + 1];
        float b = palette[i * 3 + 2];
        float L, C, H;
        color::rgb_to_oklch(r, g, b, &L, &C, &H);
        base_hues[i-1] = H;
        printf("%-12s  %5.3f      %5.3f       %6.1f°\n", names[i], L, C, H);
    }

    // Hue spacing analysis
    printf("\nHue Spacing (ideal: 60° between colors):\n");
    printf("────────────────────────────────────────────────────────────────\n");
    float min_hue_dist = 360.0f;
    const char* min_pair_a = "";
    const char* min_pair_b = "";
    for (int i = 0; i < 6; i++) {
        for (int j = i + 1; j < 6; j++) {
            float hdist = color::hue_distance(base_hues[i], base_hues[j]);
            if (hdist < min_hue_dist) {
                min_hue_dist = hdist;
                min_pair_a = names[i + 1];
                min_pair_b = names[j + 1];
            }
        }
    }
    printf("  Minimum hue distance: %.1f° (between %s and %s)\n", min_hue_dist, min_pair_a, min_pair_b);
    if (min_hue_dist >= 50.0f) {
        printf("  Status: \033[32m✓ Good spacing\033[0m\n");
    } else if (min_hue_dist >= 30.0f) {
        printf("  Status: \033[33m~ Acceptable spacing\033[0m\n");
    } else {
        printf("  Status: \033[31m✗ Colors too close in hue\033[0m\n");
    }

    // Bright color hue matching
    printf("\nBright/Base Hue Matching (bright should match base hue):\n");
    printf("────────────────────────────────────────────────────────────────\n");
    for (int i = 1; i <= 6; i++) {
        float L1, C1, H1, L2, C2, H2;
        color::rgb_to_oklch(palette[i * 3 + 0], palette[i * 3 + 1], palette[i * 3 + 2], &L1, &C1, &H1);
        color::rgb_to_oklch(palette[(i + 8) * 3 + 0], palette[(i + 8) * 3 + 1], palette[(i + 8) * 3 + 2], &L2, &C2, &H2);
        float hdist = color::hue_distance(H1, H2);
        printf("  %-10s → br.%-7s: ΔH=%5.1f°  %s\n",
               names[i], names[i], hdist,
               hdist <= 30.0f ? "\033[32m✓\033[0m" : "\033[31m✗ hue drift\033[0m");
    }
}

// =============================================================================
// Main
// =============================================================================

// Write theme to file
void write_theme_file(float* palette, const char* filepath) {
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
    float mutation_rate = 0.15f;
    float elite_ratio = 0.1f;
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
    size_t palette_size = population_size * 16 * 3 * sizeof(float);
    float *d_pop1, *d_pop2, *d_fitness;
    curandState* d_states;
    int* d_elite_indices;

    cudaMalloc(&d_pop1, palette_size);
    cudaMalloc(&d_pop2, palette_size);
    cudaMalloc(&d_fitness, population_size * sizeof(float));
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
    std::vector<float> h_fitness(population_size);
    std::vector<int> h_elite_indices(elite_count);

    // Best-ever tracking (don't rely solely on elitism)
    float best_ever_fitness = -1e9f;
    std::vector<float> best_ever_palette(16 * 3);
    int best_ever_generation = 0;

    int stagnant_generations = 0;
    float current_mutation = mutation_rate;

    printf("Starting evolution...\n\n");

    for (int gen = 0; gen < generations; gen++) {
        // Evaluate fitness
        evaluate_fitness<<<numBlocks, blockSize>>>(d_pop1, d_fitness, population_size);
        cudaDeviceSynchronize();

        // Copy fitness to host
        cudaMemcpy(h_fitness.data(), d_fitness, population_size * sizeof(float), cudaMemcpyDeviceToHost);

        // Find elite indices
        std::vector<int> indices(population_size);
        for (int i = 0; i < population_size; i++) indices[i] = i;

        std::partial_sort(indices.begin(), indices.begin() + elite_count, indices.end(),
            [&h_fitness](int a, int b) { return h_fitness[a] > h_fitness[b]; });

        for (int i = 0; i < elite_count; i++) {
            h_elite_indices[i] = indices[i];
        }

        float gen_best = h_fitness[indices[0]];

        // Track best-ever palette
        if (gen_best > best_ever_fitness) {
            best_ever_fitness = gen_best;
            best_ever_generation = gen;
            // Save the best palette from device
            int best_idx = indices[0];
            cudaMemcpy(best_ever_palette.data(), d_pop1 + best_idx * 16 * 3,
                       16 * 3 * sizeof(float), cudaMemcpyDeviceToHost);
            stagnant_generations = 0;
            current_mutation = mutation_rate;
        } else {
            stagnant_generations++;
            if (stagnant_generations > 100) {
                current_mutation = fminf(0.5f, current_mutation * 1.01f);
            }
        }

        // Progress output
        if (gen % 500 == 0 || gen == generations - 1) {
            printf("Gen %5d: best=%.2f, avg=%.2f, mutation=%.3f\n",
                   gen, gen_best,
                   std::accumulate(h_fitness.begin(), h_fitness.end(), 0.0f) / population_size,
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
    std::vector<float> rgb_palette(16 * 3);
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
