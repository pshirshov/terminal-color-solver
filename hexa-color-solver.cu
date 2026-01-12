/**
 * Hexa Color Solver v2.5 - Golden yellow, brighter red
 *
 * Uses genetic algorithm to find optimal 16-color terminal palettes.
 *
 * Constraints:
 *   1. Base colors (1-7) on black with PER-COLOR minimums:
 *      - Red >= 5.5 (force brighter)
 *      - Green >= 4.5, White >= 4.5 (needs higher contrast)
 *      - Yellow, Blue, Magenta, Cyan >= 3.5
 *      - All colors <= 7.5
 *   2. Bright on regular (br.X on X): CR >= 2.6 (br.black >= 2.2)
 *   3. FM pairs on blue (red, green, yellow, cyan, white): CR >= 3.0
 *   4. FM pairs on green (red, yellow, blue, magenta, white): CR >= 2.5
 *
 * Fixed colors:
 *   - Black: #000000
 *   - Br.White: #ffffff
 *
 * Perceptual considerations (Helmholtz-Kohlrausch effect):
 *   - Saturated red/blue/magenta appear brighter than luminance suggests
 *   - Yellow/green appear closer to their calculated luminance
 *   - Blue hues are poorly predicted by standard color spaces
 *
 * Build: cmake build
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
// Constraints
// =============================================================================
#define MAX_BASE_ON_BLACK 7.5f
#define MIN_BRIGHT_ON_REGULAR 2.6f
#define MIN_BR_BLACK_ON_BLACK 2.2f
#define MIN_ON_BLUE 2.5f
#define MIN_ON_GREEN 2.5f
#define MIN_ON_CYAN 2.5f

// Runtime flags (set via command line)
__constant__ bool d_enable_fm_pairs = true;
__constant__ bool d_enable_gb_exclusions = true;

// Per-color minimum contrast on black background
// Red and green get higher minimums since they appear darker perceptually
// Order: RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE (indices 1-7)
__constant__ float d_min_contrast[7] = {
    5.5f,   // RED - force brighter red
    4.5f,   // GREEN - needs higher minimum
    3.5f,   // YELLOW - fine as is
    4.5f,   // BLUE - needs higher minimum
    3.5f,   // MAGENTA - fine as is
    3.5f,   // CYAN - fine as is
    4.5f,   // WHITE - should have good contrast
};

// Host-side copy for printing
const float h_min_contrast[7] = {5.5f, 4.5f, 3.5f, 4.5f, 3.5f, 3.5f, 4.5f};

// =============================================================================
// Color character definitions
// =============================================================================
// Each color has RGB ranges that define its "character" - what makes it look
// like that color. The optimizer searches within these ranges.
//
// Perceptual notes (Helmholtz-Kohlrausch effect):
// - Saturated red/blue/magenta appear ~10-20% brighter than luminance predicts
// - Yellow/green track closer to calculated luminance
// - We compensate by allowing slightly lower luminance for red/blue/magenta

struct ColorRange {
    float r_min, r_max;
    float g_min, g_max;
    float b_min, b_max;
    bool fixed;
};

// Fixed colors
#define BLACK_RGB     0,   0,   0,   0,   0,   0, true
// White and Br.Black are now optimizable
//#define WHITE_RGB   191, 191, 191, 191, 191, 191, true   // #bfbfbf
//#define BR_BLACK_RGB 64,  64,  64,  64,  64,  64, true   // #404040
#define BR_WHITE_RGB 255, 255, 255, 255, 255, 255, true  // #ffffff

// Color character ranges
// Format: r_min, r_max, g_min, g_max, b_min, b_max, fixed
const ColorRange color_ranges[16] = {
    // Base colors (indices 0-7)
    {BLACK_RGB},                                    // 0: BLACK
    {140, 255,   0, 150,   0, 150, false},          // 1: RED - allow coral/salmon/orange for brightness
    {  0, 120,  80, 255,   0, 120, false},          // 2: GREEN - allow some R/B for flexibility
    {200, 255, 160, 220,   0,  20, false},          // 3: YELLOW - golden: high R, capped G, minimal B
    {  0, 160,   0, 240, 120, 255, false},          // 4: BLUE - wide range for brightness (can go cyan-ish)
    {140, 255,   0, 180, 140, 255, false},          // 5: MAGENTA - allow pink/lavender for brightness
    {  0, 100,  80, 255,  80, 255, false},          // 6: CYAN - low R, med-high G, med-high B
    {180, 220, 180, 220, 180, 220, false},          // 7: WHITE - optimizable

    // Bright colors (indices 8-15)
    { 50,  80,  50,  80,  50,  80, false},          // 8: BR_BLACK - optimizable
    {180, 255,  80, 200,  80, 200, false},          // 9: BR_RED - brighter red (more flexibility)
    { 80, 200, 160, 255,  80, 200, false},          // 10: BR_GREEN - brighter green (more flexibility)
    {180, 255, 180, 255,  80, 180, false},          // 11: BR_YELLOW - brighter yellow
    { 80, 200, 120, 255, 160, 255, false},          // 12: BR_BLUE - brighter blue (wide range)
    {180, 255,  80, 200, 180, 255, false},          // 13: BR_MAGENTA - brighter magenta (more flexibility)
    { 80, 180, 180, 255, 180, 255, false},          // 14: BR_CYAN - brighter cyan
    {BR_WHITE_RGB},                                 // 15: BR_WHITE
};

// Device constant memory for color ranges
__constant__ ColorRange d_ranges[16];

// =============================================================================
// Color Functions (from color.cuh)
// =============================================================================
// All color conversion and contrast functions are now provided by color.cuh:
// - color::wcag2::* - WCAG 2.1 contrast ratio
// - color::apca::* - APCA (WCAG 3.0) contrast
// - color::oklab::* - Oklab perceptually uniform color space
// - color::oklch::* - OKLCH (polar form of Oklab)
//
// Legacy interface functions (color::linearize, color::luminance, etc.) are
// also available for backward compatibility.

using color::linearize;
using color::luminance;
using color::contrast_ratio;
using color::rgb_to_oklab;
using color::rgb_to_oklch;
using color::oklab_distance;
using color::hue_distance;

// =============================================================================
// CUDA Kernels
// =============================================================================

__global__ void init_curand(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void init_population(float* palettes, curandState* states, int n_palettes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_palettes) return;

    curandState localState = states[idx];

    for (int color = 0; color < 16; color++) {
        ColorRange range = d_ranges[color];
        int base = idx * 16 * 3 + color * 3;

        if (range.fixed) {
            palettes[base + 0] = range.r_min;
            palettes[base + 1] = range.g_min;
            palettes[base + 2] = range.b_min;
        } else {
            palettes[base + 0] = range.r_min + curand_uniform(&localState) * (range.r_max - range.r_min);
            palettes[base + 1] = range.g_min + curand_uniform(&localState) * (range.g_max - range.g_min);
            palettes[base + 2] = range.b_min + curand_uniform(&localState) * (range.b_max - range.b_min);
        }
    }

    states[idx] = localState;
}

__global__ void evaluate_fitness(float* palettes, float* fitness, int n_palettes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_palettes) return;

    float score = 0.0f;
    int base = idx * 16 * 3;

    // =========================================================================
    // CONSTRAINT 1: Base colors (1-7) on black with per-color minimums
    // Reward higher contrast (not middle of range) to push colors brighter
    // =========================================================================
    for (int fg = 1; fg <= 7; fg++) {
        int fg_base = base + fg * 3;
        int bg_base = base + 0 * 3;  // Black

        float cr = contrast_ratio(
            palettes[fg_base + 0], palettes[fg_base + 1], palettes[fg_base + 2],
            palettes[bg_base + 0], palettes[bg_base + 1], palettes[bg_base + 2]
        );

        // Get per-color minimum (fg index 1-6 maps to d_min_contrast[0-5])
        float min_cr = d_min_contrast[fg - 1];

        if (cr >= min_cr && cr <= MAX_BASE_ON_BLACK) {
            // Within range - reward higher contrast directly (PRIMARY)
            score += 100.0f + cr * 25.0f;  // Higher contrast = more points
        } else if (cr < min_cr) {
            // Too dark - very heavy penalty (PRIMARY CONSTRAINT)
            score -= (min_cr - cr) * 500.0f;
        } else {
            // Too bright - lighter penalty
            score -= (cr - MAX_BASE_ON_BLACK) * 50.0f;
        }
    }

    // =========================================================================
    // CONSTRAINT 2: Bright on regular
    // =========================================================================
    // Iterate over all pairs: 1-6 (colors), 0 (black), 7 (white)
    // Indices: Reg 0..7, Bright 8..15
    for (int i = 0; i <= 7; i++) {
        int reg_base = base + i * 3;
        int brt_base = base + (i + 8) * 3;

        float cr = contrast_ratio(
            palettes[brt_base + 0], palettes[brt_base + 1], palettes[brt_base + 2],
            palettes[reg_base + 0], palettes[reg_base + 1], palettes[reg_base + 2]
        );

        // Determine target
        float target = MIN_BRIGHT_ON_REGULAR;
        if (i == 0) { // Black
            target = MIN_BR_BLACK_ON_BLACK;
        }

        if (cr >= target) {
            score += 20.0f + cr * 2.0f;  // Lower priority
        } else {
            score -= (target - cr) * 100.0f;  // Reduced penalty
        }
    }

    // =========================================================================
    // CONSTRAINT 3: FM pairs on blue (PRIMARY) - can be disabled with --no-fm-pairs
    // =========================================================================
    if (d_enable_fm_pairs) {
        int blue_base = base + BLUE * 3;
        float blue_r = palettes[blue_base + 0];
        float blue_g = palettes[blue_base + 1];
        float blue_b = palettes[blue_base + 2];

        // Colors that should be readable on blue: RED(1), GREEN(2), YELLOW(3), MAGENTA(5), CYAN(6), WHITE(7)
        int fg_on_blue[] = {RED, GREEN, YELLOW, MAGENTA, CYAN, WHITE};
        for (int i = 0; i < 6; i++) {
            int fg = fg_on_blue[i];
            int fg_base = base + fg * 3;

            float cr = contrast_ratio(
                palettes[fg_base + 0], palettes[fg_base + 1], palettes[fg_base + 2],
                blue_r, blue_g, blue_b
            );

            if (cr >= MIN_ON_BLUE) {
                score += 80.0f + cr * 15.0f;  // PRIMARY - high reward
            } else {
                score -= (MIN_ON_BLUE - cr) * 400.0f;  // PRIMARY - heavy penalty
            }
        }
    }

    // =========================================================================
    // CONSTRAINT 4: FM pairs on green - can be disabled with --no-fm-pairs
    // =========================================================================
    if (d_enable_fm_pairs) {
        int green_base = base + GREEN * 3;
        float green_r = palettes[green_base + 0];
        float green_g = palettes[green_base + 1];
        float green_b = palettes[green_base + 2];

        // Colors that should be readable on green: RED(1), YELLOW(3), BLUE(4), MAGENTA(5), CYAN(6), WHITE(7)
        int fg_on_green[] = {RED, YELLOW, BLUE, MAGENTA, CYAN, WHITE};
        for (int i = 0; i < 6; i++) {
            int fg = fg_on_green[i];
            int fg_base = base + fg * 3;

            float cr = contrast_ratio(
                palettes[fg_base + 0], palettes[fg_base + 1], palettes[fg_base + 2],
                green_r, green_g, green_b
            );

            if (cr >= MIN_ON_GREEN) {
                score += 60.0f + cr * 12.0f;  // PRIMARY - high reward
            } else {
                score -= (MIN_ON_GREEN - cr) * 350.0f;  // PRIMARY - heavy penalty
            }
        }
    }

    // =========================================================================
    // BONUS: Color distinctiveness (optionally excluding green and blue)
    // =========================================================================
    // Reward pairs of different base colors being distinguishable
    for (int i = 1; i <= 7; i++) {
        if (d_enable_gb_exclusions && (i == GREEN || i == BLUE)) continue;
        for (int j = i + 1; j <= 7; j++) {
            if (d_enable_gb_exclusions && (j == GREEN || j == BLUE)) continue;
            int i_base = base + i * 3;
            int j_base = base + j * 3;

            float cr = contrast_ratio(
                palettes[i_base + 0], palettes[i_base + 1], palettes[i_base + 2],
                palettes[j_base + 0], palettes[j_base + 1], palettes[j_base + 2]
            );

            // Small bonus for inter-color contrast (helps distinguish colors)
            score += cr * 2.0f;
        }
    }

    // =========================================================================
    // BONUS: Bright colors on black
    // =========================================================================
    for (int fg = 8; fg <= 14; fg++) {
        int fg_base = base + fg * 3;
        int bg_base = base + 0 * 3;

        float cr = contrast_ratio(
            palettes[fg_base + 0], palettes[fg_base + 1], palettes[fg_base + 2],
            palettes[bg_base + 0], palettes[bg_base + 1], palettes[bg_base + 2]
        );

        // Bright colors should have good contrast on black
        if (cr >= 7.0f) {
            score += 20.0f;
        }
        score += cr * 1.0f;
    }

    // =========================================================================
    // OKLCH: Hue spacing for base colors (1-6), optionally excluding green and blue
    // =========================================================================
    // Ideal: evenly spaced hues (60° for 6 colors, 90° for 4 colors)
    // We reward minimum hue distance between any two colors
    {
        float hues[6];
        float chromas[6];
        float lightnesses[6];

        // Extract OKLCH values for base colors (1-6)
        for (int i = 0; i < 6; i++) {
            int c_base = base + (i + 1) * 3;
            rgb_to_oklch(
                palettes[c_base + 0], palettes[c_base + 1], palettes[c_base + 2],
                &lightnesses[i], &chromas[i], &hues[i]
            );
        }

        // Reward good hue spacing between all pairs (optionally excluding green=1, blue=3)
        float min_hue_dist = 360.0f;
        for (int i = 0; i < 6; i++) {
            if (d_enable_gb_exclusions && (i == 1 || i == 3)) continue;
            for (int j = i + 1; j < 6; j++) {
                if (d_enable_gb_exclusions && (j == 1 || j == 3)) continue;
                float hdist = hue_distance(hues[i], hues[j]);
                if (hdist < min_hue_dist) {
                    min_hue_dist = hdist;
                }
            }
        }

        // Reward getting close to ideal spacing
        if (min_hue_dist >= 30.0f) {
            score += min_hue_dist * 2.0f;
        } else {
            score -= (30.0f - min_hue_dist) * 5.0f;  // Penalty for too close
        }

        // Bonus for good chroma (saturation) - avoid washed out colors
        for (int i = 0; i < 6; i++) {
            if (d_enable_gb_exclusions && (i == 1 || i == 3)) continue;
            if (chromas[i] >= 0.1f) {
                score += chromas[i] * 50.0f;  // Reward saturation
            }
        }
    }

    // =========================================================================
    // OKLCH: Perceptual distance between base colors (optionally excluding green and blue)
    // =========================================================================
    // Use Oklab distance for better perceptual uniformity than contrast ratio
    {
        float min_oklab_dist = 1000.0f;
        for (int i = 1; i <= 7; i++) {
            if (d_enable_gb_exclusions && (i == GREEN || i == BLUE)) continue;
            for (int j = i + 1; j <= 7; j++) {
                if (d_enable_gb_exclusions && (j == GREEN || j == BLUE)) continue;
                int i_base = base + i * 3;
                int j_base = base + j * 3;

                float dist = oklab_distance(
                    palettes[i_base + 0], palettes[i_base + 1], palettes[i_base + 2],
                    palettes[j_base + 0], palettes[j_base + 1], palettes[j_base + 2]
                );

                if (dist < min_oklab_dist) {
                    min_oklab_dist = dist;
                }
            }
        }

        // Reward minimum perceptual distance (0.15 is good separation in Oklab)
        if (min_oklab_dist >= 0.15f) {
            score += min_oklab_dist * 200.0f;
        } else {
            score -= (0.15f - min_oklab_dist) * 500.0f;  // Heavy penalty for similar colors
        }
    }

    // =========================================================================
    // OKLCH: Bright colors should match hue of their base counterparts (optionally excluding green and blue)
    // =========================================================================
    {
        for (int i = 1; i <= 6; i++) {
            if (d_enable_gb_exclusions && (i == GREEN || i == BLUE)) continue;
            int base_idx = base + i * 3;
            int bright_idx = base + (i + 8) * 3;

            float L1, C1, H1, L2, C2, H2;
            rgb_to_oklch(palettes[base_idx + 0], palettes[base_idx + 1], palettes[base_idx + 2],
                         &L1, &C1, &H1);
            rgb_to_oklch(palettes[bright_idx + 0], palettes[bright_idx + 1], palettes[bright_idx + 2],
                         &L2, &C2, &H2);

            // Reward hue similarity between base and bright version
            float hdist = hue_distance(H1, H2);
            if (hdist <= 30.0f) {
                score += (30.0f - hdist) * 2.0f;  // Up to 60 points for matching hue
            } else {
                score -= (hdist - 30.0f) * 3.0f;  // Penalty for hue drift
            }

            // Bright should have higher lightness
            if (L2 > L1) {
                score += 20.0f;
            }
        }
    }

    fitness[idx] = score;
}

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

    // Crossover and mutate each color
    for (int color = 0; color < 16; color++) {
        ColorRange range = d_ranges[color];
        int color_offset = color * 3;

        if (range.fixed) {
            new_pop[new_base + color_offset + 0] = range.r_min;
            new_pop[new_base + color_offset + 1] = range.g_min;
            new_pop[new_base + color_offset + 2] = range.b_min;
        } else {
            // Uniform crossover
            for (int c = 0; c < 3; c++) {
                float val;
                if (curand_uniform(&localState) < 0.5f) {
                    val = old_pop[p1_base + color_offset + c];
                } else {
                    val = old_pop[p2_base + color_offset + c];
                }

                // Mutation
                if (curand_uniform(&localState) < mutation_rate) {
                    float range_min, range_max;
                    if (c == 0) { range_min = range.r_min; range_max = range.r_max; }
                    else if (c == 1) { range_min = range.g_min; range_max = range.g_max; }
                    else { range_min = range.b_min; range_max = range.b_max; }

                    // Gaussian mutation
                    float range_size = range_max - range_min;
                    val += curand_normal(&localState) * range_size * 0.1f;

                    // Clamp to range
                    if (val < range_min) val = range_min;
                    if (val > range_max) val = range_max;
                }

                new_pop[new_base + color_offset + c] = val;
            }
        }
    }

    states[idx] = localState;
}

// =============================================================================
// Host Functions (Reporting)
// =============================================================================

const char* wcag_ansi_color(float cr) {
    if (cr >= 4.5f) return "\033[32m";  // Green for AA
    if (cr >= 3.0f) return "\033[33m";  // Yellow for OK
    return "\033[31m";                   // Red for BAD
}

void print_color_demo(float* palette) {
    const char* names[] = {
        "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white",
        "br.black", "br.red", "br.green", "br.yellow", "br.blue", "br.magenta", "br.cyan", "br.white"
    };

    // Extract key background colors
    int black_r = (int)palette[BLACK * 3 + 0];
    int black_g = (int)palette[BLACK * 3 + 1];
    int black_b = (int)palette[BLACK * 3 + 2];

    printf("\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("                              COLOR PALETTE RESULTS\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n\n");

    // Palette table and sample matrix side by side (using FTXUI)
    output::print_palette_and_matrix_side_by_side(palette, names);

    printf("Legend: \033[32m>=4.5 AA\033[0m  \033[33m>=3.0 OK\033[0m  \033[31m<3.0 BAD\033[0m\n\n");

    // Contrast tables side by side (using FTXUI)
    output::print_contrast_tables_side_by_side(
        palette, names,
        MIN_BRIGHT_ON_REGULAR, MIN_BR_BLACK_ON_BLACK,
        MIN_ON_BLUE, MIN_ON_GREEN, MIN_ON_CYAN
    );

    // Constraint check summary
    printf("Constraint Check:\n");
    printf("────────────────────────────────────────────────────────────────\n");

    // Base colors on black (with per-color minimums)
    printf("Base colors (1-7) on black (per-color min, max=%.1f):\n", MAX_BASE_ON_BLACK);
    for (int i = 1; i <= 7; i++) {
        int r = (int)palette[i * 3 + 0];
        int g = (int)palette[i * 3 + 1];
        int b = (int)palette[i * 3 + 2];
        float cr = contrast_ratio(r, g, b, black_r, black_g, black_b);
        float min_cr = h_min_contrast[i - 1];

        const char* status;
        if (cr >= min_cr && cr <= MAX_BASE_ON_BLACK) {
            status = "\033[32m✓\033[0m";
        } else if (cr < min_cr) {
            status = "\033[31m✗ too dark\033[0m";
        } else {
            status = "\033[33m✗ too bright\033[0m";
        }
        printf("  %-10s %5.2f:1 (min %.1f)  %s\n", names[i], cr, min_cr, status);
    }

    // OKLCH Analysis
    printf("\nOKLCH Analysis (Perceptually Uniform Color Space):\n");
    printf("────────────────────────────────────────────────────────────────\n");
    printf("%-12s  L (light)  C (chroma)  H (hue°)\n", "Color");
    printf("────────────────────────────────────────────────────────────────\n");

    float base_hues[6];
    for (int i = 1; i <= 6; i++) {
        int r = (int)palette[i * 3 + 0];
        int g = (int)palette[i * 3 + 1];
        int b = (int)palette[i * 3 + 2];
        float L, C, H;
        rgb_to_oklch(r, g, b, &L, &C, &H);
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
            float hdist = hue_distance(base_hues[i], base_hues[j]);
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
        rgb_to_oklch(palette[i * 3 + 0], palette[i * 3 + 1], palette[i * 3 + 2], &L1, &C1, &H1);
        rgb_to_oklch(palette[(i + 8) * 3 + 0], palette[(i + 8) * 3 + 1], palette[(i + 8) * 3 + 2], &L2, &C2, &H2);
        float hdist = hue_distance(H1, H2);
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

    fprintf(f, "# WCAG-Optimized Theme\n");
    fprintf(f, "# Generated by CUDA Color Optimizer v2.5\n");
    fprintf(f, "# Constraints: Red>=5.5, Green>=4.5, others>=3.5 on black\n");
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
    bool enable_fm_pairs = true;
    bool enable_gb_exclusions = true;

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
        } else if (strcmp(argv[i], "--no-fm-pairs") == 0) {
            enable_fm_pairs = false;
        } else if (strcmp(argv[i], "--no-gb-exclusions") == 0) {
            enable_gb_exclusions = false;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("CUDA Color Palette Optimizer v2.5 (Golden yellow, brighter red)\n");
            printf("Usage: %s [options]\n", argv[0]);
            printf("  -p, --population N     Population size (default: 200000)\n");
            printf("  -g, --generations N    Number of generations (default: 5000)\n");
            printf("  -m, --mutation F       Mutation rate (default: 0.15)\n");
            printf("  -o, --output FILE      Write theme to file (default: ./themes/theme-YYMMDD-HHMMSS)\n");
            printf("  --no-fm-pairs          Disable FM pairs constraints (on blue/green)\n");
            printf("  --no-gb-exclusions     Disable green/blue exclusions from perceptual rules\n");
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

    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║        CUDA Color Palette Optimizer v2.5 (Golden yellow, brighter red)      ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n\n");

    printf("Parameters:\n");
    printf("  Population: %d\n", population_size);
    printf("  Generations: %d\n", generations);
    printf("  Mutation rate: %.2f (adaptive)\n", mutation_rate);
    printf("  Elite ratio: %.2f\n", elite_ratio);
    printf("  Output: %s\n", output_file);
    printf("  FM pairs: %s\n", enable_fm_pairs ? "enabled" : "DISABLED");
    printf("  G/B exclusions: %s\n\n", enable_gb_exclusions ? "enabled" : "DISABLED");

    printf("Fixed colors:\n");
    printf("  Black:    #000000\n");
    printf("  Br.White: #ffffff\n\n");

    printf("Constraints:\n");
    printf("  1. Base colors (1-7) on black: per-color min <= CR <= %.1f\n", MAX_BASE_ON_BLACK);
    printf("     Red: >= %.1f, Green: >= %.1f, Yellow: >= %.1f\n",
           h_min_contrast[0], h_min_contrast[1], h_min_contrast[2]);
    printf("     Blue: >= %.1f, Magenta: >= %.1f, Cyan: >= %.1f, White: >= %.1f\n",
           h_min_contrast[3], h_min_contrast[4], h_min_contrast[5], h_min_contrast[6]);
    printf("  2. Bright on regular (target: >= %.1f, br.black on black: >= %.1f)\n", MIN_BRIGHT_ON_REGULAR, MIN_BR_BLACK_ON_BLACK);
    if (enable_fm_pairs) {
        printf("  3. FM pairs on blue (red, green, yellow, magenta, cyan, white): >= %.1f\n", MIN_ON_BLUE);
        printf("  4. FM pairs on green (red, yellow, blue, magenta, cyan, white): >= %.1f\n\n", MIN_ON_GREEN);
    } else {
        printf("  3. FM pairs on blue: DISABLED\n");
        printf("  4. FM pairs on green: DISABLED\n\n");
    }

    // Print color ranges
    printf("Color ranges:\n");
    const char* names[] = {
        "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white",
        "br.black", "br.red", "br.green", "br.yellow", "br.blue", "br.magenta", "br.cyan", "br.white"
    };
    for (int i = 0; i < 16; i++) {
        const ColorRange& r = color_ranges[i];
        if (r.fixed) {
            printf("  %-12s (fixed #%02x%02x%02x)\n", names[i],
                   (int)r.r_min, (int)r.g_min, (int)r.b_min);
        } else {
            printf("  %-12s R:%3.0f-%-3.0f  G:%3.0f-%-3.0f  B:%3.0f-%-3.0f\n",
                   names[i], r.r_min, r.r_max, r.g_min, r.g_max, r.b_min, r.b_max);
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

    // Copy color ranges and flags to device
    cudaMemcpyToSymbol(d_ranges, color_ranges, sizeof(color_ranges));
    cudaMemcpyToSymbol(d_enable_fm_pairs, &enable_fm_pairs, sizeof(bool));
    cudaMemcpyToSymbol(d_enable_gb_exclusions, &enable_gb_exclusions, sizeof(bool));

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

    // Print results
    print_color_demo(best_ever_palette.data());

    // Write theme file (always, defaults to ./themes/theme-YYMMDD-HHMMSS)
    write_theme_file(best_ever_palette.data(), output_file);

    // Cleanup
    cudaFree(d_pop1);
    cudaFree(d_pop2);
    cudaFree(d_fitness);
    cudaFree(d_states);
    cudaFree(d_elite_indices);

    return 0;
}
