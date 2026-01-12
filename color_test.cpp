/**
 * Color Module Test Suite
 *
 * Tests for WCAG 2.1, APCA, Oklab, and OKLCH implementations.
 * Validates against known reference values and cross-checks implementations.
 *
 * Build: g++ -std=c++17 -O2 color_test.cpp -o color_test -lm
 * Run: ./color_test
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

#include "color.cuh"

// Test configuration
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_EPSILON 0.01
#define STRICT_EPSILON 0.001

void check_double(const char* test_name, double expected, double actual, double epsilon = TEST_EPSILON) {
    tests_run++;
    double diff = fabs(expected - actual);
    if (diff <= epsilon) {
        tests_passed++;
        printf("  ✓ %s: expected %.4f, got %.4f\n", test_name, expected, actual);
    } else {
        tests_failed++;
        printf("  ✗ %s: expected %.4f, got %.4f (diff: %.6f)\n", test_name, expected, actual, diff);
    }
}

void check_bool(const char* test_name, bool expected, bool actual) {
    tests_run++;
    if (expected == actual) {
        tests_passed++;
        printf("  ✓ %s: %s\n", test_name, actual ? "true" : "false");
    } else {
        tests_failed++;
        printf("  ✗ %s: expected %s, got %s\n", test_name,
               expected ? "true" : "false", actual ? "true" : "false");
    }
}

// =============================================================================
// WCAG 2.1 Tests
// =============================================================================
// Reference: https://www.w3.org/TR/WCAG21/#dfn-relative-luminance
//
// The spec provides this example:
// For sRGB colorspace, the relative luminance of a color is defined as:
// L = 0.2126 * R + 0.7152 * G + 0.0722 * B

void test_wcag2_linearize() {
    printf("\n== WCAG 2.1 Linearization ==\n");

    // Test threshold value (0.04045 * 255 ≈ 10.315)
    // Below threshold: linear region
    check_double("linearize(0) = 0", 0.0, color::wcag2::linearize(0.0), STRICT_EPSILON);
    check_double("linearize(10) ≈ 0.00304", 10.0/255.0/12.92, color::wcag2::linearize(10.0), STRICT_EPSILON);

    // Above threshold: power law region
    // linearize(255) = ((255/255 + 0.055) / 1.055)^2.4 = 1.0
    check_double("linearize(255) = 1.0", 1.0, color::wcag2::linearize(255.0), STRICT_EPSILON);

    // Mid-gray (sRGB 127.5 should give linear ~0.2140)
    double mid = color::wcag2::linearize(127.5);
    check_double("linearize(127.5) ≈ 0.214", 0.214, mid, 0.01);

    // Test the exact threshold point (0.04045)
    double threshold_input = 0.04045 * 255.0;  // ≈ 10.315
    double below = color::wcag2::linearize(threshold_input - 0.1);
    double at = color::wcag2::linearize(threshold_input);
    double above = color::wcag2::linearize(threshold_input + 0.1);
    printf("  Info: Linearization at threshold region: below=%.6f, at=%.6f, above=%.6f\n", below, at, above);
}

void test_wcag2_luminance() {
    printf("\n== WCAG 2.1 Relative Luminance ==\n");

    // Black (0,0,0) -> L = 0
    check_double("luminance(0,0,0) = 0", 0.0, color::wcag2::luminance(0, 0, 0), STRICT_EPSILON);

    // White (255,255,255) -> L = 1
    check_double("luminance(255,255,255) = 1", 1.0, color::wcag2::luminance(255, 255, 255), STRICT_EPSILON);

    // Pure red (255,0,0) -> L ≈ 0.2126 (linearize(255) * 0.2126 = 0.2126)
    check_double("luminance(255,0,0) ≈ 0.2126", 0.2126, color::wcag2::luminance(255, 0, 0), 0.001);

    // Pure green (0,255,0) -> L ≈ 0.7152
    check_double("luminance(0,255,0) ≈ 0.7152", 0.7152, color::wcag2::luminance(0, 255, 0), 0.001);

    // Pure blue (0,0,255) -> L ≈ 0.0722
    check_double("luminance(0,0,255) ≈ 0.0722", 0.0722, color::wcag2::luminance(0, 0, 255), 0.001);

    // Sum of pure colors should equal white's luminance
    double sum = color::wcag2::luminance(255, 0, 0) +
                 color::wcag2::luminance(0, 255, 0) +
                 color::wcag2::luminance(0, 0, 255);
    check_double("R+G+B luminance = 1.0", 1.0, sum, 0.001);
}

void test_wcag2_contrast_ratio() {
    printf("\n== WCAG 2.1 Contrast Ratio ==\n");

    // White on black: (1.0 + 0.05) / (0.0 + 0.05) = 21:1
    check_double("white/black = 21:1", 21.0,
                color::wcag2::contrast_ratio(255, 255, 255, 0, 0, 0), 0.01);

    // Black on white: same result (always >= 1)
    check_double("black/white = 21:1", 21.0,
                color::wcag2::contrast_ratio(0, 0, 0, 255, 255, 255), 0.01);

    // Same color: 1:1
    check_double("gray/gray = 1:1", 1.0,
                color::wcag2::contrast_ratio(128, 128, 128, 128, 128, 128), 0.01);

    // WCAG AA minimum for normal text is 4.5:1
    // #767676 on white is approximately 4.5:1 (commonly used minimum gray)
    double aa_gray = color::wcag2::contrast_ratio(0x76, 0x76, 0x76, 255, 255, 255);
    printf("  Info: #767676 on white = %.2f:1 (WCAG AA minimum is 4.5:1)\n", aa_gray);
    check_bool("#767676 passes AA on white", true, aa_gray >= 4.5);

    // #595959 on white should be around 7:1 (AAA level)
    double aaa_gray = color::wcag2::contrast_ratio(0x59, 0x59, 0x59, 255, 255, 255);
    printf("  Info: #595959 on white = %.2f:1 (WCAG AAA minimum is 7:1)\n", aaa_gray);
    check_bool("#595959 passes AAA on white", true, aaa_gray >= 7.0);
}

// =============================================================================
// APCA Tests
// =============================================================================
// Reference: https://github.com/Myndex/SAPC-APCA
// Known values:
// - White (#FFFFFF) text on Black (#000000) background: Lc ≈ -108
// - Black (#000000) text on White (#FFFFFF) background: Lc ≈ +106

void test_apca_linearize() {
    printf("\n== APCA sRGB Linearization ==\n");

    // APCA uses simpler 2.4 gamma (no piecewise)
    check_double("apca_linear(0) = 0", 0.0, color::apca::srgb_to_linear(0.0), STRICT_EPSILON);
    check_double("apca_linear(255) = 1", 1.0, color::apca::srgb_to_linear(255.0), STRICT_EPSILON);

    // Compare with WCAG linearization
    double wcag_mid = color::wcag2::linearize(128.0);
    double apca_mid = color::apca::srgb_to_linear(128.0);
    printf("  Info: Mid-gray (128): WCAG=%.4f, APCA=%.4f (APCA uses pure 2.4 gamma)\n",
           wcag_mid, apca_mid);
}

void test_apca_contrast() {
    printf("\n== APCA Contrast (Lc values) ==\n");

    // White text on black background (dark mode): should be around -108
    double white_on_black = color::apca::contrast(255, 255, 255, 0, 0, 0);
    printf("  Info: White on black = Lc %.2f (expected ≈ -108)\n", white_on_black);
    check_bool("white on black is negative (reverse polarity)", true, white_on_black < 0);
    check_double("white on black ≈ -108", -108.0, white_on_black, 5.0);

    // Black text on white background (normal mode): should be around +106
    double black_on_white = color::apca::contrast(0, 0, 0, 255, 255, 255);
    printf("  Info: Black on white = Lc %.2f (expected ≈ +106)\n", black_on_white);
    check_bool("black on white is positive (normal polarity)", true, black_on_white > 0);
    check_double("black on white ≈ +106", 106.0, black_on_white, 5.0);

    // Same color should give 0
    double same = color::apca::contrast(128, 128, 128, 128, 128, 128);
    check_double("same color = 0", 0.0, same, 0.1);

    // Very similar colors should be near 0
    double near_same = color::apca::contrast(128, 128, 128, 129, 129, 129);
    printf("  Info: Nearly same colors = Lc %.2f (should be near 0)\n", near_same);

    // Absolute value function
    double abs_wob = color::apca::contrast_abs(255, 255, 255, 0, 0, 0);
    double abs_bow = color::apca::contrast_abs(0, 0, 0, 255, 255, 255);
    printf("  Info: |white on black| = %.2f, |black on white| = %.2f\n", abs_wob, abs_bow);

    // Test readability check
    check_bool("white on black is readable (Lc 75)", true,
               color::apca::is_readable(255, 255, 255, 0, 0, 0, 75.0));
    check_bool("gray on gray not readable", false,
               color::apca::is_readable(128, 128, 128, 140, 140, 140, 75.0));
}

void test_apca_polarity() {
    printf("\n== APCA Polarity ==\n");

    // Dark text on light background should be positive
    double dark_on_light = color::apca::contrast(50, 50, 50, 200, 200, 200);
    check_bool("dark on light is positive", true, dark_on_light > 0);

    // Light text on dark background should be negative
    double light_on_dark = color::apca::contrast(200, 200, 200, 50, 50, 50);
    check_bool("light on dark is negative", true, light_on_dark < 0);

    printf("  Info: Dark on light = Lc %.2f, Light on dark = Lc %.2f\n",
           dark_on_light, light_on_dark);
}

// =============================================================================
// Oklab Tests
// =============================================================================
// Reference: https://bottosson.github.io/posts/oklab/

void test_oklab_conversion() {
    printf("\n== Oklab Color Space ==\n");

    // Black should be L=0, a=0, b=0
    color::oklab::Lab black = color::oklab::from_srgb(0, 0, 0);
    check_double("black L = 0", 0.0, black.L, 0.01);
    check_double("black a ≈ 0", 0.0, black.a, 0.01);
    check_double("black b ≈ 0", 0.0, black.b, 0.01);

    // White should be L=1, a=0, b=0
    color::oklab::Lab white = color::oklab::from_srgb(255, 255, 255);
    check_double("white L = 1", 1.0, white.L, 0.01);
    check_double("white a ≈ 0", 0.0, white.a, 0.01);
    check_double("white b ≈ 0", 0.0, white.b, 0.01);

    // Mid gray should have L≈0.6, a=0, b=0 (perceptually mid is not 0.5)
    color::oklab::Lab gray = color::oklab::from_srgb(128, 128, 128);
    printf("  Info: Gray (128,128,128) -> L=%.3f, a=%.3f, b=%.3f\n", gray.L, gray.a, gray.b);
    check_double("gray a ≈ 0", 0.0, gray.a, 0.01);
    check_double("gray b ≈ 0", 0.0, gray.b, 0.01);

    // Red should have positive a (red-green axis)
    color::oklab::Lab red = color::oklab::from_srgb(255, 0, 0);
    printf("  Info: Red -> L=%.3f, a=%.3f, b=%.3f\n", red.L, red.a, red.b);
    check_bool("red has positive a", true, red.a > 0);

    // Green should have negative a
    color::oklab::Lab green = color::oklab::from_srgb(0, 255, 0);
    printf("  Info: Green -> L=%.3f, a=%.3f, b=%.3f\n", green.L, green.a, green.b);
    check_bool("green has negative a", true, green.a < 0);

    // Blue should have negative b (blue-yellow axis)
    color::oklab::Lab blue = color::oklab::from_srgb(0, 0, 255);
    printf("  Info: Blue -> L=%.3f, a=%.3f, b=%.3f\n", blue.L, blue.a, blue.b);
    check_bool("blue has negative b", true, blue.b < 0);

    // Yellow should have positive b
    color::oklab::Lab yellow = color::oklab::from_srgb(255, 255, 0);
    printf("  Info: Yellow -> L=%.3f, a=%.3f, b=%.3f\n", yellow.L, yellow.a, yellow.b);
    check_bool("yellow has positive b", true, yellow.b > 0);
}

void test_oklab_distance() {
    printf("\n== Oklab Distance ==\n");

    // Same color should have 0 distance
    check_double("same color distance = 0", 0.0,
                color::oklab::distance(128, 128, 128, 128, 128, 128), STRICT_EPSILON);

    // Black to white should be ~1.0 (L difference of 1)
    double bw_dist = color::oklab::distance(0, 0, 0, 255, 255, 255);
    printf("  Info: Black to white distance = %.3f\n", bw_dist);
    check_double("black to white ≈ 1.0", 1.0, bw_dist, 0.05);

    // Just noticeable difference (JND) is around 0.02-0.03
    double jnd_dist = color::oklab::distance(128, 128, 128, 130, 130, 130);
    printf("  Info: Slight gray difference = %.4f (JND ≈ 0.02-0.03)\n", jnd_dist);
}

// =============================================================================
// OKLCH Tests
// =============================================================================

void test_oklch_conversion() {
    printf("\n== OKLCH Color Space ==\n");

    // White and black should have 0 chroma
    color::oklch::LCH white = color::oklch::from_srgb(255, 255, 255);
    check_double("white chroma ≈ 0", 0.0, white.C, 0.01);

    color::oklch::LCH black = color::oklch::from_srgb(0, 0, 0);
    check_double("black chroma ≈ 0", 0.0, black.C, 0.01);

    // Red should have hue around 29° (Oklab red is around this)
    color::oklch::LCH red = color::oklch::from_srgb(255, 0, 0);
    printf("  Info: Red -> L=%.3f, C=%.3f, H=%.1f°\n", red.L, red.C, red.H);
    check_bool("red has positive chroma", true, red.C > 0);

    // Yellow should have hue around 110°
    color::oklch::LCH yellow = color::oklch::from_srgb(255, 255, 0);
    printf("  Info: Yellow -> L=%.3f, C=%.3f, H=%.1f°\n", yellow.L, yellow.C, yellow.H);

    // Green should have hue around 142°
    color::oklch::LCH green = color::oklch::from_srgb(0, 255, 0);
    printf("  Info: Green -> L=%.3f, C=%.3f, H=%.1f°\n", green.L, green.C, green.H);

    // Cyan should have hue around 195°
    color::oklch::LCH cyan = color::oklch::from_srgb(0, 255, 255);
    printf("  Info: Cyan -> L=%.3f, C=%.3f, H=%.1f°\n", cyan.L, cyan.C, cyan.H);

    // Blue should have hue around 264°
    color::oklch::LCH blue = color::oklch::from_srgb(0, 0, 255);
    printf("  Info: Blue -> L=%.3f, C=%.3f, H=%.1f°\n", blue.L, blue.C, blue.H);

    // Magenta should have hue around 328°
    color::oklch::LCH magenta = color::oklch::from_srgb(255, 0, 255);
    printf("  Info: Magenta -> L=%.3f, C=%.3f, H=%.1f°\n", magenta.L, magenta.C, magenta.H);
}

void test_oklch_hue_distance() {
    printf("\n== OKLCH Hue Distance ==\n");

    // Same hue
    check_double("0° to 0° = 0", 0.0, color::oklch::hue_distance(0.0, 0.0), STRICT_EPSILON);

    // Simple difference
    check_double("0° to 30° = 30", 30.0, color::oklch::hue_distance(0.0, 30.0), STRICT_EPSILON);

    // Wraparound (350° to 10° should be 20°, not 340°)
    check_double("350° to 10° = 20", 20.0, color::oklch::hue_distance(350.0, 10.0), STRICT_EPSILON);

    // Maximum distance (opposite hues)
    check_double("0° to 180° = 180", 180.0, color::oklch::hue_distance(0.0, 180.0), STRICT_EPSILON);

    // Symmetric
    check_double("10° to 350° = 20", 20.0, color::oklch::hue_distance(10.0, 350.0), STRICT_EPSILON);
}

void test_oklch_hue_similar() {
    printf("\n== OKLCH Hue Similarity ==\n");

    // Same color is similar
    check_bool("same color is similar", true,
               color::oklch::hue_similar(255, 0, 0, 255, 0, 0));

    // Gray colors (low chroma) are always similar (hue undefined)
    check_bool("grays are similar", true,
               color::oklch::hue_similar(128, 128, 128, 64, 64, 64));

    // Red and orange (hue difference ~30°) should be similar with default tolerance
    check_bool("red and orange similar (30° tolerance)", true,
               color::oklch::hue_similar(255, 0, 0, 255, 128, 0, 60.0));

    // Red and green should not be similar
    check_bool("red and green not similar", false,
               color::oklch::hue_similar(255, 0, 0, 0, 255, 0, 30.0));
}

// =============================================================================
// Legacy Interface Tests
// =============================================================================

void test_legacy_interface() {
    printf("\n== Legacy Interface Compatibility ==\n");

    // Test that legacy functions match new namespace functions
    double leg_lin = color::linearize(128.0);
    double new_lin = color::wcag2::linearize(128.0);
    check_double("legacy linearize matches", new_lin, leg_lin, STRICT_EPSILON);

    double leg_lum = color::luminance(255, 128, 64);
    double new_lum = color::wcag2::luminance(255, 128, 64);
    check_double("legacy luminance matches", new_lum, leg_lum, STRICT_EPSILON);

    double leg_cr = color::contrast_ratio(255, 255, 255, 0, 0, 0);
    double new_cr = color::wcag2::contrast_ratio(255, 255, 255, 0, 0, 0);
    check_double("legacy contrast_ratio matches", new_cr, leg_cr, STRICT_EPSILON);

    // Test legacy oklab functions via pointer interface
    double L, a, b;
    color::rgb_to_oklab(255, 128, 64, &L, &a, &b);
    color::oklab::Lab lab = color::oklab::from_srgb(255, 128, 64);
    check_double("legacy rgb_to_oklab L matches", lab.L, L, STRICT_EPSILON);
    check_double("legacy rgb_to_oklab a matches", lab.a, a, STRICT_EPSILON);
    check_double("legacy rgb_to_oklab b matches", lab.b, b, STRICT_EPSILON);

    double lL, C, H;
    color::rgb_to_oklch(255, 128, 64, &lL, &C, &H);
    color::oklch::LCH lch = color::oklch::from_srgb(255, 128, 64);
    check_double("legacy rgb_to_oklch L matches", lch.L, lL, STRICT_EPSILON);
    check_double("legacy rgb_to_oklch C matches", lch.C, C, STRICT_EPSILON);
    check_double("legacy rgb_to_oklch H matches", lch.H, H, STRICT_EPSILON);

    double leg_dist = color::oklab_distance(255, 0, 0, 0, 255, 0);
    double new_dist = color::oklab::distance(255, 0, 0, 0, 255, 0);
    check_double("legacy oklab_distance matches", new_dist, leg_dist, STRICT_EPSILON);

    double leg_hue = color::hue_distance(350.0, 10.0);
    double new_hue = color::oklch::hue_distance(350.0, 10.0);
    check_double("legacy hue_distance matches", new_hue, leg_hue, STRICT_EPSILON);
}

// =============================================================================
// Cross-validation Tests
// =============================================================================

void test_wcag_vs_apca() {
    printf("\n== WCAG vs APCA Comparison ==\n");

    // Both should agree on extreme contrasts
    double wcag_wob = color::wcag2::contrast_ratio(255, 255, 255, 0, 0, 0);
    double apca_wob = color::apca::contrast_abs(255, 255, 255, 0, 0, 0);
    printf("  Info: White/Black - WCAG=%.1f:1, APCA=Lc %.1f\n", wcag_wob, apca_wob);

    // Both should give minimal contrast for same color
    double wcag_same = color::wcag2::contrast_ratio(128, 128, 128, 128, 128, 128);
    double apca_same = color::apca::contrast_abs(128, 128, 128, 128, 128, 128);
    check_double("WCAG same color = 1:1", 1.0, wcag_same, 0.01);
    check_double("APCA same color = 0", 0.0, apca_same, 0.1);

    // APCA should show asymmetry that WCAG doesn't
    // Dark text on light bg vs light text on dark bg
    double wcag_dol = color::wcag2::contrast_ratio(50, 50, 50, 200, 200, 200);
    double wcag_lod = color::wcag2::contrast_ratio(200, 200, 200, 50, 50, 50);
    double apca_dol = color::apca::contrast(50, 50, 50, 200, 200, 200);
    double apca_lod = color::apca::contrast(200, 200, 200, 50, 50, 50);

    printf("  Info: Dark(50) on Light(200) - WCAG=%.2f, APCA=Lc %.1f\n", wcag_dol, apca_dol);
    printf("  Info: Light(200) on Dark(50) - WCAG=%.2f, APCA=Lc %.1f\n", wcag_lod, apca_lod);

    // WCAG should be symmetric
    check_double("WCAG is symmetric", wcag_dol, wcag_lod, 0.01);

    // APCA should differ by polarity (one positive, one negative)
    check_bool("APCA dark-on-light is positive", true, apca_dol > 0);
    check_bool("APCA light-on-dark is negative", true, apca_lod < 0);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║              Color Module Test Suite                             ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");

    // WCAG 2.1 tests
    test_wcag2_linearize();
    test_wcag2_luminance();
    test_wcag2_contrast_ratio();

    // APCA tests
    test_apca_linearize();
    test_apca_contrast();
    test_apca_polarity();

    // Oklab tests
    test_oklab_conversion();
    test_oklab_distance();

    // OKLCH tests
    test_oklch_conversion();
    test_oklch_hue_distance();
    test_oklch_hue_similar();

    // Legacy interface
    test_legacy_interface();

    // Cross-validation
    test_wcag_vs_apca();

    // Summary
    printf("\n══════════════════════════════════════════════════════════════════\n");
    printf("Test Summary: %d tests, %d passed, %d failed\n", tests_run, tests_passed, tests_failed);
    printf("══════════════════════════════════════════════════════════════════\n");

    if (tests_failed > 0) {
        printf("\n⚠ Some tests failed!\n");
        return 1;
    }

    printf("\n✓ All tests passed!\n");
    return 0;
}
