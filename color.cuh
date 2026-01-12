/**
 * Color Module - Unified CUDA/Host Color Conversion and Contrast Functions
 *
 * This module provides:
 * - WCAG 2.1 compliant contrast calculations
 * - APCA (Accessible Perceptual Contrast Algorithm) for WCAG 3.0
 * - Oklab/OKLCH perceptually uniform color space conversions
 *
 * All functions work on both CUDA device and host.
 * Uses double precision for accuracy in color space conversions.
 *
 * References:
 * - WCAG 2.1: https://www.w3.org/TR/WCAG21/#dfn-relative-luminance
 * - APCA: https://github.com/Myndex/SAPC-APCA
 * - Oklab: https://bottosson.github.io/posts/oklab/
 */

#ifndef COLOR_CUH
#define COLOR_CUH

#include <cmath>

#if defined(__CUDACC__) || defined(__HIP__)
#define COLOR_FUNC __host__ __device__
#else
#define COLOR_FUNC
#endif

namespace color {

// =============================================================================
// Mathematical Constants
// =============================================================================

constexpr double PI = 3.14159265358979323846;

// =============================================================================
// WCAG 2.1 Contrast Ratio
// =============================================================================
// Reference: https://www.w3.org/TR/WCAG21/#dfn-relative-luminance
//
// The relative luminance calculation uses the sRGB color space with the
// following transfer function:
// - For values <= 0.04045: L = V / 12.92
// - For values > 0.04045: L = ((V + 0.055) / 1.055) ^ 2.4
//
// The luminance coefficients are based on the CIE 1931 standard observer
// adapted for sRGB primaries: R=0.2126, G=0.7152, B=0.0722

namespace wcag2 {

/**
 * Linearize an sRGB channel value (0-255) to linear light.
 * Uses the exact WCAG 2.1 threshold of 0.04045.
 */
COLOR_FUNC inline double linearize(double channel) {
    double v = channel / 255.0;
    if (v <= 0.04045) {
        return v / 12.92;
    }
    return pow((v + 0.055) / 1.055, 2.4);
}

/**
 * Compute relative luminance from sRGB values (0-255).
 * Returns a value in [0, 1] where 0 is black and 1 is white.
 */
COLOR_FUNC inline double luminance(double r, double g, double b) {
    return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b);
}

/**
 * Compute WCAG 2.1 contrast ratio between two colors.
 * The result is always >= 1.0, with higher values indicating more contrast.
 * AA requires 4.5:1 for normal text, 3:1 for large text.
 * AAA requires 7:1 for normal text, 4.5:1 for large text.
 */
COLOR_FUNC inline double contrast_ratio(double r1, double g1, double b1,
                                        double r2, double g2, double b2) {
    double l1 = luminance(r1, g1, b1);
    double l2 = luminance(r2, g2, b2);

    // Ensure l1 is the lighter color
    if (l1 < l2) {
        double tmp = l1;
        l1 = l2;
        l2 = tmp;
    }

    return (l1 + 0.05) / (l2 + 0.05);
}

} // namespace wcag2

// =============================================================================
// APCA - Accessible Perceptual Contrast Algorithm (WCAG 3.0)
// =============================================================================
// Reference: https://github.com/Myndex/SAPC-APCA
// Version: 0.0.98G-4g (Feb 15, 2021 - stable constants)
//
// APCA provides perceptually uniform contrast values that better predict
// readability than WCAG 2.x, especially for:
// - Dark mode (light text on dark background)
// - Low luminance colors
// - Small text
//
// Output: Lc value from -108 to +106
// - Positive: dark text on light background
// - Negative: light text on dark background (dark mode)
// - |Lc| >= 75: minimum for body text
// - |Lc| >= 90: preferred for body text

namespace apca {

// APCA 0.0.98G-4g constants
namespace constants {
    // Exponents for normal polarity (dark text on light background)
    constexpr double normBG = 0.56;
    constexpr double normTXT = 0.57;

    // Exponents for reverse polarity (light text on dark background)
    constexpr double revBG = 0.65;
    constexpr double revTXT = 0.62;

    // Scale and offset
    constexpr double scaleBoW = 1.14;
    constexpr double scaleWoB = 1.14;
    constexpr double loBoWoffset = 0.027;
    constexpr double loWoBoffset = 0.027;

    // Clamps
    constexpr double loClip = 0.1;
    constexpr double deltaYmin = 0.0005;

    // Black level soft clamp
    constexpr double blkThrs = 0.022;
    constexpr double blkClmp = 1.414;

    // Luminance coefficients (sRGB to Y)
    constexpr double sRco = 0.2126729;
    constexpr double sGco = 0.7151522;
    constexpr double sBco = 0.0721750;
}

/**
 * sRGB to linear using simple 2.4 gamma (APCA uses this instead of piecewise).
 * Input: channel value 0-255
 * Output: linear value 0-1
 */
COLOR_FUNC inline double srgb_to_linear(double channel) {
    double v = channel / 255.0;
    return pow(v, 2.4);
}

/**
 * Compute luminance (Y) for APCA from sRGB values (0-255).
 */
COLOR_FUNC inline double luminance(double r, double g, double b) {
    return constants::sRco * srgb_to_linear(r) +
           constants::sGco * srgb_to_linear(g) +
           constants::sBco * srgb_to_linear(b);
}

/**
 * Apply soft clamp near black levels.
 * This compensates for flare and ambient light on displays.
 */
COLOR_FUNC inline double soft_clamp(double y) {
    if (y > constants::blkThrs) {
        return y;
    }
    return y + pow(constants::blkThrs - y, constants::blkClmp);
}

/**
 * Compute APCA contrast (Lc value) between text and background colors.
 *
 * @param text_r, text_g, text_b: Text/foreground color (0-255)
 * @param bg_r, bg_g, bg_b: Background color (0-255)
 * @return Lc contrast value:
 *         - Positive (0 to ~106): dark text on light background
 *         - Negative (-108 to 0): light text on dark background
 *         - |Lc| >= 75: minimum for body text
 *         - |Lc| >= 90: preferred for body text
 *         - |Lc| < 30: not readable
 */
COLOR_FUNC inline double contrast(double text_r, double text_g, double text_b,
                                  double bg_r, double bg_g, double bg_b) {
    // Compute luminance
    double txtY = luminance(text_r, text_g, text_b);
    double bgY = luminance(bg_r, bg_g, bg_b);

    // Apply soft clamps
    txtY = soft_clamp(txtY);
    bgY = soft_clamp(bgY);

    // Check for insufficient difference
    double deltaY = bgY - txtY;
    if (fabs(deltaY) < constants::deltaYmin) {
        return 0.0;
    }

    double sapc;
    double output;

    if (bgY > txtY) {
        // Normal polarity: dark text on light background
        sapc = (pow(bgY, constants::normBG) - pow(txtY, constants::normTXT)) * constants::scaleBoW;
        if (sapc < constants::loClip) {
            output = 0.0;
        } else {
            output = sapc - constants::loBoWoffset;
        }
    } else {
        // Reverse polarity: light text on dark background
        sapc = (pow(bgY, constants::revBG) - pow(txtY, constants::revTXT)) * constants::scaleWoB;
        if (sapc > -constants::loClip) {
            output = 0.0;
        } else {
            output = sapc + constants::loWoBoffset;
        }
    }

    return output * 100.0;
}

/**
 * Get the absolute contrast value (polarity-independent).
 * Useful when you just want to know "how much contrast" regardless of mode.
 */
COLOR_FUNC inline double contrast_abs(double text_r, double text_g, double text_b,
                                      double bg_r, double bg_g, double bg_b) {
    return fabs(contrast(text_r, text_g, text_b, bg_r, bg_g, bg_b));
}

/**
 * Check if contrast meets minimum readability threshold.
 * @param min_lc: Minimum |Lc| value (75 for body text, 60 for large text, etc.)
 */
COLOR_FUNC inline bool is_readable(double text_r, double text_g, double text_b,
                                    double bg_r, double bg_g, double bg_b,
                                    double min_lc = 75.0) {
    return contrast_abs(text_r, text_g, text_b, bg_r, bg_g, bg_b) >= min_lc;
}

} // namespace apca

// =============================================================================
// Oklab Color Space
// =============================================================================
// Reference: https://bottosson.github.io/posts/oklab/
//
// Oklab is a perceptually uniform color space designed for:
// - Uniform hue perception
// - Uniform lightness perception
// - Better than Lab for color manipulation
//
// Components:
// - L: Lightness (0 = black, 1 = white)
// - a: Green-red axis (negative = green, positive = red)
// - b: Blue-yellow axis (negative = blue, positive = yellow)

namespace oklab {

struct Lab {
    double L;
    double a;
    double b;
};

/**
 * Convert sRGB (0-255) to Oklab.
 */
COLOR_FUNC inline Lab from_srgb(double r, double g, double b) {
    // sRGB to linear RGB
    double lr = wcag2::linearize(r);
    double lg = wcag2::linearize(g);
    double lb = wcag2::linearize(b);

    // Linear RGB to LMS (cone response)
    double l = 0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb;
    double m = 0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb;
    double s = 0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb;

    // Cube root (perceptual nonlinearity)
    double l_ = cbrt(l);
    double m_ = cbrt(m);
    double s_ = cbrt(s);

    // LMS' to Oklab
    Lab result;
    result.L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
    result.a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
    result.b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;

    return result;
}

/**
 * Compute Euclidean distance between two colors in Oklab space.
 * This is a good approximation of perceptual color difference (ΔE).
 * Values around 0.02-0.03 are just noticeable differences.
 */
COLOR_FUNC inline double distance(double r1, double g1, double b1,
                                  double r2, double g2, double b2) {
    Lab lab1 = from_srgb(r1, g1, b1);
    Lab lab2 = from_srgb(r2, g2, b2);

    double dL = lab1.L - lab2.L;
    double da = lab1.a - lab2.a;
    double db = lab1.b - lab2.b;

    return sqrt(dL * dL + da * da + db * db);
}

/**
 * Convert Oklab to linear RGB.
 * Returns values that may be outside [0,1] if out of sRGB gamut.
 */
COLOR_FUNC inline void to_linear_rgb(Lab lab, double* lr, double* lg, double* lb) {
    // Oklab to LMS' (inverse of final matrix)
    double l_ = lab.L + 0.3963377774 * lab.a + 0.2158037573 * lab.b;
    double m_ = lab.L - 0.1055613458 * lab.a - 0.0638541728 * lab.b;
    double s_ = lab.L - 0.0894841775 * lab.a - 1.2914855480 * lab.b;

    // LMS' to LMS (cube)
    double l = l_ * l_ * l_;
    double m = m_ * m_ * m_;
    double s = s_ * s_ * s_;

    // LMS to linear RGB (inverse of forward matrix)
    *lr = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    *lg = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    *lb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;
}

/**
 * Convert a single linear RGB channel to sRGB (0-255).
 * Applies gamma correction and clamps to valid range.
 */
COLOR_FUNC inline double linear_to_srgb_channel(double linear) {
    double v;
    if (linear <= 0.0031308) {
        v = 12.92 * linear;
    } else {
        v = 1.055 * pow(linear, 1.0 / 2.4) - 0.055;
    }
    // Clamp to [0, 1] then scale to [0, 255]
    if (v < 0.0) v = 0.0;
    if (v > 1.0) v = 1.0;
    return v * 255.0;
}

/**
 * Convert Oklab to sRGB (0-255).
 * Clamps out-of-gamut colors.
 */
COLOR_FUNC inline void to_srgb(Lab lab, double* r, double* g, double* b) {
    double lr, lg, lb;
    to_linear_rgb(lab, &lr, &lg, &lb);
    *r = linear_to_srgb_channel(lr);
    *g = linear_to_srgb_channel(lg);
    *b = linear_to_srgb_channel(lb);
}

/**
 * Check if an Oklab color is within sRGB gamut.
 */
COLOR_FUNC inline bool is_in_gamut(Lab lab) {
    double lr, lg, lb;
    to_linear_rgb(lab, &lr, &lg, &lb);
    return lr >= -0.0001 && lr <= 1.0001 &&
           lg >= -0.0001 && lg <= 1.0001 &&
           lb >= -0.0001 && lb <= 1.0001;
}

} // namespace oklab

// =============================================================================
// OKLCH Color Space (Polar form of Oklab)
// =============================================================================
// OKLCH is the polar/cylindrical representation of Oklab.
//
// Components:
// - L: Lightness (0 = black, 1 = white)
// - C: Chroma (saturation, 0 = gray, higher = more saturated)
// - H: Hue (angle in degrees, 0-360)
//
// Hue values roughly correspond to:
// - 0°/360°: Red
// - 60°: Yellow
// - 120°: Green
// - 180°: Cyan
// - 240°: Blue
// - 300°: Magenta

namespace oklch {

struct LCH {
    double L;
    double C;
    double H;
};

/**
 * Convert sRGB (0-255) to OKLCH.
 */
COLOR_FUNC inline LCH from_srgb(double r, double g, double b) {
    oklab::Lab lab = oklab::from_srgb(r, g, b);

    LCH result;
    result.L = lab.L;
    result.C = sqrt(lab.a * lab.a + lab.b * lab.b);
    result.H = atan2(lab.b, lab.a) * 180.0 / PI;

    // Normalize hue to 0-360
    if (result.H < 0.0) {
        result.H += 360.0;
    }

    return result;
}

/**
 * Compute angular distance between two hues (handles wraparound).
 * Returns value in [0, 180].
 */
COLOR_FUNC inline double hue_distance(double h1, double h2) {
    double diff = fabs(h1 - h2);
    if (diff > 180.0) {
        diff = 360.0 - diff;
    }
    return diff;
}

/**
 * Check if two colors have similar hue (within tolerance).
 */
COLOR_FUNC inline bool hue_similar(double r1, double g1, double b1,
                                    double r2, double g2, double b2,
                                    double tolerance = 30.0) {
    LCH lch1 = from_srgb(r1, g1, b1);
    LCH lch2 = from_srgb(r2, g2, b2);

    // If either color is achromatic, hue comparison is meaningless
    if (lch1.C < 0.02 || lch2.C < 0.02) {
        return true;
    }

    return hue_distance(lch1.H, lch2.H) <= tolerance;
}

/**
 * Convert OKLCH to Oklab.
 */
COLOR_FUNC inline oklab::Lab to_oklab(double L, double C, double H) {
    double h_rad = H * PI / 180.0;
    oklab::Lab lab;
    lab.L = L;
    lab.a = C * cos(h_rad);
    lab.b = C * sin(h_rad);
    return lab;
}

/**
 * Convert OKLCH to sRGB (0-255).
 * Clamps out-of-gamut colors.
 */
COLOR_FUNC inline void to_srgb(double L, double C, double H, double* r, double* g, double* b) {
    oklab::Lab lab = to_oklab(L, C, H);
    oklab::to_srgb(lab, r, g, b);
}

/**
 * Check if an OKLCH color is within sRGB gamut.
 */
COLOR_FUNC inline bool is_in_gamut(double L, double C, double H) {
    oklab::Lab lab = to_oklab(L, C, H);
    return oklab::is_in_gamut(lab);
}

/**
 * Find maximum chroma at given L and H that stays in sRGB gamut.
 * Uses binary search for efficiency.
 */
COLOR_FUNC inline double max_chroma_in_gamut(double L, double H) {
    double low = 0.0;
    double high = 0.5;  // Max reasonable chroma

    for (int i = 0; i < 20; i++) {  // More iterations for double precision
        double mid = (low + high) * 0.5;
        if (is_in_gamut(L, mid, H)) {
            low = mid;
        } else {
            high = mid;
        }
    }
    return low;
}

/**
 * Normalize hue to [0, 360) range.
 */
COLOR_FUNC inline double normalize_hue(double h) {
    while (h < 0.0) h += 360.0;
    while (h >= 360.0) h -= 360.0;
    return h;
}

/**
 * Interpolate between two hues (circular interpolation).
 * @param h1 First hue (degrees)
 * @param h2 Second hue (degrees)
 * @param t Interpolation factor (0 = h1, 1 = h2)
 */
COLOR_FUNC inline double lerp_hue(double h1, double h2, double t) {
    double diff = h2 - h1;
    if (diff > 180.0) diff -= 360.0;
    if (diff < -180.0) diff += 360.0;
    return normalize_hue(h1 + t * diff);
}

} // namespace oklch

// =============================================================================
// Convenience Functions (Legacy Interface)
// =============================================================================
// These provide backward compatibility with the original function signatures.

COLOR_FUNC inline double linearize(double c) {
    return wcag2::linearize(c);
}

COLOR_FUNC inline double luminance(double r, double g, double b) {
    return wcag2::luminance(r, g, b);
}

COLOR_FUNC inline double contrast_ratio(double r1, double g1, double b1,
                                        double r2, double g2, double b2) {
    return wcag2::contrast_ratio(r1, g1, b1, r2, g2, b2);
}

COLOR_FUNC inline void rgb_to_oklab(double r, double g, double b,
                                     double* L, double* a, double* ok_b) {
    oklab::Lab lab = oklab::from_srgb(r, g, b);
    *L = lab.L;
    *a = lab.a;
    *ok_b = lab.b;
}

COLOR_FUNC inline void rgb_to_oklch(double r, double g, double b,
                                     double* L, double* C, double* H) {
    oklch::LCH lch = oklch::from_srgb(r, g, b);
    *L = lch.L;
    *C = lch.C;
    *H = lch.H;
}

COLOR_FUNC inline double oklab_distance(double r1, double g1, double b1,
                                        double r2, double g2, double b2) {
    return oklab::distance(r1, g1, b1, r2, g2, b2);
}

COLOR_FUNC inline double hue_distance(double h1, double h2) {
    return oklch::hue_distance(h1, h2);
}

COLOR_FUNC inline void oklch_to_srgb(double L, double C, double H,
                                      double* r, double* g, double* b) {
    oklch::to_srgb(L, C, H, r, g, b);
}

COLOR_FUNC inline bool oklch_in_gamut(double L, double C, double H) {
    return oklch::is_in_gamut(L, C, H);
}

COLOR_FUNC inline double oklch_max_chroma(double L, double H) {
    return oklch::max_chroma_in_gamut(L, H);
}

} // namespace color

#endif // COLOR_CUH
