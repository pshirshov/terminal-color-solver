/**
 * Color Module - Unified CUDA/Host Color Conversion and Contrast Functions
 *
 * This module provides:
 * - WCAG 2.1 compliant contrast calculations
 * - APCA (Accessible Perceptual Contrast Algorithm) for WCAG 3.0
 * - Oklab/OKLCH perceptually uniform color space conversions
 *
 * All functions work on both CUDA device and host.
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

constexpr float PI = 3.14159265358979323846f;

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
COLOR_FUNC inline float linearize(float channel) {
    float v = channel / 255.0f;
    if (v <= 0.04045f) {
        return v / 12.92f;
    }
    return powf((v + 0.055f) / 1.055f, 2.4f);
}

/**
 * Compute relative luminance from sRGB values (0-255).
 * Returns a value in [0, 1] where 0 is black and 1 is white.
 */
COLOR_FUNC inline float luminance(float r, float g, float b) {
    return 0.2126f * linearize(r) + 0.7152f * linearize(g) + 0.0722f * linearize(b);
}

/**
 * Compute WCAG 2.1 contrast ratio between two colors.
 * The result is always >= 1.0, with higher values indicating more contrast.
 * AA requires 4.5:1 for normal text, 3:1 for large text.
 * AAA requires 7:1 for normal text, 4.5:1 for large text.
 */
COLOR_FUNC inline float contrast_ratio(float r1, float g1, float b1,
                                        float r2, float g2, float b2) {
    float l1 = luminance(r1, g1, b1);
    float l2 = luminance(r2, g2, b2);

    // Ensure l1 is the lighter color
    if (l1 < l2) {
        float tmp = l1;
        l1 = l2;
        l2 = tmp;
    }

    return (l1 + 0.05f) / (l2 + 0.05f);
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
    constexpr float normBG = 0.56f;
    constexpr float normTXT = 0.57f;

    // Exponents for reverse polarity (light text on dark background)
    constexpr float revBG = 0.65f;
    constexpr float revTXT = 0.62f;

    // Scale and offset
    constexpr float scaleBoW = 1.14f;
    constexpr float scaleWoB = 1.14f;
    constexpr float loBoWoffset = 0.027f;
    constexpr float loWoBoffset = 0.027f;

    // Clamps
    constexpr float loClip = 0.1f;
    constexpr float deltaYmin = 0.0005f;

    // Black level soft clamp
    constexpr float blkThrs = 0.022f;
    constexpr float blkClmp = 1.414f;

    // Luminance coefficients (sRGB to Y)
    constexpr float sRco = 0.2126729f;
    constexpr float sGco = 0.7151522f;
    constexpr float sBco = 0.0721750f;
}

/**
 * sRGB to linear using simple 2.4 gamma (APCA uses this instead of piecewise).
 * Input: channel value 0-255
 * Output: linear value 0-1
 */
COLOR_FUNC inline float srgb_to_linear(float channel) {
    float v = channel / 255.0f;
    return powf(v, 2.4f);
}

/**
 * Compute luminance (Y) for APCA from sRGB values (0-255).
 */
COLOR_FUNC inline float luminance(float r, float g, float b) {
    return constants::sRco * srgb_to_linear(r) +
           constants::sGco * srgb_to_linear(g) +
           constants::sBco * srgb_to_linear(b);
}

/**
 * Apply soft clamp near black levels.
 * This compensates for flare and ambient light on displays.
 */
COLOR_FUNC inline float soft_clamp(float y) {
    if (y > constants::blkThrs) {
        return y;
    }
    return y + powf(constants::blkThrs - y, constants::blkClmp);
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
COLOR_FUNC inline float contrast(float text_r, float text_g, float text_b,
                                  float bg_r, float bg_g, float bg_b) {
    // Compute luminance
    float txtY = luminance(text_r, text_g, text_b);
    float bgY = luminance(bg_r, bg_g, bg_b);

    // Apply soft clamps
    txtY = soft_clamp(txtY);
    bgY = soft_clamp(bgY);

    // Check for insufficient difference
    float deltaY = bgY - txtY;
    if (fabsf(deltaY) < constants::deltaYmin) {
        return 0.0f;
    }

    float sapc;
    float output;

    if (bgY > txtY) {
        // Normal polarity: dark text on light background
        sapc = (powf(bgY, constants::normBG) - powf(txtY, constants::normTXT)) * constants::scaleBoW;
        if (sapc < constants::loClip) {
            output = 0.0f;
        } else {
            output = sapc - constants::loBoWoffset;
        }
    } else {
        // Reverse polarity: light text on dark background
        sapc = (powf(bgY, constants::revBG) - powf(txtY, constants::revTXT)) * constants::scaleWoB;
        if (sapc > -constants::loClip) {
            output = 0.0f;
        } else {
            output = sapc + constants::loWoBoffset;
        }
    }

    return output * 100.0f;
}

/**
 * Get the absolute contrast value (polarity-independent).
 * Useful when you just want to know "how much contrast" regardless of mode.
 */
COLOR_FUNC inline float contrast_abs(float text_r, float text_g, float text_b,
                                      float bg_r, float bg_g, float bg_b) {
    return fabsf(contrast(text_r, text_g, text_b, bg_r, bg_g, bg_b));
}

/**
 * Check if contrast meets minimum readability threshold.
 * @param min_lc: Minimum |Lc| value (75 for body text, 60 for large text, etc.)
 */
COLOR_FUNC inline bool is_readable(float text_r, float text_g, float text_b,
                                    float bg_r, float bg_g, float bg_b,
                                    float min_lc = 75.0f) {
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
    float L;
    float a;
    float b;
};

/**
 * Convert sRGB (0-255) to Oklab.
 */
COLOR_FUNC inline Lab from_srgb(float r, float g, float b) {
    // sRGB to linear RGB
    float lr = wcag2::linearize(r);
    float lg = wcag2::linearize(g);
    float lb = wcag2::linearize(b);

    // Linear RGB to LMS (cone response)
    float l = 0.4122214708f * lr + 0.5363325363f * lg + 0.0514459929f * lb;
    float m = 0.2119034982f * lr + 0.6806995451f * lg + 0.1073969566f * lb;
    float s = 0.0883024619f * lr + 0.2817188376f * lg + 0.6299787005f * lb;

    // Cube root (perceptual nonlinearity)
    float l_ = cbrtf(l);
    float m_ = cbrtf(m);
    float s_ = cbrtf(s);

    // LMS' to Oklab
    Lab result;
    result.L = 0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_;
    result.a = 1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_;
    result.b = 0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_;

    return result;
}

/**
 * Compute Euclidean distance between two colors in Oklab space.
 * This is a good approximation of perceptual color difference (ΔE).
 * Values around 0.02-0.03 are just noticeable differences.
 */
COLOR_FUNC inline float distance(float r1, float g1, float b1,
                                  float r2, float g2, float b2) {
    Lab lab1 = from_srgb(r1, g1, b1);
    Lab lab2 = from_srgb(r2, g2, b2);

    float dL = lab1.L - lab2.L;
    float da = lab1.a - lab2.a;
    float db = lab1.b - lab2.b;

    return sqrtf(dL * dL + da * da + db * db);
}

/**
 * Convert Oklab to linear RGB.
 * Returns values that may be outside [0,1] if out of sRGB gamut.
 */
COLOR_FUNC inline void to_linear_rgb(Lab lab, float* lr, float* lg, float* lb) {
    // Oklab to LMS' (inverse of final matrix)
    float l_ = lab.L + 0.3963377774f * lab.a + 0.2158037573f * lab.b;
    float m_ = lab.L - 0.1055613458f * lab.a - 0.0638541728f * lab.b;
    float s_ = lab.L - 0.0894841775f * lab.a - 1.2914855480f * lab.b;

    // LMS' to LMS (cube)
    float l = l_ * l_ * l_;
    float m = m_ * m_ * m_;
    float s = s_ * s_ * s_;

    // LMS to linear RGB (inverse of forward matrix)
    *lr = +4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s;
    *lg = -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s;
    *lb = -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s;
}

/**
 * Convert a single linear RGB channel to sRGB (0-255).
 * Applies gamma correction and clamps to valid range.
 */
COLOR_FUNC inline float linear_to_srgb_channel(float linear) {
    float v;
    if (linear <= 0.0031308f) {
        v = 12.92f * linear;
    } else {
        v = 1.055f * powf(linear, 1.0f / 2.4f) - 0.055f;
    }
    // Clamp to [0, 1] then scale to [0, 255]
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    return v * 255.0f;
}

/**
 * Convert Oklab to sRGB (0-255).
 * Clamps out-of-gamut colors.
 */
COLOR_FUNC inline void to_srgb(Lab lab, float* r, float* g, float* b) {
    float lr, lg, lb;
    to_linear_rgb(lab, &lr, &lg, &lb);
    *r = linear_to_srgb_channel(lr);
    *g = linear_to_srgb_channel(lg);
    *b = linear_to_srgb_channel(lb);
}

/**
 * Check if an Oklab color is within sRGB gamut.
 */
COLOR_FUNC inline bool is_in_gamut(Lab lab) {
    float lr, lg, lb;
    to_linear_rgb(lab, &lr, &lg, &lb);
    return lr >= -0.0001f && lr <= 1.0001f &&
           lg >= -0.0001f && lg <= 1.0001f &&
           lb >= -0.0001f && lb <= 1.0001f;
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
    float L;
    float C;
    float H;
};

/**
 * Convert sRGB (0-255) to OKLCH.
 */
COLOR_FUNC inline LCH from_srgb(float r, float g, float b) {
    oklab::Lab lab = oklab::from_srgb(r, g, b);

    LCH result;
    result.L = lab.L;
    result.C = sqrtf(lab.a * lab.a + lab.b * lab.b);
    result.H = atan2f(lab.b, lab.a) * 180.0f / PI;

    // Normalize hue to 0-360
    if (result.H < 0.0f) {
        result.H += 360.0f;
    }

    return result;
}

/**
 * Compute angular distance between two hues (handles wraparound).
 * Returns value in [0, 180].
 */
COLOR_FUNC inline float hue_distance(float h1, float h2) {
    float diff = fabsf(h1 - h2);
    if (diff > 180.0f) {
        diff = 360.0f - diff;
    }
    return diff;
}

/**
 * Check if two colors have similar hue (within tolerance).
 */
COLOR_FUNC inline bool hue_similar(float r1, float g1, float b1,
                                    float r2, float g2, float b2,
                                    float tolerance = 30.0f) {
    LCH lch1 = from_srgb(r1, g1, b1);
    LCH lch2 = from_srgb(r2, g2, b2);

    // If either color is achromatic, hue comparison is meaningless
    if (lch1.C < 0.02f || lch2.C < 0.02f) {
        return true;
    }

    return hue_distance(lch1.H, lch2.H) <= tolerance;
}

/**
 * Convert OKLCH to Oklab.
 */
COLOR_FUNC inline oklab::Lab to_oklab(float L, float C, float H) {
    float h_rad = H * PI / 180.0f;
    oklab::Lab lab;
    lab.L = L;
    lab.a = C * cosf(h_rad);
    lab.b = C * sinf(h_rad);
    return lab;
}

/**
 * Convert OKLCH to sRGB (0-255).
 * Clamps out-of-gamut colors.
 */
COLOR_FUNC inline void to_srgb(float L, float C, float H, float* r, float* g, float* b) {
    oklab::Lab lab = to_oklab(L, C, H);
    oklab::to_srgb(lab, r, g, b);
}

/**
 * Check if an OKLCH color is within sRGB gamut.
 */
COLOR_FUNC inline bool is_in_gamut(float L, float C, float H) {
    oklab::Lab lab = to_oklab(L, C, H);
    return oklab::is_in_gamut(lab);
}

/**
 * Find maximum chroma at given L and H that stays in sRGB gamut.
 * Uses binary search for efficiency.
 */
COLOR_FUNC inline float max_chroma_in_gamut(float L, float H) {
    float low = 0.0f;
    float high = 0.5f;  // Max reasonable chroma

    for (int i = 0; i < 16; i++) {  // Binary search iterations
        float mid = (low + high) * 0.5f;
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
COLOR_FUNC inline float normalize_hue(float h) {
    while (h < 0.0f) h += 360.0f;
    while (h >= 360.0f) h -= 360.0f;
    return h;
}

/**
 * Interpolate between two hues (circular interpolation).
 * @param h1 First hue (degrees)
 * @param h2 Second hue (degrees)
 * @param t Interpolation factor (0 = h1, 1 = h2)
 */
COLOR_FUNC inline float lerp_hue(float h1, float h2, float t) {
    float diff = h2 - h1;
    if (diff > 180.0f) diff -= 360.0f;
    if (diff < -180.0f) diff += 360.0f;
    return normalize_hue(h1 + t * diff);
}

} // namespace oklch

// =============================================================================
// Convenience Functions (Legacy Interface)
// =============================================================================
// These provide backward compatibility with the original function signatures.

COLOR_FUNC inline float linearize(float c) {
    return wcag2::linearize(c);
}

COLOR_FUNC inline float luminance(float r, float g, float b) {
    return wcag2::luminance(r, g, b);
}

COLOR_FUNC inline float contrast_ratio(float r1, float g1, float b1,
                                        float r2, float g2, float b2) {
    return wcag2::contrast_ratio(r1, g1, b1, r2, g2, b2);
}

COLOR_FUNC inline void rgb_to_oklab(float r, float g, float b,
                                     float* L, float* a, float* ok_b) {
    oklab::Lab lab = oklab::from_srgb(r, g, b);
    *L = lab.L;
    *a = lab.a;
    *ok_b = lab.b;
}

COLOR_FUNC inline void rgb_to_oklch(float r, float g, float b,
                                     float* L, float* C, float* H) {
    oklch::LCH lch = oklch::from_srgb(r, g, b);
    *L = lch.L;
    *C = lch.C;
    *H = lch.H;
}

COLOR_FUNC inline float oklab_distance(float r1, float g1, float b1,
                                        float r2, float g2, float b2) {
    return oklab::distance(r1, g1, b1, r2, g2, b2);
}

COLOR_FUNC inline float hue_distance(float h1, float h2) {
    return oklch::hue_distance(h1, h2);
}

COLOR_FUNC inline void oklch_to_srgb(float L, float C, float H,
                                      float* r, float* g, float* b) {
    oklch::to_srgb(L, C, H, r, g, b);
}

COLOR_FUNC inline bool oklch_in_gamut(float L, float C, float H) {
    return oklch::is_in_gamut(L, C, H);
}

COLOR_FUNC inline float oklch_max_chroma(float L, float H) {
    return oklch::max_chroma_in_gamut(L, H);
}

} // namespace color

#endif // COLOR_CUH
