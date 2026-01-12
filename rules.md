# Hexa Color Solver Optimization Rules

This document describes the complete optimization rules used by the genetic algorithm color solver to generate WCAG-compliant terminal color palettes.

## Color Palette Structure

The solver generates a 16-color ANSI terminal palette:

| Index | Name       | Status    | Notes                          |
|-------|------------|-----------|--------------------------------|
| 0     | Black      | Fixed     | Always `#000000`               |
| 1     | Red        | Optimized | Base color                     |
| 2     | Green      | Optimized | Base color                     |
| 3     | Yellow     | Optimized | Base color                     |
| 4     | Blue       | Optimized | Base color                     |
| 5     | Magenta    | Optimized | Base color                     |
| 6     | Cyan       | Optimized | Base color                     |
| 7     | White      | Optimized | Base color                     |
| 8     | Br.Black   | Optimized | Bright variant                 |
| 9     | Br.Red     | Optimized | Bright variant                 |
| 10    | Br.Green   | Optimized | Bright variant                 |
| 11    | Br.Yellow  | Optimized | Bright variant                 |
| 12    | Br.Blue    | Optimized | Bright variant                 |
| 13    | Br.Magenta | Optimized | Bright variant                 |
| 14    | Br.Cyan    | Optimized | Bright variant                 |
| 15    | Br.White   | Fixed     | Always `#ffffff`               |

## Hard Constraints

These constraints must be satisfied for a palette to be considered valid. Violations result in severe fitness penalties.

### Constraint 1: Base Colors on Black

Colors 1-7 must meet minimum WCAG contrast ratios against black (`#000000`).

| FG Color | Min Contrast | Rationale                        |
|----------|--------------|----------------------------------|
| Red      | 5.5:1        | Force brighter red for readability |
| Green    | 4.5:1        | Needs higher minimum             |
| Yellow   | 3.5:1        | Fine as is                       |
| Blue     | 3.5:1        | Fine as is                       |
| Magenta  | 3.5:1        | Fine as is                       |
| Cyan     | 3.5:1        | Fine as is                       |
| White    | 4.5:1        | Should have good contrast        |

**Penalty**: `(required - actual) * 500.0` per violation (PRIMARY)
**Reward**: `100 + cr * 25` when in range

### Constraint 2: Bright Colors on Regular Colors

Each bright color (8-14) must meet contrast requirements against its corresponding base color (0-6).

| Bright Color | Base Color | Target Contrast | Rationale                    |
|--------------|------------|-----------------|------------------------------|
| Br.Black (8) | Black (0)  | 2.2:1           | Special lower target         |
| Br.Red (9)   | Red (1)    | 2.6:1           | Standard bright/base         |
| Br.Green (10)| Green (2)  | 2.6:1           | Standard bright/base         |
| Br.Yellow (11)| Yellow (3)| 2.6:1           | Standard bright/base         |
| Br.Blue (12) | Blue (4)   | 2.6:1           | Standard bright/base         |
| Br.Magenta (13)| Magenta (5)| 2.6:1         | Standard bright/base         |
| Br.Cyan (14) | Cyan (6)   | 2.6:1           | Standard bright/base         |

**Penalty**: `(required - actual) * 100.0` per violation (lower priority)

### Constraint 3: FM Pairs on Blue (PRIMARY)

Foreground-on-medium pairs: certain colors must be readable on blue background.

| FG Color | BG Color | Min Contrast | Rationale                    |
|----------|----------|--------------|------------------------------|
| Red      | Blue     | 3.0:1        | Common syntax highlighting   |
| Green    | Blue     | 3.0:1        | Common syntax highlighting   |
| Yellow   | Blue     | 3.0:1        | Common syntax highlighting   |
| Magenta  | Blue     | 3.0:1        | Common syntax highlighting   |
| Cyan     | Blue     | 3.0:1        | Common syntax highlighting   |
| White    | Blue     | 3.0:1        | Selection/status bar text    |

**Penalty**: `(required - actual) * 400.0` per violation (PRIMARY)
**Reward**: `80 + cr * 15` when met

### Constraint 4: FM Pairs on Green (PRIMARY)

Foreground-on-medium pairs: certain colors must be readable on green background.

| FG Color | BG Color | Min Contrast | Rationale                    |
|----------|----------|--------------|------------------------------|
| Red      | Green    | 2.5:1        | Common syntax highlighting   |
| Yellow   | Green    | 2.5:1        | Common syntax highlighting   |
| Blue     | Green    | 2.5:1        | Common syntax highlighting   |
| Magenta  | Green    | 2.5:1        | Common syntax highlighting   |
| Cyan     | Green    | 2.5:1        | Common syntax highlighting   |
| White    | Green    | 2.5:1        | Selection/status bar text    |

**Penalty**: `(required - actual) * 350.0` per violation (PRIMARY)
**Reward**: `60 + cr * 12` when met

## Bonus Rules (Soft Constraints)

These rules provide fitness bonuses to encourage perceptually better palettes.

### Color Distinctiveness (Weight: 100.0)

Colors 1-7 should be visually distinct from each other.

| Metric          | Threshold     | Effect                           |
|-----------------|---------------|----------------------------------|
| RGB distance    | > 80 units    | Bonus for exceeding threshold    |
| Color pairs     | All (1-7)×(1-7) | Pairwise comparison            |

**Bonus**: `min(distance - 80, 50) / 50 * weight` for each pair exceeding threshold

### Bright Colors on Black (Weight: 50.0)

Bright colors (8-14) should have good contrast on black background.

| FG Colors | BG Color | Target  | Effect                        |
|-----------|----------|---------|-------------------------------|
| 8-14      | Black    | 4.5:1   | Bonus for meeting/exceeding   |

**Bonus**: `(contrast - 3.0) / 1.5 * weight`, clamped to [0, weight]

### OKLCH Hue Spacing (Weight: 150.0)

Chromatic colors (1-6) should have evenly distributed hues in OKLCH color space.

| Colors    | Ideal Spacing | Tolerance | Effect                       |
|-----------|---------------|-----------|------------------------------|
| 1-6       | 60°           | ±15°      | Bonus for staying in range   |

**Bonus**: Full weight if within ±15° of ideal 60° spacing; linear falloff to 0 at ±30°

### Perceptual Distance (Weight: 200.0)

Adjacent colors (1-7) should be perceptually distinct in Oklab color space.

| Color Pairs | Min Distance | Effect                          |
|-------------|--------------|----------------------------------|
| Adjacent (1-7) | 0.15 ΔE   | Bonus for exceeding threshold   |

**Bonus**: `(distance / 0.15) * weight` for each adjacent pair

### Bright/Base Hue Matching (Weight: 100.0)

Bright colors should have similar hue to their base counterparts.

| Bright | Base    | Max Hue Diff | Effect                       |
|--------|---------|--------------|------------------------------|
| 9      | 1 (Red) | 30°          | Bonus for matching hue       |
| 10     | 2 (Grn) | 30°          | Bonus for matching hue       |
| 11     | 3 (Yel) | 30°          | Bonus for matching hue       |
| 12     | 4 (Blu) | 30°          | Bonus for matching hue       |
| 13     | 5 (Mag) | 30°          | Bonus for matching hue       |
| 14     | 6 (Cyn) | 30°          | Bonus for matching hue       |

**Bonus**: Full weight if hue difference < 30°; linear falloff to 0 at 60°

## Color Search Ranges

The genetic algorithm searches within these RGB ranges to maintain color identity.

### Base Colors (1-7)

| Color   | R Range   | G Range   | B Range   |
|---------|-----------|-----------|-----------|
| Red     | 128-255   | 0-100     | 0-100     |
| Green   | 0-100     | 128-255   | 0-100     |
| Yellow  | 180-255   | 180-255   | 0-100     |
| Blue    | 0-150     | 0-150     | 128-255   |
| Magenta | 128-255   | 0-128     | 128-255   |
| Cyan    | 0-100     | 128-255   | 128-255   |
| White   | 170-220   | 170-220   | 170-220   |

### Bright Colors (8-14)

| Color      | R Range   | G Range   | B Range   |
|------------|-----------|-----------|-----------|
| Br.Black   | 80-140    | 80-140    | 80-140    |
| Br.Red     | 200-255   | 50-150    | 50-150    |
| Br.Green   | 50-150    | 200-255   | 50-150    |
| Br.Yellow  | 220-255   | 220-255   | 80-180    |
| Br.Blue    | 80-180    | 80-180    | 200-255   |
| Br.Magenta | 200-255   | 80-180    | 200-255   |
| Br.Cyan    | 80-180    | 220-255   | 220-255   |

## Fitness Function Summary

The total fitness is computed as:

```
fitness = base_fitness - constraint_penalties + bonus_rewards
```

Where:
- `base_fitness` = 1000.0
- `constraint_penalties` = sum of hard constraint violations
- `bonus_rewards` = sum of soft constraint bonuses

Higher fitness indicates a better palette.

## Color Space Conversions

The solver uses multiple color spaces:

| Space  | Purpose                                    |
|--------|-------------------------------------------|
| sRGB   | Input/output, constraint evaluation        |
| Linear RGB | Luminance calculation for WCAG contrast |
| Oklab  | Perceptual distance calculations          |
| OKLCH  | Hue spacing and matching                  |

### WCAG Contrast Ratio Formula

```
L = 0.2126 * R + 0.7152 * G + 0.0722 * B  (where R,G,B are linearized)
contrast = (L1 + 0.05) / (L2 + 0.05)  (L1 >= L2)
```

### Linearization (sRGB to Linear)

```
if (sRGB <= 0.04045)
    linear = sRGB / 12.92
else
    linear = pow((sRGB + 0.055) / 1.055, 2.4)
```
