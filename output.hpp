/**
 * Output Module - FTXUI-based terminal output for color palette results
 */

#ifndef OUTPUT_HPP
#define OUTPUT_HPP

#include <vector>
#include <string>
#include <cmath>
#include <iostream>

#include <ftxui/dom/elements.hpp>
#include <ftxui/dom/table.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/screen/color.hpp>

#include "color.cuh"

namespace output {

struct ColorRGB {
    double r, g, b;

    ColorRGB(double* palette, int index) {
        r = palette[index * 3 + 0];
        g = palette[index * 3 + 1];
        b = palette[index * 3 + 2];
    }

    ftxui::Color to_color() const { return ftxui::Color::RGB((int)r, (int)g, (int)b); }

    std::string hex() const {
        char buf[8];
        snprintf(buf, sizeof(buf), "#%02x%02x%02x", (int)r, (int)g, (int)b);
        return buf;
    }
};

struct ContrastResult {
    double apca;
    bool apca_pass;

    ContrastResult(ColorRGB fg, ColorRGB bg, double apca_target) {
        apca = ::color::apca::contrast(fg.r, fg.g, fg.b, bg.r, bg.g, bg.b);
        apca_pass = fabs(apca) >= apca_target;
    }

    bool pass() const { return apca_pass; }
};

// APCA Lc contrast levels:
// >= 90: Preferred for body text
// >= 75: Minimum for body text
// >= 60: Large text minimum
// >= 45: Non-text, large bold minimum
// < 45: Fail for most uses
inline ftxui::Color apca_status_color(double lc) {
    double abs_lc = fabs(lc);
    if (abs_lc >= 90.0) return ftxui::Color::Cyan;
    if (abs_lc >= 75.0) return ftxui::Color::Green;
    if (abs_lc >= 60.0) return ftxui::Color::Yellow;
    if (abs_lc >= 45.0) return ftxui::Color::RGB(255, 165, 0); // Orange
    return ftxui::Color::Red;
}

inline const char* apca_status_symbol(double lc) {
    double abs_lc = fabs(lc);
    if (abs_lc >= 90.0) return "★";  // Excellent
    if (abs_lc >= 75.0) return "✓";  // Body text
    if (abs_lc >= 60.0) return "~";  // Large text
    if (abs_lc >= 45.0) return "○";  // Non-text/bold
    return "✗";                        // Fail
}

struct ContrastPair {
    int fg_index;
    std::string fg_name;
};

inline ftxui::Element make_contrast_table(
    const std::string& title,
    double apca_target,
    double* palette,
    const std::vector<ContrastPair>& pairs,
    ColorRGB bg,
    const std::string& bg_name
) {
    namespace f = ftxui;

    std::vector<std::vector<f::Element>> rows;

    rows.push_back({
        f::text("Pair") | f::bold,
        f::text("APCA") | f::bold
    });

    for (const auto& pair : pairs) {
        ColorRGB fg(palette, pair.fg_index);
        ContrastResult cr(fg, bg, apca_target);

        char apca_str[24];
        snprintf(apca_str, sizeof(apca_str), "%s%6.1f", apca_status_symbol(cr.apca), cr.apca);

        std::string pair_label = " " + pair.fg_name + " on " + bg_name + " ";

        rows.push_back({
            f::text(pair_label) | f::color(fg.to_color()) | f::bgcolor(bg.to_color()),
            f::text(apca_str) | f::color(apca_status_color(cr.apca))
        });
    }

    auto table = f::Table(rows);
    table.SelectAll().SeparatorVertical(f::LIGHT);
    table.SelectRow(0).BorderBottom(f::LIGHT);

    char header[128];
    snprintf(header, sizeof(header), "%s on %s (APCA≥%.0f)",
             title.c_str(), bg_name.c_str(), apca_target);

    return f::vbox({
        f::text(header) | f::bold,
        f::separator(),
        table.Render()
    });
}

inline ftxui::Element make_bright_on_regular_table(
    double* palette,
    const char** names,
    double apca_target,
    double apca_target_black
) {
    namespace f = ftxui;

    std::vector<std::vector<f::Element>> rows;

    rows.push_back({
        f::text("Pair") | f::bold,
        f::text("APCA") | f::bold
    });

    for (int i = 0; i <= 7; i++) {
        ColorRGB reg(palette, i);
        ColorRGB brt(palette, i + 8);
        double target = (i == 0) ? apca_target_black : apca_target;

        ContrastResult cr(brt, reg, target);

        char apca_str[24];
        snprintf(apca_str, sizeof(apca_str), "%s%6.1f", apca_status_symbol(cr.apca), cr.apca);

        std::string pair_label = std::string(" br.") + names[i] + " on " + names[i] + " ";

        rows.push_back({
            f::text(pair_label) | f::color(brt.to_color()) | f::bgcolor(reg.to_color()),
            f::text(apca_str) | f::color(apca_status_color(cr.apca))
        });
    }

    auto table = f::Table(rows);
    table.SelectAll().SeparatorVertical(f::LIGHT);
    table.SelectRow(0).BorderBottom(f::LIGHT);

    char header[128];
    snprintf(header, sizeof(header), "Bright on Regular (APCA≥%.0f, br.black≥%.0f)",
             apca_target, apca_target_black);

    return f::vbox({
        f::text(header) | f::bold,
        f::separator(),
        table.Render()
    });
}

inline ftxui::Element make_palette_table(
    double* palette,
    const char** names
) {
    namespace f = ftxui;

    ColorRGB black(palette, 0);
    ColorRGB green(palette, 2);
    ColorRGB blue(palette, 4);
    ColorRGB cyan(palette, 6);

    // Background colors with their indices for "---" detection
    struct BgInfo { ColorRGB color; int index; const char* name; };
    std::vector<BgInfo> backgrounds = {
        {black, 0, "Black"},
        {blue, 4, "Blue"},
        {cyan, 6, "Cyan"},
        {green, 2, "Green"}
    };

    std::vector<std::vector<f::Element>> rows;

    // Header row
    std::vector<f::Element> header = {
        f::text(" # ") | f::bold,
        f::text(" Name ") | f::bold,
        f::text("  ") | f::bold,
        f::text(" Hex ") | f::bold
    };
    // APCA columns per background
    for (const auto& bg : backgrounds) {
        char col_name[32];
        snprintf(col_name, sizeof(col_name), " on %s ", bg.name);
        header.push_back(f::text(col_name) | f::bold);
    }
    rows.push_back(header);

    for (int i = 0; i < 16; i++) {
        ColorRGB col(palette, i);

        char hex_str[12];
        snprintf(hex_str, sizeof(hex_str), " %s ", col.hex().c_str());

        char idx_str[8];
        snprintf(idx_str, sizeof(idx_str), " %2d ", i);

        std::vector<f::Element> row = {
            f::text(idx_str),
            f::text(std::string(" ") + names[i] + " "),
            f::text("    ") | f::bgcolor(col.to_color()),
            f::text(hex_str)
        };

        // APCA columns
        for (const auto& bg : backgrounds) {
            if (i == bg.index) {
                row.push_back(f::text("  ---  ") | f::bgcolor(bg.color.to_color()));
            } else {
                double apca = ::color::apca::contrast(col.r, col.g, col.b, bg.color.r, bg.color.g, bg.color.b);
                char apca_str[24];
                snprintf(apca_str, sizeof(apca_str), "%s%6.1f",
                         apca_status_symbol(apca), apca);
                row.push_back(f::text(apca_str) | f::color(col.to_color()) | f::bgcolor(bg.color.to_color()));
            }
        }

        rows.push_back(row);
    }

    auto table = f::Table(rows);
    table.SelectAll().SeparatorVertical(f::LIGHT);
    table.SelectRow(0).BorderBottom(f::LIGHT);

    return f::vbox({
        f::text("Optimized Palette") | f::bold,
        f::separator(),
        table.Render()
    });
}

inline ftxui::Element make_sample_matrix(
    double* palette
) {
    namespace f = ftxui;

    std::vector<std::vector<f::Element>> rows;

    // Header row with background indices
    std::vector<f::Element> header_row;
    header_row.push_back(f::text(" FG\\BG") | f::bold);
    for (int bg = 0; bg < 16; bg++) {
        char bg_str[8];
        snprintf(bg_str, sizeof(bg_str), " %02d ", bg);
        header_row.push_back(f::text(bg_str) | f::bold);
    }
    rows.push_back(header_row);

    // Data rows
    for (int fg = 0; fg < 16; fg++) {
        ColorRGB fg_col(palette, fg);

        std::vector<f::Element> row;
        char fg_str[8];
        snprintf(fg_str, sizeof(fg_str), "  %02d  ", fg);
        row.push_back(f::text(fg_str) | f::bold);

        for (int bg = 0; bg < 16; bg++) {
            ColorRGB bg_col(palette, bg);
            double apca = ::color::apca::contrast(fg_col.r, fg_col.g, fg_col.b, bg_col.r, bg_col.g, bg_col.b);

            char cell_str[12];
            snprintf(cell_str, sizeof(cell_str), "%s%4.0f",
                     apca_status_symbol(apca), apca);
            row.push_back(f::text(cell_str) | f::color(fg_col.to_color()) | f::bgcolor(bg_col.to_color()));
        }
        rows.push_back(row);
    }

    auto table = f::Table(rows);
    table.SelectAll().SeparatorVertical(f::LIGHT);
    table.SelectRow(0).BorderBottom(f::LIGHT);
    table.SelectColumn(0).BorderRight(f::LIGHT);

    return f::vbox({
        f::text("APCA Contrast Matrix (FG on BG)") | f::bold,
        f::separator(),
        table.Render()
    });
}

inline void print_palette_and_matrix(
    double* palette,
    const char** names
) {
    namespace f = ftxui;

    auto palette_table = make_palette_table(palette, names);
    auto matrix = make_sample_matrix(palette);

    // Print palette table
    auto palette_screen = f::Screen::Create(f::Dimension::Fit(palette_table));
    f::Render(palette_screen, palette_table);
    palette_screen.Print();
    std::cout << std::endl;

    // Print contrast matrix below
    auto matrix_screen = f::Screen::Create(f::Dimension::Fit(matrix));
    f::Render(matrix_screen, matrix);
    matrix_screen.Print();
    std::cout << std::endl;
}

// Print FM pairs tables (APCA-focused)
inline void print_fm_pairs_tables(
    double* palette,
    const char** names
) {
    namespace f = ftxui;

    ColorRGB blue(palette, 4);   // BLUE
    ColorRGB green(palette, 2);  // GREEN
    ColorRGB cyan(palette, 6);   // CYAN

    std::vector<ContrastPair> pairs_on_blue = {
        {0, "Black"}, {1, "Red"}, {2, "Green"}, {3, "Yellow"},
        {5, "Magenta"}, {6, "Cyan"}, {7, "White"}
    };

    std::vector<ContrastPair> pairs_on_green = {
        {0, "Black"}, {1, "Red"}, {3, "Yellow"}, {4, "Blue"},
        {5, "Magenta"}, {6, "Cyan"}, {7, "White"}
    };

    std::vector<ContrastPair> pairs_on_cyan = {
        {0, "Black"}, {1, "Red"}, {2, "Green"}, {3, "Yellow"},
        {4, "Blue"}, {5, "Magenta"}, {7, "White"}
    };

    // APCA thresholds matching our constraints
    const double APCA_BRIGHT_ON_REGULAR = 30.0;
    const double APCA_BR_BLACK_ON_BLACK = 15.0;
    const double APCA_ON_BLUE = 30.0;
    const double APCA_ON_GREEN = 30.0;
    const double APCA_ON_CYAN = 20.0;

    auto table1 = make_bright_on_regular_table(
        palette, names, APCA_BRIGHT_ON_REGULAR, APCA_BR_BLACK_ON_BLACK
    );

    auto table2 = make_contrast_table(
        "Colors", APCA_ON_BLUE, palette, pairs_on_blue, blue, "blue"
    );

    auto table3 = make_contrast_table(
        "Colors", APCA_ON_GREEN, palette, pairs_on_green, green, "green"
    );

    auto table4 = make_contrast_table(
        "Colors", APCA_ON_CYAN, palette, pairs_on_cyan, cyan, "cyan"
    );

    auto layout = f::hbox({
        table1,
        f::text("  "),
        table2,
        f::text("  "),
        table3,
        f::text("  "),
        table4
    });

    auto screen = f::Screen::Create(f::Dimension::Fit(layout));
    f::Render(screen, layout);
    screen.Print();
    std::cout << std::endl;

    // APCA legend
    std::cout << "APCA: \033[36m★\033[0m≥90 \033[32m✓\033[0m≥75(body) \033[33m~\033[0m≥60(large) \033[38;2;255;165;0m○\033[0m≥45(bold) \033[31m✗\033[0m<45" << std::endl;
}

} // namespace output

#endif // OUTPUT_HPP
