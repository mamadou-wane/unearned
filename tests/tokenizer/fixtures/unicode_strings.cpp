#include <string>
#include <string_view>

// UTF-8 comment: naïve café 日本語
constexpr std::string_view kGreeting = "héllo, wörld";
constexpr std::string_view kSymbols = "∀x ∈ ℝ: x² ≥ 0";

std::string repeat(std::string_view text, int times) {
    std::string out;
    for (int i = 0; i < times; ++i) {
        out += text;
    }
    return out;
}
