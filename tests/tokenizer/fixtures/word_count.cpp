#include <cctype>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace text {

struct WordCount {
    std::string word;
    std::uint32_t count = 0;
};

std::vector<std::string> split_words(const std::string& line) {
    std::vector<std::string> words;
    std::string current;
    for (unsigned char ch : line) {
        if (std::isalnum(ch) || ch == '_') {
            current.push_back(static_cast<char>(std::tolower(ch)));
        } else if (!current.empty()) {
            words.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) {
        words.push_back(current);
    }
    return words;
}

std::vector<WordCount> count_words(std::istream& input) {
    std::map<std::string, std::uint32_t> counts;
    std::string line;
    while (std::getline(input, line)) {
        for (const auto& word : split_words(line)) {
            counts[word] += 1;
        }
    }
    std::vector<WordCount> result;
    for (const auto& [word, count] : counts) {
        result.push_back({word, count});
    }
    return result;
}

}  // namespace text

int main() {
    for (const auto& entry : text::count_words(std::cin)) {
        std::cout << entry.word << '\t' << entry.count << '\n';
    }
    return 0;
}
