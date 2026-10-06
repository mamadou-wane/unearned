#include <array>
#include <cstddef>
#include <optional>

template <typename T, std::size_t Capacity>
class RingBuffer {
public:
    bool push(const T& value) {
        if (size_ == Capacity) {
            return false;
        }
        items_[(head_ + size_) % Capacity] = value;
        ++size_;
        return true;
    }

    std::optional<T> pop() {
        if (size_ == 0) {
            return std::nullopt;
        }
        T value = items_[head_];
        head_ = (head_ + 1) % Capacity;
        --size_;
        return value;
    }

    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    bool full() const { return size_ == Capacity; }

private:
    std::array<T, Capacity> items_{};
    std::size_t head_ = 0;
    std::size_t size_ = 0;
};

int main() {
    RingBuffer<int, 4> buffer;
    for (int i = 0; i < 6; ++i) {
        buffer.push(i * i);
    }
    int total = 0;
    while (auto value = buffer.pop()) {
        total += *value;
    }
    return total == 14 ? 0 : 1;
}
