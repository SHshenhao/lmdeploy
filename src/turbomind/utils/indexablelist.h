#pragma once
#include <list>

namespace turbomind {

template <typename T>
class IndexableList : public std::list<T> {
public:
    IndexableList() : std::list<T>() {}
    
    IndexableList(std::initializer_list<T> initList) : std::list<T>(initList) {}

    IndexableList(std::size_t initialSize) {
        for (std::size_t i = 0; i < initialSize; ++i) {
            this->emplace_back(); 
        }
    }

    T& operator[](std::size_t index) {
        auto it = this->begin();
        std::advance(it, index);
        return *it;
    }

    const T& operator[](std::size_t index) const {
        auto it = this->begin();
        std::advance(it, index);
        return *it;
    }
};

}  // namespace turbomind