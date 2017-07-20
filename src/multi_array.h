#ifndef multi_array_h
#define multi_array_h

#include <cstdint>
#include <vector>
#include <iostream>
#include <cassert>

namespace qbasis {
    template <typename> class multi_array;
    
    template <typename T> void swap(multi_array<T>&, multi_array<T>&);
    
    template <typename T> class multi_array {
        friend void swap <> (multi_array<T>&, multi_array<T>&);
    public:
        uint32_t dim;
        uint64_t size;
        std::vector<uint64_t> linear_size;
        
        multi_array(): dim(0), size(0) {}
        
        multi_array(const std::vector<uint64_t> &linear_size_input);
        
        multi_array(const std::vector<uint64_t> &linear_size_input, const T &element);
        
        // copy constructor
        multi_array(const multi_array<T> &old):
            dim(old.dim),
            size(old.size),
            linear_size(old.linear_size),
            data(old.data) {}
        
        // move constructor
        multi_array(multi_array<T> &&old) noexcept :
            dim(old.dim),
            size(old.size),
            linear_size(std::move(old.linear_size)),
            data(std::move(old.data)) {}
        
        // copy assignment constructor and move assignment constructor, using "swap and copy"
        multi_array& operator=(multi_array<T> old) { swap(*this, old); return *this; }
        
        multi_array& operator=(const T &element);
        
        ~multi_array() {}
        
        T& index(const std::vector<uint64_t> &pos);
        
        const T& index(const std::vector<uint64_t> &pos) const;
        
    private:
        std::vector<T> data;
    };
    
}

#endif