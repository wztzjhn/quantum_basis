#include "multi_array.h"

namespace qbasis {
    
    template <typename T>
    multi_array<T>::multi_array(const std::vector<uint64_t> &linear_size_input):
        dim(linear_size_input.size()),
        linear_size(linear_size_input)
    {
        assert(dim > 0);
        size = linear_size[0];
        for (decltype(linear_size.size()) j = 1; j < linear_size.size(); j++)
            size *= linear_size[j];
        data = std::vector<T>(size);
    }
    
    template <typename T>
    multi_array<T>::multi_array(const std::vector<uint64_t> &linear_size_input, const T &element):
        dim(linear_size_input.size()),
        linear_size(linear_size_input)
    {
        assert(dim > 0);
        size = linear_size[0];
        for (decltype(linear_size.size()) j = 1; j < linear_size.size(); j++)
            size *= linear_size[j];
        data = std::vector<T>(size, element);
    }
    
    template <typename T>
    multi_array<T>& multi_array<T>::operator=(const T &element)
    {
        for (uint64_t j = 0; j < size; j++) data[j] = element;
        return *this;
    }
    
    template <typename T>
    T& multi_array<T>::index(const std::vector<uint64_t> &pos)
    {
        assert(pos.size() == linear_size.size());
        uint64_t res = 0;
        auto j = linear_size.size() - 1;
        while (pos[j] == 0 && j > 0) j--;
        while (j > 0) {
            assert(pos[j] < linear_size[j]);
            res = (res + pos[j]) * linear_size[j-1];
            j--;
        }
        assert(pos[0] < linear_size[0]);
        res += pos[0];
        return data[res];
    }

    template <typename T>
    const T& multi_array<T>::index(const std::vector<uint64_t> &pos) const
    {
        assert(pos.size() == linear_size.size());
        uint64_t res = 0;
        auto j = linear_size.size() - 1;
        while (pos[j] == 0 && j > 0) j--;
        while (j > 0) {
            assert(pos[j] < linear_size[j]);
            res = (res + pos[j]) * linear_size[j-1];
            j--;
        }
        assert(pos[0] < linear_size[0]);
        res += pos[0];
        return data[res];
    }
    
    template <typename T>
    void swap(multi_array<T> &lhs, multi_array<T> &rhs)
    {
        using std::swap;
        swap(lhs.dim, rhs.dim);
        swap(lhs.size, rhs.size);
        swap(lhs.linear_size, rhs.linear_size);
        swap(lhs.data, rhs.data);
    }

    // explicit instantiation
    template class multi_array<double>;
    template class multi_array<std::pair<std::vector<uint32_t>,std::vector<uint32_t>>>;
    
    template void swap(multi_array<double>&, multi_array<double>&);
    template void swap(multi_array<std::pair<std::vector<uint32_t>,std::vector<uint32_t>>>&,
                       multi_array<std::pair<std::vector<uint32_t>,std::vector<uint32_t>>>&);
}
