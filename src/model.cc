#include <iostream>
#include "qbasis.h"

namespace qbasis {

    template <typename T>
    MKL_INT binary_search(const std::vector<T> &basis_all, const T &val)
    {
        MKL_INT low = 0;
        MKL_INT high = basis_all.size() - 1;
        MKL_INT mid;
        while(low <= high) {
            mid = (low + high) / 2;
            if (val == basis_all[mid]) return mid;
            else if (val < basis_all[mid]) high = mid - 1;
            else low = mid + 1;
        }
        assert(false);
        return -1;
    }
    
    
    // Explicit instantiation
    template MKL_INT binary_search(const std::vector<qbasis::mbasis_elem> &basis_all, const qbasis::mbasis_elem &val);

}
