#include "qbasis.h"

namespace qbasis {
    
    MKL_INT int_pow(const MKL_INT &base, const MKL_INT &index)
    {
        assert(index >= 0);
        MKL_INT res = 1;
        for (MKL_INT j = 0; j < index; j++) res *= base;
        return res;
    }
    
    MKL_INT dynamic_base(const std::vector<MKL_INT> &nums, const std::vector<MKL_INT> &base)
    {
        assert(nums.size() == base.size());
        assert(nums.size() > 0);
        MKL_INT res = 0;
        for (MKL_INT j = nums.size() - 1; j > 0; j--) {
            assert(nums[j] < base[j]);
            res = (res + nums[j]) * base[j-1];
        }
        assert(nums[0] < base[0]);
        res += nums[0];
        return res;
    }
    
    std::vector<MKL_INT> dynamic_base(const MKL_INT &total, const std::vector<MKL_INT> &base)
    {
        assert(base.size() > 0);
        assert(total >= 0);
        MKL_INT len = base.size();
        std::vector<MKL_INT> res(len);
        auto temp = total;     // temp == i + j * base[0] + k * base[0] * base[1] + ...
        for (MKL_INT n = 0; n < len - 1; n++) {
            res[n] = temp % base[n];
            temp = (temp - res[n]) / base[n];
        }
        res[len-1] = temp;
        assert(temp < base[len-1]);
        return res;
    }
    
    template <typename T>
    MKL_INT binary_search(const std::vector<T> &array, const T &val,
                          const MKL_INT &bgn, const MKL_INT &end)
    {
        if(array.size() == 0 && bgn == end) return -1;            // not found
        assert(bgn < end);
        assert(bgn >= 0 && bgn < array.size());
        assert(end > 0 && end <= array.size());
        MKL_INT low  = bgn;
        MKL_INT high = end - 1;
        MKL_INT mid;
        while(low <= high) {
            mid = (low + high) / 2;
            if (val == array[mid]) return mid;
            else if (val < array[mid]) high = mid - 1;
            else low = mid + 1;
        }
        return -1;
    }
    template MKL_INT binary_search(const std::vector<qbasis::mbasis_elem> &array, const qbasis::mbasis_elem &val,
                                   const MKL_INT &bgn, const MKL_INT &end);
    template MKL_INT binary_search(const std::vector<MKL_INT> &array, const MKL_INT &val,
                                   const MKL_INT &bgn, const MKL_INT &end);
    
    
    template <typename T>
    T continued_fraction(T a[], T b[], const MKL_INT &len)
    {
        T res = static_cast<T>(0.0);
        for (MKL_INT j = len - 1; j > 0; j--) res = b[j] / (a[j] + res);
        return a[0] + res;
    }
    template double continued_fraction(double a[], double b[], const MKL_INT &len);
    template std::complex<double> continued_fraction(std::complex<double> a[], std::complex<double> b[], const MKL_INT &len);
    
}

