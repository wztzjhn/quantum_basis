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
    T continued_fraction(T a[], T b[], const MKL_INT &len)
    {
        T res = static_cast<T>(0.0);
        for (MKL_INT j = len - 1; j > 0; j--) res = b[j] / (a[j] + res);
        return a[0] + res;
    }
    template double continued_fraction(double a[], double b[], const MKL_INT &len);
    template std::complex<double> continued_fraction(std::complex<double> a[], std::complex<double> b[], const MKL_INT &len);
    
}

