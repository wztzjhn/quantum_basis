#include "qbasis.h"

namespace qbasis {
    template <typename T1, typename T2>
    T2 int_pow(const T1 &base, const T1 &index)
    {
        T2 res = 1;
        for (T1 j = 0; j < index; j++) res *= base;
        return res;
    }
    template uint64_t int_pow(const uint8_t &base, const uint8_t &index);
    template uint64_t int_pow(const uint32_t &base, const uint32_t &index);
    template MKL_INT int_pow(const MKL_INT &base, const MKL_INT &index);
    
    template <typename T1, typename T2>
    T2 dynamic_base(const std::vector<T1> &nums, const std::vector<T1> &base)
    {
        assert(nums.size() == base.size());
        assert(nums.size() > 0);
        T2 res = 0;
        auto j = nums.size() - 1;
        while (nums[j] == 0 && j > 0) j--;
        while (j > 0) {
            assert(nums[j] < base[j]);
            res = (res + nums[j]) * base[j-1];
            j--;
        }
        assert(nums[0] < base[0]);
        res += nums[0];
        return res;
    }
    template uint32_t dynamic_base(const std::vector<uint8_t> &nums, const std::vector<uint8_t> &base);
    template uint64_t dynamic_base(const std::vector<uint8_t> &nums, const std::vector<uint8_t> &base);
    template uint32_t dynamic_base(const std::vector<uint32_t> &nums, const std::vector<uint32_t> &base);
    template uint64_t dynamic_base(const std::vector<uint64_t> &nums, const std::vector<uint64_t> &base);
    template MKL_INT dynamic_base(const std::vector<MKL_INT> &nums, const std::vector<MKL_INT> &base);
    
    template <typename T>
    std::vector<T> dynamic_base(const T &total, const std::vector<T> &base)
    {
        assert(base.size() > 0);
        assert(total >= 0);
        auto len = base.size();
        std::vector<T> res(len);
        T temp = total;        // temp == i + j * base[0] + k * base[0] * base[1] + ...
        for (decltype(len) n = 0; n < len - 1; n++) {
            res[n] = temp % base[n];
            temp = (temp - res[n]) / base[n];
            if (temp == 0) {
                for (decltype(len) j = n + 1; j < len - 1; j++) res[j] = 0;
                break;
            }
        }
        res[len-1] = temp;
        assert(temp < base[len-1]);
        return res;
    }
    template std::vector<MKL_INT> dynamic_base(const MKL_INT &total, const std::vector<MKL_INT> &base);
    template std::vector<uint32_t> dynamic_base(const uint32_t &total, const std::vector<uint32_t> &base);
    
    template <typename T>
    std::vector<T> dynamic_base_plus1(const std::vector<T> &nums, const std::vector<T> &base)
    {
        auto len = nums.size();
        assert(len > 0);
        assert(len == base.size());
        for (decltype(len) j = 0; j < len; j++) assert(nums[j] >= 0 && nums[j] < base[j]);
        auto res = nums;
        res[len-1]++;
        auto i = len - 1;
        while (i > 0 && res[i] == base[i]) {
            res[i] = 0;
            res[i-1]++;
            i--;
        }
        return res;
    }
    template std::vector<MKL_INT> dynamic_base_plus1(const std::vector<MKL_INT> &nums, const std::vector<MKL_INT> &base);
    template std::vector<uint32_t> dynamic_base_plus1(const std::vector<uint32_t> &nums, const std::vector<uint32_t> &base);
    
    template <typename T>
    bool is_sorted_norepeat(const std::vector<T> &array)
    {
        auto it = array.begin();
        auto it_prev = it++;
        while (it != array.end()) {
            if (! (*it_prev < *it)) return false;
            it_prev = it++;
        }
        return true;
    }
    template bool is_sorted_norepeat(const std::vector<MKL_INT> &array);
    template bool is_sorted_norepeat(const std::vector<mbasis_elem> &array);
    
    template <typename T>
    MKL_INT binary_search(const std::vector<T> &array, const T &val,
                          const MKL_INT &bgn, const MKL_INT &end)
    {
        if(array.size() == 0 && bgn == end) return -1;            // not found
        assert(bgn < end);
        assert(bgn >= 0 && bgn < static_cast<MKL_INT>(array.size()));
        assert(end > 0 && end <= static_cast<MKL_INT>(array.size()));
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
    template MKL_INT binary_search(const std::vector<mbasis_elem> &array, const mbasis_elem &val,
                                   const MKL_INT &bgn, const MKL_INT &end);
    template MKL_INT binary_search(const std::vector<MKL_INT> &array, const MKL_INT &val,
                                   const MKL_INT &bgn, const MKL_INT &end);
    
    
    template <typename T>
    int bubble_sort(std::vector<T> &array, const int &bgn, const int &end)
    {
        if(array.size() == 0 && bgn == end) return 0;            // no exchange at all
        assert(bgn < end);
        assert(bgn >= 0 && bgn < static_cast<int>(array.size()));
        assert(end > 0 && end <= static_cast<int>(array.size()));
        int len = end - bgn;
        int cnt = 0;
        
        using std::swap;
        for (int j = 1; j < len; j++) {  // e.g. if len = 3, need bubble sort with 2 loops
            int cnt0 = 0;
            for (int i = bgn; i < end - j; i++) {
                if (array[i+1] < array[i]) {
                    swap(array[i], array[i+1]);
                    cnt0++;
                }
            }
            if (cnt0 == 0) {  // already sorted
                break;
            } else {
                cnt += cnt0;
            }
        }
        return cnt;
    }
    template int bubble_sort(std::vector<uint32_t> &array, const int &bgn, const int &end);
    template int bubble_sort(std::vector<MKL_INT> &array, const int &bgn, const int &end);
    
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

