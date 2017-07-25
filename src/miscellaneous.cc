#include "qbasis.h"
#include "graph.h"

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
    
    template <typename T1, typename T2>
    std::vector<T1> dynamic_base(const T2 &total, const std::vector<T1> &base)
    {
        assert(base.size() > 0);
        assert(total >= 0);
        auto len = base.size();
        std::vector<T1> res(len);
        T2 temp = total;        // temp == i + j * base[0] + k * base[0] * base[1] + ...
        for (decltype(len) n = 0; n < len - 1; n++) {
            res[n] = static_cast<T1>(temp % base[n]);
            temp = (temp - res[n]) / base[n];
            if (temp == 0) {
                for (decltype(len) j = n + 1; j < len - 1; j++) res[j] = 0;
                break;
            }
        }
        res[len-1] = static_cast<T1>(temp);
        assert(res[len-1] < base[len-1]);
        return res;
    }
    template std::vector<MKL_INT> dynamic_base(const MKL_INT &total, const std::vector<MKL_INT> &base);
    template std::vector<uint8_t> dynamic_base(const uint64_t &total, const std::vector<uint8_t> &base);
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
    bool dynamic_base_maximized(const std::vector<T> &nums, const std::vector<T> &base)
    {
        auto len = nums.size();
        assert(len > 0);
        assert(len == base.size());
        for (decltype(len) j = 0; j < len; j++) {
            assert(base[j] > 0 && nums[j] < base[j]);
            if (nums[j] != base[j] - 1) return false;
        }
        return true;
    }
    template bool dynamic_base_maximized(const std::vector<uint8_t> &nums, const std::vector<uint8_t> &base);
    template bool dynamic_base_maximized(const std::vector<uint32_t> &nums, const std::vector<uint32_t> &base);
    
    template <typename T>
    bool dynamic_base_overflow(const std::vector<T> &nums, const std::vector<T> &base)
    {
        auto len = nums.size();
        assert(len > 0);
        assert(len == base.size());
        for (decltype(len) j = 0; j < len; j++)
            if (nums[j] > base[j] - 1) return true;
        return false;
    }
    template bool dynamic_base_overflow(const std::vector<uint8_t> &nums, const std::vector<uint8_t> &base);
    template bool dynamic_base_overflow(const std::vector<uint32_t> &nums, const std::vector<uint32_t> &base);
    
    
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
    template bool is_sorted_norepeat(const std::vector<uint32_t> &array);
    template bool is_sorted_norepeat(const std::vector<mbasis_elem> &array);
    template bool is_sorted_norepeat(const std::vector<std::vector<uint32_t>> &array);
    
    template <typename T1, typename T2>
    T2 binary_search(const std::vector<T1> &array, const T1 &val,
                          const T2 &bgn, const T2 &end)
    {
        if(array.size() == 0 && bgn == end) return static_cast<T2>(array.size());            // not found
        assert(bgn < end);
        assert(bgn >= 0 && bgn < static_cast<T2>(array.size()));
        assert(end > 0 && end <= static_cast<T2>(array.size()));
        T2 low  = bgn;
        T2 high = end - 1;
        T2 mid;
        while(low <= high) {
            mid = (low + high) / 2;
            if (val == array[mid]) return mid;
            else if (val < array[mid]) high = mid - 1;
            else low = mid + 1;
        }
        return static_cast<T2>(array.size());
    }
    template uint32_t binary_search(const std::vector<std::vector<uint32_t>> &array, const std::vector<uint32_t> &val,
                                    const uint32_t &bgn, const uint32_t &end);
    template uint64_t binary_search(const std::vector<std::vector<uint64_t>> &array, const std::vector<uint64_t> &val,
                                    const uint64_t &bgn, const uint64_t &end);
    template uint64_t binary_search(const std::vector<mbasis_elem> &array, const mbasis_elem &val,
                                   const uint64_t &bgn, const uint64_t &end);
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
    
    
    

    //  -------------- Multi-dimensional array data structure ------------------
    template <typename T>
    multi_array<T>::multi_array(const std::vector<uint64_t> &linear_size_input):
    dim_(linear_size_input.size()),
    linear_size_(linear_size_input)
    {
        assert(dim_ > 0);
        size_ = linear_size_[0];
        for (decltype(linear_size_.size()) j = 1; j < linear_size_.size(); j++)
            size_ *= linear_size_[j];
        data = std::vector<T>(size_);
    }
    
    template <typename T>
    multi_array<T>::multi_array(const std::vector<uint64_t> &linear_size_input, const T &element):
    dim_(linear_size_input.size()),
    linear_size_(linear_size_input)
    {
        assert(dim_ > 0);
        size_ = linear_size_[0];
        for (decltype(linear_size_.size()) j = 1; j < linear_size_.size(); j++)
            size_ *= linear_size_[j];
        data = std::vector<T>(size_, element);
    }
    
    template <typename T>
    multi_array<T>& multi_array<T>::operator=(const T &element)
    {
        for (uint64_t j = 0; j < size_; j++) data[j] = element;
        return *this;
    }
    
    template <typename T>
    T& multi_array<T>::index(const std::vector<uint64_t> &pos)
    {
        assert(pos.size() == linear_size_.size());
        uint64_t res = 0;
        auto j = linear_size_.size() - 1;
        while (pos[j] == 0 && j > 0) j--;
        while (j > 0) {
            assert(pos[j] < linear_size_[j]);
            res = (res + pos[j]) * linear_size_[j-1];
            j--;
        }
        assert(pos[0] < linear_size_[0]);
        res += pos[0];
        return data[res];
    }
    
    template <typename T>
    const T& multi_array<T>::index(const std::vector<uint64_t> &pos) const
    {
        assert(pos.size() == linear_size_.size());
        uint64_t res = 0;
        auto j = linear_size_.size() - 1;
        while (pos[j] == 0 && j > 0) j--;
        while (j > 0) {
            assert(pos[j] < linear_size_[j]);
            res = (res + pos[j]) * linear_size_[j-1];
            j--;
        }
        assert(pos[0] < linear_size_[0]);
        res += pos[0];
        return data[res];
    }
    
    template <typename T>
    void swap(multi_array<T> &lhs, multi_array<T> &rhs)
    {
        using std::swap;
        swap(lhs.dim_, rhs.dim_);
        swap(lhs.size_, rhs.size_);
        swap(lhs.linear_size_, rhs.linear_size_);
        swap(lhs.data, rhs.data);
    }
    
    // explicit instantiation
    template class multi_array<std::vector<uint32_t>>;
    template class multi_array<std::pair<std::vector<uint32_t>,std::vector<uint32_t>>>;
    
    template void swap(multi_array<std::vector<uint32_t>>&, multi_array<std::vector<uint32_t>>&);
    template void swap(multi_array<std::pair<std::vector<uint32_t>,std::vector<uint32_t>>>&,
                       multi_array<std::pair<std::vector<uint32_t>,std::vector<uint32_t>>>&);
    
    
    
    //  ----------------------- graph data structure ---------------------------
    void ALGraph::add_edge(const uint64_t &n1, const uint64_t &n2)
    {
        assert(n1 != n2);
        assert(n1 < vertices.size() && n2 < vertices.size());
        vertices[n1].arcs.push_back(n2);
        vertices[n2].arcs.push_back(n1);
        arcnum++;
    }
    
    void ALGraph::prt() const
    {
        for (uint64_t v = 0; v < vertices.size(); v++) {
            std::cout << v << " -> ";
            for (auto w : vertices[v].arcs) {
                std::cout << w << ",";
            }
            std::cout << std::endl;
        }
    }
    
    void ALGraph::BSF_set_JaJb(std::vector<MKL_INT> &ja, std::vector<MKL_INT> &jb)
    {
        MKL_INT magic = -19900917;
        for (decltype(ja.size()) j = 0; j < ja.size(); j++) ja[j] = magic;
        for (decltype(jb.size()) j = 0; j < jb.size(); j++) jb[j] = magic;
        
        std::vector<bool> visited(vertices.size(),false);
        std::list<uint64_t> Q;
        for (uint64_t v = 0; v < vertices.size(); v++) {
            if (! visited[v]) {
                visited[v] = true;
                // ja, jb arbitrary for this node
                //std::cout << "v = " << v << std::endl;
                assert(ja[vertices[v].i_a] == magic && jb[vertices[v].i_b] == magic);
                jb[vertices[v].i_b] = static_cast<MKL_INT>(v)/2;
                ja[vertices[v].i_a] = static_cast<MKL_INT>(v) - jb[vertices[v].i_b];
                //std::cout << "ia,ib -> ja,jb : " << vertices[v].i_a << "," << vertices[v].i_b << " -> " <<
                //ja[vertices[v].i_a] << "," << jb[vertices[v].i_b] << std::endl;
                Q.push_back(v);
                while (Q.size() != 0) {
                    uint64_t u = Q.front();
                    Q.pop_front();
                    auto u_arcs = vertices[u].arcs;
                    for (auto w : u_arcs) {
                        if (! visited[w]) {
                            visited[w] = true;
                            // ja, jb fixed from connected neighbors
                            //std::cout << "w = " << w << std::endl;
                            if (ja[vertices[w].i_a] == magic) {
                                assert(jb[vertices[w].i_b] != magic);
                                ja[vertices[w].i_a] = static_cast<MKL_INT>(w) - jb[vertices[w].i_b];
                            } else if (jb[vertices[w].i_b] == magic) {
                                assert(ja[vertices[w].i_a] != magic);
                                jb[vertices[w].i_b] = static_cast<MKL_INT>(w) - ja[vertices[w].i_a];
                            } else {
                                assert(ja[vertices[w].i_a] + jb[vertices[w].i_b] == static_cast<MKL_INT>(w));
                            }
                            //std::cout << "ia,ib -> ja,jb : " << vertices[w].i_a << "," << vertices[w].i_b << " -> " <<
                            //ja[vertices[w].i_a] << "," << jb[vertices[w].i_b] << std::endl;
                            Q.push_back(w);
                        }
                    }
                }
            }
        }
    }
    
}

