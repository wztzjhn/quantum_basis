//
//  qbasis.h
//  qbasis
//
//  Created by Zhentao Wang on 11/21/16.
//
//

#ifndef qbasis_h
#define qbasis_h

#define MKL_Complex16 std::complex<double>

#ifndef lapack_int
#define lapack_int MKL_INT
#endif

#ifndef lapack_complex_double
#define lapack_complex_double   MKL_Complex16
#endif

#include <cstdint>
#include <cmath>
#include <complex>
#include <algorithm>
#include <string>
#include <vector>
#include <list>
#include <forward_list>
#include <utility>
#include <initializer_list>
#include <chrono>
#include <cassert>
#include "mkl.h"

#ifdef _OPENMP
  #include <omp.h>
#else
  #define omp_get_thread_num() -1
  #define omp_get_num_threads() 1
  #define omp_get_num_procs() 1
#endif

#if defined(__clang__)
#elif defined(__GNUC__) && defined(_OPENMP)
  #include <parallel/algorithm>
  #define use_gnu_parallel_sort
#endif


// FUNCTIONS which need further change for Fermions:
// opr& transform(const std::vector<MKL_INT> &plan);
// opr_prod& transform(const std::vector<MKL_INT> &plan);

namespace qbasis {

//  -------------part 0: global variables, forward declarations ----------------
//  ----------------------------------------------------------------------------
    static const double pi = 3.141592653589793238462643;
    // later let's try to combine these three as a unified name "precision"
    static const double machine_prec = std::numeric_limits<double>::epsilon();
    static const double opr_precision = 1e-12; // used as the threshold value in comparison
    static const double sparse_precision = 1e-14;
    static const double lanczos_precision = 1e-12;
    
    // Multi-dimensional array
    template <typename> class multi_array;
    template <typename T> void swap(multi_array<T>&, multi_array<T>&);
    template <typename T> class multi_array {
        friend void swap <> (multi_array<T>&, multi_array<T>&);
    public:
        multi_array(): dim_(0), size_(0) {}
        multi_array(const std::vector<uint64_t> &linear_size_input);
        multi_array(const std::vector<uint64_t> &linear_size_input, const T &element);
        multi_array(const multi_array<T> &old):                                 // copy constructor
            dim_(old.dim_),
            size_(old.size_),
            linear_size_(old.linear_size_),
            data(old.data) {}
        multi_array(multi_array<T> &&old) noexcept :                            // move constructor
            dim_(old.dim_),
            size_(old.size_),
            linear_size_(std::move(old.linear_size_)),
            data(std::move(old.data)) {}
        multi_array& operator=(multi_array<T> old) { swap(*this, old); return *this; }
        multi_array& operator=(const T &element);
        ~multi_array() {}
        
        uint32_t dim() { return dim_; }
        uint64_t size() { return size_; }
        std::vector<uint64_t> linear_size() { return linear_size_; }
        T& index(const std::vector<uint64_t> &pos);
        const T& index(const std::vector<uint64_t> &pos) const;
    private:
        uint32_t dim_;
        uint64_t size_;
        std::vector<uint64_t> linear_size_;
        std::vector<T> data;
    };
    typedef multi_array<std::vector<uint32_t>> array_3D;
    typedef multi_array<std::pair<std::vector<uint32_t>,std::vector<uint32_t>>> array_4D;
    
    
    class basis_prop;
    class mbasis_elem;
    template <typename> class wavefunction;
    template <typename> class opr;
    template <typename> class opr_prod;
    template <typename> class mopr;
    class lattice;
    template <typename> class csr_mat;
    template <typename> class model;
    
    
    // Operations on basis
    void basis_props_split(const std::vector<basis_prop> &parent,
                           std::vector<basis_prop> &sub1, std::vector<basis_prop> &sub2);
    void swap(mbasis_elem&, mbasis_elem&);
    bool operator<(const mbasis_elem&, const mbasis_elem&);
    bool operator==(const mbasis_elem&, const mbasis_elem&);
    bool operator!=(const mbasis_elem&, const mbasis_elem&);
    bool trans_equiv(const mbasis_elem&, const mbasis_elem&, const std::vector<basis_prop> &props, const lattice&);   // computational heavy, use with caution
    
    void zipper_basis(const std::vector<basis_prop> &props_parent,              // defined in Weisse's PRE 87, 043305 (2013)
                      const std::vector<basis_prop> &props_sub_a,
                      const std::vector<basis_prop> &props_sub_b,
                      const mbasis_elem &sub_a, const mbasis_elem &sub_b, mbasis_elem &parent);
    void unzipper_basis(const std::vector<basis_prop> &props_parent,
                        const std::vector<basis_prop> &props_sub_a,
                        const std::vector<basis_prop> &props_sub_b,
                        const mbasis_elem &parent, mbasis_elem &sub_a, mbasis_elem &sub_b);
    
    // generate states compatible with given symmetry
    template <typename T>
    void enumerate_basis(const std::vector<basis_prop> &props,
                         std::vector<qbasis::mbasis_elem> &basis,
                         std::vector<mopr<T>> conserve_lst = {},
                         std::vector<double> val_lst = {});
    
    // sort basis according to Lin Table convention (Ib, then Ia)
    void sort_basis_Lin_order(const std::vector<basis_prop> &props, std::vector<qbasis::mbasis_elem> &basis);
    
    // generate Lin Tables for a given basis
    void fill_Lin_table(const std::vector<basis_prop> &props, const std::vector<qbasis::mbasis_elem> &basis,
                        std::vector<MKL_INT> &Lin_Ja, std::vector<MKL_INT> &Lin_Jb);
    
    // (sublattice) for a given list of full basis, find the reps according to translational symmetry
    void classify_trans_full2rep(const std::vector<basis_prop> &props,
                                 const std::vector<mbasis_elem> &basis_all,
                                 const lattice &latt,
                                 const std::vector<bool> &trans_sym,
                                 std::vector<mbasis_elem> &reps,
                                 std::vector<uint64_t> &belong2rep,
                                 std::vector<std::vector<int>> &dist2rep);
    // (sublattice) for a given list of reps, find the corresponding translation group
    void classify_trans_rep2group(const std::vector<basis_prop> &props,
                                  const std::vector<mbasis_elem> &reps,
                                  const lattice &latt,
                                  const std::vector<bool> &trans_sym,
                                  std::vector<std::vector<uint32_t>> &groups,
                                  std::vector<uint32_t> &omega_g,
                                  std::vector<uint32_t> &belong2group);
    // tabulate the 4-dim tables for (ga,gb,ja,jb) -> (i,j)
    void classify_Weisse_tables(const std::vector<basis_prop> &props_parent,
                             const std::vector<basis_prop> &props_sub,
                             const std::vector<mbasis_elem> &basis_sub_full,
                             const std::vector<mbasis_elem> &basis_sub_repr,
                             const lattice &latt_parent,
                             const std::vector<bool> &trans_sym,
                             const std::vector<uint64_t> &belong2rep,
                             const std::vector<std::vector<int>> &dist2rep,
                             const std::vector<std::vector<uint32_t>> &groups,
                             const std::vector<uint32_t> &omega_g,
                             const std::vector<uint32_t> &belong2group,
                             array_4D &table_e_lt,
                             array_4D &table_e_eq,
                             array_4D &table_e_gt,
                             array_3D &table_w_lt,
                             array_3D &table_w_eq);
    
    template <typename T> void swap(wavefunction<T>&, wavefunction<T>&);
    template <typename T> wavefunction<T> operator+(const wavefunction<T>&, const wavefunction<T>&);
    template <typename T> wavefunction<T> operator*(const wavefunction<T>&, const T&);
    template <typename T> wavefunction<T> operator*(const T&, const wavefunction<T>&);
    template <typename T> wavefunction<T> operator*(const mbasis_elem&, const T&);
    template <typename T> wavefunction<T> operator*(const T&, const mbasis_elem&);
    
    // -------- operators -----------
    template <typename T> void swap(opr<T>&, opr<T>&);
    template <typename T> bool operator==(const opr<T>&, const opr<T>&);
    template <typename T> bool operator!=(const opr<T>&, const opr<T>&);
    template <typename T> bool operator<(const opr<T>&, const opr<T>&); // only compares site, orbital and fermion, for sorting purpose
    template <typename T> opr<T> operator*(const T&, const opr<T>&);
    template <typename T> opr<T> operator*(const opr<T>&, const T&);
    template <typename T> opr<T> normalize(const opr<T>&, T&); // sum_{i,j} mat[i,j]^2 == dim; the 1st nonzero element (in memory) be real positive
    
    template <typename T> void swap(opr_prod<T>&, opr_prod<T>&);
    template <typename T> bool operator==(const opr_prod<T>&, const opr_prod<T>&);
    template <typename T> bool operator!=(const opr_prod<T>&, const opr_prod<T>&);
    template <typename T> bool operator<(const opr_prod<T>&, const opr_prod<T>&); // compare only length, and if each lhs.mat_prod < rhs.mat_prod
    template <typename T> opr_prod<T> operator*(const opr_prod<T>&, const opr_prod<T>&);
    template <typename T> opr_prod<T> operator*(const opr_prod<T>&, const opr<T>&);
    template <typename T> opr_prod<T> operator*(const opr<T>&, const opr_prod<T>&);
    template <typename T> opr_prod<T> operator*(const opr_prod<T>&, const T&);
    template <typename T> opr_prod<T> operator*(const T&, const opr_prod<T>&);
    template <typename T> opr_prod<T> operator*(const opr<T>&, const opr<T>&);       // cast up
    
    template <typename T> void swap(mopr<T>&, mopr<T>&);
    template <typename T> bool operator==(const mopr<T>&, const mopr<T>&);
    template <typename T> bool operator!=(const mopr<T>&, const mopr<T>&);
    template <typename T> mopr<T> operator+(const mopr<T>&, const mopr<T>&);
    template <typename T> mopr<T> operator+(const mopr<T>&, const opr_prod<T>&);
    template <typename T> mopr<T> operator+(const opr_prod<T>&, const mopr<T>&);
    template <typename T> mopr<T> operator+(const mopr<T>&, const opr<T>&);
    template <typename T> mopr<T> operator+(const opr<T>&, const mopr<T>&);
    template <typename T> mopr<T> operator+(const opr_prod<T>&, const opr_prod<T>&); // cast up
    template <typename T> mopr<T> operator+(const opr_prod<T>&, const opr<T>&);      // cast up
    template <typename T> mopr<T> operator+(const opr<T>&, const opr_prod<T>&);      // cast up
    template <typename T> mopr<T> operator+(const opr<T>&, const opr<T>&);           // cast up
    template <typename T> mopr<T> operator-(const mopr<T>&, const mopr<T>&);
    template <typename T> mopr<T> operator-(const mopr<T>&, const opr_prod<T>&);
    template <typename T> mopr<T> operator-(const opr_prod<T>&, const mopr<T>&);
    template <typename T> mopr<T> operator-(const mopr<T>&, const opr<T>&);
    template <typename T> mopr<T> operator-(const opr<T>&, const mopr<T>&);
    template <typename T> mopr<T> operator-(const opr_prod<T>&, const opr_prod<T>&); // cast up
    template <typename T> mopr<T> operator-(const opr_prod<T>&, const opr<T>&);      // cast up
    template <typename T> mopr<T> operator-(const opr<T>&, const opr_prod<T>&);      // cast up
    template <typename T> mopr<T> operator-(const opr<T>&, const opr<T>&);           // cast up
    template <typename T> mopr<T> operator*(const mopr<T>&, const mopr<T>&);
    template <typename T> mopr<T> operator*(const mopr<T>&, const opr_prod<T>&);
    template <typename T> mopr<T> operator*(const opr_prod<T>&, const mopr<T>&);
    template <typename T> mopr<T> operator*(const mopr<T>&, const opr<T>&);
    template <typename T> mopr<T> operator*(const opr<T>&, const mopr<T>&);
    template <typename T> mopr<T> operator*(const mopr<T>&, const T&);
    template <typename T> mopr<T> operator*(const T&, const mopr<T>&);

    // opr * | orb0, orb1, ..., ORB, ... > = | orb0, orb1, ..., opr*ORB, ... >, fermionic sign has to be computed when traversing orbitals
    template <typename T> wavefunction<T> oprXphi(const opr<T>&, const mbasis_elem&, const std::vector<basis_prop>&);
    template <typename T> wavefunction<T> oprXphi(const opr<T>&, const wavefunction<T>&, const std::vector<basis_prop>&);
    template <typename T> wavefunction<T> oprXphi(const opr_prod<T>&, const mbasis_elem&, const std::vector<basis_prop>&);
    template <typename T> wavefunction<T> oprXphi(const opr_prod<T>&, const wavefunction<T>&, const std::vector<basis_prop>&);
    template <typename T> wavefunction<T> oprXphi(const mopr<T>&, const mbasis_elem&, const std::vector<basis_prop>&);
    template <typename T> wavefunction<T> oprXphi(const mopr<T>&, const wavefunction<T>&, const std::vector<basis_prop>&);
    
    // mopr * {a list of mbasis} -->> {a new list of mbasis}
    template <typename T> void gen_mbasis_by_mopr(const mopr<T>&, std::list<mbasis_elem>&, const std::vector<basis_prop>&);
    
    template <typename T> void swap(csr_mat<T>&, csr_mat<T>&);
    
    // divide into two identical sublattices, if Nsites even. To be used in the divide and conquer method
    lattice divide_lattice(const lattice &parent);
    
    
    

    
    

//  --------------------  part 1: basis of the wave functions ------------------
//  ----------------------------------------------------------------------------
    // ------------ basic info of a particular basis -----------------
    struct extra_info {
        uint8_t Nmax;   // maximum number of particles
        // more items can be filled here...
    };
    
    class basis_prop {
    public:
        basis_prop() = default;
        basis_prop(const uint32_t &n_sites, const uint8_t &dim_local_,
                   const std::vector<uint32_t> &Nf_map = std::vector<uint32_t>(),
                   const bool &dilute_ = false);
        
        // current choices of name s:
        // ***   spin-1/2            ***
        // ***   spin-1              ***
        // ***   dimer               ***
        // ***   electron            ***
        // ***   tJ                  ***
        // ***   spinless-fermion    ***
        basis_prop(const uint32_t &n_sites, const std::string &s, const extra_info &ex = extra_info{0});
        
        bool q_fermion() const { return (! Nfermion_map.empty()); }
        
        void split(basis_prop &sub1, basis_prop &sub2) const;
        
        uint8_t dim_local;                      // local (single-site, single-orbital) dimension < 256
        uint8_t bits_per_site;                  // <= 8
        uint8_t bits_ignore;                    // for each orbital (with many sites), there are a few bits ignored
        uint16_t num_bytes;                     // for multi-orbital system, sum of num_bytes < 65536
        uint32_t num_sites;
        std::vector<uint32_t> Nfermion_map;     // Nfermion_map[i] corresponds to the number of fermions of state i
        std::string name;                       // store the name of the basis
        bool dilute;                            // if dilute, bit-rep is not a good representation
    };
    
    
    // -------------- fundamental class for basis elements ---------------
    // -------------- class for basis with several orbitals---------------
    // for given number of sites, and several orbitals, store the vectors of bits
    class mbasis_elem {
        friend void swap(mbasis_elem&, mbasis_elem&);
        friend bool operator<(const mbasis_elem&, const mbasis_elem&);
        friend bool operator==(const mbasis_elem&, const mbasis_elem&);
        friend bool trans_equiv(const mbasis_elem&, const mbasis_elem&, const std::vector<basis_prop> &props, const lattice&);
        template <typename T> friend wavefunction<T> oprXphi(const opr<T>&, const mbasis_elem&, const std::vector<basis_prop>&);
    public:
        // default constructor
        mbasis_elem() : mbits(nullptr) {}
        
        // constructor with its properties
        mbasis_elem(const std::vector<basis_prop> &props);
        
        // copy constructor
        mbasis_elem(const mbasis_elem& old);
        
        // move constructor
        mbasis_elem(mbasis_elem &&old) noexcept;
        
        // copy assignment constructor and move assignment constructor, using "swap and copy"
        mbasis_elem& operator=(mbasis_elem old) { swap(*this, old); return *this; }
        
        // destructor
        ~mbasis_elem();
        
        //     ---------- status changes -----------
        // read out a state from a given site and given orbital
        uint8_t siteRead(const std::vector<basis_prop> &props,
                         const uint32_t &site, const uint32_t &orbital) const;
        
        // write to a given site and given orbital
        mbasis_elem& siteWrite(const std::vector<basis_prop> &props,
                               const uint32_t &site, const uint32_t &orbital, const uint8_t &val);
        
        // reset all bits to 0 for a partical orbital
        mbasis_elem& reset(const std::vector<basis_prop> &props, const uint32_t &orbital);
        
        // reset all bits to 0 in all orbitals
        mbasis_elem& reset();
        
        // change mbasis_elem to the next available state, for a particular orbital
        mbasis_elem& increment(const std::vector<basis_prop> &props, const uint32_t &orbital);
        
        // change mbasis_elem to the next available state
        mbasis_elem& increment(const std::vector<basis_prop> &props);
        
        //    ---------------- print ---------------
        void prt_bits(const std::vector<basis_prop> &props) const;       // print the bits
        
        void prt_states(const std::vector<basis_prop> &props) const;     // print non-vacuum states
        
        //    ----------- basic inquiries ----------
        bool q_zero(const std::vector<basis_prop> &props, const uint32_t &orbital) const;
        
        bool q_zero() const;
        
        bool q_maximized(const std::vector<basis_prop> &props, const uint32_t &orbital) const;
        
        bool q_maximized(const std::vector<basis_prop> &props) const;
        
        bool q_same_state_all_site(const std::vector<basis_prop> &props, const uint32_t &orbital) const;
        
        bool q_same_state_all_site(const std::vector<basis_prop> &props) const;
        
        // get a label
        uint64_t label(const std::vector<basis_prop> &props, const uint32_t &orbital) const;
        
        uint64_t label(const std::vector<basis_prop> &props) const;
        
        void label_sub(const std::vector<basis_prop> &props, const uint32_t &orbital,
                       uint64_t &label1, uint64_t &label2) const;
        
        void label_sub(const std::vector<basis_prop> &props,
                       uint64_t &label1, uint64_t &label2) const;
        
        // return a vector of length dim_local (for orbital), reporting # of each state
        std::vector<uint32_t> statistics(const std::vector<basis_prop> &props, const uint32_t &orbital) const;
        
        // a direct product of the statistics of all orbitals, size: dim_orb1 * dim_orb2 * ...
        std::vector<uint32_t> statistics(const std::vector<basis_prop> &props) const;
        
        //    ---------- transformations -----------
        // translate the basis according to the given plan, transform a single orbital
        // sgn = 0 or 1 denoting if extra minus sign generated by translating fermions
        mbasis_elem& transform(const std::vector<basis_prop> &props,
                               const std::vector<uint32_t> &plan, int &sgn,
                               const uint32_t &orbital);
        // translate the basis according to the given plan, all orbitals transform in same way: site1 -> site2
        mbasis_elem& transform(const std::vector<basis_prop> &props,
                               const std::vector<uint32_t> &plan, int &sgn);
        
        // different orbs transform in different ways: (site1, orb1) -> (site2, orb2)
        // outer vector: each element denotes one orbital
        // middle vector: each element denotes one site
        // inner pair: first=site, second=orbital
        mbasis_elem& transform(const std::vector<basis_prop> &props,
                               const std::vector<std::vector<std::pair<uint32_t,uint32_t>>> &plan, int &sgn);
        
        mbasis_elem& translate(const std::vector<basis_prop> &props,
                               const lattice &latt, const std::vector<int> &disp, int &sgn,
                               const uint32_t &orbital);
        mbasis_elem& translate(const std::vector<basis_prop> &props,
                               const lattice &latt, const std::vector<int> &disp, int &sgn);
        
        // change to a basis element which is the unique (fully determined by the lattice and its state) among its translational equivalents
        // Translation(disp_vec) * old state = new state
        mbasis_elem& translate_to_unique_state(const std::vector<basis_prop> &props,
                                               const lattice &latt, std::vector<int> &disp_vec);
        
        //    ------------ measurements ------------
        double diagonal_operator(const std::vector<basis_prop> &props, const opr<double> &lhs) const;
        
        std::complex<double> diagonal_operator(const std::vector<basis_prop> &props, const opr<std::complex<double>> &lhs) const;
        
        double diagonal_operator(const std::vector<basis_prop> &props, const opr_prod<double> &lhs) const;
        
        std::complex<double> diagonal_operator(const std::vector<basis_prop> &props, const opr_prod<std::complex<double>> &lhs) const;
        
        double diagonal_operator(const std::vector<basis_prop> &props, const mopr<double> &lhs) const;
        
        std::complex<double> diagonal_operator(const std::vector<basis_prop> &props, const mopr<std::complex<double>> &lhs) const;
        
    private:
        // store an array of basis elements, for multi-orbital site (or unit cell)
        // the first 2 bytes are used to store the total number of bytes used by this array
        /*
         in terms of bits when perfoming "<" comparison (e.g. bits_per_site = 2):
         e.g.  0 1 0 1 0 0 0 0 1 1 0 1 0 0 0 0 1 1 1 0
                                                   ^ ^
                                                 /     \
                                            bits[1]   bits[0]
                                                 \     /
                                                 site[0]
         Note1: for arrangement of orbitals, they are similar:
                ...   orb[2]    orb[1]    orb[0]
         Note2: The wavefunction is always defined as:
                |alpha_0, beta_1, gamma_2, ... > = alpha_0^\dagger beta_1^\dagger gamma_2^\dagger ... |GS>
                (where alpha_i^\dagger is creation operator of state alpha on site i)
         */
        uint8_t* mbits;
    };
    
    
    // -------------- class for wave functions ---------------
    // Use with caution, may hurt speed when not used properly
    template <typename T> class wavefunction {
        friend void swap <> (wavefunction<T> &, wavefunction<T> &);
        friend wavefunction<T> oprXphi <> (const opr<T>&, const wavefunction<T>&, const std::vector<basis_prop>&);
        friend wavefunction<T> oprXphi <> (const opr_prod<T>&, const wavefunction<T>&, const std::vector<basis_prop>&);
        friend wavefunction<T> oprXphi <> (const mopr<T>&, const wavefunction<T>&, const std::vector<basis_prop>&);
    public:
        // default constructor
        wavefunction() = default;
        
        // constructor from an element
        wavefunction(const mbasis_elem &old) : elements(1, std::pair<mbasis_elem, T>(old, static_cast<T>(1.0))) {}
        wavefunction(mbasis_elem &&old)      : elements(1, std::pair<mbasis_elem, T>(old, static_cast<T>(1.0))) {}
        
        // copy constructor
        wavefunction(const wavefunction<T> &old) : elements(old.elements) {}
        
        // move constructor
        wavefunction(wavefunction<T> &&old) noexcept : elements(std::move(old.elements)) {}
        
        // copy assignment constructor and move assignment constructor, using "swap and copy"
        wavefunction& operator=(wavefunction<T> old)
        {
            swap(*this, old);
            return *this;
        }
        
        // destructor
        ~wavefunction() {}
        
        std::pair<mbasis_elem, T>& operator[](uint32_t n);
        
        const std::pair<mbasis_elem, T>& operator[](uint32_t n) const;
        
        //    ---------------- print ---------------
        void prt_bits(const std::vector<basis_prop> &props) const;
        
        void prt_states(const std::vector<basis_prop> &props) const;
        
        //    ----------- basic inquiries ----------
        // check if zero
        bool q_zero() const { return elements.empty(); }
        
        // check if sorted
        bool q_sorted() const;
        
        // check if sorted and there are no dulplicated terms
        bool q_sorted_fully() const;
        
        uint32_t size() const { return static_cast<uint32_t>(elements.size()); }
        
        // for \sum_i \alpha_i * element[i], return \sum_i |\alpha_i|^2
        double amplitude();
        
        //    ------------ arithmetics -------------
        // add one element
        wavefunction& operator+=(std::pair<mbasis_elem, T> ele);
        
        wavefunction& operator+=(const mbasis_elem &ele);
        
        // add a wave function
        wavefunction& operator+=(wavefunction<T> rhs);
        
        // multiply by a constant
        wavefunction& operator*=(const T &rhs);
        
        // simplify
        wavefunction& simplify();
        
    private:
        // store an array of basis elements, and their corresponding coefficients
        // note: there should not be any dulplicated elements
        std::list<std::pair<mbasis_elem, T>> elements;
    };
    
    
    
    
//  -----------------------  part 2: basis of the operators --------------------
//  ----------------------------------------------------------------------------
    
    // ---------------- fundamental class for operators ------------------
    // an operator on a given site and orbital
    template <typename T> class opr {
        friend void swap <> (opr<T>&, opr<T>&);
        friend bool operator== <> (const opr<T>&, const opr<T>&);
        friend bool operator!= <> (const opr<T>&, const opr<T>&);
        friend bool operator< <> (const opr<T>&, const opr<T>&);
        friend opr<T> operator* <> (const T&, const opr<T>&);
        friend opr<T> operator* <> (const opr<T>&, const T&);
        friend opr<T> normalize <> (const opr<T>&, T&);
        friend class opr_prod<T>;
        friend class mopr<T>;
        friend class mbasis_elem;
        friend wavefunction<T> oprXphi <> (const opr<T>&, const mbasis_elem&, const std::vector<basis_prop>&);
    public:
        // default constructor
        opr() : mat(nullptr) {}
        
        // constructor from diagonal elements
        opr(const uint32_t &site_, const uint32_t &orbital_, const bool &fermion_, const std::vector<T> &mat_);
        
        // constructor from a matrix
        opr(const uint32_t &site_, const uint32_t &orbital_, const bool &fermion_, const std::vector<std::vector<T>> &mat_);
        
        // copy constructor
        opr(const opr<T> &old);
        
        // move constructor
        opr(opr<T> &&old) noexcept;
        
        // copy assignment constructor and move assignment constructor, using "swap and copy"
        opr& operator=(opr<T> old) { swap(*this, old); return *this; }
        
        // destructor
        ~opr();
        
        void prt() const;
        
        //    ----------- basic inquiries ----------
        // question if it is zero operator
        bool q_zero() const;
        
        // question if it is identity operator
        bool q_diagonal() const { return diagonal; }
        
        // question if it is identity operator
        bool q_identity() const;
        
        uint32_t pos_site() const { return site; }
        
        uint32_t pos_orb() const { return orbital; }
        
        //    ------------ arithmetics -------------
        // \sqrt { sum_{i,j} |mat[i,j]|^2 }
        double norm() const;
        
        // simplify the structure if possible
        opr& simplify();
        
        // invert the sign
        opr& negative();
        
        // take Hermitian conjugate
        opr& dagger();
        
        // change site index
        opr& change_site(const uint32_t &site_);
        
        // fermions not implemented yet
        opr& transform(const std::vector<uint32_t> &plan);
        
        // compound assignment operators
        opr& operator+=(const opr<T> &rhs);
        
        opr& operator-=(const opr<T> &rhs);
        
        opr& operator*=(const opr<T> &rhs);
        
        opr& operator*=(const T &rhs);

    private:
        uint32_t site;      // site No.
        uint32_t orbital;   // orbital No.
        uint32_t dim;       // number of rows(columns) of the matrix
        bool fermion;       // fermion or not
        bool diagonal;      // diagonal in matrix form
        T *mat;             // matrix form, or diagonal elements if diagonal
    };
    
    
    // -------------- class for operator products ----------------
    // note: when mat_prod is empty, it represents identity operator, with coefficient coeff
    // all matrices in this class should have the same type (may decrease effiency, think later how we can improve)
    template <typename T> class opr_prod {
        friend void swap <> (opr_prod<T>&, opr_prod<T>&);
        friend bool operator== <> (const opr_prod<T>&, const opr_prod<T>&);
        friend bool operator!= <> (const opr_prod<T>&, const opr_prod<T>&);
        friend bool operator< <> (const opr_prod<T>&, const opr_prod<T>&);
        friend opr_prod<T> operator* <> (const opr_prod<T>&, const opr_prod<T>&);
        friend opr_prod<T> operator* <> (const opr_prod<T>&, const opr<T>&);
        friend opr_prod<T> operator* <> (const opr<T>&, const opr_prod<T>&);
        friend opr_prod<T> operator* <> (const opr_prod<T>&, const T&);
        friend opr_prod<T> operator* <> (const T&, const opr_prod<T>&);
        friend opr_prod<T> operator* <> (const opr<T>&, const opr<T>&);
        friend class mopr<T>;
        friend class mbasis_elem;
        friend wavefunction<T> oprXphi <> (const opr_prod<T>&, const mbasis_elem&, const std::vector<basis_prop>&);
        friend wavefunction<T> oprXphi <> (const opr_prod<T>&, const wavefunction<T>&, const std::vector<basis_prop>&);
    public:
        // default constructor
        opr_prod() = default;
        
        // constructor from one fundamental operator
        opr_prod(const opr<T> &ele);
        
        // copy constructor
        opr_prod(const opr_prod<T> &old): coeff(old.coeff), mat_prod(old.mat_prod) {}
        
        // move constructor
        opr_prod(opr_prod<T> &&old) noexcept : coeff(std::move(old.coeff)), mat_prod(std::move(old.mat_prod)) {}
        
        // copy assignment constructor and move assignment constructor, using "swap and copy"
        opr_prod& operator=(opr_prod<T> old) { swap(*this, old); return *this; }
        
        // destructor
        ~opr_prod() {}
        
        void prt() const;
        
        //    ----------- basic inquiries ----------
        // question if it is zero operator
        bool q_zero() const;
        
        // question if each opr is diagonal
        bool q_diagonal() const;
        
        // question if it is proportional to identity operator
        bool q_prop_identity() const;
        
        uint32_t len() const;
        
        opr<T>& operator[](uint32_t n);
        
        const opr<T>& operator[](uint32_t n) const;
        
        //    ------------ arithmetics -------------
        // invert the sign
        opr_prod& negative();
        
        opr_prod& transform(const std::vector<uint32_t> &plan);
        
        opr_prod& operator*=(opr<T> rhs);
        
        opr_prod& operator*=(opr_prod<T> rhs); // in this form to avoid self-assignment
        
        opr_prod& operator*=(const T &rhs);
        
    private:
        T coeff;
        std::list<opr<T>> mat_prod; // each opr<T> should be normalized
    };
    
    
    // -------------- class for a combination of operators ----------------
    // a linear combination of products of operators
    template <typename T> class mopr {
        friend void swap <> (mopr<T>&, mopr<T>&);
        friend bool operator== <> (const mopr<T>&, const mopr<T>&);
        friend bool operator!= <> (const mopr<T>&, const mopr<T>&);
        friend mopr<T> operator+ <> (const mopr<T>&, const mopr<T>&);
        friend mopr<T> operator+ <> (const mopr<T>&, const opr_prod<T>&);
        friend mopr<T> operator+ <> (const opr_prod<T>&, const mopr<T>&);
        friend mopr<T> operator+ <> (const mopr<T>&, const opr<T>&);
        friend mopr<T> operator+ <> (const opr<T>&, const mopr<T>&);
        friend mopr<T> operator+ <> (const opr_prod<T>&, const opr_prod<T>&);
        friend mopr<T> operator+ <> (const opr_prod<T>&, const opr<T>&);
        friend mopr<T> operator+ <> (const opr<T>&, const opr_prod<T>&);
        friend mopr<T> operator+ <> (const opr<T>&, const opr<T>&);
        friend mopr<T> operator- <> (const mopr<T>&, const mopr<T>&);
        friend mopr<T> operator- <> (const mopr<T>&, const opr_prod<T>&);
        friend mopr<T> operator- <> (const opr_prod<T>&, const mopr<T>&);
        friend mopr<T> operator- <> (const mopr<T>&, const opr<T>&);
        friend mopr<T> operator- <> (const opr<T>&, const mopr<T>&);
        friend mopr<T> operator- <> (const opr_prod<T>&, const opr_prod<T>&);
        friend mopr<T> operator- <> (const opr_prod<T>&, const opr<T>&);
        friend mopr<T> operator- <> (const opr<T>&, const opr_prod<T>&);
        friend mopr<T> operator- <> (const opr<T>&, const opr<T>&);
        friend mopr<T> operator* <> (const mopr<T>&, const mopr<T>&);
        friend mopr<T> operator* <> (const mopr<T>&, const opr_prod<T>&);
        friend mopr<T> operator* <> (const opr_prod<T>&, const mopr<T>&);
        friend mopr<T> operator* <> (const mopr<T>&, const opr<T>&);
        friend mopr<T> operator* <> (const opr<T>&, const mopr<T>&);
        friend mopr<T> operator* <> (const mopr<T>&, const T&);
        friend mopr<T> operator* <> (const T&, const mopr<T>&);
        friend wavefunction<T> oprXphi <> (const mopr<T>&, const mbasis_elem&, const std::vector<basis_prop>&);
        friend wavefunction<T> oprXphi <> (const mopr<T>&, const wavefunction<T>&, const std::vector<basis_prop>&);
    public:
        // default constructor
        mopr() = default;
        
        // constructor from one fundamental operator
        mopr(const opr<T> &ele);
        
        // constructor from operator products
        mopr(const opr_prod<T> &ele);
        
        // copy constructor
        mopr(const mopr<T> &old): mats(old.mats) {}
        
        // move constructor
        mopr(mopr<T> &&old) noexcept : mats(std::move(old.mats)) {}
        
        // copy assignment constructor and move assignment constructor, using "swap and copy"
        mopr& operator=(mopr<T> old) { swap(*this, old); return *this; }
        
        // destructor
        ~mopr() {}
        
        void prt() const;
        
        //    ----------- basic inquiries ----------
        bool q_zero() const { return mats.empty(); }
        
        // question if each opr_prod is diagonal
        bool q_diagonal() const;
        
        uint32_t size() const { return static_cast<uint32_t>(mats.size()); }
        
        opr_prod<T>& operator[](uint32_t n);
        
        const opr_prod<T>& operator[](uint32_t n) const;
        
        //    ------------ arithmetics -------------
        mopr& simplify();
        
        // invert the sign
        mopr& negative();
        
        mopr& transform(const std::vector<uint32_t> &plan);
        
        mopr& operator+=(opr<T> rhs);
        
        mopr& operator+=(opr_prod<T> rhs);
        
        mopr& operator+=(mopr<T> rhs);
        
        mopr& operator-=(opr<T> rhs);
        
        mopr& operator-=(opr_prod<T> rhs);
        
        mopr& operator-=(mopr<T> rhs);
        
        mopr& operator*=(opr<T> rhs);
        
        mopr& operator*=(opr_prod<T> rhs);
        
        mopr& operator*=(mopr<T> rhs);
        
        mopr& operator*=(const T &rhs);
        
    private:
        // the outer list represents the sum of operators, inner data structure taken care by operator products
        std::list<opr_prod<T>> mats;
    };
    
    
    
    
//  --------------------------  part 3: sparse matrices ------------------------
//  ----------------------------------------------------------------------------
    
    template <typename T> struct lil_mat_elem {
        T val;
        MKL_INT col;
    };
    
    template <typename T> class lil_mat {
        friend class csr_mat<T>;
    public:
        // default constructor
        lil_mat() = default;
        
        // constructor with the Hilbert space dimension
        lil_mat(const MKL_INT &n, bool sym_ = false);
        
        // add one element
        void add(const MKL_INT &row, const MKL_INT &col, const T &val);
        
        // explicitly destroy, free space
        void destroy(const MKL_INT &row);
        
        void destroy();
        
        // destructor
        ~lil_mat() {};
        
        MKL_INT dimension() const { return dim; }
        
        MKL_INT num_nonzero() const { return nnz; }
        
        bool q_sym() { return sym; }
        
        // print
        void prt() const;
        
        void use_full_matrix() {sym = false; }
        
    private:
        MKL_INT dim;    // dimension of the matrix
        MKL_INT nnz;    // number of non-zero entries
        bool sym;       // if storing only upper triangle
        std::vector<std::forward_list<lil_mat_elem<T>>> mat;
    };
    
    
    // 3-array form of csr sparse matrix format, zero based
    template <typename T> class csr_mat {
        friend void swap <> (csr_mat<T>&, csr_mat<T>&);
    public:
        // default constructor
        csr_mat() : dim(0), nnz(0), val(nullptr), ja(nullptr), ia(nullptr) {}
        
        // copy constructor
        csr_mat(const csr_mat<T> &old);
        
        // move constructor
        csr_mat(csr_mat<T> &&old) noexcept;
        
        // copy assignment constructor and move assignment constructor, using "swap and copy"
        csr_mat& operator=(csr_mat<T> old) { swap(*this, old); return *this; }
        
        // explicitly destroy, free space
        void destroy();
        
        // destructor
        ~csr_mat();
        
        // print
        void prt() const;
        
        MKL_INT dimension() const {return dim; }
        
        // construcotr from a lil_mat, and if sym_ == true, use only the upper triangle
        // then destroy the lil_mat
        csr_mat(lil_mat<T> &old);
        
        // matrix vector product
        void MultMv(const T *x, T *y) const;
        void MultMv(T *x, T *y);  // to be compatible with arpack++
        
        std::vector<T> to_dense() const;
        
        // matrix matrix product, x and y of shape dim * n
        //void MultMm(const T *x, T *y, MKL_INT n) const;
        
    private:
        MKL_INT dim;
        MKL_INT nnz;        // number of non-zero entries
        bool sym;           // if storing only upper triangle
        T *val;
        MKL_INT *ja;
        MKL_INT *ia;
    };
    
    
    
//  ------------------------------part 4: Lanczos ------------------------------
//  ----------------------------------------------------------------------------
    
    // Note: sparse matrices in this code are using zero-based convention
    
    // By default, all diagonal elements are stored, even if they are zero (to be compatible with pardiso, if used in future)
    
    
    // m = k + np step of Lanczos
    // v of length m+1, hessenberg matrix of size m*m (m-step Lanczos)
    // after decomposition, mat * v[0:m-1] = v[0:m-1] * hessenberg + rnorm * resid * e_m^T,
    // where e_m has only one nonzero element: e[0:m-2] == 0, e[m-1] = 1
    
    // on entry, assuming k steps of Lanczos already performed:
    // v_0, ..., v_{k-1} stored in v, v{k} stored in resid
    // alpha_0, ..., alpha_{k-1} in hessenberg matrix
    // beta_1,  ..., beta_{k-1} in hessenberg matrix, beta_k as rnorm
    // if on entry k==0, then beta_k=0, v_0=resid, entry value of rnorm irrelevant
    
    // ldh: leading dimension of hessenberg
    // alpha[j] = hessenberg[j+ldh], diagonal of hessenberg matrix
    // beta[j]  = hessenberg[j]
    //  a[0]  b[1]      -> note: beta[0] not used
    //  b[1]  a[1]  b[2]
    //        b[2]  a[2]  b[3]
    //              b[3]  a[3] b[4]
    //                    ..  ..  ..    b[k-1]
    //                          b[k-1]  a[k-1]
    template <typename T, typename MAT>
    void lanczos(MKL_INT k, MKL_INT np, const MKL_INT &dim, const MAT &mat, double &rnorm, T resid[],
                 T v[], double hessenberg[], const MKL_INT &ldh, const bool &MemoSteps = true);
    
    // if possible, add a block Arnoldi version here
    
    // transform from band storage to general storage
    template <typename T>
    void hess2matform(const double hessenberg[], T mat[], const MKL_INT &m, const MKL_INT &ldh);
    
    // compute eigenvalues (and optionally eigenvectors, stored in s) of hessenberg matrix
    // on entry, hessenberg and s should have the same leading dimension: ldh
    // order = "sm", "lm", "sr", "lr", where 's': small, 'l': large, 'm': magnitude, 'r': real part
    void select_shifts(const double hessenberg[], const MKL_INT &ldh, const MKL_INT &m,
                       const std::string &order, double ritz[], double s[] = nullptr);
    
    // --------------------------
    // ideally, here we should use the bulge-chasing algorithm; for this moment, we simply use the less efficient brute force QR factorization
    // --------------------------
    // QR factorization of hessenberg matrix, using np selected eigenvalues from ritz
    // [H, Q] = QR(H, shift1, shift2, ..., shift_np)
    // \tilde{H} = Q_np^T ... Q_1^T H Q_1 ... Q_np
    // \tilde{V} = V Q
    template <typename T>
    void perform_shifts(const MKL_INT &dim, const MKL_INT &m, const MKL_INT &np, const double shift[],
                        double &rnorm, T resid[], T v[], double hessenberg[], const MKL_INT &ldh,
                        double Q[], const MKL_INT &ldq);
    
    // implicitly restarted Arnoldi method
    // nev: number of eigenvalues needed
    // ncv: length of each individual lanczos process
    // 2 < nev + 2 <= ncv
    // when not using arpack++, we can modify the property of mat to be const
    template <typename T, typename MAT>
    void iram(const MKL_INT &dim, MAT &mat, T v0[], const MKL_INT &nev, const MKL_INT &ncv,
              const MKL_INT &maxit, const std::string &order,
              MKL_INT &nconv, double eigenvals[], T eigenvecs[],
              const bool &use_arpack = true);
    
    
//  ----------------------------- part 5: Lattices  ----------------------------
//  ----------------------------------------------------------------------------
    class lattice {
        friend lattice divide_lattice(const lattice &parent);
    public:
        lattice() = default;
        
        // constructor from particular requirements. e.g. square, triangular...
        lattice(const std::string &name, const std::vector<uint32_t> &L_, const std::vector<std::string> &bc_);
        
        // coordinates <-> site indices
        // first find a direction (dim_spec) which has even size, if not successful, use the following:
        // 1D: site = i * num_sub + sub
        // 2D: site = (i + j * L[0]) * num_sub + sub
        // 3D: site = (i + j * L[0] + k * L[0] * L[1]) * num_sub + sub
        // otherwise, the dim_spec should be counted first
        void coor2site(const std::vector<int> &coor, const int &sub, uint32_t &site) const;
        
        void site2coor(std::vector<int> &coor, int &sub, const uint32_t &site) const;
        
        // return a vector containing the positions of each site after translation
        std::vector<uint32_t> translation_plan(const std::vector<int> &disp) const;
        
        // return a vector containing the positions of each site after c2 (180) or c4 (90) rotation
        std::vector<uint32_t> c2_rotation_plan() const;
        std::vector<uint32_t> c4_rotation_plan() const;
        std::vector<uint32_t> reflection_plan() const;
        
        // combine two plans
        std::vector<std::vector<std::pair<uint32_t,uint32_t>>> plan_product(const std::vector<std::vector<std::pair<uint32_t,uint32_t>>> &lhs,
                                                                            const std::vector<std::vector<std::pair<uint32_t,uint32_t>>> &rhs) const;
        
        // inverse of a transformation
        std::vector<std::vector<std::pair<uint32_t,uint32_t>>> plan_inverse(const std::vector<std::vector<std::pair<uint32_t,uint32_t>>> &old) const;
        
        std::vector<std::string> boundary() const {
            return bc;
        }
        
        uint32_t dimension() const {
            return dim;
        }
        
        uint32_t num_sublattice() const {
            return num_sub;
        }
        
        uint32_t total_sites() const {
            return Nsites;
        }
        
        std::vector<uint32_t> Linear_size() const { return L; }
        
        uint32_t Lx() const { return L[0]; }
        uint32_t Ly() const { assert(L.size() > 1); return L[1]; }
        uint32_t Lz() const { assert(L.size() > 2); return L[2]; }
        
        // obtain all possible divisors of a lattice, for the divide and conquer method
        // the returned value is a list of lists: {{divisors for Lx}, {divisors for Ly}, ...}
        std::vector<std::vector<uint32_t>> divisor_v1(const std::vector<bool> &trans_sym) const;
        // {{1,1,1}, {1,1,2}, {1,1,5}, {1,2,1}, {1,2,2}, {1,2,5},...}, i.e., combine results from v1 to a single list
        std::vector<std::vector<uint32_t>> divisor_v2(const std::vector<bool> &trans_sym) const;
        
    private:
        std::vector<uint32_t> L;             // linear size in each dimension
        std::vector<std::string> bc;         // boundary condition
        std::vector<std::vector<double>> a;  // real space basis
        std::vector<std::vector<double>> b;  // momentum space basis
        // one more variable here, denoting the divide and conquer partition
        // if empty(false), then force to store the matrix when working with translational symmetry
        uint32_t dim;
        uint32_t num_sub;
        uint32_t Nsites;
        uint32_t dim_spec;                   // the code starts labeling sites from a dimension which has even # of sites
    };
    
    
//  ---------------part 6: Routines to construct Hamiltonian -------------------
//  ----------------------------------------------------------------------------
    
    template <typename T> class model {
//        friend MKL_INT generate_Ham_all_AtRow <> (model<T> &,
//                                                  threads_pool &);
    public:
        bool matrix_free;
        std::vector<basis_prop> props;
        std::vector<basis_prop> props_sub_a, props_sub_b;
        mopr<T> Ham_diag;
        mopr<T> Ham_off_diag;
        MKL_INT nconv;
        
        std::vector<bool> trans_sym;
        
        MKL_INT dim_target_full;
        MKL_INT dim_excite_full;
        MKL_INT dim_target_repr;
        MKL_INT dim_excite_repr;
        std::vector<qbasis::mbasis_elem> basis_target_full;    // full basis for ground state sector, without translation sym
        std::vector<qbasis::mbasis_elem> basis_excite_full;    // full basis for some intermeidate state sector (e.g. calculating correlation functions)
        std::vector<qbasis::mbasis_elem> basis_target_repr;
        std::vector<qbasis::mbasis_elem> basis_excite_repr;
        std::vector<qbasis::mbasis_elem> basis_sub_full;       // basis for half lattice, used for building Weisse Table
        std::vector<qbasis::mbasis_elem> basis_sub_repr;       // reps for half lattice
        
        std::vector<MKL_INT> Lin_Ja_target_full;               // Lin tables for the full basis
        std::vector<MKL_INT> Lin_Jb_target_full;
        std::vector<MKL_INT> Lin_Ja_excite_full;
        std::vector<MKL_INT> Lin_Jb_excite_full;
        std::vector<MKL_INT> Lin_Ja_target_repr;               // Lin tables for the repr basis
        std::vector<MKL_INT> Lin_Jb_target_repr;
        std::vector<MKL_INT> Lin_Ja_excite_repr;
        std::vector<MKL_INT> Lin_Jb_excite_repr;
        
        std::vector<uint64_t> belong2rep_sub;
        std::vector<std::vector<int>> dist2rep_sub;
        std::vector<std::vector<uint32_t>> groups_sub;
        std::vector<uint32_t> omega_g_sub;
        std::vector<uint32_t> belong2group_sub;
        array_4D table_e_lt, table_e_eq, table_e_gt;           // Weisse Tables for translation symmetry
        array_3D table_w_lt, table_w_eq;
        
        csr_mat<T> HamMat_csr_target_full;
        csr_mat<T> HamMat_csr_excite_full;
        csr_mat<std::complex<double>> HamMat_csr_target_repr;
        csr_mat<std::complex<double>> HamMat_csr_excite_repr;
        
        std::vector<double> eigenvals_full;
        std::vector<T> eigenvecs_full;
        std::vector<double> eigenvals_repr;
        std::vector<std::complex<double>> eigenvecs_repr;
        
        
        // ---------------- deprecated --------------------
        std::vector<MKL_INT> basis_belong;                // size: dim_target_full, store the position of its repr
        std::vector<std::complex<double>> basis_coeff;    // size: dim_target_full, store the coeff
        std::vector<MKL_INT> basis_repr_deprecated;
        // ---------------- deprecated --------------------
        
        
        model();
        
        ~model() {}
        
        void prt_Ham_diag() const { Ham_diag.prt(); }
        
        void prt_Ham_offdiag() const { Ham_off_diag.prt(); }
        
        void add_orbital(const uint32_t &n_sites, const uint8_t &dim_local_,
                         const std::vector<uint32_t> &Nf_map = std::vector<uint32_t>(),
                         const bool &dilute_ = false)
        {
            props.emplace_back(n_sites,dim_local_,Nf_map,dilute_);
            basis_props_split(props, props_sub_a, props_sub_b);
        }
        
        void add_orbital(const uint32_t &n_sites, const std::string &s, const extra_info &ex = extra_info{0})
        {
            props.emplace_back(n_sites, s, ex);
            basis_props_split(props, props_sub_a, props_sub_b);
        }
        
        uint32_t local_dimension() const;
        
        void add_diagonal_Ham(const opr<T> &rhs)      { assert(rhs.q_diagonal()); Ham_diag += rhs; }
        
        void add_diagonal_Ham(const opr_prod<T> &rhs) { assert(rhs.q_diagonal()); Ham_diag += rhs; }
        
        void add_diagonal_Ham(const mopr<T> &rhs)     { assert(rhs.q_diagonal()); Ham_diag += rhs; }
        
        void add_offdiagonal_Ham(const opr<T> &rhs)      { Ham_off_diag += rhs; }
        
        void add_offdiagonal_Ham(const opr_prod<T> &rhs) { Ham_off_diag += rhs; }
        
        void add_offdiagonal_Ham(const mopr<T> &rhs)     { Ham_off_diag += rhs; }
        
        void fill_Weisse_table(const lattice &latt);
        
        // naive way of enumerating all possible basis state
        void enumerate_basis_full(const lattice &latt,
                                  MKL_INT &dim_full,
                                  std::vector<qbasis::mbasis_elem> &basis_full,
                                  std::initializer_list<mopr<std::complex<double>>> conserve_lst = {},
                                  std::initializer_list<double> val_lst = {});
        
        // Need to build Weiss Tables before enumerating representatives
        void enumerate_basis_repr(const lattice &latt,
                                  const std::vector<int> &momentum,
                                  MKL_INT &dim_repr,
                                  std::vector<qbasis::mbasis_elem> &basis_repr,
                                  std::initializer_list<mopr<std::complex<double>>> conserve_lst = {},
                                  std::initializer_list<double> val_lst = {});
        
        // momentum has to be in format {m,n,...} corresponding to (m/L1) b_1 + (n/L2) b_2 + ...
        void basis_init_repr_deprecated(const std::vector<int> &momentum, const lattice &latt);
        
        
        
        void generate_Ham_sparse_full(const bool &upper_triangle = true); // generate the full Hamiltonian in sparse matrix format
        
        void generate_Ham_sparse_repr(const bool &upper_triangle = true); // generate the Hamiltonian using basis_repr
        
        // generate a dense matrix of the Hamiltonian
        std::vector<std::complex<double>> to_dense();
        
        // generate matrix on the fly
        void MultMv(const T *x, T *y) const;
        void MultMv(T *x, T *y);  // to be compatible with arpack++
        
        void locate_E0_full(const MKL_INT &nev = 2, const MKL_INT &ncv = 6, MKL_INT maxit = 0);
        
        // Don't use! Accuracy not good enough yet.
        void locate_E0_full_lanczos();
        
        void locate_Emax_full(const MKL_INT &nev = 2, const MKL_INT &ncv = 6, MKL_INT maxit = 0);
        
        void locate_E0_repr(const MKL_INT &nev = 2, const MKL_INT &ncv = 6, MKL_INT maxit = 0);
        
        void locate_Emax_repr(const MKL_INT &nev = 2, const MKL_INT &ncv = 6, MKL_INT maxit = 0);
        
        double energy_min() { return E0; }
        
        double energy_max() { return Emax; }
        
        double energy_gap() { return gap; }
        
        // lhs | phi >
        void moprXeigenvec_full(const mopr<T> &lhs, T* vec_new, const MKL_INT &which_col = 0);
        // < phi | lhs | phi >
        T measure(const mopr<T> &lhs, const MKL_INT &which_col=0);
        // < phi | lhs1^\dagger lhs2 | phi >
        T measure(const mopr<T> &lhs1, const mopr<T> &lhs2, const MKL_INT &which_col=0);
        

        // later add conserved quantum operators and corresponding quantum numbers
        // later add measurement operators
    
    private:
        double Emax;
        double E0;
        double gap;
        
        // check if translational symmetry satisfied
        void check_translation(const lattice &latt);
    };
    

    

    
//  --------------------------- Miscellaneous stuff ----------------------------
//  ----------------------------------------------------------------------------
    
    inline double conjugate(const double &rhs) { return rhs; }
    inline std::complex<double> conjugate(const std::complex<double> &rhs) { return std::conj(rhs); }
    
    // calculate base^index, in the case both are integers
    template <typename T1, typename T2>
    T2 int_pow(const T1 &base, const T1 &index);
    
    // given two arrays: num & base, get the result of:
    // num[0] + num[1] * base[0] + num[2] * base[0] * base[1] + num[3] * base[0] * base[1] * base[2] + ...
    // e.g. (base = {2,2,2,2,2})
    // 0      0      1      0      1
    //                      ^      ^
    //                      |      |
    //                   num[1]   num[0]
    // 1 + 0 * 2 + 1 * 2^2 + 0 * 2^3 + 0 * 2^4
    template <typename T1, typename T2>
    T2 dynamic_base(const std::vector<T1> &nums, const std::vector<T1> &base);
    // the other way around
    template <typename T1, typename T2>
    std::vector<T1> dynamic_base(const T2 &total, const std::vector<T1> &base);
    // nums + 1
    template <typename T>
    std::vector<T> dynamic_base_plus1(const std::vector<T> &nums, const std::vector<T> &base);
    // check if maximized
    template <typename T>
    bool dynamic_base_maximized(const std::vector<T> &nums, const std::vector<T> &base);
    // check overflow
    template <typename T>
    bool dynamic_base_overflow(const std::vector<T> &nums, const std::vector<T> &base);
    
    template <typename T>
    bool is_sorted_norepeat(const std::vector<T> &array);
    
    // note: end means the position which has already passed the last element
    template <typename T1, typename T2>
    T2 binary_search(const std::vector<T1> &array, const T1 &val,
                     const T2 &bgn, const T2 &end);
    
    // return the number of exchanges happened during the bubble sort
    template <typename T>
    int bubble_sort(std::vector<T> &array, const int &bgn, const int &end);
    
    //             b1
    // a0 +  ---------------
    //                b2
    //       a1 + ----------
    //            a2 + ...
    template <typename T>
    T continued_fraction(T a[], T b[], const MKL_INT &len); // b0 not used
    
    
}



//  -------------------------  interface to mkl library ------------------------
//  ----------------------------------------------------------------------------
namespace qbasis {
    // blas level 1, y = a*x + y
    inline // double
    void axpy(const MKL_INT n, const double alpha, const double *x, const MKL_INT incx, double *y, const MKL_INT incy) {
        daxpy(&n, &alpha, x, &incx, y, &incy);
    }
    inline // complex double
    void axpy(const MKL_INT n, const std::complex<double> alpha, const std::complex<double> *x, const MKL_INT incx, std::complex<double> *y, const MKL_INT incy) {
        zaxpy(&n, &alpha, x, &incx, y, &incy);
    }
    
    
    // blas level 1, y = x
    inline // double
    void copy(const MKL_INT n, const double *x, const MKL_INT incx, double *y, const MKL_INT incy) {
        dcopy(&n, x, &incx, y, &incy);
    }
    inline // complex double
    void copy(const MKL_INT n, const std::complex<double> *x, const MKL_INT incx, std::complex<double> *y, const MKL_INT incy) {
        zcopy(&n, x, &incx, y, &incy);
    }
    
    // blas level 1, Euclidean norm of vector
    inline // double
    double nrm2(const MKL_INT n, const double *x, const MKL_INT incx) {
        return dnrm2(&n, x, &incx);
    }
    inline // complex double
    double nrm2(const MKL_INT n, const std::complex<double> *x, const MKL_INT incx) {
        return dznrm2(&n, x, &incx);
    }
    
    // blas level 1, rescale: x = a*x
    inline // double * double vector
    void scal(const MKL_INT n, const double a, double *x, const MKL_INT incx) {
        dscal(&n, &a, x, &incx);
    }
    inline // double complex * double complex vector
    void scal(const MKL_INT n, const std::complex<double> a, std::complex<double> *x, const MKL_INT incx) {
        zscal(&n, &a, x, &incx);
    }
    inline // double * double complex vector
    void scal(const MKL_INT n, const double a, std::complex<double> *x, const MKL_INT incx) {
        zdscal(&n, &a, x, &incx);
    }
    
    
    // blas level 1, conjugated vector dot vector
    inline // double
    double dotc(const MKL_INT n, const double *x, const MKL_INT incx, const double *y, const MKL_INT incy) {
        return ddot(&n, x, &incx, y, &incy);
    }
    // comment 1: zdotc is a problematic function in lapack, when returning std::complex
    // comment 2: with my own version of dotc, it will slow things down without parallelization, need fix later
    inline // complex double
    std::complex<double> dotc(const MKL_INT n, const std::complex<double> *x, const MKL_INT incx,
                              const std::complex<double> *y, const MKL_INT incy) {
        std::complex<double> result(0.0, 0.0);
        cblas_zdotc_sub(n, x, incx, y, incy, &result);
        return result;
    }
    
    // blas level 3, matrix matrix product
    inline // double
    void gemm(const char transa, const char transb, const MKL_INT m, const MKL_INT n, const MKL_INT k,
              const double alpha, const double *a, const MKL_INT lda, const double *b, const MKL_INT ldb,
              const double beta, double *c, const MKL_INT ldc) {
        dgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }
    inline // complex double
    void gemm(const char transa, const char transb, const MKL_INT m, const MKL_INT n, const MKL_INT k,
              const std::complex<double> alpha, const std::complex<double> *a, const MKL_INT lda,
              const std::complex<double> *b, const MKL_INT ldb,
              const std::complex<double> beta, std::complex<double> *c, const MKL_INT ldc) {
        zgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }
    
    
    // sparse blas routines
    inline // double
    void csrgemv(const char transa, const MKL_INT m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y) {
        mkl_cspblas_dcsrgemv(&transa, &m, a, ia, ja, x, y);
    }
    inline // complex double
    void csrgemv(const char transa, const MKL_INT m, const std::complex<double> *a, const MKL_INT *ia, const MKL_INT *ja, const std::complex<double> *x, std::complex<double> *y) {
        mkl_cspblas_zcsrgemv(&transa, &m, a, ia, ja, x, y);
    }
    
    // for symmetric matrix (NOT Hermitian matrix)
    inline // double
    void csrsymv(const char uplo, const MKL_INT m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y) {
        mkl_cspblas_dcsrsymv(&uplo, &m, a, ia, ja, x, y);
    }
    inline // complex double
    void csrsymv(const char uplo, const MKL_INT m, const std::complex<double> *a, const MKL_INT *ia, const MKL_INT *ja, const std::complex<double> *x, std::complex<double> *y) {
        mkl_cspblas_zcsrsymv(&uplo, &m, a, ia, ja, x, y);
    }
    
    // more general function to perform matrix vector product in mkl
    inline // double
    void mkl_csrmv(const char transa, const MKL_INT m, const MKL_INT k, const double alpha, const char *matdescra,
                   const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre,
                   const double *x, const double beta, double *y) {
        mkl_dcsrmv(&transa, &m, &k, &alpha, matdescra, val, indx, pntrb, pntre, x, &beta, y);
    }
    inline // complex double
    void mkl_csrmv(const char transa, const MKL_INT m, const MKL_INT k, const std::complex<double> alpha, const char *matdescra,
                   const std::complex<double> *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre,
                   const std::complex<double> *x, const std::complex<double> beta, std::complex<double> *y) {
        mkl_zcsrmv(&transa, &m, &k, &alpha, matdescra, val, indx, pntrb, pntre, x, &beta, y);
    }
    
    inline // double
    void mkl_csrmm(const char transa, const MKL_INT m, const MKL_INT n, const MKL_INT k, const double alpha, const char *matdescra,
                   const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre,
                   const double *b, const MKL_INT ldb, const double beta, double *c, const MKL_INT ldc) {
        mkl_dcsrmm(&transa, &m, &n, &k, &alpha, matdescra, val, indx, pntrb, pntre, b, &ldb, &beta, c, &ldc);
    }
    inline // complex double
    void mkl_csrmm(const char transa, const MKL_INT m, const MKL_INT n, const MKL_INT k, const std::complex<double> alpha, const char *matdescra,
                   const std::complex<double> *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre,
                   const std::complex<double> *b, const MKL_INT ldb, const std::complex<double> beta, std::complex<double> *c, const MKL_INT ldc) {
        mkl_zcsrmm(&transa, &m, &n, &k, &alpha, matdescra, val, indx, pntrb, pntre, b, &ldb, &beta, c, &ldc);
    }
    
    // sparse blas, convert csr to csc
    inline // double
    void mkl_csrcsc(const MKL_INT *job, const MKL_INT n, double *Acsr, MKL_INT *AJ0, MKL_INT *AI0,
                    double *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info) {
        mkl_dcsrcsc(job, &n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info);
    }
    inline // complex double
    void mkl_csrcsc(const MKL_INT *job, const MKL_INT n, std::complex<double> *Acsr, MKL_INT *AJ0, MKL_INT *AI0,
                    std::complex<double> *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info) {
        mkl_zcsrcsc(job, &n, Acsr, AJ0, AI0, Acsc, AJ1, AI1, info);
    }
    
    // lapack computational routine, computes all eigenvalues of a real symmetric TRIDIAGONAL matrix using QR algorithm.
    inline // double
    lapack_int sterf(const lapack_int &n, double *d, double *e) {
        return LAPACKE_dsterf(n, d, e);
    }
    
    // lapack computational routine, computes all eigenvalues and (optionally) eigenvectors of a symmetric/hermitian TRIDIAGONAL matrix using the divide and conquer method.
    inline // double
    lapack_int stedc(const int &matrix_layout, const char &compz, const lapack_int &n, double *d, double *e, double *z, const lapack_int &ldz) {
        return LAPACKE_dstedc(matrix_layout, compz, n, d, e, z, ldz);
    }
    inline // complex double (for the unitary matrix which brings the original matrix to tridiagonal form)
    lapack_int stedc(const int &matrix_layout, const char &compz, const lapack_int &n, double *d, double *e, std::complex<double> *z, const lapack_int &ldz) {
        return LAPACKE_zstedc(matrix_layout, compz, n, d, e, z, ldz);
    }
    
    
    
    //// lapack symmetric eigenvalue driver routine, using divide and conquer, for band matrix
    //inline // double
    //lapack_int bevd(const int &matrix_layout, const char &jobz, const char &uplo, const lapack_int &n, const lapack_int &kd,
    //                double *ab, const lapack_int &ldab, double *w, double *z, const lapack_int &ldz) {
    //    return LAPACKE_dsbevd(matrix_layout, jobz, uplo, n, kd, ab, ldab, w, z, ldz);
    //}
    //lapack_int bevd(const int &matrix_layout, const char &jobz, const char &uplo, const lapack_int &n, const lapack_int &kd,
    //                std::complex<double> *ab, const lapack_int &ldab, double *w, std::complex<double> *z, const lapack_int &ldz) {
    //    return LAPACKE_zhbevd(matrix_layout, jobz, uplo, n, kd, ab, ldab, w, z, ldz);
    //}
    
    
    // lapack, Computes the QR factorization of a general m-by-n matrix.
    inline // double
    lapack_int geqrf(const int &matrix_layout, const lapack_int &m, const lapack_int &n, double *a, const lapack_int &lda, double *tau) {
        return LAPACKE_dgeqrf(matrix_layout, m, n, a, lda, tau);
    }
    inline // complex double
    lapack_int geqrf(const int &matrix_layout, const lapack_int &m, const lapack_int &n, std::complex<double> *a, const lapack_int &lda, std::complex<double> *tau) {
        return LAPACKE_zgeqrf(matrix_layout, m, n, a, lda, tau);
    }
    
    // lapack, Multiplies a real matrix by the orthogonal matrix Q of the QR factorization formed by ?geqrf or ?geqpf.
    inline // double
    lapack_int ormqr(const int &matrix_layout, const char &side, const char &trans,
                     const lapack_int &m, const lapack_int &n, const lapack_int &k,
                     const double *a, const lapack_int &lda, const double *tau, double *c, const lapack_int &ldc) {
        return LAPACKE_dormqr(matrix_layout, side, trans, m, n, k, a, lda, tau, c, ldc);
    }
    
    // lapack driver routine, computes all eigenvalues and, optionally, all eigenvectors of a hermitian matrix using divide and conquer algorithm.
    inline // double
    lapack_int heevd(const int &matrix_layout, const char &jobz, const char &uplo, const lapack_int &n, double *a, const lapack_int &lda, double *w) {
        return LAPACKE_dsyevd(matrix_layout, jobz, uplo, n, a, lda, w);
    }
    inline // complex double
    lapack_int heevd(const int &matrix_layout, const char &jobz, const char &uplo, const lapack_int &n, std::complex<double> *a, const lapack_int &lda, double *w) {
        return LAPACKE_zheevd(matrix_layout, jobz, uplo, n, a, lda, w);
    }
}


#endif /* qbasis_h */
