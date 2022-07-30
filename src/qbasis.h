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
#define lapack_complex_double MKL_Complex16
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <complex>
#include <filesystem>
#include <forward_list>
#include <functional>
#include <initializer_list>
#include <list>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "mkl.h"

#ifdef _OPENMP
  #include <omp.h>
#else
  #define omp_get_thread_num() 0
  #define omp_get_num_threads() 1
  #define omp_get_num_procs() 1
  #define omp_get_proc_bind() 0
#endif

#if defined(__clang__)
#elif defined(__GNUC__) && defined(_OPENMP)
  #include <parallel/algorithm>
  #define use_gnu_parallel_sort
#endif


namespace fs = std::filesystem;

namespace qbasis {

//  -------------part 0: global variables, forward declarations ----------------
//  ----------------------------------------------------------------------------
    /** @file qbasis.h
     *  \var const double pi
     *  \brief pi = 3.1415926...
     */
    extern const double pi;
    /** @file qbasis.h
     *  \var const double machine_prec
     *  \brief machine precision
     */
    extern const double machine_prec;

    extern const double opr_precision;
    extern const double sparse_precision;
    extern const double lanczos_precision;

    /** @file qbasis.h
     *  \var bool enable_ckpt
     *  \brief use checkpoint and restart in Lanczos, for long runs
     */
    extern bool enable_ckpt;

    /** @file qbasis.h
     *  \fn void initialize(const bool &enable_ckpt_)
     *  \brief initialize global variables & print out info
     */
    void initialize(const bool &enable_ckpt_=false);

    template <typename> class multi_array;
    template <typename T> void swap(multi_array<T>&, multi_array<T>&);

    /** \brief Multi-dimensional array */
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

        uint32_t dim() const { return dim_; }
        uint64_t size() const { return size_; }
        std::vector<uint64_t> linear_size() const { return linear_size_; }
        T& index(const std::vector<uint64_t> &pos);
        const T& index(const std::vector<uint64_t> &pos) const;
    private:
        uint32_t dim_;
        uint64_t size_;
        std::vector<uint64_t> linear_size_;
        std::vector<T> data;
    };
    typedef multi_array<uint32_t> MltArray_uint32;
    typedef multi_array<double> MltArray_double;
    typedef multi_array<std::pair<std::vector<uint32_t>,std::vector<uint32_t>>> MltArray_PairVec;


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
    bool q_bosonic(const std::vector<basis_prop> &props);
    void swap(mbasis_elem&, mbasis_elem&);
    bool operator<(const mbasis_elem&, const mbasis_elem&);
    bool operator==(const mbasis_elem&, const mbasis_elem&);
    bool operator!=(const mbasis_elem&, const mbasis_elem&);
    /** \brief DEPRECATED! */
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

    /** @file qbasis.h
     *  \fn void sort_basis_normal_order(std::vector<qbasis::mbasis_elem> &basis)
     *  \brief sort basis with a simple "<" comparison
     */
    void sort_basis_normal_order(std::vector<qbasis::mbasis_elem> &basis);

    /** @file qbasis.h
     *  \fn void sort_basis_Lin_order(const std::vector<basis_prop> &props, std::vector<qbasis::mbasis_elem> &basis)
     *  \brief sort basis according to Lin Table convention (Ib, then Ia)
     */
    void sort_basis_Lin_order(const std::vector<basis_prop> &props, std::vector<qbasis::mbasis_elem> &basis);

    // generate Lin Tables for a given basis
    void fill_Lin_table(const std::vector<basis_prop> &props, const std::vector<qbasis::mbasis_elem> &basis,
                        std::vector<MKL_INT> &Lin_Ja, std::vector<MKL_INT> &Lin_Jb);

    /** \brief (sublattice) for a given list of full basis, find the reps according to translational symmetry.
     *  Note: any state = T(disp2rep) * |rep>
     */
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
                                  const std::vector<std::pair<std::vector<std::vector<uint32_t>>,uint32_t>> &groups,
                                  std::vector<uint32_t> &omega_g,
                                  std::vector<uint32_t> &belong2group);
    // tabulate the maps for (ga,gb,ja,jb) -> (i,j), and (ga,gb,j) -> w
    void classify_Weisse_tables(const std::vector<basis_prop> &props_parent,
                                const std::vector<basis_prop> &props_sub,
                                const std::vector<mbasis_elem> &basis_sub_repr,
                                const lattice &latt_parent,
                                const std::vector<bool> &trans_sym,
                                const std::vector<uint64_t> &belong2rep,
                                const std::vector<std::vector<int>> &dist2rep,
                                const std::vector<uint32_t> &belong2group,
                                const std::vector<std::pair<std::vector<std::vector<uint32_t>>,uint32_t>> &groups_parent,
                                const std::vector<std::pair<std::vector<std::vector<uint32_t>>,uint32_t>> &groups_sub,
                                MltArray_PairVec &Weisse_e_lt, MltArray_PairVec &Weisse_e_eq, MltArray_PairVec &Weisse_e_gt,
                                MltArray_uint32 &Weisse_w_lt, MltArray_uint32 &Weisse_w_eq, MltArray_uint32 &Weisse_w_gt);
    // <r|P_k|r>^{-1}
    double norm_trans_repr(const std::vector<basis_prop> &props, const mbasis_elem &repr,
                           const lattice &latt_parent, const std::pair<std::vector<std::vector<uint32_t>>,uint32_t> &group_parent,
                           const std::vector<int> &momentum);

    // Operations on wavefunctions
    template <typename T> void swap(wavefunction<T>&, wavefunction<T>&);
    template <typename T> wavefunction<T> operator+(const wavefunction<T>&, const wavefunction<T>&);
    template <typename T> wavefunction<T> operator*(const wavefunction<T>&, const T&);
    template <typename T> wavefunction<T> operator*(const T&, const wavefunction<T>&);
    template <typename T> wavefunction<T> operator*(const mbasis_elem&, const T&);
    template <typename T> wavefunction<T> operator*(const T&, const mbasis_elem&);
    template <typename T> T inner_product(const wavefunction<T>&, const wavefunction<T>&);

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
    // wavefunction = opr * wavefunction (overwritten)
    template <typename T> void oprXphi(const opr<T>&, const std::vector<basis_prop>&, wavefunction<T>&, const bool &append = false);
    // wavefunction = opr * mbasis_elem
    template <typename T> void oprXphi(const opr<T>&, const std::vector<basis_prop>&, wavefunction<T>&, mbasis_elem, const bool &append = false);
    // wavefunction = opr * wavefunction0
    template <typename T> void oprXphi(const opr<T>&, const std::vector<basis_prop>&, wavefunction<T>&, wavefunction<T> phi0, const bool &append = false);

    // wavefunction = opr_prod * wavefunction (overwritten)
    template <typename T> void oprXphi(const opr_prod<T>&, const std::vector<basis_prop>&, wavefunction<T>&);
    // wavefunction = opr_prod * mbasis_elem
    template <typename T> void oprXphi(const opr_prod<T>&, const std::vector<basis_prop>&, wavefunction<T>&, mbasis_elem, const bool &append = false);
    // wavefunction = opr_prod * wavefunction0
    template <typename T> void oprXphi(const opr_prod<T>&, const std::vector<basis_prop>&, wavefunction<T>&, wavefunction<T> phi0, const bool &append = false);

    // wavefunction = mopr * mbasis_elem
    template <typename T> void oprXphi(const mopr<T>&, const std::vector<basis_prop>&, wavefunction<T>&, mbasis_elem, const bool &append = false);
    // wavefunction = mopr * wavefunction0
    template <typename T> void oprXphi(const mopr<T>&, const std::vector<basis_prop>&, wavefunction<T>&, wavefunction<T> phi0, const bool &append = false);

    // mopr * {a list of mbasis} -->> {a new list of mbasis}
    template <typename T> void gen_mbasis_by_mopr(const mopr<T>&, std::list<mbasis_elem>&, const std::vector<basis_prop>&,
                                                  std::vector<mopr<T>> conserve_lst = {},
                                                  std::vector<double> val_lst = {});

    void rm_mbasis_dulp_trans(const lattice&, std::list<mbasis_elem> &, const std::vector<basis_prop>&);

    // csr matrix
    template <typename T> void swap(csr_mat<T>&, csr_mat<T>&);

    // divide into two identical sublattices, if Nsites even. To be used in the divide and conquer method
    lattice divide_lattice(const lattice &parent);

    int basis_disk_read(const std::string &filename, std::vector<mbasis_elem> &basis);
    int basis_disk_write(const std::string &filename, const std::vector<mbasis_elem> &basis);



//  --------------------  part 1: basis of the wave functions ------------------
//  ----------------------------------------------------------------------------
    // ------------ basic info of a particular basis -----------------
    struct extra_info {
        uint8_t Nmax;   // maximum number of particles
        // more items can be filled here...
    };


    /** \brief The common properties of a given basis */
    class basis_prop {
    public:
        basis_prop() = default;
        basis_prop(const uint32_t &n_sites, const uint8_t &dim_local_,
                   const std::vector<uint32_t> &Nf_map = std::vector<uint32_t>(),
                   const bool &dilute_ = false);

        /** \brief constructor of basis_prop, by proving number of sites and name of basis
         *
         *  Implemented choices of basis name (s):
         *    - spin-1/2
         *    - spin-1
         *    - dimer
         *    - electron
         *    - tJ
         *    - spinless-fermion
         */
        basis_prop(const uint32_t &n_sites, const std::string &s, const extra_info &ex = extra_info{0});

        /** \brief question if corresponding to fermionic basis */
        bool q_fermion() const { return (! Nfermion_map.empty()); }

        /** \brief split (site by site) into basis_prop for two basis */
        void split(basis_prop &sub1, basis_prop &sub2) const;

        uint8_t dim_local;                      ///< local (single-site, single-orbital) dimension < 256
        uint8_t bits_per_site;                  ///< <= 8
        uint8_t bits_ignore;                    ///< for each orbital (with many sites), there are a few bits ignored
        uint16_t num_bytes;                     ///< for multi-orbital system, sum of num_bytes < 65536
        uint32_t num_sites;
        std::vector<uint32_t> Nfermion_map;     ///< Nfermion_map[i] corresponds to the number of fermions of state i
        std::string name;                       ///< store the name of the basis
        bool dilute;                            ///< if dilute, bit-rep is not a good representation
    };

    /** \brief Class for representing a quantum basis using bits
     *
     *  Fundamental class for basis elements.
     *  For given number of sites, and several orbitals, store the vectors of bits
     */
    class mbasis_elem {
        friend void swap(mbasis_elem&, mbasis_elem&);
        friend bool operator<(const mbasis_elem&, const mbasis_elem&);
        friend bool operator==(const mbasis_elem&, const mbasis_elem&);
        friend bool trans_equiv(const mbasis_elem&, const mbasis_elem&, const std::vector<basis_prop> &props, const lattice&);
        template <typename T> friend class wavefunction;
        template <typename T> friend void oprXphi(const opr<T>&, const std::vector<basis_prop>&, wavefunction<T>&, const bool&);
        template <typename T> friend void oprXphi(const opr<T>&, const std::vector<basis_prop>&, wavefunction<T>&, mbasis_elem, const bool&);
        template <typename T> friend void oprXphi(const opr_prod<T>&, const std::vector<basis_prop>&, wavefunction<T>&, mbasis_elem, const bool&);
        template <typename T> friend T inner_product(const wavefunction<T> &, const wavefunction<T> &);
        friend int basis_disk_read(const std::string&, std::vector<mbasis_elem>&);
        friend int basis_disk_write(const std::string&, const std::vector<mbasis_elem>&);
    public:
        /** \brief default constructor */
        mbasis_elem() : mbits(nullptr) {}

        /** \brief constructor with its properties */
        mbasis_elem(const std::vector<basis_prop> &props);

        /** \brief copy constructor */
        mbasis_elem(const mbasis_elem& old);

        /** \brief move constructor */
        mbasis_elem(mbasis_elem &&old) noexcept;

        /** \brief copy/move assignment constructor */
        mbasis_elem& operator=(mbasis_elem old) { swap(*this, old); return *this; }

        /** \brief destructor */
        ~mbasis_elem();

        //     ---------- status changes -----------
        /** \brief read out a state from a given site and given orbital */
        uint8_t siteRead(const std::vector<basis_prop> &props,
                         const uint32_t &site, const uint32_t &orbital) const;

        /** \brief write to a given site and given orbital */
        mbasis_elem& siteWrite(const std::vector<basis_prop> &props,
                               const uint32_t &site, const uint32_t &orbital, const uint8_t &val);

        /** \brief reset all bits to 0 for a particular orbital */
        mbasis_elem& reset(const std::vector<basis_prop> &props, const uint32_t &orbital);

        /** \brief reset all bits to 0 in all orbitals */
        mbasis_elem& reset();

        /** \brief change mbasis_elem to the next available state, for a particular orbital */
        mbasis_elem& increment(const std::vector<basis_prop> &props, const uint32_t &orbital);

        /** \brief change mbasis_elem to the next available state */
        mbasis_elem& increment(const std::vector<basis_prop> &props);

        /** \brief print the basis in the bit representation */
        void prt_bits(const std::vector<basis_prop> &props) const;

        /** \brief print the basis in numbers, each corresponding to a state on one site (vacuum not printed) */
        void prt_states(const std::vector<basis_prop> &props) const;

        /** \brief generate a file to plot the basis on a lattice */
        void plot_states(const std::vector<basis_prop> &props, const lattice &latt,
                         const std::string &filename) const;

        //    ----------- basic inquiries ----------
        /** \brief question if the vacuum state on a particular site */
        bool q_zero_site(const std::vector<basis_prop> &props, const uint32_t &site) const;

        /** \brief question if the vacuum state on a particular orbital */
        bool q_zero_orbital(const std::vector<basis_prop> &props, const uint32_t &orbital) const;

        /** \brief question if the vacuum state */
        bool q_zero() const;

        /** \brief question if every site occupied by the highest state, for a given orbital */
        bool q_maximized(const std::vector<basis_prop> &props, const uint32_t &orbital) const;

        /** \brief question if every site occupied by the highest state */
        bool q_maximized(const std::vector<basis_prop> &props) const;

        /** \brief question if every site occupied by the same state, for a given orbital */
        bool q_same_state_all_site(const std::vector<basis_prop> &props, const uint32_t &orbital) const;

        /** \brief question if every site occupied by the same state */
        bool q_same_state_all_site(const std::vector<basis_prop> &props) const;

        // get a label
        // preferred size of work: 2*num_sites
        uint64_t label(const std::vector<basis_prop> &props, const uint32_t &orbital,
                       std::vector<uint8_t> &work) const;

        // preferred size of work1: 2*num_sites, work2: 2*num_orbs
        uint64_t label(const std::vector<basis_prop> &props,
                       std::vector<uint8_t> &work1, std::vector<uint64_t> &work2) const;

        // preferred size of work: 2*num_sites
        void label_sub(const std::vector<basis_prop> &props, const uint32_t &orbital,
                       uint64_t &label1, uint64_t &label2, std::vector<uint8_t> &work) const;

        // preferred size of work1: 2*num_sites, work2: 4*num_orbs
        void label_sub(const std::vector<basis_prop> &props,
                       uint64_t &label1, uint64_t &label2,
                       std::vector<uint8_t> &work1, std::vector<uint64_t> &work2) const;

        // return a vector of length dim_local (for orbital), reporting # of each state
        std::vector<uint32_t> statistics(const std::vector<basis_prop> &props, const uint32_t &orbital) const;

        // a direct product of the statistics of all orbitals, size: dim_orb1 * dim_orb2 * ...
        std::vector<uint32_t> statistics(const std::vector<basis_prop> &props) const;

        /** \brief output the center of the non-vacuum states, in units of the lattice basis \f$ \vec{a}_i \f$ */
        std::vector<double> center_pos(const std::vector<basis_prop> &props, const lattice &latt) const;

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
        /*
        mbasis_elem& translate(const std::vector<basis_prop> &props,
                               const lattice &latt, const std::vector<int> &disp, int &sgn,
                               const uint32_t &orbital);
        mbasis_elem& translate(const std::vector<basis_prop> &props,
                               const lattice &latt, const std::vector<int> &disp, int &sgn);
        */

        /** \brief (OBC assumed; state 0 assumed to be vacuum) old state = T(-disp_vec) * centralized state */
        mbasis_elem& translate2center_OBC(const std::vector<basis_prop> &props,
                                          const lattice &latt, std::vector<int> &disp_vec);

        //    ------------ measurements ------------
        double diagonal_operator(const std::vector<basis_prop> &props, const opr<double> &lhs) const;

        std::complex<double> diagonal_operator(const std::vector<basis_prop> &props, const opr<std::complex<double>> &lhs) const;

        double diagonal_operator(const std::vector<basis_prop> &props, const opr_prod<double> &lhs) const;

        std::complex<double> diagonal_operator(const std::vector<basis_prop> &props, const opr_prod<std::complex<double>> &lhs) const;

        double diagonal_operator(const std::vector<basis_prop> &props, const mopr<double> &lhs) const;

        std::complex<double> diagonal_operator(const std::vector<basis_prop> &props, const mopr<std::complex<double>> &lhs) const;

    private:
        /** store an array of basis elements, for multi-orbital site (or unit cell)
         *  the first 2 bytes are used to store the total number of bytes used by this array
         *
         *  in terms of bits when perfoming "<" comparison (e.g. bits_per_site = 2):
         *  e.g.  0 1 0 1 0 0 0 0 1 1 0 1 0 0 0 0 1 1 1 0
         *                                            ^ ^
         *                                          /     \
         *                                     bits[1]   bits[0]
         *                                          \     /
         *                                          site[0]
         *  Note1: for arrangement of orbitals, they are similar:
         *          ...   orb[2]    orb[1]    orb[0]
         *  Note2: The wavefunction is always defined as:
         *         |alpha_0, beta_1, gamma_2, ... > = alpha_0^\dagger beta_1^\dagger gamma_2^\dagger ... |GS>
         *         (where alpha_i^\dagger is creation operator of state alpha on site i)
         */
        uint8_t* mbits;
    };


    // -------------- class for wave functions ---------------
    // Use with caution, may hurt speed when not used properly
    template <typename T> class wavefunction {
        friend class model<T>;
        friend void swap <> (wavefunction<T> &, wavefunction<T> &);
        friend wavefunction<T> operator+ <> (const wavefunction<T>&, const wavefunction<T>&);
        friend void oprXphi <> (const opr<T>&,      const std::vector<basis_prop>&, wavefunction<T>&, const bool&);
        friend void oprXphi <> (const opr<T>&,      const std::vector<basis_prop>&, wavefunction<T>&, mbasis_elem,     const bool&);
        friend void oprXphi <> (const opr<T>&,      const std::vector<basis_prop>&, wavefunction<T>&, wavefunction<T>, const bool&);
        friend void oprXphi <> (const opr_prod<T>&, const std::vector<basis_prop>&, wavefunction<T>&, mbasis_elem,     const bool&);
        friend void oprXphi <> (const opr_prod<T>&, const std::vector<basis_prop>&, wavefunction<T>&, wavefunction<T>, const bool&);
        friend void oprXphi <> (const mopr<T>&,     const std::vector<basis_prop>&, wavefunction<T>&, mbasis_elem,     const bool&);
        friend void oprXphi <> (const mopr<T>&,     const std::vector<basis_prop>&, wavefunction<T>&, wavefunction<T>, const bool&);
        friend T inner_product <> (const wavefunction<T> &, const wavefunction<T> &);
        friend void gen_mbasis_by_mopr <> (const mopr<T> &, std::list<mbasis_elem>&, const std::vector<basis_prop>&,
                                           std::vector<mopr<T>> conserve_lst, std::vector<double> val_lst);
    public:
        // constructor from props
        wavefunction(const std::vector<basis_prop> &props, const MKL_INT &capacity = 64);

        // constructor from an element
        wavefunction(const mbasis_elem &old, const MKL_INT &capacity = 64);
        wavefunction(mbasis_elem &&old,      const MKL_INT &capacity = 64);

        // copy constructor
        wavefunction(const wavefunction<T> &old) : bgn(old.bgn), end(old.end), total_bytes(old.total_bytes), ele(old.ele) {}

        // move constructor
        wavefunction(wavefunction<T> &&old) noexcept : bgn(old.bgn), end(old.end), total_bytes(old.total_bytes), ele(std::move(old.ele)) {}

        // copy assignment constructor and move assignment constructor, using "swap and copy"
        wavefunction& operator=(wavefunction<T> old)
        {
            swap(*this, old);
            return *this;
        }

        // destructor
        ~wavefunction() {}

        //    ----------- basic inquiries ----------
        MKL_INT size() const { MKL_INT capacity = static_cast<MKL_INT>(ele.size()); return ( (end + capacity - bgn) % capacity ); }

        // check if zero
        bool q_empty() const { return (size() == 0); }

        // check if full
        bool q_full() const { return (size() + 1 == static_cast<int>(ele.size())); }

        // check if sorted
        bool q_sorted() const;

        // check if sorted and there are no dulplicated terms
        bool q_sorted_fully() const;

        // for \sum_i \alpha_i * element[i], return \sum_i |\alpha_i|^2
        double amplitude() const;

        //    ---------------- print ---------------
        void prt_bits(const std::vector<basis_prop> &props) const;

        void prt_states(const std::vector<basis_prop> &props) const;

        //    ----------- element access -----------
        std::pair<mbasis_elem, T>& operator[](MKL_INT n);

        const std::pair<mbasis_elem, T>& operator[](MKL_INT n) const;

        //    ------------ arithmetics -------------
        wavefunction& reserve(const MKL_INT& capacity_new);                      // enlarge capacity

        wavefunction& clear();                                                   // clear but memory not released

        wavefunction& operator+=(mbasis_elem ele_new);

        wavefunction& operator+=(std::pair<mbasis_elem, T> ele_new);

        wavefunction& operator+=(wavefunction<T> rhs);

        wavefunction& operator*=(const T &rhs);

        // simplify
        wavefunction& simplify();

    private:
        // store an array of basis elements, and their corresponding coefficients
        // note: there should not be any dulplicated elements
        // using cycled queue data structure (wasting 1 element to avoid end==bgn confusion: empty or full)
        MKL_INT bgn;
        MKL_INT end;
        int total_bytes;
        std::vector<std::pair<mbasis_elem, T>> ele;

        // add one basis element (promised not to overlap)
        wavefunction& add(const mbasis_elem &rhs);
        // copy from one basis element (promised not to overlap)
        wavefunction& copy(const mbasis_elem &rhs);

        // add one basis element (promised not to overlap)
        wavefunction& add(const std::pair<mbasis_elem, T> &rhs);
        // copy from one basis element (promised not to overlap)
        wavefunction& copy(const std::pair<mbasis_elem, T> &rhs);

        // add another wavefunction (promised not to overlap)
        wavefunction& add(const wavefunction<T> &rhs);
        // copy from another wavefuction (promised not to overlap)
        wavefunction& copy(const wavefunction<T> &rhs);
    };




//  ------------------------------  part 2: operators --------------------------
//  ----------------------------------------------------------------------------

    /** \brief An operator on a given site and orbital
     *  (fundamental class for operators)
     */
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
        friend void oprXphi <> (const opr<T>&, const std::vector<basis_prop>&, wavefunction<T>&, const bool&);
        friend void oprXphi <> (const opr<T>&, const std::vector<basis_prop>&, wavefunction<T>&, mbasis_elem, const bool&);
        friend void oprXphi <> (const opr<T>&, const std::vector<basis_prop>&, wavefunction<T>&, wavefunction<T>, const bool&);
    public:
         /** \brief default constructor */
        opr() : mat(nullptr) {}

        /** \brief constructor from diagonal elements */
        opr(const uint32_t &site_, const uint32_t &orbital_, const bool &fermion_, const std::vector<T> &mat_);

        /** \brief constructor from a matrix */
        opr(const uint32_t &site_, const uint32_t &orbital_, const bool &fermion_, const std::vector<std::vector<T>> &mat_);

        /** \brief copy constructor */
        opr(const opr<T> &old);

        /** \brief move constructor */
        opr(opr<T> &&old) noexcept;

        /** \brief copy/move assignment constructor */
        opr& operator=(opr<T> old) { swap(*this, old); return *this; }

        /** \brief destructor */
        ~opr();

        /** \brief print details of opr */
        void prt() const;

        //    ----------- basic inquiries ----------
        /** \brief question if it is zero operator */
        bool q_zero() const;

        /** \brief question if it is identity operator */
        bool q_diagonal() const { return diagonal; }

        /** \brief question if it is identity operator */
        bool q_identity() const;

        /** \brief return site index */
        uint32_t pos_site() const { return site; }

        /** \brief return orbital index */
        uint32_t pos_orb() const { return orbital; }

        //    ------------ arithmetics -------------
        /**
         *  \brief \f$ \sqrt { \sum_{i,j} |M_{i,j}|^2 } \f$,
         *  where \f$ M \f$ is the matrix representation of the operator
         */
        double norm() const;

        /** \brief simplify the structure if possible */
        opr& simplify();

        /** \brief invert the sign */
        opr& negative();

        /** \brief take Hermitian conjugate */
        opr& dagger();

        /** \brief change site index */
        opr& change_site(const uint32_t &site_);

        /** \brief change the site index of operator
         *
         *  Say plan[i]=j, it means site i becomes site j after the transformation.
         *  If opr has site index i, then after transformation, the site index becomes j.
         */
        opr& transform(const std::vector<uint32_t> &plan);

        /** \brief opr = opr + rhs */
        opr& operator+=(const opr<T> &rhs);

        /** \brief opr = opr - rhs */
        opr& operator-=(const opr<T> &rhs);

        /** \brief opr = opr * rhs */
        opr& operator*=(const opr<T> &rhs);

        /** \brief opr = opr * rhs */
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
        friend void oprXphi <> (const opr_prod<T>&, const std::vector<basis_prop>&, wavefunction<T>&);
        friend void oprXphi <> (const opr_prod<T>&, const std::vector<basis_prop>&, wavefunction<T>&, mbasis_elem, const bool&);
        friend void oprXphi <> (const opr_prod<T>&, const std::vector<basis_prop>&, wavefunction<T>&, wavefunction<T>, const bool&);
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

        // Hermitian conjugate (not sorted after operation)
        opr_prod& dagger();

        // sort according to {site, orbital}. If exchanging fermions with odd times, invert sign.
        // replace in future, if used heavily
        opr_prod& bubble_sort();

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
        friend class model<T>;
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
        friend void oprXphi <> (const mopr<T>&, const std::vector<basis_prop>&, wavefunction<T>&, mbasis_elem, const bool&);
        friend void oprXphi <> (const mopr<T>&, const std::vector<basis_prop>&, wavefunction<T>&, wavefunction<T>, const bool&);
        friend void gen_mbasis_by_mopr <> (const mopr<T> &, std::list<mbasis_elem>&, const std::vector<basis_prop>&,
                                           std::vector<mopr<T>> conserve_lst, std::vector<double> val_lst);
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

        // Hermitian conjugate
        mopr& dagger();

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
// Note: sparse matrices in this code are using zero-based convention
// By default, all diagonal elements are stored, even if they are zero (to be compatible with pardiso)
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

        void use_full_matrix() { sym = false; }

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

        // construcotr from an lil_mat, and if sym_ == true, use only the upper triangle
        // then destroy the lil_mat
        csr_mat(lil_mat<T> &old);

        // matrix vector product
        // y = H * x + y
        void MultMv2(const T *x, T *y) const;
        // y = H * x
        void MultMv(T *x, T *y);              // non-const, to be compatible with arpack++

        std::vector<T> to_dense() const;

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

    // m step of Lanczos (for iram: m = k + np; for simple Lanczos, m depends on the convergence speed)
    // on entry, assuming k steps of Lanczos already performed
    // v of length m+1, hessenberg matrix of size m*m (m-step Lanczos)
    // after decomposition, mat * v[0:m-1] = v[0:m-1] * hessenberg + b[m] * v[m] * e_m^T,
    // where e_m has only one nonzero element: e[0:m-2] == 0, e[m-1] = 1
    //
    // maxit: maximum allowed Lanczos steps, m < maxit
    // maxit also serves as the leading dimension of hessenberg matrix
    // a[j] = hessenberg[j+maxit], diagonal of hessenberg matrix
    // b[j] = hessenberg[j]
    //  a[0]  b[1]      -> note: b[0] not used
    //  b[1]  a[1]  b[2]
    //        b[2]  a[2]  b[3]
    //              b[3]  a[3] b[4]
    //                    ..  ..  ..    b[k-1]
    //                          b[k-1]  a[k-1]
    // on entry:
    //     a[0], ..., a[k-1]       in hessenberg matrix
    //     b[0], ..., b[k-1], b[k] in hessenberg matrix
    // on exit:
    //     a[0], ..., a[m-1]       in hessenberg matrix
    //     b[0], ..., b[m-1], b[m] in hessenberg matrix
    //
    // if purpose == "iram":
    // on entry:
    //     v[0], ..., v[k] stored in v
    // on exit:
    //     v[0], ..., v[m] stored in v
    //
    // if purpose == "sr_val0" (smallest eigenvalue), or purpose == "dnmcs":
    // on entry:
    //     v[k-1], v[k] stored in v. If k%2==0, stored as {v[k],v[k-1]}; else stored as {v[k-1],v[k]}
    // on exit:
    //     v[m-1], v[m] stored in v. If m%2==0, stored as {v[m],v[m-1]}; else stored as {v[m-1],v[m]}
    //
    // if purpose == "sr_vec0" (smallest eigenvector):
    // same as "sr_val0", but with one extra column storing the eigenvector

    template <typename T, typename MAT>
    void lanczos(MKL_INT k, MKL_INT np, const MKL_INT &maxit, MKL_INT &m, const MKL_INT &dim,
                 const MAT &mat, T v[], double hessenberg[], const std::string &purpose);



    // Iterative sparse solver using conjugate gradient method
    // given well converged eigenvalue E0, and a good initital guess v[0], solves (H - E0) * v[j] = 0
    // on entry: assuming m-step finished, v contains the initital guess v[m]
    //           r[0] = -(H-E0)*v[0], p[0] = r[0]
    // on exit:  v rewritten by the converged solution
    template <typename T, typename MAT>
    void eigenvec_CG(const MKL_INT &dim, const MKL_INT &maxit, MKL_INT &m,
                     const MAT &mat, const T &E0, double &accu,
                     T v[], T r[], T p[], T pp[]);

    // compute eigenvalues and eigenvectors of hessenberg matrix
    // on entry, hessenberg should have leading dimension maxit
    // on exit, ritz of size m, s of size m*m
    // order = "sm", "lm", "sr", "lr", where 's': small, 'l': large, 'm': magnitude, 'r': real part
    void hess_eigen(const double hessenberg[], const MKL_INT &maxit, const MKL_INT &m,
                    const std::string &order, std::vector<double> &ritz, std::vector<double> &s);

    // transform from band storage to general storage
    // on exit, mat of size m*m
    void hess2dense(const double hessenberg[], const MKL_INT &maxit, const MKL_INT &m, std::vector<double> &mat);

    // --------------------------
    // ideally, here we should use the bulge-chasing algorithm; for this moment, we simply use the less efficient brute force QR factorization
    // --------------------------
    // QR factorization of hessenberg matrix, using np selected eigenvalues from ritz
    // [H, Q] = QR(H, shift1, shift2, ..., shift_np)
    // \tilde{H} = Q_np^T ... Q_1^T H Q_1 ... Q_np
    // \tilde{V} = V Q
    template <typename T>
    void perform_shifts(const MKL_INT &dim, const MKL_INT &m, const MKL_INT &np, const double shift[],
                        T v[], double hessenberg[], const MKL_INT &maxit, std::vector<double> &Q);

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
   /** In general, the position of any site can be represented as:
    *  \f[
    *   \vec{r} = \sum_{i=1}^{D} m_i^0 \vec{a}_i + \sum_{i=1}^D M_i \vec{A}_i + \vec{d}_s
    *           = \sum_{i=1}^{D} m_i \vec{a}_i + \vec{d}_s,
    *  \f]
    *  where \f$ m_i^0 \f$, \f$ m_i \f$, and \f$ M_i \f$ are integers,
    *  and \f$ \vec{d}_s \f$ are the position shift of sublattices.
    *  Note: \f$ m_i^0 \f$ is located in the 1st supercell defined by
    *  \f[
    *   \sum_{i=1}^D m_i^0 \vec{a}_i = \sum_{i=1}^D x_i \vec{A}_i,
    *  \f]
    *  where \f$ 0 \le x_i < 1 \f$.
    *  In other words, any site has two coordinate labels, one within the supercell,
    *  another as the supercell coordinate.
    *  Most of time, the supercell coordinate is set at origin; but they can be effective in some other cases.
    */
    class lattice {
        friend lattice divide_lattice(const lattice &parent);
    public:
        lattice() = default;

        /** \brief constructor from particular requirements. e.g. square, triangular... */
        lattice(const std::string &name, const std::vector<uint32_t> &L_,
                const std::vector<std::string> &bc_, bool auto_dim_spec = true);

        /** \brief constructor from manually designed lattices, specified from input files */
        lattice(const std::string &filename, bool auto_dim_spec = true);

        /** \brief if all directions untilted, return true */
        bool q_tilted() const;

        bool q_dividable() const;

        /** \brief for given \f$ m_i \f$, return its corresponding \f$ m_i^0 \f$ in the 1st supercell, and the supercell index \f$ M_i \f$ */
        void coor2supercell0(const int *coor, int *coor0, int *M) const;

        /** \brief with given \f$ m_i \f$, output the site index */
        void coor2site(const std::vector<int> &coor, const int &sub, uint32_t &site, std::vector<int> &work) const;

        // coordinates <-> site indices
        // first find a direction (dim_spec) which has even size, if not successful, use the following:
        // 1D: site = i * num_sub + sub
        // 2D: site = (i + j * L[0]) * num_sub + sub
        // 3D: site = (i + j * L[0] + k * L[0] * L[1]) * num_sub + sub
        // otherwise, the dim_spec should be counted first
        void coor2site_old(const std::vector<int> &coor, const int &sub, uint32_t &site) const;

        /** \brief for any site, output the coordinate \f$ m_i \f$ (NOT \f$ m_i^0 \f$). */
        void site2coor(std::vector<int> &coor, int &sub, const uint32_t &site) const;

        void site2coor_old(std::vector<int> &coor, int &sub, const uint32_t &site) const;

        /** \brief with given \f$ m_i \f$, output cartisian coordinates */
        void coor2cart(const std::vector<int> &coor, std::vector<double> &cart) const;

        /** \brief with given \f$ m_i \f$ and sublattice index, output cartisian coordinates */
        void coor2cart(const std::vector<int> &coor, std::vector<double> &cart, const int &sub) const;

        /** with given \f$ \vec{k}=\sum_i k_i \vec{b}_i \f$,
         *  return \f$ \vec{K} = \sum_i K_i \vec{B}_i \f$ and \f$ \tilde{k} = \sum_i \tilde{k}_i \vec{B}_i \f$,
         *  which satisfies \f$ \vec{k} = \vec{K} + \tilde{k} \f$.
         *  Here \f$ K_i \f$ are integers, and \f$ \tilde{k}_i \in [0,1) \f$.
         */
        void k2superBZ(const double *k, int *K, double *ktilde) const;

        /** \brief on return, the vector "plan" contains the positions of each site after translation T(disp) */
        void translation_plan(std::vector<uint32_t> &plan, const std::vector<int> &disp,
                              std::vector<int> &scratch_coor, std::vector<int> &scratch_work) const;


        // return a vector containing the positions of each site after rotation
        // x -> x', by (x' - x0) = R (x - x0)
        // roughly implemented, check before use!
        std::vector<uint32_t> rotation_plan(const uint32_t &origin, const double &angle) const;

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

        uint32_t dimension_spec() const {
            return dim_spec;
        }

        uint32_t num_sublattice() const {
            return num_sub;
        }

        uint32_t total_sites() const {
            return Nsites;
        }

        /** \brief return real space basis \f$ a_i \f$*/
        std::vector<std::vector<double>> basis_a() const {
            return a;
        }

        /** \brief return superlattice basis \f$ A_i \f$*/
        std::vector<std::vector<int>> basis_A() const {
            return A;
        }

        std::vector<std::vector<double>> basis_b() const {
            return b;
        }

        std::vector<std::vector<double>> basis_B() const {
            return B;
        }

        /** \brief return the center of the lattice */
        std::vector<double> center_pos() const {
            return center;
        }

        /** \brief return the positions of sublattices */
        std::vector<std::vector<double>> sublattice_pos() const {
            return pos_sub;
        }

        std::vector<uint32_t> Linear_size() const { return L; }

        uint32_t Lx() const { return L[0]; }
        uint32_t Ly() const { if (L.size() > 1) return L[1]; else return 1; }
        uint32_t Lz() const { if (L.size() > 2) return L[2]; else return 1; }

        // obtain all possible divisors of a lattice, for the divide and conquer method
        // the returned value is a list of lists: {{divisors for Lx}, {divisors for Ly}, ...}
        std::vector<std::vector<uint32_t>> divisor_v1(const std::vector<bool> &trans_sym) const;
        // {{1,1,1}, {1,1,2}, {1,1,5}, {1,2,1}, {1,2,2}, {1,2,5},...}, i.e., combine results from v1 to a single list
        //std::vector<std::vector<uint32_t>> divisor_v2(const std::vector<bool> &trans_sym) const;

        // enumerate all possible commensurate magnetic bravis basis, ordered by the unit cell size
        // note: equally, it gives all the possible translational subgroups (without dulplicates)
        // in the following format:
        // 1D: { pair( {{1}}, size=1 ),
        //       pair( {{2}}, size=2 ),
        //       ...
        //     }
        // 2D: { pair( {{1,0},{0,1}}, size=1 ),
        //       pair( {{1,0},{0,1}}, size=1 ),
        //        ...,
        //     }
        // 3D: { pair( {{1,0,0},{0,1,0},{0,0,1}}, size=1 ),
        //       pair( {{2,0,0},{0,1,0},{0,0,1}}, size=2 ),
        //       ...,
        //     }
        std::vector<std::pair<std::vector<std::vector<uint32_t>>,uint32_t>> trans_subgroups(const std::vector<bool> &trans_sym) const;

    private:
        /** \brief real space basis \f$ \vec{a}_i \f$ */
        std::vector<std::vector<double>> a;

        /** \brief reciprocal space basis \f$ \vec{b}_i \f$ */
        std::vector<std::vector<double>> b;  // momentum space basis

        /** \brief superlattice basis, in units of \f$ \vec{a}_i \f$ */
        std::vector<std::vector<int>> A;

        /** \brief superlattice basis in matrix format (column i -> A_i) */
        std::vector<double> Amat;

        /** \brief reciprocal superlattice basis, in units of \f$ \vec{b}_i \f$ */
        std::vector<std::vector<double>> B;

        /** \brief reciprocal superlattice basis in matrix format (column i -> B_i) */
        std::vector<double> Bmat;

        /** \brief position shift \f$ \vec{d}_s \f$ of each sublattice, in units of \f$ \vec{a}_i \f$ */
        std::vector<std::vector<double>> pos_sub;

        /** \brief true if {A1,A2,...} unparallel to {a1,a2,...} */
        std::vector<bool> tilted;

        /** \brief boundary condition. Only explicitly used when all in 1st supercell, otherwise too complicated */
        std::vector<std::string> bc;

        /** \brief dimension of lattice */
        uint32_t dim;

        /** \brief number of sublattices */
        uint32_t num_sub;

        /** \brief total number of sites in the 1st supercell */
        uint32_t Nsites;

        /** \brief the code starts labeling sites from a dimension which has even # of sites (if applicable) */
        uint32_t dim_spec;

        /** \brief DEPRECATED! linear size in each dimension */
        std::vector<uint32_t> L;

        /** \brief store the coordinate \f$ m_i \f$ (NOT \f$ m_i^0 \f$), and sublattice index for each site */
        std::vector<std::pair<std::vector<int>,int>> site2coor_map;

        /** \brief store the supercell index \f$ M_i \f$ for each site (empty vector if all in 1st supercell) */
        std::vector<std::vector<int>> site2super_map;

        /** \brief for given \f$ m_i^0 \f$ (in the 1st supercell) and sublattice index, return the site label */
        std::vector<std::map<std::vector<int>,uint32_t>> coor2site_map;

        /** \brief center of mass of the lattice, in units of \f$ \vec{a}_i \f$ */
        std::vector<double> center;

        // one more variable here, denoting the divide and conquer partition
        // if empty(false), then force to store the matrix when working with translational symmetry
    };


//  ---------------part 6: Routines to construct Hamiltonian -------------------
//  ----------------------------------------------------------------------------

    template <typename T> class model {
    public:
        bool matrix_free;                                                        ///< if generating matrix on the fly
        std::vector<basis_prop> props, props_sub_a, props_sub_b;
        mopr<T> Ham_diag;                                                        ///< diagonal part of H
        mopr<T> Ham_off_diag;                                                    ///< offdiagonal part of H
        mopr<T> Ham_vrnl;                                                        ///< used for generating Trugman's basis
        MKL_INT nconv;

        // controls which sector of basis to be active
        // by default sec_full = 0 (e.g. Sz=0 ground state sector of Heisenberg model);
        // when needed, sec_full will be switched to 1 to activate another sector (e.g. Sz=1 sector),
        // such setting can avoid messing up the code when calculating correlation functions
        uint32_t sec_sym;  ///< 0: work in dim_full; 1: work in dim_repr
        uint32_t sec_mat;  ///< which sector the matrix is relevant.

        std::vector<MKL_INT> dim_full;
        std::vector<MKL_INT> dim_repr;
        std::vector<MKL_INT> dim_vrnl;

        std::vector<std::vector<int>> momenta;
        std::vector<std::vector<double>> momenta_vrnl;

        /** \brief full basis without translation sym */
        std::vector<std::vector<mbasis_elem>> basis_full;
        /** \brief basis with translation sym */
        std::vector<std::vector<mbasis_elem>> basis_repr;
        /** \brief variational basis for Trugman's method */
        std::vector<std::vector<mbasis_elem>> basis_vrnl;
        /** \brief ground state representative for Trugman's method */
        mbasis_elem gs_vrnl;
        /** \brief basis for half lattice, used for building Weisse Table */
        std::vector<qbasis::mbasis_elem> basis_sub_full;
        /** \brief reps for half lattice */
        std::vector<qbasis::mbasis_elem> basis_sub_repr;

        // Lin tables, for both full basis and translation basis
        std::vector<std::vector<MKL_INT>> Lin_Ja_full;
        std::vector<std::vector<MKL_INT>> Lin_Jb_full;
        std::vector<std::vector<MKL_INT>> Lin_Ja_repr;
        std::vector<std::vector<MKL_INT>> Lin_Jb_repr;

        /** \brief 1 / <rep | P_k | rep> */
        std::vector<std::vector<double>> norm_repr;

        /** \brief 1 / <vac | P_k | vac> = omega_g, for the variational vacuum state */
        std::vector<double> gs_norm_vrnl;
        /** \brief omega_g, orbital size of the variational vacuum state */
        uint32_t gs_omegaG_vrnl;
        /** \brief ground state energy of the variational vacuum state */
        double gs_E0_vrnl;
        /** \brief ground state momentum */
        std::vector<double> gs_momentum_vrnl;

        std::vector<csr_mat<T>>            HamMat_csr_full;
        std::vector<csr_mat<T>>            HamMat_csr_repr;
        std::vector<csr_mat<T>>            HamMat_csr_vrnl;

        std::vector<double>                eigenvals_full;
        std::vector<T>                     eigenvecs_full;
        std::vector<double>                eigenvals_repr;
        std::vector<std::complex<double>>  eigenvecs_repr;
        std::vector<double>                eigenvals_vrnl;
        std::vector<std::complex<double>>  eigenvecs_vrnl;


        // ---------------- deprecated --------------------
        std::vector<std::vector<MKL_INT>>              basis_belong_deprec;   // size: dim_target_full, store the position of its repr
        std::vector<std::vector<std::complex<double>>> basis_coeff_deprec;    // size: dim_target_full, store the coeff
        std::vector<std::vector<MKL_INT>>              basis_repr_deprec;
        // ---------------- deprecated --------------------

        model() {}

        model(const lattice &latt, const uint32_t &num_secs = 5, const double &fake_pos_ = 100.0);

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

        void add_Ham(const opr<T> &rhs);

        void add_Ham(const opr_prod<T> &rhs);

        void add_Ham(const mopr<T> &rhs);

        void add_Ham_vrnl(const opr<T> &rhs);

        void add_Ham_vrnl(const opr_prod<T> &rhs);

        void add_Ham_vrnl(const mopr<T> &rhs);

        void switch_sec_mat(const uint32_t &sec_mat_);

        void fill_Weisse_table();

        // check if translational symmetry satisfied
        void check_translation();

        // naive way of enumerating all possible basis state
        void enumerate_basis_full(std::vector<mopr<T>> conserve_lst = {},
                                  std::vector<double> val_lst = {},
                                  const uint32_t &sec_full = 0);

        // Need to build Weiss Tables before enumerating representatives
        void enumerate_basis_repr(const std::vector<int> &momentum,
                                  std::vector<mopr<T>> conserve_lst = {},
                                  std::vector<double> val_lst = {},
                                  const uint32_t &sec_repr = 0);

        // build the variational basis to run Trugman's method
        void build_basis_vrnl(const std::list<mbasis_elem> &initial_list,
                              const mbasis_elem &gs,
                              const std::vector<double> &momentum_gs,
                              const std::vector<double> &momentum,
                              const uint32_t &iteration_depth,
                              std::vector<mopr<T>> conserve_lst = {},
                              std::vector<double> val_lst = {},
                              const uint32_t &sec_vrnl = 0);

        // momentum has to be in format {m,n,...} corresponding to (m/L1) b_1 + (n/L2) b_2 + ...
        void basis_init_repr_deprecated(const std::vector<int> &momentum,
                                        const uint32_t &sec_full = 0,
                                        const uint32_t &sec_repr = 0);

        // generate the Hamiltonian using basis_full
        void generate_Ham_sparse_full(const uint32_t &sec_full = 0,
                                      const bool &upper_triangle = true);

        // generate the Hamiltonian using basis_repr
        // a few artificial diagonal elements above 100, corresponding to zero norm states
        void generate_Ham_sparse_repr(const uint32_t &sec_repr = 0,
                                      const bool &upper_triangle = true);

        // generate the Hamiltonian using basis_vrnl
        void generate_Ham_sparse_vrnl(const uint32_t &sec_vrnl = 0,
                                      const bool &upper_triangle = true);

        // a few artificial diagonal elements above 100, corresponding to zero norm states
        void generate_Ham_sparse_repr_deprecated(const uint32_t &sec_full = 0,
                                                 const uint32_t &sec_repr = 0,
                                                 const bool &upper_triangle = true); // generate the Hamiltonian using basis_repr

        // generate a dense matrix of the Hamiltonian
        std::vector<std::complex<double>> to_dense(const uint32_t &sec_mat_ = 0);

        /** \brief y = H * x + y (matrix generated on the fly) */
        void MultMv2(const T *x, T *y) const;
        /** \brief y = H * x (matrix generated on the fly) */
        void MultMv(T *x, T *y);              // non-const, to be compatible with arpack++

        // Note: in this function, (nev, ncv, maxit) have different meanings comparing to IRAM!
        // 1 <= nev <= 2, nev-1 <= ncv <= nev
        // nev = 1, calcualte up to ground state energy
        // nev = 2, calculate up to 1st excited state energy
        // ncv = 1, calculate up to ground state eigenvector
        // ncv = 2, calculate up to 1st excited excited state eigenvector
        // sec_sym_=0: without translation; sec_sym_=1, with translation symmetry
        void locate_E0_lanczos(const uint32_t &sec_sym_, const MKL_INT &nev = 1, const MKL_INT &ncv = 1, MKL_INT maxit = 1000);

        /** \brief calculate the lowest eigenstates using IRAM
         *  nev, ncv, maxit following ARPACK definition
         *  sec_sym_ : 0 (full), 1 (repr), 2 (vrnl)
         */
        void locate_E0_iram(const uint32_t &sec_sym_, const MKL_INT &nev = 2, const MKL_INT &ncv = 6, MKL_INT maxit = 0);

        /** \brief calculate the highest eigenstates using IRAM
         *  nev, ncv, maxit following ARPACK definition.
         *  sec_sym_ : 0 (full), 1 (repr), 2 (vrnl).
         *  for repr and vrnl, there may be a few artificial eigenvalues above fake_pos (default to 100), corresponding to zero norm states.
         */
        void locate_Emax_iram(const uint32_t &sec_sym_, const MKL_INT &nev = 2, const MKL_INT &ncv = 6, MKL_INT maxit = 0);

        /** \brief return dim_full */
        std::vector<MKL_INT> dimension_full() const { return dim_full; }

        /** \brief return dim_repr */
        std::vector<MKL_INT> dimension_repr() const { return dim_repr; }

        /** \brief return dim_vrnl */
        std::vector<MKL_INT> dimension_vrnl() const { return dim_vrnl; }

        /** \brief return E0 */
        double energy_min() const { return E0; }

        /** \brief return Emax */
        double energy_max() const { return Emax; }

        /** \brief return gap */
        double energy_gap() const { return gap; }

        /** \brief lhs | phi >, where | phi > is an input state */
        void moprXvec_full(const mopr<T> &lhs, const uint32_t &sec_old, const uint32_t &sec_new,
                           const T* vec_old, T* vec_new) const;

        /** \brief lhs | phi >, where | phi > is an eigenstate */
        void moprXvec_full(const mopr<T> &lhs, const uint32_t &sec_old, const uint32_t &sec_new,
                           const MKL_INT &which_col, T* vec_new) const;

        /** transform a vector in the full space according to the plan */
        void transform_vec_full(const std::vector<uint32_t> &plan, const uint32_t &sec_full,
                                const T* vec_old, T* vec_new) const;

        /** transform an eigenvector in the full space according to the plan */
        void transform_vec_full(const std::vector<uint32_t> &plan, const uint32_t &sec_full,
                                const MKL_INT &which_col, T* vec_new) const;

        /** \brief project a state into a given momentum sector */
        void projectQ_full(const std::vector<int> &momentum, const uint32_t &sec_full,
                           const T* vec_old, T* vec_new) const;

        /** \brief project an eigenstate into a given momentum sector */
        void projectQ_full(const std::vector<int> &momentum, const uint32_t &sec_full,
                           const MKL_INT &which_col, T* vec_new) const;

        /** \brief < phi | lhs | phi > */
        T measure_full_static(const mopr<T> &lhs, const uint32_t &sec_full, const MKL_INT &which_col) const;

        /** \brief < phi |  ... * lhs2 * lhs1 * lhs0 | phi >, where sec_old has to be given for each lhs_i */
        T measure_full_static(const std::vector<mopr<T>> &lhs, const std::vector<uint32_t> &sec_old_list, const MKL_INT &which_col) const;

        /** \brief calculate dynamical structure factors
         *
         * \f[
         *  G_A(z) = \langle \phi | A_q^\dagger (z-H)^{-1} A_q | \phi \rangle,
         * \f]
         *  where \f$ z = \omega + E_0 + i \eta \f$, and
         *  norm   = \f$ \sqrt{\langle \phi | A_q^\dagger A_q | \phi \rangle} \f$.
         * \f[
         *  G_A(z) = \frac{\langle \phi | A_q^\dagger A_q | \phi \rangle}{z-a_0 -
         *           \frac{b_1^2}{z-a_1 -
         *           \frac{b_2^2}{z-a_2 - \cdots}}}
         * \f]
         *
         *  on exit: \f$ a_i \f$, \f$ b_i \f$ and norm are given.
         *  NEED enable CKPT later!!!
         */
        void measure_full_dynamic(const mopr<T> &Aq, const uint32_t &sec_old, const uint32_t &sec_new,
                                  const MKL_INT &maxit, MKL_INT &m, double &norm, double hessenberg[]) const;

        /** \brief \f$ A_q | \phi \rangle \f$
         *
         *  Requirements: after operation of Aq, the new state is still an eigenstate of translation operator.
         *  e.g. \f$ A_q = \frac{1}{\sqrt{N}} \sum_r e^{i q \cdot r} A_r \f$,
         *  the old state has momentum k, the new state has momentum k+q
         */
        void moprXvec_repr(const mopr<T> &Aq, const uint32_t &sec_old, const uint32_t &sec_new,
                           const T* vec_old, T* vec_new) const;

        /** \brief \f$ A_q | \phi \rangle  \f$, where \f$ | \phi \rangle \f$ is an eigenstate  */
        void moprXvec_repr(const mopr<T> &lhs, const uint32_t &sec_old, const uint32_t &sec_new,
                           const MKL_INT &which_col, T* vec_new) const;


        void transform_vec_repr(const std::vector<uint32_t> &plan, const uint32_t &sec_full,
                                const MKL_INT &which_col, std::vector<T> &vec_new) const;


        /** \brief < phi | lhs | phi > */
        T measure_repr_static(const mopr<T> &lhs, const uint32_t &sec_repr, const MKL_INT &which_col) const;

        /** \brief Calculate dynamical structure factors (in translational symmetric basis).
         *  For details, see measure_full_dynamic.
         */
        void measure_repr_dynamic(const mopr<T> &Aq, const uint32_t &sec_old, const uint32_t &sec_new,
                                  const MKL_INT &maxit, MKL_INT &m, double &norm, double hessenberg[]) const;

        /** \f[
         *     A_q | G(Q_0) \rangle = \sum_i \frac{p_i}{N} | \varphi_i (Q_0 + q) \rangle,
         *  \f]
         *  where \f$ | G(Q_0) \rangle \f$ is the variational ground state.
         *
         *  Note: in this input, will use
         *  \f[
         *     B_q \equiv \sqrt{N} A_q = \sum_i A_{r_i} e^{i q \cdot r_i},
         *  \f]
         *  since we cannot directly work with \f$ N \rightarrow \infty \f$.
         *  (summation only over the box)
         *
         *  Also, we output \f$ p_i \f$ instead of \f$ p_i/N \f$ for the same reason.
         */
        void moprXgs_vrnl(const mopr<T> &Bq, const uint32_t &sec_vrnl, T* vec_new) const;


        /** \f[
         *     A_q | vec \rangle = \sum_i \frac{p_i}{\sqrt{N}} | \varphi_i (k+q) \rangle + p_G | G(k+q) \rangle,
         *  \f]
         *  where \f$ | \varphi_k \rangle \f$ is a state with momentum k
         *
         *  Note: in this input, will use
         *  \f[
         *     B_q \equiv \sqrt{N} A_q = \sum_i A_{r_i} e^{i q \cdot r_i},
         *  \f]
         *  since we cannot directly work with \f$ N \rightarrow \infty \f$.
         *  (summation only over the box)
         *
         *  Also, we output \f$ p_i \f$ instead of \f$ p_i/\sqrt{N} \f$ for the same reason.
         */
        void moprXvec_vrnl(const mopr<T> &Bq, const uint32_t &sec_old, const uint32_t &sec_new,
                           const T* vec_old, T* vec_new, T &pG) const;

        void moprXvec_vrnl(const mopr<T> &Bq, const uint32_t &sec_old, const uint32_t &sec_new,
                           const MKL_INT &which_col, T* vec_new, T &pG) const;

        /** \brief < phi | lhs | phi >, where lhs is translational invariant */
        T measure_vrnl_static_trans_invariant(const mopr<T> &lhs, const uint32_t &sec_vrnl, const MKL_INT &which_col) const;

        /** Calculate dynamical structure factors (in Trugman basis).
         *  For details, see measure_full_dynamic.
         *  Note: the input \f$ B_q = \sqrt{N} A_q  \f$ (summation only over the box)
         */
        void measure_vrnl_dynamic(const mopr<T> &Bq, const uint32_t &sec_vrnl,
                                  const MKL_INT &maxit, MKL_INT &m, double &norm, double hessenberg[]) const;

        /** Build the matrix for calculating the observables in the Wannier state
         *
         *  \f[
         *     \mu_{k_1 k_2} = \langle \phi(k_1) | B_{k_1 - k_2} | \phi(k_2) \rangle ,
         *  \f]
         *  where
         *  \f[
         *     B_q = \sqrt{N} A_q =  \sum_i e^{i q \cdot r_i} A_{r_i}.
         *  \f]
         *  Note: \f$ \phi(k_1) \f$ and \f$ \phi(k_2) \f$ should be in the same band (same quantum numbers and well-defined dispersion).
         *  Thus, this function ONLY works if you really have a well defined band.
         *
         *  Each pair denotes the position \f$ r_i \f$ (in cartesin coordinates) and the operator \f$ A_{r_i} \f$.
         */
        void WannierMat_vrnl(const std::vector<std::pair<std::vector<double>,mopr<T>>> &Ar_list,
                             const uint32_t &sec_vrnl,
                             const std::vector<std::vector<double>> &momenta_list,
                             std::vector<std::complex<double>> &matrix_mu,
                             const std::function<MKL_INT(const model<T>&, const uint32_t&)> &locate_state);

        // later add conserved quantum operators and corresponding quantum numbers?
    private:
        double Emax;
        double E0;
        double E1;
        double gap;

        double fake_pos;

        lattice latt_parent;
        lattice latt_sub;
        std::vector<bool> trans_sym;                               // if translation allowed in each dimension
        bool dim_spec_involved;                                    // if translation allowed in partitioned direction

        // Weisse Tables for translation symmetry
        // Note: different from Weisse's paper, here we use Weisse_w to store the (parent) group label, instead of omega_g
        std::vector<uint64_t>              belong2rep_sub;
        std::vector<std::vector<int>>      dist2rep_sub;
        std::vector<std::pair<std::vector<std::vector<uint32_t>>,uint32_t>> groups_parent;
        std::vector<std::pair<std::vector<std::vector<uint32_t>>,uint32_t>> groups_sub;
        std::vector<uint32_t>              omega_g_sub;
        std::vector<uint32_t>              belong2group_sub;
        MltArray_PairVec                   Weisse_e_lt, Weisse_e_eq, Weisse_e_gt;
        MltArray_uint32                    Weisse_w_lt, Weisse_w_eq, Weisse_w_gt;

        void ckpt_lczsE0_init(bool &E0_done, bool &V0_done, bool &E1_done, bool &V1_done, std::vector<T> &v);

        void ckpt_lczsE0_updt(const bool &E0_done, const bool &V0_done, const bool &E1_done, const bool &V1_done);

    };



//  ---------------------------  Kernel polynomial  ----------------------------
//  ----------------------------------------------------------------------------

    /** \brief use Lanczos to determine the upper and lower bound of the eigenvalues of the matrix. */
    template <typename T, typename MAT>
    void energy_scale(const MKL_INT &dim, const MAT &mat, T v[], double &lo, double &hi,
                      const double &extend = 0.1, const MKL_INT &iters = 128);


//  --------------------------- Miscellaneous stuff ----------------------------
//  ----------------------------------------------------------------------------

    std::string date_and_time();

    inline int round2int(const double &x) { return static_cast<int>(x < 0.0 ? x - 0.5 : x + 0.5); }

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
    T2 dynamic_base(const std::vector<T1> &nums, const std::vector<T1> &base);   // deprecated
    template <typename T1, typename T2>
    void dynamic_base_vec2num(const MKL_INT &len, const T1* base, const T1* vec, T2 &num);

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


    template <typename T>
    void vec_swap(const MKL_INT &n, T *x, T *y);

    /** \beief fill x with zeros */
    template <typename T>
    void vec_zeros(const MKL_INT &n, T *x);

    /** \brief fill x with Lehmer 16807 random numbers,
     *  where seed == 0 reserved for filling 1/n to each element
     */
    template <typename T>
    void vec_randomize(const MKL_INT &n, T *x, const uint32_t &seed = 1);

    template <typename T>
    int vec_disk_read(const std::string &filename, MKL_INT n, T *x);

    template <typename T>
    int vec_disk_write(const std::string &filename, MKL_INT n, T *x);

    int basis_disk_read(const std::string &filename, std::vector<mbasis_elem> &basis);

    int basis_disk_write(const std::string &filename, const std::vector<mbasis_elem> &basis);

}



//  -------------------------  interface to mkl library ------------------------
//  ----------------------------------------------------------------------------
namespace qbasis {
    /** @file
     *  \fn void axpy(const MKL_INT n, const double alpha, const double *x, const MKL_INT incx, double *y, const MKL_INT incy)
     *  \brief blas level 1, y = a*x + y (double)
     */
    inline
    void axpy(const MKL_INT n, const double alpha, const double *x, const MKL_INT incx, double *y, const MKL_INT incy) {
        daxpy(&n, &alpha, x, &incx, y, &incy);
    }
    /** @file
     *  \fn void axpy(const MKL_INT n, const std::complex<double> alpha, const std::complex<double> *x, const MKL_INT incx, std::complex<double> *y, const MKL_INT incy)
     *  \brief blas level 1, y = a*x + y (complex double)
     */
    inline
    void axpy(const MKL_INT n, const std::complex<double> alpha, const std::complex<double> *x, const MKL_INT incx, std::complex<double> *y, const MKL_INT incy) {
        zaxpy(&n, &alpha, x, &incx, y, &incy);
    }

    /** @file
     *  \fn void copy(const MKL_INT n, const double *x, const MKL_INT incx, double *y, const MKL_INT incy)
     *  \brief blas level 1, y = x (double)
     */
    inline
    void copy(const MKL_INT n, const double *x, const MKL_INT incx, double *y, const MKL_INT incy) {
        dcopy(&n, x, &incx, y, &incy);
    }
    /** @file
     *  \fn void copy(const MKL_INT n, const std::complex<double> *x, const MKL_INT incx, std::complex<double> *y, const MKL_INT incy)
     *  \brief blas level 1, y = x (complex double)
     */
    inline
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
    // comment: zdotc is a problematic function in lapack, when returning std::complex. So using cblas here.
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
    /*
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
    */

    // general function to perform matrix vector product in mkl, deprecated since MKL 2018.3
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

    // lapack driver routine, Computes the solution to the system of linear equations with a square coefficient matrix A and multiple right-hand sides.
    inline // double
    lapack_int gesv(const int &matrix_layout, const lapack_int &n, const lapack_int &nrhs, double *a, const lapack_int &lda, lapack_int *ipiv, double *b, const lapack_int &ldb) {
        return LAPACKE_dgesv(matrix_layout, n, nrhs, a, lda, ipiv, b, ldb);
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
