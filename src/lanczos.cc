#include "lanczos.h"
#include "areig.h"
#include <iomanip>

namespace qbasis {
    
    
    
    // need further classification:
    // 1. provide an option that no intermidiate v stored, only hessenberg returned (probably in a separate routine)
    // 2. ask lanczos to restart with a new linearly independent vector when v_m+1 = 0
    // 3. need add DGKS re-orthogonalization
    template <typename T>
    void lanczos(MKL_INT k, MKL_INT np, const csr_mat<T> &mat, double &rnorm, T resid[],
                 T v[], double hessenberg[], const MKL_INT &ldh)
    {
        MKL_INT dim = mat.dimension();
        MKL_INT m   = k + np;
        assert(m <= ldh && k >= 0 && k < dim && np >=0 && np < dim);
        assert(m < dim);                                                              // # of orthogonal vectors: at most dim
        assert(std::abs(nrm2(dim, resid, 1) - 1.0) < lanczos_precision);              // normalized
        if (np == 0) return;
        std::vector<T*> vpt(m+1);
        for (MKL_INT j = 0; j < m; j++) vpt[j] = &v[j*dim];                           // pointers of v_0, v_1, ..., v_m
        vpt[m] = resid;
        copy(dim, resid, 1, vpt[k], 1);                                               // v_k = resid
        hessenberg[k] = rnorm;                                                        // beta_k = rnorm
        if (k == 0 && m > 1) {                                                        // prepare at least 2 vectors to start
            assert(std::abs(hessenberg[0]) < lanczos_precision);
            mat.MultMv(vpt[0], vpt[1]);                                               // w_0, not orthogonal to v[0] yet
            hessenberg[ldh] = std::real(dotc(dim, vpt[0], 1, vpt[1], 1));             // alpha[0]
            axpy(dim, -hessenberg[ldh], vpt[0], 1, vpt[1], 1);                        // w_0, orthogonal but not normalized yet
            hessenberg[1] = nrm2(dim, vpt[1], 1);                                     // beta[1]
            scal(dim, 1.0 / hessenberg[1], vpt[1], 1);                                // v[1]
            ++k;
            --np;
        }
        for (MKL_INT j = k; j < m-1; j++) {
            mat.MultMv(vpt[j], vpt[j+1]);
            axpy(dim, -hessenberg[j], vpt[j-1], 1, vpt[j+1], 1);                      // w_j
            hessenberg[ldh+j] = std::real(dotc(dim, vpt[j], 1, vpt[j+1], 1));         // alpha[j]
            axpy(dim, -hessenberg[ldh+j], vpt[j], 1, vpt[j+1], 1);                    // w_j
            hessenberg[j+1] = nrm2(dim, vpt[j+1], 1);                                 // beta[j+1]
            scal(dim, 1.0 / hessenberg[j+1], vpt[j+1], 1);                            // v[j+1]
        }
        mat.MultMv(vpt[m-1], vpt[m]);
        if(m > 1) axpy(dim, -hessenberg[m-1], vpt[m-2], 1, vpt[m], 1);
        hessenberg[ldh+m-1] = std::real(dotc(dim, vpt[m-1], 1, vpt[m], 1));           // alpha[k-1]
        axpy(dim, -hessenberg[ldh+m-1], vpt[m-1], 1, vpt[m], 1);                      // w_{k-1}
        rnorm = nrm2(dim, vpt[m], 1);
        scal(dim, 1.0 / rnorm, vpt[m], 1);                                            // v[k]
    }


    template <typename T>
    void hess2matform(const double hessenberg[], T mat[], const MKL_INT &m, const MKL_INT &ldh)
    {
        assert(m <= ldh);
        for (MKL_INT j = 0; j < m; j++) {
            for (MKL_INT i = 0; i < m; i++) {
                mat[i + j * ldh] = 0.0;
            }
        }
        mat[0] = hessenberg[ldh];
        if(m > 1) mat[ldh] = hessenberg[1];
        for (MKL_INT i =1; i < m-1; i++) {
            mat[i + i * ldh]     = hessenberg[i+ldh];
            mat[i + (i-1) * ldh] = hessenberg[i];
            mat[i + (i+1) * ldh] = hessenberg[i+1];
        }
        if (m > 1) {
            mat[m-1 + (m-1) * ldh] = hessenberg[ldh + m -1];
            mat[m-1 + (m-2) * ldh] = hessenberg[m-1];
        }
    }


    void select_shifts(const double hessenberg[], const MKL_INT &ldh, const MKL_INT &m,
                       const std::string &order, double ritz[], double s[])
    {
        assert(m>0 && ldh >= m);
        int info;
        copy(m, hessenberg + ldh, 1, ritz, 1);
        std::vector<double> e(m-1);
        copy(m-1, hessenberg + 1, 1, e.data(), 1);
        std::vector<double> eigenvecs;
        if (s == nullptr) {
            info = sterf(m, ritz, e.data());                                         // ritz rewritten by eigenvalues in ascending order
        } else {
            eigenvecs.resize(m*m);
            info = stedc(LAPACK_COL_MAJOR, 'I', m, ritz, e.data(), eigenvecs.data(), m);
        }
        assert(info == 0);
        std::vector<std::pair<double, MKL_INT>> eigenvals(m);
        for (decltype(eigenvals.size()) j = 0; j < eigenvals.size(); j++) {
            eigenvals[j].first = ritz[j];
            eigenvals[j].second = j;
        }
        if (order == "SR" || order == "sr") {        // smallest real part
            std::sort(eigenvals.begin(), eigenvals.end(),
                      [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return a.first < b.first; });
        } else if (order == "LR" || order == "lr") { // largest real part
            std::sort(eigenvals.begin(), eigenvals.end(),
                      [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return b.first < a.first; });
        } else if (order == "SM" || order == "sm") { // smallest magnitude
            std::sort(eigenvals.begin(), eigenvals.end(),
                      [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return std::abs(a.first) < std::abs(b.first); });
        } else if (order == "LM" || order == "lm") { // largest magnitude
            std::sort(eigenvals.begin(), eigenvals.end(),
                      [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return std::abs(b.first) < std::abs(a.first); });
        }
        for (decltype(eigenvals.size()) j = 0; j < eigenvals.size(); j++) ritz[j] = eigenvals[j].first;
        if (s != nullptr) {
            for (MKL_INT j = 0; j < m; j++) copy(m, eigenvecs.data() + m * eigenvals[j].second, 1, s + ldh * j, 1);
        }
    }

    // when there is time, re-write this subroutine with bulge-chasing
    template <typename T>
    void perform_shifts(const MKL_INT &dim, const MKL_INT &m, const MKL_INT &np, const double shift[],
                        double &rnorm, T resid[], T v[], double hessenberg[], const MKL_INT &ldh,
                        double Q[], const MKL_INT &ldq)
    {
        lapack_int info;
        const T zero = 0.0;
        const T one = 1.0;
        MKL_INT k = m - np;
        assert(np>0 && np < m);
        std::vector<double> hess_full(m*ldh), hess_qr(m*ldh);
        std::vector<double> tau(m);
        for (MKL_INT j = 0; j < m; j++) {                                              // Q = I
            for (MKL_INT i = 0; i < m; i++) Q[i + j * ldq] = (i == j ? 1.0 : 0.0);
        }
        hess2matform(hessenberg, hess_full.data(), m, ldh);
        for (MKL_INT j = 0; j < np; j++) {
            //std::cout << "QR shift with theta = " << shift[j] << std::endl;
            hess_qr = hess_full;
            for (MKL_INT i = 0; i < m; i++) hess_qr[i + i * ldh] -= shift[j];         // H - shift[j] * I
            info = geqrf(LAPACK_COL_MAJOR, m, m, hess_qr.data(), ldh, tau.data());    // upper triangle of hess_copy represents R, lower + tau represent Q
            assert(info == 0);
            info = ormqr(LAPACK_COL_MAJOR, 'L', 'T', m, m, m, hess_qr.data(), ldh,    // H_new = Q^T * H
                         tau.data(), hess_full.data(), ldh);
            assert(info == 0);
            info = ormqr(LAPACK_COL_MAJOR, 'R', 'N', m, m, m, hess_qr.data(), ldh,    // H_new = Q^T * H * Q
                         tau.data(), hess_full.data(), ldh);
            assert(info == 0);
            info = ormqr(LAPACK_COL_MAJOR, 'R', 'N', m, m, m, hess_qr.data(), ldh,    // Q_new = Q_old * Q
                         tau.data(), Q, ldq);
            assert(info == 0);
        }

        for (MKL_INT j = 0; j < k; j++) hessenberg[j + ldh] = hess_full[j + j * ldh]; // diagonal elements of hessenberg matrix
        for (MKL_INT j = 1; j < k; j++) hessenberg[j] = hess_full[j + (j-1) * ldh];   // subdiagonal
        std::vector<T> v_old(dim * m);
        copy(dim * m, v, 1, v_old.data(), 1);
        std::vector<T> Q_typeT(m*ldq);
        for (MKL_INT j = 0; j < m*ldq; j++) Q_typeT[j] = Q[j];
        gemm('n', 'n', dim, m, m, one, v_old.data(), dim, Q_typeT.data(), ldq, zero, v, dim); // v updated
        scal(dim, rnorm*Q_typeT[m-1 + ldq*(k-1)], resid, 1);
        axpy(dim, static_cast<T>(hess_full[k + (k-1)*ldh]), v+k*dim, 1, resid, 1);
        rnorm = nrm2(dim, resid, 1);                                                  // rnorm updated
        scal(dim, 1.0 / rnorm, resid, 1);                                             // resid updated




        //    // ------ check here, can be removed --------
        //    std::cout << "---------- Q ---------- " << std::endl;
        //    for (MKL_INT i=0; i < m; i++) {
        //        for (MKL_INT j = 0; j < m; j++) {
        //            double out = std::abs(Q[i + j * ldq])>lanczos_precision?Q[i + j * ldq]:0.0;
        //            std::cout << std::setw(15) << out;
        //        }
        //        std::cout << std::endl;
        //    }
        //    std::cout << std::endl;
        // ------ check here, can be removed --------
        //    // ------ check here, can be removed --------
        //    std::cout << "---------- Q^T Q ---------- " << std::endl;
        //    std::vector<double> productQ(m*m, 0.0);
        //    gemm('t', 'n', m, m, m, 1.0, Q, ldq, Q, ldq, 0.0, productQ.data(), m);
        //    for (MKL_INT i=0; i < m; i++) {
        //        for (MKL_INT j = 0; j < m; j++) {
        //            double out = std::abs(productQ[i + j * m])>lanczos_precision?productQ[i + j * m]:0.0;
        //            std::cout << std::setw(15) << out;
        //        }
        //        std::cout << std::endl;
        //    }
        //    std::cout << std::endl;
        //    // ------ check here, can be removed --------
        //    // ------ check here, can be removed --------
        //    std::cout << "---------- Q^T H Q ---------- " << std::endl;
        //    hess2matform(hessenberg, hess_qr.data(), m, ldh);
        //    for (MKL_INT j = 0; j < m*m; j++) productQ[j] = 0.0;
        //    gemm('t', 'n', m, m, m, 1.0, Q, ldq, hess_qr.data(), ldh, 0.0, productQ.data(), m);
        //    std::vector<double> productQ2(productQ);
        //    gemm('n', 'n', m, m, m, 1.0, productQ2.data(), m, Q, ldq, 0.0, productQ.data(), m);
        //    for (MKL_INT i=0; i < m; i++) {
        //        for (MKL_INT j = 0; j < m; j++) {
        //            double out = std::abs(productQ[i + j * m])>1e-12?productQ[i + j * m]:0.0;
        //            std::cout << std::setw(15) << out;
        //        }
        //        std::cout << std::endl;
        //    }
        //    std::cout << std::endl;
        //    // ------ check here, can be removed --------

    }

    // interface to arpack++
    void call_arpack(csr_mat<double> &mat, double v0[],
                     const MKL_INT &nev, const MKL_INT &ncv, MKL_INT &nconv,
                     const std::string &order, double eigenvals[], double eigenvecs[])
    {
        ARSymStdEig<double, csr_mat<double>>
            prob(mat.dimension(), nev, &mat, &csr_mat<double>::MultMv, order, ncv, 0.0, 0, v0);
        prob.EigenValVectors(eigenvecs, eigenvals);
        nconv = prob.ConvergedEigenvalues();
    }
    void call_arpack(csr_mat<std::complex<double>> &mat, std::complex<double> v0[],
                     const MKL_INT &nev, const MKL_INT &ncv, MKL_INT &nconv,
                     const std::string &order, double eigenvals[], std::complex<double> eigenvecs[])
    {
        std::complex<double> *eigenvals_copy = new std::complex<double>[nev];
        ARCompStdEig<double, csr_mat<std::complex<double>>> prob(mat.dimension(), nev, &mat,
                                                                 &csr_mat<std::complex<double>>::MultMv,
                                                                 order, ncv, 0.0, 0, v0);
        std::cout << "bench1" << std::endl;
        prob.EigenValVectors(eigenvecs, eigenvals_copy);
        std::cout << "bench2" << std::endl;
        nconv = prob.ConvergedEigenvalues();
        for (MKL_INT j = 0; j < nconv; j++) {
            assert(std::abs(eigenvals_copy[j].imag()) < lanczos_precision);
            eigenvals[j] = eigenvals_copy[j].real();
        }
        delete [] eigenvals_copy;
    }

    template <typename T>
    void iram(csr_mat<T> &mat, T v0[], const MKL_INT &nev, const MKL_INT &ncv, MKL_INT &nconv,
              const std::string &order, double eigenvals[], T eigenvecs[], const bool &use_arpack)
    {
        MKL_INT dim = mat.dimension();
        MKL_INT np = ncv - nev;

//        arma::SpMat<T> sp_csc;

        if (use_arpack) {
            std::string order_cap(order);
            std::transform(order_cap.begin(), order_cap.end(), order_cap.begin(), ::toupper);
            call_arpack(mat, v0, nev, ncv, nconv, order_cap, eigenvals, eigenvecs);
            for (MKL_INT j = 0; j < nconv; j++) {
                std::cout << "j = " << j << ", E_j = " << eigenvals[j] << std::endl;
            }
            

        } else {                                                                       // hand-coded arpack
            std::vector<T> resid(dim, static_cast<T>(0.0)), v(dim*ncv);
            std::vector<double> hessenberg(ncv*ncv), ritz(ncv), Q(ncv*ncv);
            double rnorm = nrm2(dim, v0, 1);
            axpy(dim, 1.0/rnorm, v0, 1, resid.data(), 1);                              // resid = v0 normalized
            rnorm = 0.0;
            lanczos(0, ncv, mat, rnorm, resid.data(), v.data(), hessenberg.data(), ncv);

            MKL_INT step = 0, step_max=10;
            while (step < step_max) {
                select_shifts(hessenberg.data(), ncv, ncv, order, ritz.data());
                perform_shifts(dim, ncv, np, ritz.data()+nev, rnorm, resid.data(), v.data(),
                               hessenberg.data(), ncv, Q.data(), ncv);
                lanczos(nev, np, mat, rnorm, resid.data(), v.data(), hessenberg.data(), ncv);
                for (MKL_INT j = 0; j < nev; j++) {
                    std::cout << std::setw(16) << ritz[j];
                }
                std::cout << std::endl;
                step++;
            }
        }


    }

    // Explicit instantiation
    template void lanczos(MKL_INT k, MKL_INT np, const csr_mat<double> &mat,
                          double &rnorm, double resid[], double v[], double hessenberg[], const MKL_INT &ldh);
    template void lanczos(MKL_INT k, MKL_INT np, const csr_mat<std::complex<double>> &mat,
                          double &rnorm, std::complex<double> resid[], std::complex<double> v[], double hessenberg[], const MKL_INT &ldh);

    template void hess2matform(const double hessenberg[], double mat[], const MKL_INT &m, const MKL_INT &ldh);
    template void hess2matform(const double hessenberg[], std::complex<double> mat[], const MKL_INT &m, const MKL_INT &ldh);

    template void perform_shifts(const MKL_INT &dim, const MKL_INT &m, const MKL_INT &np, const double shift[],
                                 double &rnorm, double resid[], double v[], double hessenberg[], const MKL_INT &ldh,
                                 double Q[], const MKL_INT &ldq);
    template void perform_shifts(const MKL_INT &dim, const MKL_INT &m, const MKL_INT &np, const double shift[],
                                 double &rnorm, std::complex<double> resid[], std::complex<double> v[], double hessenberg[], const MKL_INT &ldh,
                                 double Q[], const MKL_INT &ldq);

    template void iram(csr_mat<double> &mat, double v0[],
                       const MKL_INT &nev, const MKL_INT &ncv, MKL_INT &nconv,
                       const std::string &order, double eigenvals[], double eigenvecs[],
                       const bool &use_arpack);
    template void iram(csr_mat<std::complex<double>> &mat, std::complex<double> v0[],
                       const MKL_INT &nev, const MKL_INT &ncv, MKL_INT &nconv,
                       const std::string &order, double eigenvals[], std::complex<double> eigenvecs[],
                       const bool &use_arpack);

}
