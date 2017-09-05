#include <iostream>
#include <iomanip>
#include <algorithm>
#include "qbasis.h"
#include "areig.h"
namespace qbasis {
    
    
    void log_Lanczos_srval(const MKL_INT &k, const std::vector<double> &ritz,
                           const double hessenberg[], const MKL_INT &maxit,
                           const double &accuracy,
                           const double &accu_E0, const double &accu_E1,
                           const std::string &filename)
    {
        assert(k > 2);
        std::ofstream fout(filename, std::ios::out | std::ios::app);
        fout << std::setprecision(10);
        fout << std::setw(20) << "#(1)" << std::setw(20) << "(2)" << std::setw(20) << "(3)"
             << std::setw(20) << "(4)"  << std::setw(20) << "(5)" << std::setw(20) << "(6)"
             << std::setw(20) << "(7)"  << std::setw(20) << "(8)" << std::setw(20) << "(9)"
             << std::setw(20) << "(10)" << std::endl;
        fout << std::setw(20) << "Iter(k)"
             << std::setw(20) << "Ritz[0]"  << std::setw(20) << "Ritz[1]"
             << std::setw(20) << "Ritz[2]"  << std::setw(20) << "Ritz[3]"
             << std::setw(20) << "a[k-1]"     << std::setw(20) << "b[k]"
             << std::setw(20) << "accuracy" << std::setw(20) << "accu_E0"
             << std::setw(20) << "accu_E1"  << std::endl;
        fout << std::setw(20) << k
             << std::setw(20) << ritz[0] << std::setw(20) << ritz[1]
             << std::setw(20) << ritz[2] << std::setw(20) << ritz[3]
             << std::setw(20) << hessenberg[maxit+k-1] << std::setw(20) << hessenberg[k]
             << std::setw(20) << accuracy << std::setw(20) << accu_E0
             << std::setw(20) << accu_E1 << std::endl;
        fout.close();
    }
    
    // need further classification:
    // 1. ask lanczos to restart with a new linearly independent vector when v_m+1 = 0
    // 2. add DGKS re-orthogonalization (when purpose == iram)
    // 3. add partial and selective re-orthogonalization (when purpose == sr_xxx)
    template <typename T, typename MAT>
    void lanczos(MKL_INT k, MKL_INT np, const MKL_INT &maxit, MKL_INT &m, const MKL_INT &dim,
                 const MAT &mat, T v[], double hessenberg[], const std::string &purpose)
    {
        MKL_INT mm = k + np;
        assert(mm < maxit && k >= 0 && k < dim && np >=0 && np < dim);
        assert(mm < dim);                                                        // # of orthogonal vectors: at most dim
        auto &npos = std::string::npos;
        m = k;
        if (np == 0) return;
        T zero = static_cast<T>(0.0);
        std::vector<T*> vpt(mm+1);                                               // pointers of v[0],v[1],...,v[m]
        T* phipt = &v[2*dim];                                                    // for ground state eigenvector
        T* ypt = (purpose.find("vec1") != npos) ? &v[3*dim] : &v[2*dim];         // for eigenvector y
        
        if (purpose == "iram") {                                                 // v has m+1 cols
            for (MKL_INT j = 0; j <= mm; j++) vpt[j] = &v[j*dim];
        } else {                                                                 // v has only 2 or 3 cols
            for (MKL_INT j = 0; j <= mm; j++) vpt[j] = &v[(j%2)*dim];
        }
        
        std::vector<double> ritz(mm), s(mm * mm);                                // Ritz values and eigenvecs of Hess
        if (purpose.find("vec") != npos) hess_eigen(hessenberg, maxit, mm, "sr", ritz, s);
        
        assert(std::abs(nrm2(dim, vpt[k], 1) - 1.0) < lanczos_precision);        // v[k] should be normalized
        if (k == 0) {                                                            // prepare 2 vectors to start
            hessenberg[0] = 0.0;
            for (MKL_INT l = 0; l < dim; l++) vpt[1][l] = zero;                  // v[1] = 0
            mat.MultMv2(vpt[0], vpt[1]);                                         // v[1] = H * v[0] + v[1]
            if (purpose == "iram" || purpose.find("val") != npos) {              // a[0] = (v[0], v[1])
                hessenberg[maxit] = std::real(dotc(dim, vpt[0], 1, vpt[1], 1));
            } else if (purpose.find("vec") != npos) {
                assert(std::abs(hessenberg[maxit] - std::real(dotc(dim, vpt[0], 1, vpt[1], 1))) < lanczos_precision);
            } else {
                assert(false);
            }
            axpy(dim, -hessenberg[maxit], vpt[0], 1, vpt[1], 1);                 // v[1] = v[1] - a[0] * v[0]
            if (purpose == "iram" || purpose.find("val") != npos) {              // b[1] = || v[1] ||
                hessenberg[1] = nrm2(dim, vpt[1], 1);
            } else if (purpose.find("vec") != npos) {
                assert(std::abs(hessenberg[1] - nrm2(dim, vpt[1], 1)) < lanczos_precision);
            } else {
                assert(false);
            }
            scal(dim, 1.0 / hessenberg[1], vpt[1], 1);                           // v[1] = v[1] / b[1]
            m = ++k;
            --np;
            if (purpose.find("vec") != npos) axpy(dim, s[m], vpt[m], 1, ypt, 1); // y += s[m] * v[m]
        }
        
        double theta0_prev, theta1_prev;                                         // record Ritz values from last step
        int cnt_accuE0 = 0;
        do {                                                                     // while m < mm
            m++;
            for (MKL_INT l = 0; l < dim; l++)
                vpt[m][l] = -hessenberg[m-1] * vpt[m-2][l];                      // v[m] = -b[m-1] * v[m-2]
            mat.MultMv2(vpt[m-1], vpt[m]);                                       // v[m] = H * v[m-1] + v[m]
            
            if (purpose == "iram" || purpose.find("val") != npos) {              // a[m-1] = (v[m-1], v[m])
                hessenberg[maxit+m-1] = std::real(dotc(dim, vpt[m-1], 1, vpt[m], 1));
            } else if (purpose.find("vec") != npos) {
                assert(std::abs(hessenberg[maxit+m-1] - std::real(dotc(dim, vpt[m-1], 1, vpt[m], 1))) < lanczos_precision);
            } else {
                assert(false);
            }
            axpy(dim, -hessenberg[maxit+m-1], vpt[m-1], 1, vpt[m], 1);           // v[m] = v[m] - a[m-1] * v[m-1]
            if (purpose == "iram" || purpose.find("val") != npos) {              // b[m] = || v[m] ||
                hessenberg[m] = nrm2(dim, vpt[m], 1);
            } else if (purpose.find("vec") != npos) {
                assert(std::abs(hessenberg[m] - nrm2(dim, vpt[m], 1)) < lanczos_precision);
            } else {
                assert(false);
            }
            scal(dim, 1.0 / hessenberg[m], vpt[m], 1);                           // v[m] = v[m] / b[m]
            if (purpose.find("val1") != npos || purpose.find("vec1") != npos) {  // re-orthogonalization again phi0
                auto temp = dotc(dim, phipt, 1, vpt[m], 1);
                if (std::abs(temp) > lanczos_precision) {
                    std::cout << "-" << std::flush;
                    axpy(dim, -temp, phipt, 1, vpt[m], 1);
                    double rnorm = nrm2(dim, vpt[m], 1);
                    scal(dim, 1.0 / rnorm, vpt[m], 1);
                }
            }
            
            if (purpose.find("val") != npos) {
                hess_eigen(hessenberg, maxit, m, "sr", ritz, s);                  // calculate {theta, s}
                if (m > 3) {
                    double accuracy = std::abs(hessenberg[m] * s[m-1]);
                    double accu_E0  = std::abs((ritz[0] - theta0_prev) / ritz[0]);
                    double accu_E1  = std::abs((ritz[1] - theta1_prev) / ritz[1]);
                    log_Lanczos_srval(m, ritz, hessenberg, maxit, accuracy, accu_E0, accu_E1,
                                      "log_Lanczos_"+purpose+".txt");
                    if (accu_E0 < lanczos_precision) {
                        cnt_accuE0++;
                    } else {
                        cnt_accuE0 = 0;
                    }
                    if ( (cnt_accuE0 > 15) &&
                        (accuracy < lanczos_precision ||
                        (purpose.find("rough") != npos && accu_E0 < lanczos_precision))) break;
                }
                theta0_prev = ritz[0];
                theta1_prev = ritz[1];
            }
            if (purpose.find("vec") != npos) axpy(dim, s[m], vpt[m], 1, ypt, 1); // y += s[m] * v[m]
            
            // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // naive re-orthogonalization, change checking criteria and replace with DGKS later
            if (purpose == "iram") {
                for (MKL_INT l = 0; l < m-1; l++) {
                    auto q = dotc(dim, vpt[l], 1, vpt[m], 1);
                    double qabs = std::abs(q);
                    if (qabs > lanczos_precision) {
                        axpy(dim, -q, vpt[l], 1, vpt[m], 1);
                        scal(dim, 1.0 / std::sqrt(1.0 - qabs * qabs), vpt[m], 1);
                    }
                }
            }
        } while (m < mm);
        std::cout << std::endl;
    }
    template void lanczos(MKL_INT k, MKL_INT np, const MKL_INT &maxit, MKL_INT &m, const MKL_INT &dim,
                          const csr_mat<double> &mat, double v[],
                          double hessenberg[], const std::string &purpose);
    template void lanczos(MKL_INT k, MKL_INT np, const MKL_INT &maxit, MKL_INT &m, const MKL_INT &dim,
                          const csr_mat<std::complex<double>> &mat, std::complex<double> v[],
                          double hessenberg[], const std::string &purpose);
    template void lanczos(MKL_INT k, MKL_INT np, const MKL_INT &maxit, MKL_INT &m, const MKL_INT &dim,
                          const model<std::complex<double>> &mat, std::complex<double> v[],
                          double hessenberg[], const std::string &purpose);
//    template void lanczos(MKL_INT k, MKL_INT np, MKL_INT &mm, const MKL_INT &dim,
//                          const model<double> &mat, double v[],
//                          double hessenberg[], const MKL_INT &ldh, const std::string &purpose);
    
    
    template <typename T, typename MAT>
    void eigenvec_CG(const MKL_INT &dim, const MKL_INT &maxit, MKL_INT &m,
                     const MAT &mat, const T &E0, double &accu,
                     T v[], T r[], T p[], T pp[])
    {
        assert(m >= 0 && m < maxit);
        assert(std::abs(nrm2(dim, v, 1) - 1.0) < lanczos_precision);
        for (MKL_INT j = 0; j < dim; j++) assert(std::abs(p[j] - r[j]) < lanczos_precision);
        accu = nrm2(dim, r, 1);
        std::ofstream fout("log_CG.txt", std::ios::out | std::ios::app);
        fout << std::setprecision(10);
        fout << std::setw(20) << accu << std::endl;
        fout.close();
        while (m < maxit && accu > lanczos_precision) {
            copy(dim, p, 1, pp, 1);
            scal(dim, machine_prec - E0, pp, 1);
            mat.MultMv2(p,pp);                                                   // pp[m]    = (H - E0) * p[m]
            T delta = dotc(dim, p, 1, pp, 1);                                    // delta[m] = (p[m], pp[m])
            T alpha = accu * accu / delta;                                       // alpha[m] = gamma[m]^2 / delta[m]
            axpy(dim,  alpha,  p, 1, v, 1);                                      // v[m+1]   = v[m] + alpha[m] * p[m]
            axpy(dim, -alpha, pp, 1, r, 1);                                      // r[m+1]   = r[m] - alpha[m] * pp[m]
            double beta = nrm2(dim, r, 1) / accu;                                // beta[m]  = gamma[m+1] / gamma[m]
            scal(dim, static_cast<T>(beta*beta), p, 1);
            axpy(dim, static_cast<T>(1.0), r, 1, p, 1);                          // p[m+1]   = r[m+1] + beta^2 * p[m]
            accu *= beta;                                                        // gamma[m+1] = beta[m] * gamma[m]
            m++;
            std::ofstream fout("log_CG.txt", std::ios::out | std::ios::app);
            fout << std::setprecision(10);
            fout << std::setw(20) << accu << std::endl;
            fout.close();
        }
        std::cout << std::endl;
    }
    template void eigenvec_CG(const MKL_INT &dim, const MKL_INT &maxit, MKL_INT &m,
                              const csr_mat<double> &mat, const double &E0, double &accu,
                              double v[], double r[], double p[], double pp[]);
    template void eigenvec_CG(const MKL_INT &dim, const MKL_INT &maxit, MKL_INT &m,
                              const csr_mat<std::complex<double>> &mat, const std::complex<double> &E0, double &accu,
                              std::complex<double> v[], std::complex<double> r[],
                              std::complex<double> p[], std::complex<double> pp[]);
    template void eigenvec_CG(const MKL_INT &dim, const MKL_INT &maxit, MKL_INT &m,
                              const model<std::complex<double>> &mat, const std::complex<double> &E0, double &accu,
                              std::complex<double> v[], std::complex<double> r[],
                              std::complex<double> p[], std::complex<double> pp[]);
    
    
    void hess_eigen(const double hessenberg[], const MKL_INT &maxit, const MKL_INT &m,
                    const std::string &order, std::vector<double> &ritz, std::vector<double> &s)
    {
        assert(m > 0 && m < maxit);
        ritz.resize(m);
        s.resize(m*m);
        std::vector<double> b(m);
        std::vector<double> eigenvecs(m*m);
        
        copy(m, hessenberg + maxit, 1, ritz.data(), 1);                          // ritz = a
        copy(m, hessenberg,         1, b.data(),    1);                          // b
        // ritz to be rewritten by eigenvalues in ascending order
        int info = stedc(LAPACK_COL_MAJOR, 'I', m, ritz.data(), b.data() + 1, eigenvecs.data(), m);
        assert(info == 0);
        
        std::vector<std::pair<double, MKL_INT>> eigenvals(m);
        for (MKL_INT j = 0; j < m; j++) {
            eigenvals[j].first = ritz[j];
            eigenvals[j].second = j;
        }
        if (order == "SR" || order == "sr" || order == "SA" || order == "sa") {  // smallest real part
            std::sort(eigenvals.begin(), eigenvals.end(),
                      [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return a.first < b.first; });
        } else if (order == "LR" || order == "lr" || order == "LA" || order == "la") { // largest real part
            std::sort(eigenvals.begin(), eigenvals.end(),
                      [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return b.first < a.first; });
        } else if (order == "SM" || order == "sm") {                             // smallest magnitude
            std::sort(eigenvals.begin(), eigenvals.end(),
                      [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return std::abs(a.first) < std::abs(b.first); });
        } else if (order == "LM" || order == "lm") {                             // largest magnitude
            std::sort(eigenvals.begin(), eigenvals.end(),
                      [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return std::abs(b.first) < std::abs(a.first); });
        }
        for (MKL_INT j = 0; j < m; j++) ritz[j] = eigenvals[j].first;
        for (MKL_INT j = 0; j < m; j++) copy(m, eigenvecs.data() + m * eigenvals[j].second, 1, s.data() + m * j, 1);
    }


    void hess2dense(const double hessenberg[], const MKL_INT &maxit, const MKL_INT &m, std::vector<double> &mat)
    {
        assert(m > 0 && m < maxit);
        mat.assign(m*m, 0.0);
        
        mat[0] = hessenberg[maxit];
        if(m > 1) mat[m] = hessenberg[1];
        for (MKL_INT i = 1; i < m - 1; i++) {
            mat[i +     i * m] = hessenberg[i + maxit];
            mat[i + (i-1) * m] = hessenberg[i];
            mat[i + (i+1) * m] = hessenberg[i + 1];
        }
        if (m > 1) {
            mat[m-1 + (m-1) * m] = hessenberg[maxit + m -1];
            mat[m-1 + (m-2) * m] = hessenberg[m-1];
        }
    }


    // when there is time, re-write this subroutine with bulge-chasing
    template <typename T>
    void perform_shifts(const MKL_INT &dim, const MKL_INT &m, const MKL_INT &np, const double shift[],
                        T v[], double hessenberg[], const MKL_INT &maxit, std::vector<double> &Q)
    {
        lapack_int info;
        const T zero = 0.0;
        const T one = 1.0;
        MKL_INT k = m - np;
        assert(np > 0 && np < m);
        std::vector<double> hess_full(m*m), hess_qr(m*m);
        std::vector<double> tau(m);
        Q.resize(m*m);
        for (MKL_INT j = 0; j < m; j++) {                                            // Q = I
            for (MKL_INT i = 0; i < m; i++) Q[i + j * m] = (i == j ? 1.0 : 0.0);
        }
        hess2dense(hessenberg, maxit, m, hess_full);
        for (MKL_INT j = 0; j < np; j++) {
            //std::cout << "QR shift with theta = " << shift[j] << std::endl;
            hess_qr = hess_full;
            for (MKL_INT i = 0; i < m; i++) hess_qr[i + i * m] -= shift[j];       // H - shift[j] * I
            info = geqrf(LAPACK_COL_MAJOR, m, m, hess_qr.data(), m, tau.data());  // upper triangle of hess_copy represents R, lower + tau represent Q
            assert(info == 0);
            info = ormqr(LAPACK_COL_MAJOR, 'L', 'T', m, m, m, hess_qr.data(), m,  // H_new = Q^T * H
                         tau.data(), hess_full.data(), m);
            assert(info == 0);
            info = ormqr(LAPACK_COL_MAJOR, 'R', 'N', m, m, m, hess_qr.data(), m,  // H_new = Q^T * H * Q
                         tau.data(), hess_full.data(), m);
            assert(info == 0);
            info = ormqr(LAPACK_COL_MAJOR, 'R', 'N', m, m, m, hess_qr.data(), m,  // Q_new = Q_old * Q
                         tau.data(), Q.data(), m);
            assert(info == 0);
        }

        for (MKL_INT j = 0; j < k; j++) hessenberg[j + maxit] = hess_full[j + j * maxit]; // diagonal elements of hessenberg matrix
        for (MKL_INT j = 1; j < k; j++) hessenberg[j] = hess_full[j + (j-1) * maxit];   // subdiagonal
        std::vector<T> v_old(dim * m);
        copy(dim * m, v, 1, v_old.data(), 1);
        std::vector<T> Q_typeT(m*m);
        for (MKL_INT j = 0; j < m*m; j++) Q_typeT[j] = Q[j];
        gemm('n', 'n', dim, m, m, one, v_old.data(), dim, Q_typeT.data(), m, zero, v, dim); // v updated
        scal(dim, hessenberg[m]*Q_typeT[m-1 + m*(k-1)], v+m*dim, 1);
        axpy(dim, static_cast<T>(hess_full[k + (k-1)*maxit]), v+k*dim, 1, v+m*dim, 1);
        hessenberg[m] = nrm2(dim, v+m*dim, 1);                                                  // rnorm updated
        scal(dim, 1.0 / hessenberg[m], v+m*dim, 1);                                             // resid updated




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
    template void perform_shifts(const MKL_INT &dim, const MKL_INT &m, const MKL_INT &np, const double shift[],
                                 double v[], double hessenberg[], const MKL_INT &maxit, std::vector<double> &Q);
    template void perform_shifts(const MKL_INT &dim, const MKL_INT &m, const MKL_INT &np, const double shift[],
                                 std::complex<double> v[], double hessenberg[], const MKL_INT &maxit, std::vector<double> &Q);

    // interface to arpack++
    void call_arpack(const MKL_INT &dim, csr_mat<double> &mat, double v0[],
                     const MKL_INT &nev, const MKL_INT &ncv, const MKL_INT &maxit, MKL_INT &nconv, MKL_INT &niter,
                     const std::string &order, double eigenvals[], double eigenvecs[])
    {
        ARSymStdEig<double, csr_mat<double>>
            prob(dim, nev, &mat, &csr_mat<double>::MultMv, order, ncv, 0.0, 0, v0);
        //prob.Trace();
        std::cout << "ARPACK info:" << std::endl;
        std::cout << "(nev, ncv)    = (" << prob.GetNev() << "," << prob.GetNcv() << ")" << std::endl;
        std::cout << "Max iteration = " << prob.GetMaxit() << " -> ";
        prob.ChangeMaxit(maxit);
        std::cout << prob.GetMaxit() << std::endl;
        std::cout << "Max Mat*Vec   = " << prob.GetMaxit() * (ncv - nev) << std::endl;
        prob.EigenValVectors(eigenvecs, eigenvals);
        nconv = prob.ConvergedEigenvalues();
        niter = prob.GetIter();
    }
    void call_arpack(const MKL_INT &dim, csr_mat<std::complex<double>> &mat, std::complex<double> v0[],
                     const MKL_INT &nev, const MKL_INT &ncv, const MKL_INT &maxit, MKL_INT &nconv, MKL_INT &niter,
                     const std::string &order, double eigenvals[], std::complex<double> eigenvecs[])
    {
        std::complex<double> *eigenvals_copy = new std::complex<double>[nev];
        ARCompStdEig<double, csr_mat<std::complex<double>>> prob(dim, nev, &mat,
                                                                 &csr_mat<std::complex<double>>::MultMv,
                                                                 order, ncv, 0.0, 0, v0);
        //prob.Trace();
        std::cout << "ARPACK info:" << std::endl;
        std::cout << "(nev, ncv)    = (" << prob.GetNev() << "," << prob.GetNcv() << ")" << std::endl;
        std::cout << "Max iteration = " << prob.GetMaxit() << " -> ";
        prob.ChangeMaxit(maxit);
        std::cout << prob.GetMaxit() << std::endl;
        std::cout << "Max Mat*Vec   = " << prob.GetMaxit() * (ncv - nev) << std::endl;
        prob.EigenValVectors(eigenvecs, eigenvals_copy);
        nconv = prob.ConvergedEigenvalues();
        niter = prob.GetIter();
        for (MKL_INT j = 0; j < nconv; j++) {
            assert(std::abs(eigenvals_copy[j].imag()) < lanczos_precision);
            eigenvals[j] = eigenvals_copy[j].real();
        }
        delete [] eigenvals_copy;
    }
    void call_arpack(const MKL_INT &dim, model<std::complex<double>> &mat, std::complex<double> v0[],
                     const MKL_INT &nev, const MKL_INT &ncv, const MKL_INT &maxit, MKL_INT &nconv, MKL_INT &niter,
                     const std::string &order, double eigenvals[], std::complex<double> eigenvecs[])
    {
        std::complex<double> *eigenvals_copy = new std::complex<double>[nev];
        ARCompStdEig<double, model<std::complex<double>>> prob(dim, nev, &mat,
                                                               &model<std::complex<double>>::MultMv,
                                                               order, ncv, 0.0, 0, v0);
        //prob.Trace();
        std::cout << "ARPACK info:" << std::endl;
        std::cout << "(nev, ncv)    = (" << prob.GetNev() << "," << prob.GetNcv() << ")" << std::endl;
        std::cout << "Max iteration = " << prob.GetMaxit() << " -> ";
        prob.ChangeMaxit(maxit);
        std::cout << prob.GetMaxit() << std::endl;
        std::cout << "Max Mat*Vec   = " << prob.GetMaxit() * (ncv - nev) << std::endl;
        prob.EigenValVectors(eigenvecs, eigenvals_copy);
        nconv = prob.ConvergedEigenvalues();
        niter = prob.GetIter();
        for (MKL_INT j = 0; j < nconv; j++) {
            assert(std::abs(eigenvals_copy[j].imag()) < lanczos_precision);
            eigenvals[j] = eigenvals_copy[j].real();
        }
        delete [] eigenvals_copy;
    }

    template <typename T, typename MAT>
    void iram(const MKL_INT &dim, MAT &mat, T v0[], const MKL_INT &nev, const MKL_INT &ncv,
              const MKL_INT &maxit, const std::string &order,
              MKL_INT &nconv, double eigenvals[], T eigenvecs[],
              const bool &use_arpack)
    {
        assert(maxit >= 20);
        MKL_INT np = ncv - nev;
        MKL_INT niter;
        std::string orderC(order);
        std::transform(orderC.begin(), orderC.end(), orderC.begin(), ::toupper);
        
        if (dim <= 30) {                                                                // fall back to full diagonalization
            auto mat_dense = mat.to_dense();
            std::vector<double> eigenvals_all(dim);
            auto info = heevd(LAPACK_COL_MAJOR, 'V', 'U', dim, mat_dense.data(), dim, eigenvals_all.data());
            assert(info == 0);
            nconv = nev < dim ? nev : dim;
            std::vector<std::pair<double, MKL_INT>> eigenvals_copy(dim);
            for (MKL_INT j = 0; j < dim; j++) {
                eigenvals_copy[j].first = eigenvals_all[j];
                eigenvals_copy[j].second = j;
            }
            if (orderC == "SR"|| orderC == "SA") {                               // smallest real part
                std::sort(eigenvals_copy.begin(), eigenvals_copy.end(),
                          [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return a.first < b.first; });
            } else if (orderC == "LR"|| orderC == "LA") {                        // largest real part
                std::sort(eigenvals_copy.begin(), eigenvals_copy.end(),
                          [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return b.first < a.first; });
            } else if (orderC == "SM") {                                         // smallest magnitude
                std::sort(eigenvals_copy.begin(), eigenvals_copy.end(),
                          [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return std::abs(a.first) < std::abs(b.first); });
            } else if (orderC == "LM") {                                         // largest magnitude
                std::sort(eigenvals_copy.begin(), eigenvals_copy.end(),
                          [](const std::pair<double, MKL_INT> &a, const std::pair<double, MKL_INT> &b){ return std::abs(b.first) < std::abs(a.first); });
            } else {
                assert(false);
            }
            for (MKL_INT j = 0; j < nconv; j++)
                eigenvals[j] = eigenvals_copy[j].first;
            for (MKL_INT j = 0; j < nconv; j++) {
                std::cout << "j = " << j << ", E_j = " << eigenvals[j] << std::endl;
            }
            // sort the eigenvecs
            for (MKL_INT j = 0; j < nconv; j++)
                copy(dim, mat_dense.data() + dim * eigenvals_copy[j].second, 1, eigenvecs + dim * j, 1);
        } else if (use_arpack) {
            call_arpack(dim, mat, v0, nev, ncv, maxit, nconv, niter, orderC, eigenvals, eigenvecs);
            assert(nconv > 0);
            std::cout << std::endl << "(nev,ncv,nconv) = (" << nev << "," << ncv << "," << nconv << ")" << std::endl;
            std::cout << "Number of implicit restarts: " << niter << std::endl;
            auto comp = [&orderC](const double &a, const double &b)
            {
                if (orderC == "SR" || orderC == "SA") {
                    return a < b;
                } else if (orderC == "LR" || orderC == "LA") {
                    return b < a;
                } else if (orderC == "SM") {
                    return std::abs(a) < std::abs(b);
                } else {
                    assert(orderC == "LM");
                    return std::abs(b) < std::abs(a);
                }
            };
            // bubble sort the eigenvalues and eigenvecs
            using std::swap;
            for (MKL_INT j = 1; j < nconv; j++) {
                bool sorted = true;
                for (MKL_INT i = 0; i < nconv - j; i++) {
                    if (comp(eigenvals[i+1],eigenvals[i])) {
                        swap(eigenvals[i],eigenvals[i+1]);
                        swap_vec(dim, eigenvecs + i * dim, eigenvecs + (i+1) * dim);
                        sorted = false;
                    }
                }
                if (sorted) break;
            }
            for (MKL_INT j = 0; j < nconv; j++) {
                std::cout << "E_" << j << " = " << eigenvals[j] << std::endl;
            }
            std::cout << "Caution: IRAM may miss a few degenerate eigenstates!" << std::endl;
            std::cout << "(in these cases, try a different set of {nev, ncv} may help finding the missing eigenstates)" << std::endl;
        } else {                                                                       // hand-coded arpack
            MKL_INT m = ncv;
            MKL_INT maxit = m + 1;
            std::vector<T> v(dim*(m+1));
            std::vector<double> hessenberg(maxit*2), ritz(m), s(m*m), Q(m*m);
            double rnorm = nrm2(dim, v0, 1);
            axpy(dim, 1.0/rnorm, v0, 1, v.data(), 1);                                  // v[0] = v0 normalized
            lanczos(0, ncv, maxit, m, dim, mat, v.data(), hessenberg.data(), "iram");
            assert(m == ncv);

            MKL_INT step = 0, step_max=100;
            while (step < step_max) {
                hess_eigen(hessenberg.data(), maxit, m, order, ritz, s);
                perform_shifts(dim, ncv, np, ritz.data()+nev, v.data(), hessenberg.data(), maxit, Q);
                lanczos(nev, np, maxit, m, dim, mat, v.data(), hessenberg.data(), "iram");
                for (MKL_INT j = 0; j < nev; j++) {
                    std::cout << std::setw(16) << ritz[j];
                }
                std::cout << std::endl;
                std::cout << "ritz = " << ritz[0] << "\t" << ritz[0] << std::endl;
                step++;
            }
            nconv = 1;
            eigenvals[0] = ritz[0];
            
            std::cout << "Caution: IRAM may miss a few degenerate eigenstates!" << std::endl;
            std::cout << "(in these cases, try a different set of {nev, ncv} may help finding the missing eigenstates)" << std::endl;
        }
        assert(nconv > 0);


    }

    // Explicit instantiation


    


//    template void iram(const MKL_INT &dim, csr_mat<double> &mat, double v0[],
//                       const MKL_INT &nev, const MKL_INT &ncv, MKL_INT &nconv,
//                       const std::string &order, double eigenvals[], double eigenvecs[],
//                       const bool &use_arpack);
    template void iram(const MKL_INT &dim, csr_mat<std::complex<double>> &mat, std::complex<double> v0[],
                       const MKL_INT &nev, const MKL_INT &ncv,
                       const MKL_INT &maxit, const std::string &order,
                       MKL_INT &nconv, double eigenvals[], std::complex<double> eigenvecs[],
                       const bool &use_arpack);
//    template void iram(const MKL_INT &dim, model<double> &mat, double v0[],
//                       const MKL_INT &nev, const MKL_INT &ncv, MKL_INT &nconv,
//                       const std::string &order, double eigenvals[], double eigenvecs[],
//                       const bool &use_arpack);
    template void iram(const MKL_INT &dim, model<std::complex<double>> &mat, std::complex<double> v0[],
                       const MKL_INT &nev, const MKL_INT &ncv,
                       const MKL_INT &maxit, const std::string &order,
                       MKL_INT &nconv, double eigenvals[], std::complex<double> eigenvecs[],
                       const bool &use_arpack);

}
