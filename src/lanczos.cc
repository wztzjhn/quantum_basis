#include "lanczos.h"

// need further classification:
// 1. provide an option that no intermidiate v stored, only hessenberg returned (probably in a separate routine)
// 2. ask lanczos to restart with a new linearly independent vector when v_m+1 = 0
// 3. need add DGKS re-orthogonalization
template <typename T>
void lanczos(MKL_INT k, MKL_INT np, csr_mat<T> &mat, double &rnorm, T resid[],
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

// Explicit instantiation
template void lanczos(MKL_INT k, MKL_INT np, csr_mat<double> &mat,
                      double &rnorm, double resid[], double v[], double hessenberg[], const MKL_INT &ldh);
template void lanczos(MKL_INT k, MKL_INT np, csr_mat<std::complex<double>> &mat,
                      double &rnorm, std::complex<double> resid[], std::complex<double> v[], double hessenberg[], const MKL_INT &ldh);

template void hess2matform(const double hessenberg[], double mat[], const MKL_INT &m, const MKL_INT &ldh);
template void hess2matform(const double hessenberg[], std::complex<double> mat[], const MKL_INT &m, const MKL_INT &ldh);
