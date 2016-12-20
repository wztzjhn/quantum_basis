#include <iostream>
#include <iomanip>
#include <fstream>
#include "qbasis.h"


// H = J/2 \sum_{<i,j>} (S_i^+ S_j^- + S_i^- S_j^+) + \Delta J \sum_{<i,j>} (S_i^z S_j^z)
//
// S_i^+ = ( 0  1 )     S_i^- = ( 0  0 )      S_i^z = (0.5   0 )
//         ( 0  0 )             ( 1  0 )              (0  -0.5 )

MKL_INT binary_find(const std::vector<qbasis::mbasis_elem> &basis_all, const qbasis::mbasis_elem &val)
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


void pbc(const MKL_INT &Lx, const double &Delta, const double &h, const MKL_INT &step);

int main() {
    MKL_INT Lx;
    std::cout << "Lx: " << std::endl;
    std::cin >> Lx;
    std::cout << "Lx = " << Lx << std::endl;
    double Delta;
    std::cout << "Delta: " << std::endl;
    std::cin >> Delta;
    std::cout << "Delta = " << Delta << std::endl;
    double h;
    std::cout << "h: " <<std::endl;
    std::cin >> h;
    std::cout << "h = " << h << std::endl;
    MKL_INT step = 150;
    
    pbc(Lx, Delta, h, step);
    
}

qbasis::mopr<std::complex<double>> Sx_Q(const MKL_INT &Lx, const double &Q)
{
    std::vector<std::vector<std::complex<double>>> Sx(2,std::vector<std::complex<double>>(2));
    Sx[0][0] = 0.0;
    Sx[0][1] = 0.5;
    Sx[1][0] = 0.5;
    Sx[1][1] = 0.0;
    qbasis::mopr<std::complex<double>> res;
    for (MKL_INT j = 0; j < Lx; j++) {
        qbasis::opr<std::complex<double>> sjx(j,0,false,Sx);
        res += ( (std::exp(std::complex<double>(0.0, -Q*j)) / std::sqrt(static_cast<double>(Lx)) ) * sjx);
    }
    return res;
}

qbasis::mopr<std::complex<double>> Sy_Q(const MKL_INT &Lx, const double &Q)
{
    std::vector<std::vector<std::complex<double>>> Sy(2,std::vector<std::complex<double>>(2));
    Sy[0][0] = 0.0;
    Sy[0][1] = std::complex<double>(0.0, -0.5);
    Sy[1][0] = std::complex<double>(0.0, 0.5);
    Sy[1][1] = 0.0;
    qbasis::mopr<std::complex<double>> res;
    for (MKL_INT j = 0; j < Lx; j++) {
        qbasis::opr<std::complex<double>> sjy(j,0,false,Sy);
        res += ( (std::exp(std::complex<double>(0.0, -Q*j)) / std::sqrt(static_cast<double>(Lx)) ) * sjy);
    }
    return res;
}

qbasis::mopr<std::complex<double>> Sz_Q(const MKL_INT &Lx, const double &Q)
{
    std::vector<std::vector<std::complex<double>>> Sz(2,std::vector<std::complex<double>>(2));
    Sz[0][0] = 0.5;
    Sz[0][1] = 0.0;
    Sz[1][0] = 0.0;
    Sz[1][1] = -0.5;
    qbasis::mopr<std::complex<double>> res;
    for (MKL_INT j = 0; j < Lx; j++) {
        qbasis::opr<std::complex<double>> sjz(j,0,false,Sz);
        res += ( (std::exp(std::complex<double>(0.0, -Q*j)) / std::sqrt(static_cast<double>(Lx)) ) * sjz);
    }
    return res;
}


void pbc(const MKL_INT &Lx, const double &Delta, const double &h, const MKL_INT &step) {
    std::cout << "bench1" << std::endl;
    // define the basic operators: S+, S-, Sz
    std::vector<std::vector<std::complex<double>>> Splus(2,std::vector<std::complex<double>>(2));
    std::vector<std::vector<std::complex<double>>> Sminus(2,std::vector<std::complex<double>>(2));
    std::vector<std::vector<std::complex<double>>> Sx(2,std::vector<std::complex<double>>(2));
    std::vector<std::complex<double>> Sz(2);
    Splus[0][0] = 0.0;
    Splus[0][1] = 1.0;
    Splus[1][0] = 0.0;
    Splus[1][1] = 0.0;
    Sminus[0][0] = 0.0;
    Sminus[0][1] = 0.0;
    Sminus[1][0] = 1.0;
    Sminus[1][1] = 0.0;
    Sx[0][0] = 0.0;
    Sx[0][1] = 0.5;
    Sx[1][0] = 0.5;
    Sx[1][1] = 0.0;
    Sz[0] = 0.5;
    Sz[1] = -0.5;
    
    //std::cout << "bench2" << std::endl;
    // create Hamiltonian for xxz model on a chain
    qbasis::mopr<std::complex<double>> Hamiltonian_xx, Hamiltonian_z;
    for (MKL_INT i = 0; i < Lx; i++) {
        MKL_INT j = (i < Lx-1 ? i+1 : 0);
        qbasis::opr<std::complex<double>> sip(i,0,false,Splus);
        qbasis::opr<std::complex<double>> sim(i,0,false,Sminus);
        qbasis::opr<std::complex<double>> sjp(j,0,false,Splus);
        qbasis::opr<std::complex<double>> sjm(j,0,false,Sminus);
        qbasis::opr<std::complex<double>> siz(i,0,false,Sz);
        qbasis::opr<std::complex<double>> sjz(j,0,false,Sz);
        qbasis::opr<std::complex<double>> six(i,0,false,Sx);
//        std::cout << "bench2.1" << std::endl;
//        std::cout << "i,j: " << i << ", " << j << std::endl;
//        auto temp1 = sip * sjm;
//        std::cout << "bench2.2" << std::endl;
//        auto temp2 = sim * sjp;
//        std::cout << "bench2.3" << std::endl;
//        auto temp3 = temp1 + temp2;
//        std::cout << "bench2.4" << std::endl;
//        //temp3 *= std::complex<double>(0.5,0.0);
//        //temp3 = std::complex<double>(0.5,0.0) * temp3;
//        std::cout << "bench2.5" << std::endl;
//        Hamiltonian_xx += temp3;
        Hamiltonian_xx += std::complex<double>(0.5,0.0) * (sip * sjm + sim * sjp);
        if (i % 2 == 0) {
            Hamiltonian_xx  += std::complex<double>(-h, 0.0) * six;
        } else {
            Hamiltonian_xx  += std::complex<double>( h, 0.0) * six;
        }
        //std::cout << "bench2.6" << std::endl;
        Hamiltonian_z  += std::complex<double>(Delta,0.0) * (siz * sjz);
        
        
        //std::cout << "bench2.7" << std::endl;
    }
    
    //std::cout << "bench3" << std::endl;
    // create the basis, without any symmetry, this part later can be parallelized in the library
    MKL_INT dim_all = qbasis::int_pow(2, Lx);
    std::cout << "dim_all = " << dim_all << std::endl;
    std::vector<qbasis::mbasis_elem> basis_all(dim_all,qbasis::mbasis_elem(Lx,{"spin-1/2"}));
    basis_all[0].reset();
    for (MKL_INT j = 1; j < dim_all; j++) {
        basis_all[j] = basis_all[j-1];
        basis_all[j].increment();
    }
    
    // check if basis sorted
    std::cout << "checking if basis sorted..." << std::endl;
    bool sorted = true;
    for (MKL_INT j = 0; j < dim_all - 1; j++) {
        assert(basis_all[j] != basis_all[j+1]);
        if (basis_all[j+1] < basis_all[j]) {
            sorted = false;
            break;
        }
    }
    if (! sorted) {
        std::cout << "sorting basis..." << std::endl;
        std::sort(basis_all.begin(), basis_all.end());
        std::cout << "sorting finished." << std::endl << std::endl;
    } else {
        std::cout << "yes it is already sorted." << std::endl << std::endl;
    }
    
    // generating Hamiltonian matrix, remember to only only upper triangle later
    // the generation process can be parallelized
    std::cout << "Generating Hamiltonian..." << std::endl;
    qbasis::lil_mat<std::complex<double>> matrix_lil(dim_all, true);
    for (MKL_INT j = 0; j < dim_all; j++) {
        if (j%1000 == 0) std::cout << "j = " << j << std::endl;
        // diagonal part: Sz * Sz
        for (MKL_INT cnt = 0; cnt < Hamiltonian_z.size(); cnt++) {
            matrix_lil.add(j, j, basis_all[j].diagonal_operator(Hamiltonian_z[cnt]));
        }
        // non-diagonal part
        qbasis::wavefunction<std::complex<double>> intermediate_state = Hamiltonian_xx * basis_all[j];
        //std::cout << "intermediate state size = " << intermediate_state.size() << std::endl;
        for (MKL_INT cnt = 0; cnt < intermediate_state.size(); cnt++) {
            auto &ele_new = intermediate_state[cnt];
            MKL_INT i = binary_find(basis_all, ele_new.first);
            if (i <= j) {
                matrix_lil.add(i, j, ele_new.second);
            }
        }
    }
    qbasis::csr_mat<std::complex<double>> matrix_csr(matrix_lil);
    matrix_lil.destroy();
    std::cout << "Hamiltonian generated." << std::endl << std::endl;
    //matrix_csr.prt();
    
    
    // run Lanczos to obtain eigenvals
    MKL_INT nev = 5, ncv = 15;
    MKL_INT nconv;
    std::vector<std::complex<double>> v0(dim_all, 1.0);
    std::vector<double> eigenvals(nev);
    std::vector<std::complex<double>> eigenvecs(dim_all * nev);
    qbasis::iram(matrix_csr, v0.data(), nev, ncv, nconv, "lr", eigenvals.data(), eigenvecs.data());
    std::cout << "Emax  = " << eigenvals[0] << std::endl;
    double Emax = eigenvals[0];
    for (MKL_INT j = 0; j < dim_all; j++) v0[j] = 1.0;
    qbasis::iram(matrix_csr, v0.data(), nev, ncv, nconv, "sr", eigenvals.data(), eigenvecs.data());
    std::cout << "E0  = " << eigenvals[0] << std::endl;
    std::cout << "Gap = " << eigenvals[1] - eigenvals[0] << std::endl;
    assert(eigenvals[1] - eigenvals[0] > qbasis::lanczos_precision); // first only consider when the system is gapped
//    std::cout << "Eigenvector: " << std::endl;
//    for (MKL_INT j = 0; j < dim_all; j++) {
//        std::cout << eigenvecs[j] << std::endl;
//    }
    std::cout << std::endl;
    

    std::string output_name =   "L"+std::to_string(Lx)
                              + "_Delta_"+std::to_string(Delta).substr(0,4)
                              + "_h_"+std::to_string(h).substr(0,4)
                              + ".dat";
    std::ofstream fout(output_name, std::ios::out);
    fout << std::scientific << std::right;
    fout << "L\t" << Lx << std::endl;
    fout << "Delta\t" << Delta << std::endl;
    fout << "h\t" << h << std::endl;
    fout << "Emax\t" << Emax << std::endl;
    fout << "E0\t" << eigenvals[0] << std::endl;
    fout << "Gap\t" << eigenvals[1] - eigenvals[0] << std::endl << std::endl;
    
    
    // later change Q into a loop
    for (MKL_INT m = 0; m <= Lx; m++) {
        double Q = 2.0 * qbasis::pi * m / static_cast<double>(Lx);
        auto SxQ = Sx_Q(Lx, Q);
        auto SyQ = Sy_Q(Lx, Q);
        auto SzQ = Sz_Q(Lx, Q);
        //SxQ.prt();
        
        // obtain the new starting vector |phi_0_x> = S^x(-q) |varphi_0>
        std::cout << "Generating S^x(-q) | varphi_0 >, with q = " << Q << std::endl;
        std::vector<std::complex<double>> phi0_x(dim_all, 0.0);
        for (MKL_INT j = 0; j < dim_all; j++) {
            if (std::abs(eigenvecs[j]) < qbasis::lanczos_precision) continue;
            auto intermediate_state = SxQ * basis_all[j];
            for (MKL_INT cnt = 0; cnt < intermediate_state.size(); cnt++) {
                auto &ele = intermediate_state[cnt];
                MKL_INT i = binary_find(basis_all, ele.first);
                phi0_x[i] += eigenvecs[j] * ele.second;
            }
            //        std::cout << "----state " << j << " -------" << std::endl;
            //        for (MKL_INT jj = 0; jj < dim_all; jj++) {
            //            std::cout << "phi0_x[" << jj << "] = " << phi0_x[jj] << std::endl;
            //        }
        }
        double phi0_x_nrm2 = qbasis::nrm2(dim_all, phi0_x.data(), 1);
        double phi0_x_nrm2_copy;
        if (phi0_x_nrm2 > qbasis::lanczos_precision) {
            std::cout << "phi_x_nrm2 = " << phi0_x_nrm2 << std::endl;
            qbasis::scal(dim_all, 1.0/phi0_x_nrm2, phi0_x.data(), 1);
            phi0_x_nrm2_copy = qbasis::nrm2(dim_all, phi0_x.data(), 1);
        }
        
        std::cout << "Generating S^y(-q) | varphi_0 >, with q = " << Q << std::endl;
        std::vector<std::complex<double>> phi0_y(dim_all, 0.0);
        for (MKL_INT j = 0; j < dim_all; j++) {
            if (std::abs(eigenvecs[j]) < qbasis::lanczos_precision) continue;
            auto intermediate_state = SyQ * basis_all[j];
            for (MKL_INT cnt = 0; cnt < intermediate_state.size(); cnt++) {
                auto &ele = intermediate_state[cnt];
                MKL_INT i = binary_find(basis_all, ele.first);
                phi0_y[i] += eigenvecs[j] * ele.second;
            }
        }
        double phi0_y_nrm2 = qbasis::nrm2(dim_all, phi0_y.data(), 1);
        double phi0_y_nrm2_copy;
        if (phi0_y_nrm2 > qbasis::lanczos_precision) {
            std::cout << "phi_y_nrm2 = " << phi0_y_nrm2 << std::endl;
            qbasis::scal(dim_all, 1.0/phi0_y_nrm2, phi0_y.data(), 1);
            phi0_y_nrm2_copy = qbasis::nrm2(dim_all, phi0_y.data(), 1);
        }
        
        std::cout << "Generating S^z(-q) | varphi_0 >, with q = " << Q << std::endl;
        std::vector<std::complex<double>> phi0_z(dim_all, 0.0);
        for (MKL_INT j = 0; j < dim_all; j++) {
            if (std::abs(eigenvecs[j]) < qbasis::lanczos_precision) continue;
            auto intermediate_state = SzQ * basis_all[j];
            for (MKL_INT cnt = 0; cnt < intermediate_state.size(); cnt++) {
                auto &ele = intermediate_state[cnt];
                MKL_INT i = binary_find(basis_all, ele.first);
                phi0_z[i] += eigenvecs[j] * ele.second;
            }
        }
        double phi0_z_nrm2 = qbasis::nrm2(dim_all, phi0_z.data(), 1);
        double phi0_z_nrm2_copy;
        if (phi0_z_nrm2 > qbasis::lanczos_precision) {
            std::cout << "phi_z_nrm2 = " << phi0_z_nrm2 << std::endl;
            qbasis::scal(dim_all, 1.0/phi0_z_nrm2, phi0_z.data(), 1);
            phi0_z_nrm2_copy = qbasis::nrm2(dim_all, phi0_z.data(), 1);
        }
        
        std::cout << "Running continued fraction with " << step << " steps" << std::endl;
        std::vector<std::complex<double>> v(dim_all * 3);
        std::vector<double> hessenberg_x(step * 2, 0.0);
        std::vector<double> hessenberg_y(step * 2, 0.0);
        std::vector<double> hessenberg_z(step * 2, 0.0);
        fout << "Q\t" << Q << std::endl;
        fout << "nrm2_x\t" << phi0_x_nrm2 << std::endl;
        if (phi0_x_nrm2 > qbasis::lanczos_precision) {
            qbasis::lanczos(0, step, matrix_csr, phi0_x_nrm2_copy, phi0_x.data(), v.data(), hessenberg_x.data(), step, false);
            fout << "b_x\t";
            for (MKL_INT i = 0; i < step; i++) {
                fout << std::setw(30) << std::setprecision(18) << hessenberg_x[i];
            }
            fout << std::endl;
            fout << "a_x\t";
            for (MKL_INT i = step; i < 2 * step; i++) {
                fout << std::setw(30) << std::setprecision(18) << hessenberg_x[i];
            }
            fout << std::endl;
        } else {
            fout << "b_x\t" << std::endl;
            fout << "a_x\t" << std::endl;
        }
        fout << "nrm2_y\t" << phi0_y_nrm2 << std::endl;
        if (phi0_y_nrm2 > qbasis::lanczos_precision) {
            qbasis::lanczos(0, step, matrix_csr, phi0_y_nrm2_copy, phi0_y.data(), v.data(), hessenberg_y.data(), step, false);
            fout << "b_y\t";
            for (MKL_INT i = 0; i < step; i++) {
                fout << std::setw(30) << std::setprecision(18) << hessenberg_y[i];
            }
            fout << std::endl;
            fout << "a_y\t";
            for (MKL_INT i = step; i < 2 * step; i++) {
                fout << std::setw(30) << std::setprecision(18) << hessenberg_y[i];
            }
            fout << std::endl;
        } else {
            fout << "b_y\t" << std::endl;
            fout << "a_y\t" << std::endl;
        }
        fout << "nrm2_z\t" << phi0_z_nrm2 << std::endl;
        if (phi0_z_nrm2 > qbasis::lanczos_precision) {
            qbasis::lanczos(0, step, matrix_csr, phi0_z_nrm2_copy, phi0_z.data(), v.data(), hessenberg_z.data(), step, false);
            fout << "b_z\t";
            for (MKL_INT i = 0; i < step; i++) {
                fout << std::setw(30) << std::setprecision(18) << hessenberg_z[i];
            }
            fout << std::endl;
            fout << "a_z\t";
            for (MKL_INT i = step; i < 2 * step; i++) {
                fout << std::setw(30) << std::setprecision(18) << hessenberg_z[i];
            }
            fout << std::endl;
        } else {
            fout << "b_z\t" << std::endl;
            fout << "a_z\t" << std::endl;
        }
        fout << std::endl;
    }
    
    
    
    fout.close();
    
    
    
    
    
    
    
    std::cout << "------" << std::endl;
    
}
