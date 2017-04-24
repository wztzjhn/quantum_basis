

void test_dimer() {
    qbasis::mbasis_elem basis_uu(2, {"spin-1/2"});
    qbasis::mbasis_elem basis_dd(2, {"spin-1/2"});
    basis_dd.siteWrite(0, 0, 1);
    basis_dd.siteWrite(1, 0, 1);
    qbasis::mbasis_elem basis_ud(2, {"spin-1/2"});
    basis_ud.siteWrite(1, 0, 1);
    qbasis::mbasis_elem basis_du(2, {"spin-1/2"});
    basis_du.siteWrite(0, 0, 1);
    // define the basic operators: S+, S-, Sz
    std::vector<std::vector<std::complex<double>>> Sx(2,std::vector<std::complex<double>>(2));
    std::vector<std::vector<std::complex<double>>> Sy(2,std::vector<std::complex<double>>(2));
    std::vector<std::complex<double>> Sz(2);
    Sx[0][0] = 0.0;
    Sx[0][1] = 0.5;
    Sx[1][0] = 0.5;
    Sx[1][1] = 0.0;
    Sy[0][0] = 0.0;
    Sy[0][1] = std::complex<double>(0.0, -0.5);
    Sy[1][0] = std::complex<double>(0.0,  0.5);
    Sy[1][1] = 0.0;
    Sz[0] = 0.5;
    Sz[1] = -0.5;
    qbasis::opr<std::complex<double>> s0x(0,0,false,Sx);
    qbasis::opr<std::complex<double>> s0y(0,0,false,Sy);
    qbasis::opr<std::complex<double>> s0z(0,0,false,Sz);
    qbasis::opr<std::complex<double>> s1x(1,0,false,Sx);
    qbasis::opr<std::complex<double>> s1y(1,0,false,Sy);
    qbasis::opr<std::complex<double>> s1z(1,0,false,Sz);
    auto Ham = s0x * s1x + s0y * s1y + s0z * s1z;
    
    qbasis::wavefunction<std::complex<double>> singlet(  std::complex<double>( 1.0/sqrt(2.0)) * basis_ud
                                                       + std::complex<double>(-1.0/sqrt(2.0)) * basis_du);
    qbasis::wavefunction<std::complex<double>> triplet_z(  std::complex<double>(1.0/sqrt(2.0)) * basis_ud
                                                         + std::complex<double>(1.0/sqrt(2.0)) * basis_du);
    qbasis::wavefunction<std::complex<double>> triplet_u(basis_uu);
    qbasis::wavefunction<std::complex<double>> triplet_d(basis_dd);
    qbasis::wavefunction<std::complex<double>> triplet_x(  std::complex<double>(-1.0/sqrt(2.0)) * basis_uu
                                                         + std::complex<double>( 1.0/sqrt(2.0)) * basis_dd);
    qbasis::wavefunction<std::complex<double>> triplet_y(  std::complex<double>(0.0, 1.0/sqrt(2.0)) * basis_uu
                                                         + std::complex<double>(0.0, 1.0/sqrt(2.0)) * basis_dd);
    auto res0 = s0x * singlet;
    auto res1 = s0x * triplet_x;
    auto res2 = s0x * triplet_y;
    auto res3 = s0x * triplet_z;
    res0.prt();
    std::cout << std::endl << std::endl;
    
}

void test_trimer()
{
    qbasis::lattice lattice("chain",std::vector<MKL_INT>{3},std::vector<std::string>{"obc"});
    
    std::vector<std::vector<std::complex<double>>> Splus(2,std::vector<std::complex<double>>(2));
    std::vector<std::vector<std::complex<double>>> Sminus(2,std::vector<std::complex<double>>(2));
    std::vector<std::vector<std::complex<double>>> Sx(2,std::vector<std::complex<double>>(2));
    std::vector<std::vector<std::complex<double>>> Sy(2,std::vector<std::complex<double>>(2));
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
    Sy[0][0] = 0.0;
    Sy[0][1] = std::complex<double>(0.0,-0.5);
    Sy[1][0] = std::complex<double>(0.0,0.5);
    Sy[1][1] = 0.0;
    Sz[0] = 0.5;
    Sz[1] = -0.5;
    
    qbasis::model<std::complex<double>> trimer;
    qbasis::mopr<std::complex<double>> Sz_total;
    //trimer.dim_all = qbasis::int_pow(2, 3);
    //trimer.basis_all = std::vector<qbasis::mbasis_elem>(trimer.dim_all,qbasis::mbasis_elem(3,{"spin-1/2"}));
    
    qbasis::opr<std::complex<double>> s0x(0,0,false,Sx);
    qbasis::opr<std::complex<double>> s1x(1,0,false,Sx);
    qbasis::opr<std::complex<double>> s2x(2,0,false,Sx);
    qbasis::opr<std::complex<double>> s0y(0,0,false,Sy);
    qbasis::opr<std::complex<double>> s1y(1,0,false,Sy);
    qbasis::opr<std::complex<double>> s2y(2,0,false,Sy);
    qbasis::opr<std::complex<double>> s0z(0,0,false,Sz);
    qbasis::opr<std::complex<double>> s1z(1,0,false,Sz);
    qbasis::opr<std::complex<double>> s2z(2,0,false,Sz);
    
    trimer.add_offdiagonal_Ham(s0x * s1x + s0y * s1y);
    trimer.add_diagonal_Ham(s0z * s1z);
    trimer.add_offdiagonal_Ham(s1x * s2x + s1y * s2y);
    trimer.add_diagonal_Ham(s1z * s2z);
    
    Sz_total = (s0z + s1z + s2z);
    
    trimer.enumerate_basis_full(3, {"spin-1/2"}, {Sz_total}, {0.5});
    std::cout << "dim_all = " << trimer.dim_full << std::endl;
    
    for (MKL_INT j = 0; j < trimer.dim_full; j++) {
        std::cout << " j = " << j << std::endl;
        trimer.basis_full[j].prt_nonzero(); std::cout << std::endl;
    }
    
    trimer.generate_Ham_sparse_full(true);
    
    trimer.HamMat_csr_full.prt();
    
    trimer.locate_E0_full(2,4);
    for (MKL_INT j = 0; j < trimer.dim_full; j++) {
        std::cout << "j= "  << j << ", mod=" << std::abs(trimer.eigenvecs_full[j]) << ", allp= " << trimer.eigenvecs_full[j] << std::endl;
    }
    
    
}


// test t-J model on square lattice
void test_tJ()
{
    
    std::cout << "testing tJ model." << std::endl;
    MKL_INT lx = 2, ly = 3;
    qbasis::lattice lattice("square",std::vector<MKL_INT>{lx,ly},std::vector<std::string>{"pbc", "pbc"});
    MKL_INT total_up = lx * ly / 2;
    MKL_INT total_dn = lx * ly / 2;
    
    double J = 1.9;
    
    auto c_up  = std::vector<std::vector<std::complex<double>>>(3,std::vector<std::complex<double>>(3, 0.0));
    auto c_dn  = std::vector<std::vector<std::complex<double>>>(3,std::vector<std::complex<double>>(3, 0.0));
    c_up[0][1] = std::complex<double>(1.0,0.0);
    c_dn[0][2] = std::complex<double>(1.0,0.0);
    
    qbasis::mopr<std::complex<double>> N_up, N_dn;
    qbasis::model<std::complex<double>> tJ;

    for (MKL_INT m = 0; m < lattice.Lx(); m++) {
        for (MKL_INT n = 0; n < lattice.Ly(); n++) {
            MKL_INT site_i, site_j;
            lattice.coor2site(std::vector<MKL_INT>{m,n}, 0, site_i);
            auto c_up_i = qbasis::opr<std::complex<double>>(site_i,0,true,c_up);
            auto c_dn_i = qbasis::opr<std::complex<double>>(site_i,0,true,c_dn);
            auto c_up_dg_i = c_up_i; c_up_dg_i.dagger();
            auto c_dn_dg_i = c_dn_i; c_dn_dg_i.dagger();
            auto Sp_i = c_up_dg_i * c_dn_i;
            auto Sm_i = c_dn_dg_i * c_up_i;
            auto Sz_i = std::complex<double>(0.5,0.0) * (c_up_dg_i * c_up_i - c_dn_dg_i * c_dn_i);
            auto n_i  = c_up_dg_i * c_up_i + c_dn_dg_i * c_dn_i;
            
            N_up += (c_up_dg_i * c_up_i);
            N_dn += (c_dn_dg_i * c_dn_i);
            
            // right neighbor
            lattice.coor2site(std::vector<MKL_INT>{m+1,n}, 0, site_j);
            auto c_up_j = qbasis::opr<std::complex<double>>(site_j,0,true,c_up);
            auto c_dn_j = qbasis::opr<std::complex<double>>(site_j,0,true,c_dn);
            auto c_up_dg_j = c_up_j; c_up_dg_j.dagger();
            auto c_dn_dg_j = c_dn_j; c_dn_dg_j.dagger();
            auto Sp_j = c_up_dg_j * c_dn_j;
            auto Sm_j = c_dn_dg_j * c_up_j;
            auto Sz_j = std::complex<double>(0.5,0.0) * (c_up_dg_j * c_up_j - c_dn_dg_j * c_dn_j);
            auto n_j  = c_up_dg_j * c_up_j + c_dn_dg_j * c_dn_j;
            
            tJ.add_offdiagonal_Ham(std::complex<double>(-1.0,0.0) * ( c_up_dg_i * c_up_j ));
            tJ.add_offdiagonal_Ham(std::complex<double>(-1.0,0.0) * ( c_up_dg_j * c_up_i ));
            tJ.add_offdiagonal_Ham(std::complex<double>(-1.0,0.0) * ( c_dn_dg_i * c_dn_j ));
            tJ.add_offdiagonal_Ham(std::complex<double>(-1.0,0.0) * ( c_dn_dg_j * c_dn_i ));
            tJ.add_offdiagonal_Ham(std::complex<double>(0.5 * J,0.0) * (Sp_i * Sm_j));
            tJ.add_offdiagonal_Ham(std::complex<double>(0.5 * J,0.0) * (Sm_i * Sp_j));
            tJ.add_diagonal_Ham(std::complex<double>(J,0.0) * Sz_i * Sz_j);
            tJ.add_diagonal_Ham(std::complex<double>(-0.25 * J,0.0) * n_i * n_j);
            
            // top neighbor
            lattice.coor2site(std::vector<MKL_INT>{m,n+1}, 0, site_j);
            c_up_j = qbasis::opr<std::complex<double>>(site_j,0,true,c_up);
            c_dn_j = qbasis::opr<std::complex<double>>(site_j,0,true,c_dn);
            c_up_dg_j = c_up_j; c_up_dg_j.dagger();
            c_dn_dg_j = c_dn_j; c_dn_dg_j.dagger();
            Sp_j = c_up_dg_j * c_dn_j;
            Sm_j = c_dn_dg_j * c_up_j;
            Sz_j = std::complex<double>(0.5,0.0) * (c_up_dg_j * c_up_j - c_dn_dg_j * c_dn_j);
            n_j  = c_up_dg_j * c_up_j + c_dn_dg_j * c_dn_j;
            
            tJ.add_offdiagonal_Ham(std::complex<double>(-1.0,0.0) * ( c_up_dg_i * c_up_j ));
            tJ.add_offdiagonal_Ham(std::complex<double>(-1.0,0.0) * ( c_up_dg_j * c_up_i ));
            tJ.add_offdiagonal_Ham(std::complex<double>(-1.0,0.0) * ( c_dn_dg_i * c_dn_j ));
            tJ.add_offdiagonal_Ham(std::complex<double>(-1.0,0.0) * ( c_dn_dg_j * c_dn_i ));
            tJ.add_offdiagonal_Ham(std::complex<double>(0.5 * J,0.0) * (Sp_i * Sm_j));
            tJ.add_offdiagonal_Ham(std::complex<double>(0.5 * J,0.0) * (Sm_i * Sp_j));
            tJ.add_diagonal_Ham(std::complex<double>(J,0.0) * Sz_i * Sz_j);
            tJ.add_diagonal_Ham(std::complex<double>(-0.25 * J,0.0) * n_i * n_j);
        }
    }
    
    tJ.enumerate_basis_full(lattice.total_sites(), {"tJ"}, {N_up, N_dn}, {static_cast<double>(total_up),static_cast<double>(total_dn)});
    std::cout << "dim_all = " << tJ.dim_full << std::endl;
    
    // generating Hamiltonian matrix
    //tJ.generate_Ham_sparse_full(false);
    //std::cout << std::endl;
    
    //tJ.locate_E0_full();
    //std::cout << std::endl;
    
    
    MKL_INT i = 0, j = 0;
//    for (MKL_INT i = 0; i < lattice.Lx(); i++) {
//        for (MKL_INT j = 0; j < lattice.Ly(); j++) {
            tJ.basis_init_repr(std::vector<MKL_INT>{i,j}, lattice);
            tJ.generate_Ham_sparse_repr();
            std::cout << std::endl;
            
            tJ.locate_E0_repr();
            std::cout << std::endl;
//        }
//    }
    
    
    
}
