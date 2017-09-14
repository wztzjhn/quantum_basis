#include <iostream>
#include <iomanip>
#include <fstream>
#include "qbasis.h"

void test_basis();
void test_basis2();
void test_basis3();
void test_basis4();
void test_basis5();

void test_operator();

void test_lanczos_memoAll();

void test_lattice();
void test_lattice2();

void test_cfraction();
void test_dotc();
void test_bubble();




int main(){
    
    std::cout << qbasis::date_and_time() << std::endl;
    
    MKL_INT m = 1e4;
    
    fs::create_directory(fs::path("out_Qckpt"));
    
    for (auto &p : fs::directory_iterator("out_Qckpt")) {
        auto y = p.path();
        std::cout << "files: " << y.filename().string() << std::endl;
    }
    
    std::vector<std::complex<double>> xx(m), yy(m);
    for (int j = 0; j < m; j++) {
        xx[j] = std::complex<double>{j *0.5, j*0.5};
        //xx[j] = j * 0.5;
    }
    qbasis::vec_disk_write("test.bin", m, xx.data());
    
    std::cout << "ll" << std::endl;
    
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    auto info = qbasis::vec_disk_read("test.bin", m, yy.data());
    for (int j = 0; j < 16; j++) {
        std::cout << "yy[" << j << "] = " << yy[j] << std::endl;
    }
    std::cout << "info = " << info << std::endl;
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s." << std::endl;
    
    
    
    
    test_basis();
    
    test_basis2();
    
    test_basis3();
    
    test_basis4();
    
    //test_basis5();
    
    
    test_operator();
    /*
    test_lanczos_memoAll();
    
    test_lattice();
    
    test_lattice2();
    
    //test_dotc();
    
    //test_bubble();
    
    //test_cfraction();
    */
    
    
}





void test_dotc() {
    std::vector<std::complex<double>> x(2), y(2);
    x[0] = std::complex<double>(1.0,2.0);
    x[1] = std::complex<double>(2.0,3.0);
    y[0] = std::complex<double>(2.0,-2.0);
    y[1] = std::complex<double>(-5.0,7.0);
    std::complex<double> z = qbasis::dotc(2, x.data(), 1, y.data(), 1);
    std::cout << "z = " << z << std::endl;
    assert(std::abs(z - std::complex<double>(9.0,23.0)) < qbasis::lanczos_precision);
}

void test_cfraction() {
    std::vector<double> a(1000,2.0), b(1000,1.0);
    a[0] = 1.0;
    std::cout << "len =   5, sqrt(2) = " << qbasis::continued_fraction(a.data(), b.data(), 5) << std::endl;
    std::cout << "len =  10, sqrt(2) = " << qbasis::continued_fraction(a.data(), b.data(), 10) << std::endl;
    std::cout << "len =  50, sqrt(2) = " << qbasis::continued_fraction(a.data(), b.data(), 50) << std::endl;
    
    a[0] = 3.0;
    for (MKL_INT j = 1; j < a.size(); j++) {
        a[j] = 6.0;
        b[j] = (2.0 * j - 1.0) * (2.0 * j - 1.0);
    }
    std::cout << "len =   5, pi = " << qbasis::continued_fraction(a.data(), b.data(), 5) << std::endl;
    std::cout << "len =  10, pi = " << qbasis::continued_fraction(a.data(), b.data(), 10) << std::endl;
    std::cout << "len =  50, pi = " << qbasis::continued_fraction(a.data(), b.data(), 50) << std::endl;
}


void test_bubble() {
    std::vector<MKL_INT> val{3,1,10,2,5,12,9,-3};
    auto cnt = qbasis::bubble_sort(val, 0, 8);
    std::cout << "cnt = " << cnt << std::endl;
    std::cout << "vals: " << std::endl;
    for (MKL_INT j = 0; j < val.size(); j++) {
        std::cout << val[j] << "  ";
    }
    std::cout << std::endl;
}



