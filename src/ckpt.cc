#include <cassert>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <regex>
#include "qbasis.h"

namespace fs = std::filesystem;

namespace qbasis {
    bool enable_ckpt = false;

    template <typename T>
    void ckpt_lanczos_init(MKL_INT &k, const MKL_INT &maxit, const MKL_INT &dim,
                           int &cnt_accuE0, double &accuracy, double &theta0_prev, double &theta1_prev,
                           T v[], double hessenberg[], const std::string &purpose)
    {
        assert(k >= 0 && dim > 0 && maxit > 0);
        if (! enable_ckpt) return;
        fs::path outdir("out_Qckpt");
        if (fs::exists(outdir)) {
            if (! fs::is_directory(outdir)) {
                fs::remove_all(outdir);
                fs::create_directory(outdir);
            }
        } else {
            fs::create_directory(outdir);
        }

        auto &npos = std::string::npos;
        std::ofstream fout("out_Qckpt/log_Lanczos_ckpt.txt", std::ios::out | std::ios::app);
        fout << std::endl << "Log start: " << date_and_time() << std::endl;
        fout << "Initializing Lanczos, purpose = " << purpose << std::endl;
        fout << "Input step = " << k << std::endl;
        fout << "Current files on disk: " << std::endl;
        for (auto &p : fs::directory_iterator("out_Qckpt")) fout << p << std::endl;
        auto size_Qckpt1 = sizeof(MKL_INT);
        bool updating = (fs::exists(fs::path("out_Qckpt/lczs_updt.Qckpt1")) &&
                         fs::file_size(fs::path("out_Qckpt/lczs_updt.Qckpt1")) == size_Qckpt1);


        fout << "Resuming from an interrupted update? " << updating << std::endl;
        if (updating) {
            fout << "Cleaning up junks from last update." << std::endl;
            bool finished = fs::exists(fs::path("out_Qckpt/lczs_updt.Qckpt2"));  // if new data finished writing
            std::ifstream ftemp("out_Qckpt/lczs_updt.Qckpt1", std::ios::in | std::ios::binary);
            ftemp.read(reinterpret_cast<char*>(&k), sizeof(MKL_INT));
            ftemp.close();
            if (finished) {                                                      // then continue cleanup
                fout << "New data finished writing while updating Lanczos step k = " << k << std::endl;
                assert(fs::exists(fs::path("out_Qckpt/lanczosV" + std::to_string(k) + ".dat")));
                if (fs::exists(fs::path("out_Qckpt/HessenbergA.dat.new"))) {
                    fs::remove(fs::path("out_Qckpt/HessenbergA.dat"));
                    fs::rename(fs::path("out_Qckpt/HessenbergA.dat.new"), fs::path("out_Qckpt/HessenbergA.dat"));
                }
                if (fs::exists(fs::path("out_Qckpt/HessenbergB.dat.new"))) {
                    fs::remove(fs::path("out_Qckpt/HessenbergB.dat"));
                    fs::rename(fs::path("out_Qckpt/HessenbergB.dat.new"), fs::path("out_Qckpt/HessenbergB.dat"));
                }
                if (fs::exists(fs::path("out_Qckpt/lanczosY0.dat.new"))) {
                    fs::remove(fs::path("out_Qckpt/lanczosY0.dat"));
                    fs::rename(fs::path("out_Qckpt/lanczosY0.dat.new"), fs::path("out_Qckpt/lanczosY0.dat"));
                }
                if (fs::exists(fs::path("out_Qckpt/lanczosY1.dat.new"))) {
                    fs::remove(fs::path("out_Qckpt/lanczosY1.dat"));
                    fs::rename(fs::path("out_Qckpt/lanczosY1.dat.new"), fs::path("out_Qckpt/lanczosY1.dat"));
                }
                if (fs::exists(fs::path("out_Qckpt/lczs_mlns.dat.new"))) {
                    fs::remove(fs::path("out_Qckpt/lczs_mlns.dat"));
                    fs::rename(fs::path("out_Qckpt/lczs_mlns.dat.new"), fs::path("out_Qckpt/lczs_mlns.dat"));
                }
                if (purpose != "iram") {
                    for (MKL_INT kk = 0; kk < k-1; kk++)
                        fs::remove(fs::path("out_Qckpt/lanczosV" + std::to_string(kk) + ".dat"));
                }
                fs::remove(fs::path("out_Qckpt/lczs_updt.Qckpt1"));
                fs::remove(fs::path("out_Qckpt/lczs_updt.Qckpt2"));
            } else {                                                             // rewind
                fout << "New data unfinished writing while updating Lanczos step k = " << k << std::endl;
                fout << "Rewinding one step." << std::endl;
                k--;
                assert(fs::exists(fs::path("out_Qckpt/lanczosV" + std::to_string(k) + ".dat")));
                fs::remove(fs::path("out_Qckpt/lczs_mlns.dat.new"));
                fs::remove(fs::path("out_Qckpt/lanczosY1.dat.new"));
                fs::remove(fs::path("out_Qckpt/lanczosY0.dat.new"));
                MKL_INT kk = k + 1;
                while (fs::exists(fs::path("out_Qckpt/lanczosV" + std::to_string(kk) + ".dat"))) {
                    fs::remove(fs::path("out_Qckpt/lanczosV" + std::to_string(kk) + ".dat"));
                    kk++;
                }
                fs::remove(fs::path("out_Qckpt/HessenbergB.dat.new"));
                fs::remove(fs::path("out_Qckpt/HessenbergA.dat.new"));
                fs::remove(fs::path("out_Qckpt/lczs_updt.Qckpt1"));
            }
        } else {
            fs::remove(fs::path("lczs_updt.Qckpt1"));
            fs::remove(fs::path("lczs_updt.Qckpt2"));
            MKL_INT k_bgn = 0;
            while (k_bgn < maxit && ! fs::exists(fs::path("out_Qckpt/lanczosV" + std::to_string(k_bgn) + ".dat"))) k_bgn++;
            if (k_bgn == maxit) {
                k = 0;
            } else {
                k = k_bgn;
                while (fs::exists(fs::path("out_Qckpt/lanczosV" + std::to_string(k+1) + ".dat"))) k++;
            }
        }
        fout << "Initializing/Resuming from k = " << k << std::endl;
        fout << "Current files on disk: " << std::endl;
        for (auto &p : fs::directory_iterator("out_Qckpt")) fout << p << std::endl;

        if (k > 0) {
            fout << "Loading Lanczos data from disk..." << std::endl;
            int info;
            if (purpose == "iram") {
                for (MKL_INT kk = 0; kk <= k; kk++) {
                    fout << "out_Qckpt/lanczosV" << kk << ".dat" << std::endl;
                    auto info = vec_disk_read("out_Qckpt/lanczosV" + std::to_string(kk) + ".dat", dim, v + dim * kk);
                    assert(info == 0);
                }
            } else {
                fout << "out_Qckpt/lanczosV" + std::to_string(k-1) + ".dat" << std::endl;
                info = vec_disk_read("out_Qckpt/lanczosV" + std::to_string(k-1) + ".dat", dim, v + dim * ((k-1)%2));
                assert(info == 0);
                fout << "out_Qckpt/lanczosV" + std::to_string(k) + ".dat" << std::endl;
                info = vec_disk_read("out_Qckpt/lanczosV" + std::to_string(k) + ".dat", dim, v + dim * (k%2));
                assert(info == 0);
                if (purpose.find("val0") == npos){
                    fout << "out_Qckpt/lanczosY0.dat" << std::endl;
                    info = vec_disk_read("out_Qckpt/lanczosY0.dat", dim, v + 2 * dim);
                    assert(info == 0);
                }
                if (purpose.find("vec1") != npos){
                    fout << "out_Qckpt/lanczosY1.dat" << std::endl;
                    info = vec_disk_read("out_Qckpt/lanczosY1.dat", dim, v + 3 * dim);
                    assert(info == 0);
                }
                if (purpose.find("val") != npos) {
                    fout << "out_Qckpt/lczs_mlns.dat" << std::endl;
                    std::ifstream fmlns("out_Qckpt/lczs_mlns.dat", std::ios::in | std::ios::binary);
                    fmlns.read(reinterpret_cast<char*>(&cnt_accuE0), sizeof(int));
                    fmlns.read(reinterpret_cast<char*>(&accuracy), sizeof(double));
                    fmlns.read(reinterpret_cast<char*>(&theta0_prev), sizeof(double));
                    fmlns.read(reinterpret_cast<char*>(&theta1_prev), sizeof(double));
                    fmlns.close();
                    fout << "cnt_accuE0 = " << cnt_accuE0 << std::endl;
                    fout << "accuracy (* 1e12) = " << accuracy * 1e12 << std::endl;
                    fout << "theta0_prev = " << theta0_prev << std::endl;
                    fout << "theta1_prev = " << theta1_prev << std::endl;
                }
            }
            if (purpose.find("vec") != npos) {
                fout << "out_Qckpt/HessenbergA.dat" << std::endl;
                info = vec_disk_read("out_Qckpt/HessenbergA.dat", maxit, hessenberg + maxit);
                assert(info == 0);
                fout << "out_Qckpt/HessenbergB.dat" << std::endl;
                info = vec_disk_read("out_Qckpt/HessenbergB.dat", maxit, hessenberg);
                assert(info == 0);
            } else {
                fout << "out_Qckpt/HessenbergA.dat" << std::endl;
                info = vec_disk_read("out_Qckpt/HessenbergA.dat", k,   hessenberg + maxit);
                assert(info == 0);
                fout << "out_Qckpt/HessenbergB.dat" << std::endl;
                info = vec_disk_read("out_Qckpt/HessenbergB.dat", k+1, hessenberg);
                assert(info == 0);
            }
        }
        fout << "Log end: " << date_and_time() << std::endl << std::endl;
        fout.close();
    }
    template void ckpt_lanczos_init(MKL_INT &k, const MKL_INT &maxit, const MKL_INT &dim,
                                    int &cnt_accuE0, double &accuracy, double &theta0_prev, double &theta1_prev,
                                    double v[], double hessenberg[], const std::string &purpose);
    template void ckpt_lanczos_init(MKL_INT &k, const MKL_INT &maxit, const MKL_INT &dim,
                                    int &cnt_accuE0, double &accuracy, double &theta0_prev, double &theta1_prev,
                                    std::complex<double> v[], double hessenberg[], const std::string &purpose);


    template <typename T>
    void ckpt_lanczos_update(const MKL_INT &m, const MKL_INT &maxit, const MKL_INT &dim,
                             int &cnt_accuE0, double &accuracy, double &theta0_prev, double &theta1_prev,
                             T v[], double hessenberg[], const std::string &purpose)
    {
        if (! enable_ckpt) return;
        fs::path outdir("out_Qckpt");
        if (fs::exists(outdir)) {
            if (! fs::is_directory(outdir)) {
                fs::remove_all(outdir);
                fs::create_directory(outdir);
            }
        } else {
            fs::create_directory(outdir);
        }

        auto &npos = std::string::npos;
        std::ofstream fout("out_Qckpt/log_Lanczos_ckpt.txt", std::ios::out | std::ios::app);
        fout << std::endl << "Log start: " << date_and_time() << std::endl;
        fout << "Updating Lanczos, purpose = " << purpose << std::endl;
        fout << "Step = " << m << std::endl;
        fout << "Files before updating: " << std::endl;
        for (auto &p : fs::directory_iterator("out_Qckpt")) fout << p << std::endl;

        fs::remove(fs::path("out_Qckpt/lczs_updt.Qckpt1"));
        fs::remove(fs::path("out_Qckpt/lczs_updt.Qckpt2"));
        std::ofstream ftemp("out_Qckpt/lczs_updt.Qckpt1", std::ios::out | std::ios::binary);
        ftemp.write(reinterpret_cast<const char*>(&m), sizeof(MKL_INT));
        ftemp.close();
        if (purpose == "iram") {
            vec_disk_write("out_Qckpt/HessenbergA.dat.new", m,   hessenberg + maxit);
            vec_disk_write("out_Qckpt/HessenbergB.dat.new", m+1, hessenberg);
            for (MKL_INT k = 0; k <= m; k++) {
                if (! fs::exists(fs::path("out_Qckpt/lanczosV" + std::to_string(k) + ".dat")))
                    vec_disk_write("out_Qckpt/lanczosV" + std::to_string(k) + ".dat", dim, v + k * dim);
            }

            // before/after this point, have to use old/new data
            std::ofstream ftemp("out_Qckpt/lczs_updt.Qckpt2", std::ios::out | std::ios::binary);
            ftemp.write(reinterpret_cast<const char*>(&m), sizeof(MKL_INT));
            ftemp.close();

            // new data fully written, start clean up old ones
            fs::remove(fs::path("out_Qckpt/HessenbergA.dat"));
            fs::remove(fs::path("out_Qckpt/HessenbergB.dat"));

            // renaming new data to correct names
            fs::rename(fs::path("out_Qckpt/HessenbergA.dat.new"), fs::path("out_Qckpt/HessenbergA.dat"));
            fs::rename(fs::path("out_Qckpt/HessenbergB.dat.new"), fs::path("out_Qckpt/HessenbergB.dat"));
        } else if (purpose.find("val") != npos) {
            vec_disk_write("out_Qckpt/HessenbergA.dat.new", m,   hessenberg + maxit);
            vec_disk_write("out_Qckpt/HessenbergB.dat.new", m+1, hessenberg);
            if (m > 0 && ! fs::exists(fs::path("out_Qckpt/lanczosV" + std::to_string(m-1) + ".dat")))
                vec_disk_write("out_Qckpt/lanczosV" + std::to_string(m-1) + ".dat", dim, v + ((m-1)%2) * dim);
            vec_disk_write("out_Qckpt/lanczosV" + std::to_string(m) + ".dat", dim, v + (m%2) * dim);
            if (purpose.find("val0") == npos) vec_disk_write("out_Qckpt/lanczosY0.dat.new", dim, v + 2 * dim);
            fout << "cnt_accuE0 = " << cnt_accuE0 << std::endl;
            fout << "accuracy (* 1e12) = " << accuracy * 1e12 << std::endl;
            fout << "theta0_prev = " << theta0_prev << std::endl;
            fout << "theta1_prev = " << theta1_prev << std::endl;
            std::ofstream f_mlns("out_Qckpt/lczs_mlns.dat.new", std::ios::out | std::ios::binary);
            f_mlns.write(reinterpret_cast<const char*>(&cnt_accuE0), sizeof(int));
            f_mlns.write(reinterpret_cast<const char*>(&accuracy), sizeof(double));
            f_mlns.write(reinterpret_cast<const char*>(&theta0_prev), sizeof(double));
            f_mlns.write(reinterpret_cast<const char*>(&theta1_prev), sizeof(double));
            f_mlns.close();

            // before/after this point, have to use old/new data
            std::ofstream ftemp("out_Qckpt/lczs_updt.Qckpt2", std::ios::out | std::ios::binary);
            ftemp.write(reinterpret_cast<const char*>(&m), sizeof(MKL_INT));
            ftemp.close();

            // new data fully written, start clean up old ones
            fs::remove(fs::path("out_Qckpt/HessenbergA.dat"));
            fs::remove(fs::path("out_Qckpt/HessenbergB.dat"));
            for (MKL_INT k = 0; k < m-1; k++)
                fs::remove(fs::path("out_Qckpt/lanczosV" + std::to_string(k) + ".dat"));
            fs::remove(fs::path("out_Qckpt/lanczosY0.dat"));
            fs::remove(fs::path("out_Qckpt/lanczosY1.dat"));
            fs::remove(fs::path("out_Qckpt/lczs_mlns.dat"));

            // renaming new data to correct names
            fs::rename(fs::path("out_Qckpt/HessenbergA.dat.new"), fs::path("out_Qckpt/HessenbergA.dat"));
            fs::rename(fs::path("out_Qckpt/HessenbergB.dat.new"), fs::path("out_Qckpt/HessenbergB.dat"));
            if (purpose.find("val0") == npos)
                fs::rename(fs::path("out_Qckpt/lanczosY0.dat.new"), fs::path("out_Qckpt/lanczosY0.dat"));
            fs::rename(fs::path("out_Qckpt/lczs_mlns.dat.new"), fs::path("out_Qckpt/lczs_mlns.dat"));
        } else if (purpose.find("vec") != npos) {
            if (! fs::exists(fs::path("out_Qckpt/HessenbergA.dat")))
                vec_disk_write("out_Qckpt/HessenbergA.dat", maxit, hessenberg + maxit);
            if (! fs::exists(fs::path("out_Qckpt/HessenbergB.dat")))
                vec_disk_write("out_Qckpt/HessenbergA.dat", maxit, hessenberg);
            if (m > 0 && ! fs::exists(fs::path("out_Qckpt/lanczosV" + std::to_string(m-1) + ".dat")))
                vec_disk_write("out_Qckpt/lanczosV" + std::to_string(m-1) + ".dat", dim, v + ((m-1)%2) * dim);
            vec_disk_write("out_Qckpt/lanczosV" + std::to_string(m) + ".dat", dim, v + (m%2) * dim);
            vec_disk_write("out_Qckpt/lanczosY0.dat.new", dim, v + 2 * dim);
            if (purpose.find("vec1") != npos) vec_disk_write("out_Qckpt/lanczosY1.dat.new", dim, v + 3 * dim);

            // before/after this point, have to use old/new data
            std::ofstream ftemp("out_Qckpt/lczs_updt.Qckpt2", std::ios::out | std::ios::binary);
            ftemp.write(reinterpret_cast<const char*>(&m), sizeof(MKL_INT));
            ftemp.close();

            // new data fully written, start clean up old ones
            for (MKL_INT k = 0; k < m-1; k++)
                fs::remove(fs::path("out_Qckpt/lanczosV" + std::to_string(k) + ".dat"));
            fs::remove(fs::path("out_Qckpt/lanczosY0.dat"));
            fs::remove(fs::path("out_Qckpt/lanczosY1.dat"));

            // renaming new data to correct names
            fs::rename(fs::path("out_Qckpt/lanczosY0.dat.new"), fs::path("out_Qckpt/lanczosY0.dat"));
            fs::rename(fs::path("out_Qckpt/lanczosY1.dat.new"), fs::path("out_Qckpt/lanczosY1.dat"));
        }
        fs::remove(fs::path("out_Qckpt/lczs_updt.Qckpt1"));
        fs::remove(fs::path("out_Qckpt/lczs_updt.Qckpt2"));
        fout << "Files after updating: " << std::endl;
        for (auto &p : fs::directory_iterator("out_Qckpt")) fout << p << std::endl;
        fout << "Log end: " << date_and_time() << std::endl << std::endl;
        fout.close();
    }
    template void ckpt_lanczos_update(const MKL_INT &m, const MKL_INT &maxit, const MKL_INT &dim,
                                      int &cnt_accuE0, double &accuracy, double &theta0_prev, double &theta_prev1,
                                      double v[], double hessenberg[], const std::string &purpose);
    template void ckpt_lanczos_update(const MKL_INT &m, const MKL_INT &maxit, const MKL_INT &dim,
                                      int &cnt_accuE0, double &accuracy, double &theta1_prev, double &theta_prev1,
                                      std::complex<double> v[], double hessenberg[], const std::string &purpose);

    void ckpt_lanczos_clean()
    {
        if (! enable_ckpt) return;
        fs::path outdir("out_Qckpt");
        if (fs::exists(outdir)) {
            if (! fs::is_directory(outdir)) {
                fs::remove_all(outdir);
                fs::create_directory(outdir);
            }
        } else {
            fs::create_directory(outdir);
        }

        std::ofstream fout("out_Qckpt/log_Lanczos_ckpt.txt", std::ios::out | std::ios::app);
        fout << std::endl << "Log start: " << date_and_time() << std::endl;
        fout << "Cleaning up Lanczos..." << std::endl;
        fout << "Current files: " << std::endl;
        for (auto &p : fs::directory_iterator("out_Qckpt")) fout << p << std::endl;

        fs::remove(fs::path("out_Qckpt/HessenbergA.dat"));
        fs::remove(fs::path("out_Qckpt/HessenbergB.dat"));
        fs::remove(fs::path("out_Qckpt/lanczosY0.dat"));
        fs::remove(fs::path("out_Qckpt/lanczosY1.dat"));
        fs::remove(fs::path("out_Qckpt/lczs_mlns.dat"));

        for (auto &p : fs::directory_iterator("out_Qckpt"))
        {
            if (std::regex_match(p.path().filename().string(), std::regex("lanczosV[[:digit:]]+\\.dat"))) fs::remove(p.path());

        }
        fout << "Current files after clean: " << std::endl;
        for (auto &p : fs::directory_iterator("out_Qckpt")) fout << p << std::endl;

        fout << "Log end: " << date_and_time() << std::endl;
        fout.close();
    }


    template <typename T>
    void ckpt_CG_init(MKL_INT &m, const MKL_INT &maxit, const MKL_INT &dim,
                      T v[], T r[], T p[])
    {
        assert(m >= 0 && dim > 0);
        if (! enable_ckpt) return;
        fs::path outdir("out_Qckpt");
        if (fs::exists(outdir)) {
            if (! fs::is_directory(outdir)) {
                fs::remove_all(outdir);
                fs::create_directory(outdir);
            }
        } else {
            fs::create_directory(outdir);
        }

        std::ofstream fout("out_Qckpt/log_CG_ckpt.txt", std::ios::out | std::ios::app);
        fout << std::endl << "Log start: " << date_and_time() << std::endl;

        fout << "Initializing Conjugate Gradient method" << std::endl;
        fout << "Input step = " << m << std::endl;
        fout << "Current files on disk: " << std::endl;
        for (auto &p : fs::directory_iterator("out_Qckpt")) fout << p << std::endl;
        auto size_Qckpt1 = sizeof(MKL_INT);
        bool updating = (fs::exists(fs::path("out_Qckpt/CG_updt.Qckpt1")) &&
                         fs::file_size(fs::path("out_Qckpt/CG_updt.Qckpt1")) == size_Qckpt1) ? true : false;

        fout << "Resuming from an interrupted update? " << updating << std::endl;
        if (updating) {
            fout << "Cleaning up junks from last update." << std::endl;
            bool finished = fs::exists(fs::path("out_Qckpt/CG_updt.Qckpt2"));
            std::ifstream ftemp("out_Qckpt/CG_updt.Qckpt1", std::ios::in | std::ios::binary);
            ftemp.read(reinterpret_cast<char*>(&m), sizeof(MKL_INT));
            ftemp.close();
            if (finished) {
                fout << "New data finished writing while updating Lanczos step m = " << m << std::endl;
                assert(fs::exists(fs::path("out_Qckpt/CG_V" + std::to_string(m) + ".dat")));
                assert(fs::exists(fs::path("out_Qckpt/CG_R" + std::to_string(m) + ".dat")));
                assert(fs::exists(fs::path("out_Qckpt/CG_P" + std::to_string(m) + ".dat")));
                fs::remove(fs::path("out_Qckpt/CG_V" + std::to_string(m-1) + ".dat"));
                fs::remove(fs::path("out_Qckpt/CG_R" + std::to_string(m-1) + ".dat"));
                fs::remove(fs::path("out_Qckpt/CG_P" + std::to_string(m-1) + ".dat"));
                fs::remove(fs::path("out_Qckpt/CG_updt.Qckpt1"));
                fs::remove(fs::path("out_Qckpt/CG_updt.Qckpt2"));
            } else {
                fout << "New data unfinished writing while updating Lanczos step m = " << m << std::endl;
                fout << "Rewinding one step." << std::endl;
                m--;
                assert(m > 0 || fs::exists(fs::path("out_Qckpt/CG_V" + std::to_string(m) + ".dat")));
                assert(m > 0 || fs::exists(fs::path("out_Qckpt/CG_R" + std::to_string(m) + ".dat")));
                assert(m > 0 || fs::exists(fs::path("out_Qckpt/CG_P" + std::to_string(m) + ".dat")));
                fs::remove(fs::path("out_Qckpt/CG_V" + std::to_string(m+1) + ".dat"));
                fs::remove(fs::path("out_Qckpt/CG_R" + std::to_string(m+1) + ".dat"));
                fs::remove(fs::path("out_Qckpt/CG_P" + std::to_string(m+1) + ".dat"));
                fs::remove(fs::path("CG_updt.Qckpt1"));
            }
        } else {
            fs::remove(fs::path("out_Qckpt/CG_updt.Qckpt1"));
            fs::remove(fs::path("out_Qckpt/CG_updt.Qckpt2"));
            m = 0;
            while (m < maxit && ! fs::exists(fs::path("out_Qckpt/CG_V" + std::to_string(m) + ".dat"))) m++;
            if (m == maxit) m = 0;
        }
        fout << "Initializing/Resuming from m = " << m << std::endl;
        fout << "Current files on disk: " << std::endl;
        for (auto &p : fs::directory_iterator("out_Qckpt")) fout << p << std::endl;

        if (m > 0) {
            fout << "Loading CG data from disk..." << std::endl;
            fout << "out_Qckpt/CG_V" + std::to_string(m) + ".dat" << std::endl;
            auto info = vec_disk_read("out_Qckpt/CG_V" + std::to_string(m) + ".dat", dim, v);
            assert(info == 0);
            fout << "out_Qckpt/CG_R" + std::to_string(m) + ".dat" << std::endl;
            info = vec_disk_read("out_Qckpt/CG_R" + std::to_string(m) + ".dat", dim, r);
            assert(info == 0);
            fout << "out_Qckpt/CG_P" + std::to_string(m) + ".dat" << std::endl;
            info = vec_disk_read("out_Qckpt/CG_P" + std::to_string(m) + ".dat", dim, p);
            assert(info == 0);
        }
        fout << "Log end: " << date_and_time() << std::endl << std::endl;
        fout.close();
    }
    template void ckpt_CG_init(MKL_INT &m, const MKL_INT &maxit, const MKL_INT &dim,
                               double v[], double r[], double p[]);
    template void ckpt_CG_init(MKL_INT &m, const MKL_INT &maxit, const MKL_INT &dim,
                               std::complex<double> v[], std::complex<double> r[], std::complex<double> p[]);

    template <typename T>
    void ckpt_CG_update(const MKL_INT &m, const MKL_INT &dim,
                        T v[], T r[], T p[])
    {
        if (! enable_ckpt) return;
        fs::path outdir("out_Qckpt");
        if (fs::exists(outdir)) {
            if (! fs::is_directory(outdir)) {
                fs::remove_all(outdir);
                fs::create_directory(outdir);
            }
        } else {
            fs::create_directory(outdir);
        }

        std::ofstream fout("out_Qckpt/log_CG_ckpt.txt", std::ios::out | std::ios::app);
        fout << std::endl << "Log start: " << date_and_time() << std::endl;
        fout << "Updating CG..." << std::endl;
        fout << "Step = " << m << std::endl;
        fout << "Files before updating: " << std::endl;
        for (auto &p : fs::directory_iterator("out_Qckpt")) fout << p << std::endl;

        fs::remove(fs::path("out_Qckpt/CG_updt.Qckpt1"));
        fs::remove(fs::path("out_Qckpt/CG_updt.Qckpt2"));
        std::ofstream ftemp1("out_Qckpt/CG_updt.Qckpt1", std::ios::out | std::ios::binary);
        ftemp1.write(reinterpret_cast<const char*>(&m), sizeof(MKL_INT));
        ftemp1.close();

        vec_disk_write("out_Qckpt/CG_V" + std::to_string(m) + ".dat", dim, v);
        vec_disk_write("out_Qckpt/CG_R" + std::to_string(m) + ".dat", dim, r);
        vec_disk_write("out_Qckpt/CG_P" + std::to_string(m) + ".dat", dim, p);

        // before/after this point, have to use old/new data
        fs::copy(fs::path("out_Qckpt/CG_updt.Qckpt1"), fs::path("out_Qckpt/CG_updt.Qckpt2"));

        // new data fully written, start clean up old ones
        fs::remove(fs::path("out_Qckpt/CG_V" + std::to_string(m-1) + ".dat"));
        fs::remove(fs::path("out_Qckpt/CG_R" + std::to_string(m-1) + ".dat"));
        fs::remove(fs::path("out_Qckpt/CG_P" + std::to_string(m-1) + ".dat"));

        fs::remove(fs::path("out_Qckpt/CG_updt.Qckpt1"));
        fs::remove(fs::path("out_Qckpt/CG_updt.Qckpt2"));
        fout << "Files after updating: " << std::endl;
        for (auto &p : fs::directory_iterator("out_Qckpt")) fout << p << std::endl;
        fout << "Log end: " << date_and_time() << std::endl << std::endl;
        fout.close();
    }
    template void ckpt_CG_update(const MKL_INT &m, const MKL_INT &dim,
                                 double v[], double r[], double p[]);
    template void ckpt_CG_update(const MKL_INT &m, const MKL_INT &dim,
                                 std::complex<double> v[], std::complex<double> r[], std::complex<double> p[]);

    void ckpt_CG_clean()
    {
        if (! enable_ckpt) return;
        fs::path outdir("out_Qckpt");
        if (fs::exists(outdir)) {
            if (! fs::is_directory(outdir)) {
                fs::remove_all(outdir);
                fs::create_directory(outdir);
            }
        } else {
            fs::create_directory(outdir);
        }

        std::ofstream fout("out_Qckpt/log_CG_ckpt.txt", std::ios::out | std::ios::app);
        fout << std::endl << "Log start: " << date_and_time() << std::endl;
        fout << "Cleaning up CG..." << std::endl;
        fout << "Current files: " << std::endl;
        for (auto &p : fs::directory_iterator("out_Qckpt")) fout << p << std::endl;

        for (auto &p : fs::directory_iterator("out_Qckpt"))
        {
            if (std::regex_match(p.path().filename().string(), std::regex("CG_V[[:digit:]]+\\.dat"))) {
                fs::remove(p.path());
            } else if (std::regex_match(p.path().filename().string(), std::regex("CG_R[[:digit:]]+\\.dat"))) {
                fs::remove(p.path());
            } else if (std::regex_match(p.path().filename().string(), std::regex("CG_P[[:digit:]]+\\.dat"))) {
                fs::remove(p.path());
            }
        }
        fout << "Current files after clean: " << std::endl;
        for (auto &p : fs::directory_iterator("out_Qckpt")) fout << p << std::endl;

        fout << "Log end: " << date_and_time() << std::endl;
        fout.close();
    }
}
