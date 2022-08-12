#ifndef graph_h
#define graph_h

#include <cstdint>
#include <vector>
#include <list>
#include <iostream>
#include <cassert>
#include "mkl.h"

namespace qbasis {

    // node info filled with {i_a,i_b} used for Lin Table, arcs contains the next vertex connected by this arc
    struct VNode {
        MKL_INT i_a;
        MKL_INT i_b;
        std::list<uint64_t> arcs;
    };

    class ALGraph {
    public:
        ALGraph() = default;

        ALGraph(const uint64_t &num_nodes) { vertices.resize(num_nodes); arcnum = 0;}

        ~ALGraph() = default;

        uint64_t num_arcs() const { return arcnum; }

        void prt() const;

        VNode& operator[](uint64_t n){ return vertices[n]; }

        void add_edge(const uint64_t &n1, const uint64_t &n2);

        // use BSF search, to set the values of Ja, Jb for Lin Table
        // return = 0 : success
        // return = 1 : fail
        int BSF_set_JaJb(std::vector<MKL_INT> &ja, std::vector<MKL_INT> &jb);

    private:
        std::vector<VNode> vertices;
        uint64_t arcnum;
    };


}

#endif
