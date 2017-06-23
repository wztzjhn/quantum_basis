#include "graph.h"

namespace qbasis {
    
    void ALGraph::add_edge(const uint64_t &n1, const uint64_t &n2)
    {
        assert(n1 != n2);
        assert(n1 < vertices.size() && n2 < vertices.size());
        vertices[n1].arcs.push_back(n2);
        vertices[n2].arcs.push_back(n1);
        arcnum++;
    }

    void ALGraph::prt() const
    {
        for (uint64_t v = 0; v < vertices.size(); v++) {
            std::cout << v << " -> ";
            for (auto w : vertices[v].arcs) {
                std::cout << w << ",";
            }
            std::cout << std::endl;
        }
    }
    
    void ALGraph::BSF_set_JaJb(std::vector<MKL_INT> &ja, std::vector<MKL_INT> &jb)
    {
        MKL_INT magic = -19900917;
        for (decltype(ja.size()) j = 0; j < ja.size(); j++) ja[j] = magic;
        for (decltype(jb.size()) j = 0; j < jb.size(); j++) jb[j] = magic;
        
        std::vector<bool> visited(vertices.size(),false);
        std::list<uint64_t> Q;
        for (uint64_t v = 0; v < vertices.size(); v++) {
            if (! visited[v]) {
                visited[v] = true;
                // ja, jb arbitrary for this node
                //std::cout << "v = " << v << std::endl;
                assert(ja[vertices[v].i_a] == magic && jb[vertices[v].i_b] == magic);
                jb[vertices[v].i_b] = static_cast<MKL_INT>(v)/2;
                ja[vertices[v].i_a] = static_cast<MKL_INT>(v) - jb[vertices[v].i_b];
                //std::cout << "ia,ib -> ja,jb : " << vertices[v].i_a << "," << vertices[v].i_b << " -> " <<
                //ja[vertices[v].i_a] << "," << jb[vertices[v].i_b] << std::endl;
                Q.push_back(v);
                while (Q.size() != 0) {
                    uint64_t u = Q.front();
                    Q.pop_front();
                    auto u_arcs = vertices[u].arcs;
                    for (auto w : u_arcs) {
                        if (! visited[w]) {
                            visited[w] = true;
                            // ja, jb fixed from connected neighbors
                            //std::cout << "w = " << w << std::endl;
                            if (ja[vertices[w].i_a] == magic) {
                                assert(jb[vertices[w].i_b] != magic);
                                ja[vertices[w].i_a] = static_cast<MKL_INT>(w) - jb[vertices[w].i_b];
                            } else if (jb[vertices[w].i_b] == magic) {
                                assert(ja[vertices[w].i_a] != magic);
                                jb[vertices[w].i_b] = static_cast<MKL_INT>(w) - ja[vertices[w].i_a];
                            } else {
                                assert(ja[vertices[w].i_a] + jb[vertices[w].i_b] == static_cast<MKL_INT>(w));
                            }
                            //std::cout << "ia,ib -> ja,jb : " << vertices[w].i_a << "," << vertices[w].i_b << " -> " <<
                            //ja[vertices[w].i_a] << "," << jb[vertices[w].i_b] << std::endl;
                            Q.push_back(w);
                        }
                    }
                }
            }
        }
    }

}
