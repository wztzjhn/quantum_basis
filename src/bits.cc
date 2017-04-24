#include <iostream>
#include "qbasis.h"

namespace qbasis {
    // ----------------- implementation of bits ------------------

    Dbits::Dbits(const int &orb, const int &num_bits)
    {
        assert(num_bits > 0);
        
    }

    int Dbits::bit_read(const int &pos_bit_overall)
    {
        assert(pos_bit_overall >= 0);
        int pos_byte       = pos_bit_overall / 8;
        int pos_bit_detail = pos_bit_overall % 8;
        
        
        
    }

}