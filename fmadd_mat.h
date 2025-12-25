namespace cache_op {
    static inline __attribute__((always_inline)) void fmadd_mat(const double* __restrict A,
                                                                const double* __restrict B, 
                                                                double* __restrict C);
    static inline __attribute__((always_inline)) void neg_mat(double* __restrict A);

    // NB: source matrix should be 256x256, not 128x128. Output matrix C is 128x128.
    // Only process a 128x128 chunk.

    /* NB: for standard matrix layout
     * i.e., {{1,2},{3,4}} -> [1,2,3,4]
     * prototype: do operation on matrix A, B and store in C
     */
    
    static inline __attribute__((always_inline)) void add_mat_a(const double* __restrict A, 
                                                            const double* __restrict B, 
                                                            double* __restrict C);
    static inline __attribute__((always_inline)) void sub_mat_a(const double* __restrict A,
                                                            const double* __restrict B, 
                                                            double* __restrict C);
    static inline __attribute__((always_inline)) void store_mat_a(const double* __restrict A,  
                                                                double* __restrict B);
    
    /* NB: for panel matrices (matrix B in fmadd_mat) 
     * i.e., with width 2, {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}} -> strip 1 = [(1,2),5,6,9,10,13,14]; circled = width
     * strip 2 = [3,4,7,8,11,12,15,16]; store = [strip 1 | strip 2] contiguously 
     * prototype: do operation on matrix A, B and store in C. C is in panel format.
     */

    static inline __attribute__((always_inline)) void add_mat_b(const double* __restrict A, 
                                                            const double* __restrict B, 
                                                            double* __restrict C);
    static inline __attribute__((always_inline)) void sub_mat_b(const double* __restrict A,
                                                            const double* __restrict B, 
                                                            double* __restrict C);
    static inline __attribute__((always_inline)) void store_mat_b(const double* __restrict A,  
                                                                double* __restrict B);
}