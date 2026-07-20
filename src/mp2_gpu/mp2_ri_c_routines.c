#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "../mpiwrap/cp_mpi.h"
// I use it like a timer
#include "../offload/offload_library.h"

// Helper function to compute the maximum value in an array
static int array_max(int *array, int size) {
    int max_value = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max_value) {
            max_value = array[i];
        }
    }
    return max_value;
}

// Helper function to compute the sum of products
static int sum_of_product(const int *array1, const int *array2, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += array1[i] * array2[i];
    }
    return sum;
}

// Helper function to find integration group size
static int find_integ_group_size(int ngroup, int max_repl_group_size) {
    int integ_group_size = ngroup;
    int min_repl_group_size = ngroup / max_repl_group_size;

    if (max_repl_group_size < 1) {
        max_repl_group_size = 1;
    }

    if(max_repl_group_size > ngroup) {
        max_repl_group_size = ngroup;

    }

    if (min_repl_group_size < 1) {
        min_repl_group_size = 1;
    }

    if (min_repl_group_size > ngroup) {
        min_repl_group_size = ngroup;
    }

    // Find smallest divisor >= min_repl_group_size
    for (int i = min_repl_group_size; i <= max_repl_group_size; i++) {
        if (ngroup % i == 0) {
            integ_group_size = i;
            break;
        }
    }
    return integ_group_size;
}

/**
 * Helper: modulo operation that works like fortran MOD,
 * ensuring non-negative results.
 */
static int modulo_frotran(int a, int b) {
    int result = a % b;
    if (result < 0) {
        result += b;
    }
    return result;
}

/**
 * Helper: allocate 4D arrays as flat array with dimensions:
 * ranges_info_array[dim][rep_rank][exchange_rank]
 */
static int* allocate_ranges_info_array(int dim, int rep_size, int exchange_size) {
    // int total_size = 4 * rep_size * exchange_size;
    int total_size = dim * rep_size * exchange_size;
    int* array = (int*)calloc(total_size, sizeof(int));
    return array;
}

/**
 * Helper: index into ranges_info_array
 * ranges_info_array[dim * rep_size * exchange_size + rep_rank * exchange_size + exchange_rank]
 */
static int ranges_info_index(int dim, int rep_rank, int exchange_rank, int rep_size, int exchange_size) {
    return dim * rep_size * exchange_size + rep_rank * exchange_size + exchange_rank;
}

/**
 * Function to calculate the integration group size based on memory constraints and other parameters.
 * =========PARAMETERS=========
 *  // Output parameters
 * int* integ_group_size_out,
 * int* ngroup_out,
 * int* num_integ_group_out,
 * 
 * // Input parameters
 * int ngroup,  // from para_env%num_pr / para_env_sub%num_pr
 * int num_integ_group, // computed from ngroup/integ_grup_size
 * int integ_group_size,
 * 
 * // From mp2_env
 * double mp2_memory, // mp2_env%mp2_memory
 * int number_integration_groups, // mp2_env%ri_mp2%number_integration_groups
 * const int* homo,
 * int homo_size,
 * 
 * // from virtual
 * const int* virtual_arr,
 * int virtual_size,
 * int dimen_RI,
 * bool calc_forces,
 * int unit_nr,
 * 
 * // gd_array%sizes
 * const int* gd_array_sizes,
 * int gd_array_sizes_size,
 * 
 * // gd_B_virtual%sizes
 * const int* gd_B_virtual_sizes,
 * int gd_B_virtual_sizes_size,
 * 
 * // try to extract from gd_array and gd_B_virtual
 * int maxsize_gd_array, // Maximum size of gd_array
 * int maxsize_gd_B_virtual, // Maximum size of gd_B_virtual
 * int maxval_gd_B_virtual, // Maximum value in gd_B_virtual
 * int maxval_virtual, // Maximum value in virtual
 * int max_homo, // Maximum value in homo
 * int sum_homo_virtual, // Sum of products of homo and virtual
 * int product_homo, // Product of all values in homo
 * 
 * // SHOULD BE 1
 * int nspins, // 1 in this case; other case: SIZE(homo)
 * bool calc_group_size, // Flag to calculate group size
 */
void c_mp2_ri_get_integ_group_size(
    int* integ_group_size_out, int* ngroup_out, int* num_integ_group_out,
    int ngroup, int num_integ_group, int integ_group_size,
    double mp2_memory, int number_integration_groups, const int* homo,
    int homo_size, const int* virtual_arr, int virtual_size,
    int dimen_RI, bool calc_forces, int unit_nr,
    const int* gd_array_sizes, int gd_array_sizes_size,
    const int* gd_B_virtual_sizes, int gd_B_virtual_sizes_size,
    int maxsize_gd_array, int maxsize_gd_B_virtual, int maxval_gd_B_virtual,
    int maxval_virtual, int max_homo, int sum_homo_virtual,
    int product_homo, int nspins, int calc_group_size, cp_mpi_comm_t comm) {
    // Local variables
    bool calc_group_size_local = calc_group_size;
    int block_size = 1;
    int max_repl_group_size = 1;
    int min_integ_group_size = 1;
    
    // Memory calculation variables
    double mem_real = 0.0;
    double mem_base = 0.0;
    double mem_per_blk = 0.0;
    double mem_per_repl = 0.0;
    double mem_per_repl_blk = 0.0;
    double mem_min = 0.0;
    double factor = 0.0;

    mem_real = mp2_memory;
    
    // Calculate memory components (based on Fortran logic)
    
    // BIB_C_copy: MAX(MAX(homo*maxsize(gd_array)), dimen_RI) * maxsize(gd_B_virtual)
    double max_homo_gd = 0.0;
    for (int i = 0; i < homo_size; i++) {
        double temp = (double)homo[i] * maxsize_gd_array;
        if (temp > max_homo_gd) {
            max_homo_gd = temp;
        }
    }
    double max_compare = (max_homo_gd > (double)dimen_RI) ? max_homo_gd : (double)dimen_RI;
    mem_per_repl += max_compare * maxsize_gd_B_virtual * 8.0 / (1024.0 * 1024.0);
    
    // BIB_C: SUM(homo*maxsize(gd_B_virtual)) * maxsize(gd_array)
    double sum_homo_gd_B = 0.0;
    for (int i = 0; i < homo_size; i++) {
        sum_homo_gd_B += (double)homo[i] * maxsize_gd_B_virtual;
    }
    mem_per_repl += sum_homo_gd_B * maxsize_gd_array * 8.0 / (1024.0 * 1024.0);
    
    // BIB_C_rec: maxsize(gd_B_virtual) * maxsize(gd_array)
    mem_per_repl_blk += (double)maxval_gd_B_virtual * maxsize_gd_array * 8.0 / (1024.0 * 1024.0);
    
    // local_i_aL+local_j_aL: 2 * maxsize(gd_B_virtual) * dimen_RI
    mem_per_blk += 2.0 * maxval_gd_B_virtual * (double)dimen_RI * 8.0 / (1024.0 * 1024.0);
    
    // local_ab: MAX(virtual*maxsize(gd_B_virtual))
    double max_virtual_gd_B = 0.0;
    for (int i = 0; i < virtual_size; i++) {
        double temp = (double)virtual_arr[i] * maxsize_gd_B_virtual;
        if (temp > max_virtual_gd_B) {
            max_virtual_gd_B = temp;
        }
    }
    mem_base += max_virtual_gd_B * 8.0 / (1024.0 * 1024.0);
    
    // external_ab/external_i_aL: MAX(dimen_RI, max_virtual) * maxsize(gd_B_virtual)
    int max_dim = (dimen_RI > maxval_virtual) ? dimen_RI : maxval_virtual;
    mem_base += (double)max_dim * maxval_gd_B_virtual * 8.0 / (1024.0 * 1024.0);
    
    if (calc_forces) {
        // Gamma_P_ia: SUM(homo*maxsize(gd_array)*maxsize(gd_B_virtual))
        double sum_homo_gd = 0.0;
        for (int i = 0; i < homo_size; i++) {
            sum_homo_gd += (double)homo[i] * maxsize_gd_array * maxsize_gd_B_virtual;
        }
        mem_per_repl += sum_homo_gd * 8.0 / (1024.0 * 1024.0);
        
        // Y_i_aP+Y_j_aP: 2 * maxsize(gd_B_virtual) * dimen_RI
        mem_per_blk += 2.0 * maxval_gd_B_virtual * dimen_RI * 8.0 / (1024.0 * 1024.0);
        
        // local_ba/t_ab: maxsize(gd_B_virtual) * MAX(dimen_RI, max_virtual)
        mem_base += (double)maxval_gd_B_virtual * max_dim * 8.0 / (1024.0 * 1024.0);
        
        // P_ij: SUM(homo*homo)
        double sum_homo_sq = 0.0;
        for (int i = 0; i < homo_size; i++) {
            sum_homo_sq += (double)homo[i] * homo[i];
        }
        mem_base += sum_homo_sq * 8.0 / (1024.0 * 1024.0);
        
        // P_ab: SUM(virtual*maxsize(gd_B_virtual))
        double sum_virtual_gd_B = 0.0;
        for (int i = 0; i < virtual_size; i++) {
            sum_virtual_gd_B += (double)virtual_arr[i] * maxsize_gd_B_virtual;
        }
        mem_base += sum_virtual_gd_B * 8.0 / (1024.0 * 1024.0);
        
        // send_ab/send_i_aL: MAX(dimen_RI, max_virtual) * maxsize(gd_B_virtual)
        mem_base += (double)max_dim * maxval_gd_B_virtual * 8.0 / (1024.0 * 1024.0);
    }
    
    // Initial block size guess
    // block_size = MAX(1, MIN(FLOOR(SQRT(MINVAL(homo))), FLOOR(MINVAL(homo)/SQRT(2.0*ngroup))))
    int min_homo = max_homo;
    for (int i = 0; i < homo_size; i++) {
        if (homo[i] < min_homo) min_homo = homo[i];
    }
    block_size = (int)sqrt((double)min_homo);
    int temp = (int)(min_homo / sqrt(2.0 * ngroup));
    block_size = (block_size < temp) ? block_size : temp;
    block_size = (block_size < 1) ? 1 : block_size;
    
    // User-provided block size would be set here
    // if (user_block_size > 0) block_size = user_block_size;
    
    mem_min = mem_base + mem_per_repl + (mem_per_blk + mem_per_repl_blk) * block_size;
    
    // Print memory info if unit_nr > 0
    if (unit_nr > 0) {
        // Using printf for now - would use CP2K logging in production
        printf("RI_INFO| Minimum available memory per MPI process: %9.2f MiB\n", mem_real);
        printf("RI_INFO| Minimum required memory per MPI process: %9.2f MiB\n", mem_min);
    }
    
    // Calculate factor for communication model
    // factor = SUM(homovirtual) - SUM((MAX(homo)/block_size + block_size - 2)*homovirtual)/ngroup
    double factor_homo = 0.0;
    for (int i = 0; i < homo_size; i++) {
        factor_homo += (double)homo[i] * virtual_arr[i];
    }
    
    double sum_factor = 0.0;
    for (int i = 0; i < homo_size; i++) {
        double temp = ((double)max_homo / block_size + block_size - 2.0);
        sum_factor += temp * homo[i] * virtual_arr[i];
    }
    factor = factor_homo - sum_factor / ngroup;
    
    if (nspins == 2) {
        factor = factor - 2.0 * product_homo / block_size / ngroup * sum_homo_virtual;
    }
    
    // Determine integration group size
    integ_group_size = ngroup;  // Default
    
    if (factor <= 0.0) {
        // Calculate max replication group size
        // max_repl_group_size = FLOOR((mem_real - mem_base - mem_per_blk*block_size) /
        //                             (mem_per_repl + mem_per_repl_blk*block_size))
        double numerator = mem_real - mem_base - mem_per_blk * block_size;
        double denominator = mem_per_repl + mem_per_repl_blk * block_size;
        
        if (denominator > 0.0) {
            max_repl_group_size = (int)(numerator / denominator);
        } else {
            max_repl_group_size = 1;
        }
        
        // Clamp
        if (max_repl_group_size < 1) max_repl_group_size = 1;
        if (max_repl_group_size > ngroup) max_repl_group_size = ngroup;
        
        // Find integration group size
        integ_group_size = find_integ_group_size(ngroup, max_repl_group_size);
    }
    
    // If calc_group_size is false, use user-provided group size
    if (!calc_group_size_local) {
        integ_group_size = ngroup / number_integration_groups;
    }
    
    // Print result
    if (unit_nr > 0) {
        printf("RI_INFO| Group size for integral replication: %6d\n", integ_group_size);
        fflush(stdout);
    }
    
    // Compute num_integ_group
    num_integ_group = ngroup / integ_group_size;
    
    // Return values
    *integ_group_size_out = integ_group_size;
    *ngroup_out = ngroup;
    *num_integ_group_out = num_integ_group;
}

/**
 * now create a group that contains all the proc that have
 * the same virtual starting point in the integ group
 * 
 * This subroutine creates the parallel infrastructure for RI-MP2:
 * 1. Creates exchange communicator (comm_exchange)
 * 2. Creates replication communicator (comm_rep)
 * 3. Builds ranges_info_array for data redistribution
 * 4. Updates sizes_array for new group distribution
 * ===========PARAMETERS=========
 * // OUTPUTS (communicators as Fortran version)
 * int* comm_exchange_out,
 * int* comm_rep_out,
 * 
 * // INPUT/OUTPUT arrays
 * int* sizes_array,
 * int sizes_array_size,
 * 
 * // OUTPUT arrays
 * int* ranges_info_array,
 * int ranges_info_array_dim1,  // Should be 4
 * int ranges_info_array_dim2,  // comm_rep%num_pe
 * int ranges_info_array_dim3,  // comm_exchange%num_pe
 * int* integ_group_pos2color_sub,
 * int integ_group_pos2color_sub_size,
 * int* sizes_array_orig,
 * int sizes_array_orig_size,
 * 
 * // OUTPUT scalars
 * int* my_group_L_size_out,
 * int* my_group_L_size_orig_out,
 * int* my_new_group_L_size_out,
 * 
 * // INPUT scalars
 * int my_group_L_start,
 * int my_group_L_end,
 * 
 * // MPI communicators (Fortran version)
 * int para_env_comm,
 * int para_env_sub_comm,
 * 
 * // Other inputs
 * int color_sub,
 * int integ_group_size,
 * int num_integ_group,
 * bool calc_forces,
 * int my_group_L_size,
 * int my_group_L_size_orig
 */
void c_mp2_ri_create_group(
    int* comm_exchange_out, int* comm_rep_out, int* sizes_array,
    int sizes_array_size, int* ranges_info_array, int ranges_info_array_dim1,
    int ranges_info_array_dim2, int ranges_info_array_dim3, int* integ_group_pos2color_sub,
    int integ_group_pos2color_sub_size, int* sizes_array_orig, int sizes_array_orig_size,
    int* my_group_L_size_out, int* my_group_L_size_orig_out, int* my_new_group_L_size_out,
    int my_group_L_start, int my_group_L_end, int para_env_comm, int para_env_sub_comm,
    int color_sub, int integ_group_size, int num_integ_group, bool calc_forces,
    int my_group_L_size, int my_group_L_size_orig) {
    // Convert Fortran MPI communicators to C MPI communicators
    cp_mpi_comm_t comm_para_env_c_comm = cp_mpi_comm_f2c(para_env_comm);
    cp_mpi_comm_t comm_para_env_sub_c_comm = cp_mpi_comm_f2c(para_env_sub_comm);

    // Get rank and size of the sub-communicator
    int para_env_rank = cp_mpi_comm_rank(comm_para_env_c_comm);
    int para_env_size = cp_mpi_comm_size(comm_para_env_c_comm);
    int para_env_sub_rank = cp_mpi_comm_rank(comm_para_env_sub_c_comm);
    int para_env_sub_size = cp_mpi_comm_size(comm_para_env_sub_c_comm);

    // Calculate the ngroup
    int ngroup = para_env_size / integ_group_size;

    // Local variables
    cp_mpi_comm_t comm_exchange_c;
    cp_mpi_comm_t comm_rep_c;

    int comm_exchange_rank = 0;
    int comm_exchange_size = 0;
    int comm_rep_rank = 0;
    int comm_rep_size = 0;

    int my_new_group_L_size = my_group_L_size;
    int my_new_group_L_size_orig = my_group_L_size_orig;


    /**
     * ====================================================
     * STEP 1: Create exchange communicator (comm_exchange)
     * ====================================================
     * From sub_sub_color = para_env_sub%mepos * num_inte_group + color_sub/integ_group_size
     */
    int sub_sub_color_exchange = para_env_sub_rank * num_integ_group + color_sub / integ_group_size;

    // Split the world communicator
    // Use the rank as ket for consistent ordering
    int exchange_key = para_env_rank;
    cp_mpi_comm_t comm_exchange_old;

    // Create the exchange communicator
    MPI_Comm_split(comm_para_env_c_comm, sub_sub_color_exchange, exchange_key, &comm_exchange_c);

    // Get info about exchange communicator
    comm_exchange_rank = cp_mpi_comm_rank(comm_exchange_c);
    comm_exchange_size = cp_mpi_comm_size(comm_exchange_c);

    offload_timeset("mp2_ri_create_group\0");
    /**
     * ====================================================
     * STEP 2: Create replication communicator (comm_rep)
     * ====================================================
     * From sub_sub_color = para_env_sub%mepos*comm_exchange%num_pe + comm_exchange%mepos
     */
    int sub_sub_color = para_env_sub_rank * comm_exchange_size + comm_exchange_rank;

    // Create replication communicator
    MPI_Comm_split(comm_para_env_c_comm, sub_sub_color, exchange_key, &comm_rep_c);

    // Get info about replication communicator
    comm_rep_rank = cp_mpi_comm_rank(comm_rep_c);
    comm_rep_size = cp_mpi_comm_size(comm_rep_c);

    /**
     * ====================================================
     * STEP 3: Build replication information
     * ====================================================
     */

    // Allocate arrays for gathering replication infor
    int* rep_sizes_array = (int*)malloc(comm_rep_size * sizeof(int));
    int* rep_starts_array = (int*)malloc(comm_rep_size * sizeof(int));
    int* rep_ends_array = (int*)malloc(comm_rep_size * sizeof(int));

     /**
      * cp_mpi_allgather_int(
      *     const int *sendbuf,
      *     const int sendcount,
      *     int *recvbuf,
      *     const int recvcount,
      *     const cp_mpi_comm_t comm
      * );
      */
    cp_mpi_allgather_int(&my_group_L_size, 1, rep_sizes_array, 1, comm_rep_c);
    cp_mpi_allgather_int(&my_group_L_start, 1, rep_starts_array, 1, comm_rep_c);
    cp_mpi_allgather_int(&my_group_L_end, 1, rep_ends_array, 1, comm_rep_c);

    /**
     * ====================================================
     * STEP 4: Calculate new group size and build my_info
     * ====================================================
     */

    // Allocate my_ifno array (4 x comm_rep_size)
    int* my_info = (int*)malloc(4 * comm_rep_size * sizeof(int));
    
    // Info of this process
    my_info[0 * comm_rep_size + 0] = my_group_L_start; // start
    my_info[1 * comm_rep_size + 0] = my_group_L_end; // end
    my_info[2 * comm_rep_size + 0] = 1; // local_start
    my_info[3 * comm_rep_size + 0] = my_group_L_size; // local_end

    my_new_group_L_size = my_group_L_size;

    // Loop ove other processes in replication group
    for (int proc_shift = 1; proc_shift < comm_rep_size; proc_shift++) {

        /**
         * I suppose this is the ring communication pattern,
         * where each process communicates with its neighbor in the replication group.
         * The modulo_frotran function is try to ensure that the rank wraps around correctly (always +).
         * The proc_receive is the rank of the process from which this process will receive data.
         * The my_info array is being filled with the start, end, local_start, and
         * 
         * Process 0 collects from:  0, 3, 2, 1 (counter-clockwise)
         * Process 1 collects from:  1, 0, 3, 2                    
         * Process 2 collects from:  2, 1, 0, 3                    
         * Process 3 collects from:  3, 2, 1, 0
         * Each process gets data from the others
         * 
         * I think for example in the process 2, will be something like:
         * proc_shift=1:  Process 2 -> collects from Process 1
         * proc_shift=2:  Process 2 -> collects from Process 0
         * proc_shift=3:  Process 2 -> collects from Process 3
         */
        int proc_receive = modulo_frotran(comm_rep_rank - proc_shift, comm_rep_size);
        
        // Update new group size
        my_new_group_L_size += rep_sizes_array[proc_receive];

        my_info[0 * comm_rep_size + proc_shift] = rep_starts_array[proc_receive]; // start
        my_info[1 * comm_rep_size + proc_shift] = rep_ends_array[proc_receive]; // end
        my_info[2 * comm_rep_size + proc_shift] = my_info[3 * comm_rep_size + proc_shift - 1] + 1; // local_start
        my_info[3 * comm_rep_size + proc_shift] = my_new_group_L_size; // local_end
    }

    /**
     * ====================================================
     * STEP 5: Build ranges_info_array
     * ====================================================
     */

    // Allocate ranges_info_array as a flat array (4 x comm_rep_size x comm_exchange_size)
    int* new_sizes_array = (int*)malloc(comm_exchange_size * sizeof(int));
    int* my_info_temp = my_info; // Temporary pointer to my_info for easier indexing

    /**
      * Gather my_new_group_L_size from all processes in the exchange communicator
      * cp_mpi_allgather_int(
      *     const int *sendbuf,
      *     const int sendcount,
      *     int *recvbuf,
      *     const int recvcount,
      *     const cp_mpi_comm_t comm
      * );
      */
    cp_mpi_allgather_int(&my_new_group_L_size, 1, new_sizes_array, 1, comm_exchange_c);

    // Gather my_info from all processes in the exchange communicator
    int my_info_size = 4 * comm_rep_size;
    int* all_my_info =(int*)malloc(my_info_size * comm_exchange_size * sizeof(int));

    /**
      * cp_mpi_allgather_int(
      *     const int *sendbuf,
      *     const int sendcount,
      *     int *recvbuf,
      *     const int recvcount,
      *     const cp_mpi_comm_t comm
      * );
      */
    cp_mpi_allgather_int(my_info, my_info_size, ranges_info_array, my_info_size, comm_exchange_c);

    free(rep_sizes_array);
    free(rep_starts_array);
    free(rep_ends_array);

    /**
      * Build the integ_group_po2color_sub
      * 
      * cp_mpi_allgather_int(
      *     const int *sendbuf,
      *     const int sendcount,
      *     int *recvbuf,
      *     const int recvcount,
      *     const cp_mpi_comm_t comm
      * );
      */
    cp_mpi_allgather_int(&color_sub, 1, integ_group_pos2color_sub, 1, comm_exchange_c);

    /**
     * ====================================================
     * STEP 6: Update sizes_array
     * ====================================================
     */

    /**
     * cp_mpi_allgather_int(
     *      const int *sendbuf,
     *      const int sendcount,
     *      int *recvbuf,
     *      const int recvcount,
     *      const cp_mpi_comm_t comm
     * );
     */
    cp_mpi_allgather_int(&my_new_group_L_size, 1, new_sizes_array, 1, comm_exchange_c);
    
    // ALLOCATE (sizes_array(0:integ_group_size - 1))
    sizes_array = (int*)malloc(integ_group_size * sizeof(int));

    // Copy data from new_sizes_array to size_array
    for (int i = 0; i < integ_group_size && i < comm_exchange_size; i++) {
        sizes_array[i] = new_sizes_array[i];
    }

    // DEALLOCATE (new_sizes_array)
    free(new_sizes_array);

    // time stop
    offload_timestop();
}


/**
 * // Input/Output: BIb_C array (3D: L, virtual, occupied)
 * SIZE(Bib_C, 1) = L-size
 * SIZE(Bib_C, 2) = virtual orbitals
 * SIZE(Bib_C, 3) = occupied orbitals
 * 
 * double** BIb_C,          // Pointer to array pointer (can reallocate)
 * int* BIb_C_L_size,       // Current L-size of BIb_C
 * int* BIb_C_virtual,      // Virtual orbitals dimension
 * int* BIb_C_occupied,     // Occupied orbitals dimension
 * 
 * // MPI communicators (Fortran handles)
 * int comm_exchange,
 * int comm_rep,
 * 
 * // Dimensions
 * int homo,                // Number of occupied orbitals
 * const int* sizes_array,  // Distribution of L-indices
 * int sizes_array_size,    // Size of sizes_array
 * int my_B_size,           // Number of virtual orbitals for this process
 * int my_group_L_size,     // New L-size after replication
 * 
 * // Range mapping information
 * // Follow the last subroutine I think:
 * const int* ranges_info_array,  // 4 x rep_size x exchange_size
 * int ranges_info_dim1,          // Should be 4
 * int ranges_info_dim2,          // comm_rep%num_pe
 * int ranges_info_dim3           // comm_exchange%num_pe
 */
void c_replicate_iaK_2intgroup(
    double** BIb_C, int* BIb_C_L_size, int* BIb_C_virtual,
    int* BIb_C_occupied, int comm_exchange, int comm_rep,
    int homo, const int* sizes_array, int sizes_array_size,
    int my_B_size, int my_group_L_size, const int* ranges_info_array,
    int ranges_info_dim1, int ranges_info_dim2, int ranges_info_dim3
) {
    cp_mpi_comm_t comm_exchange_c = cp_mpi_comm_f2c(comm_exchange);
    cp_mpi_comm_t comm_rep_c = cp_mpi_comm_f2c(comm_rep);
    
    int comm_rep_size = cp_mpi_comm_size(comm_rep_c);
    int comm_exchange_rank = cp_mpi_comm_rank(comm_exchange_c);
    int comm_rep_rank = cp_mpi_comm_rank(comm_rep_c);

    offload_timeset("replicate_iaK_2intgroup\0");

    // Replication scheme using mpi_allgather
    // get the max L size
    int max_L_size = 0;
    for (int i = 0; i < sizes_array_size; i++) {
        if (sizes_array[i] > max_L_size) {
            max_L_size = sizes_array[i];
        }
    }

    // STEP 1: Create local copy (BIb_C_copy)

    // Get current BIb_C dimensions
    int current_L_size = *BIb_C_L_size;
    int virtual_size = *BIb_C_virtual;
    int occupied_size = *BIb_C_occupied;

    // Allocate copy buffer: [L][virtual][occupied]
    size_t copy_size = (size_t)max_L_size * my_B_size * homo;
    // fill with 0 calloc instead of malloc
    double* BIb_C_copy = (double*)calloc(copy_size, sizeof(double));

    // copy data from old BIb_C to copy buffer
    for (int i = 0; i < homo; i++) {
        for (int j = 0; j < my_B_size; j++) {
            //Source: old BIb_C (L,j,i) - colum-major
            //Source index: (i*my_B_size + j) * current_L_size
            size_t src_idx = ((size_t)i * my_B_size + j) * current_L_size;

            //Destination: new BIb_C (i,j,L) - row-major
            //Destination index: (i*my_B_size + j) * max_L_size
            size_t dst_idx = ((size_t)i * my_B_size + j) * max_L_size;

            // Copy current_L_size doubles
            // memcpy(void *dest, const void *src, size_t count);
            memcpy(&BIb_C_copy[dst_idx], &(*BIb_C)[src_idx], current_L_size * sizeof(double));
        }
    }
     // Free original BIb_C
    free(*BIb_C);
    
    //  Gather all data using cp_mpi_allgather_double
    
    // Allocate gather buffer: [comm_rep_size][max_L_size][my_B_size][homo]
    size_t gather_size = (size_t)comm_rep_size * max_L_size * my_B_size * homo;
    double* BIb_C_gather = (double*)calloc(gather_size, sizeof(double));
    
    int send_count = (int)(max_L_size * my_B_size * homo);

    cp_mpi_allgather_double(BIb_C_copy, send_count, BIb_C_gather, send_count, comm_rep_c);
    
    // Free copy buffer
    free(BIb_C_copy);
    
    // Reorder and store replicated data

    // Allocate new BIb_C: [my_group_L_size][my_B_size][homo]
    size_t new_size = (size_t)my_group_L_size * my_B_size * homo;
    double* BIb_C_new = (double*)calloc(new_size, sizeof(double));
    
    // Reorder data using ranges_info_array
    for (int proc_shift = 0; proc_shift < comm_rep_size; proc_shift++) {
        // Which process are we getting data from?
        int proc_receive = (comm_rep_rank - proc_shift) % comm_rep_size;
        if (proc_receive < 0) {
            proc_receive += comm_rep_size;
        }

        // ranges_info_array(dim, proc_shift, exchange_rank)
        // dim=3: local_start, dim=4: local_end
        int start_point = ranges_info_array[
            3 * comm_rep_size * comm_exchange_rank +
            proc_shift * comm_exchange_rank + 3
        ];
        int end_point = ranges_info_array[
            4 * comm_rep_size * comm_exchange_rank +
            proc_shift * comm_exchange_rank + 4
        ];
        int L_size = end_point - start_point + 1;
        
        // Calculate offsets
        // Each process's data in gather buffer
        size_t gather_offset = (size_t)proc_receive * max_L_size * my_B_size * homo;
        
        // Output starts at start_point-1 (0-based) in L dimension
        size_t output_offset = ((size_t)0 * my_B_size + 0) * my_group_L_size + (start_point - 1);
        
        // Copy data from gather buffer to output
        for (int i = 0; i < homo; i++) {
            for (int a = 0; a < my_B_size; a++) {
                // Source: gather buffer at (proc_receive, i, a, L)
                size_t src = gather_offset + ((size_t)i * my_B_size + a) * max_L_size;
                
                // Destination: new BIb_C at (i, a, L) from start_point to end_point
                size_t dst = ((size_t)i * my_B_size + a) * my_group_L_size + (start_point - 1);
                
                // Copy L_size doubles
                memcpy(
                    &BIb_C_new[dst],
                    &BIb_C_gather[src], 
                    L_size * sizeof(double)
                );
            }
        }
    }
    
    // Free gather buffer
    free(BIb_C_gather);
    
    // Return new BIb_C
    
    *BIb_C = BIb_C_new; //INTOUT
    *BIb_C_L_size = my_group_L_size;
    *BIb_C_virtual = my_B_size;
    *BIb_C_occupied = homo;

    // stop the timer
    offload_timestop();
}

/**
 * 
 * ===========PARAMETERS=========
 * // Output arrays
 * double** local_ab,        // local_ab(virtual(ispin), my_B_size(jspin))
 * double** t_ab,            // t_ab(virtual(ispin), my_B_size(jspin))
 * double** local_ba,        // local_ba(virtual(jspin), my_B_size(ispin))
 * 
 * // Input arrays
 * const int* homo,          // homo(2) - occupied orbitals
 * const int* virtual,       // virtual(2) - virtual orbitals
 * const int* my_B_size,     // my_B_size(2) - virtual per process
 * int my_group_L_size,      // L-size for Gamma_P_ia
 * 
 * // Control
 * bool calc_forces,         // Whether forces are being computed
 * int ispin,                // Spin index for i
 * int jspin,                // Spin index for j
 * 
 * // mp2_env%ri_grad arrays (allocated here if not already)
 * double** P_ij,            // mp2_env%ri_grad%P_ij(jspin)%array
 * double** P_ab,            // mp2_env%ri_grad%P_ab(jspin)%array
 * double** Gamma_P_ia,      // mp2_env%ri_grad%Gamma_P_ia(jspin)%array
 * 
 * // Allocation status flags (tracks if arrays already exist)
 * bool* P_ij_allocated,
 * bool* P_ab_allocated,
 * bool* Gamma_P_ia_allocated
 */
void c_mp2_ri_allocate_no_blk(
    double** local_ab, double** t_ab, double** local_ba,
    const int* homo, const int* virtual, const int* my_B_size,
    int my_group_L_size, bool cal_forces, int ispin, int jspin,
    double** P_ij, double** P_ab, double** Gamma_P_ia,
    bool* P_ij_allocated, bool* P_ab_allocated,bool* Gamma_P_ij_allocated 
) {
    //Start timer
    offload_timeset("mp2_ri_allocate_no_blk\0");

    // From fortran index (1) to C (0)
    int i_c = ispin - 1;
    int j_c = jspin - 1;

    // ALLOCATE(local_ab(virtual(ispin), my_B_size(jspin)))
    // local_ab = 0.0_dp
    *local_ab = (double*)calloc((size_t)virtual[i_c] * my_B_size[j_c], sizeof(double));

    //stopo timer
    offload_timestop();
}

/**
 * SUBROUTINE mp2_ri_get_block_size(mp2_env, para_env, para_env_sub, gd_array, gd_B_virtual, &
 *                                  homo, virtual, dimen_RI, unit_nr, &
 *                                  block_size, ngroup, num_integ_group, &
 *                                  my_open_shell_ss, calc_forces, buffer_1D)
 * 
 * -----------------------------------------------------------------------------------------
 * |            FORTRAN-SIDE               |                   C-SIDE                      |
 * ----------------------------------------------------------------------------------------|
 * |   (VAR TYPE) | (INTENT)               | (VAR TYPE)      |(INTENT) | (VAR NAME)        |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (TYPE)     | (IN) mp2_env           |                 |         |                   |
 * |    ====>  mp2_env%ri_mp2%block_size   | (int)           |         | (user_block_size) |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (TYPE)     | (IN) para_env          | (cp_mpi_comm_t) |         | (para_env)        |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (TYPE)     | (IN) para_env_sub      | (cp_mpi_comm_t) |         | (para_env_sub)    |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (TYPE)     | (IN) gd_array          | (int*)          | (const) |                   |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (TYPE(:))  | (IN) gd_B_virtual      | (int*)          | (const) |                   |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (INT(:))   | (IN) homo              | (int*)          | (const) | (homo)            |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (INT(:))   | (IN) virtual           | (int*)          | (const) | (virtual_arr)     |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (INT)      | (IN) dimen_RI          | (int)           |         | (dimen_RI)        |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (INT)      | (IN) unit_nr           | (int)           |         | (unit_nr)         |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (INT)      | (OUT) block_size       | (int*)          | (OUT)   | (block_size)      |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (INT)      | (OUT) ngroup           | (int*)          | (OUT)   | (ngroup_out)      |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (INT)      | (IN) num_integ_group   | (int)           |         | num_integ_group   |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (LOGICAL)  | (IN) my_open_shell_ss  | (bool)          |         | my_open_shell_ss  |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (LOGICAL)  | (IN) calc_forces       | (bool)          |         | calc_forces       |
 * ---------------|------------------------|-----------------|---------|-------------------|
 * |   (REAL(:))  | (OUT) buffer_1D        | (double*)       | (OUT)   | (buffer_1D)       |
 * -----------------------------------------------------------------------------------------
 * 
 * // OUTPUTS
 * int* block_size,
 * int* ngroup_out,
 * double** buffer_1D,
 * 
 * // FROM mp2_env
 * int user_block_size,
 * 
 * // COMMUNICATORS
 * cp_mpi_comm_t para_env_comm,
 * cp_mpi_comm_t para_env_sub_comm,
 * 
 * // FROM gd_array
 * const int* gd_array_sizes,
 * int gd_array_sizes_size,
 * int maxsize_gd_array,
 * 
 * // FROM gd_B_virtual
 * const int* gd_B_virtual_sizes,
 * int gd_B_virtual_sizes_size,
 * int maxsize_gd_B_virtual,
 * int maxval_gd_B_virtual,
 * 
 * // FROM homo
 * const int* homo,
 * int homo_size,
 * 
 * // FROM virtual
 * const int* virtual_arr,
 * int virtual_size,
 * int maxval_virtual,
 * 
 * int dimen_RI,
 * int unit_nr,
 * int num_integ_group,
 * bool my_open_shell_ss,
 * bool calc_forces
 */
void c_mp2_ri_get_block_size(
    int* block_size, int* ngroup_out, double** buffer_1D,
    double mp2_memory, int user_block_size, cp_mpi_comm_t para_env_comm,
    cp_mpi_comm_t para_env_sub_comm, const int* gd_array_sizes, int gd_array_sizes_size,
    int maxsize_gd_array, const int* gd_B_virtual_sizes, int gd_B_virtual_sizes_size,
    int maxsize_gd_B_virtual, int maxval_gd_B_virtual, const int* homo,
    int homo_size, const int* virtual_arr, int virtual_size,
    int maxval_virtual, int dimen_RI, int unit_nr,
    int num_integ_group, bool my_open_shell_ss, bool calc_forces
) {
    //Start timer
    offload_timeset("mp2_ri_get_block_size\0");
    // ========================================================================
    // STEP 1: Calculate ngroup
    // ========================================================================
    int para_env_size = cp_mpi_comm_size(para_env_comm);
    int para_env_sub_size = cp_mpi_comm_size(para_env_sub_comm);
    int ngroup = para_env_size / para_env_sub_size;
    *ngroup_out = ngroup;

    // ========================================================================
    // STEP 2: Get available memory
    // ========================================================================
    int64_t mem_bytes = 0;
    // In fortran-side is call m_memory()
    double mem_real = (double)((mem_bytes + 1024*1024 - 1) / (1024*1024));
    cp_mpi_max_double(&mem_real, 1, para_env_comm);
    
    // ========================================================================
    // STEP 3: Calculate memory components
    // ========================================================================
    double mem_base = 0.0;
    double mem_per_blk = 0.0;
    double mem_per_repl_blk = 0.0;
    
    // external_ab
    // (condiction) ? true : flase
    int max_dim = (dimen_RI > maxval_virtual) ? dimen_RI : maxval_virtual;
    mem_base += (double)maxval_gd_B_virtual * max_dim * 8.0 / (1024.0 * 1024.0);
    
    // BIB_C_rec
    mem_per_repl_blk += (double)maxval_gd_B_virtual * maxsize_gd_array * 8.0 / (1024.0 * 1024.0);
    
    // local_i_aL + local_j_aL
    mem_per_blk += 2.0 * maxval_gd_B_virtual * (double)dimen_RI * 8.0 / (1024.0 * 1024.0);
    
    // Copy to keep arrays contiguous
    mem_base += (double)maxval_gd_B_virtual * max_dim * 8.0 / (1024.0 * 1024.0);

    // ========================================================================
    // STEP 4: Determine block size
    // ========================================================================
    int best_block_size = 1;
    
    if (user_block_size > 0) {
        best_block_size = user_block_size;
    } else {
        double denominator = mem_per_blk + mem_per_repl_blk * ngroup / num_integ_group;
        if (denominator > 0.0) {
            best_block_size = (int)((mem_real - mem_base) / denominator);
        }
        if (best_block_size < 1) best_block_size = 1;
        
        // Loop to ensure valid block size
        while (1) {
            int num_IJ_blocks = 0;
            if (homo_size == 1) {
                if (!my_open_shell_ss) {
                    num_IJ_blocks = homo[0] / best_block_size;
                    num_IJ_blocks = (num_IJ_blocks * num_IJ_blocks - num_IJ_blocks) / 2;
                } else {
                    num_IJ_blocks = (homo[0] - 1) / best_block_size;
                    num_IJ_blocks = (num_IJ_blocks * num_IJ_blocks - num_IJ_blocks) / 2;
                }
            } else {
                num_IJ_blocks = 1;
                for (int i = 0; i < homo_size; i++) {
                    num_IJ_blocks *= (homo[i] / best_block_size);
                }
            }
            
            if ((num_IJ_blocks >= ngroup && num_IJ_blocks > 0) || best_block_size == 1) {
                break;
            } else {
                best_block_size--;
            }
        }
        
        if (homo_size == 1) {
            if (my_open_shell_ss) {
                int sqrt_val = (int)sqrt((double)(homo[0] - 1));
                best_block_size = (sqrt_val < best_block_size) ? sqrt_val : best_block_size;
            } else {
                int sqrt_val = (int)sqrt((double)homo[0]);
                best_block_size = (sqrt_val < best_block_size) ? sqrt_val : best_block_size;
            }
        }
    }
    
    *block_size = (best_block_size < 1) ? 1 : best_block_size;
    
    // ========================================================================
    // STEP 5: Print info if unit_nr > 0
    // ========================================================================
    if (unit_nr > 0) {
        printf("RI_INFO| Block size: %6d\n", *block_size);
        fflush(stdout);
    }

    // ========================================================================
    // STEP 6: Allocate buffer
    // ========================================================================
    int64_t buffer_size = 0;
    int64_t size1 = (int64_t)maxsize_gd_array * (*block_size);
    int64_t size2 = (int64_t)max_dim;
    int64_t max_size = (size1 > size2) ? size1 : size2;
    buffer_size = max_size * maxval_gd_B_virtual;
    if (calc_forces) buffer_size *= 2;
    
    *buffer_1D = (double*)malloc((size_t)buffer_size * sizeof(double));
    if (*buffer_1D == NULL) {
        fprintf(stderr, "Error: Failed to allocate buffer_1D of size %ld\n", buffer_size);
        exit(1);
    }

    // Stop timer
    offload_timestop();
}


/**
 * * 
 * 
 * SUBROUTINE mp2_ri_communication(my_alpha_beta_case, total_ij_pairs, homo, homo_beta, &
                                   block_size, ngroup, ij_map, color_sub, my_ij_pairs, my_open_shell_SS, unit_nr)
      LOGICAL, INTENT(IN)                                :: my_alpha_beta_case
      INTEGER, INTENT(OUT)                               :: total_ij_pairs
      INTEGER, INTENT(IN)                                :: homo, homo_beta, block_size, ngroup
      INTEGER, ALLOCATABLE, DIMENSION(:, :), INTENT(OUT) :: ij_map
      INTEGER, INTENT(IN)                                :: color_sub
      INTEGER, INTENT(OUT)                               :: my_ij_pairs
      LOGICAL, INTENT(IN)                                :: my_open_shell_SS
      INTEGER, INTENT(IN)                                :: unit_nr
 * ____________________________________________________________________________________________________________
 * |                  FORTRAN-SIDE                   ||                   C-SIDE                              |
 * |_________________________________________________||_______________________________________________________|
 * |   (VAR TYPE) |(INTENT) |       (VAR NAME)       ||    (VAR TYPE)      |(INTENT) |      (VAR NAME)        |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (LOGICAL)  |   (IN)  |   my_alpha_beta_case   ||       bool         |         |   my_alpha_beta_case   |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (INTEGER)  |  (OUT)  |     total_ij_pairs     ||       int*         |         |      total_ij_pairs    |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (INTEGER)  |   (IN)  |         homo           ||       int          |         |          homo          |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (INTEGER)  |   (IN)  |       homo_beta        ||       int          |         |        homo_beta       |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (INTEGER)  |   (IN)  |      block_size        ||       int          |         |       block_size       |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (INTEGER)  |   (IN)  |        ngroup          ||       int          |         |         ngroup         |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |  (INTEGER )  |         |                        ||                    |         |                        |
 * |  (DIM(:,:))  |  (IN)   |         ij_map         ||       int**        |  malloc |       ij_map           |
 * | (ALLOCATABLE)|         |                        ||                    |         |                        |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (INTEGER)  |   (IN)  |       color_sub        ||       int          |         |       color_sub        |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (INTEGER)  |  (OUT)  |     my_ij_pairs        ||       int*         |         |      my_ij_pairs       |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (LOGICAL)  |   (IN)  |   my_open_shell_SS     ||       int          |         |    my_open_shell_SS    |
 * |______________|_________|________________________||____________________|_________|________________________|
 * |   (INTEGER)  |   (IN)  |       unit_nr          ||       int          |         |        unit_nr         |
 * |______________|_________|________________________||____________________|_________|________________________|
 * 
 * // INPUTS
 * bool my_alpha_beta_case   // Alpha-beta case flag
 * int homo                  // Number of occupied orbitals (spin 1)
 * int homo_beta             // Number of occupied orbitals (spin 2)
 * int block_size            // Block size for ij pairs
 * int ngruop                // Number of groups
 * int color_sub             // Process color in subgroup
 * bool my_open_shell_SS     // Open shell same-spin flag
 * int unit_nr               // Output unit number
 * 
 * //OUTPUT
 * int* total_ij_pairs       // Total number of ij pairs
 * int** ij_map              // Map fo ij pairs (3 x total)
 * int* my_ij_pairs          // Number of pairs for this process
 */
void c_mp2_ri_communication(
    bool my_alpha_beta_case, int homo, int homo_beta,
    int block_size, int ngroup, int color_sub,
    bool my_open_shell_SS, int unit_nr, int* total_ij_pairs,
    int** ij_map, int* my_ij_pairs
){
    // start timer
    offload_timeset("mp2_ri_communication\0");

    *total_ij_pairs = homo * (1 + homo) / 2;
    int num_IJ_blocks = homo / block_size - 1;

    int first_I_block = 1;
    int last_i_block = block_size * (num_IJ_blocks - 1);
    int first_J_block = block_size + 1;
    int last_J_block = block_size * (num_IJ_blocks + 1);

    /**
     * 
         ij_block_counter = 0
         DO iiB = first_I_block, last_i_block, block_size
            DO jjB = iiB + block_size, last_J_block, block_size
               ij_block_counter = ij_block_counter + 1
            END DO
         END DO
     */
    // Count block pairs
    int ij_block_counter = 0;
    for (int iiB = first_I_block; iiB <= last_i_block; iiB += block_size) {
        for (int jjB = iiB + block_size + 1; jjB <= last_J_block; jjB += block_size) {
            ij_block_counter++;
        }
    }

    int total_ij_block = ij_block_counter;
    int num_block_per_group = total_ij_block / ngroup;
    int assigned_blocks = num_block_per_group * ngroup;
    int total_ij_pairs_blocks = assigned_blocks + (total_ij_pairs - assigned_blocks * (block_size * block_size));

    // ALLOCATE (ij_marker(homo, homo))
    // ij_marker = .TRUE.
    // array row-major flattened 1D
    // According with some forums benefits by memory, single allocation and dynamic access 
    // bool* arr = malloc(rows * cols * sizeof(bool));
    // to access: arr[i * cols + j]
    bool* ij_marker = (bool*)malloc(homo * homo * sizeof(bool));
    for (int i = 0; i < homo * homo; i++) {
        ij_marker[i] = true;
    }

    // ALLOCATE (ij_map(3, total_ij_pairs_blocks))
    // ij_map = 0
    *ij_map = (int*)calloc(3 * total_ij_pairs_blocks, sizeof(int));

    int ij_counter = 0;
    *my_ij_pairs = 0;

    /**
     * 
         DO iiB = first_I_block, last_i_block, block_size
            DO jjB = iiB + block_size, last_J_block, block_size
               IF (ij_counter + 1 > assigned_blocks) EXIT
               ij_counter = ij_counter + 1
               ij_marker(iiB:iiB + block_size - 1, jjB:jjB + block_size - 1) = .FALSE.
               ij_map(1, ij_counter) = iiB
               ij_map(2, ij_counter) = jjB
               ij_map(3, ij_counter) = block_size
               IF (MOD(ij_counter, ngroup) == color_sub) my_ij_pairs = my_ij_pairs + 1
            END DO
         END DO
     */
    for (int iiB = first_I_block; iiB <= last_i_block; iiB += block_size) {
        for (int jjB = iiB + block_size; jjB <= last_J_block; jjB += block_size) {
            // exit
            if (ij_counter + 1 > assigned_blocks) {break;}
            ij_counter++;

            // ij_marker(iiB:iiB + block_size - 1, jjB:jjB + block_size - 1) = .FALSE.
            // i = iiB - 1 (index 0 in C)
            for (int i = iiB - 1; i < iiB + block_size - 1; i++) {
                // j = jjB - 1 (index 0 in C)
                for (int j = jjB - 1; j < jjB + block_size - 1; j++) {
                    ij_marker[i * homo + j] = false;
                }

                (*ij_map)[0 * total_ij_pairs_blocks + (ij_counter - 1)] = iiB;
                (*ij_map)[1 * total_ij_pairs_blocks + (ij_counter - 1)] = jjB;
                (*ij_map)[2 * total_ij_pairs_blocks + (ij_counter - 1)] = block_size;

                if ((ij_block_counter % ngroup) == color_sub) {
                    (*my_ij_pairs)++;
                }
            }

            /**
             * 
         DO iiB = 1, homo
            DO jjB = iiB, homo
               IF (ij_marker(iiB, jjB)) THEN
                  ij_counter = ij_counter + 1
                  ij_map(1, ij_counter) = iiB
                  ij_map(2, ij_counter) = jjB
                  ij_map(3, ij_counter) = 1
                  IF (MOD(ij_counter, ngroup) == color_sub) my_ij_pairs = my_ij_pairs + 1
               END IF
            END DO
         END DO
         DEALLOCATE (ij_marker)
             */
            for (int iiB = 1; iiB <= homo; iiB++) {
                for (int jjB = iiB; jjB <= homo; jjB++) {
                    // to access: arr[i * cols + j]
                    // 0-based in C-stlr
                    if (ij_marker[(iiB - 1) * homo + (jjB - 1)]) {
                        ij_counter++;
                        (*ij_map)[0 * total_ij_pairs_blocks + (ij_counter -1)] = iiB;
                        (*ij_map)[1 * total_ij_pairs_blocks + (ij_counter -1)] = jjB;
                        (*ij_map)[3 * total_ij_pairs_blocks + (ij_counter -1)] = 1;

                        if ((ij_counter % ngroup) == color_sub) {
                            (*my_ij_pairs)++;
                        }
                    }
                }
            }
            free(ij_marker);
        }
    }
    
    
    /**
     * ============================ Should I also add this part?
         IF (unit_nr > 0) THEN
         IF (block_size == 1) THEN
            WRITE (UNIT=unit_nr, FMT="(T3,A,T66,F15.1)") &
               "RI_INFO| Percentage of ij pairs communicated with block size 1:", 100.0_dp
         ELSE
            WRITE (UNIT=unit_nr, FMT="(T3,A,T66,F15.1)") &
               "RI_INFO| Percentage of ij pairs communicated with block size 1:", &
               100.0_dp*REAL((total_ij_pairs - assigned_blocks*(block_size**2)), KIND=dp)/REAL(total_ij_pairs, KIND=dp)
         END IF
         CALL m_flush(unit_nr)
      END IF
     */

    // Stop timer
    offload_timestop();
}