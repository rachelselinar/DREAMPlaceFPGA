/**
 * @file   lut_ff_legalization_cuda_kernel.cu
 * @author Rachel Selina (DREAMPlaceFPGA-PL)
 * @date   Mar 2021
 * @brief  Legalize LUT/FF
 */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include "assert.h"
#include "cuda_runtime.h"
#include "utility/src/print.cuh"
#include "utility/src/utils.cuh"
#include "utility/src/limits.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>

DREAMPLACE_BEGIN_NAMESPACE

#define THREAD_COUNT 64
#define INVALID -1
#define INPUT_PIN 1
#define PQ_IDX 10
//Below values are fixed for ISPD'2016 benchmarks
#define NUM_LUT6_INPUTS 6
#define LUT_MAXSHARED_INPUTS 5
#define SLICE_CAPACITY 16
#define BLE_CAPACITY 2
#define N 16
#define M 24
#define M2 576

///helper functions for Edmonds Blossom Implementation based on https://codeforces.com/blog/entry/92339
inline __device__ void queue_insert(const int Q_size, int &Q_front, int &Q_back, int* Q, const int element)
{
    if (Q_back == Q_size-1)
    {
        printf("ERROR: QUEUE OVERFLOW - INCREASE SIZE\n");
    } else
    {
        if (Q_front == INVALID)
        {
            Q_front = 0;
        }
        if (Q_back == INVALID)
        {
            Q_back = 0;
        } else
        {
            Q_back += 1;
        }
        Q[Q_back] = element;
    }
}

inline __device__ void queue_pop(int &Q_front, const int Q_back)
{
    if (Q_front == INVALID || Q_front > Q_back)
    {
        printf("WARN: QUEUE UNDERFLOW\n");
    } else
    {
        ++Q_front;
    }

}

inline __device__ void add_edge(const int u, const int v, int* g)
{
    g[u*M+v] = u;
    g[v*M+u] = v;
}

inline __device__ void match(const int u, const int v, int* g, int* mate)
{
    g[u*M+v] = INVALID;
    g[v*M+u] = INVALID;
    mate[u] = v;
    mate[v] = u;
}

//Note: x should not be changed outside the function!
inline __device__ void trace(int x, const int* bl, const int* p, int* vx, int &vx_length)
{
    while(true)
    {
        while(bl[x] != x) x = bl[x];
        if(vx_length > 0 && vx[vx_length - 1] == x) break;
        vx[vx_length] = x;
        ++vx_length;
        x = p[x];
    }
}

__device__ void contract(const int c, int x, int y, int* vx, int &vx_length, int* vy, int &vy_length, int* b, int* bIndex, int* bl, int* g)
{
    bIndex[c] = 0;
    int r = vx[vx_length - 1];
    while(vx_length > 0 && vy_length > 0 && vx[vx_length - 1] == vy[vy_length - 1])
    {
        r = vx[vx_length - 1];
        --vx_length;
        --vy_length;
    }
    // b[c].push_back(r);
    b[c * M + bIndex[c]] = r;
    ++bIndex[c];
    
    // b[c].insert(b[c].end(), vx.rbegin(), vx.rend());
    for (int i = vx_length - 1; i >= 0; --i) {
        b[c * M + bIndex[c]] = vx[i];
        ++bIndex[c]; 
    }
    
    // b[c].insert(b[c].end(), vy.begin(), vy.end());
    for (int i = 0; i < vy_length; ++i)
    {
        b[c * M + bIndex[c]] = vy[i];
        ++bIndex[c];
    }

    for(int i = 0; i <= c; ++i)
    {
        g[c*M+i] = INVALID;
        g[i*M+c] = INVALID;
    }

    for (int j = 0; j < bIndex[c]; ++j)
    {
        int z = b[c * M + j];
        bl[z] = c;
        for(int i = 0; i < c; ++i)
        {
            if(g[z*M+i] != INVALID) 
            {
                g[c*M+i] = z;
                g[i*M+c] = g[i*M+z];
            }
        }
    }
}

__device__ void lift(const int n, const int* g, const int* b, const int* bIndex, int* vx, int &vx_length, int* A, int &A_length)
{
    while (vx_length >= 2)
    {
        int z = vx[vx_length-1];
        --vx_length;
        if (z < n)
        {
            A[A_length] = z;
            ++A_length;
            continue;
        }
        int w = vx[vx_length-1];
        int i = 0;
        if (A_length % 2 == 0)
        {
            //Find index of g[z][w] within b[z]
            int val = g[z*M+w];
            for (int bId = 0; bId < bIndex[z]; ++bId)
            {
                if (b[z*M+bId] == val)
                {
                    i = bId;
                    break;
                }
            }
        }
        int j = 0;
        if (A_length % 2 == 1)
        {
            //Find index of g[z][A.back()] within b[z]
            int val = g[z*M+A[A_length-1]];
            for (int bId = 0; bId < bIndex[z]; ++bId)
            {
                if (b[z*M+bId] == val)
                {
                    j = bId;
                    break;
                }
            }
        }
        int k = bIndex[z];
        int dif = (A_length % 2 == 0 ? i%2 == 1 : j%2 == 0) ? 1 : k-1;

        while(i != j)
        {
            vx[vx_length] = b[z*M+i];
            ++vx_length;
            i = (i + dif) % k;
        }
        vx[vx_length] = b[z*M+i];
        ++vx_length;
    }
}

///End of helper functions for Edmonds Blossom Implementation based on https://codeforces.com/blog/entry/92339

//Clear entries in candidate
inline __device__ void clear_cand_contents(const int tsPQ, const int SIG_IDX,
        const int CKSR_IN_CLB, const int CE_IN_CLB, int* site_sig_idx, int* site_sig,
        int* site_impl_lut, int* site_impl_ff, int* site_impl_cksr, int* site_impl_ce)
{
    int topIdx(tsPQ*SIG_IDX);
    int lutIdx = tsPQ*SLICE_CAPACITY;
    int ckIdx = tsPQ*CKSR_IN_CLB;
    int ceIdx = tsPQ*CE_IN_CLB;

    for(int sg = 0; sg < SIG_IDX; ++sg)
    {
        site_sig[topIdx + sg] = INVALID;
    }
    for(int sg = 0; sg < SLICE_CAPACITY; ++sg)
    {
        site_impl_lut[lutIdx + sg] = INVALID;
    }
    for(int sg = 0; sg < SLICE_CAPACITY; ++sg)
    {
        site_impl_ff[lutIdx + sg] = INVALID;
    }
    for(int sg = 0; sg < CKSR_IN_CLB; ++sg)
    {
        site_impl_cksr[ckIdx + sg] = INVALID;
    }
    for(int sg = 0; sg < CE_IN_CLB; ++sg)
    {
        site_impl_ce[ceIdx + sg] = INVALID;
    }
}

//Check if entry exists in array
inline __device__ bool val_in_array(const int *array, const int arraySize, const int arrayIdx, const int val)
{
    for (int idx = 0; idx < arraySize; ++idx)
    {
        if (array[arrayIdx+idx] == val)
        {
            return true;
        }
    }
    return false;
}

/// define candidate_validity_check
inline __device__ bool candidate_validity_check(const int topIdx, const int siteCurrPQSigIdx, const int siteCurrPQSiteId, const int* site_curr_pq_sig, const int* inst_curr_detSite)
{
    for (int i = 0; i < siteCurrPQSigIdx; ++i)
    {
        int pqInst = site_curr_pq_sig[topIdx + i];

        if (inst_curr_detSite[pqInst] != INVALID && 
            inst_curr_detSite[pqInst] != siteCurrPQSiteId)
        {
            return false;
        }
    }
    return true;
}

/// define check_sig_in_site_next_pq_sig
inline __device__ bool check_sig_in_site_next_pq_sig(const int* nwCand_sig, const int nwCand_sigIdx, const int siteId, const int* site_next_pq_validIdx, const int* site_next_pq_sig, const int* site_next_pq_sig_idx, const int* site_next_pq_idx, const int SIG_IDX)
{
    int sPQ = siteId*PQ_IDX;
    int cnt = 0;
    for (int i = 0; i < PQ_IDX; ++i)
    {
        int sigIdx = sPQ + i;

        if (site_next_pq_validIdx[sigIdx] != INVALID)
        {
            if (site_next_pq_sig_idx[sigIdx] == nwCand_sigIdx)
            {
                int pqIdx(sigIdx*SIG_IDX), mtch(0);
                for (int k = 0; k < nwCand_sigIdx; ++k)
                {
                    for (int l = 0; l < nwCand_sigIdx; ++l)
                    {
                        if (site_next_pq_sig[pqIdx + l] == nwCand_sig[k])
                        {
                            ++mtch;
                            break;
                        }
                    }
                }
                if (mtch == nwCand_sigIdx)
                {
                    return true;
                }
            }
            ++cnt;
            if (cnt == site_next_pq_idx[siteId])
            {
                break;
            }
        }
    }
    return false;
}

////SUBFUCTIONS////
inline __device__ bool add_flop_to_candidate_impl(const int ffCKSR, const int ffCE, const int ffId, const int HALF_SLICE_CAPACITY, const int CKSR_IN_CLB, int* res_ff, int* res_cksr, int* res_ce)
{
    for (int i = 0; i < CKSR_IN_CLB; ++i)
    {
        if (res_cksr[i] == INVALID || res_cksr[i] == ffCKSR)
        {
            for (int j = 0; j < CKSR_IN_CLB; ++j)
            {
                int ceIdx = CKSR_IN_CLB*i + j;
                if (res_ce[ceIdx] == INVALID || res_ce[ceIdx] == ffCE)
                {
                    int beg = i*HALF_SLICE_CAPACITY+j;
                    int end = beg + HALF_SLICE_CAPACITY;
                    for (int k = beg; k < end; k += BLE_CAPACITY)
                    {
                        if (res_ff[k] == INVALID)
                        {
                            res_ff[k] = ffId;
                            res_cksr[i] = ffCKSR;
                            res_ce[ceIdx] = ffCE;
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}

// define remove_invalid_neighbor
inline __device__ void remove_invalid_neighbor(const int sIdx, const int sNbrIdx, int* site_nbr_idx, int* site_nbr)
{
    int temp[512];
    int tempSize(0);
    for (int i = 0; i < site_nbr_idx[sIdx]; ++i)
    {
        int instId = site_nbr[sNbrIdx + i];

        if (instId != INVALID)
        {
            temp[tempSize] = instId;
            ++tempSize;
        }
    }

    for (int j = 0; j < tempSize; ++j)
    {
        site_nbr[sNbrIdx+j] = temp[j];
    }
    for (int j = tempSize; j < site_nbr_idx[sIdx]; ++j)
    {
        site_nbr[sNbrIdx+j] = INVALID;
    }
    site_nbr_idx[sIdx] = tempSize;

    //DBG
    if (tempSize > 500)
    {
        printf("WARN: remove_invalid_neighbor() has tempSize > 500 for site: %d\n", sIdx);
    }
    //DBG
}

//two lut compatibility
inline __device__ bool two_lut_compatibility_check(const int* lut_type, const int* flat_node2pin_start_map, const int* flat_node2pin_map, const int* pin2net_map, const int* pin_typeIds, const int* sorted_net_map, const int lutA, const int lutB)
{
    if(lut_type[lutA] == NUM_LUT6_INPUTS || lut_type[lutB] == NUM_LUT6_INPUTS)
    {
        return false;
    }

    int numInputs = lut_type[lutA] + lut_type[lutB];

    if (numInputs <= LUT_MAXSHARED_INPUTS)
    {
        return true;
    }

    int lutAIt = flat_node2pin_start_map[lutA];
    int lutBIt = flat_node2pin_start_map[lutB];
    int lutAEnd = flat_node2pin_start_map[lutA+1];
    int lutBEnd = flat_node2pin_start_map[lutB+1];

    if (pin_typeIds[flat_node2pin_map[lutAIt]] != INPUT_PIN)
    {
        ++lutAIt;
    }
    if (pin_typeIds[flat_node2pin_map[lutBIt]] != INPUT_PIN)
    {
        ++lutBIt;
    }

    int netIdA = pin2net_map[flat_node2pin_map[lutAIt]];
    int netIdB = pin2net_map[flat_node2pin_map[lutBIt]];

    while(numInputs > LUT_MAXSHARED_INPUTS)
    {
        if (sorted_net_map[netIdA] < sorted_net_map[netIdB])
        {
            ++lutAIt;
            if (pin_typeIds[flat_node2pin_map[lutAIt]] != INPUT_PIN)
            {
                ++lutAIt;
            }
            if (lutAIt < lutAEnd)
            {
                netIdA = pin2net_map[flat_node2pin_map[lutAIt]];
            } else
            {
                break;
            }
        } else if (sorted_net_map[netIdA] > sorted_net_map[netIdB])
        {
            ++lutBIt;
            if (pin_typeIds[flat_node2pin_map[lutBIt]] != INPUT_PIN)
            {
                ++lutBIt;
            }
            if (lutBIt < lutBEnd)
            {
                netIdB = pin2net_map[flat_node2pin_map[lutBIt]];
            } else
            {
                break;
            }

        } else
        {
            --numInputs;
            ++lutAIt;
            ++lutBIt;
            if (pin_typeIds[flat_node2pin_map[lutAIt]] != INPUT_PIN)
            {
                ++lutAIt;
            }
            if (pin_typeIds[flat_node2pin_map[lutBIt]] != INPUT_PIN)
            {
                ++lutBIt;
            }

            if (lutAIt < lutAEnd && lutBIt < lutBEnd)
            {
                netIdA = pin2net_map[flat_node2pin_map[lutAIt]];
                netIdB = pin2net_map[flat_node2pin_map[lutBIt]];
            } else
            {
                break;
            }
        }
    }

    return numInputs <= LUT_MAXSHARED_INPUTS;
}

//fitLUTsToCandidateImpl
inline __device__ bool fit_luts_to_candidate_impl(const int* lut_type, const int* pin2net_map, const int* pin_typeIds, const int* flat_node2pin_start_map, const int* flat_node2pin_map, const int* sorted_net_map, const int* flat_node2precluster_map, const int* node2fence_region_map, const int instPcl, const int node2prclstrCount, const int NUM_BLE_PER_SLICE, int* res_lut)
{
    int luts[N], lut6s[N];
    int lutIdx(0), lut6Idx(0);

    for (int sg = 0; sg < SLICE_CAPACITY; ++sg)
    {
        if (res_lut[sg] != INVALID)
        {
            if (lut_type[res_lut[sg]] < NUM_LUT6_INPUTS)
            {
                luts[lutIdx] = res_lut[sg];
                ++lutIdx;
            } else
            {
                lut6s[lut6Idx] = res_lut[sg];
                ++lut6Idx;
            }
        }
    }

    //int lutSize = lutIdx + lut6Idx;
    for (int idx = 0; idx < node2prclstrCount; ++idx)
    {
        int clInstId = flat_node2precluster_map[instPcl + idx];
        if (node2fence_region_map[clInstId] == 0)
        {
            if (lut_type[clInstId] < NUM_LUT6_INPUTS)
            {
                luts[lutIdx] = clInstId;
                ++lutIdx;
                ////Remove redundant entries
                for (int i = 0; i < lutIdx; ++i)
                {
                    for(int j = 0; j <i; ++j)
                    {
                        if (luts[i] == luts[j])
                        {
                            --lutIdx;
                            for (int k = 0; k < lutIdx; ++k)
                            {
                                luts[k] = luts[k+1];
                            }
                            --i;
                        }
                    }
                }
            } else
            {
                lut6s[lut6Idx] = clInstId;
                ++lut6Idx;
                ////Remove redundant entries
                for (int i = 0; i < lut6Idx; ++i)
                {
                    for(int j = 0; j <i; ++j)
                    {
                        if (lut6s[i] == lut6s[j])
                        {
                            --lut6Idx;
                            for (int k = 0; k < lut6Idx; ++k)
                            {
                                lut6s[k] = lut6s[k+1];
                            }
                            --i;
                        }
                    }
                }

            }
        }
    }

    //graph matching can be called even if res_lut if full!
    //Guard band for graph matching implementation with fixed memory
    if (lutIdx + lut6Idx > SLICE_CAPACITY)
    {
        return false;
    }

    ///Edmonds Blossom Implementation based on https://codeforces.com/blog/entry/92339
    int n = lutIdx; //n - #vertices
    //int m = (n%2 == 0) ? 3*n/2: 3*(n+1)/2; //m = 3n/2

    int mate[N]; //array of length n; For each vertex u, if exposed mate[u] = -1 or mate[u] = u
    int b[M2]; //For each blossom u, b[u] is list of all vertices contracted from u
    int bIndex[M];
    int p[M]; //array of length m; For each vertex/blossom u, p[u] is parent in the search forest
    int d[M]; //array of length m; For each vertex u, d[u] is status in search forest. d[u] = 0 if unvisited, d[u] = 1 is even depth from root and d[u] = 2 is odd depth from root
    int bl[M]; //array of length m; For each vertex/blossom u, bl[u] is the blossom containing u. If not contracted, bl[u] = u. 
    int g[M2]; //table of size mxm with information of unmatched edges.g[u][v] = -1 if no unmatched vertices; g[u][v] = u, if u is a vertex.

    //Initialize mate
    for (int mId = 0; mId < n; ++mId)
    {
        mate[mId] = INVALID;
    }
    for (int gId = 0; gId < M2; ++gId)
    {
        g[gId] = INVALID;
    }

    //Create graph with luts
    for(int ll = 0; ll < lutIdx; ++ll)
    {
        for(int rl = ll+1; rl < lutIdx; ++rl)
        {
            if (two_lut_compatibility_check(lut_type, flat_node2pin_start_map, flat_node2pin_map, pin2net_map, pin_typeIds, sorted_net_map, luts[ll], luts[rl]))
            {
                add_edge(ll, rl, g);
            }
        }
    }

    int totalPairs(0);

    for (int ans = 0; ; ++ans)
    {
        for (int dId = 0; dId < M; ++dId)
        {
            d[dId] = 0;
        }

        int Q[32];
        int Q_size(32);
        int Q_front(INVALID), Q_back(INVALID);

        for (int i = 0; i < M; ++i)
        {
            bl[i] = i;
        }
        for (int i = 0; i < n; ++i)
        {
            if (mate[i] == INVALID)
            {
                queue_insert(Q_size, Q_front, Q_back, Q, i);
                p[i] = i;
                d[i] = 1;
            }
        }

        int c = N;
        bool aug(false);

        while ((Q_front != INVALID && Q_front <= Q_back) && !aug)
        {
            int x = Q[Q_front];
            //queue_pop(Q_front, Q_back, Q);
            queue_pop(Q_front, Q_back);

            if (bl[x] != x) continue;

            for (int y = 0; y < c; ++y)
            {
                if (bl[y] == y && g[x*M+y] != INVALID)
                {
                    if (d[y] == 0)
                    {
                        p[y] = x;
                        d[y] = 2;
                        p[mate[y]] = y;
                        d[mate[y]] = 1;
                        queue_insert(Q_size, Q_front, Q_back, Q, mate[y]);
                    } else if (d[y] == 1)
                    {
                        int vx[2*M], vy[2*M];
                        int vx_length = 0, vy_length = 0;
                        trace(x, bl, p, vx, vx_length);
                        trace(y, bl, p, vy, vy_length);

                        if (vx[vx_length-1] == vy[vy_length-1])
                        {
                            contract(c, x, y, vx, vx_length, vy, vy_length, b, bIndex, bl, g);
                            queue_insert(Q_size, Q_front, Q_back, Q, c);
                            p[c] = p[b[c*M]];
                            d[c] = 1;
                            ++c;
                        } else
                        {
                            aug = true;
                            int new_vx[2*M], new_vy[2*M];
                            new_vx[0] = y;
                            for (int idx = 0; idx < vx_length; ++idx)
                            {
                                new_vx[idx+1] = vx[idx];
                            }
                            ++vx_length;
                            new_vy[0] = x;
                            for (int idx = 0; idx < vy_length; ++idx)
                            {
                                new_vy[idx+1] = vy[idx];
                            }
                            ++vy_length;

                            int A[4*M], B[2*M];
                            int A_length = 0, B_length = 0;

                            lift(n, g, b, bIndex, new_vx, vx_length, A, A_length);
                            lift(n, g, b, bIndex, new_vy, vy_length, B, B_length);

                            for (int idx = B_length-1; idx >= 0; --idx)
                            {
                                A[A_length] = B[idx];
                                ++A_length;
                            }

                            for (int i = 0; i < A_length; i += 2)
                            {
                                match(A[i], A[i+1], g, mate);
                                if (i + 2 < A_length)
                                {
                                    add_edge(A[i+1], A[i + 2], g);
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }
        if (!aug)
        {
            totalPairs = ans;
            break;
        }
    }

    if (lutIdx - totalPairs + lut6Idx > NUM_BLE_PER_SLICE)
    {
        return false;
    }

    int idxL = 0;

    for (int iil = 0; iil < lut6Idx; ++iil)
    {
        res_lut[idxL] = lut6s[iil];
        res_lut[idxL + 1] = INVALID;
        idxL += BLE_CAPACITY;
    }

    for (int mId = 0; mId < n; ++mId)
    {
        if (mate[mId] == INVALID)
        {
            res_lut[idxL] = luts[mId];
            res_lut[idxL + 1] = INVALID;
            idxL += BLE_CAPACITY;
        }
    }

    int ck[N] = {0};
    for (int mId = 0; mId < n; ++mId)
    {
        if (mate[mId] != INVALID && ck[mId] == 0 && ck[mate[mId]] == 0)
        {
            ++ck[mId];
            ++ck[mate[mId]];

            res_lut[idxL] = luts[mId];
            res_lut[idxL + 1] = luts[mate[mId]];
            idxL += BLE_CAPACITY;
        }
    }
    for (int lIdx = idxL; lIdx < SLICE_CAPACITY; ++lIdx)
    {
        res_lut[lIdx] = INVALID;
    }

    return true;
}
//////////////////
//////////////////

//addLUTToCandidateImpl
inline __device__ bool add_lut_to_cand_impl(const int* lut_type, const int* flat_node2pin_start_map, const int* flat_node2pin_map, const int* pin2net_map, const int* pin_typeIds, const int* sorted_net_map, const int lutId, int* res_lut)
{
    for (int i=0; i < SLICE_CAPACITY; i += BLE_CAPACITY)
    {
        if (res_lut[i] == INVALID)
        {
            res_lut[i] = lutId;
            return true;
        }
    }
    for (int i=1; i < SLICE_CAPACITY; i += BLE_CAPACITY)
    {
        if (res_lut[i] == INVALID && 
            two_lut_compatibility_check(lut_type, flat_node2pin_start_map, flat_node2pin_map, pin2net_map, pin_typeIds, sorted_net_map, res_lut[i-1], lutId))
        {
            res_lut[i] = lutId;
            return true;
        }
    }
    return false;
}

//template <typename T>
__device__ bool is_inst_in_cand_feasible(const int* node2fence_region_map, const int* lut_type, const int* flat_node2pin_start_map, const int* flat_node2pin_map, const int* node2pincount, const int* net2pincount, const int* pin2net_map, const int* pin_typeIds, const int* sorted_net_map, const int* flat_node2prclstrCount, const int* flat_node2precluster_map, const int* flop2ctrlSetId_map, const int* flop_ctrlSets, const int* site_det_impl_lut, const int* site_det_impl_ff, const int* site_det_impl_cksr, const int* site_det_impl_ce,
        const int siteId, const int instId, const int HALF_SLICE_CAPACITY, const int NUM_BLE_PER_SLICE, const int CKSR_IN_CLB, const int CE_IN_CLB)
{
    int instPcl = instId*3;

    int sdlutId = siteId*SLICE_CAPACITY;
    int sdckId = siteId*CKSR_IN_CLB;
    int sdceId = siteId*CE_IN_CLB;

    int res_lut[SLICE_CAPACITY];
    int res_ff[SLICE_CAPACITY];
    int res_cksr[2];
    int res_ce[4];

    for(int sg = 0; sg < SLICE_CAPACITY; ++sg)
    {
        res_lut[sg] = site_det_impl_lut[sdlutId + sg];
        res_ff[sg] = site_det_impl_ff[sdlutId + sg];
    }
    for(int sg = 0; sg < CKSR_IN_CLB; ++sg)
    {
        res_cksr[sg] = site_det_impl_cksr[sdckId + sg];
    }
    for(int sg = 0; sg < CE_IN_CLB; ++sg)
    {
        res_ce[sg] = site_det_impl_ce[sdceId + sg];
    }
    /////

    bool lutFail(false);
    for (int idx = 0; idx < flat_node2prclstrCount[instId]; ++idx)
    {
        int clInstId = flat_node2precluster_map[instPcl + idx];
        int clInstCKSR = flop2ctrlSetId_map[clInstId]*3 + 1;
        int clInstCE = flop2ctrlSetId_map[clInstId]*3 + 2;
        switch(node2fence_region_map[clInstId])
        {
            case 1: //FF
                {
                    if(!add_flop_to_candidate_impl(flop_ctrlSets[clInstCKSR], flop_ctrlSets[clInstCE], clInstId, HALF_SLICE_CAPACITY, CKSR_IN_CLB, res_ff, res_cksr, res_ce))
                    {
                        return false;
                    }
                    break;
                }
            case 0: //LUT
                {
                    if (!lutFail && !add_lut_to_cand_impl(lut_type, flat_node2pin_start_map, flat_node2pin_map, pin2net_map, pin_typeIds, sorted_net_map, clInstId, res_lut))
                    {
                        lutFail = true;
                    }
                    break;
                }
            default:
                {
                    break;
                }
        }
    }
    if (!lutFail)
    {
        return true;
    }

    return fit_luts_to_candidate_impl(lut_type, pin2net_map, pin_typeIds, flat_node2pin_start_map, flat_node2pin_map, sorted_net_map, flat_node2precluster_map, node2fence_region_map, instPcl, flat_node2prclstrCount[instId], NUM_BLE_PER_SLICE, res_lut);
}

inline __device__ bool add_inst_to_cand_impl(const int* lut_type, const int* flat_node2pin_start_map, const int* flat_node2pin_map, const int* node2pincount, const int* net2pincount, const int* pin2net_map, const int* pin_typeIds, const int* sorted_net_map, const int* flat_node2prclstrCount, const int* flat_node2precluster_map, const int* flop2ctrlSetId_map, const int* node2fence_region_map, const int* flop_ctrlSets, const int instId, const int CKSR_IN_CLB, const int CE_IN_CLB, const int HALF_SLICE_CAPACITY, const int NUM_BLE_PER_SLICE, int* nwCand_lut, int* nwCand_ff, int* nwCand_cksr, int* nwCand_ce)
{
    int instPcl = instId*3;

    //array instantiation
    int res_lut[SLICE_CAPACITY];
    int res_ff[SLICE_CAPACITY];
    int res_ce[4];
    int res_cksr[2];

    for (int sg = 0; sg < SLICE_CAPACITY; ++sg)
    {
        res_lut[sg] = nwCand_lut[sg];
        res_ff[sg] = nwCand_ff[sg];
    }
    for (int sg = 0; sg < CKSR_IN_CLB; ++sg)
    {
        res_cksr[sg] = nwCand_cksr[sg];
    }
    for (int sg = 0; sg < CE_IN_CLB; ++sg)
    {
        res_ce[sg] = nwCand_ce[sg];
    }
    /////
    //DBG

    bool lutFail(false);
    for (int idx = 0; idx < flat_node2prclstrCount[instId]; ++idx)
    {
        int clInstId = flat_node2precluster_map[instPcl + idx];
        int clInstCKSR = flop2ctrlSetId_map[clInstId]*3 + 1;
        int clInstCE = flop2ctrlSetId_map[clInstId]*3 + 2;
        switch(node2fence_region_map[clInstId])
        {
            case 1: //FF
                {
                    if(!add_flop_to_candidate_impl(flop_ctrlSets[clInstCKSR], flop_ctrlSets[clInstCE], clInstId, HALF_SLICE_CAPACITY, CKSR_IN_CLB, res_ff, res_cksr, res_ce))
                    {
                        return false;
                    }
                    break;
                }
            case 0: //LUT
                {
                    if (!lutFail && !add_lut_to_cand_impl(lut_type, flat_node2pin_start_map, flat_node2pin_map, pin2net_map, pin_typeIds, sorted_net_map, clInstId, res_lut))
                    {
                        lutFail = true;
                    }
                    break;
                }
            default:
                {
                    break;
                }
        }
    }

    if (!lutFail)
    {
        for (int sg = 0; sg < SLICE_CAPACITY; ++sg)
        {
            nwCand_lut[sg] = res_lut[sg];
            nwCand_ff[sg] = res_ff[sg];
        }
        for (int sg = 0; sg < CKSR_IN_CLB; ++sg)
        {
            nwCand_cksr[sg] = res_cksr[sg];
        }
        for (int sg = 0; sg < CE_IN_CLB; ++sg)
        {
            nwCand_ce[sg] = res_ce[sg];
        }

        return true;
    }

    if(fit_luts_to_candidate_impl(lut_type, pin2net_map, pin_typeIds, flat_node2pin_start_map, flat_node2pin_map, sorted_net_map, flat_node2precluster_map, node2fence_region_map, instPcl, flat_node2prclstrCount[instId], NUM_BLE_PER_SLICE, res_lut))
    {
        for (int sg = 0; sg < SLICE_CAPACITY; ++sg)
        {
            nwCand_lut[sg] = res_lut[sg];
            nwCand_ff[sg] = res_ff[sg];
        }
        for (int sg = 0; sg < CKSR_IN_CLB; ++sg)
        {
            nwCand_cksr[sg] = res_cksr[sg];
        }
        for (int sg = 0; sg < CE_IN_CLB; ++sg)
        {
            nwCand_ce[sg] = res_ce[sg];
        }

        return true;
    }
    return false;
}

//template <typename T>
inline __device__ void remove_incompatible_neighbors(const int* node2fence_region_map, const int* lut_type, const int* flat_node2pin_start_map, const int* flat_node2pin_map, const int* node2pincount, const int* net2pincount, const int* pin2net_map, const int* pin_typeIds, const int* sorted_net_map, const int* flat_node2prclstrCount, const int* flat_node2precluster_map, const int* flop2ctrlSetId_map, const int* flop_ctrlSets, const int* site_det_impl_lut, const int* site_det_impl_ff, const int* site_det_impl_cksr, const int* site_det_impl_ce, const int* site_det_sig,
const int* site_det_sig_idx, const int siteId, const int sNbrIdx, const int HALF_SLICE_CAPACITY, const int NUM_BLE_PER_SLICE, const int SIG_IDX, const int CKSR_IN_CLB, const int CE_IN_CLB, int* site_nbr_idx, int* site_nbr)
{
    int sdtopId = siteId*SIG_IDX;

    for (int nbrId = 0; nbrId < site_nbr_idx[siteId]; ++nbrId)
    {
        int instId = site_nbr[sNbrIdx + nbrId];


        if (val_in_array(site_det_sig, site_det_sig_idx[siteId], sdtopId, instId) || 
            !is_inst_in_cand_feasible(node2fence_region_map, lut_type, flat_node2pin_start_map, flat_node2pin_map, node2pincount, net2pincount, pin2net_map, pin_typeIds, sorted_net_map, flat_node2prclstrCount, flat_node2precluster_map, flop2ctrlSetId_map, flop_ctrlSets, site_det_impl_lut, site_det_impl_ff, site_det_impl_cksr, site_det_impl_ce, 
            siteId, instId, HALF_SLICE_CAPACITY, NUM_BLE_PER_SLICE, CKSR_IN_CLB, CE_IN_CLB))
        {
            site_nbr[sNbrIdx + nbrId] = INVALID;
        }
    }

    //Remove invalid neighbor instances
    remove_invalid_neighbor(siteId, sNbrIdx, site_nbr_idx, site_nbr);
}


//define add_inst_to_sig
inline __device__ bool add_inst_to_sig(const int node2prclstrCount, const int* flat_node2precluster_map, const int* sorted_node_map, const int instPcl, int* nwCand_sig, int& nwCand_sigIdx)
{
    if (nwCand_sigIdx == 2*SLICE_CAPACITY)
    {
        return false;
    }
    
    int temp[32]; //Max capacity is 16 FFs + 16 LUTs for Ultrascale
    int mIdx(0);

    int itA(0), itB(0);
    int endA(nwCand_sigIdx), endB(node2prclstrCount);

    while (itA != endA && itB != endB)
    {
        if(sorted_node_map[nwCand_sig[itA]] < sorted_node_map[flat_node2precluster_map[instPcl+itB]])
        {
            temp[mIdx] = nwCand_sig[itA];
            ++mIdx;
            ++itA;
        }
        else if(sorted_node_map[nwCand_sig[itA]] > sorted_node_map[flat_node2precluster_map[instPcl+itB]])
        {
            temp[mIdx] = flat_node2precluster_map[instPcl+itB];
            ++mIdx;
            ++itB;
        } else
        {
            return false;
        }
    }
    if (itA == endA)
    {
        for (int mBIdx = itB; mBIdx < endB; ++mBIdx)
        {
            temp[mIdx] = flat_node2precluster_map[instPcl+mBIdx];
            ++mIdx;
        }
    } else
    {
        for (int mAIdx = itA; mAIdx < endA; ++mAIdx)
        {
            temp[mIdx] = nwCand_sig[mAIdx];
            ++mIdx;
        }
    }

    //Remove duplicates
    for (int i = 0; i < mIdx; ++i)
    {
        for (int j=0; j < i; ++j)
        {
            if (temp[i] == temp[j])
            {
                --mIdx;
                for (int k=i; k < mIdx; ++k)
                {
                    temp[k] = temp[k+1];
                }
                --i;
            }
        }
    }

    nwCand_sigIdx = mIdx;
    for (int smIdx = 0; smIdx < mIdx; ++smIdx)
    {
        nwCand_sig[smIdx] = temp[smIdx];
    }
    return true;
}

//WL Improv
template <typename T>
__device__ void compute_wirelen_improv(const T* pos_x, const T* pos_y, const T* net_bbox, const T* pin_offset_x, const T* pin_offset_y, const T* net_weights, const T* site_xy, const int* net2pincount, const int* flat_net2pin_start_map, const int* net_pinIdArrayX, const int* net_pinIdArrayY, const int* pin2node_map, const T xWirelenWt, const T yWirelenWt, const int currNetId, const int res_siteId, const int cNIPIdx, const int* currNetIntPins, T& wirelenImprov)
{
    //Compute wirelenImprov
    int cNbId = currNetId*4;
    T netXlen = net_bbox[cNbId+2] - net_bbox[cNbId];
    T netYlen = net_bbox[cNbId+3] - net_bbox[cNbId+1];
    if (cNIPIdx == net2pincount[currNetId])
    {
        T bXLo(pin_offset_x[currNetIntPins[0]]);
        T bXHi(pin_offset_x[currNetIntPins[0]]);
        T bYLo(pin_offset_y[currNetIntPins[0]]);
        T bYHi(pin_offset_y[currNetIntPins[0]]);
        for (int poI = 1; poI < cNIPIdx; ++poI)
        {
            T poX = pin_offset_x[currNetIntPins[poI]];
            T poY = pin_offset_y[currNetIntPins[poI]];
            if (poX < bXLo)
            {
                bXLo = poX;
            } else if (poX > bXHi)
            {
                bXHi = poX;
            }
            if (poY < bYLo)
            {
                bYLo = poY;
            } else if (poY > bYHi)
            {
                bYHi = poY;
            }
        }
        wirelenImprov += net_weights[currNetId] * (xWirelenWt * (netXlen - (bXHi-bXLo)) + yWirelenWt * (netYlen - (bYHi - bYLo)));
        return;
    }

    T bXLo(net_bbox[cNbId]);
    T bYLo(net_bbox[cNbId+1]);
    T bXHi(net_bbox[cNbId+2]);
    T bYHi(net_bbox[cNbId+3]);

    int sId = res_siteId*2;
    T locX = site_xy[sId];
    T locY = site_xy[sId+1];

    if (locX <= bXLo)
    {
        bXLo = locX;
    } else
    {
        int n2pId = flat_net2pin_start_map[currNetId];
        while (n2pId < flat_net2pin_start_map[currNetId+1] && val_in_array(currNetIntPins, cNIPIdx, 0, net_pinIdArrayX[n2pId]))
        {
            ++n2pId;
        }
        int reqPId = net_pinIdArrayX[n2pId];
        T pinX = pos_x[pin2node_map[reqPId]] + pin_offset_x[reqPId];
        bXLo = DREAMPLACE_STD_NAMESPACE::min(pinX, locX);
    }

    if (locX >= bXHi)
    {
        bXHi = locX;
    } else
    {
        int n2pId = flat_net2pin_start_map[currNetId+1]-1;
        while (n2pId >= flat_net2pin_start_map[currNetId] && val_in_array(currNetIntPins, cNIPIdx, 0, net_pinIdArrayX[n2pId]))
        {
            --n2pId;
        }
        int reqPId = net_pinIdArrayX[n2pId];
        T pinX = pos_x[pin2node_map[reqPId]] + pin_offset_x[reqPId];
        bXHi = DREAMPLACE_STD_NAMESPACE::max(pinX, locX);
    }

    if (locY <= bYLo)
    {
        bYLo = locY;
    } else
    {
        int n2pId = flat_net2pin_start_map[currNetId];
        while (n2pId < flat_net2pin_start_map[currNetId+1] && val_in_array(currNetIntPins, cNIPIdx, 0, net_pinIdArrayY[n2pId]))
        {
            ++n2pId;
        }
        int reqPId = net_pinIdArrayY[n2pId];
        T pinY = pos_y[pin2node_map[reqPId]] + pin_offset_y[reqPId];
        bYLo = DREAMPLACE_STD_NAMESPACE::min(pinY, locY);
    }

    if (locY >= bYHi)
    {
        bYHi = locY;
    } else
    {
        int n2pId = flat_net2pin_start_map[currNetId+1]-1;
        while (n2pId >= flat_net2pin_start_map[currNetId] && val_in_array(currNetIntPins, cNIPIdx, 0, net_pinIdArrayY[n2pId]))
        {
            --n2pId;
        }
        int reqPId = net_pinIdArrayY[n2pId];
        T pinY = pos_y[pin2node_map[reqPId]] + pin_offset_y[reqPId];
        bYHi = DREAMPLACE_STD_NAMESPACE::max(pinY, locY);
    }
    wirelenImprov += net_weights[currNetId] * (xWirelenWt * (netXlen - (bXHi-bXLo)) + yWirelenWt * (netYlen - (bYHi - bYLo)));
    return;
}

//computeCandidateScore
template <typename T>
__device__ void compute_candidate_score(const T* pos_x, const T* pos_y, const T* pin_offset_x, const T* pin_offset_y, const T* net_bbox, const T* net_weights, const int* net_pinIdArrayX, const int* net_pinIdArrayY, const int* flat_net2pin_start_map, const int* flat_node2pin_start_map, const int* flat_node2pin_map, const int* sorted_net_map, const int* pin2net_map, const int* pin2node_map, const int* net2pincount, const T* site_xy, const T xWirelenWt, const T yWirelenWt, const T extNetCountWt, const T wirelenImprovWt, const int netShareScoreMaxNetDegree, const int wlscoreMaxNetDegree, const int* res_sig, const int res_siteId, const int res_sigIdx, T &result)
{
    T netShareScore = T(0.0);
    T wirelenImprov = T(0.0);
    int pins[128];
    int pinIdx = 0;

    for (int i = 0; i < res_sigIdx; ++i)
    {
        int instId = res_sig[i];
        for (int pId = flat_node2pin_start_map[instId]; pId < flat_node2pin_start_map[instId+1]; ++pId)
        {
            pins[pinIdx] = flat_node2pin_map[pId];
            ++pinIdx;
        }
    }

    if (pinIdx > 0)
    {
        for (int ix = 1; ix < pinIdx; ++ix)
        {
            for (int jx = 0; jx < pinIdx-1; ++jx)
            {
                if (sorted_net_map[pin2net_map[pins[jx]]] > sorted_net_map[pin2net_map[pins[jx+1]]])
                {
                    int tempVal = pins[jx];
                    pins[jx] = pins[jx+1];
                    pins[jx+1] = tempVal;
                }
            }
        }
    } else
    {
        result = T(0.0);
        return;
    } 

    int maxNetDegree = DREAMPLACE_STD_NAMESPACE::max(netShareScoreMaxNetDegree, wlscoreMaxNetDegree);
    int currNetId = pin2net_map[pins[0]];

    if (net2pincount[currNetId] > maxNetDegree)
    {
        result = T(0.0);
        return;
    } 

    int numIntNets(0), numNets(0);
    int currNetIntPins[128];
    int cNIPIdx = 0;

    currNetIntPins[cNIPIdx] = pins[0];
    ++cNIPIdx;

    for (int pId = 1; pId < pinIdx; ++pId)
    {
        int netId = pin2net_map[pins[pId]];
        if (netId == currNetId)
        {
            currNetIntPins[cNIPIdx] = pins[pId];
            ++cNIPIdx;
        } else
        {
            if (net2pincount[currNetId] <= netShareScoreMaxNetDegree)
            {
                ++numNets;
                numIntNets += (cNIPIdx == net2pincount[currNetId] ? 1 : 0);
                netShareScore += net_weights[currNetId] * (cNIPIdx - 1.0) / DREAMPLACE_STD_NAMESPACE::max(T(1.0), net2pincount[currNetId] - T(1.0));
            }
            if (net2pincount[currNetId] <= wlscoreMaxNetDegree)
            {
                compute_wirelen_improv(pos_x, pos_y, net_bbox, pin_offset_x, pin_offset_y, net_weights, site_xy, net2pincount, flat_net2pin_start_map, net_pinIdArrayX, net_pinIdArrayY, pin2node_map, xWirelenWt, yWirelenWt, currNetId, res_siteId, cNIPIdx, currNetIntPins, wirelenImprov);
            }
            currNetId = netId;
            if (net2pincount[currNetId] > maxNetDegree)
            {
                break;
            }
            cNIPIdx = 0;
            currNetIntPins[cNIPIdx] = pins[pId];
            ++cNIPIdx;
        }
    }

    //Handle last net
    if (net2pincount[currNetId] <= netShareScoreMaxNetDegree)
    {
        ++numNets;
        numIntNets += (cNIPIdx == net2pincount[currNetId] ? 1 : 0);
        netShareScore += net_weights[currNetId] * (cNIPIdx - 1.0) / DREAMPLACE_STD_NAMESPACE::max(T(1.0), net2pincount[currNetId] - T(1.0));
    }
    if (net2pincount[currNetId] <= wlscoreMaxNetDegree)
    {
        compute_wirelen_improv(pos_x, pos_y, net_bbox, pin_offset_x, pin_offset_y, net_weights, site_xy, net2pincount, flat_net2pin_start_map, net_pinIdArrayX, net_pinIdArrayY, pin2node_map, xWirelenWt, yWirelenWt, currNetId, res_siteId, cNIPIdx, currNetIntPins, wirelenImprov);
    }
    netShareScore /= (T(1.0) + extNetCountWt * (numNets - numIntNets));
    result = netShareScore + wirelenImprovWt * wirelenImprov;
}

template <typename T>
inline __device__ bool compare_pq_tops(
        const T* site_curr_pq_score, const int* site_curr_pq_top_idx, const int* site_curr_pq_validIdx,
        const int* site_curr_pq_siteId, const int* site_curr_pq_sig_idx, const int* site_curr_pq_sig,
        const int* site_curr_pq_impl_lut, const int* site_curr_pq_impl_ff, const int* site_curr_pq_impl_cksr,
        const int* site_curr_pq_impl_ce, const T* site_next_pq_score, const int* site_next_pq_top_idx,
        const int* site_next_pq_validIdx, const int* site_next_pq_siteId, const int* site_next_pq_sig_idx,
        const int* site_next_pq_sig, const int* site_next_pq_impl_lut, const int* site_next_pq_impl_ff,
        const int* site_next_pq_impl_cksr, const int* site_next_pq_impl_ce, const int siteId,
        const int sPQ, const int SIG_IDX, const int CKSR_IN_CLB, const int CE_IN_CLB)
{
    //Check site_curr_pq TOP == site_next_pq TOP
    int curr_pq_topId = sPQ+site_curr_pq_top_idx[siteId];
    int next_pq_topId = sPQ+site_next_pq_top_idx[siteId];

    if (site_curr_pq_validIdx[curr_pq_topId] != site_next_pq_validIdx[next_pq_topId] || 
            site_curr_pq_validIdx[curr_pq_topId] != 1)
    {
        return false;
    }
    if (site_curr_pq_score[curr_pq_topId] == site_next_pq_score[next_pq_topId] && 
            site_curr_pq_siteId[curr_pq_topId] == site_next_pq_siteId[next_pq_topId] &&
            site_curr_pq_sig_idx[curr_pq_topId] == site_next_pq_sig_idx[next_pq_topId])
    {
        //Check both sig
        int currPQSigIdx = curr_pq_topId*SIG_IDX;
        int nextPQSigIdx = next_pq_topId*SIG_IDX;

        for (int sg = 0; sg < site_curr_pq_sig_idx[curr_pq_topId]; ++sg)
        {
            if (site_curr_pq_sig[currPQSigIdx + sg] != site_next_pq_sig[nextPQSigIdx + sg])
            {
                return false;
            }
        }

        //Check impl
        int cCKRId = curr_pq_topId*CKSR_IN_CLB;
        int cCEId = curr_pq_topId*CE_IN_CLB;
        int cFFId = curr_pq_topId*SLICE_CAPACITY;
        int nCKRId = next_pq_topId*CKSR_IN_CLB;
        int nCEId = next_pq_topId*CE_IN_CLB;
        int nFFId = next_pq_topId*SLICE_CAPACITY;

        for (int sg = 0; sg < SLICE_CAPACITY; ++sg)
        {
            if (site_curr_pq_impl_lut[cFFId + sg] != site_next_pq_impl_lut[nFFId + sg] || 
                    site_curr_pq_impl_ff[cFFId + sg] != site_next_pq_impl_ff[nFFId + sg])
            {
                return false;
            }
        }
        for (int sg = 0; sg < CKSR_IN_CLB; ++sg)
        {
            if (site_curr_pq_impl_cksr[cCKRId + sg] != site_next_pq_impl_cksr[nCKRId + sg])
            {
                return false;
            }
        }
        for (int sg = 0; sg < CE_IN_CLB; ++sg)
        {
            if(site_curr_pq_impl_ce[cCEId + sg] != site_next_pq_impl_ce[nCEId + sg])
            {
                return false;
            }
        }
        return true;
    }
    return false;
}

////////////////////////////////
////////////////////////////////
////////////////////////////////

template <typename T>
__global__ void initNets(const T *pos_x,
                         const T *pos_y,
                         const int *sorted_net_idx,
                         const int *flat_net2pin_map,
                         const int *flat_net2pin_start_map,
                         const int *pin2node_map,
                         const T *pin_offset_x,
                         const T *pin_offset_y,
                         const int *net2pincount,
                         const int num_nets,
                         T *net_bbox,
                         int *net_pinIdArrayX,
                         int *net_pinIdArrayY,
                         const int wlscoreMaxNetDegree)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    while (i < num_nets)
    {
        const int idx = sorted_net_idx[i];

        if (net2pincount[idx] > 0 && net2pincount[idx] <= wlscoreMaxNetDegree)
        {
            int pinIdxBeg = flat_net2pin_start_map[idx];
            int pinIdxEnd = flat_net2pin_start_map[idx+1];

            int xLo = idx*4;
            int yLo = xLo+1;
            int xHi = xLo+2;
            int yHi = xLo+3;

            int pnIdx = flat_net2pin_map[pinIdxBeg];
            int nodeIdx = pin2node_map[pnIdx];

            net_bbox[xLo] = pos_x[nodeIdx] + pin_offset_x[pnIdx];
            net_bbox[yLo] = pos_y[nodeIdx] + pin_offset_y[pnIdx];
            net_bbox[xHi] = net_bbox[xLo];
            net_bbox[yHi] = net_bbox[yLo];

            //Max net2pin considered is 99
            int tempX[128];
            int tempY[128];
            T temp_flat_net2pinX[128];
            T temp_flat_net2pinY[128];
            int tempId = 0;

            temp_flat_net2pinX[tempId] = net_bbox[xLo];
            temp_flat_net2pinY[tempId] = net_bbox[yLo];
            tempX[tempId] = pnIdx;
            tempY[tempId] = pnIdx;

            ++tempId;

            //Update Net Bbox based on node location and pin offset
            for (int pId = pinIdxBeg+1; pId < pinIdxEnd; ++pId)
            {
                int pinIdx = flat_net2pin_map[pId];
                int ndIdx = pin2node_map[pinIdx];

                T valX = pos_x[ndIdx] + pin_offset_x[pinIdx];
                T valY = pos_y[ndIdx] + pin_offset_y[pinIdx];

                if (valX < net_bbox[xLo])
                {
                    net_bbox[xLo] = valX;
                } else if (valX > net_bbox[xHi])
                {
                    net_bbox[xHi] = valX;
                }

                if (valY < net_bbox[yLo])
                {
                    net_bbox[yLo] = valY;
                } else if (valY > net_bbox[yHi])
                {
                    net_bbox[yHi] = valY;
                }

                temp_flat_net2pinX[tempId] = valX;
                temp_flat_net2pinY[tempId] = valY;

                tempX[tempId] = pinIdx;
                tempY[tempId] = pinIdx;

                ++tempId;
            }

            //Sort pinIdArray based on node loc and pin offset - Bubble sort
            for (int ix = 1; ix < tempId; ++ix)
            {
                for (int jx = 0; jx < tempId-1; ++jx)
                {
                    //Sort X
                    if (temp_flat_net2pinX[jx] > temp_flat_net2pinX[jx+1])
                    {
                        int tempVal = tempX[jx];
                        tempX[jx] = tempX[jx+1];
                        tempX[jx+1] = tempVal;

                        T net2pinVal = temp_flat_net2pinX[jx];
                        temp_flat_net2pinX[jx] = temp_flat_net2pinX[jx+1];
                        temp_flat_net2pinX[jx+1] = net2pinVal;
                    }

                    //Sort Y
                    if (temp_flat_net2pinY[jx] > temp_flat_net2pinY[jx+1])
                    {
                        int tempVal = tempY[jx];
                        tempY[jx] = tempY[jx+1];
                        tempY[jx+1] = tempVal;

                        T net2pinVal = temp_flat_net2pinY[jx];
                        temp_flat_net2pinY[jx] = temp_flat_net2pinY[jx+1];
                        temp_flat_net2pinY[jx+1] = net2pinVal;
                    }
                }
            }

            //Assign sorted values back
            tempId = 0;
            for (int pId = pinIdxBeg; pId < pinIdxEnd; ++pId)
            {
                net_pinIdArrayX[pId] = tempX[tempId];
                net_pinIdArrayY[pId] = tempY[tempId];
                ++tempId;
            }
        }
        i += blockDim.x * gridDim.x;
    }
}

//Preclustering
template <typename T>
__global__ void preClustering(const T *pos_x,
                              const T *pos_y,
                              const T *pin_offset_x,
                              const T *pin_offset_y,
                              const int *sorted_node_map,
                              const int *sorted_node_idx,
                              const int *flat_net2pin_map,
                              const int *flat_net2pin_start_map,
                              const int *flop2ctrlSetId_map,
                              const int *flop_ctrlSets,
                              const int *node2fence_region_map,
                              const int *node2outpinIdx_map,
                              const int *pin2net_map,
                              const int *pin2node_map,
                              const int *pin_typeIds,
                              const int num_nodes,
                              const T preClusteringMaxDist,
                              int *flat_node2precluster_map,
                              int *flat_node2prclstrCount)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    //const int blk = blockDim.x * gridDim.x;
    while (i < num_nodes)
    {
        const int idx = sorted_node_map[i];
        if (node2fence_region_map[idx] == 0) //Only consider LUTs
        {
            int outPinId = node2outpinIdx_map[idx];
            int outNetId = pin2net_map[outPinId];

            int pinIdxBeg = flat_net2pin_start_map[outNetId];
            int pinIdxEnd = flat_net2pin_start_map[outNetId+1];

            T instLocX = pos_x[idx] + pin_offset_x[outPinId];
            T instLocY = pos_y[idx] + pin_offset_y[outPinId];

            //Assume ffs to be of size 10
            int ff_insts[16];
            T ff_dists[16];
            int ffIdx = 0;

            for (int pinId = pinIdxBeg; pinId < pinIdxEnd; ++pinId)
            {
                int pinIdx = flat_net2pin_map[pinId];
                int nodeIdx = pin2node_map[pinIdx];

                T distX = instLocX - pos_x[nodeIdx] - pin_offset_x[pinIdx];
                T distY = instLocY - pos_y[nodeIdx] - pin_offset_y[pinIdx];

                T dist = DREAMPLACE_STD_NAMESPACE::abs(distX) + DREAMPLACE_STD_NAMESPACE::abs(distY);

                if (pin_typeIds[pinIdx] == INPUT_PIN && node2fence_region_map[nodeIdx] == 1 &&
                        dist < preClusteringMaxDist)
                {
                    ff_insts[ffIdx] = nodeIdx;
                    ff_dists[ffIdx] = dist;
                    ++ffIdx;
                }
            }

            //Check if ff is empty
            if (ffIdx > 0)
            {
                //Sort ff_insts/ff_dists based on dist and sorted_node_map
                for (int ix = 1; ix < ffIdx; ++ix)
                {
                    for (int jx = 0; jx < ffIdx-1; ++jx)
                    {
                        if (ff_dists[jx] == ff_dists[jx+1])
                        {
                            if (sorted_node_map[ff_insts[jx]] > sorted_node_map[ff_insts[jx+1]])
                            {
                                int tempVal = ff_insts[jx];
                                ff_insts[jx] = ff_insts[jx+1];
                                ff_insts[jx+1] = tempVal;

                                T distVal = ff_dists[jx];
                                ff_dists[jx] = ff_dists[jx+1];
                                ff_dists[jx+1] = distVal;
                            }
                        } else
                        {
                            if (ff_dists[jx] > ff_dists[jx+1])
                            {
                                int tempVal = ff_insts[jx];
                                ff_insts[jx] = ff_insts[jx+1];
                                ff_insts[jx+1] = tempVal;

                                T distVal = ff_dists[jx];
                                ff_dists[jx] = ff_dists[jx+1];
                                ff_dists[jx+1] = distVal;
                            }
                        }
                    }
                }

                int nPIdx = idx*3;

                flat_node2precluster_map[nPIdx + flat_node2prclstrCount[idx]] = ff_insts[0];
                ++flat_node2prclstrCount[idx];

                int fcIdx = flop2ctrlSetId_map[ff_insts[0]]*3 + 1;
                int cksr = flop_ctrlSets[fcIdx];

                for (int fIdx = 1; fIdx < ffIdx; ++fIdx)
                {
                    int ctrlIdx = flop2ctrlSetId_map[ff_insts[fIdx]]*3 + 1;
                    int fCksr = flop_ctrlSets[ctrlIdx];

                    if (fCksr == cksr)
                    {
                        flat_node2precluster_map[nPIdx + flat_node2prclstrCount[idx]] = ff_insts[fIdx];
                        ++flat_node2prclstrCount[idx];
                        break;
                    }
                }

                //Sort precluster based on instId
                for (int ix = nPIdx+1; ix < nPIdx + flat_node2prclstrCount[idx]; ++ix)
                {
                    for (int jx = nPIdx; jx < nPIdx + flat_node2prclstrCount[idx]-1; ++jx)
                    {
                        if (sorted_node_map[flat_node2precluster_map[jx]] > sorted_node_map[flat_node2precluster_map[jx+1]])
                        {
                            int val = flat_node2precluster_map[jx];
                            flat_node2precluster_map[jx] = flat_node2precluster_map[jx+1];
                            flat_node2precluster_map[jx+1] = val;
                        }
                    }
                }

                for (int prcl = 0; prcl < flat_node2prclstrCount[idx]; ++prcl) // 0 is the LUT instId
                {
                    int fIdx = flat_node2precluster_map[nPIdx + prcl];
                    int fID = fIdx*3;
                    if (fIdx != idx)
                    {
                        for (int cl = 0; cl < flat_node2prclstrCount[idx]; ++cl)
                        {
                            flat_node2precluster_map[fID + cl] = flat_node2precluster_map[nPIdx + cl];
                        }
                        flat_node2prclstrCount[fIdx] = flat_node2prclstrCount[idx];
                    }
                }
            }
        }
        i += blockDim.x * gridDim.x;
    }
}

//runDLIteration
template <typename T>
__global__ void runDLIteration(const T* pos_x,
                               const T* pos_y,
                               const T* pin_offset_x,
                               const T* pin_offset_y,
                               const T* net_bbox,
                               const T* site_xy,
                               const int* net_pinIdArrayX,
                               const int* net_pinIdArrayY,
                               const int* node2fence_region_map,
                               const int* flop_ctrlSets,
                               const int* flop2ctrlSetId_map,
                               const int* lut_type,
                               const int* flat_node2pin_start_map,
                               const int* flat_node2pin_map,
                               const int* node2pincount,
                               const int* net2pincount,
                               const int* pin2net_map,
                               const int* pin_typeIds,
                               const int* flat_net2pin_start_map,
                               const int* pin2node_map,
                               const int* sorted_net_map,
                               const int* sorted_node_map,
                               const int* flat_node2prclstrCount,
                               const int* flat_node2precluster_map,
                               const int* site_nbrList,
                               const int* site_nbrRanges,
                               const int* site_nbrRanges_idx,
                               const T* net_weights,
                               const int* addr2site_map,
                               const int num_clb_sites,
                               const int minStableIter,
                               const int maxList ,
                               const int HALF_SLICE_CAPACITY,
                               const int NUM_BLE_PER_SLICE,
                               const int minNeighbors,
                               const int intMinVal,
                               const int numGroups,
                               const int netShareScoreMaxNetDegree,
                               const int wlscoreMaxNetDegree,
                               const T xWirelenWt,
                               const T yWirelenWt,
                               const T wirelenImprovWt,
                               const T extNetCountWt,
                               const int CKSR_IN_CLB,
                               const int CE_IN_CLB,
                               const int SCL_IDX,
                               const int SIG_IDX,
                               int* validIndices_curr_scl,
                               int* site_nbr_idx,
                               int* site_nbr,
                               int* site_nbrGroup_idx,
                               int* site_curr_pq_top_idx,
                               int* site_curr_pq_validIdx,
                               int* site_curr_pq_sig_idx,
                               int* site_curr_pq_sig,
                               int* site_curr_pq_idx,
                               int* site_curr_stable,
                               int* site_curr_pq_siteId,
                               T* site_curr_pq_score,
                               int* site_curr_pq_impl_lut,
                               int* site_curr_pq_impl_ff,
                               int* site_curr_pq_impl_cksr,
                               int* site_curr_pq_impl_ce,
                               T* site_curr_scl_score,
                               int* site_curr_scl_siteId,
                               int* site_curr_scl_idx,
                               int* site_curr_scl_validIdx,
                               int* site_curr_scl_sig_idx,
                               int* site_curr_scl_sig,
                               int* site_curr_scl_impl_lut,
                               int* site_curr_scl_impl_ff,
                               int* site_curr_scl_impl_cksr,
                               int* site_curr_scl_impl_ce,
                               int* site_next_pq_idx,
                               int* site_next_pq_validIdx,
                               int* site_next_pq_top_idx,
                               T* site_next_pq_score,
                               int* site_next_pq_siteId,
                               int* site_next_pq_sig_idx,
                               int* site_next_pq_sig,
                               int* site_next_pq_impl_lut,
                               int* site_next_pq_impl_ff,
                               int* site_next_pq_impl_cksr,
                               int* site_next_pq_impl_ce,
                               T* site_next_scl_score,
                               int* site_next_scl_siteId,
                               int* site_next_scl_idx,
                               int* site_next_scl_validIdx,
                               int* site_next_scl_sig_idx,
                               int* site_next_scl_sig,
                               int* site_next_scl_impl_lut,
                               int* site_next_scl_impl_ff,
                               int* site_next_scl_impl_cksr,
                               int* site_next_scl_impl_ce,
                               int* site_next_stable,
                               T* site_det_score,
                               int* site_det_siteId,
                               int* site_det_sig_idx,
                               int* site_det_sig,
                               int* site_det_impl_lut,
                               int* site_det_impl_ff,
                               int* site_det_impl_cksr,
                               int* site_det_impl_ce,
                               int* inst_curr_detSite,
                               int* inst_curr_bestSite,
                               int* inst_next_detSite,
                               T* inst_next_bestScoreImprov,
                               int* inst_next_bestSite,
                               int* inst_score_improv,
                               int* site_score_improv
                               )
{
    for (int sIdx = threadIdx.x + blockDim.x * blockIdx.x; sIdx < num_clb_sites; sIdx += blockDim.x*gridDim.x)
    {
        int siteId = addr2site_map[sIdx];
        int numNbrGroups = (site_nbrRanges_idx[sIdx] == 0) ? 0 : site_nbrRanges_idx[sIdx]-1;

        int sPQ(sIdx*PQ_IDX), sSCL(sIdx*SCL_IDX), sNbrIdx(sIdx*maxList);
        int sdtopId = sIdx*SIG_IDX;
        int sdlutId = sIdx*SLICE_CAPACITY;
        int sdckId = sIdx*CKSR_IN_CLB;
        int sdceId = sIdx*CE_IN_CLB;

        int sclSigId = sSCL*SIG_IDX;
        int scllutIdx = sSCL*SLICE_CAPACITY;
        int sclckIdx = sSCL*CKSR_IN_CLB;
        int sclceIdx = sSCL*CE_IN_CLB;

        //(a)Try to commit Top candidates
        char commitTopCandidate(INVALID);

        int tsPQ(sPQ + site_curr_pq_top_idx[sIdx]);
        int topIdx(tsPQ*SIG_IDX);
        int lutIdx = tsPQ*SLICE_CAPACITY;
        int ckIdx = tsPQ*CKSR_IN_CLB;
        int ceIdx = tsPQ*CE_IN_CLB;

        if (site_curr_pq_idx[sIdx] == 0 || site_curr_stable[sIdx] < minStableIter ||
                !candidate_validity_check(topIdx, site_curr_pq_sig_idx[tsPQ], site_curr_pq_siteId[tsPQ], site_curr_pq_sig, inst_curr_detSite))
        {
            commitTopCandidate = 0;
        } else {

            for (int pIdx = 0; pIdx < site_curr_pq_sig_idx[tsPQ]; ++pIdx)
            {
                int pqInst = site_curr_pq_sig[topIdx + pIdx];

                if (inst_curr_detSite[pqInst] != siteId && inst_curr_bestSite[pqInst] != siteId)
                {
                    commitTopCandidate = 0;
                    break;
                }
            }
        }

        if (commitTopCandidate == INVALID)
        {
            //////
            site_det_score[sIdx] = site_curr_pq_score[tsPQ];
            site_det_siteId[sIdx] = site_curr_pq_siteId[tsPQ];
            site_det_sig_idx[sIdx] = site_curr_pq_sig_idx[tsPQ];

            for(int sg = 0; sg < site_curr_pq_sig_idx[tsPQ]; ++sg)
            {
                site_det_sig[sdtopId + sg] = site_curr_pq_sig[topIdx + sg];
            }
            for(int sg = 0; sg < SLICE_CAPACITY; ++sg)
            {
                site_det_impl_lut[sdlutId + sg] = site_curr_pq_impl_lut[lutIdx + sg];
                site_det_impl_ff[sdlutId + sg] = site_curr_pq_impl_ff[lutIdx + sg];
            }
            for(int sg = 0; sg < CKSR_IN_CLB; ++sg)
            {
                site_det_impl_cksr[sdckId + sg] = site_curr_pq_impl_cksr[ckIdx + sg];
            }
            for(int sg = 0; sg < CE_IN_CLB; ++sg)
            {
                site_det_impl_ce[sdceId + sg] = site_curr_pq_impl_ce[ceIdx + sg];
            }
            //////

            for(int iSig = 0; iSig < site_det_sig_idx[sIdx]; ++iSig)
            {
                int sigInst = site_det_sig[sdtopId + iSig];
                inst_next_detSite[sigInst] = siteId;
            }

            //Remove Incompatible Neighbors
            remove_incompatible_neighbors(node2fence_region_map, lut_type, flat_node2pin_start_map, flat_node2pin_map, node2pincount, net2pincount, pin2net_map, pin_typeIds, sorted_net_map, flat_node2prclstrCount, flat_node2precluster_map, flop2ctrlSetId_map, flop_ctrlSets, site_det_impl_lut, site_det_impl_ff, site_det_impl_cksr, site_det_impl_ce, site_det_sig, 
                    //site_det_score, site_det_siteId,
                    site_det_sig_idx, sIdx, sNbrIdx, HALF_SLICE_CAPACITY, NUM_BLE_PER_SLICE, SIG_IDX, CKSR_IN_CLB, CE_IN_CLB, site_nbr_idx, site_nbr);

            //Clear pq and make scl only contain the committed candidate
            //int sclCount = 0;
            for (int vId = 0; vId < PQ_IDX; ++vId)
            {
                int nPQId = sPQ + vId;
                //if (site_next_pq_validIdx[nPQId] != INVALID)
                //{
                //Clear contents thoroughly
                clear_cand_contents(
                        nPQId, SIG_IDX, CKSR_IN_CLB, CE_IN_CLB,
                        site_next_pq_sig_idx, site_next_pq_sig,
                        site_next_pq_impl_lut, site_next_pq_impl_ff,
                        site_next_pq_impl_cksr, site_next_pq_impl_ce);

                site_next_pq_validIdx[nPQId] = INVALID;
                site_next_pq_sig_idx[nPQId] = 0;
                site_next_pq_siteId[nPQId] = INVALID;
                site_next_pq_score[nPQId] = T(0.0);
                //++sclCount;
                //if (sclCount == site_next_pq_idx[sIdx])
                //{
                //    break;
                //}
                //}
            }
            site_next_pq_idx[sIdx] = 0;
            site_next_pq_top_idx[sIdx] = INVALID;

            int sclCount = 0;
            for (int vId = 0; vId < SCL_IDX; ++vId)
            {
                int cSclId = sSCL + vId;
                if (site_curr_scl_validIdx[cSclId] != INVALID)
                {
                    //Clear contents thoroughly
                    clear_cand_contents(
                            cSclId, SIG_IDX, CKSR_IN_CLB, CE_IN_CLB,
                            site_curr_scl_sig_idx, site_curr_scl_sig,
                            site_curr_scl_impl_lut, site_curr_scl_impl_ff,
                            site_curr_scl_impl_cksr, site_curr_scl_impl_ce);

                    site_curr_scl_validIdx[cSclId] = INVALID;
                    site_curr_scl_sig_idx[cSclId] = 0;
                    site_curr_scl_siteId[cSclId] = INVALID;
                    site_curr_scl_score[cSclId] = 0.0;
                    ++sclCount;
                    if (sclCount == site_curr_scl_idx[sIdx])
                    {
                        break;
                    }
                }
            }
            site_curr_scl_idx[sIdx] = 0;

            site_curr_scl_score[sSCL] = site_det_score[sIdx];
            site_curr_scl_siteId[sSCL] = site_det_siteId[sIdx];
            site_curr_scl_sig_idx[sSCL] = site_det_sig_idx[sIdx];
            site_curr_scl_validIdx[sSCL] = 1;

            for(int sg = 0; sg < site_det_sig_idx[sIdx]; ++sg)
            {
                site_curr_scl_sig[sclSigId + sg] = site_det_sig[sdtopId + sg];
            }
            for(int sg = 0; sg < SLICE_CAPACITY; ++sg)
            {
                site_curr_scl_impl_lut[scllutIdx + sg] = site_det_impl_lut[sdlutId + sg];
                site_curr_scl_impl_ff[scllutIdx + sg] = site_det_impl_ff[sdlutId + sg];
            }
            for(int sg = 0; sg < CKSR_IN_CLB; ++sg)
            {
                site_curr_scl_impl_cksr[sclckIdx + sg] = site_det_impl_cksr[sdckId + sg];
            }
            for(int sg = 0; sg < CE_IN_CLB; ++sg)
            {
                site_curr_scl_impl_ce[sclceIdx + sg] = site_det_impl_ce[sdceId + sg];
            }
            ++site_curr_scl_idx[sIdx];
            /////
            commitTopCandidate = 1;
        }

        if (commitTopCandidate == 0)
        {
            //Remove invalid candidates from site PQ
            if (site_next_pq_idx[sIdx] > 0)
            {
                //int snCnt = 0;
                //int maxEntries = site_next_pq_idx[sIdx];
                for (int nIdx = 0; nIdx < PQ_IDX; ++nIdx)
                {
                    int ssPQ = sPQ + nIdx;
                    int topIdx = ssPQ*SIG_IDX;

                    if (site_next_pq_validIdx[ssPQ] != INVALID)
                    {
                        if (!candidate_validity_check(topIdx, site_next_pq_sig_idx[ssPQ], site_next_pq_siteId[ssPQ], site_next_pq_sig, inst_curr_detSite))
                        {
                            //Clear contents thoroughly
                            clear_cand_contents(
                                    ssPQ, SIG_IDX, CKSR_IN_CLB, CE_IN_CLB,
                                    site_next_pq_sig_idx, site_next_pq_sig,
                                    site_next_pq_impl_lut, site_next_pq_impl_ff,
                                    site_next_pq_impl_cksr, site_next_pq_impl_ce);

                            site_next_pq_validIdx[ssPQ] = INVALID;
                            site_next_pq_sig_idx[ssPQ] = 0;
                            site_next_pq_siteId[ssPQ] = INVALID;
                            site_next_pq_score[ssPQ] = 0.0;
                            --site_next_pq_idx[sIdx];
                        }
                        //++snCnt;
                        //if (snCnt == maxEntries)
                        //{
                        //    break;
                        //}
                    }
                }

                if (site_next_pq_validIdx[sPQ + site_next_pq_top_idx[sIdx]] == INVALID)
                {
                    site_next_pq_top_idx[sIdx] = INVALID;

                    if (site_next_pq_idx[sIdx] > 0)
                    {
                        int snCnt = 0;
                        int maxEntries = site_next_pq_idx[sIdx];
                        T maxScore(-1000.0);
                        int maxScoreId(INVALID);
                        //Recompute top idx
                        for (int nIdx = 0; nIdx < PQ_IDX; ++nIdx)
                        {
                            int ssPQ = sPQ + nIdx;
                            if (site_next_pq_validIdx[ssPQ] != INVALID)
                            {
                                if (site_next_pq_score[ssPQ] > maxScore)
                                {
                                    maxScore = site_next_pq_score[ssPQ];
                                    maxScoreId = nIdx;
                                }
                                ++snCnt;
                                if (snCnt == maxEntries)
                                {
                                    break;
                                }
                            }
                        }
                        site_next_pq_top_idx[sIdx] = maxScoreId;
                    }
                }
            }

            //Remove invalid candidates from seed candidate list (scl)
            if (site_curr_scl_idx[sIdx] > 0)
            {
                int sclCount = 0;
                int maxEntries = site_curr_scl_idx[sIdx];
                for (int nIdx = 0; nIdx < SCL_IDX; ++nIdx)
                {
                    int ssPQ = sSCL + nIdx;
                    int topIdx = ssPQ*SIG_IDX;

                    if (site_curr_scl_validIdx[ssPQ] != INVALID)
                    {
                        if (!candidate_validity_check(topIdx, site_curr_scl_sig_idx[ssPQ], site_curr_scl_siteId[ssPQ], site_curr_scl_sig, inst_curr_detSite))
                        {
                            //Clear contents thoroughly
                            clear_cand_contents(
                                    ssPQ, SIG_IDX, CKSR_IN_CLB, CE_IN_CLB,
                                    site_curr_scl_sig_idx, site_curr_scl_sig,
                                    site_curr_scl_impl_lut, site_curr_scl_impl_ff,
                                    site_curr_scl_impl_cksr, site_curr_scl_impl_ce);

                            site_curr_scl_validIdx[ssPQ] = INVALID;
                            site_curr_scl_sig_idx[ssPQ] = 0;
                            site_curr_scl_siteId[ssPQ] = INVALID;
                            site_curr_scl_score[ssPQ] = 0.0;
                            --site_curr_scl_idx[sIdx];
                        }
                        ++sclCount;
                        if (sclCount == maxEntries)
                        {
                            break;
                        }
                    }
                }
            }
            //If site.scl becomes empty, add site.det into it as the seed
            if (site_curr_scl_idx[sIdx] == 0)
            {

                //site.curr.scl.emplace_back(site.det);
                site_curr_scl_score[sSCL] = site_det_score[sIdx];
                site_curr_scl_siteId[sSCL] = site_det_siteId[sIdx];
                site_curr_scl_sig_idx[sSCL] = site_det_sig_idx[sIdx];
                site_curr_scl_validIdx[sSCL] = 1;

                for(int sg = 0; sg < site_det_sig_idx[sIdx]; ++sg)
                {
                    site_curr_scl_sig[sclSigId + sg] = site_det_sig[sdtopId + sg];
                }
                for(int sg = 0; sg < SLICE_CAPACITY; ++sg)
                {
                    site_curr_scl_impl_lut[scllutIdx + sg] = site_det_impl_lut[sdlutId + sg];
                    site_curr_scl_impl_ff[scllutIdx + sg] = site_det_impl_ff[sdlutId + sg];
                }
                for(int sg = 0; sg < CKSR_IN_CLB; ++sg)
                {
                    site_curr_scl_impl_cksr[sclckIdx + sg] = site_det_impl_cksr[sdckId + sg];
                }
                for(int sg = 0; sg < CE_IN_CLB; ++sg)
                {
                    site_curr_scl_impl_ce[sclceIdx + sg] = site_det_impl_ce[sdceId + sg];
                }
                ++site_curr_scl_idx[sIdx];
            }
        }

        // (c) removeCommittedNeighbors(site)
        for (int sNIdx = 0; sNIdx < site_nbr_idx[sIdx]; ++sNIdx)
        {
            int siteInst = site_nbr[sNbrIdx + sNIdx];
            if (inst_curr_detSite[siteInst] != INVALID)
            {
                site_nbr[sNbrIdx + sNIdx] = INVALID;
            }
        }
        remove_invalid_neighbor(sIdx, sNbrIdx, site_nbr_idx, site_nbr);

        // (d) addNeighbors(site)
        ////reuse site_nbrGroup_idx to store the Ids for STAGGERED NEW CANDIDATE ADDITION 
        int maxNeighbors = site_nbrRanges[sIdx*(numGroups+1) + numGroups];
        if (site_nbr_idx[sIdx] < minNeighbors && site_nbrGroup_idx[sIdx] <= maxNeighbors)
        {
            int beg = site_nbrGroup_idx[sIdx];
            ////STAGGERED ADDITION SET TO SLICE/16 or SLICE_CAPACITY/8
            ///For ISPD'2016 benchmarks, SLICE=32 and SLICE_CAPACITY=16
            int end = DREAMPLACE_STD_NAMESPACE::min(site_nbrGroup_idx[sIdx]+SLICE_CAPACITY/8, maxNeighbors);
            site_nbrGroup_idx[sIdx] = end;

            for (int aNIdx = beg; aNIdx < end; ++aNIdx)
            {
                int instId = site_nbrList[sNbrIdx + aNIdx];

                if (inst_curr_detSite[instId] == INVALID && 
                        is_inst_in_cand_feasible(node2fence_region_map, lut_type, flat_node2pin_start_map, flat_node2pin_map, node2pincount, net2pincount, pin2net_map, pin_typeIds, sorted_net_map, flat_node2prclstrCount, flat_node2precluster_map, flop2ctrlSetId_map, flop_ctrlSets, site_det_impl_lut, site_det_impl_ff, site_det_impl_cksr, site_det_impl_ce,
                            sIdx, instId, HALF_SLICE_CAPACITY, NUM_BLE_PER_SLICE, CKSR_IN_CLB, CE_IN_CLB))
                {
                    site_nbr[sNbrIdx + site_nbr_idx[sIdx]] = site_nbrList[sNbrIdx + aNIdx]; 
                    ++site_nbr_idx[sIdx];
                }
            }
        }

        //Generate indices for kernel_2
        int validId = 0;
        for (int scsIdx = 0; scsIdx < SCL_IDX; ++scsIdx)
        {
            int siteCurrIdx = sSCL + scsIdx;
            if (site_curr_scl_validIdx[siteCurrIdx] != INVALID)
            {
                validIndices_curr_scl[sSCL+validId] = siteCurrIdx;
                ++validId;
            }
            if (validId == site_curr_scl_idx[sIdx]) break;
        }

        // (e) createNewCandidates(site)
        //Generate new candidates by merging site_nbr to site_curr_scl

        const int limit_x = site_curr_scl_idx[sIdx];
        const int limit_y = site_nbr_idx[sIdx];
        ////RESTRICTED NEW CANDIDATE EXPLORATION SET TO SLICE/8 or SLICE_CAPACITY/4
        ///For ISPD'2016 benchmarks, SLICE=32 and SLICE_CAPACITY=16
        const int limit_cands = DREAMPLACE_STD_NAMESPACE::min(SLICE_CAPACITY/4,limit_x*limit_y);

        for (int scsIdx = 0; scsIdx < limit_cands; ++scsIdx)
        {
            int sclId = scsIdx/limit_y;
            int snIdx = scsIdx/limit_x;
            int siteCurrIdx = validIndices_curr_scl[sSCL + sclId];

            /////
            int sCKRId = siteCurrIdx*CKSR_IN_CLB;
            int sCEId = siteCurrIdx*CE_IN_CLB;
            int sFFId = siteCurrIdx*SLICE_CAPACITY;
            int sGId = siteCurrIdx*SIG_IDX;

            T nwCand_score = site_curr_scl_score[siteCurrIdx];
            int nwCand_siteId = site_curr_scl_siteId[siteCurrIdx];
            int nwCand_sigIdx = site_curr_scl_sig_idx[siteCurrIdx];

            //array instantiation
            int nwCand_sig[32];
            int nwCand_lut[SLICE_CAPACITY];
            int nwCand_ff[SLICE_CAPACITY];
            int nwCand_ce[4];
            int nwCand_cksr[2];

            for (int sg = 0; sg < nwCand_sigIdx; ++sg)
            {
                nwCand_sig[sg] = site_curr_scl_sig[sGId + sg];
            }
            for (int sg = 0; sg < SLICE_CAPACITY; ++sg)
            {
                nwCand_lut[sg] = site_curr_scl_impl_lut[sFFId + sg];
                nwCand_ff[sg] = site_curr_scl_impl_ff[sFFId + sg];
            }
            for (int sg = 0; sg < CKSR_IN_CLB; ++sg)
            {
                nwCand_cksr[sg] = site_curr_scl_impl_cksr[sCKRId + sg];
            }
            for (int sg = 0; sg < CE_IN_CLB; ++sg)
            {
                nwCand_ce[sg] = site_curr_scl_impl_ce[sCEId + sg];
            }

            int instId = site_nbr[sNbrIdx + snIdx];
            int instPcl = instId*3;

            int add_inst_to_sig = INVALID;
            if (nwCand_sigIdx >= 2*SLICE_CAPACITY)
            {
                add_inst_to_sig = 0;
            }

            if (add_inst_to_sig == INVALID)
            {
                int itA(0), itB(0);
                int endA(nwCand_sigIdx), endB(flat_node2prclstrCount[instId]);

                if (endA > 0)
                {
                    int temp[32]; //Max capacity is 16 FFs + 16 LUTs for Ultrascale
                    int mIdx(0);
                    while (itA != endA && itB != endB)
                    {
                        if(sorted_node_map[nwCand_sig[itA]] < sorted_node_map[flat_node2precluster_map[instPcl+itB]])
                        {
                            temp[mIdx] = nwCand_sig[itA];
                            ++mIdx;
                            ++itA;
                        }
                        else if(sorted_node_map[nwCand_sig[itA]] > sorted_node_map[flat_node2precluster_map[instPcl+itB]])
                        {
                            temp[mIdx] = flat_node2precluster_map[instPcl+itB];
                            ++mIdx;
                            ++itB;
                        } else
                        {
                            add_inst_to_sig = 0;
                            break;
                        }
                    }
                    if (add_inst_to_sig == INVALID)
                    {
                        if (itA == endA)
                        {
                            for (int mBIdx = itB; mBIdx < endB; ++mBIdx)
                            {
                                temp[mIdx] = flat_node2precluster_map[instPcl+mBIdx];
                                ++mIdx;
                            }
                        } else
                        {
                            for (int mAIdx = itA; mAIdx < endA; ++mAIdx)
                            {
                                temp[mIdx] = nwCand_sig[mAIdx];
                                ++mIdx;
                            }
                        }

                        nwCand_sigIdx = mIdx;
                        for (int smIdx = 0; smIdx < mIdx; ++smIdx)
                        {
                            nwCand_sig[smIdx] = temp[smIdx];
                        }
                        add_inst_to_sig = 1;
                    }
                } else
                {
                    for (int mBIdx = itB; mBIdx < endB; ++mBIdx)
                    {
                        nwCand_sig[nwCand_sigIdx] = flat_node2precluster_map[instPcl+mBIdx];
                        ++nwCand_sigIdx;
                    }
                    add_inst_to_sig = 1;
                }
            }

            if (add_inst_to_sig == 1)
            {
                //check cand sig is in site_next_pq
                int candSigInSiteNextPQ = INVALID;
                //int cnt = 0;
                for (int i = 0; i < PQ_IDX; ++i)
                {
                    int sigIdx = sPQ + i;
                    if (site_next_pq_validIdx[sigIdx] != INVALID)
                    {
                        if (site_next_pq_sig_idx[sigIdx] == nwCand_sigIdx)
                        {
                            int pqIdx(sigIdx*SIG_IDX), mtch(0);

                            for (int k = 0; k < nwCand_sigIdx; ++k)
                            {
                                for (int l = 0; l < nwCand_sigIdx; ++l)
                                {
                                    if (site_next_pq_sig[pqIdx + l] == nwCand_sig[k])
                                    {
                                        ++mtch;
                                        break;
                                    }
                                }
                            }
                            if (mtch == nwCand_sigIdx)
                            {
                                candSigInSiteNextPQ = 1;
                                break;
                            }
                        }
                    }
                }

                if (candSigInSiteNextPQ == INVALID &&
                        add_inst_to_cand_impl(lut_type, flat_node2pin_start_map, flat_node2pin_map, node2pincount, net2pincount, pin2net_map, pin_typeIds, sorted_net_map, flat_node2prclstrCount, flat_node2precluster_map, flop2ctrlSetId_map, node2fence_region_map, flop_ctrlSets, instId, CKSR_IN_CLB, CE_IN_CLB, HALF_SLICE_CAPACITY, NUM_BLE_PER_SLICE, nwCand_lut, nwCand_ff, nwCand_cksr, nwCand_ce))
                {
                    compute_candidate_score(pos_x, pos_y, pin_offset_x, pin_offset_y, net_bbox, net_weights, net_pinIdArrayX, net_pinIdArrayY, flat_net2pin_start_map, flat_node2pin_start_map, flat_node2pin_map, sorted_net_map, pin2net_map, pin2node_map, net2pincount, site_xy, xWirelenWt, yWirelenWt, extNetCountWt, wirelenImprovWt, netShareScoreMaxNetDegree, wlscoreMaxNetDegree, nwCand_sig, nwCand_siteId, nwCand_sigIdx, nwCand_score);

                    int nxtId(INVALID);

                    if (site_next_pq_idx[sIdx] < PQ_IDX)
                    {
                        for (int vId = 0; vId < PQ_IDX; ++vId)
                        {
                            if (site_next_pq_validIdx[sPQ+vId] == INVALID)
                            {
                                nxtId = vId;
                                ++site_next_pq_idx[sIdx];
                                break;
                            }
                        }
                    } else
                    {
                        //find least score and replace if current score is greater
                        T ckscore(nwCand_score);
                        for (int vId = 0; vId < PQ_IDX; ++vId)
                        {
                            if (ckscore > site_next_pq_score[sPQ + vId])
                            {
                                ckscore = site_next_pq_score[sPQ + vId]; 
                                nxtId = vId;
                            }
                        }
                    }

                    if (nxtId != INVALID)
                    {
                        int nTId = sPQ + nxtId;
                        int nCKRId = nTId*CKSR_IN_CLB;
                        int nCEId = nTId*CE_IN_CLB;
                        int nFFId = nTId*SLICE_CAPACITY;
                        int nSGId = nTId*SIG_IDX;

                        /////
                        site_next_pq_validIdx[nTId] = 1;
                        site_next_pq_score[nTId] = nwCand_score;
                        site_next_pq_siteId[nTId] = nwCand_siteId;
                        site_next_pq_sig_idx[nTId] = nwCand_sigIdx;

                        for (int sg = 0; sg < nwCand_sigIdx; ++sg)
                        {
                            site_next_pq_sig[nSGId + sg] = nwCand_sig[sg];
                        }
                        for (int sg = 0; sg < SLICE_CAPACITY; ++sg)
                        {
                            site_next_pq_impl_lut[nFFId + sg] = nwCand_lut[sg];
                            site_next_pq_impl_ff[nFFId + sg] = nwCand_ff[sg];
                        }
                        for (int sg = 0; sg < CKSR_IN_CLB; ++sg)
                        {
                            site_next_pq_impl_cksr[nCKRId + sg] = nwCand_cksr[sg];
                        }
                        for (int sg = 0; sg < CE_IN_CLB; ++sg)
                        {
                            site_next_pq_impl_ce[nCEId + sg] = nwCand_ce[sg];
                        }
                        /////

                        if (site_next_pq_idx[sIdx] == 1 || 
                                nwCand_score > site_next_pq_score[sPQ + site_next_pq_top_idx[sIdx]])
                        {
                            site_next_pq_top_idx[sIdx] = nxtId;
                        }

                        nxtId = INVALID;

                        if (site_next_scl_idx[sIdx] < SCL_IDX)
                        {
                            for (int vId = 0; vId < SCL_IDX; ++vId)
                            {
                                if (site_next_scl_validIdx[sSCL+vId] == INVALID)
                                {
                                    nxtId = vId;
                                    ++site_next_scl_idx[sIdx];
                                    break;
                                }
                            }
                        } else
                        {
                            //find least score and replace if current score is greater
                            T ckscore(nwCand_score);
                            for (int vId = 0; vId < SCL_IDX; ++vId)
                            {
                                if (ckscore > site_next_scl_score[sSCL+vId])
                                {
                                    ckscore = site_next_scl_score[sSCL+vId]; 
                                    nxtId = vId;
                                }
                            }
                        }

                        if (nxtId != INVALID)
                        {
                            /////
                            nTId = sSCL + nxtId;
                            nCKRId = nTId*CKSR_IN_CLB;
                            nCEId = nTId*CE_IN_CLB;
                            nFFId = nTId*SLICE_CAPACITY;
                            nSGId = nTId*SIG_IDX;

                            site_next_scl_validIdx[nTId] = 1;
                            site_next_scl_score[nTId] = nwCand_score;
                            site_next_scl_siteId[nTId] = nwCand_siteId;
                            site_next_scl_sig_idx[nTId] = nwCand_sigIdx;

                            for (int sg = 0; sg < nwCand_sigIdx; ++sg)
                            {
                                site_next_scl_sig[nSGId + sg] = nwCand_sig[sg];
                            }
                            for (int sg = 0; sg < SLICE_CAPACITY; ++sg)
                            {
                                site_next_scl_impl_lut[nFFId + sg] = nwCand_lut[sg];
                                site_next_scl_impl_ff[nFFId + sg] = nwCand_ff[sg];
                            }
                            for (int sg = 0; sg < CKSR_IN_CLB; ++sg)
                            {
                                site_next_scl_impl_cksr[nCKRId + sg] = nwCand_cksr[sg];
                            }
                            for (int sg = 0; sg < CE_IN_CLB; ++sg)
                            {
                                site_next_scl_impl_ce[nCEId + sg] = nwCand_ce[sg];
                            }
                            /////
                        }
                    }
                }
            }
        }

        //Remove all candidates in scl that is worse than the worst candidate in PQ
        if (site_next_pq_idx[sIdx] > 0)
        {
            //Find worst candidate in PQ
            T ckscore(site_next_pq_score[sPQ + site_next_pq_top_idx[sIdx]]);

            int sclCount = 0;
            for (int vId = 0; vId < PQ_IDX; ++vId)
            {
                int nPQId = sPQ + vId;
                if (site_next_pq_validIdx[nPQId] != INVALID)
                {
                    if (ckscore > site_next_pq_score[nPQId])
                    {
                        ckscore = site_next_pq_score[nPQId]; 
                    }
                    ++sclCount;
                    if (sclCount == site_next_pq_idx[sIdx])
                    {
                        break;
                    }
                }
            }

            //Invalidate worst ones in scl
            sclCount = 0;
            int maxEntries = site_next_scl_idx[sIdx];
            for (int ckId = 0; ckId < SCL_IDX; ++ckId)
            {
                int vId = sSCL + ckId;
                if (site_next_scl_validIdx[vId] != INVALID)
                {
                    if (ckscore > site_next_scl_score[vId])
                    {
                        //Clear contents thoroughly
                        clear_cand_contents(
                                vId, SIG_IDX, CKSR_IN_CLB, CE_IN_CLB,
                                site_next_scl_sig_idx, site_next_scl_sig,
                                site_next_scl_impl_lut, site_next_scl_impl_ff,
                                site_next_scl_impl_cksr, site_next_scl_impl_ce);

                        site_next_scl_validIdx[vId] = INVALID;
                        site_next_scl_sig_idx[vId] = 0;
                        site_next_scl_siteId[vId] = INVALID;
                        site_next_scl_score[vId] = 0.0;
                        --site_next_scl_idx[sIdx];
                    }
                    ++sclCount;
                    if (sclCount == maxEntries)
                    {
                        break;
                    }
                }
            }
        }

        //Update stable Iteration count
        if (site_curr_pq_idx[sIdx] > 0 && site_next_pq_idx[sIdx] > 0 && 
                compare_pq_tops(site_curr_pq_score, site_curr_pq_top_idx,
                    site_curr_pq_validIdx, site_curr_pq_siteId, site_curr_pq_sig_idx,
                    site_curr_pq_sig, site_curr_pq_impl_lut, site_curr_pq_impl_ff,
                    site_curr_pq_impl_cksr, site_curr_pq_impl_ce, site_next_pq_score,
                    site_next_pq_top_idx, site_next_pq_validIdx, site_next_pq_siteId,
                    site_next_pq_sig_idx, site_next_pq_sig, site_next_pq_impl_lut,
                    site_next_pq_impl_ff, site_next_pq_impl_cksr, site_next_pq_impl_ce,
                    sIdx, sPQ, SIG_IDX, CKSR_IN_CLB, CE_IN_CLB))
        {
            site_next_stable[sIdx] = site_curr_stable[sIdx] + 1;
        } else
        {
            site_next_stable[sIdx] = 0;
        }

        //// (f) broadcastTopCandidate(site)
        if (site_next_pq_idx[sIdx] > 0)
        {
            int topIdx = sPQ + site_next_pq_top_idx[sIdx];
            int topSigId = topIdx*SIG_IDX;

            T scoreImprov = site_next_pq_score[topIdx] - site_det_score[sIdx];

            ////UPDATED SEQUENTIAL PORTION
            int scoreImprovInt = DREAMPLACE_STD_NAMESPACE::max(int(scoreImprov*10000), intMinVal);
            site_score_improv[sIdx] = scoreImprovInt + siteId;

            for (int ssIdx = 0; ssIdx < site_next_pq_sig_idx[topIdx]; ++ssIdx)
            {
                int instId = site_next_pq_sig[topSigId + ssIdx];

                if (inst_curr_detSite[instId] == INVALID)
                {
                    atomicMax(&inst_score_improv[instId], scoreImprovInt);
                }
            }
        }
    }
}

//runDLIteration split kernel 1
template <typename T>
__global__ void runDLIteration_kernel_1(
                   const int* node2fence_region_map,
                   const int* flop_ctrlSets,
                   const int* flop2ctrlSetId_map,
                   const int* lut_type,
                   const int* flat_node2pin_start_map,
                   const int* flat_node2pin_map,
                   const int* node2pincount,
                   const int* net2pincount,
                   const int* pin2net_map,
                   const int* pin_typeIds,
                   const int* sorted_net_map,
                   const int* flat_node2prclstrCount,
                   const int* flat_node2precluster_map,
                   const int* site_nbrList,
                   const int* site_nbrRanges,
                   const int* site_nbrRanges_idx,
                   const int* addr2site_map,
                   const int num_clb_sites,
                   const int minStableIter,
                   const int maxList,
                   const int HALF_SLICE_CAPACITY,
                   const int NUM_BLE_PER_SLICE,
                   const int minNeighbors,
                   const int numGroups,
                   const int CKSR_IN_CLB,
                   const int CE_IN_CLB,
                   const int SCL_IDX,
                   const int SIG_IDX,
                   int* site_nbr_idx,
                   int* site_nbr,
                   int* site_nbrGroup_idx,
                   int* site_curr_pq_top_idx,
                   int* site_curr_pq_sig_idx,
                   int* site_curr_pq_sig,
                   int* site_curr_pq_idx,
                   int* site_curr_stable,
                   int* site_curr_pq_siteId,
                   T* site_curr_pq_score,
                   int* site_curr_pq_impl_lut,
                   int* site_curr_pq_impl_ff,
                   int* site_curr_pq_impl_cksr,
                   int* site_curr_pq_impl_ce,
                   T* site_curr_scl_score,
                   int* site_curr_scl_siteId,
                   int* site_curr_scl_idx,
                   int* site_curr_scl_validIdx,
                   int* site_curr_scl_sig_idx,
                   int* site_curr_scl_sig,
                   int* site_curr_scl_impl_lut,
                   int* site_curr_scl_impl_ff,
                   int* site_curr_scl_impl_cksr,
                   int* site_curr_scl_impl_ce,
                   int* site_next_pq_idx,
                   int* site_next_pq_validIdx,
                   int* site_next_pq_top_idx,
                   T* site_next_pq_score,
                   int* site_next_pq_siteId,
                   int* site_next_pq_sig_idx,
                   int* site_next_pq_sig,
                   int* site_next_pq_impl_lut,
                   int* site_next_pq_impl_ff,
                   int* site_next_pq_impl_cksr,
                   int* site_next_pq_impl_ce,
                   T* site_det_score,
                   int* site_det_siteId,
                   int* site_det_sig_idx,
                   int* site_det_sig,
                   int* site_det_impl_lut,
                   int* site_det_impl_ff,
                   int* site_det_impl_cksr,
                   int* site_det_impl_ce,
                   int* inst_curr_detSite,
                   int* inst_curr_bestSite,
                   int* inst_next_detSite,
                   int* validIndices_curr_scl,
                   int* cumsum_curr_scl
                   )
{
    for (int sIdx = threadIdx.x + blockDim.x * blockIdx.x; sIdx < num_clb_sites; sIdx += blockDim.x*gridDim.x)
    {
        int siteId = addr2site_map[sIdx];
        int numNbrGroups = (site_nbrRanges_idx[sIdx] == 0) ? 0 : site_nbrRanges_idx[sIdx]-1;

        int sPQ(sIdx*PQ_IDX), sSCL(sIdx*SCL_IDX), sNbrIdx(sIdx*maxList);
        int sdtopId = sIdx*SIG_IDX;
        int sdlutId = sIdx*SLICE_CAPACITY;
        int sdckId = sIdx*CKSR_IN_CLB;
        int sdceId = sIdx*CE_IN_CLB;

        int sclSigId = sSCL*SIG_IDX;
        int scllutIdx = sSCL*SLICE_CAPACITY;
        int sclckIdx = sSCL*CKSR_IN_CLB;
        int sclceIdx = sSCL*CE_IN_CLB;

        //(a)Try to commit Top candidates
        int commitTopCandidate(INVALID);

        int tsPQ(sPQ + site_curr_pq_top_idx[sIdx]);
        int topIdx(tsPQ*SIG_IDX);
        int lutIdx = tsPQ*SLICE_CAPACITY;
        int ckIdx = tsPQ*CKSR_IN_CLB;
        int ceIdx = tsPQ*CE_IN_CLB;

        if (site_curr_pq_idx[sIdx] == 0 || site_curr_stable[sIdx] < minStableIter ||
                !candidate_validity_check(topIdx, site_curr_pq_sig_idx[tsPQ], site_curr_pq_siteId[tsPQ], site_curr_pq_sig, inst_curr_detSite))
        {
            commitTopCandidate = 0;
        } else {
            for (int pIdx = 0; pIdx < site_curr_pq_sig_idx[tsPQ]; ++pIdx)
            {
                int pqInst = site_curr_pq_sig[topIdx + pIdx];

                if (inst_curr_detSite[pqInst] != siteId && inst_curr_bestSite[pqInst] != siteId)
                {
                    commitTopCandidate = 0;
                    break;
                }
            }
        }

        if (commitTopCandidate == INVALID)
        {
            //////
            site_det_score[sIdx] = site_curr_pq_score[tsPQ];
            site_det_siteId[sIdx] = site_curr_pq_siteId[tsPQ];
            site_det_sig_idx[sIdx] = site_curr_pq_sig_idx[tsPQ];

            for(int sg = 0; sg < site_curr_pq_sig_idx[tsPQ]; ++sg)
            {
                site_det_sig[sdtopId + sg] = site_curr_pq_sig[topIdx + sg];
            }
            for(int sg = 0; sg < SLICE_CAPACITY; ++sg)
            {
                site_det_impl_lut[sdlutId + sg] = site_curr_pq_impl_lut[lutIdx + sg];
                site_det_impl_ff[sdlutId + sg] = site_curr_pq_impl_ff[lutIdx + sg];
            }
            for(int sg = 0; sg < CKSR_IN_CLB; ++sg)
            {
                site_det_impl_cksr[sdckId + sg] = site_curr_pq_impl_cksr[ckIdx + sg];
            }
            for(int sg = 0; sg < CE_IN_CLB; ++sg)
            {
                site_det_impl_ce[sdceId + sg] = site_curr_pq_impl_ce[ceIdx + sg];
            }
            //////

            for(int iSig = 0; iSig < site_det_sig_idx[sIdx]; ++iSig)
            {
                int sigInst = site_det_sig[sdtopId + iSig];
                inst_next_detSite[sigInst] = siteId;
            }

            //Remove Incompatible Neighbors
            remove_incompatible_neighbors(node2fence_region_map, lut_type, flat_node2pin_start_map, flat_node2pin_map, node2pincount, net2pincount, pin2net_map, pin_typeIds, sorted_net_map, flat_node2prclstrCount, flat_node2precluster_map, flop2ctrlSetId_map, flop_ctrlSets, site_det_impl_lut, site_det_impl_ff, site_det_impl_cksr, site_det_impl_ce, site_det_sig, 
                    site_det_sig_idx, sIdx, sNbrIdx, HALF_SLICE_CAPACITY, NUM_BLE_PER_SLICE, SIG_IDX, CKSR_IN_CLB, CE_IN_CLB, site_nbr_idx, site_nbr);

            //Clear pq and make scl only contain the committed candidate
            //int sclCount = 0;
            for (int vId = 0; vId < PQ_IDX; ++vId)
            {
                int nPQId = sPQ + vId;
                //if (site_next_pq_validIdx[nPQId] != INVALID)
                //{
                //Clear contents thoroughly
                clear_cand_contents(
                        nPQId, SIG_IDX, CKSR_IN_CLB, CE_IN_CLB,
                        site_next_pq_sig_idx, site_next_pq_sig,
                        site_next_pq_impl_lut, site_next_pq_impl_ff,
                        site_next_pq_impl_cksr, site_next_pq_impl_ce);

                site_next_pq_validIdx[nPQId] = INVALID;
                site_next_pq_siteId[nPQId] = INVALID;
                site_next_pq_score[nPQId] = 0.0;
                site_next_pq_sig_idx[nPQId] = 0;

                //++sclCount;
                //if (sclCount == site_next_pq_idx[sIdx])
                //{
                //    break;
                //}
                //}
            }
            site_next_pq_idx[sIdx] = 0;
            site_next_pq_top_idx[sIdx] = INVALID;

            int sclCount = 0;
            for (int vId = 0; vId < SCL_IDX; ++vId)
            {
                int cSclId = sSCL + vId;
                if (site_curr_scl_validIdx[cSclId] != INVALID)
                {
                    //Clear contents thoroughly
                    clear_cand_contents(
                            cSclId, SIG_IDX, CKSR_IN_CLB, CE_IN_CLB,
                            site_curr_scl_sig_idx, site_curr_scl_sig,
                            site_curr_scl_impl_lut, site_curr_scl_impl_ff,
                            site_curr_scl_impl_cksr, site_curr_scl_impl_ce);

                    site_curr_scl_validIdx[cSclId] = INVALID;
                    site_curr_scl_sig_idx[cSclId] = 0;
                    site_curr_scl_siteId[cSclId] = INVALID;
                    site_curr_scl_score[cSclId] = 0.0;
                    ++sclCount;
                    if (sclCount == site_curr_scl_idx[sIdx])
                    {
                        break;
                    }
                }
            }
            site_curr_scl_idx[sIdx] = 0;

            site_curr_scl_score[sSCL] = site_det_score[sIdx];
            site_curr_scl_siteId[sSCL] = site_det_siteId[sIdx];
            site_curr_scl_sig_idx[sSCL] = site_det_sig_idx[sIdx];
            site_curr_scl_validIdx[sSCL] = 1;

            for(int sg = 0; sg < site_det_sig_idx[sIdx]; ++sg)
            {
                site_curr_scl_sig[sclSigId + sg] = site_det_sig[sdtopId + sg];
            }
            for(int sg = 0; sg < SLICE_CAPACITY; ++sg)
            {
                site_curr_scl_impl_lut[scllutIdx + sg] = site_det_impl_lut[sdlutId + sg];
                site_curr_scl_impl_ff[scllutIdx + sg] = site_det_impl_ff[sdlutId + sg];
            }
            for(int sg = 0; sg < CKSR_IN_CLB; ++sg)
            {
                site_curr_scl_impl_cksr[sclckIdx + sg] = site_det_impl_cksr[sdckId + sg];
            }
            for(int sg = 0; sg < CE_IN_CLB; ++sg)
            {
                site_curr_scl_impl_ce[sclceIdx + sg] = site_det_impl_ce[sdceId + sg];
            }
            ++site_curr_scl_idx[sIdx];
            /////
            commitTopCandidate = 1;
        }

        if (commitTopCandidate == 0)
        {
            //(b) removeInvalidCandidates

            //Remove invalid candidates from site PQ
            if (site_next_pq_idx[sIdx] > 0)
            {
                for (int nIdx = 0; nIdx < PQ_IDX; ++nIdx)
                {
                    int ssPQ = sPQ + nIdx;
                    int topIdx = ssPQ*SIG_IDX;

                    if (site_next_pq_validIdx[ssPQ] != INVALID)
                    {
                        if (!candidate_validity_check(topIdx, site_next_pq_sig_idx[ssPQ], site_next_pq_siteId[ssPQ], site_next_pq_sig, inst_curr_detSite))
                        {
                            //Clear contents thoroughly
                            clear_cand_contents(
                                    ssPQ, SIG_IDX, CKSR_IN_CLB, CE_IN_CLB,
                                    site_next_pq_sig_idx, site_next_pq_sig,
                                    site_next_pq_impl_lut, site_next_pq_impl_ff,
                                    site_next_pq_impl_cksr, site_next_pq_impl_ce);

                            site_next_pq_validIdx[ssPQ] = INVALID;
                            site_next_pq_sig_idx[ssPQ] = 0;
                            site_next_pq_siteId[ssPQ] = INVALID;
                            site_next_pq_score[ssPQ] = 0.0;
                            --site_next_pq_idx[sIdx];
                        }
                    }
                }

                if (site_next_pq_validIdx[sPQ + site_next_pq_top_idx[sIdx]] == INVALID)
                {
                    site_next_pq_top_idx[sIdx] = INVALID;

                    if (site_next_pq_idx[sIdx] > 0)
                    {
                        T maxScore(-1000.0);
                        int maxScoreId(INVALID);
                        //Recompute top idx
                        for (int nIdx = 0; nIdx < PQ_IDX; ++nIdx)
                        {
                            int ssPQ = sPQ + nIdx;
                            if (site_next_pq_validIdx[ssPQ] != INVALID)
                            {
                                if (site_next_pq_score[ssPQ] > maxScore)
                                {
                                    maxScore = site_next_pq_score[ssPQ];
                                    maxScoreId = nIdx;
                                }
                            }
                        }
                        site_next_pq_top_idx[sIdx] = maxScoreId;
                    }
                }
            }

            //Remove invalid candidates from seed candidate list (scl)
            if (site_curr_scl_idx[sIdx] > 0)
            {
                int sclCount = 0;
                int maxEntries = site_curr_scl_idx[sIdx];
                for (int nIdx = 0; nIdx < SCL_IDX; ++nIdx)
                {
                    int ssPQ = sSCL + nIdx;
                    int topIdx = ssPQ*SIG_IDX;

                    if (site_curr_scl_validIdx[ssPQ] != INVALID)
                    {
                        if (!candidate_validity_check(topIdx, site_curr_scl_sig_idx[ssPQ], site_curr_scl_siteId[ssPQ], site_curr_scl_sig, inst_curr_detSite))
                        {
                            //Clear contents thoroughly
                            clear_cand_contents(
                                    ssPQ, SIG_IDX, CKSR_IN_CLB, CE_IN_CLB,
                                    site_curr_scl_sig_idx, site_curr_scl_sig,
                                    site_curr_scl_impl_lut, site_curr_scl_impl_ff,
                                    site_curr_scl_impl_cksr, site_curr_scl_impl_ce);

                            site_curr_scl_validIdx[ssPQ] = INVALID;
                            site_curr_scl_sig_idx[ssPQ] = 0;
                            site_curr_scl_siteId[ssPQ] = INVALID;
                            site_curr_scl_score[ssPQ] = 0.0;
                            --site_curr_scl_idx[sIdx];
                        }
                        ++sclCount;
                        if (sclCount == maxEntries)
                        {
                            break;
                        }
                    }
                }
            }

            //If site.scl becomes empty, add site.det into it as the seed
            if (site_curr_scl_idx[sIdx] == 0)
            {

                site_curr_scl_score[sSCL] = site_det_score[sIdx];
                site_curr_scl_siteId[sSCL] = site_det_siteId[sIdx];
                site_curr_scl_sig_idx[sSCL] = site_det_sig_idx[sIdx];
                site_curr_scl_validIdx[sSCL] = 1;

                for(int sg = 0; sg < site_det_sig_idx[sIdx]; ++sg)
                {
                    site_curr_scl_sig[sclSigId + sg] = site_det_sig[sdtopId + sg];
                }
                for(int sg = 0; sg < SLICE_CAPACITY; ++sg)
                {
                    site_curr_scl_impl_lut[scllutIdx + sg] = site_det_impl_lut[sdlutId + sg];
                    site_curr_scl_impl_ff[scllutIdx + sg] = site_det_impl_ff[sdlutId + sg];
                }
                for(int sg = 0; sg < CKSR_IN_CLB; ++sg)
                {
                    site_curr_scl_impl_cksr[sclckIdx + sg] = site_det_impl_cksr[sdckId + sg];
                }
                for(int sg = 0; sg < CE_IN_CLB; ++sg)
                {
                    site_curr_scl_impl_ce[sclceIdx + sg] = site_det_impl_ce[sdceId + sg];
                }
                ++site_curr_scl_idx[sIdx];

            }
        }

        // (c) removeCommittedNeighbors(site)

        for (int sNIdx = 0; sNIdx < site_nbr_idx[sIdx]; ++sNIdx)
        {
            int siteInst = site_nbr[sNbrIdx + sNIdx];
            if (inst_curr_detSite[siteInst] != INVALID)
            {
                site_nbr[sNbrIdx + sNIdx] = INVALID;
            }
        }
        remove_invalid_neighbor(sIdx, sNbrIdx, site_nbr_idx, site_nbr);

        // (d) addNeighbors(site)
        ////STAGGERED ADDITION OF NEW NEIGHBORS
        int maxNeighbors = site_nbrRanges[sIdx*(numGroups+1) + numGroups];
        if (site_nbr_idx[sIdx] < minNeighbors && site_nbrGroup_idx[sIdx] <= maxNeighbors)
        {
            int beg = site_nbrGroup_idx[sIdx];
            ///STAGGERED ADDITION SET TO SLICE/16 or SLICE_CAPACITY/8
            ///FOR ISPD'2016 BENCHMARKS, SLICE=32 and SLICE_CAPACITY=16
            int end = DREAMPLACE_STD_NAMESPACE::min(site_nbrGroup_idx[sIdx]+SLICE_CAPACITY/8, maxNeighbors);
            site_nbrGroup_idx[sIdx] = end;

            for (int aNIdx = beg; aNIdx < end; ++aNIdx)
            {
                int instId = site_nbrList[sNbrIdx + aNIdx];

                if (inst_curr_detSite[instId] == INVALID && 
                        is_inst_in_cand_feasible(node2fence_region_map, lut_type, flat_node2pin_start_map, flat_node2pin_map, node2pincount, net2pincount, pin2net_map, pin_typeIds, sorted_net_map, flat_node2prclstrCount, flat_node2precluster_map, flop2ctrlSetId_map, flop_ctrlSets, site_det_impl_lut, site_det_impl_ff, site_det_impl_cksr, site_det_impl_ce,
                            sIdx, instId, HALF_SLICE_CAPACITY, NUM_BLE_PER_SLICE, CKSR_IN_CLB, CE_IN_CLB))
                {
                    site_nbr[sNbrIdx + site_nbr_idx[sIdx]] = site_nbrList[sNbrIdx + aNIdx]; 
                    ++site_nbr_idx[sIdx];
                }
            }
        }

        //Generate indices for kernel_2
        int validId = 0;
        for (int scsIdx = 0; scsIdx < SCL_IDX; ++scsIdx)
        {
            int siteCurrIdx = sSCL + scsIdx;
            if (site_curr_scl_validIdx[siteCurrIdx] != INVALID)
            {
                validIndices_curr_scl[sSCL+validId] = siteCurrIdx;
                ++validId;
            }
            if (validId == site_curr_scl_idx[sIdx]) break;
        }

        cumsum_curr_scl[sIdx] = site_curr_scl_idx[sIdx]*site_nbr_idx[sIdx];
    }
}

//runDLIteration split kernel 2
template <typename T>
__global__ void runDLIteration_kernel_2(const T* pos_x,
                   const T* pos_y,
                   const T* pin_offset_x,
                   const T* pin_offset_y,
                   const T* net_bbox,
                   const T* site_xy,
                   const int* net_pinIdArrayX,
                   const int* net_pinIdArrayY,
                   const int* node2fence_region_map,
                   const int* flop_ctrlSets,
                   const int* flop2ctrlSetId_map,
                   const int* lut_type,
                   const int* flat_node2pin_start_map,
                   const int* flat_node2pin_map,
                   const int* node2pincount,
                   const int* net2pincount,
                   const int* pin2net_map,
                   const int* pin_typeIds,
                   const int* flat_net2pin_start_map,
                   const int* pin2node_map,
                   const int* sorted_node_map,
                   const int* sorted_net_map,
                   const int* flat_node2prclstrCount,
                   const int* flat_node2precluster_map,
                   const T* net_weights,
                   const int* addr2site_map,
                   const int* validIndices_curr_scl,
                   const int* sorted_clb_siteIds,
                   const int intMinVal,
                   const int num_clb_sites,
                   const int maxList,
                   const int HALF_SLICE_CAPACITY,
                   const int NUM_BLE_PER_SLICE,
                   const int netShareScoreMaxNetDegree,
                   const int wlscoreMaxNetDegree,
                   const T xWirelenWt,
                   const T yWirelenWt,
                   const T wirelenImprovWt,
                   const T extNetCountWt,
                   const int CKSR_IN_CLB,
                   const int CE_IN_CLB,
                   const int SCL_IDX,
                   const int SIG_IDX,
                   int* site_nbr_idx,
                   int* site_nbr,
                   int* site_curr_pq_top_idx,
                   int* site_curr_pq_validIdx,
                   int* site_curr_pq_sig_idx,
                   int* site_curr_pq_sig,
                   int* site_curr_pq_impl_lut,
                   int* site_curr_pq_impl_ff,
                   int* site_curr_pq_impl_cksr,
                   int* site_curr_pq_impl_ce,
                   int* site_curr_pq_idx,
                   int* site_curr_stable,
                   int* site_curr_pq_siteId,
                   T* site_curr_pq_score,
                   T* site_curr_scl_score,
                   int* site_curr_scl_siteId,
                   int* site_curr_scl_idx,
                   int* site_curr_scl_validIdx,
                   int* site_curr_scl_sig_idx,
                   int* site_curr_scl_sig,
                   int* site_curr_scl_impl_lut,
                   int* site_curr_scl_impl_ff,
                   int* site_curr_scl_impl_cksr,
                   int* site_curr_scl_impl_ce,
                   int* site_next_pq_idx,
                   int* site_next_pq_validIdx,
                   int* site_next_pq_top_idx,
                   T* site_next_pq_score,
                   int* site_next_pq_siteId,
                   int* site_next_pq_sig_idx,
                   int* site_next_pq_sig,
                   int* site_next_pq_impl_lut,
                   int* site_next_pq_impl_ff,
                   int* site_next_pq_impl_cksr,
                   int* site_next_pq_impl_ce,
                   T* site_next_scl_score,
                   int* site_next_scl_siteId,
                   int* site_next_scl_idx,
                   int* site_next_scl_validIdx,
                   int* site_next_scl_sig_idx,
                   int* site_next_scl_sig,
                   int* site_next_scl_impl_lut,
                   int* site_next_scl_impl_ff,
                   int* site_next_scl_impl_cksr,
                   int* site_next_scl_impl_ce,
                   int* site_next_stable,
                   T* site_det_score,
                   int* inst_curr_detSite,
                   T* inst_next_bestScoreImprov,
                   int* inst_next_bestSite,
                   int* inst_score_improv,
                   int* site_score_improv
                   )
{
    for (int sId = threadIdx.x + blockDim.x * blockIdx.x; sId < num_clb_sites; sId += blockDim.x*gridDim.x)
    {
        const int sIdx  = sorted_clb_siteIds ? sorted_clb_siteIds[sId] : sId;

        int siteId = addr2site_map[sIdx];

        int sPQ(sIdx*PQ_IDX), sSCL(sIdx*SCL_IDX), sNbrIdx(sIdx*maxList);

        // (e) createNewCandidates(site)
        //Generate new candidates by merging site_nbr to site_curr_scl
        const int limit_x = site_curr_scl_idx[sIdx];
        const int limit_y = site_nbr_idx[sIdx];
        ///RESTRICTED NEW CANDIDATE SPACE EXPLORATION SET TO SLICE/8 or SLICE_CAPACITY/4
        ///FOR ISPD'2016 BENCHMARKS, SLICE=32 and SLICE_CAPACITY=16
        const int limit_cands = DREAMPLACE_STD_NAMESPACE::min(SLICE_CAPACITY/4,limit_x*limit_y);

        for (int scsIdx = 0; scsIdx < limit_cands; ++scsIdx)
        {
            int sclId = scsIdx/limit_y;
            int snIdx = scsIdx/limit_x;
            int siteCurrIdx = validIndices_curr_scl[sSCL + sclId];

            /////
            int sCKRId = siteCurrIdx*CKSR_IN_CLB;
            int sCEId = siteCurrIdx*CE_IN_CLB;
            int sFFId = siteCurrIdx*SLICE_CAPACITY;
            int sGId = siteCurrIdx*SIG_IDX;

            T nwCand_score = site_curr_scl_score[siteCurrIdx];
            int nwCand_siteId = site_curr_scl_siteId[siteCurrIdx];
            int nwCand_sigIdx = site_curr_scl_sig_idx[siteCurrIdx];

            //array instantiation
            int nwCand_sig[32];
            int nwCand_lut[SLICE_CAPACITY];
            int nwCand_ff[SLICE_CAPACITY];
            int nwCand_ce[4];
            int nwCand_cksr[2];

            for (int sg = 0; sg < nwCand_sigIdx; ++sg)
            {
                nwCand_sig[sg] = site_curr_scl_sig[sGId + sg];
            }
            for (int sg = 0; sg < SLICE_CAPACITY; ++sg)
            {
                nwCand_lut[sg] = site_curr_scl_impl_lut[sFFId + sg];
                nwCand_ff[sg] = site_curr_scl_impl_ff[sFFId + sg];
            }
            for (int sg = 0; sg < CKSR_IN_CLB; ++sg)
            {
                nwCand_cksr[sg] = site_curr_scl_impl_cksr[sCKRId + sg];
            }
            for (int sg = 0; sg < CE_IN_CLB; ++sg)
            {
                nwCand_ce[sg] = site_curr_scl_impl_ce[sCEId + sg];
            }

            int instId = site_nbr[sNbrIdx + snIdx];
            int instPcl = instId*3;

            int add_inst_to_sig = INVALID;
            if (nwCand_sigIdx >= 2*SLICE_CAPACITY)
            {
                add_inst_to_sig = 0;
            }

            if (add_inst_to_sig == INVALID)
            {
                int itA(0), itB(0);
                int endA(nwCand_sigIdx), endB(flat_node2prclstrCount[instId]);

                if (endA > 0)
                {
                    int temp[32]; //Max capacity is 16 FFs + 16 LUTs for Ultrascale
                    int mIdx(0);
                    while (itA != endA && itB != endB)
                    {
                        if(sorted_node_map[nwCand_sig[itA]] < sorted_node_map[flat_node2precluster_map[instPcl+itB]])
                        {
                            temp[mIdx] = nwCand_sig[itA];
                            ++mIdx;
                            ++itA;
                        }
                        else if(sorted_node_map[nwCand_sig[itA]] > sorted_node_map[flat_node2precluster_map[instPcl+itB]])
                        {
                            temp[mIdx] = flat_node2precluster_map[instPcl+itB];
                            ++mIdx;
                            ++itB;
                        } else
                        {
                            add_inst_to_sig = 0;
                            break;
                        }
                    }
                    if (add_inst_to_sig == INVALID)
                    {
                        if (itA == endA)
                        {
                            for (int mBIdx = itB; mBIdx < endB; ++mBIdx)
                            {
                                temp[mIdx] = flat_node2precluster_map[instPcl+mBIdx];
                                ++mIdx;
                            }
                        } else
                        {
                            for (int mAIdx = itA; mAIdx < endA; ++mAIdx)
                            {
                                temp[mIdx] = nwCand_sig[mAIdx];
                                ++mIdx;
                            }
                        }

                        nwCand_sigIdx = mIdx;
                        for (int smIdx = 0; smIdx < mIdx; ++smIdx)
                        {
                            nwCand_sig[smIdx] = temp[smIdx];
                        }
                        add_inst_to_sig = 1;
                    }
                } else
                {
                    for (int mBIdx = itB; mBIdx < endB; ++mBIdx)
                    {
                        nwCand_sig[nwCand_sigIdx] = flat_node2precluster_map[instPcl+mBIdx];
                        ++nwCand_sigIdx;
                    }
                    add_inst_to_sig = 1;
                }
            }

            if (add_inst_to_sig == 1)
            {
                //check cand sig is in site_next_pq
                int candSigInSiteNextPQ = INVALID;
                for (int i = 0; i < PQ_IDX; ++i)
                {
                    int sigIdx = sPQ + i;
                    if (site_next_pq_validIdx[sigIdx] != INVALID)
                    {
                        if (site_next_pq_sig_idx[sigIdx] == nwCand_sigIdx)
                        {
                            int pqIdx(sigIdx*SIG_IDX), mtch(0);

                            for (int k = 0; k < nwCand_sigIdx; ++k)
                            {
                                for (int l = 0; l < nwCand_sigIdx; ++l)
                                {
                                    if (site_next_pq_sig[pqIdx + l] == nwCand_sig[k])
                                    {
                                        ++mtch;
                                        break;
                                    }
                                }
                            }
                            if (mtch == nwCand_sigIdx)
                            {
                                candSigInSiteNextPQ = 1;
                                break;
                            }
                        }
                    }
                }

                if (candSigInSiteNextPQ == INVALID &&
                        add_inst_to_cand_impl(lut_type, flat_node2pin_start_map, flat_node2pin_map, node2pincount, net2pincount, pin2net_map, pin_typeIds, sorted_net_map, flat_node2prclstrCount, flat_node2precluster_map, flop2ctrlSetId_map, node2fence_region_map, flop_ctrlSets, instId, CKSR_IN_CLB, CE_IN_CLB, HALF_SLICE_CAPACITY, NUM_BLE_PER_SLICE, nwCand_lut, nwCand_ff, nwCand_cksr, nwCand_ce))
                {
                    compute_candidate_score(pos_x, pos_y, pin_offset_x, pin_offset_y, net_bbox, net_weights, net_pinIdArrayX, net_pinIdArrayY, flat_net2pin_start_map, flat_node2pin_start_map, flat_node2pin_map, sorted_net_map, pin2net_map, pin2node_map, net2pincount, site_xy, xWirelenWt, yWirelenWt, extNetCountWt, wirelenImprovWt, netShareScoreMaxNetDegree, wlscoreMaxNetDegree, nwCand_sig, nwCand_siteId, nwCand_sigIdx, nwCand_score);

                    int nxtId(INVALID);

                    if (site_next_pq_idx[sIdx] < PQ_IDX)
                    {
                        for (int vId = 0; vId < PQ_IDX; ++vId)
                        {
                            if (site_next_pq_validIdx[sPQ+vId] == INVALID)
                            {
                                nxtId = vId;
                                ++site_next_pq_idx[sIdx];
                                break;
                            }
                        }
                    } else
                    {
                        //find least score and replace if current score is greater
                        T ckscore(nwCand_score);
                        for (int vId = 0; vId < PQ_IDX; ++vId)
                        {
                            if (ckscore > site_next_pq_score[sPQ + vId])
                            {
                                ckscore = site_next_pq_score[sPQ + vId]; 
                                nxtId = vId;
                            }
                        }
                    }

                    if (nxtId != INVALID)
                    {
                        int nTId = sPQ + nxtId;
                        int nCKRId = nTId*CKSR_IN_CLB;
                        int nCEId = nTId*CE_IN_CLB;
                        int nFFId = nTId*SLICE_CAPACITY;
                        int nSGId = nTId*SIG_IDX;

                        /////
                        site_next_pq_validIdx[nTId] = 1;
                        site_next_pq_score[nTId] = nwCand_score;
                        site_next_pq_siteId[nTId] = nwCand_siteId;
                        site_next_pq_sig_idx[nTId] = nwCand_sigIdx;

                        for (int sg = 0; sg < nwCand_sigIdx; ++sg)
                        {
                            site_next_pq_sig[nSGId + sg] = nwCand_sig[sg];
                        }
                        for (int sg = 0; sg < SLICE_CAPACITY; ++sg)
                        {
                            site_next_pq_impl_lut[nFFId + sg] = nwCand_lut[sg];
                            site_next_pq_impl_ff[nFFId + sg] = nwCand_ff[sg];
                        }
                        for (int sg = 0; sg < CKSR_IN_CLB; ++sg)
                        {
                            site_next_pq_impl_cksr[nCKRId + sg] = nwCand_cksr[sg];
                        }
                        for (int sg = 0; sg < CE_IN_CLB; ++sg)
                        {
                            site_next_pq_impl_ce[nCEId + sg] = nwCand_ce[sg];
                        }
                        /////

                        if (site_next_pq_idx[sIdx] == 1 || 
                                nwCand_score > site_next_pq_score[sPQ + site_next_pq_top_idx[sIdx]])
                        {
                            site_next_pq_top_idx[sIdx] = nxtId;
                        }

                        nxtId = INVALID;

                        if (site_next_scl_idx[sIdx] < SCL_IDX)
                        {
                            for (int vId = 0; vId < SCL_IDX; ++vId)
                            {
                                if (site_next_scl_validIdx[sSCL+vId] == INVALID)
                                {
                                    nxtId = vId;
                                    ++site_next_scl_idx[sIdx];
                                    break;
                                }
                            }
                        } else
                        {
                            //find least score and replace if current score is greater
                            T ckscore(nwCand_score);
                            for (int vId = 0; vId < SCL_IDX; ++vId)
                            {
                                if (ckscore > site_next_scl_score[sSCL+vId])
                                {
                                    ckscore = site_next_scl_score[sSCL+vId]; 
                                    nxtId = vId;
                                }
                            }
                        }

                        if (nxtId != INVALID)
                        {
                            /////
                            nTId = sSCL + nxtId;
                            nCKRId = nTId*CKSR_IN_CLB;
                            nCEId = nTId*CE_IN_CLB;
                            nFFId = nTId*SLICE_CAPACITY;
                            nSGId = nTId*SIG_IDX;

                            site_next_scl_validIdx[nTId] = 1;
                            site_next_scl_score[nTId] = nwCand_score;
                            site_next_scl_siteId[nTId] = nwCand_siteId;
                            site_next_scl_sig_idx[nTId] = nwCand_sigIdx;

                            for (int sg = 0; sg < nwCand_sigIdx; ++sg)
                            {
                                site_next_scl_sig[nSGId + sg] = nwCand_sig[sg];
                            }
                            for (int sg = 0; sg < SLICE_CAPACITY; ++sg)
                            {
                                site_next_scl_impl_lut[nFFId + sg] = nwCand_lut[sg];
                                site_next_scl_impl_ff[nFFId + sg] = nwCand_ff[sg];
                            }
                            for (int sg = 0; sg < CKSR_IN_CLB; ++sg)
                            {
                                site_next_scl_impl_cksr[nCKRId + sg] = nwCand_cksr[sg];
                            }
                            for (int sg = 0; sg < CE_IN_CLB; ++sg)
                            {
                                site_next_scl_impl_ce[nCEId + sg] = nwCand_ce[sg];
                            }
                            /////
                        }
                    }
                }
            }
        }

        //Remove all candidates in scl that is worse than the worst candidate in PQ
        if (site_next_pq_idx[sIdx] > 0)
        {
            //Find worst candidate in PQ
            T ckscore(site_next_pq_score[sPQ + site_next_pq_top_idx[sIdx]]);

            for (int vId = 0; vId < PQ_IDX; ++vId)
            {
                int nPQId = sPQ + vId;
                if (site_next_pq_validIdx[nPQId] != INVALID)
                {
                    if (ckscore > site_next_pq_score[nPQId])
                    {
                        ckscore = site_next_pq_score[nPQId]; 
                    }
                }
            }

            //Invalidate worst ones in scl
            int sclCount = 0;
            int maxEntries = site_next_scl_idx[sIdx];
            for (int ckId = 0; ckId < SCL_IDX; ++ckId)
            {
                int vId = sSCL + ckId;
                if (site_next_scl_validIdx[vId] != INVALID)
                {
                    if (ckscore > site_next_scl_score[vId])
                    {
                        //Clear contents thoroughly
                        clear_cand_contents(
                                vId, SIG_IDX, CKSR_IN_CLB, CE_IN_CLB,
                                site_next_scl_sig_idx, site_next_scl_sig,
                                site_next_scl_impl_lut, site_next_scl_impl_ff,
                                site_next_scl_impl_cksr, site_next_scl_impl_ce);

                        site_next_scl_validIdx[vId] = INVALID;
                        site_next_scl_sig_idx[vId] = 0;
                        site_next_scl_siteId[vId] = INVALID;
                        site_next_scl_score[vId] = 0.0;
                        --site_next_scl_idx[sIdx];
                    }
                    ++sclCount;
                    if (sclCount == maxEntries)
                    {
                        break;
                    }
                }
            }
        }

        //Update stable Iteration count
        if (site_curr_pq_idx[sIdx] > 0 && site_next_pq_idx[sIdx] > 0 && 
                compare_pq_tops(site_curr_pq_score, site_curr_pq_top_idx,
                    site_curr_pq_validIdx, site_curr_pq_siteId, site_curr_pq_sig_idx,
                    site_curr_pq_sig, site_curr_pq_impl_lut, site_curr_pq_impl_ff,
                    site_curr_pq_impl_cksr, site_curr_pq_impl_ce, site_next_pq_score,
                    site_next_pq_top_idx, site_next_pq_validIdx, site_next_pq_siteId,
                    site_next_pq_sig_idx, site_next_pq_sig, site_next_pq_impl_lut,
                    site_next_pq_impl_ff, site_next_pq_impl_cksr, site_next_pq_impl_ce,
                    sIdx, sPQ, SIG_IDX, CKSR_IN_CLB, CE_IN_CLB))
        {
            site_next_stable[sIdx] = site_curr_stable[sIdx] + 1;
        } else
        {
            site_next_stable[sIdx] = 0;
        }

        //// (f) broadcastTopCandidate(site)
        if (site_next_pq_idx[sIdx] > 0)
        {
            int topIdx = sPQ + site_next_pq_top_idx[sIdx];
            int topSigId = topIdx*SIG_IDX;

            T scoreImprov = site_next_pq_score[topIdx] - site_det_score[sIdx];

            ////UPDATED SEQUENTIAL PORTION
            int scoreImprovInt = DREAMPLACE_STD_NAMESPACE::max(int(scoreImprov*10000), intMinVal);
            site_score_improv[sIdx] = scoreImprovInt + siteId;

            for (int ssIdx = 0; ssIdx < site_next_pq_sig_idx[topIdx]; ++ssIdx)
            {
                int instId = site_next_pq_sig[topSigId + ssIdx];

                if (inst_curr_detSite[instId] == INVALID)
                {
                    atomicMax(&inst_score_improv[instId], scoreImprovInt);
                }
            }
        }
    }
}

//runDLSyncSites
template <typename T>
__global__ void runDLSyncSites(
                               const int* site_nbrRanges_idx,
                               const int* site_nbrGroup_idx,
                               const int* addr2site_map,
                               const int num_clb_sites,
                               const int CKSR_IN_CLB,
                               const int CE_IN_CLB,
                               const int SCL_IDX,
                               const int SIG_IDX,
                               int* site_curr_pq_top_idx,
                               int* site_curr_pq_sig_idx,
                               int* site_curr_pq_sig,
                               int* site_curr_pq_idx,
                               int* site_curr_stable,
                               int* site_curr_pq_validIdx,
                               int* site_curr_pq_siteId,
                               T* site_curr_pq_score,
                               int* site_curr_pq_impl_lut,
                               int* site_curr_pq_impl_ff,
                               int* site_curr_pq_impl_cksr,
                               int* site_curr_pq_impl_ce,
                               T* site_curr_scl_score,
                               int* site_curr_scl_siteId,
                               int* site_curr_scl_idx,
                               int* site_curr_scl_validIdx,
                               int* site_curr_scl_sig_idx,
                               int* site_curr_scl_sig,
                               int* site_curr_scl_impl_lut,
                               int* site_curr_scl_impl_ff,
                               int* site_curr_scl_impl_cksr,
                               int* site_curr_scl_impl_ce,
                               int* site_next_pq_validIdx,
                               T* site_next_pq_score,
                               int* site_next_pq_top_idx,
                               int* site_next_pq_siteId,
                               int* site_next_pq_sig_idx,
                               int* site_next_pq_sig,
                               int* site_next_pq_idx,
                               int* site_next_pq_impl_lut,
                               int* site_next_pq_impl_ff,
                               int* site_next_pq_impl_cksr,
                               int* site_next_pq_impl_ce,
                               T* site_next_scl_score,
                               int* site_next_scl_siteId,
                               int* site_next_scl_idx,
                               int* site_next_scl_validIdx,
                               int* site_next_scl_sig_idx,
                               int* site_next_scl_sig,
                               int* site_next_scl_impl_lut,
                               int* site_next_scl_impl_ff,
                               int* site_next_scl_impl_cksr,
                               int* site_next_scl_impl_ce,
                               int* site_next_stable,
                               int* activeStatus)
{
    int sIdx = threadIdx.x + blockDim.x * blockIdx.x;
    while(sIdx < num_clb_sites)
    {
        int numNbrGroups = (site_nbrRanges_idx[sIdx] == 0) ? 0 : site_nbrRanges_idx[sIdx]-1;
        int sPQ = sIdx*SCL_IDX;

        site_curr_stable[sIdx] = site_next_stable[sIdx];

        int curr_scl_size = site_curr_scl_idx[sIdx];
        site_curr_scl_idx[sIdx] = 0;
        int sclCount = 0;

        //Include valid entries of site.next.scl to site.curr.scl
        if (site_next_scl_idx[sIdx] > 0)
        {
            for (int id = 0; id < SCL_IDX; ++id)
            {
                int vIdx = sPQ+id;
                if (site_next_scl_validIdx[vIdx] != INVALID)
                {
                    int currId = sPQ+site_curr_scl_idx[sIdx];

                    site_curr_scl_validIdx[currId] = 1;
                    site_curr_scl_siteId[currId] = site_next_scl_siteId[vIdx];
                    site_curr_scl_score[currId] = site_next_scl_score[vIdx];
                    site_curr_scl_sig_idx[currId] = site_next_scl_sig_idx[vIdx];

                    int currFFId(currId*SLICE_CAPACITY), nxtFFId(vIdx*SLICE_CAPACITY);
                    int currCKId(currId*CKSR_IN_CLB), nxtCKId(vIdx*CKSR_IN_CLB);
                    int currCEId(currId*CE_IN_CLB), nxtCEId(vIdx*CE_IN_CLB);
                    int currSGId(currId*SIG_IDX), nxtSGId(vIdx*SIG_IDX);

                    for (int sg = 0; sg < site_next_scl_sig_idx[vIdx]; ++sg)
                    {
                        site_curr_scl_sig[currSGId + sg] = site_next_scl_sig[nxtSGId + sg];
                    }
                    for (int sg = 0; sg < SLICE_CAPACITY; ++sg)
                    {
                        site_curr_scl_impl_lut[currFFId + sg] = site_next_scl_impl_lut[nxtFFId + sg];
                        site_curr_scl_impl_ff[currFFId + sg] = site_next_scl_impl_ff[nxtFFId + sg];
                    }
                    for (int sg = 0; sg < CKSR_IN_CLB; ++sg)
                    {
                        site_curr_scl_impl_cksr[currCKId + sg]  = site_next_scl_impl_cksr[nxtCKId + sg];
                    }
                    for (int sg = 0; sg < CE_IN_CLB; ++sg)
                    {
                        site_curr_scl_impl_ce[currCEId + sg] = site_next_scl_impl_ce[nxtCEId + sg];
                    }
                    ++site_curr_scl_idx[sIdx];
                    ++sclCount;
                    if (sclCount == site_next_scl_idx[sIdx])
                    {
                        break;
                    }
                }
            }
        }

        ////Invalidate the rest in site.curr.scl
        if (curr_scl_size > site_next_scl_idx[sIdx])
        {
            for (int ckId = site_curr_scl_idx[sIdx]; ckId < SCL_IDX; ++ckId)
            {
                int vIdx = sPQ+ckId;
                if (site_curr_scl_validIdx[vIdx] != INVALID)
                {
                    site_curr_scl_validIdx[vIdx] = INVALID;
                    site_curr_scl_sig_idx[vIdx] = 0;
                    site_curr_scl_siteId[vIdx] = INVALID;
                    site_curr_scl_score[vIdx] = T(0.0);
                    ++sclCount;
                    if (sclCount == curr_scl_size)
                    {
                        break;
                    }
                }
            }
        }

        int curr_pq_size = site_curr_pq_idx[sIdx];
        site_curr_pq_idx[sIdx] = 0;
        site_curr_pq_top_idx[sIdx] = INVALID;

        sPQ = sIdx*PQ_IDX;
        sclCount = 0;
        //Include valid entries of site.next.pq to site.curr.pq
        if (site_next_pq_idx[sIdx] > 0)
        {
            for (int id = 0; id < PQ_IDX; ++id)
            {
                int vIdx = sPQ+id;
                if (site_next_pq_validIdx[vIdx] != INVALID)
                {
                    int currId = sPQ+site_curr_pq_idx[sIdx];

                    site_curr_pq_validIdx[currId] = 1;
                    site_curr_pq_siteId[currId] = site_next_pq_siteId[vIdx];
                    site_curr_pq_score[currId] = site_next_pq_score[vIdx];
                    site_curr_pq_sig_idx[currId] = site_next_pq_sig_idx[vIdx];

                    int currFFId(currId*SLICE_CAPACITY), nxtFFId(vIdx*SLICE_CAPACITY);
                    int currCKId(currId*CKSR_IN_CLB), nxtCKId(vIdx*CKSR_IN_CLB);
                    int currCEId(currId*CE_IN_CLB), nxtCEId(vIdx*CE_IN_CLB);
                    int currSGId(currId*SIG_IDX), nxtSGId(vIdx*SIG_IDX);

                    for (int sg = 0; sg < site_next_pq_sig_idx[vIdx]; ++sg)
                    {
                        site_curr_pq_sig[currSGId + sg] = site_next_pq_sig[nxtSGId + sg];
                    }
                    for (int sg = 0; sg < SLICE_CAPACITY; ++sg)
                    {
                        site_curr_pq_impl_lut[currFFId + sg] = site_next_pq_impl_lut[nxtFFId + sg];
                        site_curr_pq_impl_ff[currFFId + sg] = site_next_pq_impl_ff[nxtFFId + sg];
                    }
                    for (int sg = 0; sg < CKSR_IN_CLB; ++sg)
                    {
                        site_curr_pq_impl_cksr[currCKId + sg]  = site_next_pq_impl_cksr[nxtCKId + sg];
                    }
                    for (int sg = 0; sg < CE_IN_CLB; ++sg)
                    {
                        site_curr_pq_impl_ce[currCEId + sg] = site_next_pq_impl_ce[nxtCEId + sg];
                    }
                    if (id == site_next_pq_top_idx[sIdx])
                    {
                        site_curr_pq_top_idx[sIdx] = site_curr_pq_idx[sIdx];
                    }
                    ++site_curr_pq_idx[sIdx];
                    ++sclCount;
                    if (sclCount == site_next_pq_idx[sIdx])
                    {
                        break;
                    }
                }
            }
        }

        //Invalidate the rest in site.curr.pq
        if (curr_pq_size > site_next_pq_idx[sIdx])
        {
            for (int ckId = site_curr_pq_idx[sIdx]; ckId < PQ_IDX; ++ckId)
            {
                int vIdx = sPQ+ckId;
                if (site_curr_pq_validIdx[vIdx] != INVALID)
                {
                    site_curr_pq_validIdx[vIdx] = INVALID;
                    site_curr_pq_sig_idx[vIdx] = 0;
                    site_curr_pq_siteId[vIdx] = INVALID;
                    site_curr_pq_score[vIdx] = T(0.0);
                    ++sclCount;
                    if (sclCount == curr_pq_size)
                    {
                        break;
                    }
                }
            }
        }

        sPQ = sIdx*SCL_IDX;
        //sclCount = 0;
        for (int ckId = 0; ckId < SCL_IDX; ++ckId)
        {
            int vIdx = sPQ+ckId;
            //if (site_next_scl_validIdx[vIdx] != INVALID)
            //{
            //Clear contents thoroughly
            clear_cand_contents(
                    vIdx, SIG_IDX, CKSR_IN_CLB, CE_IN_CLB,
                    site_next_scl_sig_idx, site_next_scl_sig,
                    site_next_scl_impl_lut, site_next_scl_impl_ff,
                    site_next_scl_impl_cksr, site_next_scl_impl_ce);

            site_next_scl_validIdx[vIdx] = INVALID;
            site_next_scl_sig_idx[vIdx] = 0;
            site_next_scl_siteId[vIdx] = INVALID;
            site_next_scl_score[vIdx] = 0.0;
            //++sclCount;
            //if (sclCount == site_next_scl_idx[sIdx])
            //{
            //    break;
            //}
            //}
        }
        site_next_scl_idx[sIdx] = 0;

        activeStatus[addr2site_map[sIdx]] = (site_curr_pq_idx[sIdx] > 0 || site_curr_scl_idx[sIdx] > 0 || site_nbrGroup_idx[sIdx] < numNbrGroups) ? 1: 0;

        sIdx += blockDim.x * gridDim.x;
    }
}

//runDLSyncInsts
template <typename T>
__global__ void runDLSyncInsts(const T* pos_x,
                               const T* pos_y,
                               const T* site_xy,
                               const int* spiral_accessor,
                               const int* site_types,
                               const int* site2addr_map,
                               const int* site_score_improv,
                               const int* lut_flop_indices,
                               int* site_curr_pq_top_idx,
                               int* site_curr_pq_sig_idx,
                               int* site_curr_pq_sig,
                               int* site_curr_pq_idx,
                               const T maxDist,
                               const int spiralBegin,
                               const int spiralEnd,
                               const int intMinVal,
                               const int num_nodes,
                               const int num_sites_x,
                               const int num_sites_y,
                               const int maxSites,
                               const int SIG_IDX,
                               int* inst_score_improv,
                               int* inst_curr_detSite,
                               int* inst_curr_bestSite,
                               T* inst_curr_bestScoreImprov,
                               int* inst_next_detSite,
                               int* inst_next_bestSite,
                               T* inst_next_bestScoreImprov,
                               int* illegalStatus)

{
    int nIdx = threadIdx.x + blockDim.x * blockIdx.x;
    while(nIdx < num_nodes)
    {
        const int nodeId = lut_flop_indices[nIdx];
        if (inst_curr_detSite[nodeId] == INVALID) //Only LUT/FF
        {
            //POST PROCESSING TO IDENTIFY INSTANCE BEST SITE
            //REPLACEMENT FOR SEQUENTIAL PORTION
            if (inst_score_improv[nodeId] > intMinVal)
            {
                int bestSite = maxSites;
                int &instScoreImprov = inst_score_improv[nodeId];
                T instScoreImprovT =  T(instScoreImprov/10000.0);
                T posX = pos_x[nodeId];
                T posY = pos_y[nodeId];

                for (int spIdx = spiralBegin; spIdx < spiralEnd; ++spIdx)
                {
                    int saIdx = spIdx*2;
                    int xVal = posX + spiral_accessor[saIdx];
                    int yVal = posY + spiral_accessor[saIdx+1];

                    int siteId = xVal * num_sites_y + yVal;

                    if (xVal >= 0 && xVal < num_sites_x && yVal >= 0 && yVal < num_sites_y && site_types[siteId] == 1) //Site is within die and of type CLB
                    {
                        int stMpId = siteId *2;
                        int sIdx = site2addr_map[siteId];
                        int tsPQ = sIdx*PQ_IDX + site_curr_pq_top_idx[sIdx];
                        int topIdx = tsPQ*SIG_IDX;
                        int site_score = site_score_improv[sIdx] - siteId;

                        T dist = DREAMPLACE_STD_NAMESPACE::abs(posX - site_xy[stMpId]) + DREAMPLACE_STD_NAMESPACE::abs(posY - site_xy[stMpId+1]);

                        if (instScoreImprov == site_score && site_curr_pq_idx[sIdx] > 0)
                        {
                            for (int idx = 0; idx < site_curr_pq_sig_idx[tsPQ]; ++idx)
                            {
                                if (site_curr_pq_sig[topIdx+idx] == nodeId && siteId < bestSite && dist < maxDist)
                                {
                                    bestSite = siteId;
                                    inst_next_bestSite[nodeId] = siteId;
                                    inst_next_bestScoreImprov[nodeId] = instScoreImprovT;
                                }
                            }
                        }
                    }
                }
                instScoreImprov = intMinVal;
            }
            //END post processing


            inst_curr_detSite[nodeId] = inst_next_detSite[nodeId];
            inst_curr_bestSite[nodeId] = inst_next_bestSite[nodeId];
            inst_curr_bestScoreImprov[nodeId] = inst_next_bestScoreImprov[nodeId];

            inst_next_bestSite[nodeId] = INVALID;
            inst_next_bestScoreImprov[nodeId] = T(-10000.0);

            illegalStatus[nodeId] = (inst_curr_detSite[nodeId] == INVALID) ? 1 : 0;
        }
        nIdx += blockDim.x * gridDim.x;
    }
}

//init nets and precluster
template <typename T>
int initNetsClstrCuda(const T *pos_x,
                      const T *pos_y,
                      const T *pin_offset_x,
                      const T *pin_offset_y,
                      const int *sorted_node_map,
                      const int *sorted_node_idx,
                      const int *sorted_net_idx,
                      const int *flat_net2pin_map,
                      const int *flat_net2pin_start_map,
                      const int *flop2ctrlSetId_map,
                      const int *flop_ctrlSets,
                      const int *node2fence_region_map,
                      const int *node2outpinIdx_map,
                      const int *pin2net_map,
                      const int *pin2node_map,
                      const int *pin_typeIds,
                      const int *net2pincount,
                      const T preClusteringMaxDist,
                      const int num_nodes,
                      const int num_nets,
                      const int wlscoreMaxNetDegree,
                      T *net_bbox,
                      int *net_pinIdArrayX,
                      int *net_pinIdArrayY,
                      int *flat_node2precluster_map,
                      int *flat_node2prclstrCount)
{
    int block_count = ceilDiv(num_nets + THREAD_COUNT-1, THREAD_COUNT);
    initNets<<<block_count, THREAD_COUNT>>>(pos_x,
                                            pos_y,
                                            sorted_net_idx,
                                            flat_net2pin_map,
                                            flat_net2pin_start_map,
                                            pin2node_map,
                                            pin_offset_x,
                                            pin_offset_y,
                                            net2pincount,
                                            num_nets,
                                            net_bbox,
                                            net_pinIdArrayX,
                                            net_pinIdArrayY,
                                            wlscoreMaxNetDegree);
    int nodes_block_count = ceilDiv(num_nodes + THREAD_COUNT-1, THREAD_COUNT);
    preClustering<<<nodes_block_count, THREAD_COUNT>>>(pos_x,
                                                 pos_y,
                                                 pin_offset_x,
                                                 pin_offset_y,
                                                 sorted_node_map,
                                                 sorted_node_idx,
                                                 flat_net2pin_map,
                                                 flat_net2pin_start_map,
                                                 flop2ctrlSetId_map,
                                                 flop_ctrlSets,
                                                 node2fence_region_map,
                                                 node2outpinIdx_map,
                                                 pin2net_map,
                                                 pin2node_map,
                                                 pin_typeIds,
                                                 num_nodes,
                                                 preClusteringMaxDist,
                                                 flat_node2precluster_map,
                                                 flat_node2prclstrCount);

    cudaDeviceSynchronize();
    return 0;
}

//runDLIter
template <typename T>
int runDLIterCuda(const T* pos_x,
                  const T* pos_y,
                  const T* pin_offset_x,
                  const T* pin_offset_y,
                  const T* net_bbox,
                  const T* site_xy,
                  const int* net_pinIdArrayX,
                  const int* net_pinIdArrayY,
                  const int* site_types,
                  const int* spiral_accessor,
                  const int* node2fence_region_map,
                  const int* lut_flop_indices,
                  const int* flop_ctrlSets,
                  const int* flop2ctrlSetId_map,
                  const int* lut_type,
                  const int* flat_node2pin_start_map,
                  const int* flat_node2pin_map,
                  const int* node2pincount,
                  const int* net2pincount,
                  const int* pin2net_map,
                  const int* pin_typeIds,
                  const int* flat_net2pin_start_map,
                  const int* pin2node_map,
                  const int* sorted_net_map,
                  const int* sorted_node_map,
                  const int* flat_node2prclstrCount,
                  const int* flat_node2precluster_map,
                  const int* site_nbrList,
                  const int* site_nbrRanges,
                  const int* site_nbrRanges_idx,
                  const T* net_weights,
                  const int* addr2site_map,
                  const int* site2addr_map,
                  const int num_sites_x,
                  const int num_sites_y,
                  const int num_clb_sites,
                  const int num_lutflops,
                  const int minStableIter,
                  const int maxList,
                  const int HALF_SLICE_CAPACITY,
                  const int NUM_BLE_PER_SLICE,
                  const int minNeighbors,
                  const int spiralBegin,
                  const int spiralEnd,
                  const int intMinVal,
                  const int numGroups,
                  const int netShareScoreMaxNetDegree,
                  const int wlscoreMaxNetDegree,
                  const T maxDist,
                  const T xWirelenWt,
                  const T yWirelenWt,
                  const T wirelenImprovWt,
                  const T extNetCountWt,
                  const int CKSR_IN_CLB,
                  const int CE_IN_CLB,
                  const int SCL_IDX,
                  const int SIG_IDX,
                  int* site_nbr_idx,
                  int* site_nbr,
                  int* site_nbrGroup_idx,
                  int* site_curr_pq_top_idx,
                  int* site_curr_pq_sig_idx,
                  int* site_curr_pq_sig,
                  int* site_curr_pq_idx,
                  int* site_curr_stable,
                  int* site_curr_pq_siteId,
                  int* site_curr_pq_validIdx,
                  T* site_curr_pq_score,
                  int* site_curr_pq_impl_lut,
                  int* site_curr_pq_impl_ff,
                  int* site_curr_pq_impl_cksr,
                  int* site_curr_pq_impl_ce,
                  T* site_curr_scl_score,
                  int* site_curr_scl_siteId,
                  int* site_curr_scl_idx,
                  int* cumsum_curr_scl,
                  int* site_curr_scl_validIdx,
                  int* validIndices_curr_scl,
                  int* site_curr_scl_sig_idx,
                  int* site_curr_scl_sig,
                  int* site_curr_scl_impl_lut,
                  int* site_curr_scl_impl_ff,
                  int* site_curr_scl_impl_cksr,
                  int* site_curr_scl_impl_ce,
                  int* site_next_pq_idx,
                  int* site_next_pq_validIdx,
                  int* site_next_pq_top_idx,
                  T* site_next_pq_score,
                  int* site_next_pq_siteId,
                  int* site_next_pq_sig_idx,
                  int* site_next_pq_sig,
                  int* site_next_pq_impl_lut,
                  int* site_next_pq_impl_ff,
                  int* site_next_pq_impl_cksr,
                  int* site_next_pq_impl_ce,
                  T* site_next_scl_score,
                  int* site_next_scl_siteId,
                  int* site_next_scl_idx,
                  int* site_next_scl_validIdx,
                  int* site_next_scl_sig_idx,
                  int* site_next_scl_sig,
                  int* site_next_scl_impl_lut,
                  int* site_next_scl_impl_ff,
                  int* site_next_scl_impl_cksr,
                  int* site_next_scl_impl_ce,
                  int* site_next_stable,
                  T* site_det_score,
                  int* site_det_siteId,
                  int* site_det_sig_idx,
                  int* site_det_sig,
                  int* site_det_impl_lut,
                  int* site_det_impl_ff,
                  int* site_det_impl_cksr,
                  int* site_det_impl_ce,
                  int* inst_curr_detSite,
                  T* inst_curr_bestScoreImprov,
                  int* inst_curr_bestSite,
                  int* inst_next_detSite,
                  T* inst_next_bestScoreImprov,
                  int* inst_next_bestSite,
                  int* activeStatus,
                  int* illegalStatus,
                  int* inst_score_improv,
                  int* site_score_improv,
                  int* sorted_clb_siteIds
                  )
{
    int block_count = (num_clb_sites + THREAD_COUNT-1)/THREAD_COUNT;


    //DLkernel split Implementation to enable rearranging
    runDLIteration_kernel_1<<<block_count, THREAD_COUNT>>>(node2fence_region_map, flop_ctrlSets, flop2ctrlSetId_map, lut_type, flat_node2pin_start_map, flat_node2pin_map, node2pincount, net2pincount, pin2net_map, pin_typeIds, sorted_net_map, flat_node2prclstrCount, flat_node2precluster_map, site_nbrList, site_nbrRanges, site_nbrRanges_idx, addr2site_map,
    num_clb_sites, minStableIter, maxList, HALF_SLICE_CAPACITY, NUM_BLE_PER_SLICE, minNeighbors, numGroups, CKSR_IN_CLB, CE_IN_CLB, SCL_IDX, SIG_IDX, site_nbr_idx, site_nbr, site_nbrGroup_idx, site_curr_pq_top_idx, site_curr_pq_sig_idx, site_curr_pq_sig, site_curr_pq_idx, site_curr_stable, site_curr_pq_siteId, site_curr_pq_score, site_curr_pq_impl_lut, site_curr_pq_impl_ff, site_curr_pq_impl_cksr, site_curr_pq_impl_ce, site_curr_scl_score, site_curr_scl_siteId, site_curr_scl_idx, site_curr_scl_validIdx, site_curr_scl_sig_idx, site_curr_scl_sig, site_curr_scl_impl_lut, site_curr_scl_impl_ff, site_curr_scl_impl_cksr, site_curr_scl_impl_ce, site_next_pq_idx, site_next_pq_validIdx, site_next_pq_top_idx, site_next_pq_score, site_next_pq_siteId, site_next_pq_sig_idx, site_next_pq_sig, site_next_pq_impl_lut, site_next_pq_impl_ff, site_next_pq_impl_cksr, site_next_pq_impl_ce,
    site_det_score, site_det_siteId, site_det_sig_idx, site_det_sig, site_det_impl_lut, site_det_impl_ff, site_det_impl_cksr, site_det_impl_ce, inst_curr_detSite, inst_curr_bestSite, inst_next_detSite, validIndices_curr_scl, cumsum_curr_scl);
    cudaDeviceSynchronize();

    //////Use thrust to sort cumsum_curr_scl to compute sorted siteIds based on load
    thrust::device_ptr<int> cumsum_ptr = thrust::device_pointer_cast(cumsum_curr_scl);
    thrust::device_ptr<int> sortedId_ptr = thrust::device_pointer_cast(sorted_clb_siteIds);

    thrust::sequence(sortedId_ptr, sortedId_ptr+num_clb_sites, 0);
    thrust::sort_by_key(cumsum_ptr, cumsum_ptr+num_clb_sites, sortedId_ptr, thrust::greater<int>());
    //Note: order of cumsum_curr_scl is also changed but it is not used in the next steps

    runDLIteration_kernel_2<<<block_count, THREAD_COUNT>>>(pos_x, pos_y, pin_offset_x, pin_offset_y, net_bbox, site_xy, net_pinIdArrayX, net_pinIdArrayY,
    node2fence_region_map, flop_ctrlSets, flop2ctrlSetId_map, lut_type, flat_node2pin_start_map, flat_node2pin_map, node2pincount, net2pincount, pin2net_map, pin_typeIds, flat_net2pin_start_map, pin2node_map, sorted_node_map, sorted_net_map, flat_node2prclstrCount, flat_node2precluster_map, net_weights, addr2site_map, validIndices_curr_scl, sorted_clb_siteIds, 
        intMinVal, num_clb_sites, maxList, HALF_SLICE_CAPACITY, NUM_BLE_PER_SLICE, netShareScoreMaxNetDegree, wlscoreMaxNetDegree, xWirelenWt, yWirelenWt, wirelenImprovWt, extNetCountWt, CKSR_IN_CLB, CE_IN_CLB, SCL_IDX, SIG_IDX, site_nbr_idx, site_nbr, site_curr_pq_top_idx, site_curr_pq_validIdx, site_curr_pq_sig_idx, site_curr_pq_sig, site_curr_pq_impl_lut, site_curr_pq_impl_ff, site_curr_pq_impl_cksr, site_curr_pq_impl_ce,
        site_curr_pq_idx, site_curr_stable, site_curr_pq_siteId, site_curr_pq_score, site_curr_scl_score, site_curr_scl_siteId, site_curr_scl_idx, site_curr_scl_validIdx, site_curr_scl_sig_idx, site_curr_scl_sig, site_curr_scl_impl_lut, site_curr_scl_impl_ff, site_curr_scl_impl_cksr, site_curr_scl_impl_ce, site_next_pq_idx, site_next_pq_validIdx, site_next_pq_top_idx, site_next_pq_score, site_next_pq_siteId, site_next_pq_sig_idx, site_next_pq_sig, site_next_pq_impl_lut, site_next_pq_impl_ff, site_next_pq_impl_cksr, site_next_pq_impl_ce, site_next_scl_score, site_next_scl_siteId, site_next_scl_idx, site_next_scl_validIdx, site_next_scl_sig_idx, site_next_scl_sig, site_next_scl_impl_lut, site_next_scl_impl_ff, site_next_scl_impl_cksr, site_next_scl_impl_ce, site_next_stable, site_det_score, inst_curr_detSite, inst_next_bestScoreImprov, inst_next_bestSite, inst_score_improv, site_score_improv);
    cudaDeviceSynchronize();

    ////TODO - Comment out single DLIter implementation - rearrange not possible
    //runDLIteration<<<block_count, THREAD_COUNT>>>(pos_x, pos_y, pin_offset_x, pin_offset_y, net_bbox, site_xy, net_pinIdArrayX, net_pinIdArrayY, node2fence_region_map, flop_ctrlSets, flop2ctrlSetId_map, lut_type, flat_node2pin_start_map, flat_node2pin_map, node2pincount, net2pincount, pin2net_map, pin_typeIds, flat_net2pin_start_map, pin2node_map, sorted_net_map, sorted_node_map, flat_node2prclstrCount, flat_node2precluster_map, site_nbrList, site_nbrRanges, site_nbrRanges_idx, net_weights, addr2site_map, num_clb_sites, minStableIter, maxList, HALF_SLICE_CAPACITY, NUM_BLE_PER_SLICE, minNeighbors, intMinVal, numGroups, netShareScoreMaxNetDegree, wlscoreMaxNetDegree, xWirelenWt, yWirelenWt, wirelenImprovWt, extNetCountWt, CKSR_IN_CLB, CE_IN_CLB, SCL_IDX, SIG_IDX, validIndices_curr_scl, site_nbr_idx, site_nbr, site_nbrGroup_idx, site_curr_pq_top_idx, site_curr_pq_validIdx, site_curr_pq_sig_idx, site_curr_pq_sig, site_curr_pq_idx, site_curr_stable, site_curr_pq_siteId, site_curr_pq_score, site_curr_pq_impl_lut, site_curr_pq_impl_ff, site_curr_pq_impl_cksr, site_curr_pq_impl_ce, site_curr_scl_score, site_curr_scl_siteId, site_curr_scl_idx, site_curr_scl_validIdx, site_curr_scl_sig_idx, site_curr_scl_sig, site_curr_scl_impl_lut, site_curr_scl_impl_ff, site_curr_scl_impl_cksr, site_curr_scl_impl_ce, site_next_pq_idx, site_next_pq_validIdx, site_next_pq_top_idx, site_next_pq_score, site_next_pq_siteId, site_next_pq_sig_idx, site_next_pq_sig, site_next_pq_impl_lut, site_next_pq_impl_ff, site_next_pq_impl_cksr, site_next_pq_impl_ce, site_next_scl_score, site_next_scl_siteId, site_next_scl_idx, site_next_scl_validIdx, site_next_scl_sig_idx, site_next_scl_sig, site_next_scl_impl_lut, site_next_scl_impl_ff, site_next_scl_impl_cksr, site_next_scl_impl_ce, site_next_stable, site_det_score, site_det_siteId, site_det_sig_idx, site_det_sig, site_det_impl_lut, site_det_impl_ff, site_det_impl_cksr, site_det_impl_ce, inst_curr_detSite, inst_curr_bestSite, inst_next_detSite, inst_next_bestScoreImprov, inst_next_bestSite, inst_score_improv, site_score_improv);

    //cudaDeviceSynchronize();

    int nodes_block_count = ceilDiv((num_lutflops + THREAD_COUNT - 1), THREAD_COUNT);
    int maxSites = num_sites_x*num_sites_y;

    runDLSyncInsts<<<nodes_block_count, THREAD_COUNT>>>(pos_x, pos_y, site_xy, spiral_accessor, site_types,
            site2addr_map, site_score_improv, lut_flop_indices, site_curr_pq_top_idx, site_curr_pq_sig_idx,
            site_curr_pq_sig, site_curr_pq_idx, maxDist, spiralBegin, spiralEnd, intMinVal, num_lutflops, num_sites_x,
            num_sites_y, maxSites, SIG_IDX, inst_score_improv, inst_curr_detSite, inst_curr_bestSite, inst_curr_bestScoreImprov, inst_next_detSite,
            inst_next_bestSite, inst_next_bestScoreImprov, illegalStatus);

    //printf("End of runDLSyncInsts");
    cudaDeviceSynchronize();

    runDLSyncSites<<<block_count, THREAD_COUNT>>>(//site_types, 
            site_nbrRanges_idx, site_nbrGroup_idx, addr2site_map, //num_sites,
            num_clb_sites, CKSR_IN_CLB, CE_IN_CLB, SCL_IDX, SIG_IDX, site_curr_pq_top_idx, site_curr_pq_sig_idx, site_curr_pq_sig,
            site_curr_pq_idx, site_curr_stable, site_curr_pq_validIdx, site_curr_pq_siteId, site_curr_pq_score,
            site_curr_pq_impl_lut, site_curr_pq_impl_ff, site_curr_pq_impl_cksr, site_curr_pq_impl_ce,
            site_curr_scl_score, site_curr_scl_siteId, site_curr_scl_idx, site_curr_scl_validIdx,
            site_curr_scl_sig_idx, site_curr_scl_sig, site_curr_scl_impl_lut, site_curr_scl_impl_ff,
            site_curr_scl_impl_cksr, site_curr_scl_impl_ce, site_next_pq_validIdx, site_next_pq_score,
            site_next_pq_top_idx, site_next_pq_siteId, site_next_pq_sig_idx, site_next_pq_sig, site_next_pq_idx, site_next_pq_impl_lut,
            site_next_pq_impl_ff, site_next_pq_impl_cksr, site_next_pq_impl_ce, site_next_scl_score,
            site_next_scl_siteId, site_next_scl_idx, site_next_scl_validIdx, site_next_scl_sig_idx,
            site_next_scl_sig, site_next_scl_impl_lut, site_next_scl_impl_ff, site_next_scl_impl_cksr,
            site_next_scl_impl_ce, site_next_stable, activeStatus);
    //printf("End of runDLSyncSites");
    cudaDeviceSynchronize();

    return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T)                                                                                \
    template int initNetsClstrCuda<T>                                                                              \
        (const T *pos_x, const T *pos_y, const T *pin_offset_x, const T *pin_offset_y,                             \
        const int *sorted_node_map, const int *sorted_node_idx, const int *sorted_net_idx,                         \
        const int *flat_net2pin_map, const int *flat_net2pin_start_map, const int *flop2ctrlSetId_map,             \
        const int *flop_ctrlSets, const int *node2fence_region_map, const int *node2outpinIdx_map,                 \
        const int *pin2net_map, const int *pin2node_map, const int *pin_typeIds, const int *net2pincount,          \
        const T preClusteringMaxDist, const int num_nodes, const int num_nets, const  int WLscoreMaxNetDegre,      \
        T *net_bbox, int *net_pinIdArrayX, int *net_pinIdArrayY, int *flat_node2precluster_map,                    \
        int *flat_node2prclstrCount);                                                                              \
                                                                                                                   \
    template int runDLIterCuda<T>                                                                                  \
        (const T* pos_x, const T* pos_y, const T* pin_offset_x, const T* pin_offset_y, const T* net_bbox,          \
        const T* site_xy, const int* net_pinIdArrayX, const int* net_pinIdArrayY, const int* site_types,           \
        const int* spiral_accessor, const int* node2fence_region_map, const int* lut_flop_indices,                 \
        const int* flop_ctrlSets, const int* flop2ctrlSetId_map, const int* lut_type,                              \
        const int* flat_node2pin_start_map, const int* flat_node2pin_map, const int* node2pincount,                \
        const int* net2pincount, const int* pin2net_map, const int* pin_typeIds, const int* flat_net2pin_start_map,\
        const int* pin2node_map, const int* sorted_net_map, const int* sorted_node_map,                            \
        const int* flat_node2prclstrCount, const int* flat_node2precluster_map, const int* site_nbrList,           \
        const int* site_nbrRanges, const int* site_nbrRanges_idx, const T* net_weights, const int* addr2site_map,  \
        const int* site2addr_map, const int num_sites_x, const int num_sites_y, const int num_clb_sites,           \
        const int num_lutflops, const int minStableIter, const int maxList, const int HALF_SLICE_CAPACITY,         \
        const int NUM_BLE_PER_SLICE, const int minNeighbors, const int spiralBegin, const int spiralEnd,           \
        const int intMinVal, const int numGroups, const int netShareScoreMaxNetDegree,                             \
        const int wlscoreMaxNetDegree, const T maxDist, const T xWirelenWt, const T yWirelenWt,                    \
        const T wirelenImprovWt, const T extNetCountWt, const int CKSR_IN_CLB, const int CE_IN_CLB,                \
        const int SCL_IDX, const int SIG_IDX, int* site_nbr_idx, int* site_nbr, int* site_nbrGroup_idx,            \
        int* site_curr_pq_top_idx, int* site_curr_pq_sig_idx, int* site_curr_pq_sig, int* site_curr_pq_idx,        \
        int* site_curr_stable, int* site_curr_pq_siteId, int* site_curr_pq_validIdx, T* site_curr_pq_score,        \
        int* site_curr_pq_impl_lut, int* site_curr_pq_impl_ff, int* site_curr_pq_impl_cksr,                        \
        int* site_curr_pq_impl_ce, T* site_curr_scl_score, int* site_curr_scl_siteId, int* site_curr_scl_idx,      \
        int* cumsum_curr_scl, int* site_curr_scl_validIdx, int* validIndices_curr_scl, int* site_curr_scl_sig_idx, \
        int* site_curr_scl_sig, int* site_curr_scl_impl_lut, int* site_curr_scl_impl_ff,                           \
        int* site_curr_scl_impl_cksr, int* site_curr_scl_impl_ce, int* site_next_pq_idx,                           \
        int* site_next_pq_validIdx, int* site_next_pq_top_idx, T* site_next_pq_score, int* site_next_pq_siteId,    \
        int* site_next_pq_sig_idx, int* site_next_pq_sig, int* site_next_pq_impl_lut, int* site_next_pq_impl_ff,   \
        int* site_next_pq_impl_cksr, int* site_next_pq_impl_ce, T* site_next_scl_score, int* site_next_scl_siteId, \
        int* site_next_scl_idx, int* site_next_scl_validIdx, int* site_next_scl_sig_idx, int* site_next_scl_sig,   \
        int* site_next_scl_impl_lut, int* site_next_scl_impl_ff, int* site_next_scl_impl_cksr,                     \
        int* site_next_scl_impl_ce, int* site_next_stable, T* site_det_score, int* site_det_siteId,                \
        int* site_det_sig_idx, int* site_det_sig, int* site_det_impl_lut, int* site_det_impl_ff,                   \
        int* site_det_impl_cksr, int* site_det_impl_ce, int* inst_curr_detSite, T* inst_curr_bestScoreImprov,      \
        int* inst_curr_bestSite, int* inst_next_detSite, T* inst_next_bestScoreImprov, int* inst_next_bestSite,    \
        int* activeStatus, int* illegalStatus, int* inst_score_improv, int* site_score_improv,                     \
        int* sorted_clb_siteIds);

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
