// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include "defs.h"
#include "RCGpuUtils.h"

//imp2 table points for KernelA
__device__ __constant__ u64 jmp2_table[8 * JMP_CNT];

#define BLOCK_CNT       gridDim.x
#define BLOCK_X         blockIdx.x
#define THREAD_X        threadIdx.x

extern __shared__ u64 LDS[];

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef OLD_GPU
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelA(const TKparams Kparams)
{
    const u32 kang_id = blockIdx.x * blockDim.x + threadIdx.x;
    u32 pnt_offset = THREAD_X;

    for (int i = 0; i < Kparams.GroupCnt; i++)
    {
        //load RndPnt for a kang
        u64* px = &LDS[pnt_offset * 12];
        u64* py = &LDS[pnt_offset * 12 + 4];
        u64* pr = &LDS[pnt_offset * 12 + 8];

        u64 rnd_offset = i * BLOCK_SIZE * Kparams.BlockCnt * 12 + (BLOCK_X * BLOCK_SIZE + THREAD_X) * 12;
        *((int4*)&px[0]) = *((int4*)&Kparams.RndPnts[rnd_offset / 8]);
        *((int4*)&py[0]) = *((int4*)&Kparams.RndPnts[rnd_offset / 8 + 4]);
        *((int4*)&pr[0]) = *((int4*)&Kparams.RndPnts[rnd_offset / 8 + 8]);
        __syncthreads();

        u64 X[4], Y[4], Z[4];
        u64 pX[4], pY[4], pZ[4];
        u64 Priv[4];
        EcPoint ecp;

        load_from_LDS_compressed(X, px, THREAD_X);
        load_from_LDS_compressed(Y, py, THREAD_X);
        load_from_LDS_compressed(Priv, pr, THREAD_X);
        pZ[0] = 1;

        u32 jmp_id, h;
        u64 Xc, Yc, Zc;
        u64 tmp[4];
        int j;
        EcInt PrivAcc;
        EcPoint PntAcc;

        for (j = 0; j < STEP_CNT; j++)
        {
            Xc = X[0] & 0xFFFFFFFFFFFFE000;
            h = (u32)(Xc >> (64 - Kparams.DP));
            jmp_id = h & JMP_MASK;

            ecp.x.data[0] = X[0];
            ecp.x.data[1] = X[1];
            ecp.x.data[2] = X[2];
            ecp.x.data[3] = X[3];
            ecp.y.data[0] = Y[0];
            ecp.y.data[1] = Y[1];
            ecp.y.data[2] = Y[2];
            ecp.y.data[3] = Y[3];

            if (h)
            {
                PntAcc.x = Kparams.EcJumps1[jmp_id].p.x;
                PntAcc.y = Kparams.EcJumps1[jmp_id].p.y;

                PrivAcc = Kparams.EcJumps1[jmp_id].dist;
                PrivAcc.data[0] += h;

                if (add_point_to_point(X, Y, PntAcc.x.data, PntAcc.y.data))
                {
                    X[0] = PntAcc.x.data[0];
                    X[1] = PntAcc.x.data[1];
                    X[2] = PntAcc.x.data[2];
                    X[3] = PntAcc.x.data[3];
                    Y[0] = PntAcc.y.data[0];
                    Y[1] = PntAcc.y.data[1];
                    Y[2] = PntAcc.y.data[2];
                    Y[3] = PntAcc.y.data[3];
                }
                
                add_256(Priv, Priv, PrivAcc.data);
                
                check_solve(X, Y, Priv, Kparams);
                
                if (*Kparams.SolvedKeyIndex != -1)
                    break;
            }
        }
        store_to_LDS_compressed(px, X, THREAD_X);
        store_to_LDS_compressed(py, Y, THREAD_X);
        store_to_LDS_compressed(pr, Priv, THREAD_X);
        __syncthreads();
    }
}

//this kernel checks for collisions with tame points
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelB(const TKparams Kparams)
{
    const u32 kang_id = blockIdx.x * blockDim.x + threadIdx.x;
    u32 pnt_offset = THREAD_X;

    for (int i = 0; i < Kparams.GroupCnt; i++)
    {
        //load point for a kang from LDS
        u64* px = &LDS[pnt_offset * 12];
        u64* py = &LDS[pnt_offset * 12 + 4];
        u64* pr = &LDS[pnt_offset * 12 + 8];
        
        u64 X[4], Y[4], Priv[4];

        load_from_LDS_compressed(X, px, THREAD_X);
        load_from_LDS_compressed(Y, py, THREAD_X);
        load_from_LDS_compressed(Priv, pr, THREAD_X);
        
        //check if collision with wild point
        for (int key_idx = 0; key_idx < Kparams.PntToSolve_len; key_idx++)
        {
            u64* PntToSolve_x = Kparams.PntsToSolve[key_idx].x.data;
            u64* PntToSolve_y = Kparams.PntsToSolve[key_idx].y.data;
            
            if (is_equal(X, PntToSolve_x) && is_equal(Y, PntToSolve_y))
            {
                atomicCAS(Kparams.SolvedKeyIndex, -1, key_idx);
                break;
            }
        }
        
        //check if we need to stop
        if (*Kparams.SolvedKeyIndex != -1)
            break;
    }
}

//this kernel finds point on the elliptic curve
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelC(const TKparams Kparams)
{
    const u32 kang_id = blockIdx.x * blockDim.x + threadIdx.x;
    u32 pnt_offset = THREAD_X;

    for (int i = 0; i < Kparams.GroupCnt; i++)
    {
        //load point for a kang from LDS
        u64* px = &LDS[pnt_offset * 12];
        u64* py = &LDS[pnt_offset * 12 + 4];
        u64* pr = &LDS[pnt_offset * 12 + 8];
        
        u64 X[4], Y[4], Priv[4];

        load_from_LDS_compressed(X, px, THREAD_X);
        load_from_LDS_compressed(Y, py, THREAD_X);
        load_from_LDS_compressed(Priv, pr, THREAD_X);

        if (check_DP(X, Y, Priv, Kparams))
        {
            //copy points to global buffer
            u32 cnt = atomicAdd(Kparams.DP_ptr, 1);
            
            u64* p_out = Kparams.DPs_out + cnt * (GPU_DP_SIZE / 4);
            
            p_out[0] = X[0];
            p_out[1] = X[1];
            p_out[2] = X[2];
            p_out[3] = X[3];
            p_out[4] = Y[0];
            p_out[5] = Y[1];
            p_out[6] = Y[2];
            p_out[7] = Y[3];
            
            //copy privkey
            p_out[8] = Priv[0];
            p_out[9] = Priv[1];
            p_out[10] = Priv[2];
            p_out[11] = Priv[3];
        }

        if (*Kparams.SolvedKeyIndex != -1)
            break;
    }
}

#endif //OLD_GPU

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CallGpuKernelABC(TKparams Kparams)
{
    KernelA <<< Kparams.BlockCnt, Kparams.BlockSize >>> (Kparams);
    KernelB <<< Kparams.BlockCnt, Kparams.BlockSize >>> (Kparams);
    KernelC <<< Kparams.BlockCnt, Kparams.BlockSize >>> (Kparams);
}

void CallGpuKernelGen(TKparams Kparams)
{
    KernelGen <<< Kparams.BlockCnt, Kparams.BlockSize, 0 >>> (Kparams);
}

cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table)

void CallGpuKernelGen(TKparams Kparams)
{
	KernelGen << < Kparams.BlockCnt, Kparams.BlockSize, 0 >> > (Kparams);
}

cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table)
{
	cudaError_t err = cudaFuncSetAttribute(KernelA, cudaFuncAttributeMaxDynamicSharedMemorySize, Kparams.KernelA_LDS_Size);
	if (err != cudaSuccess)
		return err;
	err = cudaFuncSetAttribute(KernelB, cudaFuncAttributeMaxDynamicSharedMemorySize, Kparams.KernelB_LDS_Size);
	if (err != cudaSuccess)
		return err;
	err = cudaFuncSetAttribute(KernelC, cudaFuncAttributeMaxDynamicSharedMemorySize, Kparams.KernelC_LDS_Size);
	if (err != cudaSuccess)
		return err;
	err = cudaMemcpyToSymbol(jmp2_table, _jmp2_table, JMP_CNT * 64);
	if (err != cudaSuccess)
		return err;
	return cudaSuccess;
}
