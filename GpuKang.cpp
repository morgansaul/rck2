// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"

#include "GpuKang.h"
#include "defs.h"
#include <vector>

cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table);
void CallGpuKernelGen(TKparams Kparams);
void CallGpuKernelABC(TKparams Kparams);
void AddPointsToList(u32* data, int cnt, u64 ops_cnt);
extern bool gGenMode; //tames generation mode
extern volatile int gSolvedKeyIndex;


int RCGpuKang::CalcKangCnt()
{
    Kparams.BlockCnt = mpCnt;
    Kparams.BlockSize = IsOldGpu ? 512 : 256;
    Kparams.GroupCnt = IsOldGpu ? 64 : 24;
    return Kparams.BlockSize* Kparams.GroupCnt* Kparams.BlockCnt;
}

//executes in main thread
bool RCGpuKang::Prepare(std::vector<EcPoint>& _PntsToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3)
{
    PntsToSolve = _PntsToSolve;
    Range = _Range;
    DP = _DP;
    EcJumps1 = _EcJumps1;
    EcJumps2 = _EcJumps2;
    EcJumps3 = _EcJumps3;
    
    Kparams.PntToSolve_len = PntsToSolve.size();
    
    cudaSetDevice(CudaIndex);
    
    cudaMallocManaged((void**)&Kparams.PntsToSolve, PntsToSolve.size() * sizeof(EcPoint));
    cudaMemcpy(Kparams.PntsToSolve, PntsToSolve.data(), PntsToSolve.size() * sizeof(EcPoint), cudaMemcpyHostToDevice);
    
    cudaMallocManaged((void**)&DPs_out, MAX_DP_CNT * GPU_DP_SIZE);
    
    Kparams.DPs_out = DPs_out;
    
    Kparams.Range = Range;
    Kparams.DP = DP;
    Kparams.TameOffset = Int_TameOffset;

    Kparams.EcJumps1 = EcJumps1;
    Kparams.EcJumps2 = EcJumps2;
    Kparams.EcJumps3 = EcJumps3;

    Kparams.HalfRange = Int_HalfRange;
    Kparams.PntHalfRange = Pnt_HalfRange;
    Kparams.NegPntHalfRange = Pnt_NegHalfRange;
    
    Kparams.BlockCnt = mpCnt;
    Kparams.BlockSize = IsOldGpu ? 512 : 256;
    Kparams.GroupCnt = IsOldGpu ? 64 : 24;

    GenerateRndPnts();
    SetKangParams();

    cudaMemset(Kparams.dbg_buf, 0, 1024);
    
    StopFlag = false;

    return true;
}

RCGpuKang::RCGpuKang()
{
    DPs_out = NULL;
    RndPnts = NULL;
}

RCGpuKang::~RCGpuKang()
{
    cudaSetDevice(CudaIndex);

    if (DPs_out)
        cudaFree(DPs_out);

    if (RndPnts)
        cudaFree(RndPnts);
    
    if (Kparams.PntsToSolve)
        cudaFree(Kparams.PntsToSolve);
}

void RCGpuKang::GenerateRndPnts()
{
    int size = CalcKangCnt() * 96;
    cudaMallocManaged((void**)&RndPnts, size);
    Kparams.RndPnts = RndPnts;

    PntA = g_G;
    PntA.Multiply(Int_HalfRange);
    PntB.Assign(PntA);
    PntB.Neg();

    for (int i = 0; i < CalcKangCnt(); i++)
    {
        EcInt rnd, rnd2;
        rnd.RndBits(Range / 2 + 16);
        rnd2.RndBits(Range - 16);

        EcInt priv;
        priv.Add(rnd);

        if (gGenMode)
        {
            priv = Int_TameOffset;
        }

        EcPoint p = ec.MultiplyG(priv);

        RndPnts[i].priv[0] = priv.data[0];
        RndPnts[i].priv[1] = priv.data[1];
        RndPnts[i].priv[2] = priv.data[2];
        RndPnts[i].priv[3] = priv.data[3];
        RndPnts[i].x[0] = p.x.data[0];
        RndPnts[i].x[1] = p.x.data[1];
        RndPnts[i].x[2] = p.x.data[2];
        RndPnts[i].x[3] = p.x.data[3];
        RndPnts[i].y[0] = p.y.data[0];
        RndPnts[i].y[1] = p.y.data[1];
        RndPnts[i].y[2] = p.y.data[2];
        RndPnts[i].y[3] = p.y.data[3];
    }
}

void RCGpuKang::SetKangParams()
{
    cudaSetDevice(CudaIndex);
    cudaError_t err;

    err = cudaMallocManaged((void**)&Kparams.jmp1_table, JMP_CNT * 8 * sizeof(u64));
    memcpy(Kparams.jmp1_table, EcJumps1, JMP_CNT * sizeof(EcJMP));
    err = cudaMallocManaged((void**)&Kparams.jmp2_table, JMP_CNT * 8 * sizeof(u64));
    memcpy(Kparams.jmp2_table, EcJumps2, JMP_CNT * sizeof(EcJMP));
    err = cudaMallocManaged((void**)&Kparams.jmp3_table, JMP_CNT * 8 * sizeof(u64));
    memcpy(Kparams.jmp3_table, EcJumps3, JMP_CNT * sizeof(EcJMP));
    err = cuSetGpuParams(Kparams, (u64*)Kparams.jmp2_table);
    if (err != cudaSuccess)
        printf("cuSetGpuParams failed!\r\n");
}


void RCGpuKang::Execute()
{
    cudaSetDevice(CudaIndex);
    cudaError_t err;
    u64 t1;
    
    cudaMallocManaged((void**)&Kparams.DP_ptr, 1024);
    Kparams.DP_ptr[0] = 0;

    int cur_speed_ind = 0;
    
    while (!StopFlag)
    {
        t1 = GetTickCount64();
        
        if (gGenMode)
            CallGpuKernelGen(Kparams);
        else
            CallGpuKernelABC(Kparams);
        
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            printf("Kernel run failed: %s\r\n", cudaGetErrorString(err));
            break;
        }

        u32 cnt;
        err = cudaMemcpy(&cnt, Kparams.DP_ptr, 4, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            gTotalErrors++;
            break;
        }

        if (cnt)
        {
            //copy DPs from GPU to CPU
            AddPointsToList(Kparams.DPs_out, cnt, (u64)CalcKangCnt() * STEP_CNT);
            Kparams.DP_ptr[0] = 0;
        }
        
        // Check if any key has been solved and break all kangaroos
        int solved_key_idx;
        cudaMemcpy(&solved_key_idx, Kparams.SolvedKeyIndex, 4, cudaMemcpyDeviceToHost);
        if (solved_key_idx != -1) {
            StopFlag = true;
            gSolvedKeyIndex = solved_key_idx;
            break;
        }

        u64 t2 = GetTickCount64();
        u64 tm = t2 - t1;
        if (!tm)
            tm = 1;
        
        int cur_speed = (int)((u64)CalcKangCnt() * STEP_CNT / (tm * 1000));
        
        SpeedStats[cur_stats_ind] = cur_speed;
        cur_stats_ind++;
        if (cur_stats_ind >= STATS_WND_SIZE)
            cur_stats_ind = 0;

        //if we are here, we must get out, we found a key or ops limit reached
        if (gSolved || gIsOpsLimit)
        {
            StopFlag = true;
            break;
        }
    }
}

int RCGpuKang::GetStatsSpeed()
{
    int res = 0;
    for (int i = 0; i < STATS_WND_SIZE; i++)
        res += SpeedStats[i];
    return res / STATS_WND_SIZE;
}
