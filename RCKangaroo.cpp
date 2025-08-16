// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h> 
#include <inttypes.h>
#include <stdint.h>
#include <vector>

#include "cuda_runtime.h"
#include "cuda.h"

#include "defs.h"
#include "utils.h"
#include "GpuKang.h"

#ifndef _WIN32
#include <unistd.h>
#endif

time_t program_start_time = time(NULL);  // Capture the start time


EcJMP EcJumps1[JMP_CNT];
EcJMP EcJumps2[JMP_CNT];
EcJMP EcJumps3[JMP_CNT];

RCGpuKang* GpuKangs[MAX_GPU_CNT];
int GpuCnt;
volatile long ThrCnt;
volatile bool gSolved;

EcInt Int_HalfRange;
EcPoint Pnt_HalfRange;
EcPoint Pnt_NegHalfRange;
EcInt Int_TameOffset;
Ec ec;

CriticalSection csAddPoints;
u8* pPntList;
u8* pPntList2;
volatile int PntIndex;
TFastBase db;

// New global variables for multiple public keys
std::vector<EcPoint> gPubKeysToSolve;
volatile int gSolvedKeyIndex = -1;

EcInt gPrivKey;

volatile u64 TotalOps;
u32 TotalSolved;
u32 gTotalErrors;
u64 PntTotalOps;
bool IsBench;

u32 gDP;
u32 gRange;
EcInt gStart;
bool gStartSet;
EcPoint gPubKey;
u8 gGPUs_Mask[MAX_GPU_CNT];
char gTamesFileName[1024];
char gPubKeysFileName[1024] = { 0 };
double gMax;
bool gGenMode; //tames generation mode
bool gIsOpsLimit;

#pragma pack(push, 1)
struct DBRec
{
	u8 x[12];
	u8 d[22];
	u8 type; //0 - tame, 1 - wild1, 2 - wild2
};
#pragma pack(pop)

void InitGpus()
{
	GpuCnt = 0;
	int gcnt = 0;
	cudaGetDeviceCount(&gcnt);
	if (gcnt > MAX_GPU_CNT)
		gcnt = MAX_GPU_CNT;

	//	gcnt = 1; //dbg
	if (!gcnt)
		return;

	int drv, rt;
	cudaRuntimeGetVersion(&rt);
	cudaDriverGetVersion(&drv);
	char drvver[100];
	sprintf(drvver, "%d.%d/%d.%d", drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);

	printf("CUDA devices: %d, CUDA driver/runtime: %s\r\n", gcnt, drvver);
	cudaError_t cudaStatus;
	for (int i = 0; i < gcnt; i++)
	{
		cudaStatus = cudaSetDevice(i);
		if (cudaStatus != cudaSuccess)
		{
			printf("cudaSetDevice for gpu %d failed!\r\n", i);
			continue;
		}

		if (!gGPUs_Mask[i])
			continue;

		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);
		printf("GPU %d: %s, %.2f GB, %d CUs, cap %d.%d, L2 size: %d KB\r\n", i, deviceProp.name, ((float)(deviceProp.totalGlobalMem / (1024 * 1024))) / 1024.0f, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor, deviceProp.l2CacheSize / 1024);

		if (deviceProp.major < 6)
		{
			printf("GPU %d - not supported, skip\r\n", i);
			continue;
		}

		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

		GpuKangs[GpuCnt] = new RCGpuKang();
		GpuKangs[GpuCnt]->CudaIndex = i;
		GpuKangs[GpuCnt]->persistingL2CacheMaxSize = deviceProp.persistingL2CacheMaxSize;
		GpuKangs[GpuCnt]->mpCnt = deviceProp.multiProcessorCount;
		GpuKangs[GpuCnt]->IsOldGpu = deviceProp.l2CacheSize < 16 * 1024 * 1024;
		GpuCnt++;
	}
	printf("GPUs Found: %d\r\n", GpuCnt);
}
#ifdef _WIN32
u32 __stdcall kang_thr_proc(void* data)
{
	RCGpuKang* Kang = (RCGpuKang*)data;
	Kang->Execute();
	InterlockedDecrement(&ThrCnt);
	return 0;
}
#else
void* kang_thr_proc(void* data)
{
	RCGpuKang* Kang = (RCGpuKang*)data;
	Kang->Execute();
	__sync_fetch_and_sub(&ThrCnt, 1);
	return 0;
}
#endif
void AddPointsToList(u32* data, int pnt_cnt, u64 ops_cnt)
{
	csAddPoints.Enter();
	if (PntIndex + pnt_cnt >= MAX_CNT_LIST)
	{
		csAddPoints.Leave();
		printf("DPs buffer overflow, some points lost, increase DP value!\r\n");
		return;
	}
	memcpy(pPntList + GPU_DP_SIZE * PntIndex, data, pnt_cnt * GPU_DP_SIZE);
	PntIndex += pnt_cnt;
	PntTotalOps += ops_cnt;
	csAddPoints.Leave();
}

bool Collision_SOTA(EcPoint& pnt, EcInt t, int TameType, EcInt w, int WildType, bool IsNeg)
{
	if (IsNeg)
		t.Neg();
	if (TameType == TAME)
	{
		gPrivKey = t;
		gPrivKey.Sub(w);
		EcInt sv = gPrivKey;
		gPrivKey.Add(Int_HalfRange);
		EcPoint P = ec.MultiplyG(gPrivKey);
		if (P.IsEqual(pnt))
			return true;
		gPrivKey = sv;
		gPrivKey.Neg();
		gPrivKey.Add(Int_HalfRange);
		P = ec.MultiplyG(gPrivKey);
		return P.IsEqual(pnt);
	}
	else
	{
		gPrivKey = t;
		gPrivKey.Sub(w);
		if (gPrivKey.data[4] >> 63)
			gPrivKey.Neg();
		gPrivKey.ShiftRight(1);
		EcInt sv = gPrivKey;
		gPrivKey.Add(Int_HalfRange);
		EcPoint P = ec.MultiplyG(gPrivKey);
		if (P.IsEqual(pnt))
			return true;
		gPrivKey = sv;
		gPrivKey.Neg();
		gPrivKey.Add(Int_HalfRange);
		P = ec.MultiplyG(gPrivKey);
		return P.IsEqual(pnt);
	}
}


void trim_leading_zeros(char* str) {
	char* non_zero = str;
	while (*non_zero == '0' && *(non_zero + 1) != '\0') {
		non_zero++;
	}
	if (non_zero != str) {
		memmove(str, non_zero, strlen(non_zero) + 1);
	}
}

void CheckNewPoints()
{
	csAddPoints.Enter();
	if (!PntIndex)
	{
		csAddPoints.Leave();
		return;
	}

	int cnt = PntIndex;
	memcpy(pPntList2, pPntList, GPU_DP_SIZE * cnt);
	PntIndex = 0;
	csAddPoints.Leave();

	for (int i = 0; i < cnt; i++)
	{
		DBRec nrec;
		u8* p = pPntList2 + i * GPU_DP_SIZE;
		memcpy(nrec.x, p, 12);
		memcpy(nrec.d, p + 16, 22);
		nrec.type = gGenMode ? TAME : p[40];

		DBRec* pref = (DBRec*)db.FindOrAddDataBlock((u8*)&nrec);
		if (gGenMode)
			continue;
		if (pref)
		{
			//in db we dont store first 3 bytes so restore them
			DBRec tmp_pref;
			memcpy(&tmp_pref, &nrec, 3);
			memcpy(((u8*)&tmp_pref) + 3, pref, sizeof(DBRec) - 3);
			pref = &tmp_pref;

			if (pref->type == nrec.type)
			{
				if (pref->type == TAME)
					continue;

				//if it's wild, we can find the key from the same type if distances are different
				if (*(u64*)pref->d == *(u64*)nrec.d)
					continue;
				//else
				//	ToLog("key found by same wild");
			}

			EcInt w, t;
			int TameType, WildType;
			if (pref->type != TAME)
			{
				memcpy(w.data, pref->d, sizeof(pref->d));
				if (pref->d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
				memcpy(t.data, nrec.d, sizeof(nrec.d));
				if (nrec.d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
				TameType = nrec.type;
				WildType = pref->type;
			}
			else
			{
				memcpy(w.data, nrec.d, sizeof(nrec.d));
				if (nrec.d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
				memcpy(t.data, pref->d, sizeof(pref->d));
				if (pref->d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
				TameType = TAME;
				WildType = nrec.type;
			}
            
			// Iterate through all public keys and check for a collision
			for (int key_idx = 0; key_idx < gPubKeysToSolve.size(); key_idx++)
			{
				bool res = Collision_SOTA(gPubKeysToSolve[key_idx], t, TameType, w, WildType, false) || Collision_SOTA(gPubKeysToSolve[key_idx], t, TameType, w, WildType, true);
				if (res)
				{
					gSolved = true;
					gSolvedKeyIndex = key_idx;
					break;
				}
			}

			if (gSolved)
				break;
		}
	}
}

void ShowStats(u64 tm_start, double exp_ops, double dp_val)
{
#ifdef DEBUG_MODE
	for (int i = 0; i <= MD_LEN; i++)
	{
		u64 val = 0;
		for (int j = 0; j < GpuCnt; j++)
		{
			val += GpuKangs[j]->dbg[i];
		}
		if (val)
			printf("Loop size %d: %llu\r\n", i, val);
	}
#endif

	int speed = GpuKangs[0]->GetStatsSpeed();
	for (int i = 1; i < GpuCnt; i++)
		speed += GpuKangs[i]->GetStatsSpeed();

	u64 est_dps_cnt = (u64)(exp_ops / dp_val);
	u64 exp_sec = 0xFFFFFFFFFFFFFFFFull;

	if (speed)
		exp_sec = (u64)((exp_ops / 1000000) / speed);  // Expected time in seconds

	// Expected Time Breakdown
	u64 exp_days = exp_sec / (3600 * 24);
	int exp_hours = (int)(exp_sec % (3600 * 24)) / 3600;
	int exp_min = (int)(exp_sec % 3600) / 60;
	int exp_remaining_sec = (int)(exp_sec % 60);  // Expected seconds

	// Elapsed Time Calculation
	u64 sec = (GetTickCount64() - tm_start) / 1000;  // Elapsed time in seconds
	u64 days = sec / (3600 * 24);
	int hours = (int)(sec % (3600 * 24)) / 3600;
	int min = (int)(sec % 3600) / 60;
	int remaining_sec = (int)(sec % 60);  // Elapsed seconds

	// Updated printf to include seconds in both elapsed and expected times
	printf("%sSpeed: %d MKeys/s, Err: %d, DPs: %lluK/%lluK, Time: %llud:%02dh:%02dm:%02ds/%llud:%02dh:%02dm:%02ds\r",
		gGenMode ? "GEN: " : (IsBench ? "BENCH: " : "MAIN: "),
		speed, gTotalErrors,
		db.GetBlockCnt() / 1000, est_dps_cnt / 1000,
		days, hours, min, remaining_sec,        // Elapsed Time with seconds
		exp_days, exp_hours, exp_min, exp_remaining_sec  // Expected Time with seconds
	);


	fflush(stdout);  // Force the console to update the line
}

bool SolvePoint(std::vector<EcPoint>& PntsToSolve, int Range, int DP, EcInt* pk_res)
{
	if ((Range < 32) || (Range > 180))
	{
		printf("Unsupported Range value (%d)!\r\n", Range);
		return false;
	}
	if ((DP < 14) || (DP > 60))
	{
		printf("Unsupported DP value (%d)!\r\n", DP);
		return false;
	}
	printf("\r\nSolving %d points: Range %d bits, DP %d, start...\r\n", PntsToSolve.size(), Range, DP);
	double ops = 1.15 * pow(2.0, Range / 2.0);
	double dp_val = (double)(1ull << DP);
	double ram = (32 + 4 + 4) * ops / dp_val; //+4 for grow allocation and memory fragmentation
	ram += sizeof(TListRec) * 256 * 256 * 256; //3byte-prefix table
	ram /= (1024 * 1024 * 1024); //GB
	printf("SOTA method, estimated ops: 2^%.3f, RAM for DPs: %.3f GB. DP and GPU overheads not included!\r\n", log2(ops), ram);
	gIsOpsLimit = false;
	double MaxTotalOps = 0.0;
	if (gMax > 0)
	{
		MaxTotalOps = gMax * ops;
		double ram_max = (32 + 4 + 4) * MaxTotalOps / dp_val; //+4 for grow allocation and memory fragmentation
		ram_max += sizeof(TListRec) * 256 * 256 * 256; //3byte-prefix table
		ram_max /= (1024 * 1024 * 1024); //GB
		printf("Max allowed number of ops: 2^%.3f, max RAM for DPs: %.3f GB\r\n", log2(MaxTotalOps), ram_max);
	}
	u64 total_kangs = GpuKangs[0]->CalcKangCnt();
	for (int i = 1; i < GpuCnt; i++)
		total_kangs += GpuKangs[i]->CalcKangCnt();
	double path_single_kang = ops / total_kangs;
	double DPs_per_kang = path_single_kang / dp_val;
	printf("Estimated DPs per kangaroo: %.3f.%s\r\n", DPs_per_kang, (DPs_per_kang < 5) ? " DP overhead is big, use less DP value if possible!" : "");
	if (!gGenMode && gTamesFileName[0])
	{
		printf("load tames...\r\n");
		if (db.LoadFromFile(gTamesFileName))
		{
			printf("tames loaded\r\n");
			if (db.Header[0] != gRange)
			{
				printf("loaded tames have different range, they cannot be used, clear\r\n");
				db.Clear();
			}
		}
		else printf("tames loading failed\r\n");
	}
	SetRndSeed(0); //use same seed to make tames from file compatible
	PntTotalOps = 0;
	PntIndex = 0;
	//prepare jumps
	EcInt minjump, t;
	minjump.Set(1);
	minjump.ShiftLeft(Range / 2 + 3);
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps1[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps1[i].dist.Add(t);
		EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		EcJumps1[i].p = ec.MultiplyG(EcJumps1[i].dist);
	}
	minjump.Set(1);
	minjump.ShiftLeft(Range - 10); //large jumps for L1S2 loops. Must be almost RANGE_BITS
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps2[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps2[i].dist.Add(t);
		EcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		EcJumps2[i].p = ec.MultiplyG(EcJumps2[i].dist);
	}
	minjump.Set(1);
	minjump.ShiftLeft(Range - 10);
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps3[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps3[i].dist.Add(t);
		EcJumps3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		EcJumps3[i].p = ec.MultiplyG(EcJumps3[i].dist);
	}

	for (int i = 0; i < GpuCnt; i++)
		GpuKangs[i]->Prepare(PntsToSolve, Range, DP, EcJumps1, EcJumps2, EcJumps3);

	if (!gGenMode)
	{
		for (int i = 0; i < gPubKeysToSolve.size(); i++)
		{
			printf("Searching for pubkey %d/%d: ", i + 1, gPubKeysToSolve.size());
			gPubKeysToSolve[i].PrintXY();
		}
	}
	
#ifdef _WIN32
	HANDLE* hThr = (HANDLE*)malloc(GpuCnt * sizeof(HANDLE));
#else
	pthread_t* hThr = (pthread_t*)malloc(GpuCnt * sizeof(pthread_t));
#endif
	ThrCnt = GpuCnt;
	gSolved = false;
	printf("\r\n");
	u64 tm_start = GetTickCount64();
	
	for (int i = 0; i < GpuCnt; i++)
	{
#ifdef _WIN32
		hThr[i] = (HANDLE)_beginthreadex(NULL, 0, kang_thr_proc, GpuKangs[i], 0, NULL);
#else
		pthread_create(&hThr[i], NULL, kang_thr_proc, GpuKangs[i]);
#endif
	}

	while (ThrCnt)
	{
		ShowStats(tm_start, ops, dp_val);
		Sleep(500);

		CheckNewPoints();
		if (gSolved)
		{
			for (int i = 0; i < GpuCnt; i++)
				GpuKangs[i]->Stop();

			while (ThrCnt)
				Sleep(500);

			break;
		}
		if (gIsOpsLimit && (PntTotalOps > MaxTotalOps))
		{
			printf("\r\nOps limit reached, exit...\r\n");
			for (int i = 0; i < GpuCnt; i++)
				GpuKangs[i]->Stop();

			while (ThrCnt)
				Sleep(500);
			break;
		}
	}

	for (int i = 0; i < GpuCnt; i++)
#ifdef _WIN32
		CloseHandle(hThr[i]);
#else
		pthread_join(hThr[i], NULL);
#endif

	free(hThr);

	if (gSolved)
	{
		*pk_res = gPrivKey;
		return true;
	}

	return false;
}

void ShowUsage()
{
	printf("RCKangaroo.exe -dp <val> -range <val> -pubkey <val> | -pubkeysfile <filename> | -gentames <filename> -max <val>\r\n");
	printf("\r\nExamples:\r\n");
	printf("  RCKangaroo.exe -dp 16 -range 84 -pubkey 0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a\r\n");
	printf("  RCKangaroo.exe -dp 16 -range 84 -pubkeysfile C:\\keys.txt\r\n");
	printf("  RCKangaroo.exe -dp 16 -range 76 -gentames tames76.dat -max 10\r\n");
	printf("\r\n-dp\t\tDP value in bits. Recommended range is 14-22. Default 16\r\n");
	printf("-range\t\tKey bits to search. Recommended range is 32-180. Default 78\r\n");
	printf("-gpu\t\tMask of GPUs to use. Default 0xffffffff\r\n");
	printf("-tames\t\tFile with precomputed tame points\r\n");
	printf("-gentames\tGenerate tame points file and exit\r\n");
	printf("-max\t\tMax ops to run for tame generation in units of Range^2/2. Max 1000\r\n");
}

void ParseParams(int argc, char** argv)
{
	gDP = 16;
	gRange = 78;
	gMax = 0;
	memset(gGPUs_Mask, 0xFF, sizeof(gGPUs_Mask));
	gGenMode = false;
	gTamesFileName[0] = 0;
	gPubKeysFileName[0] = 0;
	gStartSet = false;

	for (int i = 1; i < argc; i++)
	{
		if (!_stricmp(argv[i], "-dp") && (i + 1 < argc))
		{
			gDP = (u32)atoi(argv[i + 1]);
		}
		else if (!_stricmp(argv[i], "-range") && (i + 1 < argc))
		{
			gRange = (u32)atoi(argv[i + 1]);
		}
		else if (!_stricmp(argv[i], "-pubkey") && (i + 1 < argc))
		{
			if (!gPubKey.SetHexStr(argv[i + 1]))
			{
				printf("Failed to parse pubkey hex string!\r\n");
				exit(-1);
			}
			gPubKeysToSolve.push_back(gPubKey);
		}
		else if (!_stricmp(argv[i], "-pubkeysfile") && (i + 1 < argc))
		{
			strcpy(gPubKeysFileName, argv[i + 1]);
		}
		else if (!_stricmp(argv[i], "-start") && (i + 1 < argc))
		{
			if (!gStart.SetHexStr(argv[i + 1]))
			{
				printf("Failed to parse start hex string!\r\n");
				exit(-1);
			}
			gStartSet = true;
		}
		else if (!_stricmp(argv[i], "-gpu") && (i + 1 < argc))
		{
			unsigned int mask = 0;
			if (sscanf(argv[i + 1], "%x", &mask) == 1)
			{
				memset(gGPUs_Mask, 0, sizeof(gGPUs_Mask));
				for (int j = 0; j < MAX_GPU_CNT; j++)
				{
					if (mask & (1ull << j))
						gGPUs_Mask[j] = 0xFF;
				}
			}
		}
		else if (!_stricmp(argv[i], "-tames") && (i + 1 < argc))
		{
			strcpy(gTamesFileName, argv[i + 1]);
		}
		else if (!_stricmp(argv[i], "-gentames") && (i + 1 < argc))
		{
			gGenMode = true;
			strcpy(gTamesFileName, argv[i + 1]);
		}
		else if (!_stricmp(argv[i], "-max") && (i + 1 < argc))
		{
			gMax = (double)atof(argv[i + 1]);
			if (gMax < 0.1 || gMax > 1000.0)
			{
				printf("Max value can be from 0.1 to 1000.0\r\n");
				exit(-1);
			}
		}
	}
}

int main(int argc, char** argv)
{
	printf("RCKangaroo v0.2, (c) 2024, RetiredCoder (RC)\r\n");
	printf("SOTA Kangaroo for ECDLP on secp256k1\r\n");

	ParseParams(argc, argv);
	
	if (gGenMode)
	{
		printf("TAMES GENERATION MODE\r\n");
	}
	else
	{
		if (gPubKeysFileName[0] != 0)
		{
			if (!LoadPublicKeysFromFile(gPubKeysFileName, gPubKeysToSolve))
			{
				printf("Failed to load public keys from file: %s\r\n", gPubKeysFileName);
				return -1;
			}
			printf("Loaded %d public keys from file.\r\n", gPubKeysToSolve.size());
		}
		else if (gPubKeysToSolve.empty())
		{
			ShowUsage();
			return 0;
		}
	}


	if (!gPubKeysToSolve.empty() && gPubKeysToSolve.size() > 5000)
	{
		printf("Error: Max 5000 public keys are supported at once.\r\n");
		return -1;
	}

	ec.Init();

	if (gGenMode)
	{
		if (gMax == 0) gMax = 1.0;
		EcPoint PntToSolve;
		EcInt pk;
		pk.Set(1);
		PntToSolve = ec.MultiplyG(pk);
		SolvePoint({ PntToSolve }, gRange, gDP, NULL);
		if (db.SaveToFile(gTamesFileName))
			printf("tames saved to %s\r\n", gTamesFileName);
		else
			printf("tames saving failed!\r\n");
		return 0;
	}

	if (!gTamesFileName[0] && gPubKeysToSolve.size() > 1)
	{
		printf("Using a public key file requires pre-generated tames. Please use the -tames option.\r\n");
		return -1;
	}

	if (gStartSet)
	{
		EcPoint PntToSolve = ec.MultiplyG(gStart);
		if (!SolvePoint({ PntToSolve }, gRange, gDP, &gPrivKey))
		{
			printf("\r\nKey not found...\r\n");
		}
	}
	else if (!gPubKeysToSolve.empty())
	{
		if (!SolvePoint(gPubKeysToSolve, gRange, gDP, &gPrivKey))
		{
			printf("\r\nKey not found...\r\n");
		}
	}
	else
	{
		if (!gRange) gRange = 78;
		if (!gDP) gDP = 16;
		
		while (1)
		{
			//generate random pk
			EcInt pk;
			pk.RndBits(gRange);
			EcPoint PntToSolve = ec.MultiplyG(pk);

			if (!SolvePoint({ PntToSolve }, gRange, gDP, &gPrivKey))
			{
				printf("\r\nKey not found...\r\n");
			}
			else
			{
				printf("\r\nKey found: ");
				pk.PrintHex();
			}
			printf("\r\n");
			Sleep(100);
		}
	}

	if (gSolvedKeyIndex != -1)
	{
		EcPoint pubkey_found = gPubKeysToSolve[gSolvedKeyIndex];
		char s[100];
		pubkey_found.GetHexStr(s);
		trim_leading_zeros(s);
		printf("\r\nFOUND PRIVATE KEY for public key: %s\r\n", s);
		
		char pk_s[100];
		gPrivKey.GetHexStr(pk_s);
		trim_leading_zeros(pk_s);
		printf("PRIVATE KEY: %s\r\n\r\n", pk_s);

		FILE* fp = fopen("RESULTS.TXT", "a");
		if (fp)
		{
			fprintf(fp, "PRIVATE KEY for %s: %s\n", s, pk_s);
			fclose(fp);
		}
		else
		{
			printf("WARNING: Cannot save the key to RESULTS.TXT!\r\n");
			while (1)
				Sleep(100);
		}
	}
	
	for (int i = 0; i < GpuCnt; i++)
		delete GpuKangs[i];

	return 0;
}
