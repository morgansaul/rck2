// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC/Kang-2


#pragma once

#include "defs.h"
#include "utils.h"

class EcInt
{
public:
	EcInt();

	void Assign(const EcInt& val);
	void Set(u64 val);
	void SetZero();
	bool SetHexStr(const char* str);
	void GetHexStr(char* str);
	u16 GetU16(int index);

	bool Add(const EcInt& val); //returns true if carry
	bool Sub(const EcInt& val); //returns true if carry
	void Neg();
	void Neg256();
	void ShiftRight(int nbits);
	void ShiftLeft(int nbits);
	bool IsLessThanU(const EcInt& val);
	bool IsLessThanI(const EcInt& val);
	bool IsEqual(const EcInt& val);
	bool IsZero();

	void Mul_u64(const EcInt& val, u64 multiplier);
	void Mul_i64(const EcInt& val, i64 multiplier);

	void AddModP(const EcInt& val);
	void SubModP(const EcInt& val);
	void NegModP();
	void NegModN();
	void MulModP(const EcInt& val);
	void InvModP();
	void SqrtModP();

	void RndBits(int nbits);

	bool Div_i64(EcInt& val, u64 divisor);

	void DivModP(const EcInt& val);
	void GetInverseModN(const EcInt& val);
	
	//for debug
	void SetRandom();
	
	u64 data[4];
};

class EcPoint
{
public:
	EcPoint();

	void Assign(const EcPoint& pnt);
	void SetZero();
	bool IsZero();
	
	//for debug
	void SetRandom();

	void Add(const EcPoint& pnt);
	void Add(const EcPoint& pnt, const EcInt& k);
	void Double();
	void Negate();
	void Normalize();
	void FromJacobian(const EcInt& x, const EcInt& y, const EcInt& z);

	bool SetHexStr(const char* str);
	// NEW: Added GetHexStr method for EcPoint
	void GetHexStr(char* str);

	//for debug
	void PrintHex();

	EcInt x;
	EcInt y;
	EcInt z;
	EcInt priv;
};

class Ec
{
public:
	Ec();
	~Ec();
	static void Init();
	static void Cleanup();

	EcPoint MultiplyG(const EcInt& priv_key);
	EcPoint Multiply(const EcPoint& pnt, const EcInt& priv_key);

	void GenerateJumps(EcJMP* Jumps, int count, int range, int type, u64* dbg, u64 start);
	void GenerateRandomPoints(EcPoint* points, int count, EcInt offset);

	bool Verify(const EcPoint& pnt, const EcInt& priv_key);
};

struct EcJMP
{
	EcPoint p;
	EcInt dist;
};

extern EcInt g_P, g_N;
extern EcPoint g_G;
