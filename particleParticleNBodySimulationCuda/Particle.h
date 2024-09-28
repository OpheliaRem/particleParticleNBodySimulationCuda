#pragma once
#include "Vector.h"

struct Particle
{
	Vector position;
	Vector velocity;
	Vector acceleration;
	double mass;
};