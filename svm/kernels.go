package svm

import "math"

// LinearKernel is a Kernel that returns the straight dot product of the two input samples.
func LinearKernel(s1, s2 Sample) float64 {
	if len(s1.V) != len(s2.V) {
		panic("samples must be of the sample dimension")
	}
	var sum float64
	for i, x := range s1.V {
		sum += x * s2.V[i]
	}
	return sum
}

// PolynomialKernel generates a Kernel that plugs vectors x and y into the formula (x*y + b)^n.
func PolynomialKernel(b, n float64) Kernel {
	return func(x, y Sample) float64 {
		return math.Pow(LinearKernel(x, y)+b, n)
	}
}

// CachedKernel generates a Kernel which caches results from a different kernel.
// This requires that each Sample has a unique UserInfo, excepting ones with UserInfo == 0.
// The caching Kernel will not use the cache for any samples that have UserInfo values of 0.
func CachedKernel(k Kernel) Kernel {
	cache := map[int]map[int]float64{}
	return func(s1, s2 Sample) float64 {
		if s1.UserInfo == 0 || s2.UserInfo == 0 {
			return k(s1, s2)
		}
		s1Cache := cache[s1.UserInfo]
		if s1Cache == nil {
			s1Cache = map[int]float64{}
			cache[s1.UserInfo] = s1Cache
		}
		if val, ok := s1Cache[s2.UserInfo]; ok {
			return val
		} else {
			res := k(s1, s2)
			s1Cache[s2.UserInfo] = res
			return res
		}
	}
}
