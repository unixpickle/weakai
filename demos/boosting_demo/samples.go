package main

import "math/rand"

const Dimensions = 5

func RandomClassVector(count int) []float64 {
	res := make([]float64, count)
	for i := range res {
		if rand.Intn(2) == 0 {
			res[i] = -1
		} else {
			res[i] = 1
		}
	}
	return res
}

type SampleList [][]float64

func RandomSampleList(count int) SampleList {
	res := make(SampleList, count)
	for i := range res {
		sample := make([]float64, Dimensions)
		for d := 0; d < Dimensions; d++ {
			sample[d] = rand.Float64()
		}
		res[i] = sample
	}
	return res
}

func (s SampleList) Len() int {
	return len(s)
}
