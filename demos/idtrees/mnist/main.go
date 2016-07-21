package main

import (
	"log"

	"github.com/unixpickle/mnist"
	"github.com/unixpickle/weakai/idtrees"
)

const (
	ForestSize   = 100
	TrainingSize = 4000
)

func main() {
	log.Println("Creating training samples...")
	samples := trainingSamples()
	attrs := trainingAttrs()
	log.Println("Training forest...")
	forest := idtrees.BuildForest(ForestSize, samples, attrs, TrainingSize, 75,
		func(s []idtrees.Sample, a []idtrees.Attr) *idtrees.Tree {
			return idtrees.ID3(s, a, 0)
		})
	log.Println("Running classifications...")
	hist := mnist.LoadTestingDataSet().CorrectnessHistogram(func(data []float64) int {
		sample := newImageSample(mnist.Sample{Intensities: data})
		res := forest.Classify(sample)
		var maxVal float64
		var maxClass int
		for class, x := range res {
			if x > maxVal {
				maxVal = x
				maxClass = class.(int)
			}
		}
		return maxClass
	})
	log.Println("Results:", hist)
}

func trainingSamples() []idtrees.Sample {
	set := mnist.LoadTrainingDataSet()
	res := make([]idtrees.Sample, len(set.Samples))
	for i, x := range set.Samples {
		res[i] = newImageSample(x)
	}
	return res
}

func trainingAttrs() []idtrees.Attr {
	attrs := make([]idtrees.Attr, 28*28)
	for i := 0; i < 28*28; i++ {
		attrs[i] = i
	}
	return attrs
}

type imageSample struct {
	Intensities []float64
	Label       int
}

func newImageSample(s mnist.Sample) *imageSample {
	return &imageSample{
		Intensities: s.Intensities,
		Label:       s.Label,
	}
}

func (i *imageSample) Attr(idx idtrees.Attr) idtrees.Val {
	return i.Intensities[idx.(int)]
}

func (i *imageSample) Class() idtrees.Class {
	return i.Label
}
