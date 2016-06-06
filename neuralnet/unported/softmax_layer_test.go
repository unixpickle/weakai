package neuralnet

import (
	"math"
	"math/rand"
	"testing"
)

func TestSoftmaxLayerOutput(t *testing.T) {
	layer := NewSoftmaxLayer(&SoftmaxParams{Size: 10})
	layer.SetInput([]float64{
		0.175974, 0.764459, 0.573260, 0.105675, 0.320708,
		0.554257, 0.028740, 0.826560, 0.290679, 0.208740,
	})
	layer.PropagateForward()
	expOutput := []float64{
		0.078301, 0.141040, 0.116494, 0.072985, 0.090495,
		0.114302, 0.067581, 0.150076, 0.087818, 0.080909,
	}
	for i, x := range expOutput {
		actual := layer.Output()[i]
		if math.Abs(actual-x) > 1e-5 {
			t.Errorf("invalid output %d: got %f expected %f", i, actual, x)
		}
	}
}

func TestSoftmaxLayerGradient(t *testing.T) {
	layer := NewSoftmaxLayer(&SoftmaxParams{Size: 10})
	layer.SetInput([]float64{
		0.175974, 0.764459, 0.573260, 0.105675, 0.320708,
		0.554257, 0.028740, 0.826560, 0.290679, 0.208740,
	})
	layer.PropagateForward()
	softmaxTestCostFunc(layer)
	layer.PropagateBackward(true)

	grad := make([]float64, len(layer.UpstreamGradient()))
	copy(grad, layer.UpstreamGradient())

	for i, old := range layer.Input() {
		layer.Input()[i] = old - 1e-5
		layer.PropagateForward()
		cost1 := softmaxTestCostFunc(layer)
		layer.Input()[i] = old + 1e-5
		layer.PropagateForward()
		cost2 := softmaxTestCostFunc(layer)
		partial := (cost2 - cost1) / 2e-5
		if math.Abs(partial-grad[i]) > 1e-6 {
			t.Errorf("invalid partial at index %d: got %f expected %f", i, grad[i], partial)
		}
	}
}

func BenchmarkSoftmaxForward(b *testing.B) {
	rand.Seed(123)
	inputVec := make([]float64, 3000)
	for i := range inputVec {
		inputVec[i] = rand.Float64()*5 - 2.5
	}
	layer := NewSoftmaxLayer(&SoftmaxParams{Size: len(inputVec)})
	layer.SetInput(inputVec)
	layer.SetDownstreamGradient(inputVec)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.PropagateForward()
	}
}

func BenchmarkSoftmaxBackProp(b *testing.B) {
	rand.Seed(123)
	inputVec := make([]float64, 3000)
	for i := range inputVec {
		inputVec[i] = rand.Float64()*5 - 2.5
	}
	layer := NewSoftmaxLayer(&SoftmaxParams{Size: len(inputVec)})
	layer.SetInput(inputVec)
	layer.SetDownstreamGradient(inputVec)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.PropagateForward()
		layer.PropagateBackward(true)
	}
}

func softmaxTestCostFunc(s *SoftmaxLayer) float64 {
	wanted := []float64{
		0.063058, 0.115592, 0.771756, 0.722379, 0.808390,
		0.055325, 0.643085, 0.439158, 0.952511, 0.208639,
	}
	var cost float64
	grad := make([]float64, len(wanted))
	for i, x := range wanted {
		grad[i] = s.Output()[i] - x
		cost += 0.5 * math.Pow(grad[i], 2)
	}
	s.SetDownstreamGradient(grad)
	return cost
}
