package neuralnet

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
)

func TestSoftmaxLayerOutput(t *testing.T) {
	input := &autofunc.Variable{
		Vector: []float64{
			0.175974, 0.764459, 0.573260, 0.105675, 0.320708,
			0.554257, 0.028740, 0.826560, 0.290679, 0.208740,
		},
	}
	layer := SoftmaxLayer{}
	output := layer.Apply(input).Output()
	expOutput := []float64{
		0.078301, 0.141040, 0.116494, 0.072985, 0.090495,
		0.114302, 0.067581, 0.150076, 0.087818, 0.080909,
	}
	for i, x := range expOutput {
		actual := output[i]
		if math.Abs(actual-x) > 1e-5 {
			t.Errorf("invalid output %d: got %f expected %f", i, actual, x)
		}
	}
}

func TestLogSoftmaxLayerOutput(t *testing.T) {
	input := &autofunc.Variable{
		Vector: []float64{
			0.175974, 0.764459, 0.573260, 0.105675, 0.320708,
			0.554257, 0.028740, 0.826560, 0.290679, 0.208740,
		},
	}
	layer := LogSoftmaxLayer{}
	output := layer.Apply(input).Output()
	expOutput := []float64{
		-2.547194905, -1.958711741, -2.149915509, -2.617501338, -2.402460678,
		-2.168911211, -2.694428401, -1.896613447, -2.432488788, -2.514430213,
	}
	for i, x := range expOutput {
		actual := output[i]
		if math.Abs(actual-x) > 1e-5 {
			t.Errorf("invalid output %d: got %f expected %f", i, actual, x)
		}
	}
}

func BenchmarkSoftmaxForward(b *testing.B) {
	rand.Seed(123)
	inputVec := make([]float64, 3000)
	for i := range inputVec {
		inputVec[i] = rand.Float64()*5 - 2.5
	}
	inputVar := &autofunc.Variable{Vector: inputVec}
	layer := SoftmaxLayer{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Apply(inputVar)
	}
}

func BenchmarkSoftmaxBackProp(b *testing.B) {
	rand.Seed(123)
	inputVec := make([]float64, 3000)
	for i := range inputVec {
		inputVec[i] = rand.Float64()*5 - 2.5
	}
	inputVar := &autofunc.Variable{Vector: inputVec}
	outGrad := autofunc.NewGradient([]*autofunc.Variable{inputVar})
	layer := SoftmaxLayer{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Apply(inputVar).PropagateGradient(inputVec, outGrad)
	}
}
