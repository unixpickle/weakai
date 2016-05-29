package main

import (
	"fmt"

	"github.com/unixpickle/num-analysis/autodiff"
)

const varCount = 6

func main() {
	// Gradient layout: in weight, in gate, remember gate weight, output gate weight, out weight, last state.
	inWeight := autodiff.NewNumVar(0.062555, varCount, 0)
	inGate := autodiff.NewNumVar(0, varCount, 1)
	rememberGate := autodiff.NewNumVar(0, varCount, 2)
	outGate := autodiff.NewNumVar(0, varCount, 3)
	outWeight := autodiff.NewNumVar(0.075439, varCount, 4)
	lastState := autodiff.NewNumVar(0.25, varCount, 5)
	half := autodiff.NewNum(0.5, 6)

	memIn := sigmoid(inWeight.Mul(lastState))
	maskedIn := memIn.Mul(sigmoid(inGate.Mul(lastState)))
	maskedMem := sigmoid(rememberGate.Mul(lastState)).Mul(lastState)
	outState := maskedMem.Add(maskedIn)
	outMask := sigmoid(outGate.Mul(lastState))
	realOutput := sigmoid(outMask.Mul(outState).Mul(outWeight))

	outError := half.Mul(realOutput.PowScaler(2))
	fmt.Println("last state partial:", outError.Gradient[5])
}

func sigmoid(n autodiff.Num) autodiff.Num {
	one := autodiff.NewNum(1, len(n.Gradient))
	return n.Exp().Add(one).Reciprocal()
}
