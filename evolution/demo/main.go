package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/unixpickle/weakai/evolution"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	solver := evolution.Solver{
		StepCount:            100,
		StepSizeInitial:      10,
		StepSizeFinal:        0,
		MaxPopulation:        20,
		MutateProbability:    0.3,
		CrossOverProbability: 0.3,
		SelectionProbability: 0.5,
		DFTradeoff:           evolution.LinearDFTradeoff,
	}
	solutions := solver.Solve([]evolution.Entity{Point{0, 0}})
	fmt.Println("Solution:", solutions[0])
}

func maximizeMe(x, y float64) float64 {
	return -(math.Pow(x-1, 2) + math.Pow(y, 2)) * math.Pow(math.Sin(x)*math.Cos(x)-1, 2)
}

type Point struct {
	X float64
	Y float64
}

func (p Point) Fitness() float64 {
	return maximizeMe(p.X, p.Y)
}

func (p Point) Similarity(e []evolution.Entity) float64 {
	var distanceSum float64
	for _, ent := range e {
		p1 := ent.(Point)
		distanceSum += math.Sqrt(math.Pow(p1.X-p.X, 2) + math.Pow(p1.Y-p.Y, 2))
	}
	if distanceSum == 0 {
		distanceSum = 0.00001
	}
	return 1 / distanceSum
}

func (p Point) Mutate(stepSize float64) evolution.Entity {
	diffX := (rand.Float64() * stepSize * 2) - stepSize
	diffY := (rand.Float64() * stepSize * 2) - stepSize
	return Point{p.X + diffX, p.Y + diffY}
}

func (p Point) CrossOver(e evolution.Entity) evolution.Entity {
	p1 := e.(Point)
	if rand.Intn(2) == 0 {
		return Point{X: p.X, Y: p1.Y}
	} else {
		return Point{X: p1.X, Y: p.Y}
	}
}
